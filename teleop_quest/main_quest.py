#!/usr/bin/env python3
import os
import sys
import time
import argparse
import signal
import yaml
import traceback
import numpy as np

from core.ursim_control import URSimController
from core.data_logger import DataLogger
from teleop.teleoperation import (
    PyRokiIKSolver,
    SafetyGate, ServoJExecutor, TeleoperationController
)
from teleop.pyroki_ik_setup import setup_pyroki_ik, get_pyroki_fk

from teleop_quest.quest_server import QuestServer
from teleop_quest.delta_pose_source import DeltaPoseSource
from teleop_quest.timeout_watchdog import TimeoutWatchdog

import viser
import subprocess

POLL_SLEEP = 0.05

def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "singularity.yaml")
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("="*60)
    print(" KHỞI ĐỘNG HỆ THỐNG VR TELEOP (META QUEST) - PYROKI VERSION ")
    print("="*60)

    # 1. Khởi động WebXR Server & Cập nhật IP/SSL
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = '127.0.0.1'
    finally:
        s.close()

    base_dir = os.path.dirname(os.path.dirname(__file__))
    cert_path = os.path.join(base_dir, "certs", "cert.pem")
    key_path = os.path.join(base_dir, "certs", "key.pem")
    
    print(f"[Hệ thống] IP hiện tại: {local_ip}")
    os.makedirs(os.path.join(base_dir, "certs"), exist_ok=True)
    
    ip_cache_path = os.path.join(base_dir, "certs", "ip_cache.txt")
    cached_ip = ""
    if os.path.exists(ip_cache_path):
        with open(ip_cache_path, "r") as f:
            cached_ip = f.read().strip()

    if not os.path.exists(cert_path) or cached_ip != local_ip:
        print(f"[Hệ thống] IP thay đổi (hoặc chạy lần đầu). Đang tự động tạo SSL cho IP: {local_ip}...")
        subprocess.run(
            f"openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout {key_path} -out {cert_path} -subj '/CN={local_ip}'", 
            shell=True, stderr=subprocess.DEVNULL
        )
        with open(ip_cache_path, "w") as f:
            f.write(local_ip)

    quest_server = QuestServer(cert_path, key_path)
    quest_server.start()

    # 2. Khởi động Controller & URSim
    print("[Hệ thống] Đang kết nối URSim...")
    ctrl = URSimController()
    
    for _ in range(10):
        if np.linalg.norm(ctrl.joints()) > 0.01:
            break
        time.sleep(0.1)
    START_JOINTS = ctrl.joints()
    config = load_config()
    logger = DataLogger(config)

    # 2.5 Khởi tạo PyRoki IK thay cho CuRobo
    print("[Hệ thống] Đang khởi tạo PyRoki IK...")
    urdf_path = os.path.join(base_dir, "assets", "ur3e_custom.urdf")
    pk_robot, solve_ik, check_error, target_link_idx = setup_pyroki_ik(urdf_path, target_link_name="tool0")
    fk_func = get_pyroki_fk(pk_robot, target_link_idx)
    
    pyroki_ik = PyRokiIKSolver(pk_robot, solve_ik, check_error, target_link_idx)

    # 3. Viser UI (Hiển thị robot ảo thuần)
    subprocess.run("fuser -k 8080/tcp 2>/dev/null", shell=True)
    time.sleep(0.5)

    server = viser.ViserServer(port=8080)
    robot_urdf = None
    joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    try:
        import viser.extras as vex
        from pathlib import Path
        robot_urdf = vex.ViserUrdf(server, Path(urdf_path))
        if robot_urdf:
            robot_urdf.update_cfg(np.array(START_JOINTS))
    except Exception as e:
        print(f"[Cảnh báo] Không thể tải URDF lên Viser (có thể do phiên bản viser): {e}")

    # 3.5 Setup Teleoperation
    import transforms3d
    
    # Lấy pose ban đầu từ FK
    init_pose = fk_func(START_JOINTS)
    init_pos = init_pose["pos"]
    init_quat_wxyz = init_pose["quat"]

    target_widget = server.scene.add_transform_controls(
        "target_pose", 
        depth_test=False, 
        position=init_pos, 
        wxyz=init_quat_wxyz,
        scale=0.18,
        disable_sliders=False
    )
    
    # Nguồn cấp tọa độ từ Kính VR, mỏ neo vào FK
    def get_robot_fk_as_ur():
        q_real = ctrl.joints()
        fk_dict = fk_func(q_real)
        pos = fk_dict["pos"]
        quat_wxyz = fk_dict["quat"]
        axis, angle = transforms3d.quaternions.quat2axangle(quat_wxyz)
        return [pos[0], pos[1], pos[2], axis[0]*angle, axis[1]*angle, axis[2]*angle]
        
    pose_source = DeltaPoseSource(quest_server, ctrl, scale=0.7, anchor_reference_func=get_robot_fk_as_ur, alpha=0.08)
    
    class WidgetPoseSource:
        def __init__(self, alpha=0.15):
            self.alpha = alpha
            self._ema_pos = None
            self._ema_quat_xyzw = None
            
        def get_pose(self):
            target_pos = target_widget.position
            target_quat_wxyz = target_widget.wxyz
            target_quat_xyzw = np.array([target_quat_wxyz[1], target_quat_wxyz[2], target_quat_wxyz[3], target_quat_wxyz[0]])
            
            if self._ema_pos is None:
                self._ema_pos = target_pos
                self._ema_quat_xyzw = target_quat_xyzw
            else:
                self._ema_pos = self.alpha * target_pos + (1.0 - self.alpha) * self._ema_pos
                if np.dot(self._ema_quat_xyzw, target_quat_xyzw) < 0:
                    target_quat_xyzw = -target_quat_xyzw
                self._ema_quat_xyzw = self.alpha * target_quat_xyzw + (1.0 - self.alpha) * self._ema_quat_xyzw
                self._ema_quat_xyzw /= np.linalg.norm(self._ema_quat_xyzw)
                
            ema_quat_wxyz = [self._ema_quat_xyzw[3], self._ema_quat_xyzw[0], self._ema_quat_xyzw[1], self._ema_quat_xyzw[2]]
            return {"pos": self._ema_pos.tolist(), "quat": ema_quat_wxyz}
            
    widget_pose_source = WidgetPoseSource()
    from safety.singularity_guard.guard_core import SingularityGuard
    guard = SingularityGuard(config)
    
    executor = ServoJExecutor(dt=POLL_SLEEP)
    # Thêm mặt đất (Ground plane)
    server.scene.add_grid("ground_grid", width=2.0, height=2.0, width_segments=20, height_segments=20, position=(0, 0, 0))

    teleop = TeleoperationController(
        pose_source=pose_source, # SỬ DỤNG VR HEADSET LÀM NGUỒN ĐIỀU KHIỂN CHÍNH
        pyroki_solver=pyroki_ik,
        safety_gate=SafetyGate(guard=guard),
        executor=executor
    )
    
    # 4. Watchdog
    watchdog = TimeoutWatchdog(quest_server, executor)
    watchdog.start()

    with server.gui.add_folder("VR Teleoperation"):
        cb_teleop = server.gui.add_checkbox("Bat VR Teleop", initial_value=False)
        cb_record = server.gui.add_checkbox("Record AI Data", initial_value=False)
        sld_speed = server.gui.add_slider("Toc do (Scale)", min=0.1, max=2.0, step=0.1, initial_value=1.0)
        btn_home = server.gui.add_button("Reset to Home")
        ui_status = server.gui.add_text("VR Status", initial_value="Chờ...", disabled=True)
        ui_conn = server.gui.add_text("Quest Connected", initial_value="No", disabled=True)
        
    @sld_speed.on_update
    def _on_speed_update(_):
        pose_source.scale = sld_speed.value
        print(f"[Cài đặt] Đã chỉnh tốc độ Robot xuống còn {sld_speed.value}x")
        
    @cb_record.on_update
    def _on_record_update(_):
        logger.is_recording = cb_record.value
        if cb_record.value:
            print("[AI Logger] BẮT ĐẦU thu thập dữ liệu!")
        else:
            logger.flush()
            print("[AI Logger] ĐÃ DỪNG thu thập dữ liệu.")
            
    def _generate_joint_trajectory(q_start, q_end, steps=40):
        # Nội suy tuyến tính đơn giản trong không gian khớp
        traj = []
        for i in range(1, steps + 1):
            alpha = i / steps
            q_step = [q_start[j] * (1 - alpha) + q_end[j] * alpha for j in range(6)]
            traj.append(q_step)
        return traj
        
    def _on_btn_home_click(_):
        if cb_teleop.value:
            ui_status.value = "Vui lòng TẮT VR Teleop trước khi về Home!"
            return
            
        print("[Hệ thống] Đang tính toán đường về Home...")
        ui_status.value = "Đang chạy về Home an toàn..."
        q_start = ctrl.joints()
        q_home = [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, -1.5708]
        
        traj = _generate_joint_trajectory(q_start, q_home)
        
        executor.connect()
        time.sleep(1.0)
        for q in traj:
            executor._ctrl.servoJ(q, 0.0, 0.0, POLL_SLEEP, 0.1, 300)
            if robot_urdf:
                robot_urdf.update_cfg(np.array(q))
            time.sleep(POLL_SLEEP)
        executor.disconnect()
        
        ui_status.value = "Đã về Home an toàn!"
        print("[Hệ thống] Đã về Home an toàn.")
        
        fk_res = fk_func(q_home)
        target_widget.position = fk_res["pos"]
        target_widget.wxyz = fk_res["quat"]
        widget_pose_source._ema_pos = None
        widget_pose_source._ema_quat_xyzw = None
            
    btn_home.on_click(_on_btn_home_click)

    def handle_sigint(sig, frame):
        print("\n[Hệ thống] Đang tắt an toàn...")
        teleop.enabled = False
        time.sleep(0.5)
        ctrl.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    # ==== TỰ ĐỘNG CHẠY VỀ HOME LÚC KHỞI ĐỘNG ====
    print("[Hệ thống] Tự động di chuyển về Home lúc khởi động...")
    q_start = START_JOINTS
    q_home = [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, -1.5708]
    traj = _generate_joint_trajectory(q_start, q_home)
    executor.connect()
    time.sleep(1.0)
    for q in traj:
        executor._ctrl.servoJ(q, 0.0, 0.0, POLL_SLEEP, 0.1, 300)
        if robot_urdf:
            robot_urdf.update_cfg(np.array(q))
        time.sleep(POLL_SLEEP)
    executor.disconnect()
    
    fk_res = fk_func(q_home)
    target_widget.position = fk_res["pos"]
    target_widget.wxyz = fk_res["quat"]
    widget_pose_source._ema_pos = None
    widget_pose_source._ema_quat_xyzw = None
    print("[Hệ thống] Đã khởi động an toàn tại Home.")

    # Vòng lặp chính (20Hz)
    print("\n" + "="*30)
    print(f"ĐỊA CHỈ IP HIỆN TẠI CỦA MÁY TÍNH LÀ: {local_ip}")
    print("="*30)
    
    print("\n[LIÊN KẾT NHANH DÀNH CHO MÁY TÍNH]")
    print(f"-> Viser Dashboard (Giao diện điều khiển): http://localhost:8080")
    print(f"-> URSim Web (Màn hình Teach Pendant): http://localhost:6080/vnc.html")
    
    print("\n[LIÊN KẾT DÀNH CHO KÍNH VR QUEST 3S]")
    print(f"1. Thông chốt SSL (Duyệt mạng): https://{local_ip}:{quest_server.ws_port}")
    print(f"2. Vào Buồng lái Ảo (VR Cockpit): https://{local_ip}:{quest_server.http_port}")
    print("="*30 + "\n")
    
    _last_teleop_state = False
    
    while True:
        if quest_server.check_and_clear_estop():
            teleop.enabled = False
            cb_teleop.value = False
            try:
                executor._ctrl.stopJ(2.0)
            except: pass
            ui_status.value = "EMERGENCY STOP (Từ VR)!"
            print("\n[CẢNH BÁO] Đã nhận lệnh E-STOP từ kính VR! Đã ngắt động cơ.")
            quest_server.send_haptic(1.0, 1000)
            
        ui_conn.value = "Yes" if quest_server.is_connected() else "No"
        
        if cb_teleop.value != _last_teleop_state:
            teleop.enabled = cb_teleop.value
            _last_teleop_state = cb_teleop.value
            
            if teleop.enabled:
                print("[VR] KÍCH HOẠT ĐIỀU KHIỂN.")
                executor.connect()
                watchdog.arm()
            else:
                print("[VR] HỦY KÍCH HOẠT ĐIỀU KHIỂN.")
                executor.disconnect()
                watchdog.disarm()

        if teleop.enabled:
            vr_pose = pose_source.get_pose()
            if vr_pose is not None:
                target_widget.position = vr_pose["pos"]
                target_widget.wxyz = vr_pose["quat"]

            try:
                q_real = ctrl.joints()
                if robot_urdf:
                    robot_urdf.update_cfg(np.array(q_real))
                
                teleop.step(q_real)
                current_status = teleop.get_status()
                ui_status.value = current_status
                
                if "thất bại" in current_status.lower() or "cảnh báo" in current_status.lower() or "singularity" in current_status.lower():
                    quest_server.send_haptic(intensity=1.0, duration=150)
                
                if logger.is_recording:
                    logger.log_step(q_real, ctrl.joint_velocities(), ctrl.pose(), image_path="cam_frame_placeholder.jpg")
                    
            except Exception as e:
                ui_status.value = f"Lỗi: {e}"
                print(f"[Lỗi Điều khiển] {e}")
                traceback.print_exc()
        else:
            ui_status.value = "Tắt"

        time.sleep(POLL_SLEEP)

if __name__ == "__main__":
    main()
