#!/usr/bin/env python3
import socket
import time
from typing import Optional
import rtde_receive

ROBOT_IP        = "127.0.0.1"
DASHBOARD_PORT  = 29999

def setup_ursim() -> bool:
    """Power-on robot if POWER_OFF (brake release via dashboard). Returns True when RUNNING."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect((ROBOT_IP, DASHBOARD_PORT))
        time.sleep(0.2)
        s.recv(4096)

        def _cmd(c, w=0.8):
            s.sendall((c + "\n").encode())
            time.sleep(w)
            try:
                s.setblocking(False)
                r = s.recv(4096).decode().strip()
                s.setblocking(True)
            except Exception:
                s.setblocking(True)
                r = ""
            return r

        mode = _cmd("robotmode", 0.3)
        safety = _cmd("safetystatus", 0.3)
        print(f"[URSim] Robot mode: {mode}, Safety: {safety}")

        if "PROTECTIVE_STOP" in safety or "VIOLATION" in safety or "FAULT" in safety:
            print("[URSim] Unlocking protective stop...")
            _cmd("unlock protective stop")
            _cmd("close safety popup")
            time.sleep(2.0)
            mode = _cmd("robotmode", 0.3)

        if "POWER_OFF" in mode or "IDLE" in mode:
            print("[URSim] Powering on and releasing brakes...")
            _cmd("power on", w=2.0)
            _cmd("brake release", w=5.0)
            mode = _cmd("robotmode", 0.5)
            print(f"[URSim] Mode after: {mode}")

        s.close()
        return "RUNNING" in mode
    except Exception as e:
        print(f"[URSim] Lỗi kết nối Dashboard (Port 29999): {e}")
        print("[URSim] Bỏ qua lỗi Dashboard, giả định robot đã được bật nguồn...")
        return True

class URSimController:
    """
    Controller để kết nối với URSim qua RTDE mà không phụ thuộc CuRobo.
    """
    def __init__(self):
        print("\n" + "=" * 56)
        print("  URSim Controller (Non-CuRobo)")
        print("=" * 56)

        if not setup_ursim():
            raise RuntimeError("Robot not RUNNING. Check URSim container.")

        print("[Setup] Connecting RTDEReceive...")
        self.rr = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        q = self.rr.getActualQ()
        print(f"[Setup] Connected. Joints: {[round(j, 4) for j in q]}")
        print("=" * 56 + "\n")

    def joints(self) -> list:
        """Current joint positions [rad] x 6."""
        return self.rr.getActualQ()

    def joint_velocities(self) -> list:
        return self.rr.getActualQd()
        
    def tcp(self) -> list:
        """Current TCP pose [x, y, z, rx, ry, rz]."""
        return self.rr.getActualTCPPose()
        
    def pose(self) -> list:
        return self.tcp()

    def close(self):
        self.rr.disconnect()
        print("[Setup] Disconnected")
