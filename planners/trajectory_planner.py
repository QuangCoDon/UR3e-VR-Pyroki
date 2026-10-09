import time
import numpy as onp
import planners.pyroki_mp as pyroki_mp

class TrajectoryPlanner:
    def __init__(self, pyroki_mp_module, guard, logger):
        """
        pyroki_mp_module: Thư viện/module pyroki_mp
        guard: SingularityGuard
        logger: DataLogger
        """
        self.pyroki_mp = pyroki_mp_module
        self.guard = guard
        self.logger = logger

    def plan_trajectory(self, start_q, goal_pos, goal_quat, 
                        pk_robot=None, world_coll=None, robot_coll=None, target_link_name="tool0"):
        """
        Quy hoạch quỹ đạo từ góc khớp start_q đến tọa độ mục tiêu (goal_pos, goal_quat)
        Trả về: (thành_công, danh_sách_waypoints, thông_báo)
        """
        if not all([pk_robot, world_coll, robot_coll]):
            return False, [], "PyRoki thiếu thông tin môi trường!"
        
        t_start = time.perf_counter()
        timesteps = 30 
        
        sol_traj, _, _ = self.pyroki_mp.solve_online_planning(
            robot=pk_robot,
            robot_coll=robot_coll,
            world_coll=world_coll,
            target_link_name=target_link_name,
            target_position=onp.array(goal_pos),
            target_wxyz=onp.array(goal_quat),
            timesteps=timesteps,
            dt=0.05,
            start_cfg=onp.array(start_q),
            prev_sols=onp.array([start_q]),
        )
        t_end = time.perf_counter()
        plan_time = (t_end - t_start) * 1000
        
        if sol_traj is not None and len(sol_traj) > 0:
            traj = sol_traj.tolist()
            return True, traj, f"PyRoki tìm đường mất {plan_time:.2f} ms"
        else:
            return False, [], "PyRoki: Lỗi, Không thể tìm đường!"

    def filter_and_log_trajectory(self, traj, q_start):
        """
        Kiểm tra vặn xoắn và tính an toàn của quỹ đạo, sau đó ghi log
        """
        clamped_traj = [q_start]
        for tq in traj:
            last_q = clamped_traj[-1]
            if max(abs(tq[i] - last_q[i]) for i in range(6)) > 0.4:
                print(f"[Guard] Cắt bỏ waypoint do nhảy khớp > 0.4 rad!")
                continue
            clamped_traj.append(tq)
            
        if len(clamped_traj) <= 1:
            return False, [], "Từ chối: Quỹ đạo bị vặn xoắn hoàn toàn!"

        # Kiểm tra gập cổ tay sâu
        for tq in clamped_traj:
            if tq[3] < -2.9 or tq[3] > -0.2:
                return False, [], "Từ chối: Quá gập, dễ kẹp nách!"

        # Đánh giá độ an toàn
        min_sigma_traj = 1.0
        for tq in clamped_traj:
            m = self.guard.metrics_calc.get_metrics(onp.array([tq], dtype=onp.float32))
            min_sigma_traj = min(min_sigma_traj, m["sigma_min"][0].item())

        warning = ""
        if min_sigma_traj < self.guard.danger_threshold:
            warning = f"CẢNH BÁO: Xuyên qua kỳ dị (min_sigma = {min_sigma_traj:.4f})"
        elif min_sigma_traj < self.guard.warn_threshold:
            warning = f"Đi sát kỳ dị (min_sigma = {min_sigma_traj:.4f})"
        else:
            warning = "An toàn"

        # Ghi log
        for q in clamped_traj:
            m = self.guard.metrics_calc.get_metrics(onp.array([q], dtype=onp.float32))
            self.logger.log_step(q, None, None, None,
                            m["sigma_min"][0].item(), m["manipulability"][0].item(),
                            m["condition_number"][0].item(), "none", None, "OK", "mp_execute")
        self.logger.flush()

        return True, clamped_traj, warning
