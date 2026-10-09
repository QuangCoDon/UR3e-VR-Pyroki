import os
import time
import csv

class DataLogger:
    def __init__(self, config):
        self.output_dir = config.get("output_dir", "logs/ai_training_data")
        os.makedirs(self.output_dir, exist_ok=True)
        # Tạo folder con cho từng session
        session_id = f"session_{int(time.time())}"
        self.session_dir = os.path.join(self.output_dir, session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        self.images_dir = os.path.join(self.session_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.current_run_file = os.path.join(self.session_dir, "robot_states.csv")
        self.data_buffer = []
        
        # Bật/tắt cờ ghi
        self.is_recording = False

    def log_step(self, *args, **kwargs):
        """
        Nhận bất kỳ arguments nào để không bị crash khi các planner cũ gọi với format cũ.
        (Chỉ ghi nếu is_recording = True)
        """
        if not self.is_recording:
            return
            
        # Extract basic info if possible
        q = kwargs.get("q", args[0] if len(args) > 0 else None)
        qd = kwargs.get("qd", args[1] if len(args) > 1 else None)
        tcp_pose = kwargs.get("tcp_pose", args[2] if len(args) > 2 else None)
        event_tag = kwargs.get("event_tag", "teleop")
            
        row = {
            "timestamp_ms": int(time.time() * 1000),
            "event_tag": str(event_tag),
        }
        
        for i in range(6):
            row[f"q_{i}"] = float(q[i]) if (q is not None and len(q) == 6) else 0.0
            row[f"qd_{i}"] = float(qd[i]) if (qd is not None and len(qd) == 6) else 0.0
                
        if tcp_pose is not None and len(tcp_pose) >= 3:
            # Lưu [x, y, z, q_w, q_x, q_y, q_z]
            for i, axis in enumerate(['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']):
                if i < len(tcp_pose):
                    row[f"tcp_{axis}"] = float(tcp_pose[i])
                
        self.data_buffer.append(row)
        
        # Auto save every 100 steps (5 seconds at 20Hz)
        if len(self.data_buffer) >= 100:
            self.flush()

    def flush(self):
        if not self.data_buffer:
            return
            
        file_exists = os.path.exists(self.current_run_file)
        
        with open(self.current_run_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.data_buffer[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.data_buffer)
            
        print(f"[AI Logger] Đã ghi {len(self.data_buffer)} frames vào {self.current_run_file}")
        self.data_buffer = []
