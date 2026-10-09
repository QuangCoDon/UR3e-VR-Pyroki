import os
import math
import numpy as np
import yaml
from safety.singularity_guard.jacobian_metrics import JacobianMetrics

def load_config(config_path="config/singularity.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_config(config, config_path="config/singularity.yaml"):
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

def calibrate_thresholds(num_samples=100000):
    print(f"Bắt đầu lấy mẫu {num_samples} cấu hình ngẫu nhiên...")
    
    metrics_calc = JacobianMetrics()
    
    # Giới hạn khớp của UR3e
    limits = [
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        (-math.pi, math.pi)
    ]
    
    # Sinh mẫu ngẫu nhiên
    q_rand = np.zeros((num_samples, 6), dtype=np.float32)
    for i in range(6):
        lo, hi = limits[i]
        q_rand[:, i] = (np.random.rand(num_samples) * (hi - lo)) + lo
        
    print("Đang tính toán các chỉ số động học (Jacobian, SVD)...")
    
    batch_size = 10000
    all_sigma_min = []
    
    for i in range(0, num_samples, batch_size):
        q_batch = q_rand[i:i+batch_size]
        metrics = metrics_calc.get_metrics(q_batch)
        all_sigma_min.append(metrics["sigma_min"])
        
    all_sigma_min = np.concatenate(all_sigma_min)
    
    q01 = np.percentile(all_sigma_min, 1)
    q05 = np.percentile(all_sigma_min, 5)
    q10 = np.percentile(all_sigma_min, 10)
    q50 = np.percentile(all_sigma_min, 50)
    
    print("\n--- Phân phối sigma_min ---")
    print(f"Trung vị (50%): {q50:.5f}")
    print(f"Percentile 10%: {q10:.5f}")
    print(f"Percentile 5% : {q05:.5f}")
    print(f"Percentile 1% : {q01:.5f}")
    
    danger_threshold = max(0.015, float(q01))
    warn_threshold = max(0.03, float(q05))
    
    print("\n--- Đề xuất Ngưỡng ---")
    print(f"DANGER: {danger_threshold:.5f}")
    print(f"WARN  : {warn_threshold:.5f}")
    
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "singularity.yaml")
    if os.path.exists(config_path):
        config = load_config(config_path)
        config["thresholds"]["sigma_min_danger"] = float(f"{danger_threshold:.4f}")
        config["thresholds"]["sigma_min_warn"] = float(f"{warn_threshold:.4f}")
        save_config(config, config_path)
        print(f"Đã cập nhật ngưỡng vào {config_path}!")
    else:
        print(f"Lỗi: Không tìm thấy file {config_path}")

if __name__ == "__main__":
    calibrate_thresholds(100000)
