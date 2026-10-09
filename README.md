# UR3e VR Teleoperation & Motion Planning System (PyRoki Version)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Robotics](https://img.shields.io/badge/Robot-UR3e-orange)
![Framework](https://img.shields.io/badge/Framework-PyRoki-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

Hệ thống điều khiển tay máy công nghiệp UR3e toàn diện, kết hợp **Điều khiển viễn thao thời gian thực (Real-time VR Teleoperation)** và **Tự hành quy hoạch quỹ đạo (Motion Planning)**. Dự án được tối ưu hóa chạy trên CPU thông qua JAX (thư viện **PyRoki**) mà không cần đến Card đồ họa (GPU) chuyên dụng.

## Tính năng Nổi bật (Key Features)

- **Zero-Latency VR Teleoperation:** Sử dụng giao thức mạng RTDE (`servoJ` ở 20Hz) cho phép người dùng dùng kính VR (Meta Quest) để điều khiển cánh tay mô phỏng lại các cử động thực tế không có độ trễ.
- **PyRoki JAX-based Solver:** Xử lý Inverse Kinematics (IK) bằng thư viện tối ưu hóa trên nền JAX, cho thời gian hội tụ siêu nhanh trên CPU.
- **Lớp bảo vệ Động cơ (Safety Gate):**
  - **Chống vặn xoắn (Anti-Twist):** Ngăn chặn hiện tượng nhảy nhánh IK (Jump Branch) > 0.4 rad, bảo vệ động cơ.
  - **Kẹp Vận Tốc (Rate Limiter):** Giới hạn phần mềm delta góc khớp luôn dưới 1.5 rad/s.
- **3D Web Interface:** Sử dụng bộ thư viện **Viser**, cho phép giám sát robot thông qua giao diện Web trực quan.
- **Hỗ trợ thực tế ảo WebXR (Meta Quest):** Buồng lái ảo (Virtual Cockpit) với giao diện Passthrough (AR xuyên thấu).

## Kiến trúc Hệ thống (Architecture)

Hệ thống được thiết kế theo kiến trúc Module:

```text
Project_Ur3e/
├── main.py                          <-- File chạy chính (Khởi động hệ thống)
├── README.md                        <-- Hướng dẫn 
├── config/                          <-- Cấu hình an toàn
├── assets/                          <-- Chứa tài nguyên mô hình URDF
├── core/                            <-- Các thư viện kết nối mạng nội bộ (URSim/RTDE)
├── planners/                        <-- Chứa các Module quy hoạch quỹ đạo (PyRoki MP)
├── safety/                          <-- Chứa các Module bảo vệ (Jacobian Singularity)
└── teleop/                          <-- Các thuật toán giải IK, FK, và Vòng lặp Teleop
└── teleop_quest/                    <-- Server kết nối kính Meta Quest qua WebXR
```

## Hướng dẫn Cài đặt & Chạy thử (Getting Started)

### 1. Yêu cầu hệ thống
- Hệ điều hành: Ubuntu 22.04+ (khuyến nghị)
- Môi trường: Môi trường ảo Python có chứa JAX, PyRoki, và Viser.
- Tủ điều khiển thật UR3e hoặc giả lập URSim (Docker).

### 2. Chạy giả lập URSim
Khởi động máy ảo Docker chứa hệ điều hành của tủ điều khiển UR:
```bash
docker run --rm -it -e ROBOT_MODEL=UR3e -p 30001-30004:30001-30004 -p 29999:29999 -p 6080:6080 universalrobots/ursim_e-series
```

### 3. Khởi động Server Điều khiển & VR
Mở Terminal, di chuyển vào thư mục dự án và chạy:
```bash
cd /home/quang/Project_Ur3e/UR3e
python3 main.py
```
*(Hệ thống sẽ tự động dò địa chỉ IP LAN của máy tính, tạo chứng chỉ SSL nội bộ và in đường dẫn ra màn hình).*

### 4. Kết nối Kính Meta Quest
1. Đeo kính Quest (đảm bảo kết nối chung mạng Wi-Fi với máy tính).
2. Mở trình duyệt Web của Quest, truy cập **Cổng Dữ Liệu Ngầm** (ghi trên màn hình Terminal), ví dụ: `https://[IP]:8444` $\rightarrow$ Chọn Advanced $\rightarrow$ Proceed.
3. Truy cập **Cổng Giao Diện**: `https://[IP]:8443` $\rightarrow$ Advanced $\rightarrow$ Proceed.
4. Bấm **START IMMERSIVE VR** để vào chế độ điều khiển không gian 3 chiều.
5. Trên giao diện Web của máy tính (`http://localhost:8080`), hãy đánh dấu chọn **`Bat VR Teleop`** để mở khóa truyền động.

## Cơ chế Điều khiển Kính VR
- **Nắm giữ (Clutch):** Bóp giữ **ĐỒNG THỜI Cò (Trigger) + Nút hông (Grip)** để robot bắt đầu bám theo tay bạn.
- Khi đẩy tay tới trước, robot sẽ lao thẳng ra phía trước. Xoay cổ tay thì đầu ngàm (TCP) sẽ vặn theo. Thả nút ra thì robot sẽ đứng im tại chỗ.
- **Cảnh báo E-STOP:** LUÔN sẵn sàng nhấn nút Dừng Khẩn Cấp trên màn hình ảo hoặc tủ điện thực tế nếu có sự cố bất ngờ.
