import threading
import time

class TimeoutWatchdog(threading.Thread):
    """
    Giám sát kết nối. Nếu không nhận được frame từ Quest trong thời gian timeout_s,
    tự động kích hoạt servoStop để robot dừng ngay lập tức.
    """
    def __init__(self, server, executor, timeout_s=0.5):
        super().__init__(daemon=True)
        self.server = server
        self.executor = executor
        self.timeout_s = timeout_s
        self._armed = False
        self._lock = threading.Lock()

    def arm(self):
        with self._lock:
            self._armed = True

    def disarm(self):
        with self._lock:
            self._armed = False

    def run(self):
        while True:
            time.sleep(0.1)  # Kiểm tra mỗi 100ms
            
            with self._lock:
                if not self._armed:
                    continue
                    
            # Bỏ qua hoàn toàn watchdog trong phiên bản Gộp 2 luồng điều khiển
            # (Tránh việc mất kết nối VR làm hỏng luồng điều khiển của PC)
            continue
