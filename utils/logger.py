import time
from .tools import clear


class Logger:
    def __init__(self, total_epochs=None, total_batches=None, auto_newline=False):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.auto_newline = auto_newline
        self.start_time = time.time()

    def log(self, epoch=None, batch=None, **kwargs):
        clear()
        log_items = []

        if epoch is not None and self.total_epochs:
            log_items.append(f"Epoch {epoch}/{self.total_epochs}")
        if batch is not None and self.total_batches:
            log_items.append(f"Batch {batch}/{self.total_batches}")

        # 添加剩下的自定义项
        for k, v in kwargs.items():
            if isinstance(v, float):
                log_items.append(f"{k}: {v:.4f}")
            else:
                log_items.append(f"{k}: {v}")

        # 时间相关
        elapsed = time.time() - self.start_time
        log_items.append(f"Time: {elapsed:.2f}s")
        self.start_time = time.time()

        line = " | ".join(log_items)
        end_char = "\n" if self.auto_newline else "\r"
        print(line, end=end_char)
