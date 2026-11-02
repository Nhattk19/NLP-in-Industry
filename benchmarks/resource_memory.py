"""Process-memory sampling for local capacity measurements."""
import threading


class Memory:
    """Sample process RSS every 50ms, plus process-lifetime OS peak when available."""
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stopped = threading.Event()
        def sample():
            while not self.stopped.wait(.05):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stopped.set()
        self.thread.join()
        self.end = self.process.memory_info().rss
        self.peak = max(self.peak, self.end)
        self.os_peak = getattr(self.process.memory_info(), "peak_wset", None)

    def fields(self):
        return {"sampled_peak_rss_bytes": self.peak, "end_rss_bytes": self.end,
                "process_lifetime_peak_working_set_bytes": self.os_peak}
