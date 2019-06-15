"""An atomic, thread-safe incrementing counter."""

import threading

from tqdm import tqdm


class AtomicCounter:
    """An atomic, thread-safe incrementing counter.
    """

    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value (default 0)."""
        self.value = initial
        self._lock = threading.Lock()

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value += num
            return self.value


class AtomicProgressBar:
    """An atomic, thread-safe incrementing counter.
    """

    def __init__(self, total=100, desc=''):
        """Initialize a new atomic counter to given initial value (default 0)."""
        self.bar = tqdm(total=total, desc=desc)
        self._lock = threading.Lock()

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.bar.update(num)
            return self.bar
