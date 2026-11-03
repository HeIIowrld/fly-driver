"""Bounded optional cache warming; never carries sensory history or release decisions."""

from concurrent.futures import ThreadPoolExecutor


class PixelPrecomputeQueue:
    """Run pure immutable-pixel cache callbacks without blocking camera acquisition.

    Full queue skips optional cache warming, never images or observations. The actor
    retains its normal exact cache-miss path. A completed callback error is fatal.
    """

    def __init__(self, callback, *, workers=2, maximum_pending=8):
        if not callable(callback) or workers != 2 or maximum_pending != 8:
            raise ValueError('Reviewed two-worker/eight-image cache budget required')
        self.callback = callback
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='csp-pixel-cache')
        self.futures = []
        self.submitted = self.skipped = self.completed = self.peak_pending = 0
        self.closed = False

    def submit(self, pixels):
        if self.closed or pixels.flags.writeable:
            raise ValueError('Open cache queue and immutable source pixels required')
        pending = []
        for future in self.futures:
            if future.done():
                future.result()
                self.completed += 1
            else:
                pending.append(future)
        self.futures = pending
        if len(pending) == 8:
            self.skipped += 1
            return
        self.futures.append(self.pool.submit(self.callback, pixels))
        self.submitted += 1
        self.peak_pending = max(self.peak_pending, len(self.futures))

    def close(self):
        self.closed = True
        self.pool.shutdown(wait=True)
        for future in self.futures:
            future.result()
            self.completed += 1
        self.futures = []

    def statistics(self):
        return dict(policy='bounded_async_pure_pixel_cache_only', workers=2,
            maximum_pending_images=8, submitted=self.submitted, skipped_cache_warms=self.skipped,
            completed=self.completed, peak_pending_images=self.peak_pending,
            actor_cache_miss_path_unchanged=True)
