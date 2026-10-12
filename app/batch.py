"""Run many claims through the judge with a small, quota-friendly amount of parallelism."""

import logging
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.errors import NO_DOCUMENTS_MESSAGE, friendly_error

logger = logging.getLogger("app")

MAX_BATCH_SIZE = 25
MAX_CONCURRENCY = 3
# Calls are started at least this far apart (about 15 a minute): the free Gemini tier is rate limited, and calls
# that finish quickly would otherwise let three workers burst well past that.
MIN_SECONDS_BETWEEN_STARTS = 4.0


class Throttle:
    """Hands out start times at least `min_interval` seconds apart, across all threads."""

    def __init__(self, min_interval: float, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            start_at = max(now, self._next_start)
            self._next_start = start_at + self.min_interval
        delay = start_at - now
        if delay > 0:
            self._sleep(delay)


def run_batch(
    product_id: str,
    claims: list[str],
    verify_fn: Callable[[str, str], dict],
    *,
    max_workers: int = MAX_CONCURRENCY,
    min_interval: float = MIN_SECONDS_BETWEEN_STARTS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[dict]:
    """Yield one result dict per claim as each finishes (not in input order; each carries its `index`).

    A failure is reported on that claim's result and does not stop the others, except two cases where every
    remaining claim would fail the same way (the AI service's rate limit, or no documents loaded): the rest
    are then skipped, which saves quota and time and still returns everything finished so far.
    """
    throttle = Throttle(min_interval, clock, sleep)
    stop = threading.Event()
    stop_info: dict = {}

    def failure(index: int, claim: str, status: int, message: str) -> dict:
        return {"index": index, "claim": claim, "error": message, "status": status}

    def work(index: int, claim: str) -> dict:
        throttle.wait()
        if stop.is_set():
            return failure(index, claim, stop_info["status"], "Skipped: " + stop_info["message"])

        started = time.perf_counter()
        try:
            result = verify_fn(product_id, claim)
        except Exception as exc:
            mapped = friendly_error(exc)
            if mapped is None:
                logger.exception("batch_item_failed", extra={"fields": {"index": index}})
                return failure(index, claim, 500, "Unexpected error while checking this claim.")
            status, message = mapped
            if status == 429:
                stop_info.update(status=status, message=message)
                stop.set()
            return failure(index, claim, status, message)

        if not result.get("retrieved_chunks"):
            stop_info.update(status=503, message=NO_DOCUMENTS_MESSAGE)
            stop.set()
            return failure(index, claim, 503, NO_DOCUMENTS_MESSAGE)

        return {
            "index": index,
            "claim": claim,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "cited_passage": result["cited_passage"],
            "explanation": result["explanation"],
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "error": None,
        }

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = [executor.submit(work, index, claim) for index, claim in enumerate(claims)]
        for future in as_completed(futures):
            yield future.result()
    finally:
        stop.set()
        executor.shutdown(wait=False, cancel_futures=True)
