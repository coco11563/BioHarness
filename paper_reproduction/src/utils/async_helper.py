"""Async execution helpers for RLM REPL environment.

RLM Philosophy:
- RLM executes code in a synchronous REPL environment
- Our retrieval/embedding services are async (for efficiency)
- We need a clean sync-to-async bridge for REPL tool functions

DESIGN: Thread-safe with event loop reuse
- Maintains a dedicated worker thread with its own event loop
- Avoids creating new loops per call (expensive)
- Thread-safe for concurrent REPL access
- Guards against deadlock from same-thread calls
"""

import asyncio
import atexit
import threading
from concurrent.futures import Future, TimeoutError as FuturesTimeoutError
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


class AsyncExecutorError(Exception):
    """Base exception for AsyncExecutor errors."""
    pass


class ExecutorShutdownError(AsyncExecutorError):
    """Raised when executor is already shutdown."""
    pass


class ExecutorDeadlockError(AsyncExecutorError):
    """Raised when run() is called from the executor thread (would deadlock)."""
    pass


class ExecutorTimeoutError(AsyncExecutorError):
    """Raised when async operation times out."""
    pass


class AsyncExecutor:
    """Thread-safe async executor for sync-to-async bridging.

    Creates a dedicated background thread with a persistent event loop.
    All async operations are submitted to this loop, avoiding the overhead
    of creating new loops per call.

    Usage:
        executor = AsyncExecutor()

        # In sync context (REPL):
        result = executor.run(some_async_function())

        # Cleanup on exit
        executor.shutdown()
    """

    _instance: "AsyncExecutor | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "AsyncExecutor":
        """Singleton pattern for global executor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._shutdown = False

        # Start the background thread
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="AsyncExecutor")
        self._thread.start()
        self._started.wait(timeout=5.0)

        # Register cleanup
        atexit.register(self.shutdown)

        self._initialized = True

    def _run_loop(self):
        """Run the event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._started.set()

        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()

    def run(self, coro: Coroutine[None, None, T], timeout: float = 300.0) -> T:
        """Run async coroutine from sync context.

        Thread-safe: can be called from any thread except the executor thread.

        Args:
            coro: Async coroutine to execute
            timeout: Maximum seconds to wait (default 300s / 5 min)

        Returns:
            Result from the coroutine

        Raises:
            ExecutorShutdownError: If executor is shutdown
            ExecutorDeadlockError: If called from executor thread
            ExecutorTimeoutError: If operation times out
            Exception: Any exception from the coroutine
        """
        if self._shutdown or self._loop is None:
            raise ExecutorShutdownError("AsyncExecutor is shutdown")

        # Guard against deadlock: don't call from executor thread
        if threading.current_thread() == self._thread:
            raise ExecutorDeadlockError(
                "Cannot call run() from executor thread - would deadlock. "
                "Use 'await' directly inside async functions."
            )

        # Submit to the background loop
        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)

        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            # Cancel the task on timeout
            future.cancel()
            raise ExecutorTimeoutError(f"Async operation timed out after {timeout}s")

    def shutdown(self):
        """Shutdown the executor and cleanup."""
        if self._shutdown:
            return

        self._shutdown = True

        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread is not None:
            self._thread.join(timeout=2.0)


# Global singleton
_executor: AsyncExecutor | None = None


def get_executor() -> AsyncExecutor:
    """Get the global async executor singleton."""
    global _executor
    if _executor is None:
        _executor = AsyncExecutor()
    return _executor


def run_async(coro: Coroutine[None, None, T]) -> T:
    """Run async coroutine in sync context.

    This is the main entry point for REPL tools to call async functions.
    Uses the global AsyncExecutor singleton for efficient execution.

    Args:
        coro: Async coroutine to execute

    Returns:
        Result from the coroutine

    Example:
        async def fetch_data():
            return await some_async_api()

        # In REPL:
        data = run_async(fetch_data())
    """
    try:
        return get_executor().run(coro)
    except ExecutorDeadlockError:
        # Nested call: a tool already running on the executor thread called another
        # tool that also uses run_async. Submitting to the same loop would deadlock,
        # so run this coroutine on its own loop in a fresh thread instead. Observed
        # 2026-09-09: get_paper_abstracts failed this way 2/2 times inside the agent,
        # both right after it had correctly identified the target paper.
        result: dict[str, Any] = {}

        def _runner() -> None:
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                result["error"] = exc

        worker = threading.Thread(target=_runner, daemon=True)
        worker.start()
        worker.join()
        if "error" in result:
            raise result["error"]
        return result["value"]


# =============================================================================
# Alternative: Simple fallback for environments without threading
# =============================================================================

def run_async_simple(coro: Coroutine[None, None, T]) -> T:
    """Simple async runner (creates new loop each call).

    Use this only when threading is problematic.
    Less efficient but simpler.
    """
    try:
        # Check if we're already in an async context
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - create one
        return asyncio.run(coro)

    # Running loop exists - we need to run in a new thread
    # This is less efficient but necessary for nested async calls
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()
