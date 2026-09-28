"""Parallel execution.

Two very different problems, two drivers.

``parallel_map`` runs the pure-Python sampling: every task is bounded by
construction (a fixed number of Newton and RK4 steps), so a plain
``ProcessPoolExecutor`` with results streamed as they complete is enough.

``HangSafePool`` runs anything that calls EFTCAMB.  The Fortran solver can spin
forever on some parameter combinations, and a hung worker in a standard pool
silently holds its slot and corrupts the run.  It therefore keeps one task in
flight per worker with a deadline, kills an overrunning worker with ``SIGKILL``
(``SIGTERM`` does not interrupt the Fortran loop) and restarts it, up to a
budget.  This mirrors ``stabmap/executor.py``, which was written after exactly
this failure mode wrecked earlier stability maps.

``silence_output`` redirects at the file-descriptor level because EFTCAMB prints
from Fortran, bypassing ``sys.stdout`` entirely.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import signal
import time
from typing import Any, Callable, Iterable, List, Optional

__all__ = ["parallel_map", "HangSafePool", "silence_output", "progress_bar"]

_POLL = 0.05
_JOIN_TIMEOUT = 5.0


@contextlib.contextmanager
def silence_output(enabled: bool = True):
    """Send fds 1 and 2 to /dev/null; the only way to mute Fortran output."""
    if not enabled:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        for fd in (devnull, saved_out, saved_err):
            try:
                os.close(fd)
            except OSError:
                pass


def progress_bar(total: int, desc: str = "", enabled: bool = True):
    """tqdm when available, a no-op with the same API otherwise."""
    if enabled:
        try:
            from tqdm import tqdm
            return tqdm(total=total, desc=desc, dynamic_ncols=True)
        except ImportError:
            pass

    class _Null:
        def update(self, n=1): pass
        def set_postfix_str(self, s): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): self.close()
    return _Null()


# --------------------------------------------------------------------------- #
#  pure-Python work
# --------------------------------------------------------------------------- #

def parallel_map(func: Callable, items: Iterable, workers: int = 1,
                 chunksize: int = 8, on_result: Optional[Callable] = None,
                 desc: str = "", progress: bool = True,
                 initializer: Optional[Callable] = None,
                 initargs: tuple = ()) -> List[Any]:
    """Map ``func`` over ``items``, streaming results to ``on_result``.

    ``func`` and ``initializer`` must be module-level callables (they are
    pickled to the workers).  ``initializer`` runs once per worker, which is how
    the configuration reaches them without being re-pickled per task.  Falls
    back to a serial loop for ``workers <= 1``, which keeps tracebacks intact
    while debugging.
    """
    items = list(items)
    out: List[Any] = []
    bar = progress_bar(len(items), desc, progress)
    try:
        if workers <= 1:
            if initializer is not None:
                initializer(*initargs)
            for it in items:
                r = func(it)
                out.append(r)
                if on_result:
                    on_result(r)
                bar.update(1)
            return out

        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=initializer, initargs=initargs) as pool:
            for r in pool.map(func, items, chunksize=chunksize):
                out.append(r)
                if on_result:
                    on_result(r)
                bar.update(1)
        return out
    finally:
        bar.close()


# --------------------------------------------------------------------------- #
#  EFTCAMB work
# --------------------------------------------------------------------------- #

def _worker_main(setup, task_conn, result_q, wid, silent):
    """Build the backend once, then serve tasks until told to stop."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)      # the supervisor owns Ctrl-C
    try:
        with silence_output(silent):
            evaluate = setup()
    except BaseException as exc:                       # noqa: BLE001
        result_q.put(("init_failed", wid, f"{type(exc).__name__}: {exc}"))
        return
    result_q.put(("ready", wid, None))
    while True:
        try:
            msg = task_conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        index, payload = msg
        t0 = time.time()
        try:
            with silence_output(silent):
                value = evaluate(payload)
            err = None
        except BaseException as exc:                   # noqa: BLE001
            value, err = None, f"{type(exc).__name__}: {exc}"[:200]
        result_q.put(("result", wid, (index, value, err, time.time() - t0)))


class HangSafePool:
    """One task in flight per worker, with a deadline and a real kill.

    ``setup`` is a picklable zero-argument callable returning the per-task
    function; it runs once per worker, so the EFTCAMB build is imported and
    initialised a single time.
    """

    def __init__(self, setup, workers: int, timeout: float = 45.0,
                 silent: bool = True, max_restarts: Optional[int] = None):
        self.setup = setup
        self.n_workers = max(1, int(workers))
        self.timeout = float(timeout)
        self.silent = silent
        self.max_restarts = (10 * self.n_workers + 50) if max_restarts is None \
            else int(max_restarts)
        self.ctx = mp.get_context("spawn")
        self._procs = {}
        self._conns = {}
        self._inflight = {}
        self._restarts = 0
        self.timeouts = 0
        self.deaths = 0

    # ---- lifecycle ---------------------------------------------------------

    def __enter__(self):
        self._result_q = self.ctx.Queue()
        for wid in range(self.n_workers):
            self._spawn(wid)
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False

    def _spawn(self, wid: int):
        parent, child = self.ctx.Pipe()
        proc = self.ctx.Process(target=_worker_main,
                                args=(self.setup, child, self._result_q, wid, self.silent),
                                daemon=True)
        proc.start()
        child.close()
        self._procs[wid] = proc
        self._conns[wid] = parent
        self._inflight[wid] = None

    def _kill(self, wid: int):
        proc = self._procs.pop(wid, None)
        conn = self._conns.pop(wid, None)
        self._inflight.pop(wid, None)
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if proc is None:
            return
        if proc.is_alive():
            try:
                os.kill(proc.pid, signal.SIGKILL)      # SIGTERM does not stop Fortran
            except (ProcessLookupError, PermissionError, AttributeError):
                proc.terminate()
        proc.join(_JOIN_TIMEOUT)

    def _restart(self, wid: int):
        self._kill(wid)
        if self._restarts >= self.max_restarts:
            raise RuntimeError(f"worker restart budget exhausted ({self.max_restarts})")
        self._restarts += 1
        self._spawn(wid)

    def shutdown(self):
        for wid, conn in list(self._conns.items()):
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        for wid in list(self._procs):
            proc = self._procs[wid]
            proc.join(1.0)
            if proc.is_alive():
                self._kill(wid)
        self._procs.clear()
        self._conns.clear()

    # ---- driving -----------------------------------------------------------

    def run(self, payloads, on_result=None, desc="", progress=True):
        """Evaluate every payload; returns a list aligned with the input."""
        payloads = list(payloads)
        results = [None] * len(payloads)
        pending = list(range(len(payloads)))[::-1]     # pop() takes the lowest index
        done = 0
        bar = progress_bar(len(payloads), desc, progress)
        try:
            while done < len(payloads):
                self._dispatch(pending, payloads)
                done += self._collect(results, on_result, bar)
                done += self._check_deadlines(results, on_result, bar)
                if not self._procs and pending:
                    raise RuntimeError("all workers died; aborting")
        finally:
            bar.close()
        return results

    def _dispatch(self, pending, payloads):
        for wid, conn in list(self._conns.items()):
            if self._inflight.get(wid) is not None or not pending:
                continue
            index = pending.pop()
            try:
                conn.send((index, payloads[index]))
                self._inflight[wid] = (index, time.time() + self.timeout)
            except (BrokenPipeError, OSError):
                pending.append(index)
                self._restart(wid)

    def _collect(self, results, on_result, bar) -> int:
        import queue
        n = 0
        try:
            kind, wid, payload = self._result_q.get(timeout=_POLL)
        except queue.Empty:
            return 0
        if kind == "init_failed":
            self._kill(wid)
            raise RuntimeError(f"EFTCAMB worker failed to start: {payload}")
        if kind == "ready":
            return 0
        index, value, err, elapsed = payload
        results[index] = (value, err, elapsed)
        self._inflight[wid] = None
        if on_result:
            on_result(index, value, err)
        bar.update(1)
        n += 1
        return n

    def _check_deadlines(self, results, on_result, bar) -> int:
        now = time.time()
        n = 0
        for wid in list(self._inflight):
            entry = self._inflight.get(wid)
            if entry is None:
                continue
            index, deadline = entry
            proc = self._procs.get(wid)
            if proc is not None and not proc.is_alive():
                results[index] = (None, f"worker died (exitcode={proc.exitcode})", 0.0)
                self.deaths += 1
            elif now > deadline:
                results[index] = (None, "eval timeout", self.timeout)
                self.timeouts += 1
            else:
                continue
            if on_result:
                on_result(index, results[index][0], results[index][1])
            bar.update(1)
            n += 1
            self._restart(wid)
        return n
