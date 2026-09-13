"""Streaming accumulators behind the velocity features.

Every accumulator obeys one contract, and it is the contract the property tests in
``tests/test_leakage_properties.py`` check:

    ``query(key, ts)`` reflects **only** events already passed to ``update``.

The feature builder always queries before it updates, so a transaction can never see itself or
anything that comes after it in the stream's total order. That is the whole leakage defence:
there is no ``groupby`` over the full table anywhere in this package, so there is no place for
future data to enter. Windowed `groupby().rolling()` would be faster, but it is also where
leakage bugs live — off-by-one on a closed interval silently includes the current row, and the
resulting model looks excellent offline and fails in production.

All accumulators are O(1) amortised: each event is appended once and popped at most once.
"""
from __future__ import annotations

from collections import Counter, deque
from collections.abc import Hashable
from dataclasses import dataclass

__all__ = ["CountSumWindow", "DistinctWindow", "RecencyTracker", "WelfordTracker"]


class CountSumWindow:
    """Per-key count and value-sum over a trailing time window.

    The window is **half-open and backward-looking**: an event at time ``t0`` is in the window
    queried at ``ts`` iff ``ts - t0 <= span``. Events exactly ``span`` seconds old are retained;
    the current event is never included because it has not been inserted yet.
    """

    __slots__ = ("span", "_events", "_count", "_sum")

    def __init__(self, span_seconds: float) -> None:
        if span_seconds <= 0:
            raise ValueError("span_seconds must be positive")
        self.span = float(span_seconds)
        self._events: dict[Hashable, deque[tuple[float, float]]] = {}
        self._count: dict[Hashable, int] = {}
        self._sum: dict[Hashable, float] = {}

    def _evict(self, key: Hashable, ts: float) -> None:
        dq = self._events.get(key)
        if dq is None:
            return
        cutoff = ts - self.span
        cnt, tot = self._count[key], self._sum[key]
        while dq and dq[0][0] < cutoff:
            _, v = dq.popleft()
            cnt -= 1
            tot -= v
        if cnt:
            self._count[key], self._sum[key] = cnt, tot
        else:
            # Drop empty keys so memory tracks *active* entities, not lifetime cardinality.
            self._events.pop(key, None)
            self._count.pop(key, None)
            self._sum.pop(key, None)

    def query(self, key: Hashable, ts: float) -> tuple[int, float]:
        """Return ``(count, sum)`` of prior events for ``key`` inside the window at ``ts``."""
        self._evict(key, ts)
        return self._count.get(key, 0), self._sum.get(key, 0.0)

    def update(self, key: Hashable, ts: float, value: float) -> None:
        dq = self._events.get(key)
        if dq is None:
            dq = self._events[key] = deque()
            self._count[key] = 0
            self._sum[key] = 0.0
        dq.append((ts, value))
        self._count[key] += 1
        self._sum[key] += value

    @property
    def n_active_keys(self) -> int:
        return len(self._events)


class DistinctWindow:
    """Per-key count of distinct secondary values over a trailing window.

    Powers the features that actually separate the fraud archetypes: *distinct merchants touched
    by this card in the last hour* (card testing) and *distinct cards seen by this merchant in
    the last week* (merchant breach).
    """

    __slots__ = ("span", "_events", "_counts")

    def __init__(self, span_seconds: float) -> None:
        if span_seconds <= 0:
            raise ValueError("span_seconds must be positive")
        self.span = float(span_seconds)
        self._events: dict[Hashable, deque[tuple[float, Hashable]]] = {}
        self._counts: dict[Hashable, Counter] = {}

    def _evict(self, key: Hashable, ts: float) -> None:
        dq = self._events.get(key)
        if dq is None:
            return
        cutoff = ts - self.span
        ctr = self._counts[key]
        while dq and dq[0][0] < cutoff:
            _, v = dq.popleft()
            if ctr[v] <= 1:
                del ctr[v]
            else:
                ctr[v] -= 1
        if not dq:
            self._events.pop(key, None)
            self._counts.pop(key, None)

    def query(self, key: Hashable, ts: float) -> int:
        self._evict(key, ts)
        ctr = self._counts.get(key)
        return len(ctr) if ctr else 0

    def update(self, key: Hashable, ts: float, value: Hashable) -> None:
        dq = self._events.get(key)
        if dq is None:
            dq = self._events[key] = deque()
            self._counts[key] = Counter()
        dq.append((ts, value))
        self._counts[key][value] += 1


class RecencyTracker:
    """Seconds since the previous event on a key, plus the key's lifetime prior count."""

    __slots__ = ("_last", "_count")

    def __init__(self) -> None:
        self._last: dict[Hashable, float] = {}
        self._count: dict[Hashable, int] = {}

    def query(self, key: Hashable, ts: float) -> tuple[float, int]:
        """Return ``(seconds_since_last, prior_count)``.

        ``seconds_since_last`` is ``inf`` the first time a key is seen, which downstream is
        mapped to a sentinel rather than imputed: "never seen before" is itself signal (a card's
        first-ever transaction, a device nobody has used).
        """
        last = self._last.get(key)
        return (float("inf") if last is None else ts - last), self._count.get(key, 0)

    def update(self, key: Hashable, ts: float) -> None:
        self._last[key] = ts
        self._count[key] = self._count.get(key, 0) + 1


class WelfordTracker:
    """Numerically stable prior-only mean/variance per key (Welford's online algorithm).

    Used for "how unusual is this amount *for this card*" without ever touching the card's
    future spend. A batch ``groupby('card_id').amount.mean()`` would encode exactly that future
    and is the single most common leak in published fraud notebooks.
    """

    __slots__ = ("_n", "_mean", "_m2")

    def __init__(self) -> None:
        self._n: dict[Hashable, int] = {}
        self._mean: dict[Hashable, float] = {}
        self._m2: dict[Hashable, float] = {}

    def query(self, key: Hashable) -> tuple[int, float, float]:
        """Return ``(n, mean, std)`` over prior events only; ``std`` is 0 when ``n < 2``."""
        n = self._n.get(key, 0)
        if n == 0:
            return 0, 0.0, 0.0
        var = self._m2[key] / (n - 1) if n > 1 else 0.0
        return n, self._mean[key], var**0.5

    def update(self, key: Hashable, value: float) -> None:
        n = self._n.get(key, 0) + 1
        mean = self._mean.get(key, 0.0)
        delta = value - mean
        mean += delta / n
        self._n[key] = n
        self._mean[key] = mean
        self._m2[key] = self._m2.get(key, 0.0) + delta * (value - mean)


@dataclass(frozen=True)
class WindowSpec:
    """A named trailing window."""

    name: str
    seconds: float

    def __post_init__(self) -> None:
        if self.seconds <= 0:
            raise ValueError("window seconds must be positive")


HOUR = 3_600.0
DAY = 86_400.0

STANDARD_WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec("1h", HOUR),
    WindowSpec("24h", DAY),
    WindowSpec("7d", 7 * DAY),
    WindowSpec("30d", 30 * DAY),
)
