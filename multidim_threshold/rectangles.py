import operator as op
from typing import NamedTuple, Iterable
from functools import reduce

from lenses import lens

bot_lens = lens.intervals.Each().bot
top_lens = lens.intervals.Each().top
intervals_lens = lens.GetAttr('intervals')


class Interval(NamedTuple):
    bot: float
    top: float

    def __contains__(self, x):
        if isinstance(x, Interval):
            return self.bot <= x.bot and x.top <= self.top
        return self.bot <= x <= self.top

    def __and__(self, i2):
        # TODO
        return

    @property
    def radius(self):
        return self.top - self.bot


def _select_rec(intervals, j, lo, hi):
    def include_error(i, k, l, h):
        idx = (j >> k) & 1
        l2, h2 = i[idx]
        return min(l2, l), max(h, h2)

    chosen_rec = tuple(
        include_error(i, k, l, h)
        for k, (l, h, i) in enumerate(zip(lo, hi, intervals)))
    return to_rec(chosen_rec)


class Rec(NamedTuple):
    intervals: Iterable[Interval]

    @property
    def bot(self):
        return bot_lens.collect()(self)

    @property
    def top(self):
        return top_lens.collect()(self)

    @property
    def diag(self):
        return tuple(t - b for b, t in zip(self.bot, self.top))

    @property
    def dim(self):
        return len(self.intervals)

    @property
    def volume(self):
        return reduce(op.mul, lens.intervals.Each().radius.collect()(self))

    @property
    def degenerate(r):
        return any(x == 0 for x in r.diag)

    def forward_cone(self, p):
        """Computes the forward cone from point p."""
        return to_rec(zip(p, self.top))

    def backward_cone(self, p):
        """Computes the backward cone from point p."""
        return to_rec(zip(self.bot, p))

    def subdivide(self, rec2, drop_fb=True):
        """Generate all 2^n - 2 incomparable hyper-boxes.
        TODO: Do not generate unnecessary dimensions for degenerate surfaces
        """
        n = self.dim
        if n <= 1:
            return
        elif drop_fb:
            indicies = range(1, 2**n - 1)
        else:
            indicies = range(0, 2**n)
        lo, hi = rec2.bot, rec2.top
        forward, backward = self.forward_cone(lo), self.backward_cone(hi)
        intervals = list(zip(backward.intervals, forward.intervals))
        x = {_select_rec(intervals, j, lo, hi) for j in indicies}
        yield from x - {self}

    def __contains__(self, r):
        return all(i2 in i1 for i1, i2 in zip(self.intervals, r.intervals))


def to_rec(intervals):
    intervals = tuple(Interval(*i) for i in intervals)
    return Rec(intervals)
