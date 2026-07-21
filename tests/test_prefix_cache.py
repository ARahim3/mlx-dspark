"""Model-free tests for the prefix-cache manager: LCP reuse, trimming, the min-reuse gate,
bookkeeping (cache holds all-but-last generated token), reuse-eligibility detection, reset."""

from __future__ import annotations

from mlx_dspark.prefix_cache import PrefixCache, _lcp, target_cache_reusable


class KVCache:  # name matters: target_cache_reusable whitelists exactly "KVCache"
    def __init__(self, offset=0):
        self.offset = offset

    def trim(self, n):
        n = min(n, self.offset)
        self.offset -= n
        return n


class RotatingKVCache:  # legacy-shaped rotating cache without the mlx-lm rotation
    offset = 0          # machinery (max_size / is_trimmable) — must stay non-reusable

    def trim(self, n):
        return 0


class RealRotatingKVCache:
    """mlx-lm-shaped rotating cache: linear (trimmable) until offset reaches max_size."""

    def __init__(self, offset=0, max_size=512):
        self.offset = offset
        self.max_size = max_size

    def is_trimmable(self):
        return self.offset < self.max_size

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n


# make target_cache_reusable's name check match the real class
RealRotatingKVCache.__name__ = "RotatingKVCache"


class FakeCtx:
    def __init__(self):
        self.k = None
        self.v = None
        self.trimmed_to = None

    def trim_to(self, length):
        self.trimmed_to = length


def _mk_cache():
    return [KVCache(), KVCache()]


def _mk_ctx():
    return [FakeCtx(), FakeCtx()]


def test_lcp():
    assert _lcp([1, 2, 3], [1, 2, 9]) == 2
    assert _lcp([1, 2, 3], [1, 2, 3, 4]) == 3
    assert _lcp([], [1]) == 0
    assert _lcp([5], [6]) == 0


def test_reusable_detection():
    assert target_cache_reusable([KVCache(), KVCache()]) is True
    assert target_cache_reusable([KVCache(), RotatingKVCache()]) is False
    # rotating caches WITH the mlx-lm rotation machinery are structurally reusable
    # (the wrap is caught per-entry at store time)
    assert target_cache_reusable([KVCache(), RealRotatingKVCache()]) is True


def test_wrapped_rotating_cache_is_not_stored():
    pc = PrefixCache(_mk_cache, _mk_ctx, min_reuse=1)
    prompt = list(range(20))
    cache = [RealRotatingKVCache(max_size=16), KVCache()]
    ctx = _mk_ctx()
    cache[0].offset = 16                        # wrapped its window during generation
    cache[1].offset = 21
    pc.store(cache, ctx, prompt, [99, 100])
    assert pc.info()["cached_tokens"] == 0      # refused
    # under the window it stores fine
    cache[0].offset = 10
    pc.store(cache, ctx, prompt, [99, 100])
    assert pc.info()["cached_tokens"] == 21


def test_lru_two_slots_dont_evict_each_other():
    pc = PrefixCache(_mk_cache, _mk_ctx, min_reuse=4, slots=2)
    conv_a = list(range(100, 120))
    conv_b = list(range(200, 220))
    ca, xa, _ = pc.acquire(conv_a)
    for c in ca:
        c.offset = len(conv_a)
    pc.store(ca, xa, conv_a, [1, 2])
    cb, xb, _ = pc.acquire(conv_b)              # different conversation -> fresh caches
    assert cb is not ca
    for c in cb:
        c.offset = len(conv_b)
    pc.store(cb, xb, conv_b, [3, 4])
    # both conversations now hit their own slot
    got_a, _, reuse_a = pc.acquire(conv_a + [1, 130])
    assert got_a is ca and reuse_a == len(conv_a) + 1
    pc.store(got_a, xa, conv_a + [1, 130], [5, 6])
    got_b, _, reuse_b = pc.acquire(conv_b + [3, 230])
    assert got_b is cb and reuse_b == len(conv_b) + 1
    assert pc.hits == 2


def test_lru_eviction_beyond_capacity():
    pc = PrefixCache(_mk_cache, _mk_ctx, min_reuse=4, slots=2)
    convs = [list(range(i * 100, i * 100 + 20)) for i in (1, 2, 3)]
    for conv in convs:
        c, x, _ = pc.acquire(conv)
        for layer in c:
            layer.offset = len(conv)
        pc.store(c, x, conv, [7, 8])
    assert len(pc.info()["slots"]) == 2
    # the oldest conversation was evicted; the two most recent still hit
    _, _, r1 = pc.acquire(convs[0] + [9])
    assert r1 == 0
    _, _, r2 = pc.acquire(convs[2] + [9])
    assert r2 > 0


def test_acquire_empty_is_fresh():
    pc = PrefixCache(_mk_cache, _mk_ctx, min_reuse=4)
    cache, ctx, reuse_len = pc.acquire([1, 2, 3, 4, 5])
    assert reuse_len == 0 and len(cache) == 2 and len(ctx) == 2


def test_store_bookkeeping_and_reuse():
    pc = PrefixCache(_mk_cache, _mk_ctx, min_reuse=4)
    prompt = list(range(1, 11))                 # 10 prompt tokens
    gen = [11, 12, 13]                           # 3 generated
    cache, ctx, _ = pc.acquire(prompt)
    for c in cache:                             # simulate post-generation cache length
        c.offset = len(prompt) + len(gen) - 1   # holds all but the last generated token = 12
    pc.store(cache, ctx, prompt, gen)
    assert pc.info()["cached_tokens"] == 12     # prompt + gen[:-1]

    # a follow-up prompt that diverges after 6 shared tokens -> reuse 6, trim caches to 6
    cache2, ctx2, reuse_len = pc.acquire([1, 2, 3, 4, 5, 6, 50, 51])
    assert reuse_len == 6
    assert cache2 is cache and all(c.offset == 6 for c in cache2)   # trimmed 12 -> 6
    assert all(c.trimmed_to == 6 for c in ctx2)
    assert pc.hits == 1 and pc.reused_tokens == 6


def test_min_reuse_gate():
    pc = PrefixCache(_mk_cache, _mk_ctx, min_reuse=8)
    prompt = list(range(1, 11))
    cache, ctx, _ = pc.acquire(prompt)
    for c in cache:
        c.offset = len(prompt) - 1
    pc.store(cache, ctx, prompt, [99])
    # shares only 3 tokens (< min_reuse 8) -> fresh, no reuse
    _, _, reuse_len = pc.acquire([1, 2, 3, 500, 501, 502])
    assert reuse_len == 0 and pc.hits == 0


def test_reuse_len_capped_below_prompt_len():
    # even an identical prompt keeps >=1 token to prefill (need next-token logits)
    pc = PrefixCache(_mk_cache, _mk_ctx, min_reuse=1)
    prompt = [1, 2, 3, 4, 5]
    cache, ctx, _ = pc.acquire(prompt)
    for c in cache:
        c.offset = len(prompt) - 1
    pc.store(cache, ctx, prompt, [6])           # cached_tokens = [1,2,3,4,5]
    _, _, reuse_len = pc.acquire([1, 2, 3, 4, 5])
    assert reuse_len == 4                        # min(lcp=5, cached=5, len-1=4)


def test_reset_invalidates():
    pc = PrefixCache(_mk_cache, _mk_ctx, min_reuse=1)
    cache, ctx, _ = pc.acquire([1, 2, 3])
    for c in cache:
        c.offset = 2
    pc.store(cache, ctx, [1, 2, 3], [4])
    pc.reset()
    assert pc.info()["cached_tokens"] == 0
    _, _, reuse_len = pc.acquire([1, 2, 3])
    assert reuse_len == 0


def test_store_normalizes_caches_to_the_token_record():
    """A speculative round commits a block at once, so an eos landing mid-block leaves the
    KV cache (and drafter ctx) holding rows for tokens that never made it into token_ids.
    store() must trim back to the record, so the slot holds exactly `slot.tokens`."""
    pc = PrefixCache(_mk_cache, _mk_ctx, min_reuse=1)
    cache, ctx = _mk_cache(), _mk_ctx()
    prompt = [1, 2, 3, 4]
    generated = [5, 6, 7]                 # eos at 7 -> record is prompt + [5, 6]
    # verify wrote 2 extra rows past the record (the block's post-eos tokens)
    for c in cache:
        c.offset = len(prompt) + len(generated) - 1 + 2
    pc.store(cache, ctx, prompt, generated)

    slot = pc._slots[0]
    assert slot.tokens == [1, 2, 3, 4, 5, 6]
    for c in cache:
        assert c.offset == len(slot.tokens)        # was 8, the record claims 6
    for c in ctx:
        assert c.trimmed_to == len(slot.tokens)


def test_store_leaves_an_exact_cache_untouched():
    """The normal case (no eos truncation) is already exact — store must not trim it."""
    pc = PrefixCache(_mk_cache, _mk_ctx, min_reuse=1)
    cache, ctx = _mk_cache(), _mk_ctx()
    prompt, generated = [1, 2, 3], [4, 5]
    for c in cache:
        c.offset = len(prompt) + len(generated) - 1  # == len(record)
    pc.store(cache, ctx, prompt, generated)
    for c in cache:
        assert c.offset == 4
