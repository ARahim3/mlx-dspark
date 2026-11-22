"""inkling_batch: one token per REQUEST through one Inkling 2 forward, and the session around it.
Model-free: a tiny looped Inkling 2 with random bf16 weights (the fused pieces' own fixture)."""
import mlx.core as mx
import pytest
from test_inkling_fused import _bf16, _tiny_bf16_model

from mlx_dspark import inkling_batch as B
from mlx_dspark import inkling_fused as F

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernels")
TAP = [1, 5, 8]


class _Tok:
    """ids are code points; 59 ends a turn."""

    eos_token_id = 59

    def decode(self, ids):
        return "".join(chr(48 + i) for i in ids)


def _target():
    from mlx_dspark.target import Target

    t = Target(_tiny_bf16_model(), tokenizer=_Tok())
    lm = t.model.language_model
    if not B.supported(lm):
        pytest.skip("the fused pieces / in-place caches did not clear their self-test here")
    return t, lm


PROMPTS = [[3, 7, 11], [2, 9, 30, 41, 5, 8, 13, 21, 34], [4, 6, 12, 19, 27, 33, 1, 14, 22, 35, 40, 44, 51, 17]]


# ---------------------------------------------------------------- the two slot kernels
@metal
@pytest.mark.parametrize("rows", [1, 3, 8])
def test_slot_convolution_is_the_single_row_kernel_per_request(rows):
    c = 256
    weight = _bf16(c, 4, 1, seed=1)
    state = mx.random.normal((rows, 3, c), key=mx.random.key(2))
    x = _bf16(1, rows, c, seed=3)
    res16, res32 = _bf16(1, rows, c, seed=4), mx.random.normal((1, rows, c), key=mx.random.key(5))
    for res, dtype in ((None, None), (res16, mx.float32), (res32, mx.bfloat16)):
        out, nxt = B.slot_short_convolution(state, weight, x, 4, res, dtype)
        assert out.shape == (1, rows, c) and nxt.shape == (rows, 3, c) and nxt.dtype == mx.float32
        for b in range(rows):
            ref, window = F.fused_short_convolution(
                state[b: b + 1], weight, x[:, b: b + 1], 4,
                None if res is None else res[:, b: b + 1], dtype)
            assert mx.array_equal(out[:, b: b + 1], ref)
            assert mx.array_equal(nxt[b: b + 1], window[:, -3:])


@metal
@pytest.mark.parametrize("rows", [1, 4])
def test_slot_dynamic_convolution_is_the_single_row_kernel_per_request(rows):
    c, heads = 256, 4
    static = _bf16(4, c, seed=1)
    dyn = _bf16(1, rows, heads * 4, seed=2)
    state = mx.random.normal((rows, 3, c), key=mx.random.key(3))
    x = _bf16(1, rows, c, seed=4)
    out, nxt = B.slot_dynamic_convolution(state, static, dyn, x, 4, heads)
    for b in range(rows):
        ref, window = F._dyn_kernel()(
            inputs=[state[b: b + 1], static, dyn[:, b: b + 1], x[:, b: b + 1], F._dyn_divisor(4)],
            template=[("T", x.dtype), ("W", 4), ("HEADS", heads)],
            grid=(c, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(1, 1, c), (1, 4, c)], output_dtypes=[x.dtype, mx.float32])
        assert mx.array_equal(out[:, b: b + 1], ref)
        assert mx.array_equal(nxt[b: b + 1], window[:, -3:])


# ---------------------------------------------------------------- one step against single rows
def _prefilled(t, ids, tap=None):
    cache = t.make_cache()
    logits, fused = t.prefill(mx.array([ids]), cache, tap, want_logits=True)
    mx.eval(logits)
    return cache, fused


@metal
def test_a_step_gives_each_request_its_own_decode_step():
    """Prompts under and over the sliding window (8) and the log-scaling floor (4), decoded
    together for several steps: every row is the single-row decode step on that request's
    cache, and the caches and convolution states end where single-stream decoding leaves them."""
    t, lm = _target()
    forced = [[10, 20, 31, 42, 2, 9], [15, 25, 36, 47, 3, 8], [18, 28, 39, 50, 4, 7]]
    together = [_prefilled(t, ids)[0] for ids in PROMPTS]
    alone = [_prefilled(t, ids)[0] for ids in PROMPTS]
    states = B.stack_states(lm, together)
    for s in range(6):
        got = B.step(lm, mx.array([row[s] for row in forced]), together, states)
        assert got.shape == (3, 60)
        for b in range(3):
            want = t.plain(mx.array([[forced[b][s]]]), alone[b])[0, -1]
            assert mx.allclose(got[b], want, atol=2e-2, rtol=2e-2), (s, b)
            assert int(mx.argmax(got[b]).item()) == int(mx.argmax(want).item())
    B.unstack_states(states, together)
    for a, c in zip(together, alone):
        for va, vc in zip(a, c):
            assert va[0].offset == vc[0].offset
            for idx in range(4):
                assert mx.allclose(va[1][idx], vc[1][idx], atol=2e-2, rtol=2e-2)


@metal
def test_one_request_through_the_step_is_the_decode_step_bit_for_bit():
    t, lm = _target()
    a, b = _prefilled(t, PROMPTS[2])[0], _prefilled(t, PROMPTS[2])[0]
    states = B.stack_states(lm, [a])
    for token in (10, 20, 31, 42):
        got = B.step(lm, mx.array([token]), [a], states)
        assert mx.array_equal(got[0], t.plain(mx.array([[token]]), b)[0, -1])


@metal
def test_a_step_returns_the_context_rows_a_drafter_keeps():
    t, lm = _target()
    caches = [_prefilled(t, ids)[0] for ids in PROMPTS[:2]]
    states = B.stack_states(lm, caches)
    logits, fused = B.step(lm, mx.array([10, 15]), caches, states, tap=TAP)
    assert fused.shape == (1, 2, 3 * 128) and logits.shape == (2, 60)
    for b, ids in enumerate(PROMPTS[:2]):
        cache = _prefilled(t, ids)[0]
        _, want = t.prefill(mx.array([[(10, 15)[b]]]), cache, TAP, want_logits=False)
        assert mx.allclose(fused[:, b: b + 1], want, atol=2e-2, rtol=2e-2)


@metal
def test_states_follow_requests_that_join_and_leave():
    t, lm = _target()
    caches = [_prefilled(t, ids)[0] for ids in PROMPTS]
    states = B.stack_states(lm, caches)
    two = B.take_states(states, [0, 2])
    assert two[0][0].shape[0] == 2
    assert mx.array_equal(two[5][3][1:2], states[5][3][2:3])
    back = B.join_states(two, B.stack_states(lm, [caches[1]]))
    assert mx.array_equal(back[5][3][2:3], states[5][3][1:2])


# ---------------------------------------------------------------- the session
def _greedy(t, ids, n, **kw):
    from mlx_dspark.generate import greedy_generate

    return greedy_generate(t, t.tokenizer, prompt_ids=ids, max_new_tokens=n, **kw)


def _drain(session, limit=400):
    done = {}
    for _ in range(limit):
        for req, res in session.tick():
            done[req.meta] = res
        if not session.n_active:
            return done
    raise AssertionError("the session did not drain")


@metal
def test_session_decodes_each_request_as_its_own_greedy_run():
    """Three prompts of different lengths and budgets, one admitted mid-flight: each request's
    tokens are its single-stream greedy tokens, delivered when ITS budget ends."""
    t, _ = _target()
    budgets = [9, 14, 6]
    want = [_greedy(t, ids, n) for ids, n in zip(PROMPTS, budgets)]
    session = B.Session(t, t.tokenizer)
    texts = {i: [] for i in range(3)}
    for i in (0, 1):
        session.admit(B.Request(PROMPTS[i], max_new_tokens=budgets[i], meta=i,
                                on_text=texts[i].append))
    done = {}
    for _ in range(4):
        for req, res in session.tick():
            done[req.meta] = res
    session.admit(B.Request(PROMPTS[2], max_new_tokens=budgets[2], meta=2,
                            on_text=texts[2].append))
    done.update(_drain(session))
    assert session.peak == 3
    for i in range(3):
        assert done[i].token_ids == want[i].token_ids, i
        assert done[i].finish_reason == want[i].finish_reason
        assert done[i].text == want[i].text == "".join(texts[i])
        assert done[i].num_tokens == len(want[i].token_ids)
        assert 0 < done[i].batched_tokens <= done[i].num_tokens


@metal
def test_session_stops_a_request_on_its_stop_string_and_on_a_client_that_left():
    from mlx_dspark.generate import StopStreaming

    t, _ = _target()
    full = _greedy(t, PROMPTS[1], 12)
    stop = full.text[4:6]
    want = _greedy(t, PROMPTS[1], 12, stop=[stop])
    heard = []

    def gone(piece):
        heard.append(piece)
        if len("".join(heard)) >= 3:
            raise StopStreaming

    session = B.Session(t, t.tokenizer)
    session.admit(B.Request(PROMPTS[1], max_new_tokens=12, stop=[stop], meta="stop"))
    session.admit(B.Request(PROMPTS[0], max_new_tokens=12, on_text=gone, meta="gone"))
    session.admit(B.Request(PROMPTS[2], max_new_tokens=12, meta="whole"))
    done = _drain(session)
    assert done["stop"].text == want.text and done["stop"].finish_reason == "stop"
    assert done["gone"].num_tokens < 12 and done["gone"].finish_reason == "stop"
    assert done["whole"].token_ids == _greedy(t, PROMPTS[2], 12).token_ids


@metal
def test_a_long_prompt_shares_the_time_with_requests_already_decoding(monkeypatch):
    """A prompt longer than one prefill chunk is read a chunk at a time, and the requests
    already decoding get steps in between: the newcomer does not freeze them."""
    from mlx_dspark import generate as g

    monkeypatch.setattr(g, "_target_chunk", lambda target: 16)
    t, _ = _target()
    long_prompt = [(7 * i + 3) % 58 for i in range(40)]          # three chunks
    want_a, want_b = _greedy(t, PROMPTS[0], 40), _greedy(t, long_prompt, 6)
    session = B.Session(t, t.tokenizer)
    a = B.Request(PROMPTS[0], max_new_tokens=40, meta="a")
    b = B.Request(long_prompt, max_new_tokens=6, meta="b")
    session.admit(a)
    session.tick()
    session.tick()
    session.admit(b)
    before = len(a.out_ids)
    session.tick()                                               # b's first chunk, then a's share
    assert b.filled == 16 and session.filling == [b]
    assert len(a.out_ids) > before
    done = _drain(session)
    assert done["a"].token_ids == want_a.token_ids and done["b"].token_ids == want_b.token_ids
    assert session.prefill_seconds > 0 and a.shared and b.shared


class _Drafter:
    """What the session needs of a DSpark drafter: the taps and a context that takes rows."""

    class config:
        target_layer_ids = TAP

    class _Ctx:
        def __init__(self):
            self.rows, self.k = [], mx.zeros((1,))

        def trim_to(self, n):
            self.rows = self.rows[:n]

    def make_ctx_cache(self):
        return [self._Ctx()]

    def update_context(self, fused, ctx_offset, ctx_caches):
        ctx = ctx_caches[0]
        assert ctx_offset == len(ctx.rows), "context rows must arrive in order, none missing"
        ctx.rows.extend(fused[:, i] for i in range(fused.shape[1]))


def _solo_for(t, drafter, log):
    """The single-stream loop's contract, with plain decoding standing in for speculation:
    read ``prompt[reuse_len:]`` into the request's caches (taps to the drafter), then decode
    until the budget, the end of the turn, or the session's streamer says stop."""
    from mlx_dspark.generate import GenResult

    def solo(req, prompt, reuse_len, max_new_tokens, fresh):
        log.append(("fresh" if fresh else "resumed", len(prompt) - reuse_len))
        st = req.streamer
        logits, fused = t.prefill(mx.array([prompt[reuse_len:]]), req.cache, TAP, want_logits=True)
        drafter.update_context(fused, reuse_len, req.ctx)
        out = [int(mx.argmax(logits[0, -1]).item())]
        st.update(out)
        while len(out) < max_new_tokens and out[-1] != 59 and not st.stopped:
            logits, fused = t.prefill(mx.array([[out[-1]]]), req.cache, TAP, want_logits=True)
            drafter.update_context(fused, len(prompt) + len(out) - 1, req.ctx)
            out.append(int(mx.argmax(logits[0, -1]).item()))
            st.update(out)
        st.flush()
        return GenResult(text="", token_ids=out, num_tokens=len(out), num_rounds=len(out),
                         accept_lengths=[1] * len(out), target_forwards=len(out), seconds=0.0,
                         finish_reason="length" if len(out) >= max_new_tokens else "stop")

    return solo


@metal
def test_a_lone_request_runs_the_single_stream_loop_and_hands_over_both_ways():
    """Alone -> the single-stream loop; another request arrives -> it yields at a round and both
    decode together; the other finishes -> the loop picks the first one up again, with every
    context row of the batched stretch in the drafter's cache. The tokens are the greedy run's."""
    t, _ = _target()
    drafter, log, arrivals = _Drafter(), [], []
    want_a, want_b = _greedy(t, PROMPTS[2], 24), _greedy(t, PROMPTS[0], 5)
    session = B.Session(t, t.tokenizer, drafter, solo=_solo_for(t, drafter, log),
                        waiting=lambda: bool(arrivals))
    text = []
    a = B.Request(PROMPTS[2], max_new_tokens=24, meta="a", on_text=text.append)
    session.admit(a)
    polls = [0]

    def waiting():                      # the second request shows up three rounds into the first
        polls[0] += 1
        if polls[0] == 4:
            arrivals.append(B.Request(PROMPTS[0], max_new_tokens=5, meta="b"))
        return bool(arrivals)

    session.waiting = waiting
    done = {}
    for _ in range(200):
        while arrivals and session.has_free_slot:
            session.admit(arrivals.pop())
        for req, res in session.tick():
            done[req.meta] = res
        if not session.n_active and not arrivals:
            break
    assert [kind for kind, _ in log] == ["fresh", "resumed"]
    assert log[1][1] == 1                               # a resumed stretch reads only the pending token
    assert session.peak == 2 and session.steps >= 4
    assert done["a"].token_ids == want_a.token_ids and done["b"].token_ids == want_b.token_ids
    assert done["a"].text == want_a.text == "".join(text)
    # the drafter saw one row per position of request a, in order (update_context asserts it)
    assert len(a.ctx[0].rows) == len(PROMPTS[2]) + len(want_a.token_ids) - 1


@metal
def test_without_a_drafter_a_lone_request_decodes_at_one_row():
    t, _ = _target()
    session = B.Session(t, t.tokenizer, solo=lambda *a: pytest.fail("no drafter, no solo loop"))
    session.admit(B.Request(PROMPTS[1], max_new_tokens=7, meta=0))
    assert _drain(session)[0].token_ids == _greedy(t, PROMPTS[1], 7).token_ids
    assert session.peak == 1 and session.steps >= 6


# ---------------------------------------------------------------- behind the server's batch engine
@metal
def test_batch_engine_runs_concurrent_inkling_requests_as_one_session():
    """Three requests arriving together on a baseline-mode engine decode in one session and
    each returns its own greedy run; a seeded one keeps the serial path."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from mlx_dspark import server as S

    ex = ThreadPoolExecutor(max_workers=1)

    def build():                      # models live on the thread that generates
        t, _ = _target()
        return t, [_greedy(t, ids, n) for ids, n in zip(PROMPTS, (9, 14, 6))]

    t, want = ex.submit(build).result()
    engine = S.Engine(t, t.tokenizer, None, mode="baseline", model_id="tiny", target_repo="tiny",
                      drafter_repo=None, max_draft_tokens=None, prefix_cache=False, executor=ex)
    be = S.maybe_batch_engine(engine, 4)
    assert isinstance(be, S.BatchEngine) and be._inkling
    got, texts = {}, {i: [] for i in range(3)}

    def ask(i, budget, seed=None):
        got[i] = be.generate(PROMPTS[i % 3], max_tokens=budget, temperature=0.0, stop=None,
                             seed=seed, on_text=texts.setdefault(i, []).append)

    try:
        threads = [threading.Thread(target=ask, args=(i, n)) for i, n in enumerate((9, 14, 6))]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=120)
        for i in range(3):
            assert got[i].token_ids == want[i].token_ids, i
            assert got[i].text == want[i].text == "".join(texts[i])
            assert got[i].prompt_tokens == len(PROMPTS[i])
        assert be.batch_stats["max_batch_seen"] >= 2
        assert be.batch_stats["batched_requests"] == 3
        assert engine.stats["requests"] == 3
        assert engine.stats["completion_tokens"] == sum(len(w.token_ids) for w in want)
        ask(3, 5, seed=7)                                   # PROMPTS[0], seeded: serial
        assert got[3].token_ids == want[0].token_ids[:5]
        assert be.batch_stats["serial_requests"] == 1
        assert be.metrics()["batching"]["max_batch"] == 4
    finally:
        be.close()


def _tiny_engine():
    """A baseline-mode engine on the tiny model, built on the thread that will generate."""
    from concurrent.futures import ThreadPoolExecutor

    from mlx_dspark import server as S

    ex = ThreadPoolExecutor(max_workers=1)
    t = ex.submit(lambda: _target()[0]).result()
    engine = S.Engine(t, t.tokenizer, None, mode="baseline", model_id="tiny", target_repo="tiny",
                      drafter_repo=None, max_draft_tokens=None, prefix_cache=False, executor=ex)
    return S, ex, engine


@metal
def test_work_for_the_generation_thread_runs_while_the_scheduler_holds_it():
    """A BatchEngine's scheduler occupies the engine's one thread for good; a race arm or the
    memory guard's idle shed submitted to that thread used to wait behind it forever."""
    import threading

    from mlx_dspark.memory_guard import MemoryGuard

    S, ex, engine = _tiny_engine()
    be = S.BatchEngine(engine, max_batch=4)
    try:
        assert isinstance(engine._executor, S._SchedulerThread)
        assert engine._executor.submit(threading.current_thread).result(timeout=20) is not None
        with pytest.raises(ZeroDivisionError):                # an error reaches the caller
            engine._executor.submit(lambda: 1 // 0).result(timeout=20)
        # the Lab's race: every arm is a call on the generation thread
        events = []
        racer = threading.Thread(target=lambda: be.race(
            PROMPTS[0], [{"mode": "baseline"}, {"mode": "lookup"}], 6,
            lambda kind, data: events.append(kind)))
        racer.start()
        racer.join(timeout=60)
        assert not racer.is_alive(), "the race never got the generation thread"
        assert events.count("arm_done") == 2 and events[-1] == "verdict"
        # the memory guard, built the way Engine.load builds it, sheds when idle
        guard = MemoryGuard(prefix=None, submit=lambda fn: engine._executor.submit(fn),
                            is_busy=lambda: engine._busy, poll=lambda: {"label": "normal"},
                            log=lambda *_: None)
        assert guard.request("warn")
        for _ in range(200):
            if guard.info()["sheds"]:
                break
            threading.Event().wait(0.05)
        assert guard.info()["sheds"] == 1
        # a call queued while requests decode runs when they are done, and they all finish
        got = {}
        asks = [threading.Thread(target=lambda i=i: got.__setitem__(i, be.generate(
            PROMPTS[i], max_tokens=20, temperature=0.0, stop=None, seed=None)))
            for i in range(2)]
        for th in asks:
            th.start()
        call = engine._executor.submit(lambda: "after")
        for th in asks:
            th.join(timeout=120)
        assert call.result(timeout=60) == "after" and len(got) == 2
    finally:
        be.close()
    assert engine._executor is ex                              # handed back on close
    assert ex.submit(lambda: 7).result(timeout=20) == 7


@metal
def test_a_newcomer_is_held_back_when_memory_is_short(monkeypatch):
    """A lone request is always served; another joins only while the OS can spare its caches."""
    import threading

    S, _, engine = _tiny_engine()
    be = S.BatchEngine(engine, max_batch=4)
    gb = 1 << 30
    job = S._Job(PROMPTS[1], {"max_tokens": 100}, None)
    try:
        be._memory = lambda: {"pressure": "normal", "total_bytes": 48 * gb, "free_percent": 40}
        assert be._has_room(job, 0) and be._has_room(job, 3)
        be._memory = lambda: {"pressure": "warn", "total_bytes": 48 * gb, "free_percent": 40}
        assert be._has_room(job, 0) and not be._has_room(job, 1)
        be._memory = lambda: {"pressure": "normal", "total_bytes": 32 * gb, "free_percent": 12}
        assert not be._has_room(job, 1)             # 3.8 GB left, 1 GB needed, 3.2 GB reserve
        assert not be._has_room(job, 2)
        assert be.batch_stats["held_for_memory"] == 1          # counted once per request
        be._memory = lambda: {"pressure": "normal"}            # the OS does not say: no gate
        assert be._has_room(job, 1)
        # end to end under pressure: three requests arrive together and take turns
        be._memory = lambda: {"pressure": "critical", "total_bytes": 48 * gb, "free_percent": 3}
        got = {}
        asks = [threading.Thread(target=lambda i=i: got.__setitem__(i, be.generate(
            PROMPTS[i], max_tokens=8, temperature=0.0, stop=None, seed=None)))
            for i in range(3)]
        for th in asks:
            th.start()
        for th in asks:
            th.join(timeout=120)
        want = be.engine._executor.submit(
            lambda: [_greedy(engine.target, PROMPTS[i], 8).token_ids for i in range(3)]).result(60)
        assert [got[i].token_ids for i in range(3)] == want
        assert be.batch_stats["max_batch_seen"] == 1 and be.batch_stats["batched_requests"] == 0
    finally:
        be.close()
