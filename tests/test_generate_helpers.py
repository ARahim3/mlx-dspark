"""Unit tests for the model-free helpers in generate.py: the streaming/stop machinery,
finish-reason logic, and chat-template result normalization. No weights required.
"""

from __future__ import annotations

import mlx_dspark.generate as g


class _FakeTok:
    """decode(ids) == the string whose chars have those code points."""

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _run(stop, rounds, eos=frozenset()):
    tok = _FakeTok()
    chunks = []
    st = g._Streamer(tok, set(eos), chunks.append, stop)
    out = []
    for r in rounds:
        out += [ord(c) for c in r]
        st.update(out)
        if st.stopped:
            break
    st.flush()
    return "".join(chunks), st.text, st.stopped


def test_stream_no_stop_emits_everything():
    streamed, text, stopped = _run(None, ["Hel", "lo ", "world"])
    assert streamed == "Hello world" and text == "Hello world" and not stopped


def test_stop_within_round_cuts_and_no_leak():
    streamed, text, stopped = _run(["STOP"], ["abc", "deSTOPxyz"])
    assert text == "abcde" and stopped
    assert "STOP" not in streamed and streamed == "abcde"


def test_stop_straddling_rounds_held_back():
    # "ST" is emitted in round 1 but must be held back until we know it's part of "STOP"
    streamed, text, stopped = _run(["STOP"], ["abST", "OPcd"])
    assert text == "ab" and stopped and streamed == "ab"


def test_earliest_of_multiple_stops_wins():
    _streamed, text, stopped = _run(["END", "STOP"], ["xxSTOPyyENDzz"])
    assert text == "xx" and stopped


def test_incremental_feed_matches_full_decode():
    # feed one token at a time (the greedy loop's pattern): streamed text, final text,
    # and a full decode must all agree, and eos ids must never render
    tok = _FakeTok()
    chunks = []
    st = g._Streamer(tok, {9999}, chunks.append, None)
    msg = "incremental detokenization, olé! → 対応"
    out = []
    for c in msg:
        out.append(ord(c))
        st.update(out)
    out.append(9999)  # trailing eos must be filtered
    st.update(out)
    st.flush()
    assert "".join(chunks) == msg and st.text == msg


def test_streamer_uses_streaming_detokenizer_for_fast_tokenizers():
    # a real byte-level-BPE fast tokenizer (the qwen-style case) must select mlx-lm's
    # BPE streaming detokenizer — not the O(n²) full-decode fallback — and produce
    # byte-identical text to tokenizer.decode
    __import__("pytest").importorskip("tokenizers")
    from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=[])
    tk.train_from_iterator(["hello world, hello streaming detokenizers!"] * 4, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tk)

    detok = g._make_detokenizer(fast)
    assert isinstance(detok, BPEStreamingDetokenizer)

    text = "hello world, hello streaming!"
    ids = fast.encode(text)
    chunks = []
    st = g._Streamer(fast, set(), chunks.append, None)
    out = []
    for i in ids:
        out.append(i)
        st.update(out)
    st.flush()
    assert st.text == fast.decode(ids)
    assert "".join(chunks) == st.text


def test_fallback_detokenizer_for_minimal_tokenizers():
    detok = g._make_detokenizer(_FakeTok())
    assert isinstance(detok, g._FullDecodeDetokenizer)


def test_stop_streaming_ends_generation_gracefully():
    # an on_text that raises StopStreaming (the server does this on client disconnect) must
    # flip `stopped` — so the loop ends normally — without propagating the exception
    chunks = []

    def on_text(piece):
        chunks.append(piece)
        if sum(map(len, chunks)) >= 3:
            raise g.StopStreaming()

    st = g._Streamer(_FakeTok(), set(), on_text, None)
    out = []
    for ch in "abcdefgh":
        out.append(ord(ch))
        st.update(out)
        if st.stopped:
            break
    st.flush()
    assert st.stopped and st.on_text is None
    assert "".join(chunks) == "abc"       # nothing streamed past the disconnect
    assert st.text.startswith("abc")      # partial text still recorded for the GenResult


class _StreamerLike:
    stopped = False


def test_finish_reason():
    s = _StreamerLike()
    assert g._finish_reason([1, 2, 3], 3, 9, {9}, s) == "stop"       # last token is eos
    stopped = _StreamerLike()
    stopped.stopped = True
    assert g._finish_reason([1, 2, 3], 100, 5, {9}, stopped) == "stop"  # a stop string hit
    assert g._finish_reason([1, 2, 3], 3, 5, {9}, _StreamerLike()) == "length"  # hit the cap
    assert g._finish_reason([1, 2], 3, 5, {9}, _StreamerLike()) == "stop"       # under cap, no eos


def test_topp_speculative_sampling_is_lossless():
    """The committed token must be an exact sample from top-p/top-k(softmax(target/T)),
    independent of the (deliberately mismatched) draft distribution q. This is the core
    losslessness guarantee for temperature + nucleus sampling."""
    import mlx.core as mx
    import numpy as np

    from mlx_dspark.generate import _spec_sample_accept
    from mlx_dspark.sampling import sample_probs, truncate_probs

    mx.random.seed(0)
    V = 6
    target = mx.array([2.0, 1.0, 0.5, 0.0, -1.0, -2.0])
    q = mx.softmax(mx.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]), axis=-1)  # ~reverse of target
    v_logits = mx.stack([target, target], axis=0)
    qrow = q.reshape(1, V)

    for (T, tp, tk) in [(1.0, 1.0, 0), (1.0, 0.8, 0), (0.7, 1.0, 3)]:
        exp = np.array(truncate_probs(mx.softmax(target / T, axis=-1), tp, tk).tolist())
        counts = np.zeros(V)
        for _ in range(20000):
            x = int(sample_probs(q).item())
            n, repl = _spec_sample_accept(v_logits, [x], qrow, T, tp, tk)
            counts[x if n == 1 else repl] += 1
        emp = counts / counts.sum()
        assert np.abs(emp - exp).max() < 0.02, (T, tp, tk, emp, exp)


def test_ids_from_template_result_shapes():
    assert g._ids_from_template_result([1, 2, 3]) == [1, 2, 3]
    assert g._ids_from_template_result([[4, 5, 6]]) == [4, 5, 6]

    class BatchEncoding(dict):
        pass

    be = BatchEncoding(input_ids=[[7, 8, 9]])
    assert g._ids_from_template_result(be) == [7, 8, 9]


# --------------------------------------------------------------------------- stop tokens


class _MarkerTok:
    """Tokenizer stub with a fixed vocab of special markers (ids mirror Gemma-4's)."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.unk_token_id = 3
        self.eos_token_id = 1

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, self.unk_token_id)


def test_eos_includes_gemma4_tool_response_marker():
    """After a tool call Gemma-4 emits <|tool_response> instead of <turn|>, handing back to
    the harness. Its own response grammar terminates on either; if we don't stop there the
    model hallucinates the tool result and keeps going until max_tokens — which a tool-calling
    agent hits on every turn."""
    from mlx_dspark.generate import eos_token_ids

    ids = eos_token_ids(_MarkerTok({"<turn|>": 106, "<|tool_response>": 50, "<|turn>": 105}))
    assert 106 in ids and 50 in ids
    assert 105 not in ids            # turn *opener* must not stop generation


def test_eos_filters_unknown_markers_to_unk():
    from mlx_dspark.generate import eos_token_ids

    ids = eos_token_ids(_MarkerTok({"<|im_end|>": 151645}))
    assert 151645 in ids
    assert 3 not in ids              # everything else resolves to unk and is dropped


# --- prefill marks: mid-prefill snapshot positions for checkpoint prefix caching ---------


def test_mark_stops_are_suffix_relative_and_interior():
    from mlx_dspark.generate import _mark_stops

    assert _mark_stops([8, 16, 24], base=5, n=15) == [3, 11]   # 24-5=19 > 15 excluded...
    assert _mark_stops(None, base=0, n=10) == []
    assert _mark_stops([5], base=5, n=10) == []                # at the base: nothing to split
    assert _mark_stops([15], base=5, n=10) == []               # the end is not a mark stop


def test_prefill_plain_splits_chunks_at_marks_and_reports_positions():
    import mlx.core as mx

    from mlx_dspark.generate import _prefill_plain

    class Layer:
        def __init__(self):
            self.offset = 0

    class Tgt:
        def __init__(self):
            self.chunks = []

        def prefill(self, ids, cache, tap=None, want_logits=True, head_last_row=True):
            n = ids.shape[1]
            self.chunks.append(n)
            for c in cache:
                c.offset += n
            return (mx.zeros((1, 1, 4)) if want_logits else None), None

    tgt = Tgt()
    cache = [Layer()]
    seen = []
    ids = list(range(100, 112))                     # 12 suffix tokens after base 5
    _prefill_plain(tgt, ids, cache, chunk=8, base=5, marks=[8, 14],
                   on_mark=lambda p: seen.append((p, cache[0].offset)))
    # chunks split at the marks (8-5=3, 14-5=9); chunk is a max piece size, not a grid
    assert tgt.chunks == [3, 6, 3]
    # on_mark fires with the caches holding exactly the first `pos` tokens (offset is
    # suffix-relative here: 3 and 9 of the 12)
    assert seen == [(8, 3), (14, 9)]


def test_prefill_plain_on_chunk_reports_progress_after_each_evaluated_chunk():
    # Prefill progress (issue #29): on_chunk fires with the ABSOLUTE prompt position after
    # each NON-final chunk — the chunks the loop already mx.eval's — and never for the
    # final chunk (generation starting is its completion signal) or a single-chunk prompt.
    import mlx.core as mx

    from mlx_dspark.generate import _prefill_plain

    class Layer:
        def __init__(self):
            self.offset = 0

    class Tgt:
        def prefill(self, ids, cache, tap=None, want_logits=True, head_last_row=True):
            for c in cache:
                c.offset += ids.shape[1]
            return (mx.zeros((1, 1, 4)) if want_logits else None), None

    seen = []
    _prefill_plain(Tgt(), list(range(20)), [Layer()], chunk=8, base=100,
                   on_chunk=seen.append)
    assert seen == [108, 116]                       # absolute; final chunk (120) not reported
    seen = []
    _prefill_plain(Tgt(), list(range(5)), [Layer()], chunk=8, base=0,
                   on_chunk=seen.append)
    assert seen == []                               # single-chunk prompt: no progress events


def test_roundlog_publish_fans_out_without_recording():
    # publish() (prefill progress) reaches /events subscribers but must not enter round
    # history or any aggregate — /rounds and /metrics stay rounds-only.
    from mlx_dspark.telemetry import RoundLog

    log = RoundLog()
    q = log.subscribe()
    log.publish({"type": "prefill", "req": "r1", "processed": 2048, "total": 9000})
    ev = q.get_nowait()
    assert ev["type"] == "prefill" and ev["processed"] == 2048 and "t" in ev
    assert log.snapshot() == []                     # not recorded
    assert log.stats()["rounds"] == 0


# ------------------------------------------------------- enable_thinking force-close (LFM2.5)


class _CharTemplateTok:
    """Chat-template stub with a 1-char-per-token scheme so decode/encode round-trip.

    ``suffix(enable_thinking)`` decides the generation-prompt tail; subclasses override it to
    model each template family (LFM2.5 always opens ``<think>``; Qwen3 closes it; instruct
    emits none)."""

    chat_template = "present"

    def suffix(self, enable_thinking):
        return "<think>"                                        # LFM2.5: always opens, ignores flag

    def apply_chat_template(self, messages, add_generation_prompt=True, **kw):
        text = "u:" + messages[-1]["content"] + "\na\n"
        if add_generation_prompt:
            text += self.suffix(kw.get("enable_thinking"))
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


def _decode_all(tok, ids):
    return "".join(chr(i) for i in ids)


def test_force_close_thinking_fires_on_open_think():
    tok = _CharTemplateTok()
    msgs = [{"role": "user", "content": "hi"}]
    a = g.encode_messages(tok, msgs)                            # thinking on (default)
    b = g.encode_messages(tok, msgs, enable_thinking=False)     # force-close
    assert _decode_all(tok, a).endswith("<think>")             # template always opens
    assert _decode_all(tok, b).endswith("<think></think>\n\n")  # closed by the fix
    assert _decode_all(tok, b).count("</think>") == 1


def test_force_close_thinking_noop_when_template_closes():
    class _Qwen3Like(_CharTemplateTok):
        def suffix(self, enable_thinking):
            return "<think>\n\n</think>\n\n" if enable_thinking is False else "<think>\n"

    tok = _Qwen3Like()
    b = g.encode_messages(tok, [{"role": "user", "content": "hi"}], enable_thinking=False)
    # template already closed the block -> the fix must NOT append a second </think>
    assert _decode_all(tok, b).count("</think>") == 1


def test_force_close_thinking_noop_when_no_think():
    class _Instruct(_CharTemplateTok):
        def suffix(self, enable_thinking):
            return ""                                          # instruct: never opens a block

    tok = _Instruct()
    msgs = [{"role": "user", "content": "hi"}]
    a = g.encode_messages(tok, msgs)
    b = g.encode_messages(tok, msgs, enable_thinking=False)
    assert a == b                                              # no-op, identical prompts


def test_force_close_thinking_skipped_without_generation_prompt():
    # Rendering history (add_generation_prompt=False) must never inject a closer.
    tok = _CharTemplateTok()
    ids = g.encode_messages(tok, [{"role": "user", "content": "hi"}],
                            add_generation_prompt=False, enable_thinking=False)
    assert "</think>" not in _decode_all(tok, ids)
