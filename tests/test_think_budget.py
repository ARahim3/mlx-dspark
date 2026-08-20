"""Reasoning-budget tests: the ThinkBudget tracker (detection, forced-close ids) and the
forced-injection mechanics inside every generation loop, driven by scripted fake targets and
a fake DSpark/DFlash drafter. No weights required.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from mlx_dspark.generate import (
    DEFAULT_BUDGET_MESSAGE,
    ThinkBudget,
    dflash_generate,
    greedy_generate,
    speculative_generate,
)
from mlx_dspark.lookup import lookup_generate

V = 512   # fake vocab: ids are character code points (ASCII only)

FORCED_TEXT = "\n\n" + DEFAULT_BUDGET_MESSAGE + "\n</think>\n\n"


class _Tok:
    """1 id per character; convert_tokens_to_ids knows nothing (or one closer id)."""

    unk_token_id = 0
    eos_token_id = 1

    def __init__(self, closer_id=None):
        self.closer_id = closer_id

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)

    def convert_tokens_to_ids(self, t):
        if self.closer_id is not None and t == "</think>":
            return self.closer_id
        return self.unk_token_id


def _ids(text):
    return [ord(c) for c in text]


def _feed(tb, text, per=1):
    """Feed `text` to the tracker `per` tokens at a time; returns (out_ids, fired_at) where
    fired_at is the out_ids length at the first True from note(), or None."""
    out = []
    for i in range(0, len(text), per):
        out += _ids(text[i:i + per])
        if tb.note(out):
            return out, len(out)
    return out, None


# --------------------------------------------------------------------- ThinkBudget unit


def test_fires_at_budget_and_builds_the_forced_close():
    tb = ThinkBudget(_Tok(), budget=10)
    out, fired_at = _feed(tb, "<think>" + "x" * 30)
    # opener completes at 7 tokens; 10 thinking tokens later -> fires at 17
    assert fired_at == 17
    forced = tb.take_forced_ids(remaining=1000)
    assert _Tok().decode(forced) == FORCED_TEXT
    assert tb.fired
    assert not tb.note(out)                       # once-only: disarmed after firing


def test_opener_straddling_note_calls_is_still_seen():
    tb = ThinkBudget(_Tok(), budget=5)
    out = _ids("<th")
    assert not tb.note(out)
    out += _ids("ink>")
    assert not tb.note(out)
    out += _ids("abcde")
    assert tb.note(out)                            # 5 tokens past the (straddled) opener


def test_note_is_idempotent():
    tb = ThinkBudget(_Tok(), budget=3)
    out = _ids("<think>")
    assert not tb.note(out) and not tb.note(out)   # same out_ids -> same answer, twice
    out = out + _ids("abc")
    assert tb.note(out) and tb.note(out)


def test_holds_fire_while_the_tail_might_be_the_closer():
    # budget exceeded exactly while the model is mid-`</think>`: wait one round rather
    # than injecting a second close into a block that is closing itself
    tb = ThinkBudget(_Tok(), budget=3)
    out = _ids("<think>abc</thi")
    for n in range(8, len(out) + 1):
        assert not tb.note(out[:n])
    assert tb.note(out + _ids("s is fine"))        # a false alarm: it WAS thinking text
    tb2 = ThinkBudget(_Tok(), budget=3)
    out2 = _ids("<think>abc</thi")
    assert not tb2.note(out2)
    assert not tb2.note(out2 + _ids("nk>done"))    # it WAS the closer: disarmed


def test_prefilled_opener_counts_from_token_zero():
    # Qwen3-2507-style: the template ends the prompt with the opener, so the output never
    # contains one — auto-detected from the prompt tail, counting starts at token 0.
    tb = ThinkBudget(_Tok(), budget=4, prompt_tail_ids=_ids("Q: hi\n<think>"))
    _out, fired_at = _feed(tb, "yyyyyyyy")
    assert fired_at == 4


def test_gemma_pair_discovers_its_own_closer():
    tb = ThinkBudget(_Tok(), budget=3)
    _out, fired_at = _feed(tb, "<|channel>thought\n" + "z" * 10)
    assert fired_at is not None
    assert tb.close_marker == "<channel|>"
    forced = tb.take_forced_ids(remaining=1000)
    assert _Tok().decode(forced).endswith("<channel|>\n\n")


def test_model_closing_its_own_block_disarms():
    tb = ThinkBudget(_Tok(), budget=5)
    _out, fired_at = _feed(tb, "<think>ab</think>the answer, at length" + "!" * 30)
    assert fired_at is None and not tb.fired


def test_plain_answer_without_thinking_never_fires():
    tb = ThinkBudget(_Tok(), budget=2)
    _out, fired_at = _feed(tb, "just prose, no thinking block here at all")
    assert fired_at is None


def test_eos_ids_are_filtered_from_the_forced_close():
    eos = {ord("n")}
    tb = ThinkBudget(_Tok(), budget=2, eos_ids=eos)
    _feed(tb, "<think>xxxx")
    forced = tb.take_forced_ids(remaining=1000)
    assert not set(forced) & eos
    assert _Tok().decode(forced) == FORCED_TEXT.replace("n", "")


def test_insufficient_remaining_skips_injection_without_arming():
    tb = ThinkBudget(_Tok(), budget=2)
    _feed(tb, "<think>xxxx")
    assert tb.take_forced_ids(remaining=3) == []   # never truncate the closer
    assert not tb.fired
    assert tb.take_forced_ids(remaining=1000)      # a later, roomier call still fires


def test_non_positive_budget_is_rejected():
    import pytest

    for bad in (0, -1):
        with pytest.raises(ValueError):
            ThinkBudget(_Tok(), budget=bad)


def test_empty_message_closes_the_block_with_no_message():
    tb = ThinkBudget(_Tok(), budget=2, message="")
    _feed(tb, "<think>xxxx")
    forced = tb.take_forced_ids(remaining=1000)
    assert _Tok().decode(forced) == "\n</think>\n\n"


def test_custom_message_and_single_token_closer():
    tb = ThinkBudget(_Tok(closer_id=400), budget=2, message="Stop.")
    _feed(tb, "<think>xxxx")
    forced = tb.take_forced_ids(remaining=1000)
    assert forced.count(400) == 1                  # </think> resolved as ONE special token
    assert _ids("</think>")[0] not in forced or forced[forced.index(400) - 1] != ord("k")
    text = "".join(chr(i) for i in forced if i != 400)
    assert text == "\n\nStop.\n" + "\n\n"


# ------------------------------------------------------------------ scripted fake target


class _ScriptTgt:
    """Greedy target that 'wants' to emit ``full_text[len(prompt):]``: the argmax after
    consuming T tokens is full_text[T] (pad past the end; pad is never eos). Records every
    token fed through the cache plus verify/rollback events, so tests can assert exactly
    what the KV cache saw."""

    def __init__(self, full_text, pad="A"):
        self.full = _ids(full_text)
        self.pad = ord(pad)
        self.consumed = []       # the cache contents, in order
        self.events = []         # ("verify", width) / ("rollback", n_rejected)

    def make_cache(self):
        return []

    def reset_spec(self):
        pass

    def _pred(self, t):
        return self.full[t] if t < len(self.full) else self.pad

    def _logits(self, preds):
        rows = []
        for p in preds:
            r = [0.0] * V
            r[p] = 10.0
            rows.append(r)
        return mx.array([rows])

    def _consume(self, ids):
        toks = [int(x) for x in ids[0].tolist()]
        preds = []
        for t in toks:
            self.consumed.append(t)
            preds.append(self._pred(len(self.consumed)))
        return self._logits(preds)

    def prefill(self, ids, cache, tap=None, want_logits=True, head_last_row=True):
        logits = self._consume(ids)
        fused = mx.zeros((1, logits.shape[1], 4)) if tap is not None else None
        return (logits[:, -1:, :] if want_logits else None), fused

    def plain(self, ids, cache):
        return self._consume(ids)

    def verify(self, ids, cache, tap):
        self.events.append(("verify", int(ids.shape[1])))
        logits = self._consume(ids)
        fused = mx.zeros((1, logits.shape[1], 4)) if tap is not None else None
        return logits, fused

    def rollback(self, cache, n_rejected, accepted):
        self.events.append(("rollback", int(n_rejected)))
        if n_rejected:
            del self.consumed[-n_rejected:]


PROMPT = "PQ"
THINK_FOREVER = PROMPT + "<think>" + "x" * 200     # opens a block and never closes it


# --------------------------------------------------------------- greedy_generate loops


def test_greedy_pipelined_injects_and_continues():
    tgt = _ScriptTgt(THINK_FOREVER)
    res = greedy_generate(tgt, _Tok(), prompt_ids=_ids(PROMPT), max_new_tokens=70,
                          think_budget=10)
    # opener = 7 tokens, budget 10 -> injection right after committed token 17
    assert res.budget_forced
    assert res.text.startswith("<think>" + "x" * 10 + FORCED_TEXT)
    # the cache saw the forced tokens, contiguously, right where the text has them
    assert FORCED_TEXT in _Tok().decode(tgt.consumed)
    assert res.finish_reason == "length" and res.num_tokens == 70
    # accounting: one injection == one forward that committed len(FORCED_TEXT) tokens,
    # recorded chronologically — 17 plain rounds, then the injection, then plain again
    assert res.target_forwards == res.num_rounds == 70 - len(FORCED_TEXT) + 1
    assert res.accept_lengths[17] == len(FORCED_TEXT)
    assert set(res.accept_lengths[:17]) == {1} and set(res.accept_lengths[18:]) == {1}


def test_greedy_sequential_branch_injects_with_aligned_logprobs():
    tgt = _ScriptTgt(THINK_FOREVER)
    res = greedy_generate(tgt, _Tok(), prompt_ids=_ids(PROMPT), max_new_tokens=60,
                          think_budget=10, logprobs=0)
    assert res.budget_forced and FORCED_TEXT in res.text
    assert len(res.logprobs) == res.num_tokens == 60
    assert [e["token_id"] for e in res.logprobs] == res.token_ids


def test_greedy_injection_at_the_exact_remaining_boundary_terminates_cleanly():
    tgt = _ScriptTgt(THINK_FOREVER)
    cap = 17 + len(FORCED_TEXT)          # fires at 17; forced close exactly fills the rest
    res = greedy_generate(tgt, _Tok(), prompt_ids=_ids(PROMPT), max_new_tokens=cap,
                          think_budget=10)
    assert res.budget_forced and res.num_tokens == cap
    assert res.finish_reason == "length"
    assert res.text.endswith("</think>\n\n")
    assert len(tgt.consumed) == len(PROMPT) + cap    # not one token generated past the cap


def test_greedy_one_token_short_of_the_close_skips_injection():
    tgt = _ScriptTgt(THINK_FOREVER)
    cap = 17 + len(FORCED_TEXT) - 1
    res = greedy_generate(tgt, _Tok(), prompt_ids=_ids(PROMPT), max_new_tokens=cap,
                          think_budget=10)
    assert not res.budget_forced and FORCED_TEXT not in res.text
    assert res.finish_reason == "length" and res.num_tokens == cap


def test_greedy_stop_string_inside_the_forced_text_stops_generation():
    tgt = _ScriptTgt(THINK_FOREVER)
    res = greedy_generate(tgt, _Tok(), prompt_ids=_ids(PROMPT), max_new_tokens=200,
                          think_budget=10, stop=["answer now"])
    assert res.budget_forced
    assert res.finish_reason == "stop" and "answer now" not in res.text
    assert res.num_tokens == 17 + len(FORCED_TEXT)   # committed the close, then stopped


def test_greedy_prefilled_opener_counts_from_the_first_generated_token():
    prompt = "hi\n<think>"
    tgt = _ScriptTgt(prompt + "x" * 200)
    res = greedy_generate(tgt, _Tok(), prompt_ids=_ids(prompt), max_new_tokens=70,
                          think_budget=10)
    assert res.budget_forced
    assert res.text.startswith("x" * 10 + FORCED_TEXT)


def test_greedy_without_budget_is_untouched():
    tgt = _ScriptTgt(THINK_FOREVER)
    res = greedy_generate(tgt, _Tok(), prompt_ids=_ids(PROMPT), max_new_tokens=30)
    assert not res.budget_forced and res.text == "<think>" + "x" * 23
    assert res.target_forwards == 30 and res.accept_lengths == [1] * 30


# --------------------------------------------------------------- lookup_generate loop


def test_lookup_injects_through_one_forced_verify_with_zero_rollback():
    tgt = _ScriptTgt(THINK_FOREVER)
    res = lookup_generate(tgt, _Tok(), prompt_ids=_ids(PROMPT), max_new_tokens=90,
                          think_budget=10)
    assert res.budget_forced and FORCED_TEXT in res.text
    # the injection is one verify of exactly [pending] + forced[:-1] rows...
    w = len(FORCED_TEXT)
    idx = tgt.events.index(("verify", w))
    # ...committed unconditionally: the very next rollback rejects nothing
    nxt = next(e for e in tgt.events[idx + 1:] if e[0] == "rollback")
    assert nxt == ("rollback", 0)
    assert FORCED_TEXT in _Tok().decode(tgt.consumed)


# ------------------------------------------------------- speculative_generate (DSpark)


class _CtxCache:
    def __init__(self):
        self.k = mx.zeros((1,))
        self.offset = 0


class _SpecDrafter:
    """Drafts a constant guess token; records every update_context (offset, rows) so the
    ctx-alignment invariant can be checked across an injection."""

    max_draft = 3
    confidence_head = None

    def __init__(self, guess="x"):
        self.config = SimpleNamespace(target_layer_ids=[0], mask_token_id=0,
                                      has_own_lm_head=True, has_own_embed=True)
        self.guess = ord(guess)
        self.ctx_updates = []      # (ctx_offset, n_rows)

    def make_ctx_cache(self):
        return [_CtxCache()]

    def draft_width(self, cap):
        return cap

    def embed(self, block):
        return mx.zeros((1, int(block.shape[1]), 4))

    def backbone(self, noise, n_cached, ctx_caches):
        return noise

    def head_slice(self, hidden, cap):
        return hidden[:, :cap]

    def compute_logits(self, hidden):
        L = int(hidden.shape[1])
        rows = []
        for _ in range(L):
            r = [0.0] * V
            r[self.guess] = 10.0
            rows.append(r)
        return mx.array([rows])

    def sample_block(self, base_logits, first_prev_token=None):
        return mx.argmax(base_logits, axis=-1)

    def update_context(self, fused, ctx_offset=0, ctx_caches=None):
        self.ctx_updates.append((int(ctx_offset), int(fused.shape[1])))


def test_speculative_injection_keeps_drafter_ctx_offsets_aligned():
    tgt = _ScriptTgt(THINK_FOREVER)
    drafter = _SpecDrafter()
    res = speculative_generate(tgt, _Tok(), drafter, prompt_ids=_ids(PROMPT),
                               max_new_tokens=120, max_draft_tokens=2,
                               lookup_drafts=False, think_budget=10)
    assert res.budget_forced and FORCED_TEXT in res.text
    # ctx rows must tile [0, n_cached) contiguously: every update starts exactly where the
    # previous one ended — a dropped or double-fed injection breaks this immediately
    pos = 0
    for off, rows in drafter.ctx_updates:
        assert off == pos, drafter.ctx_updates
        pos += rows
    # everything committed except the final pending token is in the drafter ctx
    assert pos == len(PROMPT) + res.num_tokens - 1
    # the injection round is visible as one forward committing the whole forced close
    assert len(FORCED_TEXT) in res.accept_lengths
    # and its verify was committed unconditionally (rollback 0 right after the wide verify)
    idx = tgt.events.index(("verify", len(FORCED_TEXT)))
    assert tgt.events[idx + 1] == ("rollback", 0)


def test_speculative_without_budget_is_untouched():
    tgt = _ScriptTgt(THINK_FOREVER)
    res = speculative_generate(tgt, _Tok(), _SpecDrafter(), prompt_ids=_ids(PROMPT),
                               max_new_tokens=40, max_draft_tokens=2, lookup_drafts=False)
    assert not res.budget_forced and FORCED_TEXT not in res.text


# --------------------------------------------------------------- dflash_generate loop


class _DFlashDrafter:
    """Block-diffusion-shaped fake: callable, drafts a constant guess; records the width of
    every pending_ctx it is handed (DFlash appends pending_ctx to its draft cache at the
    NEXT call, so an injection must concatenate onto it, not replace it)."""

    embed_tokens = object()      # non-None: skip bind()

    def __init__(self, guess="x", block_size=4):
        self.config = SimpleNamespace(target_layer_ids=[0], block_size=block_size,
                                      mask_token_id=0)
        self.guess = ord(guess)
        self.ctx_widths = []

    def make_cache(self):
        return [SimpleNamespace(offset=0)]

    def __call__(self, block, pending_ctx, dcache, logits_start=1):
        self.ctx_widths.append(int(pending_ctx.shape[1]))
        k = int(block.shape[1]) - logits_start
        rows = []
        for _ in range(k):
            r = [0.0] * V
            r[self.guess] = 10.0
            rows.append(r)
        return mx.array([rows])


def test_dflash_injection_concatenates_onto_the_deferred_pending_ctx():
    tgt = _ScriptTgt(THINK_FOREVER)
    drafter = _DFlashDrafter()
    res = dflash_generate(tgt, _Tok(), drafter, prompt_ids=_ids(PROMPT),
                          max_new_tokens=120, think_budget=10)
    assert res.budget_forced and FORCED_TEXT in res.text
    # the round after the injection must see the PREVIOUS round's rows plus the forced
    # rows — a plain overwrite would show len(FORCED_TEXT) alone and lose committed ctx
    w = len(FORCED_TEXT)
    k = res.accept_lengths.index(w)                # the injection "round"
    assert k > 0
    expected = res.accept_lengths[k - 1] + w
    assert expected in drafter.ctx_widths, (drafter.ctx_widths, res.accept_lengths)
    # every committed token's ctx row reaches the drafter exactly once: widths tile the
    # sequence [prompt .. last uncommitted pending)
    assert sum(drafter.ctx_widths[:-1]) <= len(PROMPT) + res.num_tokens - 1
    idx = tgt.events.index(("verify", w))
    assert tgt.events[idx + 1] == ("rollback", 0)


def test_dflash_without_budget_is_untouched():
    tgt = _ScriptTgt(THINK_FOREVER)
    res = dflash_generate(tgt, _Tok(), _DFlashDrafter(), prompt_ids=_ids(PROMPT),
                          max_new_tokens=40)
    assert not res.budget_forced and FORCED_TEXT not in res.text
