"""Prompt-lookup (n-gram) drafting — drafter-free speculative decoding for ANY target.

The oldest trick in speculative decoding (Saxena's prompt-lookup decoding; llama.cpp's
``lookup``): when generating text that copies from its own context — quoting a RAG passage,
editing code, summarizing a document — the continuation often already exists earlier in the
sequence. So: find the most recent earlier occurrence of the current suffix n-gram, propose
the tokens that followed it, and let the target verify them exactly like any other draft.
No drafter model, no hidden-state tap (the plain forward suffices), works with every target
mlx-lm/mlx-vlm can load.

Losslessness is inherited from the verify loop: greedy accept is exact argmax match, and at
temperature > 0 the proposal is a point mass (q = one-hot), so the standard accept rule
``min(1, p/q)`` reduces to accepting with probability p(x) with the residual
``norm(max(0, p - q))`` = p renormalized without x — still an exact sample from the target.

Drafts only fire on a confident match (>= ``ngram_min`` suffix tokens, latest earlier
occurrence preferred); otherwise the round is a plain 1-token forward, so the miss cost is
zero. On a hit the cost is one wider forward — the same convex verify curve as the drafter
modes (knee at width ~4 on M-series; keep ``max_draft`` modest on big targets). The default
minimum is a **trigram**: bigrams fire constantly on natural chat text and their rejected
drafts cost wider forwards (measured net-negative on M-series), while genuine copying
produces trigram matches in abundance.
"""

from __future__ import annotations

import time

import mlx.core as mx

from .generate import (
    GenResult,
    _finish_reason,
    _make_target_cache,
    _pick,
    _prefill_plain,
    _spec_sample_accept,
    _Streamer,
    encode_prompt,
    eos_token_ids,
)


class NGramIndex:
    """Incremental n-gram → position index over the growing token sequence.

    For every completed n-gram (``min_n <= n <= max_n``) it keeps the positions of the two
    most recent occurrences, so a query for the *current* suffix can skip the occurrence
    that is the suffix itself and return the latest genuinely earlier one (recency beats
    the first match for chat: the most recent repetition is the likeliest continuation).
    """

    def __init__(self, min_n: int = 3, max_n: int = 4, max_draft: int = 6):
        self.min_n = max(1, min_n)
        self.max_n = max(self.min_n, max_n)
        self.max_draft = max(1, max_draft)
        self.tokens: list[int] = []
        # ngram tuple -> (previous end position, latest end position); "end position" is the
        # index right AFTER the ngram, i.e. where its continuation starts.
        self._index: dict[tuple, tuple[int | None, int]] = {}

    def extend(self, new_tokens: list[int]) -> None:
        for t in new_tokens:
            self.tokens.append(t)
            end = len(self.tokens)
            for n in range(self.min_n, self.max_n + 1):
                if end >= n:
                    ng = tuple(self.tokens[end - n:end])
                    prev = self._index.get(ng)
                    self._index[ng] = (prev[1] if prev else None, end)

    def propose(self, long_draft: int | None = None) -> list[int]:
        """Draft tokens continuing the latest earlier occurrence of the current suffix
        n-gram (longest n first). Empty list = no confident match, do a plain step.

        ``long_draft`` (> ``max_draft``) enables **match-scaled drafting** (after llama.cpp's
        ngram-mod, which requires a 24-token matched context before drafting 48-64 tokens):
        the suffix match is extended *backwards* token-by-token to measure how much context
        actually matches the earlier occurrence, and a match of ``m >= 8`` tokens earns a
        ``2*m``-token draft (clipped to ``long_draft``). Deep inside a verbatim copy run the
        evidence — and so the draft — grows to the ceiling, where the M-series verify-width
        plateau makes the extra rows near-free (8-bit, M4 Pro: width 16-32 costs ~2.5x one
        step); a bare 4-5-gram hit stays at ``max_draft``, so insertion-heavy edits and chat
        behave exactly as before. Purely a length policy: output is unchanged either way."""
        end = len(self.tokens)
        for n in range(self.max_n, self.min_n - 1, -1):
            if end < n:
                continue
            hit = self._index.get(tuple(self.tokens[end - n:end]))
            if hit is None:
                continue
            prev, latest = hit
            pos = latest if latest != end else prev   # skip the suffix's own occurrence
            if pos is None:
                continue
            cap = self.max_draft
            if long_draft is not None and long_draft > cap:
                m = n                            # matched context: the n-gram itself ...
                i, j = end - n - 1, pos - n - 1  # ... extended backwards while it keeps matching
                while j >= 0 and m < long_draft and self.tokens[i] == self.tokens[j]:
                    m += 1
                    i -= 1
                    j -= 1
                if m >= 8:
                    cap = min(long_draft, 2 * m)
            draft = self.tokens[pos:pos + max(1, cap)]
            if draft:
                return draft
        return []


class LongDraftGate:
    """Acceptance guard for match-scaled long drafts (after llama.cpp ngram-mod's
    low-acceptance-streak reset). Deep backward match is *usually* a real copy run — but at
    an insertion point (docstring pass, per-line edits) the evidence is deepest exactly
    where the continuation diverges, and every long draft gets chopped early. Two
    consecutive long drafts accepted under half their length park the scaling; a single
    long probe every ``probe_every`` lookup rounds re-enters when the content turns
    copy-friendly again (same parked-probe pattern as the auto-cap controller). Base-length
    lookup drafts are never gated. Speed-only: output is unchanged either way."""

    def __init__(self, probe_every: int = 8):
        self.probe_every = probe_every
        self.fails = 0
        self.rounds_since = 0     # base-length lookup rounds since the last long draft

    @property
    def allowed(self) -> bool:
        return self.fails < 2 or self.rounds_since >= self.probe_every

    def update(self, drafted: int, accepted: int, base: int) -> None:
        """Feed every lookup round's (drafted, accepted); ``base`` = the un-scaled length."""
        if drafted <= base:
            self.rounds_since += 1
            return
        self.fails = self.fails + 1 if 2 * accepted < drafted else 0
        self.rounds_since = 0


def lookup_generate(
    target_model,
    tokenizer,
    prompt: str = "",
    *,
    prompt_ids: list[int] | None = None,
    cache=None,
    reuse_len: int = 0,
    max_new_tokens: int = 128,
    max_draft_tokens: int = 6,
    long_draft_tokens: int = 32,
    ngram_min: int = 3,
    ngram_max: int = 4,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    seed: int | None = None,
    apply_chat_template: bool = True,
    stop: list[str] | None = None,
    on_text=None,
) -> GenResult:
    """Prompt-lookup speculative decoding (batch=1) — no drafter model.

    Same contract as the drafter loops: greedy (``temperature == 0``) output equals plain
    greedy decoding up to fp ties; ``temperature > 0`` is an exact sample from the target
    (one-hot-proposal speculative sampling). ``cache``/``reuse_len`` support prefix caching
    exactly like ``greedy_generate`` (plain KV cache, no drafter state to roll back).

    ``long_draft_tokens``: match-scaled long-draft ceiling — a deep context match (a real
    copy run) earns drafts up to this length; see :meth:`NGramIndex.propose`. Set equal to
    ``max_draft_tokens`` to disable.
    """
    if seed is not None:
        mx.random.seed(seed)
    eos_ids = eos_token_ids(tokenizer)
    ids = prompt_ids if prompt_ids is not None else encode_prompt(
        tokenizer, prompt, use_chat=apply_chat_template)
    if cache is None:
        cache = _make_target_cache(target_model)
        reuse_len = 0
    target_model.reset_spec()                          # hybrid targets: clear capture state
    st = _Streamer(tokenizer, eos_ids, on_text, stop)
    base_draft = max(1, max_draft_tokens)
    index = NGramIndex(min_n=ngram_min, max_n=ngram_max, max_draft=base_draft)
    index.extend(ids)
    gate = LongDraftGate()

    t0 = time.time()
    suffix = ids[reuse_len:] if reuse_len else ids
    logits = _prefill_plain(target_model, suffix, cache)
    pending = _pick(logits[0, -1], temperature, top_p, top_k)
    out_ids: list[int] = [pending]
    index.extend([pending])
    accept_lengths: list[int] = []
    target_forwards = 1

    st.update(out_ids)
    while len(out_ids) < max_new_tokens and pending not in eos_ids and not st.stopped:
        # clamp to the remaining budget: no verify width wasted past max_new_tokens
        draft = index.propose(
            long_draft=long_draft_tokens if gate.allowed else None,
        )[:max_new_tokens - len(out_ids)]

        if not draft:
            # no confident match -> plain 1-token step (zero miss cost; verify() with no
            # draft tokens is exactly a plain committed step for every family)
            logits, _ = target_model.verify(mx.array([[pending]]), cache, None)
            committed = [_pick(logits[0, -1], temperature, top_p, top_k)]
            n = 0
        elif temperature > 0.0:
            verify_ids = mx.array([[pending] + draft])
            v_logits, _ = target_model.verify(verify_ids, cache, None)
            mx.eval(v_logits)
            # point-mass proposal: q is one-hot at each drafted token
            vocab = v_logits.shape[-1]
            q_probs = (mx.arange(vocab)[None, :] == mx.array(draft)[:, None]).astype(mx.float32)
            n, repl = _spec_sample_accept(v_logits[0], draft, q_probs, temperature, top_p, top_k)
            committed = draft[:n] + [repl]
        else:
            # fused greedy verify: single sync per round (same pattern as the drafter loops)
            draft_arr = mx.array(draft)
            verify_ids = mx.concatenate(
                [mx.array([pending], dtype=draft_arr.dtype), draft_arr]).reshape(1, -1)
            v_logits, _ = target_model.verify(verify_ids, cache, None)
            tt_arr = mx.argmax(v_logits[0], axis=-1)
            match = (draft_arr == tt_arr[: len(draft)].astype(draft_arr.dtype)).astype(mx.int32)
            n_arr = mx.cumprod(match).sum()
            mx.eval(n_arr, tt_arr)
            n = int(n_arr.item())
            tt = tt_arr.tolist()
            committed = draft[:n] + [tt[n]]
        target_forwards += 1
        accept_lengths.append(len(committed))
        if draft:
            gate.update(len(draft), n, base_draft)

        target_model.rollback(cache, len(draft) - n, draft[:n])

        appended = []
        for tok in committed:
            out_ids.append(tok)
            appended.append(tok)
            if tok in eos_ids:
                break
        index.extend(appended)
        pending = out_ids[-1]        # eos mid-committed ends the loop (not committed[-1])
        st.update(out_ids)
    st.flush()

    secs = time.time() - t0
    text = st.text if st.stopped else tokenizer.decode([t for t in out_ids if t not in eos_ids])
    return GenResult(
        text=text,
        token_ids=out_ids,
        num_tokens=len(out_ids),
        num_rounds=len(accept_lengths),
        accept_lengths=accept_lengths,
        target_forwards=target_forwards,
        seconds=secs,
        finish_reason=_finish_reason(out_ids, max_new_tokens, pending, eos_ids, st),
    )
