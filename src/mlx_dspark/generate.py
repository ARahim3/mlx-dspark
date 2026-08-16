"""DSpark speculative decoding loop (greedy, batch=1) for Apple Silicon.

Per round:
  1. draft a block of K tokens from the parallel backbone + Markov head,
  2. verify them in one target forward,
  3. accept the matching prefix + 1 bonus token (so >=1 token/round always),
  4. trim the target KV cache and grow the fused-hidden context buffer.

Because the target verifies every token, the *output is exactly greedy target
decoding* regardless of drafter quality — drafter quality only shows up as the
acceptance length (tokens committed per target forward).
"""

from __future__ import annotations

import functools
import json
import time
from dataclasses import dataclass

import mlx.core as mx
from mlx.utils import tree_flatten

from .sampling import sample_probs, truncate_probs
from .small_m_qmm import small_m_matmul
from .wide_gemm import wide_matmul

TAP = None  # set from drafter config at call time

SMALL_M_IDS = None  # QuantizedLinear instances (by id) routed through the small-M MMA
# kernel for verify-window forwards of 6-8 rows (see small_m_qmm.py). None disables.
# Same doctrine as WIDE_GEMM_*: the library default stays off — a plain generate call
# must not silently change numerics class — and the CLI/server set it from
# calibrate.apply_small_m()'s measured per-shape gate.


def _with_small_m(fn):
    """Run a generation loop inside :func:`small_m_matmul` (a no-op when SMALL_M_IDS is
    unset). A decorator rather than an inline ``with`` so the loop bodies stay
    untouched; the global is read at call time, so hot swaps re-resolve naturally."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with small_m_matmul(SMALL_M_IDS):
            return fn(*args, **kwargs)
    return wrapper


@dataclass
class GenResult:
    text: str
    token_ids: list[int]
    num_tokens: int
    num_rounds: int
    accept_lengths: list[int]
    target_forwards: int
    seconds: float
    finish_reason: str = "stop"  # "stop" (eos/stop-string) | "length" (hit max_new_tokens)
    lookup_rounds: int = 0       # rounds whose draft came from the free n-gram lookup
    logprobs: list | None = None  # per-token [{token_id, logprob, top:[(id, logprob), …]}], if requested

    @property
    def mean_accept_len(self) -> float:
        return self.num_tokens / max(self.num_rounds, 1)

    @property
    def tokens_per_sec(self) -> float:
        return self.num_tokens / max(self.seconds, 1e-9)


def _ids_from_template_result(r) -> list[int] | None:
    """Normalize whatever ``apply_chat_template`` returns (list[int], nested list,
    or a BatchEncoding) into a flat list[int]. Returns None if it can't."""
    if isinstance(r, (list, tuple)):
        if r and isinstance(r[0], int):
            return list(r)
        if r and isinstance(r[0], (list, tuple)):
            return list(r[0])
        return list(r)
    ii = None
    if hasattr(r, "__contains__") and "input_ids" in r:
        ii = r["input_ids"]
    elif hasattr(r, "input_ids"):
        ii = r.input_ids
    if ii is not None:
        ii = list(ii)
        return list(ii[0]) if ii and isinstance(ii[0], (list, tuple)) else ii
    if hasattr(r, "ids"):
        return list(r.ids)
    return None


def encode_messages(tokenizer, messages: list[dict], add_generation_prompt: bool = True,
                    **template_kwargs) -> list[int]:
    """Token ids for a full chat transcript (multi-turn), via the model's chat template.

    ``messages`` is the OpenAI shape: ``[{"role": "system"|"user"|"assistant", "content": ...}]``.
    This is what the OpenAI-compatible server uses so conversations, system prompts, and
    assistant history all reach the model exactly as its template expects. Falls back to
    concatenating contents if the tokenizer has no chat template.

    ``template_kwargs`` are passed straight to the chat template — this is how the server
    forwards e.g. ``enable_thinking=False`` (Qwen3) or ``tools=[...]``. Unknown kwargs are
    harmless for templates that ignore them; if a tokenizer rejects them outright we retry
    without them rather than fail the request.
    """
    if getattr(tokenizer, "chat_template", None):
        try:
            r = tokenizer.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt, **template_kwargs)
        except (TypeError, ValueError):
            r = tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
        ids = _ids_from_template_result(r)
        if ids is not None:
            return ids
    # No template: best-effort flat concat (rare for the instruct targets we ship).
    text = "\n".join(str(m.get("content", "")) for m in messages)
    return list(tokenizer.encode(text))


def encode_prompt(tokenizer, prompt: str, use_chat: bool = True) -> list[int]:
    """Token ids for a single user prompt, using the model's chat template when present.

    Gemma-4 uses `<|turn>` / `<channel|>` markers (NOT Gemma-3's `<start_of_turn>`),
    so the template must be applied via the tokenizer — hand-formatting breaks the
    instruct model. Thin wrapper over :func:`encode_messages` for the one-user-turn case.
    """
    if use_chat and getattr(tokenizer, "chat_template", None):
        return encode_messages(tokenizer, [{"role": "user", "content": prompt}])
    return list(tokenizer.encode(prompt))


def eos_token_ids(tokenizer) -> set[int]:
    """Collect stop-token ids: eos + turn-end markers (Gemma-4 uses <turn|>=106; note
    <end_of_turn> is the UNK id in Gemma-4, so it must be filtered out).

    ``<|tool_response>`` is a turn-end marker too, and a load-bearing one: after a tool call
    Gemma-4 does *not* emit ``<turn|>`` — it emits ``<|tool_response>`` to hand back to the
    harness for the tool result. Its own response grammar
    (``tokenizer_config.json`` -> ``response_schema.x-regex``) terminates on either. Without it
    the model runs straight past its turn and hallucinates the tool result and the following
    conversation, burning the whole ``max_tokens`` budget on fiction — which is exactly what a
    tool-calling agent hits on every single turn.
    """
    ids: set[int] = set()
    e = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(e, int):
        ids.add(e)
    elif e:
        ids.update(int(x) for x in e)
    e1 = getattr(tokenizer, "eos_token_id", None)
    if isinstance(e1, int):
        ids.add(e1)
    unk = getattr(tokenizer, "unk_token_id", None)
    # Gemma-4 (<turn|>, <|tool_response>), Gemma-3 (<end_of_turn>), Qwen (<|im_end|>),
    # muse_glimmer (<|eot|>=200008 ends the assistant turn; its config eos is [200001, 200008]
    # but the tokenizer only exposes 200001 as eos_token_id, so the turn-end must be named here
    # or every chat turn overruns to max_tokens), raw eos
    for t in ("<turn|>", "<|tool_response>", "<end_of_turn>", "<|im_end|>", "<|eot|>",
              "<|endoftext|>", "<eos>"):
        try:
            i = tokenizer.convert_tokens_to_ids(t)
        except Exception:  # noqa: BLE001 — tokenizers raise various types for unknown tokens
            continue
        if isinstance(i, int) and i >= 0 and i != unk:
            ids.add(i)
    return ids


def _sample_arr(logits_row, temperature: float, top_p: float = 1.0, top_k: int = 0) -> mx.array:
    """Chosen token as a (lazy) mx scalar: argmax at temperature 0, else a temperature /
    top-p / top-k sample. No device sync — callers decide when to materialize."""
    if temperature > 0.0:
        probs = truncate_probs(mx.softmax(logits_row / temperature, axis=-1), top_p, top_k)
        return sample_probs(probs)
    return mx.argmax(logits_row)


def _pick(logits_row, temperature: float, top_p: float = 1.0, top_k: int = 0) -> int:
    """argmax (temperature 0) or a temperature / top-p / top-k sample (temperature > 0)."""
    return int(_sample_arr(logits_row, temperature, top_p, top_k).item())


def _logprobs_for_block(logits_rows, token_ids, top_k: int) -> list[dict]:
    """Per-token logprobs from a block of target logits ``[P, V]`` and the ``P`` tokens actually
    committed at those positions. Returns ``[{token_id, logprob, top:[(id, logprob), …]}]`` using
    the **raw** target log-softmax (temperature/penalty-independent — it reports the target's own
    distribution, which is what OpenAI logprobs are read for). ``top_k`` 0 = chosen token only.
    Gathers on-GPU (one eval per block) so it adds a small, bounded cost only when requested."""
    x = logits_rows.astype(mx.float32)                                # stable log-softmax (max-shift)
    m = mx.max(x, axis=-1, keepdims=True)
    logp = x - m - mx.log(mx.sum(mx.exp(x - m), axis=-1, keepdims=True))
    P = len(token_ids)
    chosen = logp[mx.arange(P), mx.array(token_ids)]                   # [P]
    top = None
    if top_k > 0:
        kth = min(top_k, logp.shape[-1])
        idx = mx.argsort(-logp, axis=-1)[:, :kth]                      # [P, k] top ids per row
        vals = mx.take_along_axis(logp, idx, axis=-1)                 # [P, k]
        mx.eval(chosen, idx, vals)
        top = (idx.tolist(), vals.tolist())
    else:
        mx.eval(chosen)
    ch = chosen.tolist()
    out = []
    for i in range(P):
        e = {"token_id": int(token_ids[i]), "logprob": float(ch[i])}
        if top is not None:
            e["top"] = [(int(t), float(l)) for t, l in zip(top[0][i], top[1][i])]
        out.append(e)
    return out


class _Penalizer:
    """OpenAI ``presence_penalty`` / ``frequency_penalty`` applied to the **target** logits, so
    the greedy/spec output equals sequential decoding of the penalized target (lossless wrt the
    penalized target — for temp>0 too: speculative sampling is exact wrt whatever distribution the
    target logits define, so penalizing only the target ``p`` suffices; the drafter proposal ``q``
    just loses a little acceptance). Running completion-token counts are kept incrementally.
    Inactive (both penalties 0) → a no-op that leaves the default decode path byte-for-byte
    unchanged. Penalized logit for token v: ``logit[v] - presence*(count[v]>0) - frequency*count[v]``
    over the generated completion only (OpenAI semantics)."""

    def __init__(self, presence: float = 0.0, frequency: float = 0.0):
        self.presence = float(presence or 0.0)
        self.frequency = float(frequency or 0.0)
        self.active = bool(self.presence or self.frequency)
        self.counts: dict[int, int] = {}

    def add(self, tokens) -> None:
        if not self.active:
            return
        for t in tokens:
            t = int(t)
            self.counts[t] = self.counts.get(t, 0) + 1

    def block_penalty(self, vocab: int, draft_prefix, dtype) -> mx.array:
        """``[len(draft_prefix)+1, vocab]`` penalty to subtract from a verify block's logits:
        row i penalizes by the base completion counts **plus** the block's own ``draft_prefix[:i]``
        — so the accepted prefix's (penalized) argmax matches sequential penalized decoding.
        ``draft_prefix=[]`` gives a single ``[1, vocab]`` row (the baseline one-token case)."""
        base = mx.zeros((vocab,), dtype=dtype)
        if self.counts:
            ids = mx.array(list(self.counts.keys()))
            cs = mx.array(list(self.counts.values()), dtype=dtype)
            base[ids] = self.presence + self.frequency * cs
        rows = [base]
        extra = dict(self.counts)
        for d in draft_prefix:
            d = int(d)
            inc = self.frequency + (self.presence if extra.get(d, 0) == 0 else 0.0)
            nxt = rows[-1] + 0.0
            nxt[d] = nxt[d] + inc
            rows.append(nxt)
            extra[d] = extra.get(d, 0) + 1
        return mx.stack(rows)

    def apply(self, v_logits_rows, draft_prefix):
        """Penalize target verify logits ``[M+1, V]`` in place-of; identity when inactive."""
        if not self.active:
            return v_logits_rows
        return v_logits_rows - self.block_penalty(
            v_logits_rows.shape[-1], draft_prefix, v_logits_rows.dtype)


class StopStreaming(Exception):
    """Raise from an ``on_text`` callback to end generation gracefully: the loop stops at
    the next round boundary and returns a normal (partial) GenResult, leaving caches in a
    consistent, storable state. The server uses this when a streaming client disconnects,
    so the prefix cache survives instead of being invalidated by an error."""


class _FullDecodeDetokenizer:
    """Fallback detokenizer: full re-decode of all tokens on every ``.text`` access (the
    pre-0.1.1 _Streamer behavior, O(n²) over a generation). Used only when no streaming
    detokenizer can be built for this tokenizer (e.g. minimal test doubles)."""

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self.tokens: list[int] = []

    def add_token(self, token: int) -> None:
        self.tokens.append(token)

    def finalize(self) -> None:
        pass

    @property
    def text(self) -> str:
        return self._tok.decode(self.tokens)


def _make_detokenizer(tokenizer):
    """Best available *streaming* detokenizer for this tokenizer, so streaming decodes only
    the new tokens each round instead of re-decoding the whole output (O(n) vs O(n²) over a
    generation — the re-decode dominated long/thinking outputs).

    - mlx-lm's ``TokenizerWrapper`` (the qwen3 target path) carries one: use it.
    - A plain HF fast tokenizer (the mlx-vlm/gemma path): pick mlx-lm's SPM/BPE streaming
      class by inspecting the backend decoder, exactly like ``mlx_lm.tokenizer_utils.load``.
    - Anything else falls back to full re-decode (prior behavior).
    """
    detok = getattr(tokenizer, "detokenizer", None)   # mlx-lm TokenizerWrapper property
    if detok is not None:
        return detok
    try:
        from mlx_lm.tokenizer_utils import (
            BPEStreamingDetokenizer,
            SPMStreamingDetokenizer,
            _is_bpe_decoder,
            _is_spm_decoder,
            _is_spm_decoder_no_space,
        )

        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None:
            decoder = json.loads(backend.to_str()).get("decoder") or {}
            if _is_spm_decoder(decoder):
                return SPMStreamingDetokenizer(tokenizer)
            if _is_spm_decoder_no_space(decoder):
                return SPMStreamingDetokenizer(tokenizer, trim_space=False)
            if _is_bpe_decoder(decoder):
                return BPEStreamingDetokenizer(tokenizer)
    except Exception:  # noqa: BLE001 — any wrapping failure means: use the safe fallback
        pass
    return _FullDecodeDetokenizer(tokenizer)


class _Streamer:
    """Round-granular text streaming + ``stop`` string detection, shared by every loop.

    Each round the caller pushes the running ``out_ids``; the *new* tokens are fed to a
    streaming detokenizer (so each update decodes only what's new — O(n) over a generation),
    the new tail is emitted via ``on_text``, and — when ``stop`` strings are configured —
    the output is cut at the earliest stop occurrence (setting ``stopped``). Stop strings are
    scanned incrementally with a ``max_stop - 1`` lookback so one straddling two rounds is
    still caught, and emission holds back the last ``max_stop - 1`` chars until it's safe (or
    we finish). With no stop strings and no ``on_text`` this is a no-op, so the greedy/spec
    loops keep their exact prior behavior.
    """

    def __init__(self, tokenizer, eos_ids: set[int], on_text, stop):
        self.eos = eos_ids
        self.on_text = on_text
        self.stop = [s for s in (stop or []) if s]
        self.max_stop = max((len(s) for s in self.stop), default=0)
        self.detok = (
            _make_detokenizer(tokenizer) if (on_text is not None or self.stop) else None
        )
        self.n_fed = 0        # how many of out_ids have been fed to the detokenizer
        self.scan_from = 0    # text index the incremental stop-scan resumes at
        self.streamed = 0     # chars already emitted via on_text
        self.text = ""
        self.stopped = False  # a stop string was hit -> caller should end the loop

    def update(self, out_ids: list[int]) -> None:
        if self.detok is None or self.stopped:
            return
        for t in out_ids[self.n_fed:]:
            if t not in self.eos:
                self.detok.add_token(t)
        self.n_fed = len(out_ids)
        self._advance(final=False)

    def flush(self) -> None:
        if self.detok is None or self.stopped:
            return
        self.detok.finalize()
        self._advance(final=True)

    def _advance(self, final: bool) -> None:
        text = self.detok.text
        if self.stop:
            cut = None
            for s in self.stop:
                i = text.find(s, self.scan_from)
                if i != -1:
                    cut = i if cut is None else min(cut, i)
            if cut is not None:
                text = text[:cut]
                self.stopped = True
            else:
                self.scan_from = max(0, len(text) - (self.max_stop - 1))
        self.text = text
        if self.on_text is None:
            return
        emit_to = len(text)
        if not (self.stopped or final) and self.max_stop > 1:
            emit_to = max(self.streamed, len(text) - (self.max_stop - 1))
        if emit_to > self.streamed:
            try:
                self.on_text(text[self.streamed:emit_to])
            except StopStreaming:
                self.on_text = None      # nobody is listening; stop emitting
                self.stopped = True      # -> loops end at the next boundary, GenResult is normal
                return
            self.streamed = emit_to


def _finish_reason(out_ids: list[int], max_new_tokens: int, last_tok: int,
                   eos_ids: set[int], streamer: _Streamer) -> str:
    """'stop' if a stop string / eos ended it, else 'length' if we hit the token cap."""
    if streamer.stopped or last_tok in eos_ids:
        return "stop"
    return "length" if len(out_ids) >= max_new_tokens else "stop"


def greedy_generate(
    target_model,
    tokenizer,
    prompt: str = "",
    *,
    prompt_ids: list[int] | None = None,
    cache=None,
    reuse_len: int = 0,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    logprobs: int | None = None,
    seed: int | None = None,
    apply_chat_template: bool = True,
    stop: list[str] | None = None,
    on_text=None,
    on_round=None,
    on_prefill=None,
    prefill_marks=None,
) -> GenResult:
    """Plain decoding of the target (no drafter, no hidden-state capture) — the fair
    'run the model normally' baseline. ``temperature`` 0 = greedy, > 0 = sampling (matches
    the spec path so a temp>0 A/B compares like-for-like). Streams via on_text.
    ``presence_penalty`` / ``frequency_penalty`` (OpenAI) penalize the completion's own tokens.

    ``prompt_ids`` overrides ``prompt`` with a pre-tokenized prompt (the server passes a full
    multi-turn transcript this way); ``stop`` adds string stop-sequences (OpenAI ``stop``)."""
    if seed is not None:
        mx.random.seed(seed)
    eos_ids = eos_token_ids(tokenizer)
    ids = prompt_ids if prompt_ids is not None else encode_prompt(
        tokenizer, prompt, use_chat=apply_chat_template)
    if cache is None:                                  # fresh, or reuse a prefix-cached one
        cache = target_model.make_cache()
        reuse_len = 0
    st = _Streamer(tokenizer, eos_ids, on_text, stop)

    t0 = time.time()
    suffix = ids[reuse_len:] if reuse_len else ids      # only prefill past the reused prefix
    logits = _prefill_plain(
        target_model, suffix, cache, base=reuse_len, marks=prefill_marks,
        on_mark=(lambda p: on_prefill(cache, None, p)) if on_prefill else None)
    if on_prefill is not None:
        on_prefill(cache, None, len(ids))   # caches hold exactly `ids` right now
    out_ids: list[int] = []
    pen = _Penalizer(presence_penalty, frequency_penalty)
    lp_list: list | None = [] if logprobs is not None else None

    if pen.active or logprobs is not None:
        # Sequential decode: needed when penalties are on (penalty state must include the just-
        # emitted token before predicting the next) or logprobs are requested (we read each
        # committed token's logits row). The fast pipeline below stays the default path.
        logits_row = logits[0, -1]
        while True:
            pen0 = (pen.block_penalty(logits_row.shape[-1], [], logits_row.dtype)[0]
                    if pen.active else 0.0)
            nxt = int(_sample_arr(logits_row - pen0, temperature, top_p, top_k).item())
            out_ids.append(nxt)
            pen.add([nxt])
            if on_round is not None:
                on_round(drafted=0, accepted=0, committed=1, cap=0, source="plain")
            if lp_list is not None:
                lp_list.extend(_logprobs_for_block(logits_row[None, :], [nxt], logprobs))
            st.update(out_ids)
            if len(out_ids) >= max_new_tokens or nxt in eos_ids or st.stopped:
                break
            logits_row = target_model.plain(mx.array([[nxt]]), cache)[0, -1]
    else:
        # Pipelined decode (mlx-lm style): schedule step t+1 on the GPU *before* syncing on
        # step t's token, so detokenize/emit overlaps GPU compute instead of serializing with
        # it. The one forward scheduled past the final token is wasted work; the cache being a
        # token ahead is harmless (prefix reuse trims to the recorded token count).
        y = _sample_arr(logits[0, -1], temperature, top_p, top_k)
        mx.async_eval(y)
        while True:
            logits = target_model.plain(y.reshape(1, 1), cache)
            y_next = _sample_arr(logits[0, -1], temperature, top_p, top_k)
            mx.async_eval(y_next)
            nxt = int(y.item())
            out_ids.append(nxt)
            # Baseline has no speculation, but reporting each step keeps the live view's
            # event shape identical across modes — which is what makes a side-by-side race
            # between baseline and dspark a single rendering path.
            if on_round is not None:
                on_round(drafted=0, accepted=0, committed=1, cap=0, source="plain")
            st.update(out_ids)
            if len(out_ids) >= max_new_tokens or nxt in eos_ids or st.stopped:
                break
            y = y_next
    st.flush()

    secs = time.time() - t0
    text = st.text if st.stopped else tokenizer.decode([t for t in out_ids if t not in eos_ids])
    return GenResult(
        text=text,
        token_ids=out_ids,
        num_tokens=len(out_ids),
        num_rounds=len(out_ids),
        accept_lengths=[1] * len(out_ids),
        target_forwards=len(out_ids),
        seconds=secs,
        finish_reason=_finish_reason(out_ids, max_new_tokens, nxt, eos_ids, st),
        logprobs=lp_list,
    )


@_with_small_m
def dflash_generate(
    target_model,
    tokenizer,
    drafter,
    prompt: str = "",
    *,
    prompt_ids: list[int] | None = None,
    max_new_tokens: int = 128,
    max_draft_tokens: int | None = None,
    cap_controller=None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    apply_chat_template: bool = True,
    seed: int | None = None,
    stop: list[str] | None = None,
    on_text=None,
    on_round=None,
) -> GenResult:
    """Speculative decoding with a **z-lab DFlash** (block-diffusion) drafter.

    DFlash differs from DSpark in two ways that matter to this loop:
      - it feeds ``[anchor] + (block-1) masks`` and reads logits at the **mask** positions
        (``logits_start=1``), i.e. predict-the-masks, not DSpark's anchor-as-position-0;
      - it has no own embed/lm_head — it reuses the target's (we ``bind`` once here).

    ``temperature == 0`` → greedy (exact argmax-match verify; output == greedy decoding up to
    fp ties). ``temperature > 0`` → speculative sampling (paper §2.1): drafts sampled from the
    block-diffusion proposal q, accepted w.p. ``min(1, p/q)`` vs the target p, residual-resampled
    on first reject — an exact sample from the target at temperature T (lossless).

    The backbone always drafts the full block width (it's trained at that width / block
    diffusion is bidirectional); ``max_draft_tokens`` only bounds how many drafted tokens
    are *verified* per round. ``None`` = verify the whole block (DFlash's native operating
    point — best on structured content; on open chat a short cap is faster).
    """
    if seed is not None:
        mx.random.seed(seed)
    if getattr(drafter, "embed_tokens", None) is None:
        drafter.bind(target_model.model)

    cfg = drafter.config
    tap = list(cfg.target_layer_ids)
    bs = int(cfg.block_size)
    mask_id = int(cfg.mask_token_id)
    kdraft = bs - 1
    cap_ceiling = kdraft if max_draft_tokens is None else max(1, min(max_draft_tokens, kdraft))
    cap = cap_ceiling
    eos_ids = eos_token_ids(tokenizer)

    ids = prompt_ids if prompt_ids is not None else encode_prompt(
        tokenizer, prompt, use_chat=apply_chat_template)
    cache = _make_target_cache(target_model)
    target_model.reset_spec()                          # hybrid targets: clear capture state
    dcache = drafter.make_cache()                      # persistent draft ctx cache
    st = _Streamer(tokenizer, eos_ids, on_text, stop)
    t0 = time.time()

    # prefill (chunked; DFlash's first draft call consumes the whole prompt's fused states)
    logits, fused = _prefill_tapped(target_model, ids, cache, tap)
    pending_ctx = fused                                # fused hidden appended to draft ctx next round
    pending = _pick(logits[0, -1], temperature, top_p, top_k)
    out_ids: list[int] = [pending]
    accept_lengths: list[int] = []
    target_forwards = 1

    st.update(out_ids)
    while len(out_ids) < max_new_tokens and pending not in eos_ids and not st.stopped:
        # ---- draft full-width block; feeding pending_ctx appends exactly the just-
        # committed positions to the draft KV cache (DFlash caches only ctx KV, never
        # block KV) -> correct absolute RoPE offsets, no trim needed.
        if cap_controller is not None:
            cap = max(1, min(cap_controller.cap, cap_ceiling))
        block = mx.array([[pending] + [mask_id] * (bs - 1)])
        head = drafter(block, pending_ctx, dcache, logits_start=1)[0][:cap]  # [cap, V] mask logits
        if temperature > 0.0:
            # two-phase: the accept test needs the q distributions on hand
            q_probs = truncate_probs(mx.softmax(head / temperature, axis=-1), top_p, top_k)
            draft_arr = sample_probs(q_probs)
            mx.eval(draft_arr, q_probs)
            drafted = [int(x) for x in draft_arr.tolist()]

            verify_ids = mx.array([[pending] + drafted])
            v_logits, v_fused = target_model.verify(verify_ids, cache, tap)
            mx.eval(v_logits, v_fused)

            n, repl = _spec_sample_accept(v_logits[0], drafted, q_probs, temperature, top_p, top_k)
            committed = drafted[:n] + [repl]             # accepted prefix + residual/bonus
        else:
            # fused greedy path: draft + verify + accept reach the device as one graph with
            # a single sync per round (see speculative_generate for the same pattern).
            draft_arr = mx.argmax(head, axis=-1)
            verify_ids = mx.concatenate(
                [mx.array([pending], dtype=draft_arr.dtype), draft_arr]).reshape(1, -1)
            v_logits, v_fused = target_model.verify(verify_ids, cache, tap)
            tt_arr = mx.argmax(v_logits[0], axis=-1)
            match = (draft_arr == tt_arr[: draft_arr.shape[0]]).astype(mx.int32)
            n_arr = mx.cumprod(match).sum()
            mx.eval(n_arr, tt_arr, draft_arr)
            n = int(n_arr.item())
            drafted = draft_arr.tolist()
            tt = tt_arr.tolist()
            committed = drafted[:n] + [tt[n]]            # accepted prefix + bonus
        target_forwards += 1
        accept_lengths.append(len(committed))
        if cap_controller is not None:
            cap_controller.update(n, len(drafted))
        if on_round is not None:
            on_round(drafted=len(drafted), accepted=n, committed=len(committed),
                     cap=cap, source="drafter")

        # ---- update target cache + draft ctx ----
        target_model.rollback(cache, len(drafted) - n, drafted[:n])
        pending_ctx = v_fused[:, : n + 1, :]             # [anchor, accepted] -> next draft ctx

        for tok in committed:
            out_ids.append(tok)
            if tok in eos_ids:
                break
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


def _make_target_cache(target):
    return target.make_cache()


def _run_target(target, ids: mx.array, cache, tap: list[int]):
    """ids: [1, L]. Returns (logits[1,L,V], fused_hidden[1,L,n_tap*H])."""
    return target.run(ids, cache, tap)


PREFILL_CHUNK = 2048  # long prompts prefill in pieces so activations stay bounded, with
# mx.clear_cache() between pieces. Prompts within one chunk take exactly the old
# single-forward path, and a chunked prefill is the same cached multi-pass forward the
# verify rounds already use (lossless to the usual fp-tie standard). Since the [chunk,
# vocab] logits are no longer materialized (see below) this is a pure activation knob;
# don't go below ~2048, SDPA's prefill efficiency falls off under q_len 1024 (measured
# 50% of peak at 512, 70% at 1024, 85%+ from 2048 — NOTES "Prefill pass").

PREFILL_LAST_ROW_HEAD = True  # Project only the LAST row of the final prefill chunk to
# vocab. Every prefill caller reads exactly logits[0, -1] (it becomes the first sampled
# token), so the rest of that [chunk, vocab] matmul is thrown away — ~7-8% of prefill
# FLOPs on an 8-bit Qwen3-8B. Skipping it on NON-final chunks is bit-identical and is
# unconditional; this knob additionally slices the final chunk, which moves that one row
# onto the quantized matvec path (the path every decode step takes): the usual fp-tie
# class, measured max|Δlogit| = 0.06 = 1-2 bf16 ulps. Set False for a bit-identical
# prefill at the cost of the whole win on prompts that fit in one chunk.

WIDE_GEMM_MIN_ROWS = None  # Rows from which QuantizedLinear switches to dequantize-once +
# GEMM (see wide_gemm.py). None disables. Left None for the library API — a library call
# must not silently trigger a device benchmark — and set by the CLI/server from
# wide_gemm.measure_crossover(), cached to disk like every other tuned width here.
WIDE_GEMM_SHAPES = None    # allowlist of weight shapes verified bit-identical at that
# width (None = all eligible). Set together with WIDE_GEMM_MIN_ROWS.


def _cache_arrays(cache) -> list:
    """Every mx.array living in a cache list, however the layer caches nest their state
    (KVCache tuples, ArraysCache lists with None holes, CacheList of caches). Used to
    force a chunk's forward when there are no logits to force it — without logits to
    depend on, only the caches tie the whole graph together."""
    states = [getattr(c, "state", None) for c in cache]
    return [v for _, v in tree_flatten(states) if isinstance(v, mx.array)]


def _mark_stops(marks, base: int, n: int) -> list[int]:
    """Suffix-relative positions the chunked prefill must break at so the caches hold
    exactly ``base + stop`` tokens there (``marks`` are absolute prompt positions; the
    prompt boundary itself is not a stop — the loop's final on_prefill covers it)."""
    return sorted({m - base for m in (marks or ()) if 0 < m - base < n})


def _prefill_plain(target, ids: list[int], cache, chunk: int | None = None,
                   base: int = 0, marks=None, on_mark=None):
    """Chunked no-tap prefill; returns the last chunk's logits [1, 1, V]. ``marks``
    (absolute positions, with ``base`` = tokens already in the caches) split chunks so
    ``on_mark(pos)`` runs while the caches hold exactly the first ``pos`` tokens — the
    prefix cache snapshots its interior boundaries there."""
    chunk = chunk or PREFILL_CHUNK      # read the module knob at call time
    stops = _mark_stops(marks, base, len(ids))
    logits = None
    many = len(ids) > chunk or bool(stops)
    with wide_matmul(WIDE_GEMM_MIN_ROWS, WIDE_GEMM_SHAPES):
        i = 0
        while i < len(ids):
            end = min(i + chunk, len(ids))
            end = min([end] + [s for s in stops if i < s < end])
            last = end == len(ids)
            logits, _ = target.prefill(mx.array([ids[i:end]]), cache,
                                       want_logits=last,
                                       head_last_row=PREFILL_LAST_ROW_HEAD)
            if many and not last:
                mx.eval(_cache_arrays(cache))
                mx.clear_cache()
            if on_mark is not None and end in stops:
                on_mark(base + end)
            i = end
    return logits


def _prefill_tapped(target, ids: list[int], cache, tap, drafter=None, ctx_caches=None,
                    ctx_offset: int = 0, chunk: int | None = None,
                    marks=None, on_mark=None):
    """Chunked prefill with the hidden-state tap. When ``drafter`` is given, each chunk's
    fused states feed the drafter context immediately (so a long prompt's fused activations
    never all materialize at once); returns (last logits, last chunk's fused). Without a
    drafter the fused chunks are concatenated (DFlash needs the whole prompt's fused).
    ``marks``/``on_mark``: see :func:`_prefill_plain` (``ctx_offset`` is the base)."""
    chunk = chunk or PREFILL_CHUNK      # read the module knob at call time
    stops = _mark_stops(marks, ctx_offset, len(ids))
    logits = fused = None
    parts = []
    pos = ctx_offset
    many = len(ids) > chunk or bool(stops)
    with wide_matmul(WIDE_GEMM_MIN_ROWS, WIDE_GEMM_SHAPES):
        i = 0
        while i < len(ids):
            end = min(i + chunk, len(ids))
            end = min([end] + [s for s in stops if i < s < end])
            piece = ids[i:end]
            last = end == len(ids)
            logits, fused = target.prefill(mx.array([piece]), cache, tap,
                                           want_logits=last,
                                           head_last_row=PREFILL_LAST_ROW_HEAD)
            if drafter is not None:
                drafter.update_context(fused, ctx_offset=pos, ctx_caches=ctx_caches)
            else:
                parts.append(fused)
            pos += len(piece)
            if many and not last:
                # the tap only reaches the tapped layers, so the caches (not `fused`) are
                # what force the rest of the forward and keep the graph from growing
                mx.eval(_cache_arrays(cache),
                        [c.k for c in ctx_caches] if ctx_caches else fused)
                mx.clear_cache()
            if on_mark is not None and end in stops:
                on_mark(ctx_offset + end)
            i = end
    if drafter is None:
        fused = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
    return logits, fused


def _spec_sample_accept(v_logits, draft, q_probs, temperature, top_p=1.0, top_k=0):
    """Speculative-sampling acceptance (Leviathan/Chen 2023) for one verified block.

    ``v_logits`` [1+L, V] are the target logits at the verify positions; ``draft`` is the
    list of L sampled tokens; ``q_probs`` [>=L, V] the draft distributions they were sampled
    from. Each token is accepted w.p. ``min(1, p(x)/q(x))``; the first rejection stops the
    block and resamples from the residual ``norm(max(0, p-q))``; if all accept, a bonus is
    sampled from the target. Returns ``(n_accepted, replacement_token)``. This is the rule
    that makes the output an exact sample from the target's temperature-T distribution.

    With ``top_p`` / ``top_k`` the *target* distribution ``p`` is truncated first, so the
    output is an exact sample from ``top-p/top-k(softmax(target / T))`` — still lossless, now
    wrt the client's requested truncation (the draft ``q_probs`` were truncated to match)."""
    L = len(draft)
    p = truncate_probs(mx.softmax(v_logits / temperature, axis=-1), top_p, top_k)  # [1+L, V]
    rows = mx.arange(L)
    idx = mx.array(draft)
    pd = p[rows, idx]                                          # target prob of each drafted token
    qd = q_probs[rows, idx]                                    # draft prob it was sampled from
    u = mx.random.uniform(shape=(L,))
    accepted = u < mx.minimum(1.0, pd / mx.maximum(qd, 1e-9))
    # accepted-prefix length in-graph (cumprod stops at the first reject)
    n = int(mx.cumprod(accepted.astype(mx.int32)).sum().item())
    if n < L:
        resid = mx.maximum(p[n] - q_probs[n], 0.0)            # residual at the rejected position
        resid = resid / mx.maximum(resid.sum(), 1e-9)
        repl = int(mx.random.categorical(mx.log(resid + 1e-20)).item())
    else:
        repl = int(sample_probs(p[L]).item())                # target bonus from the (truncated) p
    return n, repl


@_with_small_m
def speculative_generate(
    target_model,
    tokenizer,
    drafter,
    prompt: str = "",
    *,
    prompt_ids: list[int] | None = None,
    cache=None,
    ctx_caches=None,
    reuse_len: int = 0,
    max_new_tokens: int = 128,
    confidence_threshold: float = 0.0,
    max_draft_tokens: int | None = 2,
    cap_controller=None,
    lookup_drafts: bool = True,
    lookup_max_draft: int = 6,
    lookup_long_draft: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    logprobs: int | None = None,
    seed: int | None = None,
    apply_chat_template: bool = True,
    stop: list[str] | None = None,
    on_text=None,
    on_round=None,
    on_prefill=None,
    prefill_marks=None,
    verbose: bool = False,
) -> GenResult:
    """Speculative decoding (batch=1).

    ``temperature == 0`` → **greedy**: argmax draft, exact-argmax-match verify. Output is
    target-greedy by construction (up to fp tie-breaking on near-ties).

    ``temperature > 0`` → **speculative sampling** (the paper's setup, §2.1): each draft
    position is sampled from its temperature-scaled distribution q, then accepted with
    probability ``min(1, p(x)/q(x))`` against the target distribution p; on the first
    rejection the token is resampled from the residual ``norm(max(0, p-q))`` and the rest
    of the block is discarded; if all are accepted a bonus is sampled from the target. This
    preserves the target's temperature-T sampling distribution exactly (lossless), and
    accepts more per round than greedy (greedy's exact-match is the strictest possible rule).

    ``max_draft_tokens`` (``cap``) bounds how many of the 7-token block are drafted *and*
    verified per round: on Apple Silicon the verify cost grows with tokens and the marginal
    draft token rarely survives, so cap=2 is the measured optimum (the drafter only runs
    lm_head/markov over these ``cap`` positions). ``None`` = full block. ``confidence_threshold``
    > 0 truncates the block adaptively via the drafter's confidence head (cumulative survival).

    ``cap_controller`` (a :class:`~mlx_dspark.calibrate.CapController`) picks the cap per
    round from this machine's measured cost curves + a live acceptance estimate, within
    ``max_draft_tokens``'s ceiling. The cap only affects speed, never output (the target
    verifies every token), so adapting it mid-generation stays lossless.

    ``lookup_drafts`` (hybrid drafting, on by default): when the current suffix n-gram
    already occurred earlier in the context (a copy run — quoting, code edits, repeats),
    that free continuation (up to ``lookup_max_draft`` tokens) is verified *instead of*
    running the drafter this round; otherwise the DSpark block drafts as usual. Verification
    is unchanged either way, so this composes losslessly — it just lets copy-heavy spans
    commit ~6 tokens/round where the drafter block would cap out at ~2-3.

    ``lookup_long_draft``: match-scaled long-draft ceiling. When the suffix match extends
    backwards ≥8 tokens (real copy run, not a bare 4-5-gram hit), the lookup draft grows to
    ~2x the matched length, up to this ceiling — the M-series verify-width plateau (width
    16-32 ≈ 2.5x one step) makes the extra rows near-free, so verbatim spans commit ~30
    tokens per verify instead of ~6. A :class:`~mlx_dspark.lookup.LongDraftGate` parks the
    scaling when long drafts keep getting chopped early (insertion-heavy edits) and probes
    back in. Set equal to ``lookup_max_draft`` to disable. Speed-only; output unchanged
    (see :meth:`NGramIndex.propose`).
    """
    if seed is not None:
        mx.random.seed(seed)
    cfg = drafter.config
    # DFlash-warm-started heads reuse the target's own embed_tokens and/or lm_head (they ship
    # neither weight): Nemotron reuses the head, Muse-Glimmer reuses both. Bind what's reused.
    if not getattr(cfg, "has_own_lm_head", True):
        drafter.bind_lm_head(target_model.lm_head_proj)
    if not getattr(cfg, "has_own_embed", True):
        drafter.bind_embed(target_model.draft_embed)
    tap = list(cfg.target_layer_ids)
    mask_id = cfg.mask_token_id
    # A DFlash-derived head spends slot 0 on the anchor, so it can propose block_size-1.
    kdraft = drafter.max_draft
    cap_ceiling = kdraft if max_draft_tokens is None else max(1, min(max_draft_tokens, kdraft))
    cap = cap_ceiling

    eos_ids = eos_token_ids(tokenizer)

    # --- tokenize prompt ---
    ids = prompt_ids if prompt_ids is not None else encode_prompt(
        tokenizer, prompt, use_chat=apply_chat_template)

    # Prefix caching: the caller may pass a target cache + drafter ctx already holding the
    # first `reuse_len` tokens (a shared conversation prefix); then we only prefill the
    # suffix. `cache is None` = the standalone/library path (fresh caches, reuse_len=0).
    if cache is None:
        cache = _make_target_cache(target_model)
        ctx_caches = drafter.make_ctx_cache()
        reuse_len = 0
    target_model.reset_spec()                          # hybrid targets: clear capture state
    st = _Streamer(tokenizer, eos_ids, on_text, stop)
    t0 = time.time()

    # --- prefill (only the suffix past any reused prefix; chunked, feeding the drafter
    # context per chunk so long prompts never materialize all fused states at once) ---
    suffix = ids[reuse_len:] if reuse_len else ids
    logits, _ = _prefill_tapped(
        target_model, suffix, cache, tap,
        drafter=drafter, ctx_caches=ctx_caches, ctx_offset=reuse_len, marks=prefill_marks,
        on_mark=(lambda p: on_prefill(cache, ctx_caches, p)) if on_prefill else None)
    if on_prefill is not None:
        on_prefill(cache, ctx_caches, len(ids))   # caches hold exactly `ids` right now
    n_cached = len(ids)
    pending = _pick(logits[0, -1], temperature, top_p, top_k)  # first committed token
    mx.async_eval([c.k for c in ctx_caches])   # schedule; round 1's sync will wait on it
    pen = _Penalizer(presence_penalty, frequency_penalty)      # OpenAI presence/frequency penalties
    pen.add([pending])
    lp_list: list | None = [] if logprobs is not None else None
    if lp_list is not None:                                    # first token came from prefill logits
        lp_list.extend(_logprobs_for_block(logits[0, -1][None, :], [pending], logprobs))

    index = None
    lk_gate = None
    if lookup_drafts:
        from .lookup import LongDraftGate, NGramIndex  # deferred: lookup.py imports this module

        # Hybrid uses a stricter 4-gram minimum than pure lookup mode: here a spurious hit
        # doesn't just cost a wider verify — it forgoes a productive drafter round (~2.3
        # tokens). Trigrams fired on ~4-10% of chat rounds (measured 2-6% slower); 4-grams
        # almost never fire spuriously, while genuine copying has them in abundance.
        index = NGramIndex(min_n=4, max_n=5, max_draft=max(1, lookup_max_draft))
        index.extend(ids + [pending])
        lk_gate = LongDraftGate()

    out_ids: list[int] = [pending]
    accept_lengths: list[int] = []
    target_forwards = 1
    lookup_rounds = 0
    _rt = time.perf_counter()      # round-period clock for the controller's observed timings

    st.update(out_ids)
    while len(out_ids) < max_new_tokens and pending not in eos_ids and not st.stopped:
        lk_draft = []
        if index is not None:
            # clamp to the remaining budget: no verify width wasted past max_new_tokens
            lk_draft = index.propose(
                long_draft=lookup_long_draft if lk_gate.allowed else None,
            )[:max_new_tokens - len(out_ids)]
        use_conf = confidence_threshold > 0.0 and drafter.confidence_head is not None
        if cap_controller is not None and not lk_draft:
            # cap 0 = the controller parked speculation (live acceptance too low for this
            # machine's verify slope): run a plain committed step; probe rounds re-enter.
            cap = max(0, min(cap_controller.cap, cap_ceiling))
        if not lk_draft and cap == 0 and not (temperature > 0.0 or pen.active
                                              or lp_list is not None):
            # ---- parked sprint ----
            # The controller decided speculation loses on the current content. A one-step
            # parked round would still pay this loop's per-round sync (~15% over the
            # pipelined baseline), so run the plain steps exactly like greedy_generate's
            # fast path — schedule step t+1 before syncing step t — until the controller
            # schedules a probe (or flips back to a positive cap). The tap rides along so
            # the drafter context stays current and probe rounds draft correctly.
            steps = cap_controller.probe_every
            y = mx.array([[pending]])
            logits, fused = target_model.verify(y, cache, tap)
            nxt_arr = mx.argmax(logits[0, -1])
            drafter.update_context(fused, ctx_offset=n_cached, ctx_caches=ctx_caches)
            n_cached += 1
            mx.async_eval(nxt_arr, [c.k for c in ctx_caches])
            while True:
                steps -= 1
                now = time.perf_counter()
                cap_controller.update(0, 0, round_ms=(now - _rt) * 1e3, committed=1)
                _rt = now
                nxt = int(nxt_arr.item())
                out_ids.append(nxt)
                accept_lengths.append(1)
                target_forwards += 1
                if on_round is not None:
                    # Parked: the controller judged speculation a loss on this content, so
                    # these are plain committed steps. Reporting them keeps the live view
                    # honest about *why* throughput dropped instead of just going quiet.
                    on_round(drafted=0, accepted=0, committed=1, cap=0, source="plain")
                if index is not None:
                    index.extend([nxt])
                pending = nxt
                st.update(out_ids)
                if (steps <= 0 or len(out_ids) >= max_new_tokens or pending in eos_ids
                        or st.stopped or cap_controller.cap != 0):
                    break
                y = nxt_arr.reshape(1, 1)
                logits, fused = target_model.verify(y, cache, tap)
                nxt_next = mx.argmax(logits[0, -1])
                drafter.update_context(fused, ctx_offset=n_cached, ctx_caches=ctx_caches)
                n_cached += 1
                mx.async_eval(nxt_next, [c.k for c in ctx_caches])
                nxt_arr = nxt_next
            continue
        if not lk_draft and cap == 0:
            # parked, but sampling / penalties / logprobs need per-token materialization:
            # a single plain committed step through the round's shared bookkeeping below
            draft, n = [], 0
            verify_ids = mx.array([[pending]])
            v_logits, v_fused = target_model.verify(verify_ids, cache, tap)
            tt0 = mx.argmax(pen.apply(v_logits[0], draft), axis=-1)
            mx.eval(tt0, v_fused)
            committed = [int(tt0[0].item())] if temperature == 0.0 else [
                _pick(pen.apply(v_logits[0], draft)[0], temperature, top_p, top_k)]
        elif lk_draft:
            # ---- free lookup draft (a copy run was detected): verify the continuation of
            # the earlier occurrence instead of running the drafter this round. The drafter
            # context still updates below (from v_fused), so drafter rounds stay correct.
            lookup_rounds += 1
            draft = lk_draft
            if temperature > 0.0:
                verify_ids = mx.array([[pending] + draft])
                v_logits, v_fused = target_model.verify(verify_ids, cache, tap)
                mx.eval(v_logits, v_fused)
                vocab = v_logits.shape[-1]     # point-mass proposal: q is one-hot per token
                q_probs = (mx.arange(vocab)[None, :]
                           == mx.array(draft)[:, None]).astype(mx.float32)
                n, repl = _spec_sample_accept(
                    pen.apply(v_logits[0], draft), draft, q_probs, temperature, top_p, top_k)
                committed = draft[:n] + [repl]
            else:
                draft_arr = mx.array(draft)
                verify_ids = mx.concatenate(
                    [mx.array([pending], dtype=draft_arr.dtype), draft_arr]).reshape(1, -1)
                v_logits, v_fused = target_model.verify(verify_ids, cache, tap)
                tt_arr = mx.argmax(pen.apply(v_logits[0], draft), axis=-1)
                match = (draft_arr
                         == tt_arr[: len(draft)].astype(draft_arr.dtype)).astype(mx.int32)
                n_arr = mx.cumprod(match).sum()
                mx.eval(n_arr, tt_arr)
                n = int(n_arr.item())
                tt = tt_arr.tolist()
                committed = draft[:n] + [tt[n]]
        else:
            # ---- 1. draft a block ----
            # A bidirectional block (DeepSpec-native heads) runs the backbone full-width:
            # each position's hidden depends on the whole block, and shrinking it would
            # change the distribution the drafter was trained on. A CAUSAL block
            # (DFlash-lineage: Nemotron, Muse-Glimmer) truncates to the rows the head
            # actually reads — bit-identical, and on Muse's 15-wide 2.3B backbone worth
            # ~16 ms/round (see DSparkDrafter.draft_width). Either way the lm_head and the
            # sequential markov head run over just the first `cap` positions instead of
            # all `k` — the rest used to be computed and thrown away every round (the
            # dominant slice of drafter time at small caps).
            w = drafter.draft_width(cap)
            block_ids = [pending] + [mask_id] * (w - 1)
            noise = drafter.embed(mx.array([block_ids]))            # [1, k, H]
            block_hidden = drafter.backbone(noise, n_cached, ctx_caches)
            head_hidden = drafter.head_slice(block_hidden, cap)     # only the verified positions
            base_logits = drafter.compute_logits(head_hidden)[0]    # [cap, V]

            if temperature > 0.0 or use_conf or pen.active:
                # Two-phase path: the sampled path needs the q distributions, the confidence head
                # truncates the draft, and penalties need the draft as a list (to penalize each
                # verify position by the base counts + its draft prefix) — so the drafted tokens
                # are materialized *before* the verify forward, one extra sync per round.
                if temperature > 0.0:
                    draft_arr, q_probs = drafter.sample_block_probs(
                        base_logits, pending, temperature, top_p, top_k)
                    mx.eval(draft_arr, q_probs)
                else:
                    draft_arr = drafter.sample_block(base_logits, first_prev_token=pending)
                    mx.eval(draft_arr)
                    q_probs = None
                draft = [int(x) for x in draft_arr.tolist()]

                # optional confidence-based truncation (adaptive block length, within cap).
                # Paper §3.2.1: c_k is the *conditional* survival prob of position k given
                # the prefix accepted; the prefix survival prob is the cumulative product
                # a_j = ∏_{i<=j} c_i (Eq 7-8). Keep extending the draft while a_j stays
                # above the threshold, i.e. while the next token likely survives verify.
                if use_conf:
                    prev_tokens = mx.array([pending] + draft[:-1])
                    conf = mx.sigmoid(drafter.confidence_logits(head_hidden[0], prev_tokens))
                    mx.eval(conf)
                    surv, keep = 1.0, 0
                    for i, c in enumerate(conf.tolist()):
                        surv *= c
                        if surv < confidence_threshold:
                            break
                        keep = i + 1
                    draft = draft[:keep]
                if not draft:
                    draft = [int(draft_arr[0].item())]  # always propose >=1 (q_probs[0] aligns)

                # ---- 2. verify with the target ----
                verify_ids = mx.array([[pending] + draft])     # [1, 1+len(draft)]
                v_logits, v_fused = target_model.verify(verify_ids, cache, tap)
                mx.eval(v_logits, v_fused)

                # ---- 3. accept ----
                if temperature > 0.0:
                    n, repl = _spec_sample_accept(
                        pen.apply(v_logits[0], draft), draft, q_probs, temperature, top_p, top_k)
                    committed = draft[:n] + [repl]             # accepted prefix + residual/bonus
                else:
                    tt = [int(x) for x in mx.argmax(pen.apply(v_logits[0], draft), axis=-1).tolist()]
                    n = 0
                    while n < len(draft) and draft[n] == tt[n]:
                        n += 1
                    committed = draft[:n] + [tt[n]]            # accepted prefix + bonus
            else:
                # Fused greedy path (the default). The drafted tokens never round-trip to
                # the CPU before verify: verify_ids is assembled on-GPU and the accepted-
                # prefix length is computed in-graph (cumprod of positionwise argmax
                # matches), so the whole round — draft heads + verify forward + accept —
                # reaches the device as one batched graph with a single sync.
                draft_arr = drafter.sample_block(base_logits, first_prev_token=pending)
                verify_ids = mx.concatenate(
                    [mx.array([pending], dtype=draft_arr.dtype), draft_arr]).reshape(1, -1)
                v_logits, v_fused = target_model.verify(verify_ids, cache, tap)
                tt_arr = mx.argmax(v_logits[0], axis=-1)
                match = (draft_arr == tt_arr[: draft_arr.shape[0]]).astype(mx.int32)
                n_arr = mx.cumprod(match).sum()
                mx.eval(n_arr, tt_arr, draft_arr)
                n = int(n_arr.item())
                draft = draft_arr.tolist()
                tt = tt_arr.tolist()
                committed = draft[:n] + [tt[n]]                # accepted prefix + bonus
        target_forwards += 1
        accept_lengths.append(len(committed))
        if lk_draft:
            lk_gate.update(len(draft), n, max(1, lookup_max_draft))
        if cap_controller is not None:
            now = time.perf_counter()
            if not lk_draft:
                # only drafter/parked rounds inform the cap (lookup drafts have their own
                # acceptance); the measured round period grounds the controller's cost
                # model in observed reality (replay cascades, syncs, Python overhead)
                cap_controller.update(n, len(draft), round_ms=(now - _rt) * 1e3,
                                      committed=len(committed))
            _rt = now
        if on_round is not None:
            # A round is milliseconds, so this dict + queue put is noise; guard it anyway so
            # the default path (no listener) costs exactly one `is not None` test.
            on_round(drafted=len(draft), accepted=n, committed=len(committed),
                     cap=cap,
                     source=("lookup" if lk_draft else
                             "plain" if not draft else "drafter"))

        # ---- 4. update caches/context ----
        target_model.rollback(cache, len(draft) - n, draft[:n])
        # commit [pending, accepted drafts] (positions n_cached..n_cached+n) as context
        drafter.update_context(
            v_fused[:, : n + 1, :], ctx_offset=n_cached, ctx_caches=ctx_caches
        )
        n_cached = n_cached + n + 1
        # schedule (don't block): the ctx projections run while Python commits tokens and
        # streams text; the next round's single sync waits on them anyway.
        mx.async_eval([c.k for c in ctx_caches])

        appended = []
        for tok in committed:
            out_ids.append(tok)
            appended.append(tok)
            if tok in eos_ids:
                break
        pen.add(appended)
        if lp_list is not None and appended:
            # raw target logits at the committed verify positions (0..len(appended)-1)
            lp_list.extend(_logprobs_for_block(v_logits[0][:len(appended)], appended, logprobs))
        if index is not None:
            index.extend(appended)
        pending = out_ids[-1]        # eos mid-committed ends the loop (not committed[-1])
        st.update(out_ids)

        if verbose:
            src = "lookup" if lk_draft else "drafter"
            print(f"  round {len(accept_lengths):3d}: {src} drafted {len(draft)}, "
                  f"accepted {n}, committed {len(committed)}")
    st.flush()

    secs = time.time() - t0
    # strip trailing eos for display (or cut at a stop string)
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
        lookup_rounds=lookup_rounds,
        logprobs=lp_list,
    )
