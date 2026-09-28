/* mlx-dspark docs: the interactive pieces. Each initializes only if its element is on the page.
   Data comes from <script id="site-data"> (built from REGISTRY + docs/data/models.json). */
(function () {
  "use strict";
  var dataEl = document.getElementById("site-data");
  var MODELS = dataEl ? JSON.parse(dataEl.textContent) : [];
  var root = document.body.getAttribute("data-root") || "";
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function best(v) {
    for (var i = 0; i < v.arms.length; i++) if (v.arms[i].default) return v.arms[i];
    return v.arms[0];
  }
  function fx(n) { return n.toFixed(2) + "×"; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  var armLabel = function (a) { return a.label || a.mode; };

  /* ================================================================ the race */
  var RACE_TEXT = [
    "def is_prime(n: int) -> bool:",
    '    """Check whether a number is prime.',
    "",
    "    Args:",
    "        n: The integer to check for primality.",
    "",
    "    Returns:",
    "        True if n is a prime number, False otherwise.",
    '    """',
    "    if n < 2:",
    "        return False",
    "    if n == 2:",
    "        return True",
    "    if n % 2 == 0:",
    "        return False",
    "    for i in range(3, int(n**0.5) + 1, 2):",
    "        if n % i == 0:",
    "            return False",
    "    return True"
  ].join("\n");
  // BPE-ish: a leading space rides with the next word; a newline carries its indentation
  function tokenize(s) {
    var re = /\n+ *|(?: ?[A-Za-z_]+)|(?: ?\d+)|(?: ?"""|\*\*|==|->)|(?: ?[^\sA-Za-z_\d])| +/g;
    return s.match(re);
  }
  var TOKENS = tokenize(RACE_TEXT);
  if (TOKENS.join("") !== RACE_TEXT) TOKENS = RACE_TEXT.split(/(?=\s)/);   // never drop a character
  var OFF = [0];                                     // char offset where each token starts
  TOKENS.forEach(function (t, i) { OFF.push(OFF[i] + t.length); });
  // per-character syntax classes: a tiny Python highlighter, first match wins
  var CLS = (function () {
    var c = new Array(RACE_TEXT.length).fill("");
    function mark(re, cls, group) {
      var m;
      while ((m = re.exec(RACE_TEXT))) {
        var sub = group ? m[group] : m[0];
        var from = m.index + (group ? m[0].indexOf(sub) : 0);
        for (var i = from; i < from + sub.length; i++) if (!c[i]) c[i] = cls;
      }
    }
    mark(/"""[\s\S]*?"""/g, "s-str");
    mark(/def (\w+)/g, "s-fn", 1);
    mark(/\b(def|if|return|for|in|and|or|not|else|elif)\b/g, "s-kw");
    mark(/\b(int|bool|range|True|False|None)\b/g, "s-bi");
    mark(/\b\d+(\.\d+)?\b/g, "s-num");
    mark(/->|\*\*|==|[=<>%*+:(),-]/g, "s-op");
    return c;
  })();
  function hl(from, to) {
    var out = "", cur = null, buf = "";
    function flush() { if (buf) out += cur ? '<span class="' + cur + '">' + esc(buf) + "</span>" : esc(buf); buf = ""; }
    for (var i = from; i < to; i++) {
      var k = CLS[i] || "";
      if (k !== cur) { flush(); cur = k; }
      buf += RACE_TEXT[i];
    }
    flush();
    return out;
  }
  function hlTok(i) { return hl(OFF[i], OFF[i + 1]); }
  // plausible wrong guesses for a rejected draft
  var DECOYS = [" n", " x", " the", " is", " 1", " return", ":", " (", " value", " True", " in", " num"];

  function solveP(A, C) {
    // per-position acceptance p such that E[committed] = (1 - p^(C+1)) / (1 - p) = A
    A = Math.max(1.01, Math.min(A, C + 0.95));
    var lo = 0.001, hi = 0.999;
    for (var i = 0; i < 50; i++) {
      var p = (lo + hi) / 2;
      var e = (1 - Math.pow(p, C + 1)) / (1 - p);
      if (e < A) lo = p; else hi = p;
    }
    return (lo + hi) / 2;
  }

  function initRace(fig) {
    var wrap = fig.closest(".ide-wrap") || fig;
    var replay = fig.querySelector("[data-race-replay]");
    var chipsBox = wrap.querySelector("[data-race-chips]");
    var more = wrap.querySelector("[data-race-more]");
    var moreWrap = wrap.querySelector("[data-race-more-wrap]");
    var title = fig.querySelector("[data-race-title]");
    var hint = wrap.querySelector("[data-race-hint]");
    var verdict = fig.querySelector("[data-verdict]");
    var specLabel = fig.querySelector("[data-spec-label]");
    var panes = fig.querySelectorAll(".pane");
    var L = [0, 1].map(function (i) {
      var p = panes[i];
      return { text: p.querySelector("[data-text]"), rate: p.querySelector("[data-rate]"),
               passes: p.querySelector("[data-passes]").parentElement, gutter: p.querySelector("[data-gutter]"),
               line: p.querySelector("[data-line]"), scroll: p.querySelector("[data-scroll]"), lines: 0 };
    });
    var options = MODELS.filter(function (v) { return v.arms && v.arms.length && best(v).content; });
    options.sort(function (a, b) { return best(b).speedup - best(a).speedup; });
    // a handful of visible choices that span the range (best ratio, big + fast, small, tiny,
    // and a modest MoE), the rest one click away in "More models"
    var FEATURED = [["qwen3.8-27b", "8-bit"], ["gemma-4-12b", "8-bit"], ["minicpm5-2b", "bf16"],
                    ["lfm2.5-1.2b", "bf16"], ["qwen3.6-35b-a3b", "4-bit"]];
    var featured = FEATURED.map(function (f) {
      return options.findIndex(function (v) { return v.slug === f[0] && v.quant === f[1]; });
    }).filter(function (i) { return i >= 0; });
    function chip(i) {
      var v = options[i], b = el("button", "race-chip");
      b.type = "button";
      b.setAttribute("aria-pressed", "false");
      b.title = v.name + " " + v.quant + " with " + armLabel(best(v));
      b.innerHTML = esc(v.name) + " <small>" + fx(best(v).speedup) + "</small>";
      b.addEventListener("click", function () { choose(i); });
      chipsBox.insertBefore(b, moreWrap);
      return b;
    }
    var chips = featured.map(chip);
    var extra = null;   // a model picked from "More models" shows up as its own chip
    options.forEach(function (v, i) {
      if (featured.indexOf(i) >= 0) return;
      var o = el("option", null, v.name + " " + v.quant + " (" + fx(best(v).speedup) + ")");
      o.value = String(i);
      more.appendChild(o);
    });
    more.addEventListener("change", function () { if (more.value !== "") choose(+more.value); });
    var current = featured.length ? featured[0] : 0;
    function mark() {
      chips.forEach(function (b, j) { b.setAttribute("aria-pressed", String(featured[j] === current)); });
      if (extra) { extra.remove(); extra = null; }
      if (featured.indexOf(current) < 0) {
        extra = chip(current);
        extra.setAttribute("aria-pressed", "true");
      }
      more.value = "";
    }
    function choose(i) { current = i; mark(); run = run || {}; go(); }
    function nextIndex() {
      var at = featured.indexOf(current);
      return featured[(at + 1) % featured.length];
    }
    mark();

    var run = null;
    function plan(v) {
      var a = best(v);
      var base = a.baseline;
      var codeX = (a.content && a.content.code) || a.speedup;
      var specTps = base * codeX;
      var C = a.cap || 7;
      var A = a.accept ? Math.min(C + 0.9, a.accept * codeX / a.speedup) : Math.min(C, codeX + 1);
      var p = solveP(A, C);
      var roundMs = 1000 * A / specTps;
      var rnd = mulberry32(v.slug.length * 7919 + v.quant.length * 31 + C);
      var rounds = [], pos = 0, t = 0;
      while (pos < TOKENS.length) {
        var k = 0;
        while (k < C && rnd() < p) k++;
        var room = TOKENS.length - pos;
        var drafts = Math.min(C, room);
        k = Math.min(k, room);
        var commit = Math.min(k + 1, room);
        var decoy = DECOYS[Math.floor(rnd() * DECOYS.length)];
        if (decoy === TOKENS[pos + k]) decoy = " x";
        // round time scales with how many tokens it produced, so the lane averages specTps
        var dur = roundMs * (0.55 + 0.45 * commit / A);
        rounds.push({ t0: t, t1: t + dur, pos: pos, k: k, drafts: drafts, commit: commit, decoy: decoy });
        t += dur; pos += commit;
      }
      var scale = (TOKENS.length / specTps * 1000) / t;   // land exactly on the measured rate
      rounds.forEach(function (r) { r.t0 *= scale; r.t1 *= scale; });
      return { v: v, a: a, base: base, specTps: specTps, codeX: codeX, rounds: rounds,
               baseMs: 1000 / base, baseEnd: TOKENS.length * 1000 / base, specEnd: t * scale };
    }

    function tokSpan(cls, inner, raw) {
      var ws = /^\s+$/.test(raw) ? " ws" : "";
      return '<span class="tok ' + cls + ws + '">' + inner + "</span>";
    }
    function rate(n, ms) { return n >= 6 ? (n / ms * 1000).toFixed(1) : "…"; }
    function setPasses(lane, n) { lane.passes.innerHTML = "<b>" + n + "</b> forward pass" + (n === 1 ? "" : "es"); }
    var CURSOR = '<span class="cursor" aria-hidden="true"></span>';
    // write the editor: code, line numbers, current-line highlight, keep the cursor in view
    function paint(lane, html, committedChars, shownText) {
      lane.text.innerHTML = html;
      var lines = shownText.split("\n").length;
      if (lines !== lane.lines) {
        var g = [];
        for (var i = 1; i <= lines; i++) g.push(String(i));
        lane.gutter.textContent = g.join("\n");
        lane.lines = lines;
      }
      var cur = RACE_TEXT.slice(0, committedChars).split("\n").length - 1;
      lane.line.style.setProperty("--line", cur);
      var lh = lane.line.offsetHeight || 20;
      var want = (cur + 3) * lh - lane.scroll.clientHeight;
      lane.scroll.scrollTop = Math.max(0, want);
    }
    function renderBase(P, now) {
      var n = Math.min(TOKENS.length, Math.floor(now / P.baseMs));
      var done = n >= TOKENS.length;
      paint(L[0], hl(0, OFF[n]) + (done ? "" : CURSOR), OFF[n], RACE_TEXT.slice(0, OFF[n]));
      setPasses(L[0], n);
      L[0].rate.textContent = rate(n, Math.min(now, P.baseEnd));
      return done;
    }
    function renderSpec(P, now) {
      var R = P.rounds, i = 0;
      while (i < R.length && R[i].t1 <= now) i++;
      var done = i >= R.length;
      var committed = done ? TOKENS.length : R[i].pos;
      var prev = i > 0 ? R[i - 1] : null;
      var cur = done ? null : R[i];
      var phase = cur ? (now - cur.t0) / (cur.t1 - cur.t0) : 1;
      var html, shown;
      if (prev && cur && phase < 0.45) {
        // how the last round ended: accepted drafts, the rejected one, the model's own token
        var fixI = prev.pos + prev.k;
        html = hl(0, OFF[fixI]);
        shown = RACE_TEXT.slice(0, OFF[fixI]);
        if (prev.k < prev.drafts) { html += tokSpan("rej", esc(prev.decoy), prev.decoy); shown += prev.decoy; }
        if (fixI < TOKENS.length) { html += tokSpan("fix", hlTok(fixI), TOKENS[fixI]); shown += TOKENS[fixI]; }
      } else {
        html = hl(0, OFF[committed]);
        shown = RACE_TEXT.slice(0, OFF[committed]);
      }
      if (cur) {
        html += CURSOR;
        var cls = phase < 0.45 ? "draft" : "check";
        for (var j = 0; j < cur.drafts; j++) {
          var ti = cur.pos + j;
          if (j === cur.k) { html += tokSpan(cls, esc(cur.decoy), cur.decoy); shown += cur.decoy; }
          else if (ti < TOKENS.length) { html += tokSpan(cls, hlTok(ti), TOKENS[ti]); shown += TOKENS[ti]; }
        }
      }
      paint(L[1], html, OFF[committed], shown);
      setPasses(L[1], done ? R.length : i);
      L[1].rate.textContent = rate(committed, Math.min(now, P.specEnd));
      return done;
    }
    function setVerdict(P, state) {
      verdict.classList.toggle("is-done", state === "done");
      if (state === "done") {
        var nxt = options[nextIndex()];
        verdict.innerHTML = '<span><span class="ok-mark" aria-hidden="true">✓</span> Identical output, <span class="big">' +
          fx(P.baseEnd / P.specEnd) + "</span> faster: " + TOKENS.length + " tokens in " + P.rounds.length +
          " passes instead of " + TOKENS.length + '</span><button class="ide-next" type="button">Race ' +
          esc(nxt.name) + " next</button>";
        verdict.querySelector(".ide-next").addEventListener("click", function () { choose(nextIndex()); });
      } else if (state === "spec-done") {
        verdict.textContent = "mlx-dspark finished in " + (P.specEnd / 1000).toFixed(1) + " s. Plain decoding is still writing…";
      } else {
        verdict.textContent = "Both editors receive the same tokens. Only the number of passes through the model differs.";
      }
    }
    function describe(P) {
      if (title) title.innerHTML = "Racing <b>" + esc(P.v.name + " " + P.v.quant) + "</b>";
      if (specLabel) specLabel.textContent = "mlx-dspark with " + armLabel(P.a);
      if (hint) hint.textContent = "Replayed at this pair's measured code speed on an M4 Pro: " + P.base +
        " tok/s plain, " + P.specTps.toFixed(1) + " tok/s with " + armLabel(P.a) + ".";
    }
    function go() {
      if (run) cancelAnimationFrame(run.raf);
      var P = plan(options[current]);
      describe(P);
      if (reduceMotion && !run) {
        // no autoplay: show the finished state, animate only when asked
        renderBase(P, P.baseEnd + 1); renderSpec(P, P.specEnd + 1); setVerdict(P, "done");
        run = { raf: 0 };
        return;
      }
      var now = 0, last = null, specDone = false;
      setVerdict(P, "start");
      run = { raf: 0 };
      function frame(ts) {
        if (last === null) last = ts;
        now += Math.min(1000, ts - last);  // real time, but a backgrounded tab pauses instead of jumping ahead
        last = ts;
        var b = renderBase(P, now);
        var s2 = renderSpec(P, now);
        if (s2 && !specDone) { specDone = true; setVerdict(P, "spec-done"); }
        if (b && s2) { setVerdict(P, "done"); return; }
        run.raf = requestAnimationFrame(frame);
      }
      run.raf = requestAnimationFrame(frame);
    }
    replay.addEventListener("click", function () { run = run || {}; go(); });
    if ("IntersectionObserver" in window && !reduceMotion) {
      var io = new IntersectionObserver(function (en) {
        if (en[0].isIntersecting) { io.disconnect(); go(); }
      }, { threshold: 0.25 });
      io.observe(fig);
    } else { go(); }
  }

  /* ================================================================ the picker */
  var MEM_COMFORT = 0.70, MEM_TIGHT = 0.85;
  function fit(v, mem) {
    if (v.ram_gb == null) return null;
    if (v.ram_gb <= mem * MEM_COMFORT) return "ok";
    if (v.ram_gb <= mem * MEM_TIGHT) return "tight";
    return null;
  }
  function balanced(v) { return best(v).speedup * Math.sqrt(v.params_b || 1); }
  var GOALS = {
    balanced: { score: balanced, why: function (v) { return v.picker_note || null; } },
    speedup: { score: function (v) { return best(v).speedup; } },
    // raw speed among models that make sense for this much memory (else a 1.2B wins everywhere)
    tps: { filter: function (v, mem) { return (v.params_b || 0) >= mem / 8; },
           score: function (v) { var a = best(v); return a.tps || a.baseline * a.speedup; } },
    agents: { filter: function (v) { return v.tools === true; }, score: function (v) { return balanced(v) * (v.kind === "dense" ? 1.08 : 1); } },
    size: { score: function (v) { return (v.params_b || 0) * 10 + best(v).speedup; } }
  };
  function initPicker(box) {
    var results = box.querySelector("[data-picker-results]");
    function current(name) { var c = box.querySelector('input[name="' + name + '"]:checked'); return c ? c.value : null; }
    function command(v, goal) {
      var a = best(v);
      var parts = ["mlx-dspark serve", "--model " + v.target];
      if (v.slug === "nanbeige4.2-3b") parts.push("--trust-remote-code");
      if (v.slug === "ternary-bonsai-27b") parts.push("--max-draft auto");
      if (a.flags) parts.push(a.flags);
      if (goal === "agents") parts.push("--no-thinking");
      return parts.join(" \\\n  ");
    }
    function render() {
      var mem = +current("mem"), goal = current("goal");
      var G = GOALS[goal];
      var cands = MODELS.filter(function (v) {
        return v.arms && v.arms.length && v.recommended !== false && fit(v, mem) && (!G.filter || G.filter(v, mem));
      });
      cands.forEach(function (v) { v._s = G.score(v) * (fit(v, mem) === "tight" ? 0.82 : 1); });
      cands.sort(function (a, b) { return b._s - a._s; });
      // one row per model: the better-scoring quant wins
      var seen = {}, top = [];
      cands.forEach(function (v) { if (!seen[v.slug] && top.length < 3) { seen[v.slug] = 1; top.push(v); } });
      results.innerHTML = "";
      if (!top.length) {
        results.appendChild(el("li", "pick-empty", "Nothing measured fits comfortably at this size yet. Every model still runs with drafter-free lookup speculation: mlx-dspark serve --model <any MLX model> --mode lookup."));
        return;
      }
      top.forEach(function (v) {
        var a = best(v), f = fit(v, mem);
        var li = el("li", "pick-row");
        var range = a.content ? [a.baseline * Math.min(a.content.chat, a.content.code, a.content.math), a.baseline * Math.max(a.content.chat, a.content.code, a.content.math)] : null;
        var why = v.picker_note || "";
        if (goal === "agents") why = v.kind === "dense" ? "tool calling, and its prefix cache trims cleanly between agent turns" : "tool calling, with checkpoint prefix caching between agent turns";
        li.innerHTML =
          '<div><h3 class="pick-name"><a href="' + root + "models/" + v.slug + '/">' + esc(v.name) + '</a><span class="q">' + esc(v.quant) + "</span></h3>" +
          (why ? '<p class="pick-why">' + esc(why.charAt(0).toUpperCase() + why.slice(1)) + ".</p>" : "") +
          '<span class="fit ' + (f === "ok" ? "fit-ok" : "fit-tight") + '">' + (f === "ok" ? "Fits comfortably, " : "Tight fit, close other apps, ") + esc(v.ram) + "</span></div>" +
          '<div class="pick-stats"><div class="pick-stat"><b>' + fx(a.speedup) + "</b><span>faster, mean</span></div>" +
          (range ? '<div class="pick-stat"><b>' + Math.round(range[0]) + "–" + Math.round(range[1]) + "</b><span>tokens / s</span></div>" : "") + "</div>" +
          '<div class="pick-cmd"><div class="code" data-lang="shell"><pre><code>' + esc(command(v, goal)) + '</code></pre><button class="copy" type="button" aria-label="Copy command"><span>Copy</span></button></div>' +
          (goal === "agents" ? '<p class="small muted" style="margin:8px 0 0">Then run <code>mlx-dspark claude</code> in a second terminal.</p>' : "") + "</div>";
        results.appendChild(li);
      });
    }
    box.addEventListener("change", render);
    render();
  }

  /* ================================================================ dot plot */
  var SHAPES = {
    chat: function (x, y) { return '<circle class="dp-mk dp-chat" cx="' + x + '" cy="' + y + '" r="5.5"/>'; },
    code: function (x, y) { return '<rect class="dp-mk dp-code" x="' + (x - 5) + '" y="' + (y - 5) + '" width="10" height="10" rx="1.5"/>'; },
    math: function (x, y) { return '<path class="dp-mk dp-math" d="M' + x + " " + (y - 6.5) + "L" + (x + 6.5) + " " + y + "L" + x + " " + (y + 6.5) + "L" + (x - 6.5) + " " + y + 'Z"/>'; }
  };
  function initDotplot(box) {
    var rows = MODELS.filter(function (v) { return v.arms && v.arms.length && best(v).content; })
      .map(function (v) { return { v: v, a: best(v) }; })
      .sort(function (p, q) { return q.a.speedup - p.a.speedup; });
    var legend = el("ul", "dp-legend");
    legend.innerHTML =
      '<li><svg viewBox="-7 -7 14 14" aria-hidden="true">' + SHAPES.chat(0, 0) + "</svg>Chat</li>" +
      '<li><svg viewBox="-7 -7 14 14" aria-hidden="true">' + SHAPES.code(0, 0) + "</svg>Code</li>" +
      '<li><svg viewBox="-7 -7 14 14" aria-hidden="true">' + SHAPES.math(0, 0) + "</svg>Math</li>" +
      '<li><svg viewBox="0 0 14 14" aria-hidden="true"><line x1="7" y1="1" x2="7" y2="13" class="dp-mean"/></svg>Mean of the three</li>' +
      '<li><svg viewBox="0 0 14 14" aria-hidden="true"><line x1="7" y1="0" x2="7" y2="14" class="dp-one"/></svg>Plain decoding (1×)</li>';
    var frame = el("div", "dotplot-bg");
    var holder = el("div");
    var tip = el("div", "dp-tip"); tip.hidden = true;
    frame.appendChild(legend); frame.appendChild(holder);
    box.appendChild(frame); box.appendChild(tip);

    function draw() {
      var W = Math.max(300, holder.clientWidth);
      var narrow = W < 600;
      var labelW = narrow ? 0 : Math.min(290, W * 0.34);
      var rowH = narrow ? 46 : 32, top = 26, right = 54;
      var x0 = labelW + 8, x1 = W - right;
      var lo = 0.8, hi = 4.6;
      var X = function (s) { return x0 + (s - lo) / (hi - lo) * (x1 - x0); };
      var H = top + rows.length * rowH + 6;
      var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Speedup over plain decoding for each measured model, by kind of prompt">';
      [1, 2, 3, 4].forEach(function (t) {
        s += '<line class="' + (t === 1 ? "dp-one" : "dp-grid") + '" x1="' + X(t) + '" x2="' + X(t) + '" y1="' + (top - 6) + '" y2="' + (H - 4) + '"/>';
        s += '<text class="dp-tick" x="' + X(t) + '" y="' + (top - 12) + '" text-anchor="middle">' + t + "×</text>";
      });
      rows.forEach(function (r, i) {
        var a = r.a, c = a.content;
        var y = top + i * rowH + (narrow ? 30 : rowH / 2);
        var mn = Math.min(c.chat, c.code, c.math), mx = Math.max(c.chat, c.code, c.math);
        s += '<g class="dp-row" tabindex="0" role="link" data-i="' + i + '" aria-label="' + esc(r.v.name + " " + r.v.quant + ": " + fx(a.speedup) + " mean; chat " + fx(c.chat) + ", code " + fx(c.code) + ", math " + fx(c.math)) + '">';
        s += '<rect class="dp-hit" x="0" y="' + (top + i * rowH) + '" width="' + W + '" height="' + rowH + '" rx="6"/>';
        if (narrow) s += '<text class="dp-name" x="4" y="' + (y - 13) + '">' + esc(r.v.name) + '<tspan class="q"> ' + esc(r.v.quant) + "</tspan></text>";
        else s += '<text class="dp-name" x="' + labelW + '" y="' + (y + 4.5) + '" text-anchor="end">' + esc(r.v.name) + '<tspan class="q"> ' + esc(r.v.quant) + "</tspan></text>";
        s += '<line class="dp-range" x1="' + X(mn) + '" x2="' + X(mx) + '" y1="' + y + '" y2="' + y + '"/>';
        s += '<line class="dp-mean" x1="' + X(a.speedup) + '" x2="' + X(a.speedup) + '" y1="' + (y - 8) + '" y2="' + (y + 8) + '"/>';
        // draw the one furthest from the mean last so overlaps stay readable
        ["chat", "code", "math"].sort(function (p, q) { return Math.abs(c[p] - a.speedup) - Math.abs(c[q] - a.speedup); })
          .forEach(function (k) { s += SHAPES[k](X(c[k]), y); });
        s += '<text class="dp-val" x="' + (X(mx) + 12) + '" y="' + (y + 4.5) + '">' + fx(a.speedup) + "</text>";
        s += "</g>";
      });
      s += "</svg>";
      holder.innerHTML = s;
      holder.querySelectorAll(".dp-row").forEach(function (g) {
        var r = rows[+g.getAttribute("data-i")];
        var show = function (evt) {
          var a = r.a, c = a.content, v = r.v;
          tip.innerHTML = "<b>" + esc(v.name) + " " + esc(v.quant) + "</b><dl>" +
            "<dt>Mean speedup</dt><dd>" + fx(a.speedup) + "</dd>" +
            "<dt>Chat / code / math</dt><dd>" + c.chat.toFixed(2) + " / " + c.code.toFixed(2) + " / " + c.math.toFixed(2) + "</dd>" +
            "<dt>Plain decoding</dt><dd>" + a.baseline + " tok/s</dd>" +
            (a.tps ? "<dt>With mlx-dspark</dt><dd>" + a.tps + " tok/s</dd>" : "") +
            (a.accept ? "<dt>Accepted per round</dt><dd>" + a.accept.toFixed(2) + "</dd>" : "") +
            "<dt>Drafter</dt><dd>" + esc(armLabel(a)) + "</dd>" +
            (v.measured ? "<dt>Measured</dt><dd>" + esc(v.measured.when) + "</dd>" : "") + "</dl>";
          tip.hidden = false;
          var br = box.getBoundingClientRect(), gr = g.getBoundingClientRect();
          var left = evt && evt.clientX ? evt.clientX - br.left + 16 : Math.min(br.width - 300, labelW + 40);
          left = Math.max(0, Math.min(left, br.width - tip.offsetWidth - 4));
          var topPx = gr.bottom - br.top + 6;
          if (topPx + tip.offsetHeight > br.height) topPx = gr.top - br.top - tip.offsetHeight - 6;
          tip.style.left = left + "px"; tip.style.top = topPx + "px";
        };
        g.addEventListener("mousemove", show);
        g.addEventListener("focus", function () { show(null); });
        g.addEventListener("mouseleave", function () { tip.hidden = true; });
        g.addEventListener("blur", function () { tip.hidden = true; });
        var go = function () { location.href = root + "models/" + r.v.slug + "/"; };
        g.addEventListener("click", go);
        g.addEventListener("keydown", function (e) { if (e.key === "Enter") go(); });
      });
    }
    draw();
    var w = holder.clientWidth;
    if ("ResizeObserver" in window) new ResizeObserver(function () {
      if (Math.abs(holder.clientWidth - w) > 8) { w = holder.clientWidth; draw(); }
    }).observe(holder);
  }

  /* ================================================================ cost model */
  var CURVES = {
    // relative cost of verifying w rows, in single-token decode steps (illustrative shapes)
    flat: function (w) { return w <= 8 ? 1 + 0.02 * (w - 1) : 1.55 + 0.02 * (w - 9); },
    rising: function (w) { return w <= 4 ? 1 + 0.12 * (w - 1) : (w <= 8 ? 1.4 : 2.4); },
    steep: function (w) { return 1 + 0.22 * (w - 1); }
  };
  function initCostModel(box) {
    var pIn = box.querySelector('[name="cm-p"]'), dIn = box.querySelector('[name="cm-d"]');
    var pOut = box.querySelector("[data-cm-p]"), dOut = box.querySelector("[data-cm-d]");
    var chart = box.querySelector(".cm-chart"), read = box.querySelector(".cm-readout");
    function draw() {
      var p = +pIn.value / 100, d = +dIn.value / 100;
      var curve = CURVES[box.querySelector('input[name="cm-curve"]:checked').value];
      pOut.textContent = Math.round(p * 100) + "%";
      dOut.textContent = Math.round(d * 100) + "% of a step";
      var caps = [], bestC = 1, bestV = 0;
      for (var C = 1; C <= 12; C++) {
        var E = (1 - Math.pow(p, C + 1)) / (1 - p);
        var v = E / (d + curve(C + 1));
        caps.push([C, v, E]);
        if (v > bestV) { bestV = v; bestC = C; }
      }
      var W = Math.max(300, chart.clientWidth), H = 190, padL = 30, padB = 26, top = 18;
      var max = Math.max(4, Math.ceil(bestV));
      var bw = (W - padL) / caps.length;
      var Y = function (v) { return top + (H - top - padB) * (1 - v / max); };
      var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Predicted speedup for draft caps 1 to 12; best is cap ' + bestC + '">';
      for (var g = 1; g <= max; g++) {
        s += '<line class="' + (g === 1 ? "dp-one" : "dp-grid") + '" x1="' + padL + '" x2="' + W + '" y1="' + Y(g) + '" y2="' + Y(g) + '"/>';
        s += '<text class="cm-label" x="' + (padL - 6) + '" y="' + (Y(g) + 4) + '" text-anchor="end">' + g + "×</text>";
      }
      caps.forEach(function (c, i) {
        var x = padL + i * bw + bw * 0.18, w = bw * 0.64;
        var y = Y(Math.max(0, c[1]));
        s += '<rect class="cm-bar' + (c[0] === bestC ? " is-best" : "") + '" x="' + x + '" y="' + y + '" width="' + w + '" height="' + (H - padB - y) + '" rx="3"/>';
        s += '<text class="cm-label" x="' + (x + w / 2) + '" y="' + (H - 8) + '" text-anchor="middle">' + c[0] + "</text>";
        if (c[0] === bestC) s += '<text class="cm-val" x="' + (x + w / 2) + '" y="' + (y - 6) + '" text-anchor="middle">' + c[1].toFixed(2) + "×</text>";
      });
      s += "</svg>";
      chart.innerHTML = s;
      read.innerHTML = "Best cap here: <strong>" + bestC + "</strong>, about <strong>" + bestV.toFixed(2) +
        "×</strong> (accepting " + caps[bestC - 1][2].toFixed(1) + " tokens per round). Change the curve and watch the best cap move: that is why mlx-dspark measures it on your Mac instead of shipping a number.";
    }
    box.addEventListener("input", draw);
    draw();
    if ("ResizeObserver" in window) new ResizeObserver(draw).observe(chart);
  }

  /* ================================================================ models table filter */
  function initModelFilter(box) {
    var table = document.querySelector(box.getAttribute("data-filter-table"));
    if (!table) return;
    box.addEventListener("change", function () {
      var want = box.querySelector("input:checked").value;
      table.querySelectorAll("tbody tr").forEach(function (tr) {
        var kinds = (tr.getAttribute("data-kind") || "").split(" ");
        tr.hidden = want !== "all" && kinds.indexOf(want) < 0;
      });
    });
  }

  document.querySelectorAll("[data-race]").forEach(initRace);
  document.querySelectorAll("[data-picker]").forEach(initPicker);
  document.querySelectorAll("[data-dotplot]").forEach(initDotplot);
  document.querySelectorAll("[data-costmodel]").forEach(initCostModel);
  document.querySelectorAll("[data-model-filter]").forEach(initModelFilter);
})();
