/* mlx-dspark docs: site-wide behaviour (theme, nav, copy, tabs, search, toc). No deps. */
(function () {
  "use strict";
  var root = document.body.getAttribute("data-root") || "";
  var docEl = document.documentElement;

  /* ---------------------------------------------------------------- theme */
  function effectiveTheme() {
    var t = docEl.dataset.theme;
    if (t === "light" || t === "dark") return t;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  document.querySelectorAll(".theme-toggle").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var next = effectiveTheme() === "dark" ? "light" : "dark";
      docEl.dataset.theme = next;
      try { localStorage.setItem("theme", next); } catch (e) {}
      document.dispatchEvent(new CustomEvent("themechange"));
    });
  });

  /* ---------------------------------------------------------------- mobile nav */
  var navBtn = document.querySelector(".nav-toggle");
  var sidenav = document.getElementById("sidenav");
  if (navBtn) {
    if (!sidenav) navBtn.addEventListener("click", function () { location.href = root + "start/install/"; });
    else navBtn.addEventListener("click", function () {
      var open = !document.body.classList.contains("nav-open");
      document.body.classList.toggle("nav-open", open);
      navBtn.setAttribute("aria-expanded", String(open));
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && document.body.classList.contains("nav-open")) {
        document.body.classList.remove("nav-open");
        navBtn.setAttribute("aria-expanded", "false");
        navBtn.focus();
      }
    });
  }

  /* ---------------------------------------------------------------- copy buttons */
  function copyText(text, btn) {
    var done = function () {
      btn.classList.add("is-done");
      var label = btn.querySelector("span") || btn;
      label.textContent = "Copied";
      setTimeout(function () { btn.classList.remove("is-done"); label.textContent = "Copy"; }, 1600);
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text); done(); });
    } else { fallbackCopy(text); done(); }
  }
  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e) {}
    document.body.removeChild(ta);
  }
  document.addEventListener("click", function (e) {
    var btn = e.target.closest(".copy");
    if (!btn) return;
    var block = btn.closest(".code");
    var src = block ? block.querySelector("pre code") : btn.parentElement.querySelector("code");
    if (!src) return;
    // drop full-line comments from shell blocks: they are for reading, not pasting
    var text = src.textContent.replace(/\n$/, "");
    if (block && block.getAttribute("data-lang") === "shell") {
      text = text.split("\n").filter(function (l) { return !/^\s*#/.test(l); }).join("\n").replace(/\n{3,}/g, "\n\n").trim();
    }
    copyText(text, btn);
  });

  /* ---------------------------------------------------------------- tabs */
  function makeTabs(container) {
    var panes = Array.prototype.filter.call(container.children, function (c) { return c.classList.contains("tab"); });
    if (!panes.length) return;
    var list = document.createElement("div");
    list.className = "tablist"; list.setAttribute("role", "tablist");
    var group = container.getAttribute("data-group");
    var buttons = panes.map(function (pane, i) {
      var id = "tab-" + Math.random().toString(36).slice(2, 8);
      var b = document.createElement("button");
      b.type = "button"; b.setAttribute("role", "tab"); b.id = id;
      b.textContent = pane.getAttribute("data-label") || "Tab " + (i + 1);
      pane.setAttribute("role", "tabpanel"); pane.setAttribute("aria-labelledby", id);
      list.appendChild(b);
      return b;
    });
    function select(i, focus) {
      buttons.forEach(function (b, j) {
        b.setAttribute("aria-selected", String(i === j));
        b.tabIndex = i === j ? 0 : -1;
        panes[j].hidden = i !== j;
      });
      if (focus) buttons[i].focus();
    }
    buttons.forEach(function (b, i) {
      b.addEventListener("click", function () {
        select(i);
        if (group) {
          try { localStorage.setItem("tab:" + group, b.textContent); } catch (e) {}
          document.querySelectorAll('.tabs[data-group="' + group + '"]').forEach(function (other) {
            if (other !== container && other._selectLabel) other._selectLabel(b.textContent);
          });
        }
      });
      b.addEventListener("keydown", function (e) {
        var k = e.key, n = buttons.length;
        if (k === "ArrowRight" || k === "ArrowLeft") {
          e.preventDefault();
          select((i + (k === "ArrowRight" ? 1 : n - 1)) % n, true);
        }
      });
    });
    container._selectLabel = function (label) {
      var idx = buttons.findIndex(function (b) { return b.textContent === label; });
      if (idx >= 0) select(idx);
    };
    container.insertBefore(list, container.firstChild);
    var start = 0;
    if (group) {
      try {
        var saved = localStorage.getItem("tab:" + group);
        var idx = buttons.findIndex(function (b) { return b.textContent === saved; });
        if (idx >= 0) start = idx;
      } catch (e) {}
    }
    select(start);
  }
  document.querySelectorAll(".tabs").forEach(makeTabs);

  /* the home page's install switcher */
  document.querySelectorAll("[data-install]").forEach(function (box) {
    var tabs = box.querySelectorAll("[data-cmd]");
    var cmd = box.querySelector("[data-install-cmd]");
    var note = box.querySelector("[data-install-note]");
    var defaultNote = note ? note.textContent : "";
    tabs.forEach(function (t, i) {
      t.addEventListener("click", function () {
        tabs.forEach(function (o) { o.setAttribute("aria-selected", String(o === t)); o.tabIndex = o === t ? 0 : -1; });
        cmd.textContent = t.getAttribute("data-cmd");
        if (note) note.textContent = t.getAttribute("data-note") || defaultNote;
      });
      t.addEventListener("keydown", function (e) {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        e.preventDefault();
        var n = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
        n.click(); n.focus();
      });
    });
  });

  /* model page quant switch */
  document.querySelectorAll(".variant-switch").forEach(function (sw) {
    var btns = sw.querySelectorAll("button");
    btns.forEach(function (b, i) {
      b.addEventListener("click", function () {
        btns.forEach(function (o) {
          var on = o === b;
          o.setAttribute("aria-selected", String(on)); o.tabIndex = on ? 0 : -1;
          document.getElementById(o.getAttribute("aria-controls")).hidden = !on;
        });
      });
      b.addEventListener("keydown", function (e) {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        e.preventDefault();
        var n = btns[(i + (e.key === "ArrowRight" ? 1 : btns.length - 1)) % btns.length];
        n.click(); n.focus();
      });
    });
    // deep link: #8-bit selects that variant
    var want = decodeURIComponent(location.hash.slice(1));
    btns.forEach(function (b) { if (want && b.firstChild.textContent.trim() === want) b.click(); });
  });

  /* ---------------------------------------------------------------- toc scroll-spy */
  var tocLinks = document.querySelectorAll(".toc a");
  if (tocLinks.length && "IntersectionObserver" in window) {
    var map = {};
    tocLinks.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
    var heads = Object.keys(map).map(function (id) { return document.getElementById(id); }).filter(Boolean);
    var visible = new Set();
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) visible.add(en.target.id); else visible.delete(en.target.id); });
      var current = null;
      for (var i = 0; i < heads.length; i++) {
        if (visible.has(heads[i].id)) { current = heads[i].id; break; }
      }
      if (!current) {
        // between headings: the last one above the fold
        for (var j = heads.length - 1; j >= 0; j--) {
          if (heads[j].getBoundingClientRect().top < 120) { current = heads[j].id; break; }
        }
      }
      tocLinks.forEach(function (a) { a.classList.toggle("is-active", a.getAttribute("href") === "#" + current); });
    }, { rootMargin: "-70px 0px -65% 0px" });
    heads.forEach(function (h) { io.observe(h); });
  }

  /* ---------------------------------------------------------------- search */
  var dialog = document.querySelector(".search");
  if (!dialog) return;
  var input = dialog.querySelector("input");
  var list = dialog.querySelector(".search-results");
  var empty = dialog.querySelector(".search-empty");
  var index = null, loading = null, sel = 0, lastFocus = null;

  function load() {
    if (index || loading) return loading;
    loading = fetch(root + "static/search-index.json").then(function (r) { return r.json(); })
      .then(function (d) {
        index = d.map(function (e) {
          e.lp = e.p.toLowerCase(); e.lh = (e.h || "").toLowerCase(); e.lx = e.x.toLowerCase();
          return e;
        });
      }).catch(function () { index = []; });
    return loading;
  }
  function openSearch() {
    lastFocus = document.activeElement;
    dialog.hidden = false;
    document.body.style.overflow = "hidden";
    input.value = ""; render([]); empty.hidden = true;
    input.focus();
    load();
  }
  function closeSearch() {
    dialog.hidden = true;
    document.body.style.overflow = "";
    if (lastFocus) lastFocus.focus();
  }
  document.querySelectorAll("[data-search-open]").forEach(function (b) { b.addEventListener("click", openSearch); });
  dialog.querySelectorAll("[data-search-close]").forEach(function (b) { b.addEventListener("click", closeSearch); });
  document.addEventListener("keydown", function (e) {
    var typing = /^(INPUT|TEXTAREA|SELECT)$/.test((e.target.tagName || "")) || e.target.isContentEditable;
    if ((e.key === "k" && (e.metaKey || e.ctrlKey)) || (e.key === "/" && !typing && dialog.hidden)) {
      e.preventDefault(); dialog.hidden ? openSearch() : closeSearch();
    } else if (e.key === "Escape" && !dialog.hidden) { closeSearch(); }
  });

  function esc(s) { return s.replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function highlight(text, terms) {
    var out = esc(text);
    terms.forEach(function (t) {
      if (t.length < 2) return;
      out = out.replace(new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig"), "<mark>$1</mark>");
    });
    return out;
  }
  function snippet(x, terms) {
    var lx = x.toLowerCase(), pos = -1;
    for (var i = 0; i < terms.length && pos < 0; i++) pos = lx.indexOf(terms[i]);
    var start = Math.max(0, pos - 40);
    return (start > 0 ? "…" : "") + x.slice(start, start + 160);
  }
  function search(q) {
    var terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length || !index) return [];
    var scored = [];
    index.forEach(function (e) {
      var s = 0;
      for (var i = 0; i < terms.length; i++) {
        var t = terms[i], hit = 0;
        if (e.lh.indexOf(t) >= 0) hit += e.lh.indexOf(t) === 0 ? 14 : 9;
        if (e.lp.indexOf(t) >= 0) hit += 6;
        if (e.lx.indexOf(t) >= 0) hit += 2 + Math.min(3, e.lx.split(t).length - 2);
        if (!hit) return;
        s += hit;
      }
      if (!e.h) s += 1;
      scored.push([s, e]);
    });
    scored.sort(function (a, b) { return b[0] - a[0]; });
    return scored.slice(0, 14).map(function (p) { return p[1]; });
  }
  function render(results, terms) {
    list.innerHTML = "";
    sel = 0;
    results.forEach(function (e, i) {
      var li = document.createElement("li");
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", String(i === 0));
      li.innerHTML = '<a href="' + root + e.u.replace(/^\//, "") + '">' +
        '<span class="sr-page">' + esc(e.p) + "</span>" +
        '<span class="sr-head">' + highlight(e.h || e.p, terms) + "</span>" +
        '<span class="sr-text">' + highlight(snippet(e.x, terms), terms) + "</span></a>";
      li.querySelector("a").addEventListener("click", closeSearchSoft);
      list.appendChild(li);
    });
  }
  function closeSearchSoft() { dialog.hidden = true; document.body.style.overflow = ""; }
  function move(d) {
    var items = list.querySelectorAll("li");
    if (!items.length) return;
    items[sel].setAttribute("aria-selected", "false");
    sel = (sel + d + items.length) % items.length;
    items[sel].setAttribute("aria-selected", "true");
    items[sel].scrollIntoView({ block: "nearest" });
  }
  input.addEventListener("input", function () {
    var q = input.value.trim();
    (load() || Promise.resolve()).then(function () {
      if (input.value.trim() !== q) return;
      var terms = q.toLowerCase().split(/\s+/).filter(Boolean);
      var res = search(q);
      render(res, terms);
      empty.hidden = !(q && !res.length);
    });
  });
  input.addEventListener("keydown", function (e) {
    if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
    else if (e.key === "Enter") {
      var a = list.querySelector('li[aria-selected="true"] a');
      if (a) { e.preventDefault(); closeSearchSoft(); location.href = a.href; }
    }
  });
})();
