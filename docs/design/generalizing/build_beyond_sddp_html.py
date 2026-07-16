#!/usr/bin/env python3
"""Convert the Beyond-SDDP roadmap Markdown into a self-contained, styled HTML
artifact (engineering-monograph identity: copper accent on graphite neutrals)."""
import re, html

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_HERE, "beyond-sddp-generalization.md")
OUT = os.path.join(_HERE, "beyond-sddp-generalization.html")

# Private-use-area sentinels delimit code-span placeholders so the restore step
# can never collide with real digits in the prose.
SL, SR = "", ""

SEV = {"High", "Medium", "Low", "High–Med", "Low–Med", "Medium–High", "Low–Medium"}
SEV_CLASS = {"High": "high", "Medium": "med", "Low": "low", "High–Med": "high",
             "Medium–High": "high", "Low–Med": "med", "Low–Medium": "med"}

slug_seen = {}
def slugify(text):
    s = re.sub(r"[`_*]", "", text)
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    if s in slug_seen:
        slug_seen[s] += 1
        s = f"{s}-{slug_seen[s]}"
    else:
        slug_seen[s] = 1
    return s

def inline(text):
    codes = []
    def stash(m):
        codes.append(html.escape(m.group(1)))
        return SL + str(len(codes) - 1) + SR
    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
                  lambda m: f'<a href="{m.group(2)}" target="_blank" rel="noopener">{m.group(1)}</a>', text)
    text = re.sub(r"&lt;(https?://[^&]+)&gt;",
                  lambda m: f'<a href="{m.group(1)}" target="_blank" rel="noopener">{m.group(1)}</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![A-Za-z0-9_])_([^_\s][^_]*?)_(?![A-Za-z0-9_])", r"<em>\1</em>", text)
    text = re.sub(SL + r"(\d+)" + SR,
                  lambda m: f"<code>{codes[int(m.group(1))]}</code>", text)
    return text

def cell_html(raw):
    raw = raw.strip()
    if raw in SEV:
        return f'<span class="sev sev-{SEV_CLASS[raw]}">{html.escape(raw)}</span>'
    return inline(raw)

lines = open(SRC, encoding="utf-8").read().split("\n")
out = []
toc = []
i = 0
n = len(lines)

def is_table_sep(s):
    return bool(re.match(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$", s)) and "-" in s

while i < n:
    line = lines[i]
    if line.strip() == "":
        i += 1; continue
    m = re.match(r"^```(\w*)\s*$", line)
    if m:
        lang = m.group(1); i += 1; buf = []
        while i < n and not lines[i].startswith("```"):
            buf.append(lines[i]); i += 1
        i += 1
        content = "\n".join(buf)
        if lang == "mermaid":
            out.append(f'<figure class="diagram"><pre class="mermaid">{html.escape(content)}</pre></figure>')
        else:
            out.append(f'<figure class="code"><pre><code>{html.escape(content)}</code></pre></figure>')
        continue
    if re.match(r"^---+\s*$", line):
        out.append('<hr>'); i += 1; continue
    m = re.match(r"^(#{1,3})\s+(.*)$", line)
    if m:
        level = len(m.group(1)); txt = m.group(2).strip()
        htmltxt = inline(txt)
        if level == 1:
            out.append(f'<h1 class="doc-title">{htmltxt}</h1>')
        else:
            sid = slugify(txt)
            toc.append((level, sid, txt))
            out.append(f'<h{level} id="{sid}">{htmltxt}</h{level}>')
        i += 1; continue
    if "|" in line and i + 1 < n and is_table_sep(lines[i + 1]):
        header = [c.strip() for c in line.strip().strip("|").split("|")]
        i += 2
        rows = []
        while i < n and "|" in lines[i] and lines[i].strip():
            rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
            i += 1
        th = "".join(f"<th>{inline(c)}</th>" for c in header)
        trs = []
        for r in rows:
            tds = "".join(f"<td>{cell_html(c)}</td>" for c in r)
            trs.append(f"<tr>{tds}</tr>")
        out.append(f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead>'
                   f'<tbody>{"".join(trs)}</tbody></table></div>')
        continue
    m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
    if m:
        ordered = bool(re.match(r"\d+\.", m.group(2)))
        items = []
        while i < n:
            lm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
            if lm:
                items.append(lm.group(3)); i += 1
            elif lines[i].strip() != "" and re.match(r"^\s+\S", lines[i]) and not lines[i].startswith("```"):
                items[-1] += " " + lines[i].strip(); i += 1
            else:
                break
        tag = "ol" if ordered else "ul"
        lis = "".join(f"<li>{inline(it)}</li>" for it in items)
        out.append(f"<{tag}>{lis}</{tag}>")
        continue
    buf = [line]; i += 1
    while (i < n and lines[i].strip() != ""
           and not re.match(r"^(#{1,3}\s|```|---+\s*$|\s*([-*]|\d+\.)\s)", lines[i])
           and "|" not in lines[i]):
        buf.append(lines[i]); i += 1
    para = " ".join(x.strip() for x in buf)
    cls = ""
    if para.startswith("**Decision:**"):
        cls = ' class="decision"'
    elif re.match(r"^\*\*D\d+ —", para):
        cls = ' class="fork"'
    out.append(f"<p{cls}>{inline(para)}</p>")

body = "\n".join(out)

toc_html = ['<div class="toc-title">Contents</div>', '<ul class="toc-l2">']
open_sub = False
for level, sid, txt in toc:
    label = re.sub(r"[`*]", "", txt)
    label = re.sub(r"_\((.*?)\)_", "", label).strip()
    label = re.sub(r"\s+", " ", label)
    label = html.escape(label)
    if level == 2:
        if open_sub:
            toc_html.append("</ul></li>"); open_sub = False
        toc_html.append(f'<li><a href="#{sid}" class="t2">{label}</a>')
        toc_html.append('<ul class="toc-l3">'); open_sub = True
    else:
        toc_html.append(f'<li><a href="#{sid}" class="t3">{label}</a></li>')
if open_sub:
    toc_html.append("</ul></li>")
toc_html.append("</ul>")
toc_nav = "\n".join(toc_html)

CSS = r"""
<style>
:root{
  --bg:#f5f5f2; --surface:#ffffff; --surface-2:#fbfbf9;
  --ink:#1c1b19; --ink-soft:#403d38; --muted:#6b6862; --faint:#8f8b83;
  --line:#e4e2dc; --line-soft:#eeece7;
  --accent:#a65e2e; --accent-strong:#8f4d24; --accent-soft:#f2e7dc; --accent-line:#d9b892;
  --good:#3f7a56; --warn:#b07a2e; --bad:#b0473b;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,"Times New Roman",serif;
  --sans:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,system-ui,sans-serif;
  --mono:ui-monospace,"SF Mono","JetBrains Mono","Cascadia Code",Menlo,Consolas,"Liberation Mono",monospace;
  --maxw:44rem;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#14161a; --surface:#1b1e23; --surface-2:#181b20;
    --ink:#e8e4dd; --ink-soft:#cdc8bf; --muted:#9a968e; --faint:#726e66;
    --line:#2a2e34; --line-soft:#23272c;
    --accent:#d08a4f; --accent-strong:#e0a06a; --accent-soft:#2a2118; --accent-line:#5c4630;
    --good:#6fae86; --warn:#d7a655; --bad:#d97a6d;
  }
}
:root[data-theme="light"]{
  --bg:#f5f5f2; --surface:#ffffff; --surface-2:#fbfbf9;
  --ink:#1c1b19; --ink-soft:#403d38; --muted:#6b6862; --faint:#8f8b83;
  --line:#e4e2dc; --line-soft:#eeece7;
  --accent:#a65e2e; --accent-strong:#8f4d24; --accent-soft:#f2e7dc; --accent-line:#d9b892;
  --good:#3f7a56; --warn:#b07a2e; --bad:#b0473b;
}
:root[data-theme="dark"]{
  --bg:#14161a; --surface:#1b1e23; --surface-2:#181b20;
  --ink:#e8e4dd; --ink-soft:#cdc8bf; --muted:#9a968e; --faint:#726e66;
  --line:#2a2e34; --line-soft:#23272c;
  --accent:#d08a4f; --accent-strong:#e0a06a; --accent-soft:#2a2118; --accent-line:#5c4630;
  --good:#6fae86; --warn:#d7a655; --bad:#d97a6d;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:17px;line-height:1.66;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline;text-underline-offset:2px}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px;border-radius:2px}

.shell{display:grid;grid-template-columns:16.5rem minmax(0,1fr);gap:0;max-width:74rem;margin:0 auto}
.rail{position:sticky;top:0;align-self:start;height:100vh;overflow-y:auto;padding:2rem 1.25rem 3rem 1.5rem;
  border-right:1px solid var(--line);background:var(--surface-2)}
.rail .brand{font-family:var(--mono);font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);font-weight:600;margin-bottom:.15rem}
.rail .brand-sub{font-family:var(--mono);font-size:.66rem;letter-spacing:.08em;color:var(--faint);margin-bottom:1.4rem}
.toc-title{font-family:var(--mono);font-size:.66rem;letter-spacing:.16em;text-transform:uppercase;
  color:var(--faint);margin:0 0 .6rem}
.rail ul{list-style:none;margin:0;padding:0}
.toc-l2>li{margin:0 0 .15rem}
.toc-l2>li>a.t2{display:block;font-weight:600;font-size:.82rem;color:var(--ink-soft);padding:.2rem 0;line-height:1.35}
.toc-l3{margin:.1rem 0 .55rem .1rem;border-left:1px solid var(--line)}
.toc-l3>li>a.t3{display:block;font-size:.76rem;color:var(--muted);padding:.14rem 0 .14rem .7rem;line-height:1.3}
.rail a.active{color:var(--accent)}
.toc-l3>li>a.active{box-shadow:inset 2px 0 0 var(--accent);margin-left:-1px}

main{padding:0 clamp(1.1rem,4vw,3.5rem) 6rem}
.wrap{max-width:var(--maxw);margin:0 auto}
.hero{max-width:var(--maxw);margin:0 auto;padding:3.2rem 0 1.6rem;border-bottom:1px solid var(--line)}
.eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);font-weight:600}
.eyebrow::before{content:"";display:inline-block;width:1.6rem;height:2px;background:var(--accent);vertical-align:middle;margin-right:.6rem;transform:translateY(-2px)}
h1.doc-title{font-family:var(--serif);font-weight:600;font-size:clamp(2rem,5vw,2.9rem);line-height:1.08;
  letter-spacing:-.01em;margin:1rem 0 0;text-wrap:balance;color:var(--ink)}
.lede{font-family:var(--serif);font-size:1.18rem;line-height:1.5;color:var(--ink-soft);margin:1.1rem 0 0;max-width:38rem;text-wrap:pretty}
.meta{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1.5rem}
.chip{font-family:var(--mono);font-size:.7rem;letter-spacing:.03em;color:var(--muted);
  border:1px solid var(--line);border-radius:999px;padding:.28rem .7rem;background:var(--surface)}
.chip b{color:var(--accent);font-weight:600}

main h2{font-family:var(--serif);font-weight:600;font-size:1.72rem;line-height:1.15;letter-spacing:-.01em;
  margin:3.4rem 0 .2rem;padding-top:1.6rem;border-top:1px solid var(--line);text-wrap:balance;color:var(--ink)}
main h2:first-of-type{border-top:none}
main h3{font-family:var(--serif);font-weight:600;font-size:1.24rem;line-height:1.25;
  margin:2.3rem 0 .1rem;color:var(--ink);text-wrap:balance}
main h3::before{content:"\A7";color:var(--accent-line);font-family:var(--mono);font-size:.9em;margin-right:.45rem}
p{margin:.95rem 0;max-width:var(--maxw)}
strong{font-weight:650;color:var(--ink)}
em{font-style:italic}
code{font-family:var(--mono);font-size:.85em;background:var(--surface);border:1px solid var(--line-soft);
  border-radius:4px;padding:.08em .35em;color:var(--accent-strong);word-break:break-word}
hr{border:none;height:0;margin:2.2rem 0 0}
ul,ol{margin:.9rem 0;padding-left:1.3rem;max-width:var(--maxw)}
li{margin:.42rem 0;padding-left:.2rem}
li::marker{color:var(--accent);font-variant-numeric:tabular-nums}
ol>li::marker{font-family:var(--mono);font-size:.85em}

p.decision{background:var(--accent-soft);border-left:3px solid var(--accent);border-radius:0 6px 6px 0;
  padding:.75rem 1rem;margin:1.2rem 0}
p.decision strong:first-child{color:var(--accent-strong);font-family:var(--mono);font-size:.78em;
  letter-spacing:.06em;text-transform:uppercase}
p.fork{border-left:3px solid var(--line);padding:.15rem 0 .15rem 1rem;margin:1.5rem 0}
p.fork strong:first-child{color:var(--accent-strong)}

.table-wrap{overflow-x:auto;margin:1.5rem 0;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.83rem;line-height:1.45}
thead th{background:var(--surface-2);text-align:left;font-family:var(--sans);font-weight:650;color:var(--ink);
  padding:.6rem .8rem;border-bottom:2px solid var(--accent-line);white-space:nowrap;vertical-align:bottom}
tbody td{padding:.55rem .8rem;border-bottom:1px solid var(--line-soft);vertical-align:top;color:var(--ink-soft)}
tbody tr:last-child td{border-bottom:none}
tbody tr:nth-child(even){background:var(--surface-2)}
td code,th code{background:transparent;border:none;padding:0;color:var(--accent-strong)}
.sev{font-family:var(--mono);font-size:.72rem;font-weight:600;padding:.12rem .5rem;border-radius:999px;
  white-space:nowrap;display:inline-block}
.sev-high{color:var(--bad);background:color-mix(in srgb,var(--bad) 14%,transparent)}
.sev-med{color:var(--warn);background:color-mix(in srgb,var(--warn) 16%,transparent)}
.sev-low{color:var(--good);background:color-mix(in srgb,var(--good) 16%,transparent)}

figure{margin:1.6rem 0}
figure.diagram{background:#fbfbf9;border:1px solid var(--line);border-radius:8px;padding:1.25rem;overflow-x:auto;text-align:center}
figure.diagram .mermaid{margin:0 auto;min-width:0}
figure.code{background:var(--surface);border:1px solid var(--line);border-radius:8px;overflow-x:auto;border-left:3px solid var(--accent-line)}
figure.code pre{margin:0;padding:1rem 1.15rem}
figure.code code{background:transparent;border:none;padding:0;color:var(--ink);font-size:.82rem;line-height:1.55;white-space:pre}

@media (max-width:900px){
  .shell{grid-template-columns:1fr}
  .rail{position:static;height:auto;border-right:none;border-bottom:1px solid var(--line);
    padding:1rem 1.1rem;max-height:60vh}
  .rail.collapsed .toc-title,.rail.collapsed ul{display:none}
  .railtoggle{display:inline-flex}
}
.railtoggle{display:none;align-items:center;gap:.5rem;font-family:var(--mono);font-size:.72rem;
  letter-spacing:.08em;text-transform:uppercase;color:var(--accent);background:none;border:1px solid var(--line);
  border-radius:6px;padding:.4rem .7rem;cursor:pointer;margin-top:.4rem}
html{scroll-behavior:smooth}
main h2,main h3{scroll-margin-top:1.2rem}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}*{transition:none!important}}
</style>
"""

HERO = """
<header class="hero">
  <div class="eyebrow">Cobre &middot; Architecture Roadmap</div>
  <h1 class="doc-title">Beyond SDDP</h1>
  <p class="lede">Architecting the Cobre ecosystem from a single-vertical SDDP solver into a general power-system optimization platform &mdash; data model, dispatch axis, crate borders, and a sequenced roadmap.</p>
  <div class="meta">
    <span class="chip"><b>Dated</b>&nbsp; 2026-07-15</span>
    <span class="chip"><b>Status</b>&nbsp; Research proposal / roadmap</span>
    <span class="chip"><b>Verticals</b>&nbsp; UC &middot; hydro routing &middot; OPF &middot; expansion</span>
    <span class="chip"><b>Method</b>&nbsp; multi-source, adversarially verified</span>
  </div>
</header>
"""

JS = r"""
<script>
(function(){
  var rail=document.querySelector('.rail');
  var toggle=document.querySelector('.railtoggle');
  if(toggle){toggle.addEventListener('click',function(){rail.classList.toggle('collapsed');});}
  var links=[].slice.call(document.querySelectorAll('.rail a'));
  var map={};links.forEach(function(a){map[a.getAttribute('href').slice(1)]=a;});
  var heads=[].slice.call(document.querySelectorAll('main h2, main h3'));
  var obs=new IntersectionObserver(function(entries){
    entries.forEach(function(e){
      if(e.isIntersecting){
        links.forEach(function(a){a.classList.remove('active');});
        var a=map[e.target.id];
        if(a){a.classList.add('active');}
      }
    });
  },{rootMargin:"-8% 0px -80% 0px",threshold:0});
  heads.forEach(function(h){obs.observe(h);});
})();
</script>
"""

# Mermaid renders client-side. Preferred path: mermaid.min.js beside this
# script is INLINED, so the HTML is fully self-contained and the diagrams
# render offline (file://, no CDN). The file is deliberately NOT tracked in git
# (see .gitignore) — when missing, it is auto-downloaded once from the pinned
# URL below; if that fails (offline build), the HTML falls back to a CDN ESM
# import. Every path calls mermaid.run() explicitly — startOnLoad:true races an
# async import and silently skips rendering when the page's load event fires
# before the module arrives, which is exactly the bug the CDN-only version had.
_MERMAID_PINNED_URL = "https://cdn.jsdelivr.net/npm/mermaid@11.16.0/dist/mermaid.min.js"
_MERMAID_RUN_JS = r"""
  const _mermaidGo = async () => {
    try {
      mermaid.initialize({ startOnLoad: false, theme: "neutral", securityLevel: "loose", flowchart: { htmlLabels: true } });
      await mermaid.run({ querySelector: ".mermaid" });
    } catch (e) { console.error("mermaid render failed:", e); /* source stays visible */ }
  };
  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", _mermaidGo); }
  else { _mermaidGo(); }
"""
_MERMAID_LOCAL = os.path.join(_HERE, "mermaid.min.js")
if not os.path.exists(_MERMAID_LOCAL):
    try:
        import sys
        import urllib.request

        print(f"mermaid.min.js missing — downloading {_MERMAID_PINNED_URL}", file=sys.stderr)
        urllib.request.urlretrieve(_MERMAID_PINNED_URL, _MERMAID_LOCAL)
    except Exception as exc:  # offline build: HTML falls back to the CDN loader
        print(f"mermaid download failed ({exc}); emitting CDN-loader HTML", file=sys.stderr)
if os.path.exists(_MERMAID_LOCAL):
    MERMAID = (
        "<script>"
        + open(_MERMAID_LOCAL, encoding="utf-8").read()
        + "</script>\n<script>"
        + _MERMAID_RUN_JS
        + "</script>"
    )
else:
    MERMAID = (
        '<script type="module">\n'
        "try {\n"
        '  const mermaid = (await import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs")).default;\n'
        + _MERMAID_RUN_JS
        + "\n} catch (e) { /* offline or CDN blocked: leave the mermaid source visible */ }\n"
        "</script>"
    )

body_html = (
    '<div class="shell">'
    + '<aside class="rail">'
    + '<div class="brand">Cobre</div><div class="brand-sub">power-system optimization &middot; Rust</div>'
    + '<button class="railtoggle">Contents ▾</button>'
    + toc_nav
    + "</aside>"
    + '<main>' + HERO + '<div class="wrap">' + body + "</div></main>"
    + "</div>"
)

doc = (
    "<!doctype html>\n"
    '<html lang="en">\n<head>\n'
    '<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    "<title>Beyond SDDP — Cobre Architecture Roadmap</title>\n"
    + CSS
    + "</head>\n<body>\n"
    + body_html
    + JS
    + MERMAID
    + "\n</body>\n</html>\n"
)

open(OUT, "w", encoding="utf-8").write(doc)
print("wrote", OUT, len(doc.encode("utf-8")), "bytes;", len(toc), "toc entries")
