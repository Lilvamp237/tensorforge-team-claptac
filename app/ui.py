"""Presentation helpers for the Streamlit app: one stylesheet and a few HTML components.

Everything user-supplied goes through html.escape before it is placed in markup.
"""

from html import escape

import streamlit as st

RISK = {
    "high": {"label": "High risk", "fg": "#B42318", "bg": "#FEF3F2", "border": "#FECDCA",
             "advice": "Do not pay and do not send documents. Check the agency with SLBFE (hotline 1989) first."},
    "medium": {"label": "Some warning signs", "fg": "#B54708", "bg": "#FFFAEB", "border": "#FEDF89",
               "advice": "Verify the agency and the fee with SLBFE (1989) before you pay anything."},
    "low": {"label": "Low risk", "fg": "#067647", "bg": "#ECFDF3", "border": "#ABEFC6",
            "advice": "No strong warning signs found. Still, pay only official fees and always get a receipt."},
}
STATUS_ICON = {"consistent": ("✓", "#067647", "#ECFDF3"), "contradicted": ("✕", "#B42318", "#FEF3F2"),
               "unverifiable": ("?", "#475467", "#F2F4F7")}

CSS = """
<style>
:root {
  --ink: #101828; --muted: #475467; --faint: #667085; --line: #E4E7EC; --surface: #FFFFFF;
  --canvas: #F6F8FB; --brand: #0B3B60; --brand-2: #0F766E; --brand-soft: #E6F4F1;
}
html, body, [class*="css"], .stApp { font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif; }
.stApp { background: var(--canvas); }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }
/* Keep Streamlit's toolbar visible: it contains the control to reopen a collapsed sidebar. */
#MainMenu, footer { visibility: hidden; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: 10px 18px; font-weight: 600; color: var(--muted); border-radius: 8px 8px 0 0; }
.stTabs [aria-selected="true"] { color: var(--brand) !important; background: var(--surface); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--brand-2); height: 3px; }

/* bordered containers become cards */
[data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface); border-radius: 14px !important;
  border: 1px solid var(--line) !important; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.stButton > button, .stFormSubmitButton > button { border-radius: 10px; font-weight: 600; }
.stButton > button[kind="primary"] { background: var(--brand); border-color: var(--brand); }
.stButton > button[kind="primary"]:hover { background: #082C48; border-color: #082C48; }
[data-testid="stSidebar"] { background: #0B2A45; min-width: 250px !important; max-width: 250px !important; }
[data-testid="stSidebar"] * { color: #E4ECF4; }
[data-testid="stSidebar"] .sp-side-muted { color: #9FB3C8; }

/* components */
.sp-hero { display: flex; align-items: center; justify-content: space-between; gap: 24px; flex-wrap: wrap;
  background: linear-gradient(120deg, #0B3B60 0%, #0F5E73 55%, #0F766E 100%); color: #fff;
  border-radius: 18px; padding: 22px 28px; margin-bottom: 14px; }
.sp-hero h1 { color: #fff; font-size: 1.9rem; margin: 0; padding: 0; letter-spacing: -.01em; }
.sp-hero p { color: #D6E6F2; margin: 4px 0 0; font-size: .98rem; }
.sp-brand { display: flex; align-items: center; gap: 14px; }
.sp-logo { width: 46px; height: 46px; border-radius: 12px; background: rgba(255,255,255,.14);
  display: grid; place-items: center; font-size: 26px; }
.sp-hero-stats { display: flex; gap: 10px; flex-wrap: wrap; }
.sp-hero-stat { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18); border-radius: 12px;
  padding: 8px 14px; min-width: 110px; }
.sp-hero-stat b { display: block; font-size: 1.15rem; color: #fff; }
.sp-hero-stat span { font-size: .78rem; color: #CFE0EE; }

.sp-section { font-size: .78rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  color: var(--faint); margin: 2px 0 8px; }
.sp-risk { border-radius: 14px; padding: 18px 20px; border: 1px solid; margin-bottom: 12px; }
.sp-risk-top { display: flex; align-items: baseline; justify-content: space-between; gap: 8px 12px; flex-wrap: wrap; }
.sp-risk-label { font-size: 1.35rem; font-weight: 750; white-space: nowrap; }
.sp-risk-score { font-size: 2.1rem; font-weight: 800; line-height: 1; white-space: nowrap; }
.sp-risk-score small { font-size: .8rem; font-weight: 600; opacity: .8; margin-left: 4px; }
.sp-meter { height: 8px; border-radius: 99px; background: rgba(16,24,40,.08); margin: 12px 0 10px; overflow: hidden; }
.sp-meter > div { height: 100%; border-radius: 99px; }
.sp-risk-advice { color: var(--ink); font-size: .97rem; }

.sp-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 12px; }
.sp-chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 99px;
  font-size: .82rem; font-weight: 600; background: #EEF2F6; color: #344054; border: 1px solid var(--line); }
.sp-chip.brand { background: var(--brand-soft); color: #0F5E57; border-color: #B8E0D9; }

.sp-list { list-style: none; padding: 0; margin: 0; }
.sp-list li { display: flex; gap: 10px; align-items: flex-start; padding: 9px 0; border-bottom: 1px solid #F2F4F7; }
.sp-list li:last-child { border-bottom: 0; }
.sp-dot { flex: 0 0 22px; height: 22px; border-radius: 99px; display: grid; place-items: center;
  font-size: .78rem; font-weight: 800; }
.sp-item-title { color: var(--ink); font-size: .93rem; font-weight: 550; }
.sp-item-src { color: var(--faint); font-size: .78rem; margin-top: 2px; word-break: break-word; }
.sp-item-src a { color: var(--brand-2); }
.sp-empty { color: var(--faint); font-size: .9rem; padding: 6px 0; }

.sp-kpi { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; min-height: 112px; }
.sp-kpi span { color: var(--faint); font-size: .8rem; font-weight: 600; }
.sp-kpi b { display: block; font-size: 1.7rem; color: var(--ink); margin-top: 2px; }
.sp-kpi small { color: var(--muted); font-size: .78rem; }

.sp-banner { border-radius: 14px; padding: 14px 18px; border: 1px solid; margin-bottom: 14px;
  display: flex; gap: 14px; align-items: center; }
.sp-banner .sp-pulse { width: 12px; height: 12px; border-radius: 99px; flex: 0 0 12px; }
.sp-banner b { font-size: 1.02rem; }
.sp-banner div { color: var(--ink); font-size: .92rem; }

.sp-signal { padding: 8px 0; }
.sp-signal-top { display: flex; justify-content: space-between; font-size: .88rem; color: var(--ink); }
.sp-signal-top span:last-child { color: var(--faint); font-variant-numeric: tabular-nums; }
.sp-bar { position: relative; height: 8px; background: #EEF2F6; border-radius: 99px; margin-top: 6px; overflow: hidden; }
.sp-bar > div { height: 100%; border-radius: 99px; }

.sp-quote { background: #F9FAFB; border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px;
  color: #344054; font-size: .9rem; white-space: pre-wrap; max-height: 180px; overflow: auto; }
.sp-pill { display: inline-block; padding: 2px 10px; border-radius: 99px; font-size: .78rem; font-weight: 700; }
.sp-side-card { background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12); border-radius: 12px;
  padding: 12px 14px; margin-bottom: 12px; }
.sp-side-card .big { font-size: 1.6rem; font-weight: 800; color: #fff; }
.sp-steps { padding-left: 18px; margin: 0; }
.sp-steps li { margin-bottom: 6px; font-size: .88rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def section(title: str) -> None:
    html(f'<div class="sp-section">{escape(title)}</div>')


def hero(stats: list[tuple[str, str]]) -> None:
    cells = "".join(f'<div class="sp-hero-stat"><b>{escape(v)}</b><span>{escape(k)}</span></div>' for k, v in stats)
    html(f"""<div class="sp-hero"><div class="sp-brand"><div class="sp-logo">🛡️</div><div>
<h1>SafePassage LK</h1><p>Check a job, study or visa offer before you pay. English · සිංහල · தமிழ் · Singlish</p>
</div></div><div class="sp-hero-stats">{cells}</div></div>""")


def risk_card(label: str, probability: float) -> None:
    r = RISK[label]
    pct = round(probability * 100)
    shown = ">99%" if probability > 0.99 else "<1%" if probability < 0.01 else f"{pct}%"  # never claim certainty
    html(f"""<div class="sp-risk" style="background:{r['bg']};border-color:{r['border']}">
<div class="sp-risk-top"><div class="sp-risk-label" style="color:{r['fg']}">{r['label']}</div>
<div class="sp-risk-score" style="color:{r['fg']}">{shown}<small>scam likelihood</small></div></div>
<div class="sp-meter"><div style="width:{pct}%;background:{r['fg']}"></div></div>
<div class="sp-risk-advice">{escape(r['advice'])}</div></div>""")


def chips(items: list[str], brand: bool = False) -> None:
    if items:
        cls = "sp-chip brand" if brand else "sp-chip"
        html('<div class="sp-chips">' + "".join(f'<span class="{cls}">{escape(i)}</span>' for i in items) + "</div>")


def _source(src: str | None) -> str:
    if not src:
        return ""
    words = []
    for w in src.split():
        bare = w.rstrip(").,")
        if bare.startswith("http"):
            words.append(f'<a href="{escape(bare)}" target="_blank">source</a>' + escape(w[len(bare):]))
        else:
            words.append(escape(w))
    return f'<div class="sp-item-src">{" ".join(words)}</div>'


def item_list(rows: list[tuple[str, str, str, str | None]], empty: str) -> None:
    """rows: (icon, colour, background, title) + optional source as the 5th element via tuple length."""
    if not rows:
        html(f'<div class="sp-empty">{escape(empty)}</div>')
        return
    out = []
    for icon, fg, bg, title, *rest in rows:
        src = rest[0] if rest else None
        out.append(f'<li><div class="sp-dot" style="color:{fg};background:{bg}">{icon}</div>'
                   f'<div><div class="sp-item-title">{escape(title)}</div>{_source(src)}</div></li>')
    html('<ul class="sp-list">' + "".join(out) + "</ul>")


def kpi(label: str, value: str, note: str = "") -> str:
    return (f'<div class="sp-kpi"><span>{escape(label)}</span><b>{escape(value)}</b>'
            f'<small>{escape(note)}</small></div>')


def banner(ok: bool, title: str, body: str) -> None:
    fg, bg, border = ("#067647", "#ECFDF3", "#ABEFC6") if ok else ("#B42318", "#FEF3F2", "#FECDCA")
    html(f'<div class="sp-banner" style="background:{bg};border-color:{border}">'
         f'<div class="sp-pulse" style="background:{fg}"></div>'
         f'<div><b style="color:{fg}">{escape(title)}</b><div>{escape(body)}</div></div></div>')


def signal_bar(name: str, value: float, threshold: float) -> str:
    ratio = 0.0 if threshold <= 0 else max(0.0, value) / threshold
    width = min(100, round(ratio * 50))  # the threshold sits at the middle of the bar
    colour = "#B42318" if ratio > 1 else "#B54708" if ratio > 0.7 else "#0F766E"
    return (f'<div class="sp-signal"><div class="sp-signal-top"><span>{escape(name)}</span>'
            f'<span>{value:.3f} / alert above {threshold:.3f}</span></div>'
            f'<div class="sp-bar"><div style="width:{width}%;background:{colour}"></div>'
            f'<div style="position:absolute;left:50%;top:0;bottom:0;width:2px;background:#98A2B3"></div></div></div>')


def quote(text: str) -> None:
    html(f'<div class="sp-quote">{escape(text)}</div>')


def pill(text: str, label: str) -> str:
    r = RISK.get(label, {"fg": "#344054", "bg": "#F2F4F7"})
    return f'<span class="sp-pill" style="color:{r["fg"]};background:{r["bg"]}">{escape(text)}</span>'
