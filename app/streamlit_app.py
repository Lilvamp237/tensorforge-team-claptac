"""Streamlit UI: scan page, monitoring dashboard and analyst console. Owner: Person B.

Run: PYTHONPATH=src streamlit run app/streamlit_app.py
"""

from functools import lru_cache

import altair as alt
import pandas as pd
import streamlit as st

import ui
from safepassage import config, reputation, retrain
from safepassage.fetch import UnsafeURL
from safepassage.media import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, MediaError, is_audio, is_image
from safepassage.monitoring import pending_reviews, read_alerts, read_log
from safepassage.response import check_drift, trigger_retrain
from safepassage.service import reload_model, scan_audio, scan_file, scan_image, scan_public_url, scan_text

st.set_page_config(page_title="SafePassage LK · Check before you pay", page_icon="🛡️", layout="wide")
ui.inject_css()

# Synthetic examples for a quick try (fictional agencies, demo licence numbers, placeholder phones).
EXAMPLES = {
    "Scam · EN": ("Dear candidate, security guard jobs in Russia, USD 2,500 monthly, no experience needed. "
                       "Pay the document verification charge of Rs. 200,000 to my personal account today and send "
                       "your passport copy on WhatsApp 07X XXX XXXX."),
    "Scam · සිං": ("හදිසියි!!! කටාර් සඳහා රියදුරු පුරප්පාඩු, මාසික වැටුප රු. 450,000. ලියාපදිංචි ගාස්තුව රු. 150,000 "
                     "අදම ගෙවන්න. 100% වීසා සහතිකයි. WhatsApp 07X XXX XXXX"),
    "Legit offer": ("Kelani Gateway Agency (SLBFE licence 9316). Welders for Qatar, salary QAR 1,400. Interviews "
                        "at our office on Monday. Only payment: SLBFE registration fee of Rs. 22,027, paid at the "
                        "SLBFE office with a receipt."),
}
LANGUAGES = {"en": "English", "si": "Sinhala", "ta": "Tamil", "singlish": "Singlish", "mixed": "Mixed language"}
SIGNALS = {"language_psi": "Language mix changed", "category_psi": "Offer-type mix changed",
           "confidence_drop": "Model confidence dropped", "novelty_share": "Unfamiliar offers rising"}


@lru_cache(maxsize=1)
def official_counts() -> tuple[int, int]:
    def rows(name):
        path = config.KB_DIR / name
        return sum(1 for _ in path.open(encoding="utf-8")) - 1 if path.exists() else 0
    return rows("slbfe_registry.csv"), rows("slbfe_job_orders.csv")


def use_example(name: str) -> None:
    st.session_state.mode = "Paste text"
    st.session_state.offer_text = EXAMPLES[name]


# ---------------------------------------------------------------- sidebar
drift_now = check_drift()
with st.sidebar:
    ui.html(f'<div style="display:flex;align-items:center;gap:8px;font-size:1.2rem;font-weight:800;margin:4px 0 14px">'
            f'{ui.SHIELD} SafePassage LK</div>')
    ui.html('<div class="sp-side-card"><div class="sp-side-muted">SLBFE 24-hour hotline</div>'
            '<div class="big">1989</div><div class="sp-side-muted">Check any agency or fee before you pay.</div></div>')
    ui.html("""<div class="sp-side-card"><b>How it works</b><ol class="sp-steps" style="margin-top:8px">
<li>Paste an offer, a link, a screenshot or a voice note.</li>
<li>The model scores it and lists the warning signs.</li>
<li>We check the agency and fee against official SLBFE records.</li>
<li>Unusual offers go to a human analyst.</li></ol></div>""")
    state = "Drift detected" if drift_now["drift"] else "Stable"
    ui.html(f'<div class="sp-side-card"><b>System status</b><div class="sp-side-muted" style="margin-top:6px">'
            f'Model: {config.MODEL_VERSION}<br>Monitoring: {state} · {drift_now["n"]} recent scans</div></div>')
    ui.html('<div class="sp-side-muted" style="font-size:.78rem">A risk assessment, not a fraud determination. '
            'Examples in this app are synthetic.</div>')

agencies, orders = official_counts()
ui.hero([("SLBFE agencies on record", f"{agencies:,}"), ("Approved job orders", f"{orders:,}"),
         ("Languages", "EN · SI · TA")])

scan_tab, agency_tab, dash_tab, analyst_tab = st.tabs([":material/search: Check an offer", ":material/apartment: Agency lookup",
                                                      ":material/monitoring: Monitoring",
                                                      ":material/admin_panel_settings: Analyst console"])

# ---------------------------------------------------------------- check an offer
with scan_tab:
    left, right = st.columns([5, 6], gap="large")
    with left:
        with st.container(border=True):
            ui.section("Your offer")
            st.session_state.setdefault("mode", "Paste text")
            mode = st.segmented_control("Input", ["Paste text", "Public link", "Upload file"], key="mode",
                                        label_visibility="collapsed") or "Paste text"
            text = url = upload = None
            hint = "Auto-detect"
            if mode == "Paste text":
                text = st.text_area("Offer text", key="offer_text", height=190, label_visibility="collapsed",
                                    placeholder="Paste the job, study or visa offer, or the message you received...")
            elif mode == "Public link":
                url = st.text_input("Public link", placeholder="https://...", label_visibility="collapsed")
                st.caption("We fetch only public pages, never run scripts, and never submit forms.")
            else:
                upload = st.file_uploader("Upload", label_visibility="collapsed",
                                          type=["pdf", "txt"] + [e.lstrip(".") for e in AUDIO_EXTENSIONS + IMAGE_EXTENSIONS])
                st.caption("Letters (PDF/TXT), screenshots (PNG/JPG) or voice notes (MP3/WAV/M4A/OGG), up to 10 MB. "
                           "Files are read in memory and never stored.")
                if upload and is_audio(upload.name):
                    hint = st.selectbox("Voice note language", ["Auto-detect", "English", "Tamil", "Sinhala"],
                                        help="Optional. Transcription is weaker for Sinhala.")
            check = st.button("Check this offer", type="primary", use_container_width=True)
            ui.html('<div class="sp-section" style="margin-top:14px">Try an example</div>')
            cols = st.columns(len(EXAMPLES))
            for col, name in zip(cols, EXAMPLES):
                col.button(name, key=f"ex:{name}", use_container_width=True, on_click=use_example, args=(name,))

        with st.expander("Report a suspicious website"):
            with st.form("report", border=False):
                report_url = st.text_input("Website or listing link")
                verdict = st.radio("This offer is", ["scam", "legit"], horizontal=True)
                comment = st.text_input("What happened? (no names or phone numbers)")
                rater_id = st.text_input("Your SafePassage ID (optional)",
                                         help="Lets reliable reporters count for more. Stored only as a salted hash.")
                if st.form_submit_button("Send report", use_container_width=True) and report_url:
                    r = reputation.add_report(report_url, verdict, comment, rater_id.strip() or None)
                    st.success(f"Thank you. {r['scam_reports']} scam report(s) on {r['domain']}. "
                               "Analysts review sites with repeated reports.")

    if check:
        st.session_state.extracted = None
        try:
            if upload and is_audio(upload.name):
                lang = {"English": "en", "Tamil": "ta", "Sinhala": "si"}.get(hint)
                with st.spinner("Listening to the voice note..."):
                    scan = scan_audio(upload.read(), upload.name, lang)
                st.session_state.extracted, st.session_state.result = ("What we heard", scan.text), scan.result
            elif upload and is_image(upload.name):
                with st.spinner("Reading the screenshot..."):
                    scan = scan_image(upload.read())
                st.session_state.extracted, st.session_state.result = ("What we read", scan.text), scan.result
            elif upload:
                st.session_state.result = scan_file(upload.read(), upload.name)
            elif url:
                with st.spinner("Fetching the page safely..."):
                    st.session_state.result = scan_public_url(url)
            elif text and text.strip():
                st.session_state.result = scan_text(text)
            else:
                st.warning("Add an offer first: paste text, a link, or upload a file.")
        except UnsafeURL as e:
            st.error(f"We can't check that link: {e}")
        except MediaError as e:
            st.error(f"We can't read that file: {e}")

    with right:
        result = st.session_state.get("result")
        if not result:
            with st.container(border=True):
                ui.section("Result")
                ui.html('<div style="padding:10px 2px 0;color:#667085"><div style="font-size:1.1rem;font-weight:700;'
                        'color:#101828;margin-bottom:4px">Check before you pay</div>Paste an offer, share a link, or '
                        'upload a screenshot or voice note.</div>')
                ui.tiles([("1", "Risk score", "A trained model scores the offer in English, Sinhala, Tamil or Singlish."),
                          ("2", "Warning signs", "Plain-language red flags: advance fees, guarantees, personal accounts."),
                          ("3", "Official records", "Agency licence and fee checked against SLBFE's public records.")])
        else:
            with st.container(border=True):
                ui.section("Result")
                ui.risk_card(result.risk.label, result.risk.probability)
                c = result.content
                facts = [c.type.replace("_", " ").capitalize() if c.type != "unknown" else None, c.claimed_country,
                         c.claimed_agency, LANGUAGES.get(c.language)]
                ui.chips([f for f in facts if f], brand=True)
                if result.needs_review:
                    ui.html('<div class="sp-chip" style="margin-bottom:10px">An analyst will double-check this one</div>')
                ui.next_steps(result.risk.label)
                extracted = st.session_state.get("extracted")
                if extracted:
                    ui.section(extracted[0])
                    ui.quote(extracted[1])
            a, b = st.columns(2)
            with a, st.container(border=True):
                ui.section("Warning signs")
                ui.item_list([("!", "#B54708", "#FFFAEB", flag) for flag in result.indicators], "No warning signs found.")
            with b, st.container(border=True):
                ui.section("Official records")
                v = result.verification
                agency = {"matched": ("✓", "#067647", "#ECFDF3", "Agency found in the SLBFE registry"),
                          "not_matched": ("✕", "#B42318", "#FEF3F2", "Agency not confirmed by SLBFE records"),
                          "unknown": ("?", "#475467", "#F2F4F7", "No agency or licence named")}[v.agency_status]
                rows = [agency] + [(*ui.STATUS_ICON[cl.status], cl.claim, cl.source) for cl in v.claims]
                ui.item_list(rows, "")
            with st.expander("Technical details (JSON)"):
                st.json(result.model_dump())


# ---------------------------------------------------------------- agency lookup (scraped SLBFE data)
@st.cache_data
def registry() -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = pd.read_csv(config.KB_DIR / "slbfe_registry.csv", dtype=str).fillna("")
    orders = pd.read_csv(config.KB_DIR / "slbfe_job_orders.csv", dtype=str).fillna("")
    return reg, orders


with agency_tab:
    reg, orders = registry()
    left, right = st.columns([5, 7], gap="large")
    with left, st.container(border=True):
        ui.section("Look up an agency")
        query = st.text_input("Agency name or SLBFE licence number", placeholder="e.g. 2963 or Katari",
                              label_visibility="collapsed")
        counts = reg["status"].value_counts()
        st.caption(f"{len(reg):,} agencies scraped from SLBFE's public lists: {counts.get('licensed', 0):,} licensed, "
                   f"{counts.get('suspended', 0):,} suspended, {counts.get('cancelled', 0):,} cancelled. "
                   f"{len(orders):,} approved job orders from the SLBFE job bank, with the maximum fee each allows.")
    with right:
        q = (query or "").strip()
        if not q:
            with st.container(border=True):
                ui.section("Why this matters")
                st.markdown("Scammers quote real-looking licence numbers, or a licence that was **cancelled**. "
                            "Every scan is checked against these records automatically; here you can look an agency "
                            "up yourself before you pay.")
        else:
            digits = q.lstrip("0")
            hits = reg[reg["licence_no"].str.lstrip("0") == digits] if q.isdigit() else                 reg[reg["name"].str.contains(q, case=False, regex=False)]
            if hits.empty:
                with st.container(border=True):
                    ui.banner(False, "Not found in SLBFE records",
                              "No agency matches. A real foreign-employment agency must hold an SLBFE licence. "
                              "Call 1989 before paying anyone.")
            for _, a in hits.head(5).iterrows():
                mine = orders[orders["licence_no"].str.lstrip("0") == a["licence_no"].lstrip("0")]
                with st.container(border=True):
                    ui.agency_card(a["name"], a["licence_no"], a["status"], a["source"], a["retrieved"], len(mine))
                    if len(mine):
                        view = mine.copy()
                        view["max_fee_lkr"] = pd.to_numeric(view["max_fee_lkr"], errors="coerce")
                        view = view.sort_values("valid_until", ascending=False).head(50)
                        st.dataframe(view[["country", "job_title", "vacancies_approved", "max_fee_lkr", "valid_until"]],
                                     hide_index=True, use_container_width=True, height=min(38 + 35 * len(view), 280),
                                     column_config={
                                         "country": "Country", "job_title": "Job", "vacancies_approved": "Vacancies",
                                         "max_fee_lkr": st.column_config.NumberColumn("Max fee (Rs.)", format="%,d"),
                                         "valid_until": "Valid until"})
            if len(hits) > 5:
                st.caption(f"{len(hits)} matches; showing the first 5. Type more of the name to narrow it down.")

# ---------------------------------------------------------------- monitoring
with dash_tab:
    report = drift_now
    sig, thr = report.get("signals", {}), report.get("thresholds", {})
    if report["drift"]:
        fired = ", ".join(SIGNALS.get(k, k).lower() for k in report.get("triggered", []))
        ui.banner(False, "Drift detected", f"Signals fired: {fired}. High-risk scans now also go to a human analyst.")
    else:
        ui.banner(True, "Model is stable", f"No drift in the last {report['n']} scans.")

    rows = read_log()
    df = pd.json_normalize(rows) if rows else pd.DataFrame()
    n = len(df)
    high = int((df["risk.label"] == "high").sum()) if n else 0
    review = int(df["needs_review"].sum()) if n else 0
    alerts = read_alerts()
    k = st.columns(4)
    k[0].markdown(ui.kpi("Scans logged", f"{n:,}", "all channels"), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("High risk", f"{high:,}", f"{(high / n * 100 if n else 0):.0f}% of scans"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Sent to review", f"{review:,}", "medium risk, novel, or during drift"), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("Alerts", f"{len(alerts):,}", "drift, retraining"), unsafe_allow_html=True)
    st.write("")

    s_col, c_col = st.columns([5, 7], gap="large")
    with s_col, st.container(border=True):
        ui.section("Drift signals (latest window)")
        if sig and thr:
            ui.html("".join(ui.signal_bar(SIGNALS.get(key, key), sig.get(key, 0.0), t) for key, t in thr.items()))
            st.caption("The grey line marks each alert threshold, calibrated on held-out vs shifted data.")
        else:
            st.caption("Not enough scans yet.")
    with c_col, st.container(border=True):
        ui.section("Risk mix and trend")
        if n:
            counts = df["risk.label"].value_counts().reindex(["low", "medium", "high"], fill_value=0)
            mix = counts.rename_axis("Risk").reset_index(name="Scans")
            st.altair_chart(alt.Chart(mix, height=150).mark_bar(cornerRadiusEnd=6).encode(
                y=alt.Y("Risk:N", sort=["low", "medium", "high"], title=None), x=alt.X("Scans:Q", title=None),
                color=alt.Color("Risk:N", legend=None, scale=alt.Scale(
                    domain=["low", "medium", "high"], range=["#12B76A", "#F79009", "#F04438"]))),
                use_container_width=True)
            trend = df[["risk.probability", "novelty.score"]].rolling(10, min_periods=1).mean().reset_index(names="Scan")
            trend = trend.rename(columns={"risk.probability": "Scam likelihood", "novelty.score": "Novelty"}).melt(
                "Scan", var_name="Series", value_name="Rolling mean")
            st.altair_chart(alt.Chart(trend, height=190).mark_line(strokeWidth=2.5).encode(
                x=alt.X("Scan:Q", title="Scan number"),
                y=alt.Y("Rolling mean:Q", title="Rolling mean (10 scans)", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Series:N", scale=alt.Scale(range=["#0B3B60", "#0F766E"]),
                                legend=alt.Legend(orient="top", title=None))),
                use_container_width=True)  # not .interactive(): scrolling the page never zooms the chart
        else:
            st.caption("No scans yet.")

    with st.container(border=True):
        ui.section("Alert log")
        if alerts:
            log = pd.DataFrame(alerts[::-1])
            log["ts"] = pd.to_datetime(log["ts"], errors="coerce").dt.strftime("%d %b %H:%M:%S")
            log["kind"] = log["kind"].str.replace("_", " ").str.capitalize()
            if "triggered" in log.columns:
                log["triggered"] = log["triggered"].apply(
                    lambda t: ", ".join(SIGNALS.get(x, x) for x in t) if isinstance(t, list) else "")
            st.dataframe(log[[c for c in ("ts", "kind", "triggered", "n") if c in log.columns]].rename(
                columns={"ts": "Time", "kind": "Event", "triggered": "Signals", "n": "Window"}),
                hide_index=True, use_container_width=True)
        else:
            st.caption("No alerts yet.")

# ---------------------------------------------------------------- analyst console
with analyst_tab:
    queue = pending_reviews()
    models = retrain.status()
    lists = reputation.lists()
    k = st.columns(4)
    k[0].markdown(ui.kpi("Waiting for review", f"{len(queue):,}"), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("Blocklisted sites", f"{len(lists['block']):,}"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Verified sites", f"{len(lists['allow']):,}"), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("Model version", models["current"] or "initial"), unsafe_allow_html=True)
    st.write("")

    q_col, m_col = st.columns([7, 5], gap="large")
    with q_col:
        ui.section("Review queue")
        if not queue:
            with st.container(border=True):
                st.caption("Nothing waiting. Medium-risk, novel and drift-time scans appear here.")
        for item in queue[:25]:
            rid = item["review_id"]
            with st.container(border=True):
                if item.get("kind") == "domain":
                    ui.html(f'{ui.pill("Reported site", "medium")} &nbsp;<b>{ui.escape(item["domain"])}</b> '
                            f'· {item["scam_reports"]} scam reports')
                    with st.expander("Reports"):
                        st.dataframe(pd.DataFrame(reputation.reports(item["domain"])), use_container_width=True)
                    options = ["block", "allow", "dismiss"]
                else:
                    label, p = item["risk"]["label"], item["risk"]["probability"]
                    media = {"audio": " · 🎙️ transcript", "image": " · 🖼️ OCR text"}.get(item.get("media"), "")
                    ui.html(f'{ui.pill(f"{label.upper()} · {p:.2f}", label)}'
                            f'<span style="color:#667085;font-size:.82rem">{media}</span>')
                    ui.chips(item.get("indicators", [])[:6])
                    ui.quote(item.get("text", "")[:500])
                    options = ["scam", "legit", "dismiss"]
                cols = st.columns(len(options))
                for col, decision in zip(cols, options):
                    if col.button(decision.capitalize(), key=f"{rid}:{decision}", use_container_width=True,
                                  type="primary" if decision in ("scam", "block") else "secondary"):
                        reputation.apply_review(rid, decision)
                        st.rerun()
        if len(queue) > 25:
            st.caption(f"Showing 25 of {len(queue)}.")

    with m_col:
        with st.container(border=True):
            ui.section("Model")
            last = models["last_retrain"]
            st.markdown(f"**Current:** {models['current'] or 'initial (unversioned)'}  \n"
                        f"**Previous:** {models['previous'] or 'none'}")
            if last:
                held = (last.get("candidate") or {}).get("heldout", {})
                msg = (f"Last retrain {last['version']}: {last['status'].upper()} "
                       f"(held-out F1 {held.get('f1', '-')}, recall {held.get('recall', '-')})")
                {"promoted": st.success, "rejected": st.warning}.get(last["status"], st.error)(msg)
                st.caption("; ".join(last.get("gate", [])) or last.get("error", ""))
            if st.button("Retrain with confirmed labels", type="primary", use_container_width=True):
                st.info(trigger_retrain("analyst console"))
            x, y = st.columns(2)
            if x.button("Reload model", use_container_width=True):
                st.info({"model_loaded": reload_model()})
            if y.button("Roll back", use_container_width=True, disabled=not models["previous"]):
                st.info(retrain.rollback())
        with st.container(border=True):
            ui.section("Site lists")
            for title, key in (("Blocklist", "block"), ("Verified", "allow")):
                st.markdown(f"**{title}**")
                entries = [{"Site": d, "Reason": (v or {}).get("reason", ""), "By": (v or {}).get("by", "")}
                           for d, v in lists[key].items()]
                if entries:
                    st.dataframe(pd.DataFrame(entries), hide_index=True, use_container_width=True)
                else:
                    st.caption("Empty. Sites are listed only by an analyst decision.")

ui.footer()
