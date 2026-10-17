"""Streamlit UI: scan page, monitoring dashboard and analyst console. Owner: Person B.

Run: PYTHONPATH=src streamlit run app/streamlit_app.py
"""

import pandas as pd
import streamlit as st

from safepassage import reputation, retrain
from safepassage.fetch import UnsafeURL
from safepassage.monitoring import pending_reviews, read_alerts, read_log
from safepassage.response import check_drift, trigger_retrain
from safepassage.media import AUDIO_EXTENSIONS, MediaError, is_audio
from safepassage.service import reload_model, scan_audio, scan_file, scan_public_url, scan_text

st.set_page_config(page_title="SafePassage LK", layout="wide")
scan_tab, dash_tab, analyst_tab = st.tabs(["Check an offer", "Monitoring", "Analyst console"])

with scan_tab:
    st.title("SafePassage LK")
    st.caption("Check a job, study or visa offer before you pay. Risk assessment only, not a fraud determination.")
    text = st.text_area("Paste the offer or message")
    url = st.text_input("...or a public link")
    upload = st.file_uploader("...or upload a letter (PDF/TXT) or a voice note (MP3/WAV/M4A/OGG)",
                              type=["pdf", "txt"] + [e.lstrip(".") for e in AUDIO_EXTENSIONS])
    hint = st.selectbox("Voice note language", ["Auto-detect", "English", "Tamil", "Sinhala"],
                        help="Optional. Transcription is weaker for Sinhala.")
    if st.button("Check", type="primary"):
        st.session_state.extracted = None
        try:
            if upload and is_audio(upload.name):
                lang = {"English": "en", "Tamil": "ta", "Sinhala": "si"}.get(hint)
                with st.spinner("Listening... (the first voice note downloads the speech model, about 460 MB)"):
                    scan = scan_audio(upload.read(), upload.name, lang)
                st.session_state.extracted = ("What we heard", scan.text)
                st.session_state.result = scan.result
            elif upload:
                st.session_state.result = scan_file(upload.read(), upload.name)
            elif url:
                st.session_state.result = scan_public_url(url)
            elif text.strip():
                st.session_state.result = scan_text(text)
        except UnsafeURL as e:
            st.error(f"Can't check that link: {e}")
        except MediaError as e:
            st.error(f"Can't read that file: {e}")

    result = st.session_state.get("result")
    extracted = st.session_state.get("extracted")
    if result and extracted:
        st.text_area(extracted[0], extracted[1], disabled=True)
    if result:
        st.metric("Risk", result.risk.label.upper(), f"P(scam) = {result.risk.probability:.2f}", delta_color="off")
        for flag in result.indicators:
            st.write(f"- {flag}")
        if result.needs_review:
            st.info("An analyst will double-check this one.")
        with st.expander("Full result"):
            st.json(result.model_dump())

    st.divider()
    st.subheader("Report a website")
    with st.form("report"):
        report_url = st.text_input("Website or listing link")
        verdict = st.radio("This offer is", ["scam", "legit"], horizontal=True)
        comment = st.text_input("What happened? (no names or phone numbers)")
        if st.form_submit_button("Send report") and report_url:
            r = reputation.add_report(report_url, verdict, comment)
            st.success(f"Thanks. {r['scam_reports']} scam report(s) on {r['domain']}. Analysts review sites with repeated reports.")

with dash_tab:
    report = check_drift()
    if report["drift"]:
        st.error(f"Drift detected: high-risk scans now go to human review. Signals: {report['signals']}")
    else:
        st.success(f"No drift ({report['n']} recent scans). Signals: {report['signals']}")
    rows = read_log()
    if rows:
        df = pd.json_normalize(rows)
        c1, c2, c3 = st.columns(3)
        c1.metric("Scans", len(df))
        c2.metric("High risk", int((df["risk.label"] == "high").sum()))
        c3.metric("Sent to review", int(df["needs_review"].sum()))
        st.bar_chart(df["risk.label"].value_counts())
        st.line_chart(df[["risk.probability", "novelty.score"]].rolling(10, min_periods=1).mean())
    st.subheader("Alerts")
    alerts = read_alerts()
    st.dataframe(pd.DataFrame(alerts[::-1]) if alerts else pd.DataFrame(), use_container_width=True)

with analyst_tab:
    st.subheader("Review queue")
    queue = pending_reviews()
    if not queue:
        st.write("Nothing waiting.")
    for item in queue:
        rid = item["review_id"]
        with st.container(border=True):
            if item.get("kind") == "domain":
                st.write(f"**Reported site:** {item['domain']} ({item['scam_reports']} scam reports)")
                with st.expander("Reports"):
                    st.dataframe(pd.DataFrame(reputation.reports(item["domain"])), use_container_width=True)
                options = ["block", "allow", "dismiss"]
            else:
                st.write(f"**{item['risk']['label'].upper()}** ({item['risk']['probability']:.2f}) · {', '.join(item['indicators']) or 'no red flags'}")
                st.caption(item.get("text", "")[:500])
                options = ["scam", "legit", "dismiss"]
            cols = st.columns(len(options))
            for col, decision in zip(cols, options):
                if col.button(decision.capitalize(), key=f"{rid}:{decision}"):
                    reputation.apply_review(rid, decision)
                    st.rerun()

    st.subheader("Site lists")
    lists = reputation.lists()
    left, right = st.columns(2)
    left.write("**Blocklist**")
    left.json(lists["block"])
    right.write("**Verified list**")
    right.json(lists["allow"])

    st.subheader("Model")
    models = retrain.status()
    last = models["last_retrain"]
    st.write(f"**Current version:** {models['current'] or 'initial (unversioned)'}"
             f" · previous: {models['previous'] or 'none'}")
    if last:
        held = (last.get("candidate") or {}).get("heldout", {})
        msg = (f"Last retrain {last['version']}: **{last['status'].upper()}** "
               f"(held-out F1 {held.get('f1', '-')}, recall {held.get('recall', '-')})")
        {"promoted": st.success, "rejected": st.warning}.get(last["status"], st.error)(msg)
        st.caption("; ".join(last.get("gate", [])) or last.get("error", ""))
    a, b, c = st.columns(3)
    if a.button("Retrain with confirmed labels"):
        st.write(trigger_retrain("analyst console"))
    if b.button("Reload model"):
        st.write({"model_loaded": reload_model()})
    if c.button("Roll back to previous version", disabled=not models["previous"]):
        st.write(retrain.rollback())
