"""Streamlit UI: scan page, monitoring dashboard and analyst console. Owner: Person B.

Run: PYTHONPATH=src streamlit run app/streamlit_app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from safepassage import reputation, retrain
from safepassage.fetch import UnsafeURL
from safepassage.monitoring import pending_reviews, read_alerts, read_log
from safepassage.response import check_drift, trigger_retrain
from safepassage.media import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, MediaError, is_audio, is_image
from safepassage.service import reload_model, scan_audio, scan_file, scan_image, scan_public_url, scan_text

st.set_page_config(page_title="SafePassage LK", layout="wide")
scan_tab, dash_tab, analyst_tab = st.tabs(["Check an offer", "Monitoring", "Analyst console"])

with scan_tab:
    st.title("SafePassage LK")
    st.caption("Check a job, study or visa offer before you pay. Risk assessment only, not a fraud determination.")
    text = st.text_area("Paste the offer or message")
    url = st.text_input("...or a public link")
    upload = st.file_uploader("...or upload a letter (PDF/TXT), a screenshot (PNG/JPG) or a voice note (MP3/WAV/M4A/OGG)",
                              type=["pdf", "txt"] + [e.lstrip(".") for e in AUDIO_EXTENSIONS + IMAGE_EXTENSIONS])
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
            elif upload and is_image(upload.name):
                with st.spinner("Reading the image..."):
                    scan = scan_image(upload.read())
                st.session_state.extracted = ("What we read", scan.text)
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
        advice = {
            "high": (st.error, "High risk. Do not pay or send documents. Check the agency with SLBFE (hotline 1989) first."),
            "medium": (st.warning, "Some warning signs. Verify the agency and the fee with SLBFE (1989) before you pay."),
            "low": (st.success, "No strong warning signs found. Still, pay only official fees, with a receipt."),
        }
        show, text = advice[result.risk.label]
        show(f"**{result.risk.label.upper()} RISK** (P(scam) = {result.risk.probability:.2f}). {text}")
        c = result.content
        facts = [x for x in (c.type.replace("_", " ") if c.type != "unknown" else None, c.claimed_country,
                             c.claimed_agency, {"en": "English", "si": "Sinhala", "ta": "Tamil", "singlish": "Singlish",
                                                "mixed": "mixed language"}.get(c.language)) if x]
        if facts:
            st.caption("Read as: " + " · ".join(facts))
        left, right = st.columns(2)
        with left:
            st.markdown("**Warning signs**")
            for flag in result.indicators or ["None found"]:
                st.write(f"- {flag}")
        with right:
            st.markdown("**Official records**")
            v = result.verification
            st.write({"matched": "✅ Agency found in the SLBFE registry", "not_matched": "❌ Agency not confirmed by SLBFE records",
                      "unknown": "❔ No agency or licence named"}[v.agency_status])
            icon = {"consistent": "✅", "contradicted": "❌", "unverifiable": "❔"}
            for claim in v.claims:
                st.write(f"{icon[claim.status]} {claim.claim}")
                if claim.source:
                    st.caption(claim.source)
        if result.needs_review:
            st.info("An analyst will double-check this one.")
        with st.expander("Full result (JSON)"):
            st.json(result.model_dump())

    st.divider()
    st.subheader("Report a website")
    with st.form("report"):
        report_url = st.text_input("Website or listing link")
        verdict = st.radio("This offer is", ["scam", "legit"], horizontal=True)
        comment = st.text_input("What happened? (no names or phone numbers)")
        rater_id = st.text_input("Your SafePassage ID (optional)",
                                 help="Lets reliable reporters count for more. Stored only as a salted hash.")
        if st.form_submit_button("Send report") and report_url:
            r = reputation.add_report(report_url, verdict, comment, rater_id.strip() or None)
            st.success(f"Thanks. {r['scam_reports']} scam report(s) on {r['domain']}. Analysts review sites with repeated reports.")

with dash_tab:
    report = check_drift()
    names = {"language_psi": "Language mix changed", "category_psi": "Offer-type mix changed",
             "confidence_drop": "Model confidence dropped", "novelty_share": "Unfamiliar offers rising"}
    sig, thr = report.get("signals", {}), report.get("thresholds", {})
    if report["drift"]:
        fired = [f"{names.get(k, k)} ({sig.get(k, 0):.2f} > {thr.get(k, 0):.2f})" for k in report.get("triggered", [])]
        st.error("**Drift detected.** High-risk scans now also go to human review. " + "; ".join(fired))
    else:
        st.success(f"**No drift** in the last {report['n']} scans.")
    if sig and thr:
        st.dataframe(pd.DataFrame([{"Signal": names.get(k, k), "Now": round(sig.get(k, 0), 3), "Alert above": round(v, 3)}
                                   for k, v in thr.items()]), hide_index=True, use_container_width=True)
    rows = read_log()
    if rows:
        df = pd.json_normalize(rows)
        c1, c2, c3 = st.columns(3)
        c1.metric("Scans", len(df))
        c2.metric("High risk", int((df["risk.label"] == "high").sum()))
        c3.metric("Sent to review", int(df["needs_review"].sum()))
        counts = df["risk.label"].value_counts().reindex(["low", "medium", "high"], fill_value=0)
        st.altair_chart(alt.Chart(counts.rename_axis("Risk").reset_index(name="Scans")).mark_bar().encode(
            x=alt.X("Risk:N", sort=["low", "medium", "high"]), y="Scans:Q",
            color=alt.Color("Risk:N", scale=alt.Scale(domain=["low", "medium", "high"],
                                                      range=["#2e7d32", "#f9a825", "#c62828"]), legend=None)),
            use_container_width=True)
        trend = df[["risk.probability", "novelty.score"]].rolling(10, min_periods=1).mean().reset_index(names="Scan")
        trend = trend.rename(columns={"risk.probability": "P(scam)", "novelty.score": "Novelty"}).melt(
            "Scan", var_name="Series", value_name="Rolling mean (10 scans)")
        st.altair_chart(alt.Chart(trend).mark_line().encode(x="Scan:Q", y="Rolling mean (10 scans):Q", color="Series:N"),
                        use_container_width=True)  # not .interactive(), so scrolling the page never zooms the chart
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
