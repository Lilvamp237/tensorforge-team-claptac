"""Streamlit UI: scan page + monitoring dashboard. Owner: Person B.

Run: PYTHONPATH=src streamlit run app/streamlit_app.py
"""

import pandas as pd
import streamlit as st

from safepassage.drift import drift_report
from safepassage.monitoring import read_log
from safepassage.service import scan_file, scan_public_url, scan_text

st.set_page_config(page_title="SafePassage LK", layout="wide")
scan_tab, dash_tab = st.tabs(["Check an offer", "Monitoring dashboard"])

with scan_tab:
    st.title("SafePassage LK")
    st.caption("Check a job, study or visa offer before you pay. Risk assessment only, not a fraud determination.")
    text = st.text_area("Paste the offer or message")
    url = st.text_input("...or a public link")
    upload = st.file_uploader("...or upload a letter (PDF/TXT)", type=["pdf", "txt"])
    if st.button("Check", type="primary"):
        if upload:
            result = scan_file(upload.read(), upload.name)
        elif url:
            result = scan_public_url(url)
        elif text.strip():
            result = scan_text(text)
        else:
            st.stop()
        st.metric("Risk", result.risk.label.upper(), f"P(scam) = {result.risk.probability:.2f}")
        for flag in result.indicators:
            st.write(f"- {flag}")
        if result.needs_review:
            st.info("Sent to human review.")
        with st.expander("Full result"):
            st.json(result.model_dump())

with dash_tab:
    report = drift_report()
    if report["drift"]:
        st.error(f"Drift detected: {report['signals']}")
    else:
        st.success(f"No drift ({report['n']} recent scans)")
    rows = read_log()
    if rows:
        df = pd.json_normalize(rows)
        st.bar_chart(df["risk.label"].value_counts())
        st.line_chart(df[["risk.probability", "novelty.score"]])
        # TODO(B): review-queue table with approve/relabel, retrain button, alert history
