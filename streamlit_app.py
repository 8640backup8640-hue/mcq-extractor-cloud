import os
import sys
import time
import json
import tempfile
import threading
import pandas as pd
import streamlit as st

from sheets_engine import AppsScriptManager, SheetsManager
from gemini_engine import GeminiMCQEngine, MULTIMODAL_MODEL_POOL

# Default credentials pre-configured
DEFAULT_API_KEY = "AQ.Ab8RN6K-gTsF7rGMP0FhuJkXBmPeWlBkikE3umS0snn9AyIAHQ"
DEFAULT_WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbw8VNSd5evA8RG7Bb8U-g-AqJU8_MI88YaTr-FQfVop8XhuyGxFzjniwc4q05EYKtD_oA/exec"

st.set_page_config(
    page_title="Gemini Smart MCQ Extractor (Cloud Edition)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38BDF8;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-bottom: 20px;
    }
    .card-box {
        background-color: #1E293B;
        padding: 16px;
        border-radius: 10px;
        border-left: 4px solid #38BDF8;
        margin-bottom: 12px;
    }
    .mcq-card {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .mcq-title {
        color: #38BDF8;
        font-weight: bold;
        font-size: 1.05rem;
    }
    .mcq-ans {
        color: #10B981;
        font-weight: bold;
        background-color: #064E3B;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "extracted_mcqs" not in st.session_state:
    st.session_state.extracted_mcqs = []
if "mapped_manifest" not in st.session_state:
    st.session_state.mapped_manifest = []
if "live_logs" not in st.session_state:
    st.session_state.live_logs = []
if "is_running" not in st.session_state:
    st.session_state.is_running = False

st.markdown('<div class="main-title">⚡ Gemini Smart MCQ Extractor (Cloud Edition)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">100% Online Cloud Extraction • Boundary Mapping • Auto-Continuation • Multimodal Auto-Switching Pool</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Cloud Configuration")
    
    api_key = st.text_input(
        "Gemini API Key",
        value=DEFAULT_API_KEY,
        type="password",
        help="Google Gemini API Key"
    )
    
    webhook_url = st.text_input(
        "Apps Script Webhook URL",
        value=DEFAULT_WEBHOOK_URL,
        help="Google Apps Script Web App URL for Google Sheets export"
    )
    
    tab_name = st.text_input("Sheet Tab Name", value="Sheet1")
    
    selected_model = st.selectbox(
        "Primary Model",
        options=MULTIMODAL_MODEL_POOL,
        index=0,
        help="Select starting model from the multimodal pool"
    )
    
    auto_switch = st.checkbox(
        "⚡ Auto-Switch Model Pool (If 429 Quota Hit)",
        value=True,
        help="Cascades to other free models if quota limit is reached"
    )
    
    delay_sec = st.number_input(
        "Rate-Limit Delay Buffer (sec)",
        min_value=2.0,
        max_value=30.0,
        value=6.0,
        step=1.0,
        help="Sleep between iterations to respect RPM limits"
    )

    st.markdown("---")
    st.markdown("💡 **Tip**: Running on GitHub/Streamlit Cloud allows you to start extraction from your phone or PC and let it complete in the background!")

# Main Upload Area
col_up, col_action = st.columns([3, 1])

with col_up:
    uploaded_file = st.file_uploader(
        "📄 Select or Drag & Drop PDF File (Up to 50MB)",
        type=["pdf"],
        help="Upload the question bank PDF document"
    )

with col_action:
    st.write("")
    st.write("")
    start_btn = st.button("▶ Start Cloud Extraction", type="primary", use_container_width=True, disabled=st.session_state.is_running)

# Main Dashboard Tabs
tab_preview, tab_manifest, tab_logs, tab_download = st.tabs([
    "👁 Live MCQ Preview",
    "🗺️ Mapped Sets & Boundaries",
    "💻 Live Execution Logs",
    "📥 Download Local Backup"
])

# Containers for real-time updates
preview_container = tab_preview.container()
manifest_container = tab_manifest.container()
logs_container = tab_logs.container()

def display_live_preview():
    with preview_container:
        count = len(st.session_state.extracted_mcqs)
        st.subheader(f"Total Extracted Questions: {count}")
        for item in reversed(st.session_state.extracted_mcqs[-50:]):
            st.markdown(f"""
            <div class="mcq-card">
                <div class="mcq-title">#{item.get('Serial', '')} [{item.get('Set', '')}]</div>
                <div style="margin: 8px 0; color: #F8FAFC;"><b>প্রশ্ন:</b> {item.get('Question', '')}</div>
                <div style="color: #CBD5E1; font-size: 0.95rem;">
                    <b>(A)</b> {item.get('Option-A', '')} &nbsp;&nbsp;|&nbsp;&nbsp; <b>(B)</b> {item.get('Option-B', '')}<br>
                    <b>(C)</b> {item.get('Option-C', '')} &nbsp;&nbsp;|&nbsp;&nbsp; <b>(D)</b> {item.get('Option-D', '')}
                </div>
                <div style="margin-top: 8px;">
                    <span class="mcq-ans">✓ Answer: {item.get('Answer', '')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_manifest():
    with manifest_container:
        if st.session_state.mapped_manifest:
            st.subheader(f"Discovered Sets: {len(st.session_state.mapped_manifest)}")
            df = pd.DataFrame(st.session_state.mapped_manifest)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Manifest will populate during Phase 1 Discovery.")

def display_logs():
    with logs_container:
        st.text_area("Live Log Output", value="\n".join(st.session_state.live_logs[-200:]), height=350)

# Display existing states
display_live_preview()
display_manifest()
display_logs()

# Execution Logic
if start_btn:
    if not uploaded_file:
        st.error("Please select a PDF file first.")
    elif not api_key:
        st.error("Please enter a valid Gemini API Key.")
    elif not webhook_url:
        st.error("Please enter a valid Google Apps Script Webhook URL.")
    else:
        st.session_state.is_running = True
        st.session_state.extracted_mcqs = []
        st.session_state.mapped_manifest = []
        st.session_state.live_logs = []

        # Save uploaded file to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_pdf_path = tmp.name

        progress_bar = st.progress(0.0)
        status_box = st.empty()

        def log_cb(lvl: str, msg: str):
            entry = f"[{time.strftime('%H:%M:%S')}] [{lvl}] {msg}"
            st.session_state.live_logs.append(entry)
            status_box.info(f"{lvl}: {msg}")

        def prog_cb(cur: float, tot: float, txt: str):
            progress_bar.progress(min(1.0, max(0.0, cur / tot if tot > 0 else 0.0)))
            status_box.info(txt)

        def prev_cb(item: dict):
            st.session_state.extracted_mcqs.append(item)

        def man_cb(manifest: list):
            st.session_state.mapped_manifest = manifest

        sheets_mgr = AppsScriptManager(webhook_url, tab_name)
        engine = GeminiMCQEngine(
            api_key=api_key,
            model_name=selected_model,
            inter_request_delay=delay_sec,
            log_callback=log_cb,
            progress_callback=prog_cb,
            preview_callback=prev_cb,
            manifest_callback=man_cb,
            auto_switch_models=auto_switch
        )

        with st.spinner("Processing PDF on Google Cloud Engine..."):
            backup_json = temp_pdf_path + "_backup.json"
            success = engine.run_pipeline(
                pdf_path=temp_pdf_path,
                sheets_manager=sheets_mgr,
                local_backup_path=backup_json
            )

        st.session_state.is_running = False

        # Cleanup temp file
        try:
            os.remove(temp_pdf_path)
        except Exception:
            pass

        if success:
            st.success(f"🎉 Extraction Completed! Total Extracted: {len(st.session_state.extracted_mcqs)} questions sent to Google Sheets.")
        else:
            st.warning("Extraction process halted or experienced warnings. Check Live Logs tab.")

        display_live_preview()
        display_manifest()
        display_logs()

# Download Backup Tab
with tab_download:
    st.subheader("📥 Export Extracted Questions Directly")
    if st.session_state.extracted_mcqs:
        df_export = pd.DataFrame(st.session_state.extracted_mcqs)
        csv_data = df_export.to_csv(index=False).encode('utf-8-sig')
        json_data = json.dumps(st.session_state.extracted_mcqs, ensure_ascii=False, indent=2).encode('utf-8')

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                label="📄 Download CSV File",
                data=csv_data,
                file_name="extracted_mcqs.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c2:
            st.download_button(
                label="📋 Download JSON File",
                data=json_data,
                file_name="extracted_mcqs.json",
                mime="application/json",
                use_container_width=True
            )
    else:
        st.info("No questions extracted in current session yet.")
