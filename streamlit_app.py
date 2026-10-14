import os
import sys
import time
import json
import tempfile
import threading
from datetime import datetime
import pandas as pd
import streamlit as st

from sheets_engine import AppsScriptManager, SheetsManager
from gemini_engine import GeminiMCQEngine, MULTIMODAL_MODEL_POOL

# Default credentials
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
        margin-bottom: 18px;
    }
    .status-card {
        background-color: #064E3B;
        border: 2px solid #10B981;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
        color: #ECFDF5;
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

# Global Persistent Task Manager (Runs completely detached from browser session)
class CloudTaskManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.status_text = "Idle. Ready for extraction."
        self.progress_val = 0.0
        self.extracted_mcqs = []
        self.manifest = []
        self.logs = []
        self.thread = None
        self.start_time = None
        self.finished_time = None
        self.error_message = None

    def add_log(self, lvl: str, msg: str):
        timestamp = time.strftime("[%H:%M:%S]")
        entry = f"{timestamp} [{lvl}] {msg}"
        with self.lock:
            self.logs.append(entry)
            self.status_text = f"{lvl}: {msg}"

    def update_progress(self, cur: float, tot: float, txt: str):
        with self.lock:
            self.progress_val = min(1.0, max(0.0, cur / tot if tot > 0 else 0.0))
            self.status_text = txt

    def add_mcq(self, item: dict):
        with self.lock:
            self.extracted_mcqs.append(item)

    def set_manifest(self, manifest_data: list):
        with self.lock:
            self.manifest = manifest_data

@st.cache_resource
def get_global_task_manager():
    return CloudTaskManager()

task_mgr = get_global_task_manager()

st.markdown('<div class="main-title">⚡ Gemini Smart MCQ Extractor (Cloud Edition)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">100% Background Execution • Close Browser / Laptop Off Anytime • Google Sheets Auto-Sync</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Cloud Configuration")
    
    api_key = st.text_input(
        "Gemini API Key (AIzaSy...)",
        value="",
        type="password",
        help="Paste your Gemini API Key from Google AI Studio"
    )
    
    webhook_url = st.text_input(
        "Apps Script Webhook URL",
        value=DEFAULT_WEBHOOK_URL,
        help="Google Apps Script Web App URL for Google Sheets"
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
        help="Automatically switches to next free multimodal model if quota limit is reached"
    )
    
    delay_sec = st.number_input(
        "Rate-Limit Delay Buffer (sec)",
        min_value=2.0,
        max_value=30.0,
        value=6.0,
        step=1.0,
        help="Delay between sets to stay under RPM limits"
    )

    st.markdown("---")
    st.success("✅ **Background Mode Enabled**: Once started, you can close this browser or turn off your computer. The cloud will finish extracting every question into your Google Sheet automatically!")

# Background Task Worker Function
def background_worker(pdf_bytes: bytes, filename: str, g_api_key: str, wh_url: str, t_name: str, model: str, delay: float, do_switch: bool):
    task_mgr.is_running = True
    task_mgr.start_time = datetime.now()
    task_mgr.extracted_mcqs = []
    task_mgr.manifest = []
    task_mgr.logs = []
    task_mgr.error_message = None

    task_mgr.add_log("INFO", f"Cloud background worker started for '{filename}'.")
    task_mgr.add_log("SUCCESS", "Background worker is running independently. Safe to close browser/laptop!")

    # Write PDF to server temp file
    temp_dir = tempfile.gettempdir()
    temp_pdf = os.path.join(temp_dir, f"cloud_mcq_{int(time.time())}.pdf")
    with open(temp_pdf, "wb") as f:
        f.write(pdf_bytes)

    try:
        sheets_mgr = AppsScriptManager(wh_url, t_name)
        engine = GeminiMCQEngine(
            api_key=g_api_key,
            model_name=model,
            inter_request_delay=delay,
            log_callback=task_mgr.add_log,
            progress_callback=task_mgr.update_progress,
            preview_callback=task_mgr.add_mcq,
            manifest_callback=task_mgr.set_manifest,
            auto_switch_models=do_switch
        )

        backup_file = temp_pdf + "_backup.json"
        success = engine.run_pipeline(
            pdf_path=temp_pdf,
            sheets_manager=sheets_mgr,
            local_backup_path=backup_file
        )

        if success:
            task_mgr.add_log("SUCCESS", f"Extraction completed! Total {len(task_mgr.extracted_mcqs)} questions pushed to Google Sheet.")
        else:
            task_mgr.add_log("WARNING", "Extraction pipeline ended with warnings.")

    except Exception as e:
        task_mgr.error_message = str(e)
        task_mgr.add_log("ERROR", f"Cloud worker error: {e}")

    finally:
        task_mgr.is_running = False
        task_mgr.finished_time = datetime.now()
        try:
            if os.path.exists(temp_pdf):
                os.remove(temp_pdf)
        except Exception:
            pass

# Check Active State Banner
if task_mgr.is_running:
    st.markdown(f"""
    <div class="status-card">
        <h3 style="margin:0 0 6px 0; color:#34D399;">⚡ Background Extraction is IN PROGRESS on Cloud Server!</h3>
        <p style="margin:0; font-size:1.05rem;">
            <b>Status:</b> {task_mgr.status_text}<br>
            <b>Extracted so far:</b> {len(task_mgr.extracted_mcqs)} questions<br>
            <b>Safe to Exit:</b> You can now safely <b>CLOSE this browser tab, disconnect internet, or SHUT DOWN your laptop</b>. 
            The cloud server will continue extracting and posting every question to your Google Sheet automatically!
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.progress(task_mgr.progress_val)
    time.sleep(2)
    st.rerun()

# Main Upload Area
col_up, col_action = st.columns([3, 1])

with col_up:
    uploaded_file = st.file_uploader(
        "📄 Select PDF File (Up to 50MB)",
        type=["pdf"],
        help="Upload the question bank PDF document"
    )

with col_action:
    st.write("")
    st.write("")
    start_btn = st.button(
        "▶ Start Background Extraction",
        type="primary",
        use_container_width=True,
        disabled=task_mgr.is_running
    )

if start_btn:
    if not uploaded_file:
        st.error("Please upload a PDF file first.")
    elif not api_key:
        st.error("Please provide your Gemini API Key in the sidebar.")
    elif not webhook_url:
        st.error("Please provide your Google Apps Script Webhook URL.")
    else:
        pdf_content = uploaded_file.getvalue()
        fname = uploaded_file.name

        # Launch detached background thread (daemon=False ensures it outlives the web session!)
        worker_thread = threading.Thread(
            target=background_worker,
            args=(pdf_content, fname, api_key, webhook_url, tab_name, selected_model, delay_sec, auto_switch),
            daemon=False
        )
        task_mgr.thread = worker_thread
        worker_thread.start()
        st.success("🚀 Background task launched on cloud server! You can now close your laptop or browser.")
        time.sleep(1)
        st.rerun()

# Tabs
tab_preview, tab_manifest, tab_logs, tab_download = st.tabs([
    "👁 Live MCQ Preview",
    "🗺️ Mapped Sets & Boundaries",
    "💻 Live Execution Logs",
    "📥 Download Local Backup"
])

# 1. Preview Tab
with tab_preview:
    count = len(task_mgr.extracted_mcqs)
    st.subheader(f"Total Extracted Questions: {count}")
    if task_mgr.extracted_mcqs:
        for item in reversed(task_mgr.extracted_mcqs[-40:]):
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
    else:
        st.info("No questions in buffer. Click 'Start Background Extraction' to begin.")

# 2. Manifest Tab
with tab_manifest:
    if task_mgr.manifest:
        st.subheader(f"Discovered Sets: {len(task_mgr.manifest)}")
        df = pd.DataFrame(task_mgr.manifest)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Topic and question boundary manifest will appear once extraction starts.")

# 3. Logs Tab
with tab_logs:
    st.text_area("Live Log Output", value="\n".join(task_mgr.logs[-200:]), height=380)

# 4. Download Tab
with tab_download:
    st.subheader("📥 Export Questions Directly")
    if task_mgr.extracted_mcqs:
        df_export = pd.DataFrame(task_mgr.extracted_mcqs)
        csv_data = df_export.to_csv(index=False).encode('utf-8-sig')
        json_data = json.dumps(task_mgr.extracted_mcqs, ensure_ascii=False, indent=2).encode('utf-8')

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
        st.info("No questions extracted yet.")
