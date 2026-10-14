import os
import sys
import time
import json
import tempfile
import threading
from datetime import datetime
import pandas as pd
import streamlit as st

from sheets_engine import AppsScriptManager
from gemini_engine import GeminiMCQEngine, MULTIMODAL_MODEL_POOL

# Default credentials
DEFAULT_WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbw8VNSd5evA8RG7Bb8U-g-AqJU8_MI88YaTr-FQfVop8XhuyGxFzjniwc4q05EYKtD_oA/exec"

st.set_page_config(
    page_title="Gemini Smart MCQ Extractor (Cloud Edition)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium, Vibrant & 100% Mobile/Desktop Responsive CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;600;700&family=Outfit:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Hind+Siliguri', sans-serif;
    }

    /* Vibrant Gradient Header */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin: 0;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 8px;
        margin-top: 8px;
    }
    .badge-cyan { background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.15); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-green { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }

    /* HUD Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 14px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #1E293B;
        border-radius: 12px;
        padding: 16px;
        border-left: 5px solid;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-card.purple { border-color: #8B5CF6; background: linear-gradient(145deg, #1E1B4B 0%, #1E293B 100%); }
    .metric-card.cyan { border-color: #06B6D4; background: linear-gradient(145deg, #083344 0%, #1E293B 100%); }
    .metric-card.green { border-color: #10B981; background: linear-gradient(145deg, #064E3B 0%, #1E293B 100%); }
    .metric-card.amber { border-color: #F59E0B; background: linear-gradient(145deg, #451A03 0%, #1E293B 100%); }
    .metric-num { font-size: 1.8rem; font-weight: 800; color: #F8FAFC; margin-bottom: 2px; }
    .metric-label { font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; font-weight: 600; }

    /* MCQ Preview Card */
    .mcq-card {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .mcq-card:hover {
        border-color: #38BDF8;
        transform: translateY(-2px);
    }
    .mcq-tag {
        font-size: 0.8rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        background: #1E293B;
        color: #38BDF8;
        display: inline-block;
        margin-bottom: 8px;
    }
    .mcq-q {
        font-size: 1.08rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 12px;
        line-height: 1.5;
    }
    .opt-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 8px;
        margin-bottom: 12px;
    }
    .opt-box {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.92rem;
        color: #CBD5E1;
    }
    .ans-badge {
        background: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
    }

    /* Mobile Responsive Optimizations */
    @media (max-width: 768px) {
        .hero-title { font-size: 1.6rem !important; }
        .hero-subtitle { font-size: 0.9rem !important; }
        .hero-container { padding: 16px !important; }
        .metric-grid { grid-template-columns: 1fr 1fr !important; }
        .opt-grid { grid-template-columns: 1fr !important; }
    }
</style>
""", unsafe_allow_html=True)

# Global Persistent Task Manager (Runs detached from browser session)
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

# Top Hero Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ Gemini Smart MCQ Extractor (Cloud Edition)</div>
    <div class="hero-subtitle">
        100% Strict Mapping & Extraction • Zero Duplicates • Zero Missed Questions • Micro-Batch Continuation
    </div>
    <div style="margin-top: 10px;">
        <span class="badge-pill badge-cyan">✓ Chapters & Topics</span>
        <span class="badge-pill badge-purple">✓ Page-Range Tracking</span>
        <span class="badge-pill badge-green">✓ Laptop Shut Down Safe</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Cloud Configuration")
    
    api_key = st.text_input(
        "Gemini API Key (AIzaSy...)",
        value="",
        type="password",
        help="Paste your Gemini API Key from Google AI Studio (aistudio.google.com/apikey)"
    )
    
    webhook_url = st.text_input(
        "Google Sheets Webhook URL",
        value=DEFAULT_WEBHOOK_URL,
        help="Google Apps Script Web App URL for Google Sheets"
    )
    
    tab_name = st.text_input("Sheet Tab Name", value="Sheet1")
    
    selected_model = st.selectbox(
        "Primary Model",
        options=MULTIMODAL_MODEL_POOL,
        index=0,
        help="Starting model. Pool switches automatically on quota limits."
    )
    
    chunk_size = st.slider(
        "Micro-Batch Size (Questions per request)",
        min_value=15,
        max_value=35,
        value=30,
        step=5,
        help="Extracting 25-30 questions per request guarantees zero token truncation and no missed questions."
    )

    auto_switch = st.checkbox(
        "⚡ Auto-Switch Model Pool (If 429 Quota Hit)",
        value=True,
        help="Switches model without losing context or repeating questions"
    )
    
    delay_sec = st.number_input(
        "Inter-Batch Delay Buffer (sec)",
        min_value=2.0,
        max_value=20.0,
        value=5.0,
        step=1.0,
        help="Safe delay between batches to stay under rate limits"
    )

    st.markdown("---")
    st.info("💡 **Background Cloud Feature:** Once you click 'Start Background Extraction', you can close your phone or laptop completely. The server will complete everything automatically!")

# Real-time Metrics HUD
total_qs = len(task_mgr.extracted_mcqs)
total_sets = len(task_mgr.manifest)
unique_chapters = len(set([item.get('chapter', 'General') for item in task_mgr.manifest])) if task_mgr.manifest else 0
status_label = "Running" if task_mgr.is_running else "Idle"

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card purple">
        <div class="metric-num">{unique_chapters}</div>
        <div class="metric-label">Chapters Audited</div>
    </div>
    <div class="metric-card cyan">
        <div class="metric-num">{total_sets}</div>
        <div class="metric-label">Topics / Sets Mapped</div>
    </div>
    <div class="metric-card green">
        <div class="metric-num">{total_qs}</div>
        <div class="metric-label">Authentic MCQs Extracted</div>
    </div>
    <div class="metric-card amber">
        <div class="metric-num">{status_label}</div>
        <div class="metric-label">Cloud Status</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Background Task Worker Function
def background_worker(pdf_bytes: bytes, filename: str, g_api_key: str, wh_url: str, t_name: str, model: str, delay: float, do_switch: bool, c_size: int):
    task_mgr.is_running = True
    task_mgr.start_time = datetime.now()
    task_mgr.extracted_mcqs = []
    task_mgr.manifest = []
    task_mgr.logs = []
    task_mgr.error_message = None

    task_mgr.add_log("INFO", f"Cloud worker initiated for '{filename}'.")
    task_mgr.add_log("SUCCESS", "Background worker detached. 100% safe to close browser or shut down laptop!")

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
            auto_switch_models=do_switch,
            chunk_size=c_size
        )

        backup_file = temp_pdf + "_backup.json"
        success = engine.run_pipeline(
            pdf_path=temp_pdf,
            sheets_manager=sheets_mgr,
            local_backup_path=backup_file
        )

        if success:
            task_mgr.add_log("SUCCESS", f"Extraction completed! Total {len(task_mgr.extracted_mcqs)} questions synced to Google Sheet.")
        else:
            task_mgr.add_log("WARNING", "Pipeline completed with notices.")

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

# Active State Notification Banner
if task_mgr.is_running:
    st.markdown(f"""
    <div style="background-color: #064E3B; border: 2px solid #10B981; border-radius: 12px; padding: 18px; margin-bottom: 20px;">
        <h4 style="margin:0 0 6px 0; color:#34D399;">⚡ Background Extraction IN PROGRESS on Cloud Server</h4>
        <p style="margin:0; font-size:1.0rem; color: #ECFDF5;">
            <b>Status:</b> {task_mgr.status_text}<br>
            <b>Extracted so far:</b> {len(task_mgr.extracted_mcqs)} questions<br>
            <b>Safe to Exit:</b> You can now safely <b>CLOSE this browser tab or SHUT DOWN your laptop</b>. 
            The cloud server will continue extracting and syncing all questions directly to your Google Sheet!
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.progress(task_mgr.progress_val)
    time.sleep(2)
    st.rerun()

# Upload & Action Row
col_up, col_action = st.columns([3, 1])

with col_up:
    uploaded_file = st.file_uploader(
        "📄 Select Question Bank PDF (Up to 50MB)",
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

        worker_thread = threading.Thread(
            target=background_worker,
            args=(pdf_content, fname, api_key, webhook_url, tab_name, selected_model, delay_sec, auto_switch, chunk_size),
            daemon=False
        )
        task_mgr.thread = worker_thread
        worker_thread.start()
        st.success("🚀 Background task launched on Cloud! You can now safely close your browser or turn off your device.")
        time.sleep(1)
        st.rerun()

# Tabs
tab_manifest, tab_preview, tab_logs, tab_download = st.tabs([
    "🗺️ Chapters & Sets Manifest",
    "👁 Live MCQ Stream",
    "💻 Live Engine Logs",
    "📥 Download Backups"
])

# 1. Manifest Tab
with tab_manifest:
    if task_mgr.manifest:
        st.subheader(f"Audited Chapters & Sets ({len(task_mgr.manifest)} Total)")
        df_manifest = pd.DataFrame(task_mgr.manifest)
        cols_to_show = ["chapter", "topic", "set", "page_start", "page_end", "total_questions", "first_question_preview", "last_question_preview"]
        available_cols = [c for c in cols_to_show if c in df_manifest.columns]
        st.dataframe(df_manifest[available_cols], use_container_width=True)
    else:
        st.info("Chapter, Topic, and Set mapping will automatically display here once extraction starts.")

# 2. Live Preview Tab
with tab_preview:
    count = len(task_mgr.extracted_mcqs)
    st.subheader(f"Extracted Questions Stream ({count} Captured)")
    if task_mgr.extracted_mcqs:
        for item in reversed(task_mgr.extracted_mcqs[-30:]):
            st.markdown(f"""
            <div class="mcq-card">
                <span class="mcq-tag">#{item.get('Serial', '')} • {item.get('Set', '')}</span>
                <div class="mcq-q"><b>প্রশ্ন:</b> {item.get('Question', '')}</div>
                <div class="opt-grid">
                    <div class="opt-box"><b>(A)</b> {item.get('Option-A', '')}</div>
                    <div class="opt-box"><b>(B)</b> {item.get('Option-B', '')}</div>
                    <div class="opt-box"><b>(C)</b> {item.get('Option-C', '')}</div>
                    <div class="opt-box"><b>(D)</b> {item.get('Option-D', '')}</div>
                </div>
                <div>
                    <span class="ans-badge">✓ সঠিক উত্তর: {item.get('Answer', '')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No questions in stream yet. Upload a PDF and click 'Start Background Extraction'.")

# 3. Logs Tab
with tab_logs:
    st.subheader("System Execution Log")
    st.text_area("Live Terminal", value="\n".join(task_mgr.logs[-250:]), height=380)

# 4. Download Tab
with tab_download:
    st.subheader("📥 Export Local Backup Files")
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
        st.info("Questions will be available for download once extracted.")
