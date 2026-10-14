# ⚡ Gemini Smart MCQ Extractor (Cloud Edition)

A 100% online, cloud-ready Web Application built with **Streamlit** and the official **Google GenAI SDK** for automated MCQ extraction from PDF files directly into **Google Sheets**.

---

## 🌟 Key Features

- **100% Cloud Deployment**: Run on Streamlit Community Cloud (100% Free, no server or PC needed).
- **Mobile & Desktop Friendly**: Access your own web app URL from any smartphone, tablet, or PC.
- **Deep Boundary Mapping**: Automatically maps every topic, set, total expected questions, and first/last question boundaries.
- **Auto-Continuation**: Automatically queries remaining questions if Gemini reaches token output boundaries—guaranteeing zero missed questions!
- **Multimodal Model Auto-Switching Pool**: If quota limit (429) is reached on one model, the engine automatically cascades to the next multimodal model (`gemini-3.8-flash` ➔ `gemini-3.7-flash` ➔ `gemini-3.6-flash` ➔ `gemini-3.5-flash-lite` ➔ `gemini-3.1-flash-lite` ➔ `gemini-2.5-flash`) giving you 1,000+ daily free queries!
- **Safe Batch Google Sheets Posting**: Submits rows in safe batches of 25 to prevent Google Apps Script timeouts.
- **Fail-Safe Local Download**: Directly download extracted MCQs as `.csv` or `.json` right from the browser.

---

## 🚀 How to Deploy on GitHub & Streamlit Cloud (100% Free in 2 Minutes)

### Step 1: Upload to GitHub
1. Go to [GitHub.com](https://github.com/) and click **New Repository**.
2. Name it (e.g., `gemini-mcq-extractor`). Keep it **Public** (or Private).
3. Upload all files from this folder (`streamlit_app.py`, `gemini_engine.py`, `sheets_engine.py`, `json_repair.py`, `requirements.txt`, `.gitignore`, `README.md`) to the repository.

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `your-username/gemini-mcq-extractor`.
4. Main file path: `streamlit_app.py`.
5. Click **"Deploy!"**.

Within 1-2 minutes, you will get a permanent live web link (e.g. `https://your-mcq-app.streamlit.app`).

---

## 📱 How to Use from Phone or PC
1. Open your Streamlit web app link in any browser.
2. Upload your PDF file.
3. Click **"▶ Start Cloud Extraction"**.
4. You can see questions live in the **"👁 Live MCQ Preview"** tab, while the cloud automatically pushes every question into your Google Sheet!
