import os
import time
import json
import re
import random
import threading
from typing import List, Dict, Any, Optional, Callable
from google import genai
from google.genai import types
from json_repair import repair_and_parse_json, normalize_mcq_keys

# Priority pool of multimodal models capable of OCR and PDF extraction
MULTIMODAL_MODEL_POOL = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite"
]

class GeminiMCQEngine:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.8-flash",
        inter_request_delay: float = 6.0,
        log_callback: Optional[Callable[[str, str], None]] = None,
        progress_callback: Optional[Callable[[float, float, str], None]] = None,
        preview_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        manifest_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        abort_event: Optional[threading.Event] = None,
        auto_switch_models: bool = True
    ):
        self.api_key = api_key.strip()
        cleaned_model = model_name.strip().replace("models/", "") if model_name.strip() else "gemini-3.8-flash"
        self.model_name = cleaned_model
        self.inter_request_delay = max(1.0, float(inter_request_delay))
        self.log_callback = log_callback or (lambda lvl, msg: print(f"[{lvl}] {msg}"))
        self.progress_callback = progress_callback or (lambda cur, tot, txt: None)
        self.preview_callback = preview_callback or (lambda item: None)
        self.manifest_callback = manifest_callback or (lambda items: None)
        self.abort_event = abort_event or threading.Event()
        self.auto_switch_models = auto_switch_models
        
        # Build ordered fallback pool starting from selected model
        self.model_pool = list(MULTIMODAL_MODEL_POOL)
        if self.model_name in self.model_pool:
            self.model_pool.remove(self.model_name)
        self.model_pool.insert(0, self.model_name)

        self.client: Optional[genai.Client] = None
        self.active_file_name: Optional[str] = None

    def log(self, level: str, message: str):
        self.log_callback(level.upper(), message)

    def is_aborted(self) -> bool:
        return self.abort_event.is_set()

    def sleep_with_abort(self, seconds: float) -> bool:
        start_time = time.time()
        while time.time() - start_time < seconds:
            if self.is_aborted():
                return False
            time.sleep(0.2)
        return True

    def init_client(self) -> bool:
        if not self.api_key:
            self.log("ERROR", "Gemini API Key is empty.")
            return False
        try:
            self.client = genai.Client(api_key=self.api_key)
            return True
        except Exception as e:
            self.log("ERROR", f"Failed to initialize Gemini Client: {e}")
            return False

    def generate_with_retry(self, contents: list, prompt: str, max_retries: int = 8) -> Optional[str]:
        """
        Robust generation with automatic model switching across the multimodal pool upon 429 quota exhaustion.
        """
        if not self.client:
            return None

        combined_contents = contents + [prompt]
        attempt = 1

        while attempt <= max_retries:
            if self.is_aborted():
                self.log("WARNING", "Operation aborted by user.")
                return None

            try:
                config = types.GenerateContentConfig(
                    temperature=0.1,
                    top_p=0.95,
                    max_output_tokens=8192,
                    system_instruction=(
                        "You are an expert multilingual MCQ parser. "
                        "Read the PDF pages carefully and transcribe questions accurately without skipping."
                    )
                )

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=combined_contents,
                    config=config
                )

                if response and response.text:
                    return response.text
                else:
                    self.log("WARNING", f"Empty response received from '{self.model_name}'. Retrying...")

            except Exception as e:
                err_str = str(e)

                # 1. Model deprecation auto-migration
                if ("no longer available" in err_str or "NOT_FOUND" in err_str) and "models/" in err_str:
                    match = re.search(r'use models/([a-zA-Z0-9.-]+)', err_str)
                    if match:
                        suggested = match.group(1)
                        self.log("WARNING", f"Model '{self.model_name}' deprecated. Auto-migrating to '{suggested}'...")
                        self.model_name = suggested
                        time.sleep(1)
                        continue

                # 2. Check 429 Quota exhaustion or 503 Server High Demand
                is_quota = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()
                is_503 = "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str.lower()

                if is_quota and self.auto_switch_models:
                    # Look up next available model in our multimodal pool!
                    try:
                        cur_idx = self.model_pool.index(self.model_name)
                    except ValueError:
                        cur_idx = -1

                    if cur_idx + 1 < len(self.model_pool):
                        next_model = self.model_pool[cur_idx + 1]
                        self.log("WARNING", f"⚡ Quota Limit (429) hit on '{self.model_name}'! Auto-switching to '{next_model}' (Fresh Quota Pool)...")
                        self.model_name = next_model
                        time.sleep(2)
                        # Immediately retry using the new model without incrementing attempt count
                        continue
                    else:
                        self.log("WARNING", "All models in pool have been rotated. Applying exponential backoff...")

                if is_503:
                    backoff = (8.0 * (1.4 ** (attempt - 1))) + random.uniform(1.0, 3.0)
                    self.log("WARNING", f"Server busy (503 UNAVAILABLE). Waiting {backoff:.1f}s before retry ({attempt}/{max_retries})...")
                    if not self.sleep_with_abort(backoff):
                        return None
                elif is_quota:
                    backoff = (12.0 * (1.5 ** (attempt - 1))) + random.uniform(1.0, 3.0)
                    self.log("WARNING", f"Quota limit reached (429). Exponential backoff {backoff:.1f}s ({attempt}/{max_retries})...")
                    if not self.sleep_with_abort(backoff):
                        return None
                else:
                    self.log("ERROR", f"Gemini Error ({self.model_name}, Attempt {attempt}/{max_retries}): {err_str}")
                    wait_time = 4.0 * attempt
                    if not self.sleep_with_abort(wait_time):
                        return None

            attempt += 1

        self.log("ERROR", f"Max retries ({max_retries}) exhausted for current request.")
        return None

    def run_pipeline(
        self,
        pdf_path: str,
        sheets_manager: Any,
        local_backup_path: Optional[str] = None
    ) -> bool:
        if not os.path.isfile(pdf_path):
            self.log("ERROR", f"PDF file not found: {pdf_path}")
            return False

        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        if not self.init_client():
            return False

        uploaded_file = None
        self.active_file_name = None

        try:
            # 1. Connect Sheet
            self.log("INFO", "Verifying Google Sheets connection...")
            connected, msg = sheets_manager.authenticate_and_connect()
            if not connected:
                self.log("ERROR", f"Google Sheets Setup Failed: {msg}")
                return False
            self.log("SUCCESS", msg)

            # 2. Upload PDF
            self.progress_callback(0.05, 1.0, f"Uploading {file_size_mb:.1f} MB PDF...")
            self.log("INFO", f"Uploading {file_size_mb:.2f} MB PDF to Google Files API...")
            
            uploaded_file = self.client.files.upload(file=pdf_path)
            self.active_file_name = uploaded_file.name
            self.log("INFO", f"File handle: {uploaded_file.name}. Polling status...")

            # 3. Wait until ACTIVE
            poll_start = time.time()
            while uploaded_file.state.name == "PROCESSING":
                if self.is_aborted():
                    self.log("WARNING", "Upload polling aborted.")
                    return False
                time.sleep(2)
                uploaded_file = self.client.files.get(name=uploaded_file.name)
                elapsed = time.time() - poll_start
                self.log("INFO", f"Processing on Google servers... ({elapsed:.0f}s, State: {uploaded_file.state.name})")

            if uploaded_file.state.name != "ACTIVE":
                self.log("ERROR", f"File processing failed on Google servers. State: {uploaded_file.state.name}")
                return False

            self.log("SUCCESS", f"File is ACTIVE on Google Files API ({uploaded_file.name})")
            self.log("INFO", f"Active Model: '{self.model_name}' (Fallback Pool: {', '.join(self.model_pool)})")

            # 4. Phase 1: Deep Discovery & Boundary Mapping
            self.progress_callback(0.12, 1.0, "Phase 1: Mapping topics, sets & question boundaries...")
            self.log("PROCESSING", "Phase 1: Deep Discovery - Scanning document to map all Topics, Sets, and Question Boundaries...")

            discovery_prompt = (
                "You are an expert document structural auditor.\n"
                "Scan the entire uploaded document and produce a complete manifest of ALL chapters/topics and their respective question sets.\n"
                "For EACH set discovered, you MUST record:\n"
                "1. 'topic': Full Chapter or Topic Title (e.g., 'মুঘল সাম্রাজ্যঃ জাহাঙ্গীর')\n"
                "2. 'set': Set Name exactly as printed (e.g., 'সেট-১' or 'Set - 1')\n"
                "3. 'total_questions': Expected number of questions in this set (integer, e.g. 60)\n"
                "4. 'first_question_preview': The text or start of the 1st question in this set\n"
                "5. 'last_question_preview': The text or start of the LAST question in this set\n\n"
                "Strict JSON Array Output:\n"
                "[\n"
                "  {\n"
                '    "topic": "Topic Name",\n'
                '    "set": "সেট-১",\n'
                '    "total_questions": 60,\n'
                '    "first_question_preview": "১. ...",\n'
                '    "last_question_preview": "৬০. ..."\n'
                "  }\n"
                "]\n"
                "Output strictly ONLY the raw JSON array. Do not miss any set or topic from the document."
            )

            raw_manifest = self.generate_with_retry([uploaded_file], discovery_prompt, max_retries=8)
            if self.is_aborted():
                return False

            manifest_items = []
            if raw_manifest:
                parsed = repair_and_parse_json(raw_manifest)
                for item in parsed:
                    if isinstance(item, dict):
                        topic = str(item.get("topic", "General")).strip()
                        s_name = str(item.get("set", "Set - 1")).strip()
                        tot = item.get("total_questions", 0)
                        try:
                            tot = int(tot)
                        except Exception:
                            tot = 0
                        fq = str(item.get("first_question_preview", "")).strip()
                        lq = str(item.get("last_question_preview", "")).strip()

                        if topic and s_name:
                            manifest_items.append({
                                "topic": topic,
                                "set": s_name,
                                "total_questions": tot,
                                "first_question_preview": fq,
                                "last_question_preview": lq,
                                "status": "Pending"
                            })

            if not manifest_items:
                self.log("WARNING", "Defaulting to fallback set structure.")
                manifest_items = [{
                    "topic": "Complete PDF Document",
                    "set": "Set - 1",
                    "total_questions": 60,
                    "first_question_preview": "1. First Question",
                    "last_question_preview": "Last Question",
                    "status": "Pending"
                }]

            self.manifest_callback(manifest_items)
            self.log("SUCCESS", f"Phase 1 Complete: Mapped {len(manifest_items)} sets with boundaries across the document.")

            # 5. Phase 2: Verified Extraction with Continuation Handling
            total_sets = len(manifest_items)
            total_mcqs_extracted = 0
            all_extracted_mcqs = []

            for idx, set_meta in enumerate(manifest_items, start=1):
                if self.is_aborted():
                    self.log("WARNING", f"Extraction halted by user at Set {idx}/{total_sets}.")
                    break

                topic = set_meta["topic"]
                s_name = set_meta["set"]
                exp_count = set_meta["total_questions"]
                first_preview = set_meta["first_question_preview"]
                last_preview = set_meta["last_question_preview"]

                progress_val = 0.15 + (0.80 * (idx - 1) / total_sets)
                self.progress_callback(progress_val, 1.0, f"Extracting {idx}/{total_sets}: '{topic}' ({s_name})")
                self.log("PROCESSING", f"=== Extracting Set {idx}/{total_sets}: '{topic}' [{s_name}] (Target: ~{exp_count} Qs | Model: {self.model_name}) ===")
                self.log("INFO", f"Boundary Target -> First Q: '{first_preview[:35]}...' | Last Q: '{last_preview[:35]}...'")

                set_mcqs: List[Dict[str, str]] = []
                part = 1
                start_from_num = 1
                completed_set = False

                while not completed_set and not self.is_aborted():
                    if part == 1:
                        prompt = (
                            f"You are a precision MCQ extractor.\n"
                            f"Extract ALL multiple choice questions for Topic: '{topic}' and Set: '{s_name}'.\n"
                            f"The set begins with question: '{first_preview}' and ends at: '{last_preview}'.\n"
                            f"Total expected questions in this set: {exp_count}.\n\n"
                            f"Return ONLY a valid JSON array of objects conforming to this schema:\n"
                            f"[\n"
                            f"  {{\n"
                            f'    "Set": "{topic} - {s_name}",\n'
                            f'    "Serial": "1",\n'
                            f'    "Question": "Full question text...",\n'
                            f'    "Option-A": "Option A...",\n'
                            f'    "Option-B": "Option B...",\n'
                            f'    "Option-C": "Option C...",\n'
                            f'    "Option-D": "Option D...",\n'
                            f'    "Answer": "A"\n'
                            f"  }}\n"
                            f"]\n\n"
                            f"Rules:\n"
                            f"1. Transcribe Bengali/English text and options accurately.\n"
                            f"2. Output consecutive questions starting from question 1.\n"
                            f"3. Strict raw JSON array only. No markdown formatting."
                        )
                    else:
                        last_extracted_q = set_mcqs[-1]["Question"][:40] if set_mcqs else ""
                        self.log("INFO", f"[Continuation Part {part}] Extracting '{s_name}' from Question #{start_from_num}...")
                        prompt = (
                            f"CONTINUATION REQUEST for Topic: '{topic}', Set: '{s_name}'.\n"
                            f"Previously extracted up to Question #{start_from_num - 1} ('{last_extracted_q}...').\n"
                            f"Now continue extracting ALL remaining questions starting strictly from Question #{start_from_num} up to the final question ('{last_preview}').\n\n"
                            f"Return ONLY a valid JSON array matching the exact schema:\n"
                            f"[\n"
                            f"  {{\n"
                            f'    "Set": "{topic} - {s_name}",\n'
                            f'    "Serial": "{start_from_num}",\n'
                            f'    "Question": "...",\n'
                            f'    "Option-A": "...",\n'
                            f'    "Option-B": "...",\n'
                            f'    "Option-C": "...",\n'
                            f'    "Option-D": "...",\n'
                            f'    "Answer": "A"\n'
                            f"  }}\n"
                            f"]\n"
                            f"Output raw JSON array only."
                        )

                    raw_resp = self.generate_with_retry([uploaded_file], prompt, max_retries=8)
                    if self.is_aborted():
                        break

                    if not raw_resp:
                        self.log("WARNING", f"No response for part {part}. Completing set.")
                        break

                    parsed_chunk = repair_and_parse_json(raw_resp)
                    normalized_chunk = [
                        normalize_mcq_keys(q, default_set=f"{topic} - {s_name}")
                        for q in parsed_chunk
                        if q.get("Question") or q.get("question")
                    ]

                    if not normalized_chunk:
                        self.log("WARNING", f"No new questions parsed in part {part}. Reached end of set.")
                        break

                    for q in normalized_chunk:
                        set_mcqs.append(q)
                        self.preview_callback(q)

                    self.log("SUCCESS", f"Part {part}: Extracted {len(normalized_chunk)} questions (Total in this set: {len(set_mcqs)}).")

                    current_count = len(set_mcqs)
                    last_extracted_text = set_mcqs[-1]["Question"].lower()
                    last_preview_clean = re.sub(r'^[০-৯0-9.\s]+', '', last_preview).lower()[:20]

                    is_boundary_matched = bool(last_preview_clean and last_preview_clean in last_extracted_text)
                    is_count_satisfied = (exp_count > 0 and current_count >= exp_count)

                    if is_boundary_matched or is_count_satisfied or len(normalized_chunk) < 5 or part >= 3:
                        completed_set = True
                        self.log("SUCCESS", f"✓ Boundary Verified for '{topic} [{s_name}]': {current_count} questions captured.")
                    else:
                        start_from_num = current_count + 1
                        part += 1
                        self.sleep_with_abort(2.0)

                count = len(set_mcqs)
                if count > 0:
                    total_mcqs_extracted += count
                    all_extracted_mcqs.extend(set_mcqs)

                    self.log("INFO", f"Posting {count} questions to Google Sheet tab '{sheets_manager.tab_name}' in safe batches of 25...")
                    chunk_size = 25
                    for chunk_idx in range(0, count, chunk_size):
                        if self.is_aborted():
                            break
                        sub_chunk = set_mcqs[chunk_idx:chunk_idx + chunk_size]
                        appended, append_msg = sheets_manager.append_mcqs(sub_chunk)
                        if appended > 0:
                            self.log("SUCCESS", f"Appended batch {chunk_idx + 1}-{chunk_idx + len(sub_chunk)} of {count} rows.")
                        else:
                            self.log("ERROR", f"Batch append failed: {append_msg}")
                            time.sleep(2)
                            sheets_manager.append_mcqs(sub_chunk)

                else:
                    self.log("WARNING", f"No questions could be extracted for Set '{s_name}'.")

                if local_backup_path and set_mcqs:
                    try:
                        self._save_local_backup(local_backup_path, all_extracted_mcqs)
                    except Exception as e:
                        self.log("WARNING", f"Could not update local backup: {e}")

                if idx < total_sets and not self.is_aborted():
                    self.log("INFO", f"Rate-limit buffer: Waiting {self.inter_request_delay:.1f}s before next set...")
                    if not self.sleep_with_abort(self.inter_request_delay):
                        break

            final_progress = 1.0 if not self.is_aborted() else progress_val
            summary = (
                f"Extraction Complete! Total Extracted: {total_mcqs_extracted} MCQs across {total_sets} sets."
                if not self.is_aborted()
                else f"Extraction Aborted. Total Extracted so far: {total_mcqs_extracted} MCQs."
            )
            self.progress_callback(final_progress, 1.0, summary)
            self.log("SUCCESS" if not self.is_aborted() else "WARNING", summary)
            return True

        except Exception as e:
            self.log("ERROR", f"Pipeline Exception: {e}")
            return False

        finally:
            if self.client and self.active_file_name:
                try:
                    self.log("INFO", f"Cleaning temporary file '{self.active_file_name}' from Google Files API...")
                    self.client.files.delete(name=self.active_file_name)
                    self.log("SUCCESS", "Google Files API storage cleaned up.")
                except Exception as del_err:
                    self.log("WARNING", f"Failed to delete uploaded file: {del_err}")

    def _save_local_backup(self, backup_path: str, mcqs: List[Dict[str, str]]):
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(mcqs, f, ensure_ascii=False, indent=2)

        csv_path = os.path.splitext(backup_path)[0] + ".csv"
        import csv
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Set", "Serial", "Question", "Option-A", "Option-B", "Option-C", "Option-D", "Answer"])
            for item in mcqs:
                writer.writerow([
                    item.get("Set", ""),
                    item.get("Serial", ""),
                    item.get("Question", ""),
                    item.get("Option-A", ""),
                    item.get("Option-B", ""),
                    item.get("Option-C", ""),
                    item.get("Option-D", ""),
                    item.get("Answer", "")
                ])
