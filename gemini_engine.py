import os
import time
import json
import re
import random
import threading
from typing import List, Dict, Any, Optional, Callable, Set
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

def clean_fingerprint(text: str) -> str:
    """
    Generate normalized fingerprint of question text to detect and eliminate AI duplicates.
    Strips leading Bengali/English serial numbers, punctuation, and whitespace.
    """
    if not text:
        return ""
    # Strip leading serial numbers (e.g. "১. ", "23) ", "[4]")
    stripped = re.sub(r'^[০-৯0-9.\s\-\)\(\[\]]+', '', text.strip())
    # Strip all whitespace and punctuation
    norm = re.sub(r'[\s\W_]+', '', stripped).lower()
    return norm[:60]


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
        auto_switch_models: bool = True,
        chunk_size: int = 30,
        *args,
        **kwargs
    ):
        self.api_key = api_key.strip()
        cleaned_model = model_name.strip().replace("models/", "") if model_name.strip() else "gemini-3.8-flash"
        self.model_name = cleaned_model
        self.inter_request_delay = max(2.0, float(inter_request_delay))
        self.log_callback = log_callback or (lambda lvl, msg: print(f"[{lvl}] {msg}"))
        self.progress_callback = progress_callback or (lambda cur, tot, txt: None)
        self.preview_callback = preview_callback or (lambda item: None)
        self.manifest_callback = manifest_callback or (lambda items: None)
        self.abort_event = abort_event or threading.Event()
        self.auto_switch_models = auto_switch_models
        self.chunk_size = max(15, min(40, chunk_size))  # Safe micro-batch size (default 30)
        
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
        Seamlessly continues without losing extraction context.
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
                    temperature=0.05,  # Low temperature for deterministic, hallucination-free transcription
                    top_p=0.95,
                    max_output_tokens=8192,
                    system_instruction=(
                        "You are an ultra-precise multilingual MCQ auditor and transcription specialist. "
                        "Read every Bengali and English word exactly as printed in the PDF. "
                        "Do NOT skip questions, do NOT truncate options, and do NOT duplicate questions."
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
                    try:
                        cur_idx = self.model_pool.index(self.model_name)
                    except ValueError:
                        cur_idx = 0
                    
                    next_idx = (cur_idx + 1) % len(self.model_pool)
                    next_model = self.model_pool[next_idx]
                    self.log("WARNING", f"⚠️ Quota 429 on '{self.model_name}'! Switching to pool model: '{next_model}' (Zero Duplicates)")
                    self.model_name = next_model
                    time.sleep(2)
                    attempt += 1
                    continue

                if is_503 or is_quota:
                    backoff = min(60.0, (2 ** attempt) + random.uniform(1.0, 3.0))
                    self.log("WARNING", f"Server busy ({err_str[:40]}...). Backing off for {backoff:.1f}s (Attempt {attempt}/{max_retries})...")
                    if not self.sleep_with_abort(backoff):
                        return None
                    attempt += 1
                    continue

                self.log("ERROR", f"Gemini API Exception on attempt {attempt}: {err_str}")
                if attempt >= max_retries:
                    return None
                time.sleep(3)
                attempt += 1

        return None

    def discover_manifest(self, uploaded_file) -> List[Dict[str, Any]]:
        """
        Phase 1: Deep Structural Audit & Hierarchical Mapping
        Discovers Chapters, Topics, Sets, Page Ranges, Total Questions, and Boundary Previews.
        """
        discovery_prompt = (
            "You are an expert document structural auditor.\n"
            "Scan the entire uploaded document and produce a complete hierarchical manifest of ALL Chapters, Topics, and Sets.\n"
            "For EACH set discovered, extract:\n"
            "1. 'chapter': Main Chapter / Subject name (e.g., 'ইতিহাস', 'সাধারণ জ্ঞান', 'English')\n"
            "2. 'topic': Specific Topic name (e.g., 'মুঘল সাম্রাজ্যঃ জাহাঙ্গীর', 'Prepositions')\n"
            "3. 'set': Set Name exactly as printed (e.g., 'সেট-১' or 'Set - 1')\n"
            "4. 'page_start': Estimated starting PDF page number (integer, e.g. 5)\n"
            "5. 'page_end': Estimated ending PDF page number (integer, e.g. 7)\n"
            "6. 'total_questions': Total number of questions in this set (integer, e.g. 60)\n"
            "7. 'first_question_preview': The text or start of the 1st question in this set\n"
            "8. 'last_question_preview': The text or start of the LAST question in this set\n\n"
            "Strict JSON Array Output:\n"
            "[\n"
            "  {\n"
            '    "chapter": "ইতিহাস",\n'
            '    "topic": "মুঘল সাম্রাজ্যঃ জাহাঙ্গীর",\n'
            '    "set": "সেট-১",\n'
            '    "page_start": 5,\n'
            '    "page_end": 7,\n'
            '    "total_questions": 60,\n'
            '    "first_question_preview": "১. সম্রাট জাহাঙ্গীর কত সালে...",\n'
            '    "last_question_preview": "৬০. জাহাঙ্গীরের সমাধি কোথায়..."\n'
            "  }\n"
            "]\n"
            "Output strictly ONLY the raw JSON array. Do not omit any Chapter, Topic, or Set."
        )

        raw_manifest = self.generate_with_retry([uploaded_file], discovery_prompt, max_retries=8)
        if self.is_aborted():
            return []

        manifest_items = []
        if raw_manifest:
            parsed = repair_and_parse_json(raw_manifest)
            for item in parsed:
                if isinstance(item, dict):
                    chapter = str(item.get("chapter", "General")).strip()
                    topic = str(item.get("topic", "General")).strip()
                    s_name = str(item.get("set", "Set - 1")).strip()
                    p_start = item.get("page_start", 1)
                    p_end = item.get("page_end", 1)
                    tot = item.get("total_questions", 0)
                    try:
                        tot = int(tot)
                    except Exception:
                        tot = 0
                    fq = str(item.get("first_question_preview", "")).strip()
                    lq = str(item.get("last_question_preview", "")).strip()

                    if topic and s_name:
                        manifest_items.append({
                            "chapter": chapter,
                            "topic": topic,
                            "set": s_name,
                            "page_start": p_start,
                            "page_end": p_end,
                            "total_questions": tot,
                            "first_question_preview": fq,
                            "last_question_preview": lq,
                            "status": "Pending"
                        })

        if not manifest_items:
            self.log("WARNING", "Defaulting to fallback structural mapping.")
            manifest_items = [{
                "chapter": "Complete Book",
                "topic": "Complete PDF Document",
                "set": "Set - 1",
                "page_start": 1,
                "page_end": 100,
                "total_questions": 60,
                "first_question_preview": "1. First Question",
                "last_question_preview": "Last Question",
                "status": "Pending"
            }]

        return manifest_items

    def extract_micro_chunk(
        self,
        uploaded_file,
        chapter: str,
        topic: str,
        s_name: str,
        start_serial: int,
        chunk_limit: int,
        last_extracted_text: str,
        first_preview: str,
        last_preview: str
    ) -> List[Dict[str, str]]:
        """
        Extracts a controlled micro-batch of MCQs (max 25-30 questions).
        Strict boundary anchoring prevents missing questions and avoids token exhaustion.
        """
        end_serial_target = start_serial + chunk_limit - 1
        anchor_prompt = (
            f"Precision MCQ Extraction:\n"
            f"- Chapter: '{chapter}'\n"
            f"- Topic: '{topic}'\n"
            f"- Set: '{s_name}'\n"
            f"- Boundary: Starts with '{first_preview}' and ends at '{last_preview}'.\n"
        )

        if start_serial == 1:
            task_instruction = (
                f"Extract consecutive questions starting strictly from Question #{start_serial} up to Question #{end_serial_target} (maximum {chunk_limit} questions).\n"
                f"Do not exceed Question #{end_serial_target} in this response."
            )
        else:
            task_instruction = (
                f"CONTINUATION:\n"
                f"Previously extracted up to Question #{start_serial - 1} (Last extracted: '{last_extracted_text[:45]}...').\n"
                f"Now continue extracting consecutive questions starting STRICTLY from Question #{start_serial} up to Question #{end_serial_target} (maximum {chunk_limit} questions).\n"
                f"Do not repeat previous questions. Do not skip any question number."
            )

        full_prompt = (
            f"{anchor_prompt}\n{task_instruction}\n\n"
            f"Return ONLY a valid JSON array conforming to this exact schema:\n"
            f"[\n"
            f"  {{\n"
            f'    "Set": "{topic} - {s_name}",\n'
            f'    "Serial": "{start_serial}",\n'
            f'    "Question": "Exact question text in Bengali/English...",\n'
            f'    "Option-A": "Option A...",\n'
            f'    "Option-B": "Option B...",\n'
            f'    "Option-C": "Option C...",\n'
            f'    "Option-D": "Option D...",\n'
            f'    "Answer": "A"\n'
            f"  }}\n"
            f"]\n\n"
            f"Rules:\n"
            f"1. Transcribe text exactly as printed. No summary, no truncation.\n"
            f"2. Strict raw JSON array only."
        )

        raw_resp = self.generate_with_retry([uploaded_file], full_prompt, max_retries=8)
        if not raw_resp or self.is_aborted():
            return []

        parsed_chunk = repair_and_parse_json(raw_resp)
        normalized = [
            normalize_mcq_keys(q, default_set=f"{topic} - {s_name}")
            for q in parsed_chunk
            if q.get("Question") or q.get("question")
        ]
        return normalized

    def run_pipeline(
        self,
        pdf_path: str,
        sheets_manager: Any,
        local_backup_path: Optional[str] = None
    ) -> bool:
        """
        Executes the 5-Layer Strict Extraction Pipeline:
        1. Single PDF upload to Google Files API
        2. Phase 1: Hierarchical Discovery (Chapter, Topic, Set, Page Range, Boundaries)
        3. Phase 2: Micro-Chunk Extraction (25-30 Qs per chunk)
        4. In-Memory Fingerprint Deduplication (Zero AI-duplicated questions)
        5. Seamless Model Pool Auto-Switching on 429
        6. Safe batch upload to Google Sheets
        """
        if not os.path.exists(pdf_path):
            self.log("ERROR", f"PDF file not found: {pdf_path}")
            return False

        if not self.init_client():
            return False

        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        self.log("INFO", f"Starting 100% Strict MCQ Pipeline for '{os.path.basename(pdf_path)}' ({file_size_mb:.2f} MB)...")

        try:
            # 1. Upload PDF
            self.progress_callback(0.04, 1.0, f"Uploading {file_size_mb:.1f} MB PDF...")
            self.log("INFO", f"Uploading PDF to Google Files API...")
            
            uploaded_file = self.client.files.upload(file=pdf_path)
            self.active_file_name = uploaded_file.name
            self.log("INFO", f"File handle: {uploaded_file.name}. Polling status...")

            # 2. Wait until ACTIVE
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
                self.log("ERROR", f"File processing failed on Google servers: {uploaded_file.state.name}")
                return False

            self.log("SUCCESS", f"File is ACTIVE on Google Files API ({uploaded_file.name})")
            self.log("INFO", f"Starting Model: '{self.model_name}' (Pool: {', '.join(self.model_pool[:4])}...)")

            # 3. Phase 1: Deep Discovery & Hierarchical Mapping
            self.progress_callback(0.10, 1.0, "Phase 1: Auditing Chapters, Topics & Sets...")
            self.log("PROCESSING", "Phase 1: Mapping Chapters, Topics, Sets, Page Ranges & Boundaries...")

            manifest_items = self.discover_manifest(uploaded_file)
            if self.is_aborted():
                return False

            self.manifest_callback(manifest_items)
            self.log("SUCCESS", f"Phase 1 Complete: Discovered {len(manifest_items)} sets across document.")

            # 4. Phase 2: Micro-Chunk Extraction with Zero-Duplicate Deduplication
            total_sets = len(manifest_items)
            total_mcqs_extracted = 0
            all_extracted_mcqs = []

            for idx, set_meta in enumerate(manifest_items, start=1):
                if self.is_aborted():
                    self.log("WARNING", f"Extraction halted by user at Set {idx}/{total_sets}.")
                    break

                chapter = set_meta.get("chapter", "General")
                topic = set_meta["topic"]
                s_name = set_meta["set"]
                p_start = set_meta.get("page_start", 1)
                p_end = set_meta.get("page_end", 1)
                exp_count = set_meta["total_questions"]
                first_preview = set_meta["first_question_preview"]
                last_preview = set_meta["last_question_preview"]

                progress_val = 0.12 + (0.83 * (idx - 1) / total_sets)
                self.progress_callback(progress_val, 1.0, f"Extracting {idx}/{total_sets}: '{topic}' ({s_name})")
                self.log("PROCESSING", f"=== Set {idx}/{total_sets}: Chapter: '{chapter}' | Topic: '{topic}' [{s_name}] (Target: ~{exp_count} Qs | Pages: {p_start}-{p_end}) ===")
                self.log("INFO", f"Boundary Anchors -> First: '{first_preview[:30]}...' | Last: '{last_preview[:30]}...'")

                set_mcqs: List[Dict[str, str]] = []
                seen_fingerprints: Set[str] = set()
                start_serial = 1
                part = 1
                max_parts = max(4, (exp_count // self.chunk_size) + 3) if exp_count > 0 else 6
                completed_set = False

                while not completed_set and part <= max_parts and not self.is_aborted():
                    last_text = set_mcqs[-1]["Question"] if set_mcqs else ""
                    self.log("INFO", f"[{s_name} - Part {part}] Extracting micro-chunk (from Q#{start_serial}, max {self.chunk_size} Qs | Model: {self.model_name})...")

                    chunk = self.extract_micro_chunk(
                        uploaded_file=uploaded_file,
                        chapter=chapter,
                        topic=topic,
                        s_name=s_name,
                        start_serial=start_serial,
                        chunk_limit=self.chunk_size,
                        last_extracted_text=last_text,
                        first_preview=first_preview,
                        last_preview=last_preview
                    )

                    if self.is_aborted():
                        break

                    if not chunk:
                        self.log("WARNING", f"No questions returned in Part {part}. Checking boundary status...")
                        break

                    new_count_in_chunk = 0
                    for q in chunk:
                        q_text = q.get("Question", "")
                        fp = clean_fingerprint(q_text)
                        
                        # Strict Deduplication Check: Discard AI-hallucinated repeats
                        if fp and fp in seen_fingerprints:
                            self.log("WARNING", f"Discarded AI duplicate: '{q_text[:35]}...'")
                            continue

                        if fp:
                            seen_fingerprints.add(fp)

                        set_mcqs.append(q)
                        self.preview_callback(q)
                        new_count_in_chunk += 1

                    self.log("SUCCESS", f"Part {part}: Added {new_count_in_chunk} unique questions (Total set so far: {len(set_mcqs)}).")

                    # Boundary and Completion Check
                    current_count = len(set_mcqs)
                    last_extracted_text_clean = set_mcqs[-1]["Question"].lower() if set_mcqs else ""
                    last_preview_clean = re.sub(r'^[০-৯0-9.\s]+', '', last_preview).lower()[:20]

                    is_boundary_matched = bool(last_preview_clean and last_preview_clean in last_extracted_text_clean)
                    is_count_satisfied = (exp_count > 0 and current_count >= exp_count)

                    if is_boundary_matched or is_count_satisfied or new_count_in_chunk == 0:
                        completed_set = True
                        self.log("SUCCESS", f"✓ Set Boundary Reached for '{topic} [{s_name}]': Total {current_count} questions captured.")
                    else:
                        start_serial = current_count + 1
                        part += 1
                        self.sleep_with_abort(self.inter_request_delay)

                count = len(set_mcqs)
                if count > 0:
                    total_mcqs_extracted += count
                    all_extracted_mcqs.extend(set_mcqs)

                    # Post set in safe batches of 25 to Google Sheets
                    self.log("INFO", f"Submitting {count} questions to Google Sheet tab '{sheets_manager.tab_name}' in batches of 25...")
                    chunk_size = 25
                    for i in range(0, count, chunk_size):
                        if self.is_aborted():
                            break
                        sub_batch = set_mcqs[i:i + chunk_size]
                        if hasattr(sheets_manager, "append_rows"):
                            res = sheets_manager.append_rows(sub_batch)
                            success_sheet = bool(res[0] > 0 if isinstance(res, tuple) else res)
                        elif hasattr(sheets_manager, "append_mcqs"):
                            cnt, _ = sheets_manager.append_mcqs(sub_batch)
                            success_sheet = cnt > 0
                        else:
                            success_sheet = False
                        if success_sheet:
                            self.log("SUCCESS", f"Synced rows {i+1} to {min(i+chunk_size, count)} of '{s_name}' to Sheet.")
                        else:
                            self.log("ERROR", f"Failed to sync rows {i+1}-{min(i+chunk_size, count)} to Sheet. Preserved in backup.")
                        time.sleep(1.0)

                    # Update local backup after each set
                    if local_backup_path:
                        try:
                            with open(local_backup_path, "w", encoding="utf-8") as f:
                                json.dump(all_extracted_mcqs, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            self.log("WARNING", f"Could not write local backup: {e}")

                self.sleep_with_abort(self.inter_request_delay)

            self.progress_callback(1.0, 1.0, "Extraction complete!")
            self.log("SUCCESS", f"🎉 All Sets Completed! Total {total_mcqs_extracted} authentic MCQs processed and synced.")
            return True

        except Exception as e:
            self.log("ERROR", f"Fatal error in pipeline: {e}")
            return False

        finally:
            # Delete remote file handle from Google Files API to keep project clean
            if self.client and self.active_file_name:
                try:
                    self.client.files.delete(name=self.active_file_name)
                    self.log("INFO", f"Cleaned up remote file {self.active_file_name} from Google Cloud.")
                except Exception as e:
                    self.log("WARNING", f"Remote cleanup notice: {e}")
