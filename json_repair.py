import json
import re
from typing import List, Dict, Any, Optional

def clean_markdown_fences(text: str) -> str:
    """Strip markdown code block wrappers like ```json ... ```"""
    text = text.strip()
    # Match ```json ... ``` or ``` ... ```
    pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
    match = re.match(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # If not full wrapper, find the first [ and last ]
    first_bracket = text.find('[')
    last_bracket = text.rfind(']')
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        return text[first_bracket:last_bracket+1].strip()
    
    return text

def repair_and_parse_json(text: str) -> List[Dict[str, Any]]:
    """
    Robustly parses JSON array of MCQ items from Gemini.
    Handles:
    - Markdown fences
    - Trailing commas
    - Truncated JSON where Gemini ran out of tokens
    - Extra text before or after JSON
    """
    if not text or not text.strip():
        return []

    cleaned = clean_markdown_fences(text)

    # 1. Direct parse attempt
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        elif isinstance(data, dict):
            # Sometimes model wraps it in {"questions": [...]} or {"mcqs": [...]}
            for key in ["questions", "mcqs", "data", "items", "Set"]:
                if key in data and isinstance(data[key], list):
                    return [d for d in data[key] if isinstance(d, dict)]
            return [data]
    except json.JSONDecodeError:
        pass

    # 2. Extract valid individual JSON objects {...} from the text using regex
    # Even if the whole array is truncated, we can extract all completed { ... } objects!
    extracted_items = []
    
    # Try regex finding completed objects
    # Pattern to match { "Set": ..., "Serial": ..., "Question": ..., ... }
    # We can match balanced braces or standard object chunks
    object_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}')
    for match in object_pattern.finditer(cleaned):
        chunk = match.group(0)
        try:
            # Clean common syntax errors in chunk like trailing commas before }
            chunk_clean = re.sub(r',\s*\}', '}', chunk)
            obj = json.loads(chunk_clean)
            if isinstance(obj, dict) and ("Question" in obj or "question" in obj or "Serial" in obj):
                extracted_items.append(obj)
        except Exception:
            continue

    if extracted_items:
        return extracted_items

    # 3. Truncation repair: Attempt to close open strings, objects, and arrays
    # Find start of array
    start_idx = cleaned.find('[')
    if start_idx != -1:
        sub = cleaned[start_idx:]
        # Remove trailing incomplete string/property
        # If it ends with an unclosed quote, close it or strip it back to last complete property
        # Step back to the last complete '}'
        last_obj_close = sub.rfind('}')
        if last_obj_close != -1:
            truncated_sub = sub[:last_obj_close+1] + "]"
            # Remove any trailing comma before ]
            truncated_sub = re.sub(r',\s*\]', ']', truncated_sub)
            try:
                data = json.loads(truncated_sub)
                if isinstance(data, list):
                    return [d for d in data if isinstance(d, dict)]
            except Exception:
                pass

    # 4. Fallback: Line-by-line / heuristic key extraction
    return []

def normalize_mcq_keys(item: Dict[str, Any], default_set: str = "Set - 1") -> Dict[str, str]:
    """
    Normalizes dictionary keys to match strict schema:
    Set, Serial, Question, Option-A, Option-B, Option-C, Option-D, Answer
    """
    key_map = {}
    for k, v in item.items():
        key_map[k.lower().strip()] = str(v).strip() if v is not None else ""

    set_val = key_map.get("set", default_set)
    serial_val = key_map.get("serial", key_map.get("sl", key_map.get("id", key_map.get("number", ""))))
    question_val = key_map.get("question", key_map.get("q", ""))
    
    opt_a = key_map.get("option-a", key_map.get("option_a", key_map.get("a", key_map.get("option a", ""))))
    opt_b = key_map.get("option-b", key_map.get("option_b", key_map.get("b", key_map.get("option b", ""))))
    opt_c = key_map.get("option-c", key_map.get("option_c", key_map.get("c", key_map.get("option c", ""))))
    opt_d = key_map.get("option-d", key_map.get("option_d", key_map.get("d", key_map.get("option d", ""))))
    
    answer_val = key_map.get("answer", key_map.get("ans", key_map.get("correct", "")))
    # Normalize answer (e.g. if answer is "Option-A" or "A. Option Text" -> extract "A")
    ans_match = re.search(r'\b([A-D]|ক|খ|গ|ঘ)\b', answer_val, re.IGNORECASE)
    if ans_match:
        ans_raw = ans_match.group(1).upper()
        # Convert Bengali letters if any
        bengali_to_eng = {"ক": "A", "খ": "B", "গ": "C", "ঘ": "D"}
        answer_val = bengali_to_eng.get(ans_raw, ans_raw)

    return {
        "Set": set_val,
        "Serial": serial_val,
        "Question": question_val,
        "Option-A": opt_a,
        "Option-B": opt_b,
        "Option-C": opt_c,
        "Option-D": opt_d,
        "Answer": answer_val
    }
