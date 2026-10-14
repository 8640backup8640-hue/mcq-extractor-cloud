import re
import os
import json
from typing import List, Dict, Any, Optional, Tuple
import requests
import gspread
from google.oauth2.service_account import Credentials

REQUIRED_HEADERS = [
    "Set", "Serial", "Question", "Option-A", "Option-B", "Option-C", "Option-D", "Answer"
]

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def extract_spreadsheet_id(input_str: str) -> str:
    """Extracts spreadsheet ID from raw ID or full Google Sheets URL."""
    clean_str = input_str.strip()
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", clean_str)
    if match:
        return match.group(1)
    return clean_str

class AppsScriptManager:
    """
    Zero-config Google Sheets exporter using Google Apps Script Webhook URL.
    No JSON files, No Google Cloud Console required.
    """
    def __init__(self, webhook_url: str, tab_name: str = "Sheet1"):
        self.webhook_url = webhook_url.strip()
        self.tab_name = tab_name.strip() if tab_name.strip() else "Sheet1"

    def authenticate_and_connect(self) -> Tuple[bool, str]:
        """Tests the Webhook URL connection and initializes header row."""
        if not self.webhook_url:
            return False, "Google Apps Script Webhook URL is empty."

        if not (self.webhook_url.startswith("http://") or self.webhook_url.startswith("https://")):
            return False, "Invalid Webhook URL. Must begin with https://script.google.com/..."

        payload = {
            "action": "ping",
            "tabName": self.tab_name,
            "headers": REQUIRED_HEADERS
        }

        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
                allow_redirects=True
            )

            if resp.status_code == 200:
                try:
                    res_data = resp.json()
                    msg = res_data.get("message", "Connected successfully.")
                    return True, f"Google Apps Script Webhook connected! {msg} (Tab: '{self.tab_name}')"
                except Exception:
                    # Sometimes apps script returns plain text
                    return True, f"Google Apps Script Webhook connected! (Tab: '{self.tab_name}')"
            else:
                return False, f"HTTP Error {resp.status_code} while connecting to Webhook URL: {resp.text[:200]}"

        except requests.exceptions.RequestException as e:
            return False, f"Failed to reach Google Apps Script Webhook: {e}"

    def append_mcqs(self, mcq_list: List[Dict[str, str]]) -> Tuple[int, str]:
        """Appends rows to Google Sheet via Google Apps Script Webhook."""
        if not mcq_list:
            return 0, "No MCQs to append."

        rows_to_append = []
        for item in mcq_list:
            row = [
                item.get("Set", ""),
                item.get("Serial", ""),
                item.get("Question", ""),
                item.get("Option-A", ""),
                item.get("Option-B", ""),
                item.get("Option-C", ""),
                item.get("Option-D", ""),
                item.get("Answer", "")
            ]
            rows_to_append.append(row)

        payload = {
            "action": "append",
            "tabName": self.tab_name,
            "headers": REQUIRED_HEADERS,
            "rows": rows_to_append
        }

        for attempt in range(1, 3):
            try:
                resp = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=120,
                    allow_redirects=True
                )

                if resp.status_code == 200:
                    return len(rows_to_append), f"Successfully posted {len(rows_to_append)} rows to Google Sheet ('{self.tab_name}')"
                else:
                    if attempt == 2:
                        return 0, f"Webhook returned HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                if attempt == 2:
                    return 0, f"Error sending data to Google Apps Script Webhook: {e}"
                time.sleep(2)
        return 0, "Failed after retries."

    def append_rows(self, mcq_list: List[Dict[str, str]]) -> bool:
        """Alias for append_mcqs for unified manager interface."""
        count, _ = self.append_mcqs(mcq_list)
        return count > 0

class SheetsManager:
    """Google Sheets integration using Service Account JSON via gspread."""
    def __init__(self, service_account_path: str, spreadsheet_id: str, tab_name: str = "Sheet1"):
        self.service_account_path = service_account_path.strip()
        self.spreadsheet_id = extract_spreadsheet_id(spreadsheet_id)
        self.tab_name = tab_name.strip() if tab_name.strip() else "Sheet1"
        self.client: Optional[gspread.Client] = None
        self.spreadsheet: Optional[gspread.Spreadsheet] = None
        self.worksheet: Optional[gspread.Worksheet] = None
        self.service_account_email: str = ""

    def authenticate_and_connect(self) -> Tuple[bool, str]:
        if not os.path.isfile(self.service_account_path):
            return False, f"Service Account file not found: {self.service_account_path}"

        if not self.spreadsheet_id:
            return False, "Google Spreadsheet ID cannot be empty."

        try:
            with open(self.service_account_path, 'r', encoding='utf-8') as f:
                sa_data = json.load(f)
                self.service_account_email = sa_data.get("client_email", "Unknown")
        except Exception as e:
            return False, f"Failed to read Service Account JSON: {e}"

        try:
            creds = Credentials.from_service_account_file(
                self.service_account_path,
                scopes=SCOPES
            )
            self.client = gspread.authorize(creds)
        except Exception as e:
            return False, f"Authentication error with credentials: {e}"

        try:
            self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
        except gspread.exceptions.SpreadsheetNotFound:
            return False, (
                f"Spreadsheet not found (ID: {self.spreadsheet_id}). "
                f"Make sure you shared the sheet with Editor permission to: {self.service_account_email}"
            )
        except gspread.exceptions.APIError as e:
            if "PERMISSION_DENIED" in str(e) or "403" in str(e):
                return False, (
                    f"Access Denied (403)! You MUST share the Google Sheet with Editor permission to: "
                    f"'{self.service_account_email}'"
                )
            return False, f"Google Sheets API Error: {e}"
        except Exception as e:
            return False, f"Failed to open spreadsheet: {e}"

        try:
            self.worksheet = self.spreadsheet.worksheet(self.tab_name)
        except gspread.exceptions.WorksheetNotFound:
            try:
                self.worksheet = self.spreadsheet.add_worksheet(title=self.tab_name, rows="1000", cols="20")
            except Exception:
                self.worksheet = self.spreadsheet.sheet1
                self.tab_name = self.worksheet.title

        try:
            self._ensure_headers()
        except Exception as e:
            return False, f"Failed to initialize header row in Sheet: {e}"

        return True, f"Connected to Sheet '{self.spreadsheet.title}' (Tab: '{self.worksheet.title}'). Service Email: {self.service_account_email}"

    def _ensure_headers(self):
        if not self.worksheet:
            return
        existing_values = self.worksheet.row_values(1)
        if not existing_values or [h.lower() for h in existing_values[:len(REQUIRED_HEADERS)]] != [h.lower() for h in REQUIRED_HEADERS]:
            if not existing_values:
                self.worksheet.insert_row(REQUIRED_HEADERS, 1, value_input_option='USER_ENTERED')
            else:
                cell_list = self.worksheet.range(1, 1, 1, len(REQUIRED_HEADERS))
                for idx, cell in enumerate(cell_list):
                    cell.value = REQUIRED_HEADERS[idx]
                self.worksheet.update_cells(cell_list, value_input_option='USER_ENTERED')

    def append_mcqs(self, mcq_list: List[Dict[str, str]]) -> Tuple[int, str]:
        if not self.worksheet:
            return 0, "Worksheet is not connected."

        if not mcq_list:
            return 0, "No MCQs to append."

        rows_to_append = []
        for item in mcq_list:
            row = [
                item.get("Set", ""),
                item.get("Serial", ""),
                item.get("Question", ""),
                item.get("Option-A", ""),
                item.get("Option-B", ""),
                item.get("Option-C", ""),
                item.get("Option-D", ""),
                item.get("Answer", "")
            ]
            rows_to_append.append(row)

        try:
            self.worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
            return len(rows_to_append), f"Successfully appended {len(rows_to_append)} rows to '{self.tab_name}'"
        except Exception as e:
            return 0, f"Error appending rows to Google Sheet: {e}"
