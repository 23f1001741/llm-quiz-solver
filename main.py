import sys
import asyncio
import os
import json
import traceback
import requests
import re
import pandas as pd
import numpy as np
import bs4 
import io
import base64
import hashlib
import mimetypes
import tempfile
from urllib.parse import urljoin, urlparse 
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from playwright.async_api import async_playwright
import google.generativeai as genai

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

app = FastAPI()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
STUDENT_EMAIL = os.environ.get("STUDENT_EMAIL")
STUDENT_SECRET = os.environ.get("STUDENT_SECRET")

if not GOOGLE_API_KEY:
    print("⚠️ WARNING: GOOGLE_API_KEY not found via Env Vars")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('models/gemini-2.0-flash')

class TaskPayload(BaseModel):
    email: str
    secret: str
    url: str

# ==============================================================================
# 1. THE FORENSIC GATHERER (BREADTH-FIRST STRATEGY)
# ==============================================================================
def transcribe_media(content: bytes, mime_type: str) -> str:
    """Uploads media to Gemini and gets a description."""
    try:
        suffix = ".mp3" if "audio" in mime_type else ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        myfile = genai.upload_file(tmp_path)
        prompt = "Transcribe this file exactly. If it contains instructions (like 'sum column X' or 'password is Y'), output ONLY the instruction."
        result = model.generate_content([myfile, prompt])
        
        os.unlink(tmp_path)
        return f"[TRANSCRIPT OF {mime_type}]: {result.text.strip()}"
    except Exception as e:
        return f"[ERROR TRANSCRIBING MEDIA]: {e}"

def inspect_csv(content: str) -> str:
    """Peeks at a CSV to get columns and first row."""
    try:
        df = pd.read_csv(io.StringIO(content))
        return f"[CSV STRUCTURE]: Columns={list(df.columns)}, First Row={df.iloc[0].to_dict()}"
    except Exception as e:
        return f"[CSV ERROR]: {e}"

def process_asset(link: str, content: bytes = None) -> str:
    """Helper to process a single asset URL (Audio/Image/CSV)."""
    try:
        ext = link.lower().split('.')[-1]
        
        if content is None:
            resp = requests.get(link, timeout=5)
            content = resp.content
            text_content = resp.text
        else:
            text_content = content.decode('utf-8', errors='ignore')

        if ext in ['mp3', 'wav']:
            return f"--- AUDIO FOUND: {link} ---\n" + transcribe_media(content, "audio/mp3")
        
        elif ext in ['png', 'jpg', 'jpeg']:
            return f"--- IMAGE FOUND: {link} ---\n" + transcribe_media(content, "image/png")
        
        elif ext == 'csv':
            return f"--- CSV FOUND: {link} ---\n" + inspect_csv(text_content)
        
        elif ext in ['json', 'txt']:
            return f"--- DATA FOUND: {link} ---\nCONTENT: {text_content[:1000]}"
            
    except Exception as e:
        return f"⚠️ Error processing {link}: {e}"
    return ""

async def gather_deep_context(page, base_url):
    """
    Scans Level 1 (Main Page + Assets) completely BEFORE scanning Level 2 (Sub-pages).
    """
    print("🕵️ GATHERING DEEP CONTEXT (Breadth-First Mode)...")
    
    # --- PHASE 1: LEVEL 1 TEXT ---
    html_content = await page.content()
    visible_text = await page.inner_text("body")
    soup = bs4.BeautifulSoup(html_content, 'html.parser')
    
    context_log = [f"=== LEVEL 1: MAIN PAGE TEXT ===\n{visible_text[:3000]}"]
    
    # Sort links into categories
    asset_links = set()
    page_links = set()
    
    for tag in soup.find_all(['a', 'script', 'img', 'source']):
        href = tag.get('href') or tag.get('src')
        if href:
            full = urljoin(base_url, href)
            ext = full.lower().split('.')[-1]
            
            if "s-anand.net" not in full and "localhost" not in full:
                continue

            # Identify Assets
            if ext in ['mp3', 'wav', 'png', 'jpg', 'csv', 'json', 'txt']:
                asset_links.add(full)
            # Identify Internal Pages (Not Assets, Not CSS/JS)
            elif full != base_url and 'http' in full and ext not in ['js', 'css']:
                 if len(full) < len(base_url) + 50: # Simple heuristics for relative pages
                    page_links.add(full)

    # --- PHASE 2: LEVEL 1 ASSETS (Process these FIRST) ---
    print(f"🔎 Level 1: Found {len(asset_links)} assets.")
    for link in asset_links:
        print(f"   📦 Processing L1 Asset: {link}")
        context_log.append(process_asset(link))

    # --- PHASE 3: LEVEL 2 PAGES (Process these LAST) ---
    print(f"🔎 Level 1: Found {len(page_links)} sub-pages. Diving in...")
    for link in page_links:
        print(f"   📄 Scanning L2 Page: {link}")
        try:
            resp = requests.get(link, timeout=4)
            sub_soup = bs4.BeautifulSoup(resp.content, 'html.parser')
            
            # 3A. Sub-Page Text
            clean_text = sub_soup.get_text(separator=' ', strip=True)
            context_log.append(f"=== LEVEL 2: SUB-PAGE TEXT ({link}) ===\n{clean_text[:1000]}")
            
            # 3B. Sub-Page Nested Assets
            sub_assets = sub_soup.find_all(['a', 'source'])
            for sub_tag in sub_assets:
                sub_href = sub_tag.get('href') or sub_tag.get('src')
                if sub_href:
                    full_sub = urljoin(link, sub_href)
                    sub_ext = full_sub.lower().split('.')[-1]
                    
                    if sub_ext in ['mp3', 'wav', 'csv', 'txt']:
                        print(f"      🎯 Found L2 Nested Asset: {full_sub}")
                        context_log.append(process_asset(full_sub))
                        
        except Exception as e:
            print(f"   ⚠️ Error scanning L2 page: {e}")

    return "\n\n".join(context_log)

# ==============================================================================
# 2. EXECUTION ENGINE
# ==============================================================================
def execute_generated_code(code_str: str):
    print("⚡ Executing Python code...")
    code_str = code_str.replace("```python", "").replace("```", "").strip()
    
    allowed_globals = {
        "pd": pd, "np": np, "requests": requests, "json": json, "re": re,
        "bs4": bs4, "BeautifulSoup": bs4.BeautifulSoup, "urljoin": urljoin,
        "io": io, "base64": base64, "hashlib": hashlib, "print": print,
        "genai": genai, "os": os
    }
    local_vars = {}

    try:
        exec(code_str, allowed_globals, local_vars)
        if 'solution' in local_vars:
            return local_vars['solution']
        else:
            return "Error: Code executed but 'solution' variable was never defined."
    except Exception as e:
        return f"Execution Error: {traceback.format_exc()}"

# ==============================================================================
# 3. GEMINI ANALYST
# ==============================================================================
async def analyze_task(deep_context: str, current_url: str):
    print("🧠 Gemini is analyzing the gathered context...")
    
    prompt = r"""
    You are an intelligent Data Extraction Agent.
    
    I have already visited the page, scanned sub-pages, transcribed audio, and inspected CSV headers.
    
    ALL THE CONTEXT IS BELOW (Level 1 is the main page, Level 2 are links):
    =========================================
    {deep_context}
    =========================================
    
    CURRENT URL: {current_url}
    MY EMAIL: {email}
    
    YOUR JOB: Write a Python script to calculate the final `solution`.
    
    RULES:
    1. **Context Integration**: Combine instructions from LEVEL 1 (e.g., "Cutoff is 50") with data from LEVEL 2 (e.g., "CSV File").
    2. **CSV Logic**: I have shown you the headers. You MUST write `requests.get()` to download the full CSV and process it.
    3. **Instructions**: If a transcript says "Sum column X", do exactly that.
    4. **Hashing**: If you see hashing instructions (SHA1, etc), implement it in Python using `hashlib`.
    5. **Output**: Return ONLY Python code. Initialize `solution` and `target`.
    """
    
    safe_context = deep_context.replace('{', '{{').replace('}', '}}').replace("'", "")
    final_prompt = prompt.format(deep_context=safe_context, current_url=current_url, email=STUDENT_EMAIL)

    try:
        response = await asyncio.to_thread(model.generate_content, final_prompt)
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return ""

def extract_submit_url(text: str, base_url: str):
    match = re.search(r'(?:POST|submit).*?to\s+((?:https?://|/)\S+)', text, re.IGNORECASE)
    if match:
        raw_url = match.group(1).strip(".,;\"'")
        return urljoin(base_url, raw_url)
    return None

# ==============================================================================
# 4. AGENT LOOP
# ==============================================================================
async def run_quiz_chain(start_url: str):
    current_url = start_url
    steps = 0
    max_steps = 15 
    
    print(f"🚀 STARTING QUIZ CHAIN at {start_url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        while current_url and steps < max_steps:
            steps += 1
            print(f"\n--- STEP {steps}: Visiting {current_url} ---")

            try:
                await page.goto(current_url)
                
                try:
                    await page.wait_for_load_state("networkidle", timeout=3000)
                except:
                    pass
                
                # --- GATHER DEEP CONTEXT (BREADTH FIRST) ---
                deep_context = await gather_deep_context(page, current_url)
                
                # --- ANALYZE ---
                code = await analyze_task(deep_context, current_url)
                answer = execute_generated_code(code)
                
                # --- RETRY LOOP ---
                retries = 0
                while "Error" in str(answer) and retries < 2:
                    print(f"⚠️ Code failed. Retrying... ({retries+1}/2)")
                    fix_prompt = f"Previous code failed with: {answer}\n\nCode:\n{code}\n\nFIX IT."
                    retry_resp = await asyncio.to_thread(model.generate_content, fix_prompt)
                    code = retry_resp.text
                    answer = execute_generated_code(code)
                    retries += 1
                
                print(f"💡 Final Answer: {answer}")
                
                # --- SUBMIT ---
                visible_text = await page.inner_text("body")
                submit_url = extract_submit_url(visible_text, current_url)
                
                if not submit_url:
                    if "demo" in current_url:
                         submit_url = "https://tds-llm-analysis.s-anand.net/demo/submit"
                    else:
                        print("⚠️ FATAL: No submit URL found.")
                        break

                payload = {
                    "email": STUDENT_EMAIL,
                    "secret": STUDENT_SECRET,
                    "url": current_url,
                    "answer": answer
                }
                
                print(f"📤 Submitting to {submit_url}...")
                response = requests.post(submit_url, json=payload)
                
                try:
                    resp_data = response.json()
                    print(f"✅ Response: {resp_data}")
                    
                    if resp_data.get("url"):
                        current_url = resp_data.get("url")
                    else:
                        print("🏆 QUIZ COMPLETED.")
                        break
                except:
                    print(f"❌ Submission Error: {response.text}")
                    break
                    
            except Exception as e:
                print(f"💥 Error: {e}")
                traceback.print_exc()
                break
        
        await browser.close()
        print("🏁 Agent finished.")

@app.post("/llm-agent")
async def start_task(payload: TaskPayload, background_tasks: BackgroundTasks):
    if payload.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")
    background_tasks.add_task(run_quiz_chain, payload.url)
    return {"message": "Agent started."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)