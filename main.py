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
# 1. THE RECURSIVE CRAWLER (FULL BROWSER EDITION)
# ==============================================================================

visited_urls = set()
context_log = []

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

def inspect_csv(content: bytes) -> str:
    try:
        text_val = content.decode('utf-8', errors='ignore')
        try:
            df = pd.read_csv(io.StringIO(text_val))
        except:
            df = pd.read_csv(io.StringIO(text_val), sep=None, engine='python')
        return f"[CSV STRUCTURE]: Columns={list(df.columns)}, First Row={df.iloc[0].to_dict()}"
    except Exception as e:
        return f"[CSV ERROR]: {e}"

def process_puzzle_piece(url: str, mime_type: str, content: bytes) -> str:
    print(f"   🧩 Found Puzzle Piece ({mime_type}): {url}")
    if "audio" in mime_type:
        return f"--- AUDIO FILE ({url}) ---\n" + transcribe_media(content, mime_type)
    elif "image" in mime_type:
        return f"--- IMAGE FILE ({url}) ---\n" + transcribe_media(content, mime_type)
    elif "csv" in mime_type or url.endswith(".csv"):
        return f"--- CSV DATA ({url}) ---\n" + inspect_csv(content)
    elif "json" in mime_type or "text" in mime_type:
        text_val = content.decode('utf-8', errors='ignore')
        return f"--- TEXT DATA ({url}) ---\nCONTENT: {text_val[:1500]}"
    return ""

async def recursive_crawl(url: str, depth: int, browser, max_depth: int = 2):
    """
    Async Crawler that uses Playwright for HTML pages to ensure JS execution.
    """
    if url in visited_urls or depth > max_depth:
        return
    
    visited_urls.add(url)
    prefix = "  " * depth
    print(f"{prefix}👉 Crawling: {url} (Depth {depth})")

    try:
        # 1. HEAD Request (Fast Sniff)
        try:
            head = requests.head(url, timeout=5, allow_redirects=True)
            content_type = head.headers.get("Content-Type", "").lower()
        except:
            content_type = ""

        is_asset = any(x in content_type for x in ['audio', 'image', 'csv', 'json', 'text/plain'])
        is_html = 'html' in content_type

        # --- CASE A: ASSET (Use Requests - Fast) ---
        if is_asset or (not is_html and depth > 0): 
            resp = requests.get(url, timeout=10)
            piece_info = process_puzzle_piece(url, content_type, resp.content)
            if piece_info:
                context_log.append(piece_info)
            return

        # --- CASE B: HTML PAGE (Use Playwright - Slow but Smart) ---
        if is_html:
            # Open a new tab/page to run the JS
            page = await browser.new_page()
            try:
                await page.goto(url)
                try:
                    await page.wait_for_load_state("networkidle", timeout=4000)
                except:
                    pass
                
                # Get rendered text
                visible_text = await page.inner_text("body")
                context_log.append(f"=== PAGE TEXT ({url}) ===\n{visible_text[:2000]}")
                
                if depth >= max_depth:
                    await page.close()
                    return

                # Extract Links from DOM
                html = await page.content()
                soup = bs4.BeautifulSoup(html, 'html.parser')
                links = []
                for tag in soup.find_all(['a', 'script', 'img', 'source', 'audio']):
                    href = tag.get('href') or tag.get('src')
                    if href:
                        full_link = urljoin(url, href)
                        if "s-anand.net" in full_link or "localhost" in full_link:
                            if full_link not in visited_urls:
                                links.append(full_link)
                
                await page.close() # Close tab to free memory
                
                print(f"{prefix}   ↳ Found {len(links)} links. Recursing...")
                
                # RECURSE
                for link in links:
                    await recursive_crawl(link, depth + 1, browser, max_depth)

            except Exception as e:
                print(f"{prefix}⚠️ Error rendering page {url}: {e}")
                await page.close()

    except Exception as e:
        print(f"{prefix}⚠️ Error crawling {url}: {e}")

async def build_complete_context(start_url):
    print("🕵️ STARTING RECURSIVE CRAWL (Full Browser)...")
    visited_urls.clear()
    context_log.clear()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # We pass the BROWSER instance to the recursive function
        await recursive_crawl(start_url, 0, browser, max_depth=2)
        await browser.close()

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
    
    I have crawled the site using a full browser, executed all JavaScript, and transcribed all audio.
    
    FULL CONTEXT DUMP:
    =========================================
    {deep_context}
    =========================================
    
    CURRENT URL: {current_url}
    MY EMAIL: {email}
    
    TASK: Write a Python script to calculate the `solution`.
    
    CRITICAL RULES:
    1. **Puzzle Assembly**:
       - The answer is somewhere in the context above.
       - If you see "Secret code is X" in the PAGE TEXT logs, `solution = "X"`.
       - If you see "Sum column X" in AUDIO logs, do that.
    
    2. **Variable Safety**:
       - Initialize variables. Example: `limit = 0` before checking regex.
       - Always provide a fallback: `solution = "Not Found"`.
    
    3. **CSV Handling**:
       - I have provided the CSV headers. You MUST write `requests.get()` to download the full CSV and process it.
    
    4. **Output**: Return ONLY Python code.
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

    # Note: We now launch a SEPARATE browser instance inside recursive_crawl
    # But for the main loop, we still need one for submitting/navigating.
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
                
                # --- START RECURSIVE CRAWL (Passes URL, creates its own browser) ---
                deep_context = await build_complete_context(current_url)
                
                code = await analyze_task(deep_context, current_url)
                answer = execute_generated_code(code)
                
                retries = 0
                while "Error" in str(answer) and retries < 2:
                    print(f"⚠️ Code failed. Retrying... ({retries+1}/2)")
                    fix_prompt = f"Previous code failed with: {answer}\n\nCode:\n{code}\n\nFIX IT. Initialize all variables."
                    retry_resp = await asyncio.to_thread(model.generate_content, fix_prompt)
                    code = retry_resp.text
                    answer = execute_generated_code(code)
                    retries += 1
                
                print(f"💡 Final Answer: {answer}")
                
                # Re-fetch page text for submission link logic
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