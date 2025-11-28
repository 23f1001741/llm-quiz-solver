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
from urllib.parse import urljoin 
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from playwright.async_api import async_playwright
import google.generativeai as genai

# ==============================================================================
# 0. CONFIGURATION & SETUP
# ==============================================================================

# Fix for Windows Event Loop Policy
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

app = FastAPI()

# Environment Variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
STUDENT_EMAIL = os.environ.get("STUDENT_EMAIL")
STUDENT_SECRET = os.environ.get("STUDENT_SECRET")

if not GOOGLE_API_KEY:
    print("⚠️ WARNING: GOOGLE_API_KEY not found via Env Vars")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('models/gemini-2.0-flash')

class TaskPayload(BaseModel):
    email: str
    secret: str
    url: str

# ==============================================================================
# 1. EXECUTION ENGINE (SANDBOX)
# ==============================================================================
def execute_generated_code(code_str: str):
    print("⚡ Executing Python code...")
    # Clean up markdown formatting if present
    code_str = code_str.replace("```python", "").replace("```", "").strip()
    
    # Define the environment for the generated code
    allowed_globals = {
        "pd": pd,
        "np": np,
        "requests": requests,
        "json": json,
        "re": re,
        "bs4": bs4,
        "BeautifulSoup": bs4.BeautifulSoup,
        "urljoin": urljoin,
        "io": io,
        "base64": base64,
        "hashlib": hashlib,
        "print": print
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
# 2. GEMINI ANALYSIS (THE BRAIN)
# ==============================================================================
async def analyze_task(context_block: str, current_url: str):
    print("🧠 Gemini is thinking...")
    
    prompt = r"""
    You are an automated Data Analyst Agent.
    
    CURRENT URL: {current_url}
    MY EMAIL: {email}
    
    PAGE CONTEXT:
    {context_block}
    
    YOUR GOAL: Write a Python script to solve the task.
    
    CRITICAL RULES (FOLLOW THESE OR YOU WILL FAIL):
    
    1. **INITIALIZATION:**
       - ALWAYS start with:
         `solution = "Not Found"`
         `target = '{current_url}'`
       
    2. **HANDLING HIDDEN DATA (Step 2 - Client Side):**
       - Look at the 'DETECTED SCRIPTS' section in the context.
       - If the page text is empty or cryptic, the secret is likely inside a JS file.
       - **YOU MUST** write code to `requests.get()` these script URLs using `urljoin(target, script_url)`.
       - Check the script content for "var code =", "const secret =", or dynamic logic like "sha1(email)".
       - If you see hashing (SHA-256, SHA-1), implement it in Python using `hashlib`. DO NOT try to execute JS.
    
    3. **HANDLING FILES (Step 3 - Data Analysis):**
       - Look at the 'DETECTED LINKS' section.
       - IF you see a link ending in .csv, .pdf, .json, or .txt:
         a. **YOU MUST** write `requests.get(link_url)` to download it.
         b. **YOU MUST** use `io.StringIO` (for CSV/Text) or `io.BytesIO` (for PDF/Excel) to read it.
         c. **DO NOT** assume you know the file content. You must read the file.
       - If filtering numbers: `nums = [n for n in nums if n > limit]` (Be careful with strict inequality vs inclusive).
    
    4. **GENERAL:**
       - Use `requests` to fetch data.
       - Use `bs4` (BeautifulSoup) for HTML parsing.
       - Use `pd` (pandas) for heavy data, or `re` (regex) for simple extraction.
       - **Output ONLY the Python code.** No markdown, no explanation.
    """
    
    # Safety: Escape braces in the context to prevent .format() from crashing on JS code
    safe_context = context_block.replace('{', '{{').replace('}', '}}').replace("'", "")
    
    final_prompt = prompt.format(current_url=current_url, context_block=safe_context, email=STUDENT_EMAIL)

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
# 3. AGENT LOOP (THE BODY)
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
                    await page.wait_for_selector("body", timeout=5000)
                except:
                    pass
                
                # --- [CRITICAL FIX 1] EXTRACT CONTEXT LOCALLY ---
                # We do this HERE so Gemini doesn't have to guess where the scripts are.
                html_content = await page.content()
                visible_text = await page.inner_text("body")
                
                soup = bs4.BeautifulSoup(html_content, 'html.parser')
                
                # Extract Script URLs (recursively useful for Step 2)
                scripts = []
                for s in soup.find_all('script'):
                    src = s.get('src')
                    if src:
                        scripts.append(urljoin(current_url, src))
                
                # Extract Links (useful for Step 3 CSVs)
                links = []
                for a in soup.find_all('a'):
                    href = a.get('href')
                    if href:
                        links.append(urljoin(current_url, href))

                # Build the Context Block
                context_block = f"""
                VISIBLE TEXT ON PAGE:
                {visible_text[:2000]}
                
                DETECTED SCRIPTS (Might contain the secret):
                {json.dumps(scripts, indent=2)}
                
                DETECTED LINKS (Might be data files):
                {json.dumps(links, indent=2)}
                """
                
                print(f"🔎 Context Extracted: {len(scripts)} scripts, {len(links)} links.")
                
                # --- ANALYZE ---
                code = await analyze_task(context_block, current_url)
                answer = execute_generated_code(code)
                
                # --- [CRITICAL FIX 2] RETRY LOOP ---
                # If the code crashes (NameError, SyntaxError, etc), we give Gemini ONE chance to fix it.
                retries = 0
                while "Error" in str(answer) and retries < 1:
                    print(f"⚠️ Code failed. Asking Gemini to fix... (Error: {str(answer)[:100]}...)")
                    
                    fix_prompt = f"""
                    The previous code you wrote failed.
                    
                    THE CODE:
                    {code}
                    
                    THE ERROR:
                    {answer}
                    
                    TASK: Fix the code. Initialize all variables. Handle the error. Output ONLY Python.
                    """
                    
                    retry_resp = await asyncio.to_thread(model.generate_content, fix_prompt)
                    code = retry_resp.text
                    answer = execute_generated_code(code)
                    retries += 1
                
                print(f"💡 Final Answer: {answer}")
                
                # --- SUBMISSION LOGIC ---
                submit_url = extract_submit_url(visible_text, current_url)
                
                if not submit_url:
                    if "demo" in current_url and "submit" not in current_url:
                         submit_url = "https://tds-llm-analysis.s-anand.net/demo/submit"
                         print("⚠️ Regex failed. Using Hardcoded Demo Submit URL.")
                    else:
                        print("⚠️ FATAL: Could not find submit URL. Stopping.")
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
                    print(f"✅ Server Response: {resp_data}")

                    next_url = resp_data.get("url")
                    if next_url:
                        if resp_data.get("correct") is True:
                            print("🎉 Correct! Moving on...")
                        else:
                            print("❌ Wrong, but server gave a skip URL. Moving on...")
                        current_url = next_url
                    else:
                        print("🏆 QUIZ COMPLETED (No new URL provided).")
                        break
                        
                except:
                    print(f"❌ Submission failed. Status: {response.status_code}, Text: {response.text}")
                    break
                    
            except Exception as e:
                print(f"💥 Error in chain: {e}")
                traceback.print_exc()
                break
        
        await browser.close()
        print("🏁 Agent finished.")

# ==============================================================================
# 4. API SERVER
# ==============================================================================
@app.post("/llm-agent")
async def start_task(payload: TaskPayload, background_tasks: BackgroundTasks):
    if payload.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")
    background_tasks.add_task(run_quiz_chain, payload.url)
    return {"message": "Agent started."}

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 to allow external access (Required for Render/Docker)
    uvicorn.run(app, host="0.0.0.0", port=8000)