import sys
import asyncio

# 0. WINDOWS FIX
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

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
# 1. CONFIGURATION
# ==============================================================================
app = FastAPI()

# GET FROM ENVIRONMENT
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
# 2. EXECUTION HANDS
# ==============================================================================
def execute_generated_code(code_str: str):
    print("⚡ Executing Python code...")
    code_str = code_str.replace("```python", "").replace("```", "").strip()
    
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
# 3. GEMINI BRAIN (CRASH PROOF EDITION)
# ==============================================================================
async def analyze_task(page_text: str, current_url: str):
    print("🧠 Gemini is thinking...")
    
    # RAW STRING
    prompt = r"""
    You are an automated Data Analyst Agent.
    
    CURRENT URL: {current_url}
    MY EMAIL: {email}
    INPUT TEXT FROM WEBPAGE:
    {page_text}
    
    YOUR GOAL: Write a Python script to solve the task.
    
    CRITICAL INSTRUCTIONS:
    1. **INITIALIZE VARIABLES (MANDATORY):** - Start the script EXACTLY like this:
         `solution = "Not Found"`
         `target = '{current_url}'`
         `limit = None`
    
    2. **SCENARIO A: "Scrape /path"**
       - Code pattern:
         `match = re.search(r'Scrape\s+([^\s]+)', r'{page_text}')`
         `if match:`
         `    target = urljoin('{current_url}', match.group(1))`
         `    print(f"DEBUG: Fetching {{target}}")`
         `    resp = requests.get(target, headers={{'User-Agent': 'Mozilla/5.0'}})`
         `    # 1. Try JSON`
         `    try: solution = resp.json()['message']`
         `    except: pass`
         `    # 2. Script Hunter`
         `    if solution == "Not Found":`
         `        soup = BeautifulSoup(resp.text, 'html.parser')`
         `        # USE resp.url to resolve relative links safely`
         `        scripts = [urljoin(resp.url, s['src']) for s in soup.find_all('script', src=True)]`
         `        imports = re.findall(r'from\s+[\"\']\./([^\"\']+)[\"\']', resp.text)`
         `        for i in imports: scripts.append(urljoin(resp.url, i))`
         `        `
         `        for js_url in scripts:`
         `            try:`
         `                js_text = requests.get(js_url).text`
         `                # NESTED IMPORTS`
         `                nested = re.findall(r'from\s+[\"\']\./([^\"\']+)[\"\']', js_text)`
         `                for n in nested:`
         `                     js_text += "\n" + requests.get(urljoin(js_url, n)).text`
         `                `
         `                print(f"DEBUG: JS Analysis on {{len(js_text)}} chars")`
         `                # LOGIC HUNTER: SHA1 / EMAIL`
         `                if "sha1(email)" in js_text or "crypto.subtle" in js_text:`
         `                    print("DEBUG: Found SHA1 logic. Calculating locally.")`
         `                    import hashlib`
         `                    # Logic: parseInt(sha1(email).slice(0, 4), 16)`
         `                    hash_obj = hashlib.sha1('{email}'.encode())`
         `                    hex_dig = hash_obj.hexdigest()`
         `                    solution = int(hex_dig[:4], 16)`
         `                    break`
         `                `
         `                # VARIABLE HUNTER`
         `                if solution == "Not Found":`
         `                    var_match = re.search(r'(?:const|let|var)\s+\w+\s*=\s*[\"\']([a-zA-Z0-9]+)[\"\']', js_text)`
         `                    if var_match: solution = var_match.group(1)`
         `            except: pass`
    
    3. **SCENARIO B: "CSV" or "Download"**
       - Code pattern:
         `resp = requests.get('{current_url}')`
         `soup = BeautifulSoup(resp.content, 'html.parser')`
         `link = soup.find('a', string=re.compile(r'CSV|Download', re.I))`
         `if link:`
         `    d_url = urljoin('{current_url}', link['href'])`
         `    print(f"DEBUG: Downloading {{d_url}}")`
         `    content = requests.get(d_url).text`
         `    nums = [int(n) for n in re.findall(r'-?\d+', content)]`
         `    if "Cutoff" in r'{page_text}':`
         `        cutoff_match = re.search(r'Cutoff:\s*(\d+)', r'{page_text}')`
         `        if cutoff_match:`
         `             limit = int(cutoff_match.group(1))`
         `    # SAFE FILTERING: Only filter if limit was actually found`
         `    if limit is not None:`
         `         nums = [n for n in nums if n > limit]`
         `    solution = sum(nums)`
         
    4. **SCENARIO C: Simple Answer**
       - If text says "answer": `solution = "anything you want"`
    
    5. **OUTPUT:** Output ONLY the Python code.
    """
    
    safe_page_text = page_text.replace('{', '{{').replace('}', '}}').replace("'", "")
    final_prompt = prompt.format(current_url=current_url, page_text=safe_page_text, email=STUDENT_EMAIL)

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
                    await page.wait_for_selector("body", timeout=5000)
                except:
                    pass
                
                page_text = await page.inner_text("body")
                print(f"📄 RAW TEXT SEEN:\n{page_text[:300]}...\n[...]\n")

                code = await analyze_task(page_text, current_url)
                answer = execute_generated_code(code)
                print(f"💡 Calculated Answer: {answer}")
                
                submit_url = extract_submit_url(page_text, current_url)
                
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

@app.post("/llm-agent")
async def start_task(payload: TaskPayload, background_tasks: BackgroundTasks):
    if payload.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")
    background_tasks.add_task(run_quiz_chain, payload.url)
    return {"message": "Agent started."}

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 to allow external access
    uvicorn.run(app, host="0.0.0.0", port=8000)