import pandas as pd
import traceback

def run_ai_code(code_string):
    print("🤖 Robot Hand: I received code to run.")
    
    # 1. Sandbox setup (The allowed tools)
    # We give the code access to pandas, print, etc.
    allowed_tools = {
        "pd": pd,
        "print": print
    }
    
    # This dictionary will hold any variables the code creates
    memory = {}
    
    try:
        # 2. The Magic Command: exec()
        # It runs 'code_string' using 'allowed_tools'
        exec(code_string, allowed_tools, memory)
        
        # 3. Look for the answer
        # We expect the AI to save the answer in a variable named 'solution'
        if 'solution' in memory:
            return memory['solution']
        else:
            return "Error: The code ran, but didn't create a 'solution' variable."
            
    except Exception as e:
        return f"CRASH! The code failed: {e}"

# --- TEST SCENARIO ---

# Imagine Gemini sent this text:
fake_gemini_response = """
data = {'A': [10, 20, 30], 'B': [1, 2, 3]}
df = pd.DataFrame(data)

# Calculate sum of column A
solution = df['A'].sum()
"""

print("--- TESTING ---")
result = run_ai_code(fake_gemini_response)
print(f"✅ Final Result: {result}")