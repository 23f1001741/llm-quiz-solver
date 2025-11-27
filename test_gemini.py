import google.generativeai as genai
import os

# 1. Setup the "Brain"
# In the real project, we will use an Environment Variable for safety.
# For this test, it's okay to paste it here TEMPORARILY.
API_KEY = "AIzaSyBE4UkZxm0pOMil7yac2SO-EcBlmf4rXVI"

genai.configure(api_key=API_KEY)

# 2. Pick the model
# "gemini-1.5-flash" is the fast, efficient version we want.
model = genai.GenerativeModel('models/gemini-2.0-flash')

def main():
    print("Thinking...")
    
    # 3. Send a message
    response = model.generate_content("Write a short, funny poem about a Python coding snake.")
    
    # 4. Print the result
    print("--------------------------------")
    print(response.text)
    print("--------------------------------")

if __name__ == "__main__":
    main()