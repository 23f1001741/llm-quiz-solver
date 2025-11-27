import google.generativeai as genai
import os

API_KEY = "AIzaSyBE4UkZxm0pOMil7yac2SO-EcBlmf4rXVI"
genai.configure(api_key=API_KEY)

print("Listing available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")