from google import genai
from dotenv import load_dotenv
import os

# 1. Load .env into environment variables
load_dotenv()

# 2. Explicitly retrieve the key value from the environment
GEMINI_KEY = os.getenv("GEMINI_API_KEY") # Use the variable name from your .env file

if GEMINI_KEY is None:
    # Raise a clear error if the key is still missing
    raise ValueError("API Key not found! Please check your .env file and variable name.")

# 3. Create client, explicitly passing the key
client = genai.Client(api_key=GEMINI_KEY)

# Simple generation example
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Write a short upbeat 4-line poem about coding in VS Code."
)

print("--- Gemini output ---")
print(response.text)