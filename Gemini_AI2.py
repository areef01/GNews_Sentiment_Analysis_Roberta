import pandas as pd
import os
import time # **NEW: Import for sleep duration**
from google import genai
from google.genai import types # Needed for GenerateContentConfig
from dotenv import load_dotenv
from pydantic import BaseModel, Field # Needed for structured output

# --- CONFIGURATION ---
load_dotenv()

CSV_INPUT_FILE = "Roberta_News_with_language_processed.csv"
EXCEL_OUTPUT_FILE = "Roberta_News_Analysis_FIRST_100_ROWS_AGRITECH.xlsx" # **UPDATED output name**
ARTICLE_TEXT_COLUMN_NAME = "article_text"
ROWS_TO_PROCESS = 100 # **UPDATED: To process the first 100 rows**
SLEEP_DURATION_SECONDS = 1 # **UPDATED: 1 second delay per call to stay within limits**
# --- END CONFIGURATION ---

# --- STRUCTURED OUTPUT DEFINITION ---
class AnalysisOutput(BaseModel):
    """Schema for the model's output: Summary and Technology Keyword."""
    Summary: str = Field(
        min_length=2,
        description="A single, short phrase or sentence summarizing the core content (must be at least 2 words)."
    )
    Keyword: str = Field(
        description="Exactly one technology-related keyword or category (e.g., 'Precision Farming', 'AI Ethics')."
    )
# --- END STRUCTURED OUTPUT DEFINITION ---


def safe_load_csv(file_path):
    """Attempts to read the CSV file using multiple encodings and separators."""
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
    separators_to_try = [',', '|']

    print(f"Attempting to read file: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ FATAL ERROR: File not found at '{file_path}'. Check the file path.")
        return None

    for sep in separators_to_try:
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                print(f"✅ Data loaded successfully with encoding: {encoding} and separator: '{sep}'")
                return df
            except Exception:
                continue

    print(f"❌ FATAL ERROR: Could not read the file '{file_path}' with any common combination.")
    return None


def summarize_and_extract_keywords(article_text, client: genai.Client) -> dict:
    """Calls Gemini API to produce a short summary and extract a keyword using structured output."""
    
    # **NEW: Updated Prompt with AgriTech Categories**
    agritech_categories = [
        "Precision Agriculture", "Digital Transformation (DX)", "Biotech & Genetics",
        "Farm Automation & Robotics", "Agri-Sustainability", "Supply Chain & Traceability",
        "Remote Sensing & Imagery"
    ]
    category_list = ", ".join([f"'{c}'" for c in agritech_categories])
    
    try:
        prompt = f"""
        Analyze the following article text. Your task is to perform two things:
        1. Summarize the core content in a single, short phrase or sentence.
        2. Extract exactly one (1) keyword or category from the following list that best describes the main technological focus of the article.
        
        **TECHNOLOGY CATEGORIES (Select ONLY ONE):** {category_list}

        **Article Text:**
        {article_text}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalysisOutput, # Use the Pydantic class as the schema
            )
        )

        # The response.parsed attribute provides the Pydantic object
        result_object: AnalysisOutput = response.parsed

        return {
            "Summary": result_object.Summary,
            "Keyword": result_object.Keyword
        }

    except Exception as e:
        # Catch and report any API-related errors
        return {"Summary": f"Processing Failed: {e}", "Keyword": "Processing Error"}


def process_data(input_file, output_file, text_column):
    """
    Main orchestration function: Loads CSV, processes a defined number of rows,
    adds a sleep delay, and saves the results to Excel.
    """
    # 1. Initialize Client
    API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not API_KEY:
        print("❌ FATAL ERROR: API Key not found in environment variables (GEMINI_API_KEY/GOOGLE_API_KEY).")
        return

    try:
        # Initialize the client
        client = genai.Client(api_key=API_KEY)

        # 2. Load Data Safely
        df = safe_load_csv(input_file)
        if df is None:
            return

        # 3. Prepare DataFrame
        if text_column not in df.columns:
            print(f"❌ FATAL ERROR: Column '{text_column}' not found in the loaded data.")
            return

        # Initialize new columns
        df['Summary'] = None
        df['Keyword'] = None

        # --- Limit to First N Rows ---
        rows_to_process_actual = min(ROWS_TO_PROCESS, len(df))
        print(f"\nStarting Gemini AI processing for the FIRST {rows_to_process_actual} rows...")

        # 4. Process Data Row-by-Row
        for index, row in df.iloc[:rows_to_process_actual].iterrows():
            article_text = str(row[text_column]).strip()
            if not article_text or article_text.lower() == 'nan':
                continue

            print(f"Processing row {index + 1}/{rows_to_process_actual}...")

            # Call the Gemini function
            result = summarize_and_extract_keywords(article_text, client)

            # 5. Store Results
            df.loc[index, 'Summary'] = result['Summary']
            df.loc[index, 'Keyword'] = result['Keyword']
            
            # **NEW: Rate Limiting Implementation**
            print(f"Sleeping for {SLEEP_DURATION_SECONDS} second(s) to respect API rate limits...")
            time.sleep(SLEEP_DURATION_SECONDS)
            
            if (index + 1) % 10 == 0:
                 print(f"Processed {index + 1} rows so far.")


        # 6. Save Results to Excel
        df.to_excel(output_file, index=False)
        print(f"\n✨ Processing complete. Results for first {rows_to_process_actual} rows saved to: {output_file}")

        # Print the relevant columns for quick verification
        print("\n--- Processed Data Head (Summary & Keyword) ---")
        print(df.head(rows_to_process_actual)[['Summary', 'Keyword']].tail(5))

    except Exception as e:
        # This catches errors that occur after client initialization
        print(f"An unexpected error occurred during processing: {e}")


if __name__ == '__main__':
    process_data(CSV_INPUT_FILE, EXCEL_OUTPUT_FILE, ARTICLE_TEXT_COLUMN_NAME)