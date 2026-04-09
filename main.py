import pandas as pd
from jobspy import scrape_jobs
from google import genai
from sentence_transformers import SentenceTransformer, util
import time
import os
import re
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials

# Load local environment variables
load_dotenv()

# --- CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SHEET_NAME = "AI Job Tracker"

RESUME_SUMMARY = """
Candidate: Mohammed Adnan Khalick
Education: Master of Digital Transformation (Applied Data Science), uOttawa (GPA 3.8/4.0) - Currently in final academic semester. B.Eng in Electronics.
Recent Experience: Data Engineer Intern at Vosyn (3-month completed internship - Django, PostgreSQL, Docker, GCP, REST APIs).
Past Experience: 3 years combined as Senior Software Engineer & Programmer Analyst at Capgemini and Cognizant.
Core Skills: Python, SQL (T-SQL, PostgreSQL, MySQL), Data Modeling, ETL Pipelines, Apache Airflow, Docker.
Analytics & BI: Power BI, Tableau, Pandas, NumPy.
Cloud & Frameworks: AWS, GCP, Snowflake, Django REST Framework, Flask.
Target Roles: Data Engineer, Database Engineer, Data Analyst, Power BI Developer, Backend Engineer.
Target Work Type: Full-time or Internship. Remote, Hybrid, or In-person across Canada.
"""

client = genai.Client()

def authenticate_gsheets():
    """Connects to Google Sheets using the service account credentials."""
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file('credentials.json', scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open(SHEET_NAME).sheet1

def fetch_jobs():
    print("🔍 Scraping a wide net across Canada (LinkedIn, Indeed, Glassdoor)...")
    try:
        search_query = '"Data Engineer" OR "Database Developer" OR "Data Analyst" OR "Power BI" OR "Backend Engineer"'
        jobs = scrape_jobs(
            site_name=["linkedin", "indeed", "glassdoor"],
            search_term=search_query,
            location="Canada",
            results_wanted=60, 
            hours_old=48,       
            country_indeed='canada',
            linkedin_fetch_description=True
        )
        return jobs
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        return pd.DataFrame()

def analyze_job(title, company, description):
    prompt = f"""
    You are an expert technical recruiter matching a job to this candidate:
    {RESUME_SUMMARY}
    
    Job Title: {title} at {company}
    Job Description: {description[:2500]} 
    
    CRITICAL RULES:
    1. The candidate DOES NOT speak French. If the description is in French, or requires bilingual French/English, return SCORE: 0% and CATEGORY: Language.
    2. If the role has nothing to do with data, software, or analytics, return SCORE: 0% and CATEGORY: Not Relevant.
    
    Evaluate the fit. Provide the output strictly in this format:
    SCORE: [0-100]%
    REASON: [1 concise sentence on why they fit, or exactly why they were rejected]
    CATEGORY: [Match | Skill Gap | Experience Gap | Clearance/Language | Not Relevant]
    MISSING_SKILLS: [Comma-separated list of technical skills required that the candidate lacks. If none, write 'None']
    COVER LETTER: [Draft a 2-sentence highly tailored opening for a cold email. If SCORE is 0%, write 'N/A']
    """

    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            error_msg = str(e)
        # If it's a 503 (Server Busy) or 500 (Server Error), we wait and retry
            if '503' in error_msg or '500' in error_msg:
                if attempt < max_retries - 1:
                    sleep_time = 5 * (2 ** attempt) # Waits 5s, then 10s...
                    print(f"⚠️ AI Server busy (503). Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(sleep_time)
                    continue # Jumps back to the top of the loop to try again
            
            # If it's a 429 (Daily Quota empty) or we ran out of retries, fail gracefully
            return f"SCORE: 0%\nREASON: Analysis failed: {e}\nCATEGORY: Error\nMISSING_SKILLS: None\nCOVER LETTER: N/A"

def run_pipeline():
    # 1. Connect to the Cloud Database (Google Sheets)
    print("☁️ Connecting to Google Sheets...")
    sheet = authenticate_gsheets()
    
    df_new = fetch_jobs()
    if df_new.empty:
        print("📭 No jobs fetched.")
        return

    # 2. Cross-Portal Deduplication
    df_new['title_lower'] = df_new['title'].str.lower()
    df_new['company_lower'] = df_new['company'].str.lower()
    df_new = df_new.drop_duplicates(subset=['title_lower', 'company_lower'])
    
    # 3. Cloud Database Deduplication
    # Pull existing records from Google Sheets to ensure we never re-analyze old jobs
    existing_records = sheet.get_all_records()
    if existing_records:
        df_existing = pd.DataFrame(existing_records)
        if 'Title' in df_existing.columns and 'Company' in df_existing.columns:
            df_existing['combo'] = df_existing['Title'].astype(str).str.lower() + df_existing['Company'].astype(str).str.lower()
            df_new['combo'] = df_new['title_lower'] + df_new['company_lower']
            known_combos = set(df_existing['combo'].dropna())
            df_new = df_new[~df_new['combo'].isin(known_combos)]

    if df_new.empty:
        print("✅ No new jobs to analyze right now. Try again later!")
        return

    print(f"🧠 Loading ML Model and Vectorizing {len(df_new)} jobs...")
    
    # 4. Machine Learning Semantic Pre-Filter
    ml_model = SentenceTransformer('all-MiniLM-L6-v2')
    resume_embedding = ml_model.encode(RESUME_SUMMARY, convert_to_tensor=True)
    job_texts = (df_new['title'].fillna('') + " " + df_new['description'].fillna('')).tolist()
    job_embeddings = ml_model.encode(job_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(resume_embedding, job_embeddings)[0].tolist()
    
    df_new['ml_similarity_score'] = cosine_scores
    df_new = df_new.sort_values(by='ml_similarity_score', ascending=False)
    
    # 5. Strict Safety Funnel
    df_to_analyze = df_new.head(15).copy()
    df_to_analyze = df_to_analyze.fillna("")
    print(f"⏳ Sending the Top {len(df_to_analyze)} most semantically relevant jobs to Gemini 2.5...")
    
    rows_to_insert = []
    
    for _, row in df_to_analyze.iterrows():
        desc = str(row.get('description', 'No description provided'))
        result = analyze_job(row['title'], row['company'], desc)
        
        score_match = re.search(r'SCORE:\s*(\d+)', result)
        category_match = re.search(r'CATEGORY:\s*(.*)', result)
        skills_match = re.search(r'MISSING_SKILLS:\s*(.*)', result)
        
        score = int(score_match.group(1)) if score_match else 0
        category = category_match.group(1).strip() if category_match else "Unknown"
        missing_skills = skills_match.group(1).strip() if skills_match else "None"
        
        # Prepare the exact row format for Google Sheets based on our headers
        status = "To Apply" if score > 0 else "Skipped"
        if score > 0:
            print(f"🎯 AI Match ({score}%): {row['title']} at {row['company']}")
        else:
            print(f"⏭️ AI Skipped ({category}): {row['title']} at {row['company']}")

        new_row = [
            status,                                # Status
            row.get('site', ''),                   # Site
            row.get('title', ''),                  # Title
            row.get('company', ''),                # Company
            row.get('location', ''),               # Location
            row.get('job_type', ''),               # Job Type
            row.get('job_url', ''),                # URL
            str(row.get('date_posted', '')),       # Date Posted
            round(cosine_scores[0], 4),            # ML Score
            score,                                 # AI Score
            category,                              # Rejection Category
            missing_skills,                        # Missing Skills
            result                                 # AI Analysis
        ]
        rows_to_insert.append(new_row)
        time.sleep(2)

    # 6. Push to Google Sheets
    if rows_to_insert:
        # Sort so high AI scores are at the top of the batch
        rows_to_insert.sort(key=lambda x: x[9], reverse=True) 
        sheet.append_rows(rows_to_insert)
        print(f"\n🎉 Success! Pushed {len(rows_to_insert)} jobs directly to your Google Sheet.")

if __name__ == "__main__":
    run_pipeline()