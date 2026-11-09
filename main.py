import pandas as pd
from jobspy import scrape_jobs
from google import genai
from sentence_transformers import SentenceTransformer, util
import time
from datetime import datetime
import os
import re
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials

# Load local environment variables
load_dotenv()

#adding comment to keep the job hunt pipeline active

# --- CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SHEET_NAME = "AI Job Tracker"

RESUME_SUMMARY = """
Candidate: Mohammed Adnan Khalick
Education: Master of Digital Transformation (Applied Data Science), uOttawa (GPA 3.8/4.0) - Recent Graduate. B.Eng in Electronics.
Recent Experience: Data Engineer Intern at Vosyn (3-month completed internship - Django, PostgreSQL, Docker, GCP, REST APIs).
Past Experience: 3 years combined as Senior Software Engineer & Programmer Analyst at Capgemini and Cognizant.
Total Years of Professional Experience: Exactly 3 Years.
Core Skills: Python, SQL (T-SQL, PostgreSQL, MySQL), Data Modeling, ETL Pipelines, Apache Airflow, Docker.
Analytics & BI: Power BI, Tableau, Pandas, NumPy.
Cloud & Frameworks: AWS, GCP, Snowflake, Django REST Framework, Flask.
Target Roles: Data Engineer, Database Engineer, Data Analyst, Power BI Developer, Backend Engineer.
Target Work Type: Full-time or Internship. Remote, Hybrid, or In-person across Canada.
"""

client = genai.Client()

def authenticate_gsheets():
    """Connects to Google Sheets and returns both worksheets."""
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file('credentials.json', scopes=scopes)
    gc = gspread.authorize(creds)
    spreadsheet = gc.open(SHEET_NAME)
    
    # Return both tabs
    job_sheet = spreadsheet.worksheet("Job Postings")
    company_sheet = spreadsheet.worksheet("Company Directory")
    return job_sheet, company_sheet

def fetch_jobs():
    print("Scraping a wide net across Canada (LinkedIn, Indeed, Glassdoor)...")
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
        print(f"Error during scraping: {e}")
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
    3. EXPERIENCE GATE: The candidate has exactly 3 years of experience. If the job description explicitly requires 4+, 5+, 6+, or more years of experience, or states it is intended for "Senior" or "Lead" profiles, you MUST evaluate it strictly, return SCORE: 0% and CATEGORY: Experience Gap.
    
    Evaluate the fit and rate the chances of getting an interview based on the alignment of skills and experience. 
    Provide the output strictly in this format:
    SCORE: [0-100]%
    REASON: [1 concise sentence evaluating requirements against the profile, explaining exactly why they fit or why they were gated/rejected]
    CATEGORY: [Match | Skill Gap | Experience Gap | Clearance/Language | Not Relevant | Agency]
    MISSING_SKILLS: [Comma-separated list of technical skills required that the candidate lacks. If none, write 'None']
    COMPANY_SECTOR: [1-3 words classifying the company's industry (e.g., Banking, Healthcare, IT Consulting). Guess based on context if not explicitly stated.]
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
        # If it's a 503 (Server Busy) or 500 (Server Error), wait and retry
            if '503' in error_msg or '500' in error_msg:
                if attempt < max_retries - 1:
                    sleep_time = 5 * (2 ** attempt) # Waits 5s, then 10s...
                    print(f"AI Server busy (503). Retrying in {sleep_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(sleep_time)
                    continue
            
            # If it's a 429 (Daily Quota empty) or we ran out of retries, fail gracefully
            return f"SCORE: 0%\nREASON: Analysis failed: {e}\nCATEGORY: Error\nMISSING_SKILLS: None\nCOVER LETTER: N/A"

def run_pipeline():
    print("Connecting to Google Sheets...")
    job_sheet, company_sheet = authenticate_gsheets()
    
    df_new = fetch_jobs()
    if df_new.empty:
        print("No jobs fetched.")
        return

    # 1. Cross-Portal Deduplication
    df_new['title_lower'] = df_new['title'].str.lower()
    df_new['company_lower'] = df_new['company'].str.lower()
    df_new = df_new.drop_duplicates(subset=['title_lower', 'company_lower'])

    # NEW OPTIMIZATION: Early Senior/Lead Title Exclusion (Saves local compute & cleans pool)
    exclude_keywords = ['senior', 'sr.', 'sr ', 'lead', 'principal', 'staff', 'manager', 'director', 'architect', 'head of']
    initial_scrape_count = len(df_new)
    df_new = df_new[~df_new['title_lower'].str.contains('|'.join(exclude_keywords), na=False)]
    print(f"✂️ Filtered out {initial_scrape_count - len(df_new)} senior/high-level job titles automatically.")
    
    # 2. Job History Deduplication (From Tab 1)
    existing_jobs = job_sheet.get_all_records()
    if existing_jobs:
        df_existing = pd.DataFrame(existing_jobs)
        if 'Title' in df_existing.columns and 'Company' in df_existing.columns:
            df_existing['combo'] = df_existing['Title'].astype(str).str.lower() + df_existing['Company'].astype(str).str.lower()
            df_new['combo'] = df_new['title_lower'] + df_new['company_lower']
            known_combos = set(df_existing['combo'].dropna())
            df_new = df_new[~df_new['combo'].isin(known_combos)]

    if df_new.empty:
        print("No new jobs to analyze right now.")
        return

    print("Checking Company Directory...")
    company_records = company_sheet.get_all_records()
    known_companies = pd.DataFrame(company_records) if company_records else pd.DataFrame(columns=['Company Name', 'Status', 'Sector'])
    
    # Clean up names for matching
    if not known_companies.empty:
        known_companies['name_lower'] = known_companies['Company Name'].astype(str).str.lower().str.strip()
        spam_companies = set(known_companies[known_companies['Status'].str.lower() == 'spam']['name_lower'])
        all_known_names = set(known_companies['name_lower'])
    else:
        spam_companies = set()
        all_known_names = set()
    
    # 3. Pre-Filter: Drop the Known Spam
    initial_count = len(df_new)
    df_new = df_new[~df_new['company_lower'].isin(spam_companies)]
    print(f"Blocked {initial_count - len(df_new)} jobs from your known Spam list.")

    if df_new.empty:
        print("All new jobs were filtered out as spam.")
        return

    print(f"Loading ML Model and Vectorizing {len(df_new)} jobs...")
    
    # 4. Machine Learning Semantic Pre-Filter
    ml_model = SentenceTransformer('all-MiniLM-L6-v2')
    resume_embedding = ml_model.encode(RESUME_SUMMARY, convert_to_tensor=True)
    job_texts = (df_new['title'].fillna('') + " " + df_new['description'].fillna('')).tolist()
    job_embeddings = ml_model.encode(job_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(resume_embedding, job_embeddings)[0].tolist()
    
    df_new['ml_similarity_score'] = cosine_scores
    df_new = df_new.sort_values(by='ml_similarity_score', ascending=False)
    
    # 5. Strict Safety Funnel (Top 15 Only)
    df_to_analyze = df_new.head(15).copy()
    df_to_analyze = df_to_analyze.fillna("")
    
    print(f"Sending the Top {len(df_to_analyze)} most semantically relevant jobs to Gemini 2.5...")
    
    rows_to_insert = []
    new_companies_to_add = []
    processed_new_companies = set() # Prevents adding the same company twice in one batch
    
    for _, row in df_to_analyze.iterrows():
        # Stop processing once we have filled our Top 10 ideal recommendations
        if len(rows_to_insert) >= 10:
            print("🎯 Successfully compiled the top 10 recommended jobs matching your profile. Stopping run.")
            break
        
        desc = str(row.get('description', 'No description provided'))
        result = analyze_job(row['title'], row['company'], desc)
        
        # Regex parsers
        score_match = re.search(r'SCORE:\s*(\d+)', result)
        category_match = re.search(r'CATEGORY:\s*(.*)', result)
        skills_match = re.search(r'MISSING_SKILLS:\s*(.*)', result)
        sector_match = re.search(r'COMPANY_SECTOR:\s*(.*)', result)
        
        score = int(score_match.group(1)) if score_match else 0
        category = category_match.group(1).strip() if category_match else "Unknown"
        missing_skills = skills_match.group(1).strip() if skills_match else "None"
        sector = sector_match.group(1).strip() if sector_match else "Unknown"
        
        # COMPANY DIRECTORY WITH THEIR SECTOR
        comp_lower = row['company_lower']
        if comp_lower not in all_known_names and comp_lower not in processed_new_companies:
            # We append the 3-column payload: [Name, Status, Sector]
            new_companies_to_add.append([row['company'], "Pending", sector])
            processed_new_companies.add(comp_lower)
        
       # NEW OPTIMIZATION: Only add to spreadsheet if it survives the experience/eligibility filters
        if score > 0 and category != "Experience Gap":
            print(f"🎯 Match Found ({score}%): {row['title']} at {row['company']}")
            date_fetched = datetime.now().strftime("%Y-%m-%d")
            new_row = [
                "To Apply", row.get('site', ''), row.get('title', ''), row.get('company', ''), 
                row.get('location', ''), row.get('job_type', ''), row.get('job_url', ''), 
                date_fetched, round(row['ml_similarity_score'], 4), score, 
                category, missing_skills, result
            ]
            rows_to_insert.append(new_row)
        else:
            print(f"AI Skipped ({category}): {row['title']} at {row['company']}")
            
        time.sleep(2)

    # 6. Push to Google Sheets
    if new_companies_to_add:
        company_sheet.append_rows(new_companies_to_add)
        print(f"Added {len(new_companies_to_add)} new companies to the directory with their Sectors.")

    if rows_to_insert:
        rows_to_insert.sort(key=lambda x: x[9], reverse=True) 
        job_sheet.append_rows(rows_to_insert)
        print(f"Success! Pushed {len(rows_to_insert)} jobs to the Job Tracker.")
    else:
        print("Run complete. No job-matches found for this batch.")

if __name__ == "__main__":
    run_pipeline()