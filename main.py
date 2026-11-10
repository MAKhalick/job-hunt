import os
import re
import time
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from dotenv import load_dotenv
import gspread
import pandas as pd
from google import genai
from google.oauth2.service_account import Credentials
from jobspy import scrape_jobs
from sentence_transformers import SentenceTransformer, util

load_dotenv()

# --- CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
SHEET_NAME = "AI Job Tracker"
LOCAL_TIMEZONE = ZoneInfo("America/Toronto")

HERE = Path(__file__).resolve().parent
POSSIBLE_MASTER_PATHS = [
    HERE.parent / "Resume_Automator" / "master.json",
    HERE / "master.json",
    Path(r"C:\Users\moham\Desktop\PythonProjects\Resume_Automator\master.json")
]

def load_master_profile():
    """Dynamically loads master.json from Resume_Automator as single source of truth."""
    for p in POSSIBLE_MASTER_PATHS:
        if p.exists():
            try:
                master = json.loads(p.read_text(encoding="utf-8"))
                skills_list = ", ".join([f"{s['label']}: {s['items']}" for s in master.get("skills", [])])
                exp_summary = " ".join([f"{e['title']} - {' '.join(e['bullets'][:2])}" 
                                        for g in master.get("experience_groups", []) for e in g.get("entries", [])])
                proj_summary = " ".join([f"{pr['name']}: {' '.join(pr['bullets'][:2])}" for pr in master.get("projects", [])])
                
                print(f"Loaded master profile from: {p}")
                return f"""Candidate: Mohammed Adnan Khalick
Education: Master of Engineering in Applied Data Science, University of Ottawa (CGPA 9.65/10.0). B.Eng in Electronics & Communication.
Experience: 3+ years experience as Data Engineer, Analytics Engineer, and Senior Software Engineer at Cognizant, Capgemini, and Vosyn.
Skills: {skills_list}
Professional Work: {exp_summary}
Core Projects: {proj_summary}
Target Roles: Data Analytics Engineer, Data Engineer, Power BI Developer, Database Developer, BI/Data Analyst.
Target Location: Canada (Remote / Hybrid / Relocation across Canadian cities)."""
            except Exception as e:
                print(f"Warning: Failed reading {p}: {e}")
                break

    return """Candidate: Mohammed Adnan Khalick
Education: Master of Engineering in Applied Data Science, uOttawa (CGPA 9.65/10).
Experience: 3+ years in Data Engineering, Analytics Engineering, and SQL Database Systems (Cognizant, Capgemini, Vosyn).
Skills: SQL, T-SQL, Python, Snowflake, Azure SQL, Databricks, dbt, Apache Airflow, Power BI, DAX, RLS, Docker, GCP, AWS.
Target Roles: Data Analytics Engineer, Data Engineer, Database Engineer, Data Analyst, Power BI Developer."""

RESUME_SUMMARY = load_master_profile()
client = genai.Client()

def clean_location(loc_str):
    """Formats Canadian locations cleanly to 'City, Province'."""
    if not loc_str or pd.isna(loc_str):
        return "Ottawa, ON"
    s = str(loc_str).replace(", CA", "").replace(", Canada", "").strip()
    s = re.sub(r'Rexdale,\s*ON', 'Toronto, ON', s, flags=re.I)
    s = re.sub(r'North York,\s*ON', 'Toronto, ON', s, flags=re.I)
    s = re.sub(r'Mississauga,\s*ON', 'Mississauga, ON', s, flags=re.I)
    return s

def authenticate_gsheets():
    creds_file = HERE / "credentials.json"
    if not creds_file.exists():
        creds_file = Path(r"C:\Users\moham\Desktop\PythonProjects\job-hunter-pipeline\credentials.json")
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(str(creds_file), scopes=scopes)
    gc = gspread.authorize(creds)
    spreadsheet = gc.open(SHEET_NAME)
    return spreadsheet.worksheet("Job Postings"), spreadsheet.worksheet("Company Directory")

def fetch_jobs():
    print("Scraping a wide net across Canada (LinkedIn, Indeed, Glassdoor)...")
    try:
        search_query = ('"Data Engineer" OR "Analytics Engineer" OR "Database Developer" '
                        'OR "Data Analyst" OR "Power BI Developer" OR "BI Developer"')
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
    
    EVALUATION RULES:
    1. Language Gate: The candidate DOES NOT speak French. If the posting requires French or bilingualism, return SCORE: 0% and CATEGORY: Language.
    2. Relevance Gate: The role must involve working with data (SQL, Python, ETL, BI, databases, analytics, data engineering). If it is purely general IT support, sales, or customer service, return SCORE: 0% and CATEGORY: Not Relevant.
    3. Experience Gate: The candidate has 3+ years of professional experience plus a top-tier Master's degree.
       - Roles asking for 0-4 years: evaluate fully based on skills (80-100%).
       - Roles asking for 5 years: eligible if technical match is strong (75-84%).
       - Roles strictly requiring 6+ years or Lead/Principal/Manager: return SCORE: 0% and CATEGORY: Experience Gap.
    
    Output strictly in this format:
    SCORE: [0-100]%
    REASON: [1 concise sentence evaluating requirements against the profile]
    CATEGORY: [Match | Skill Gap | Experience Gap | Language | Not Relevant]
    MISSING_SKILLS: [Comma-separated list of technical skills required that the candidate lacks. If none, write 'None']
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            error_msg = str(e)
            if '503' in error_msg or '500' in error_msg or '429' in error_msg:
                if attempt < max_retries - 1:
                    sleep_time = 5 * (2 ** attempt)
                    print(f"API busy ({error_msg[:30]}). Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue
            return f"SCORE: 0%\nREASON: Analysis failed: {e}\nCATEGORY: Error\nMISSING_SKILLS: None"

def run_pipeline():
    print("Connecting to Google Sheets...")
    job_sheet, company_sheet = authenticate_gsheets()
    
    df_new = fetch_jobs()
    if df_new.empty:
        print("No jobs fetched.")
        return

    # 1. Deduplicate by Title & Company
    df_new['title_lower'] = df_new['title'].str.lower()
    df_new['company_lower'] = df_new['company'].str.lower()
    df_new = df_new.drop_duplicates(subset=['title_lower', 'company_lower'])

    exclude_keywords = ['senior', 'sr.', 'sr ', 'lead', 'principal', 'staff', 'manager', 'director', 'architect', 'head of']
    initial_scrape_count = len(df_new)
    df_new = df_new[~df_new['title_lower'].str.contains('|'.join(exclude_keywords), na=False)]
    print(f"✂️ Filtered out {initial_scrape_count - len(df_new)} senior/high-level job titles automatically.")
    
    # 2. Deduplicate Against Historical Jobs
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

    # 3. Filter Out Spam Companies
    company_records = company_sheet.get_all_records()
    known_companies = pd.DataFrame(company_records) if company_records else pd.DataFrame(columns=['Company Name', 'Status'])
    
    if not known_companies.empty:
        known_companies['name_lower'] = known_companies['Company Name'].astype(str).str.lower().str.strip()
        spam_companies = set(known_companies[known_companies['Status'].str.lower() == 'spam']['name_lower'])
        all_known_names = set(known_companies['name_lower'])
    else:
        spam_companies = set()
        all_known_names = set()
    
    df_new = df_new[~df_new['company_lower'].isin(spam_companies)]

    if df_new.empty:
        print("All fetched jobs were from flagged spam companies.")
        return

    print(f"Loading ML Model and Vectorizing {len(df_new)} jobs against updated Master Profile...")
    ml_model = SentenceTransformer('all-MiniLM-L6-v2')
    resume_embedding = ml_model.encode(RESUME_SUMMARY, convert_to_tensor=True)
    job_texts = (df_new['title'].fillna('') + " " + df_new['description'].fillna('')).tolist()
    job_embeddings = ml_model.encode(job_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(resume_embedding, job_embeddings)[0].tolist()
    
    df_new['ml_similarity_score'] = cosine_scores
    df_new = df_new.sort_values(by='ml_similarity_score', ascending=False)
    
    # Send top candidates to Gemini
    df_to_analyze = df_new.head(15).copy().fillna("")
    print(f"Sending the Top {len(df_to_analyze)} jobs to {GEMINI_MODEL}...")
    
    rows_to_insert = []
    new_companies_to_add = []
    processed_new_companies = set()
    
    for _, row in df_to_analyze.iterrows():
        desc = str(row.get('description', '')).strip()
        result = analyze_job(row['title'], row['company'], desc)
        
        score_match = re.search(r'SCORE:\s*(\d+)', result)
        reason_match = re.search(r'REASON:\s*(.*)', result)
        category_match = re.search(r'CATEGORY:\s*(.*)', result)
        skills_match = re.search(r'MISSING_SKILLS:\s*(.*)', result)
        
        score = int(score_match.group(1)) if score_match else 0
        reason = reason_match.group(1).strip() if reason_match else ""
        category = category_match.group(1).strip() if category_match else "Unknown"
        missing_skills = skills_match.group(1).strip() if skills_match else "None"
        
        comp_lower = row['company_lower']
        if comp_lower not in all_known_names and comp_lower not in processed_new_companies:
            new_companies_to_add.append([row['company'], "Pending"])
            processed_new_companies.add(comp_lower)
        
        if score >= 75 and category in ["Match", "Skill Gap"]:
            print(f"🎯 Match Found ({score}%): {row['title']} at {row['company']}")
            date_fetched = datetime.now(LOCAL_TIMEZONE).strftime("%Y-%m-%d")
            clean_loc = clean_location(row.get('location', ''))
            
            new_row = [
                "To Apply", 
                str(row.get('site', '')), 
                str(row.get('title', '')), 
                str(row.get('company', '')), 
                clean_loc, 
                str(row.get('job_type', '')), 
                str(row.get('job_url', '')), 
                date_fetched, 
                round(row['ml_similarity_score'], 4), 
                score, 
                reason, 
                missing_skills, 
                desc 
            ]
            rows_to_insert.append(new_row)
        else:
            print(f"AI Skipped ({score}%, {category}): {row['title']} at {row['company']}")
            
        time.sleep(2)

    # Push to Google Sheets
    if new_companies_to_add:
        company_sheet.append_rows(new_companies_to_add)
        print(f"Added {len(new_companies_to_add)} new companies to directory.")

    if rows_to_insert:
        # Sort by AI score descending
        rows_to_insert.sort(key=lambda x: x[9], reverse=True)
        job_sheet.append_rows(rows_to_insert)
        print(f"Success! Inserted {len(rows_to_insert)} jobs sorted by highest score into Google Sheets.")
    else:
        print("Run complete. No new matches met the threshold.")

if __name__ == "__main__":
    run_pipeline()