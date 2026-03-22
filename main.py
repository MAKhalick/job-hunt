import pandas as pd
from jobspy import scrape_jobs
from google import genai
from dotenv import load_dotenv
import time
import os

# --- CONFIGURATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Ensure your environment variable is set
OUTPUT_FILE = "main_job_matches.csv"

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

def fetch_jobs():
    print("🔍 Scraping jobs across Canada...")
    try:
        search_query = '"Data Engineer" OR "Database Developer" OR "Data Analyst" OR "Power BI" OR "Backend Engineer"'
        jobs = scrape_jobs(
            site_name=["linkedin", "indeed", "glassdoor"],
            search_term=search_query,
            location="Canada",
            results_wanted=30,  
            hours_old=48,       
            country_indeed='canada',
            linkedin_fetch_description=True
        )
        return jobs
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        return pd.DataFrame()

def analyze_job(title, company, description):
    print(f"🤖 AI Analyzing: {title} at {company}...")
    
    prompt = f"""
    You are an expert technical recruiter matching a job to this candidate:
    {RESUME_SUMMARY}
    
    Job Title: {title} at {company}
    Job Description: {description[:2500]} 
    
    CRITICAL RULES:
    1. The candidate DOES NOT speak French. If the description is in French, or requires bilingual French/English, return SCORE: 0% and REASON: French language requirement.
    
    Evaluate the fit. Provide the output strictly in this format:
    SCORE: [0-100]%
    REASON: [1 concise sentence on why their specific skills fit this role]
    COVER LETTER: [Draft a 2-sentence highly tailored opening for a cold email]
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"SCORE: 0%\nREASON: Analysis failed: {e}\nCOVER LETTER: N/A"

def run_pipeline():
    df_new = fetch_jobs()
    
    if df_new.empty:
        print("📭 No jobs fetched.")
        return

    # 1. Deduplication Logic
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        known_urls = set(df_existing['job_url'].dropna())
        # Keep only jobs whose URL is NOT in the known_urls set
        df_new = df_new[~df_new['job_url'].isin(known_urls)]
        print(f"🧹 Filtered out previously seen jobs. {len(df_new)} brand new jobs remain.")
    else:
        print(f"🆕 No existing main file found. Creating a new one with {len(df_new)} jobs.")

    if df_new.empty:
        print("✅ No new jobs to analyze right now. Try again tomorrow!")
        return

    # 2. Sort to get freshest jobs, and mix up the sites
    df_new = df_new.sort_values(by=['date_posted', 'site'], ascending=[False, True])

    print(f"⏳ Starting AI analysis on the top 15 new jobs...")
    
    analysis_results = []
    df_to_analyze = df_new.head(15).copy()
    
    for _, row in df_to_analyze.iterrows():
        desc = str(row.get('description', 'No description provided'))
        result = analyze_job(row['title'], row['company'], desc)
        analysis_results.append(result)
        time.sleep(3) 

    df_to_analyze['ai_analysis'] = analysis_results
    
    # 3. Keep the requested columns, including job_type and site
    columns_to_keep = ['site', 'title', 'company', 'location', 'job_type', 'job_url', 'date_posted', 'ai_analysis']
    df_clean = df_to_analyze[columns_to_keep]
    
    # 4. Save/Append to the Main CSV
    if os.path.exists(OUTPUT_FILE):
        df_clean.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    else:
        df_clean.to_csv(OUTPUT_FILE, index=False)
        
    print(f"\n🎉 Success! Added {len(df_clean)} new analyzed jobs to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    run_pipeline()