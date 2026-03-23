import pandas as pd
from jobspy import scrape_jobs
from google import genai
from sentence_transformers import SentenceTransformer, util
import time
import os
import re
from dotenv import load_dotenv

# Load local environment variables (API Key)
load_dotenv()

# --- CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OUTPUT_FILE = "master_job_matches.csv"

# The Profile used for both the ML Pre-Filter and the Gemini AI
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
    1. The candidate DOES NOT speak French. If the description is in French, or requires bilingual French/English, return SCORE: 0%.
    2. If the role has nothing to do with data, software, or analytics, return SCORE: 0%.
    
    Evaluate the fit. Provide the output strictly in this format:
    SCORE: [0-100]%
    REASON: [1 concise sentence on why their specific skills fit this role]
    COVER LETTER: [Draft a 2-sentence highly tailored opening for a cold email]
    """
    try:
        # Using Gemini 2.5 Flash (Strict 20/day limit)
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

    # 1. Cross-Portal Deduplication
    initial_count = len(df_new)
    df_new['title_lower'] = df_new['title'].str.lower()
    df_new['company_lower'] = df_new['company'].str.lower()
    df_new = df_new.drop_duplicates(subset=['title_lower', 'company_lower'])
    
    # 2. History Deduplication
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        df_existing['combo'] = df_existing['title'].str.lower() + df_existing['company'].str.lower()
        df_new['combo'] = df_new['title_lower'] + df_new['company_lower']
        known_combos = set(df_existing['combo'].dropna())
        df_new = df_new[~df_new['combo'].isin(known_combos)]

    if df_new.empty:
        print("✅ No new jobs to analyze right now. Try again later!")
        return

    print(f"🧠 Loading ML Model and Vectorizing {len(df_new)} jobs...")
    
    # 3. Machine Learning Semantic Pre-Filter
    # Load the lightweight NLP model (downloads automatically on first run)
    ml_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Embed the candidate's resume
    resume_embedding = ml_model.encode(RESUME_SUMMARY, convert_to_tensor=True)
    
    # Combine title and description for context, fill empty fields
    job_texts = (df_new['title'].fillna('') + " " + df_new['description'].fillna('')).tolist()
    
    # Embed all jobs at once (Batch Processing)
    job_embeddings = ml_model.encode(job_texts, convert_to_tensor=True)
    
    # Calculate Cosine Similarity between Resume and all Jobs
    cosine_scores = util.cos_sim(resume_embedding, job_embeddings)[0].tolist()
    
    # Add ML scores to the dataframe and sort
    df_new['ml_similarity_score'] = cosine_scores
    df_new = df_new.sort_values(by='ml_similarity_score', ascending=False)
    
    # 4. Strict Safety Funnel (Top 15 ONLY to protect Gemini 2.5 Quota)
    df_to_analyze = df_new.head(15).copy()
    print(f"⏳ Sending the Top {len(df_to_analyze)} most semantically relevant jobs to Gemini 2.5...")
    
    valid_jobs = []
    
    for _, row in df_to_analyze.iterrows():
        desc = str(row.get('description', 'No description provided'))
        result = analyze_job(row['title'], row['company'], desc)
        
        score_match = re.search(r'SCORE:\s*(\d+)', result)
        score = int(score_match.group(1)) if score_match else 0
        
        if score > 0:
            print(f"🎯 AI Final Match ({score}%): {row['title']} at {row['company']}")
            row_copy = row.copy()
            row_copy['ai_analysis'] = result
            row_copy['numeric_score'] = score
            valid_jobs.append(row_copy)
        else:
            print(f"⏭️ AI Skipped (0%): {row['title']} at {row['company']}")
            
        time.sleep(2) # Respect API limits

    if not valid_jobs:
        print("🤷‍♂️ None of the pre-filtered jobs passed the final Gemini check.")
        return

    # 5. Final Sort by Gemini Score
    df_valid = pd.DataFrame(valid_jobs)
    df_valid = df_valid.sort_values(by='numeric_score', ascending=False)
    
    columns_to_keep = ['site', 'title', 'company', 'location', 'job_type', 'job_url', 'date_posted', 'ml_similarity_score', 'ai_analysis']
    df_clean = df_valid[columns_to_keep]
    
    if os.path.exists(OUTPUT_FILE):
        df_clean.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    else:
        df_clean.to_csv(OUTPUT_FILE, index=False)
        
    print(f"\n🎉 Success! Appended {len(df_clean)} highly relevant, AI-sorted jobs to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    run_pipeline()