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
OUTPUT_FILE = "main_job_matches.csv"

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
    1. The candidate DOES NOT speak French. If the description is in French, or requires bilingual French/English, return SCORE: 0% and CATEGORY: Language.
    2. If the role has nothing to do with data, software, or analytics, return SCORE: 0% and CATEGORY: Not Relevant.
    
    Evaluate the fit. Provide the output strictly in this format:
    SCORE: [0-100]%
    REASON: [1 concise sentence on why they fit, or exactly why they were rejected]
    CATEGORY: [Match | Skill Gap | Experience Gap | Clearance/Language | Not Relevant]
    MISSING_SKILLS: [Comma-separated list of technical skills required that the candidate lacks (e.g., Databricks, Scala, Kafka). If none, write 'None']
    COVER LETTER: [Draft a 2-sentence highly tailored opening for a cold email. If SCORE is 0%, write 'N/A']
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"SCORE: 0%\nREASON: Analysis failed: {e}\nCATEGORY: Error\nMISSING_SKILLS: None\nCOVER LETTER: N/A"

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
    ml_model = SentenceTransformer('all-MiniLM-L6-v2')
    resume_embedding = ml_model.encode(RESUME_SUMMARY, convert_to_tensor=True)
    job_texts = (df_new['title'].fillna('') + " " + df_new['description'].fillna('')).tolist()
    job_embeddings = ml_model.encode(job_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(resume_embedding, job_embeddings)[0].tolist()
    
    df_new['ml_similarity_score'] = cosine_scores
    df_new = df_new.sort_values(by='ml_similarity_score', ascending=False)
    
    # 4. Strict Safety Funnel
    df_to_analyze = df_new.head(15).copy()
    print(f"⏳ Sending the Top {len(df_to_analyze)} most semantically relevant jobs to Gemini 2.5...")
    
    # We will now keep ALL analyzed jobs, categorized into matches and skipped
    matched_jobs = []
    skipped_jobs = []
    
    for _, row in df_to_analyze.iterrows():
        desc = str(row.get('description', 'No description provided'))
        result = analyze_job(row['title'], row['company'], desc)
        
        # Regex to extract the new fields
        score_match = re.search(r'SCORE:\s*(\d+)', result)
        category_match = re.search(r'CATEGORY:\s*(.*)', result)
        skills_match = re.search(r'MISSING_SKILLS:\s*(.*)', result)
        
        score = int(score_match.group(1)) if score_match else 0
        category = category_match.group(1).strip() if category_match else "Unknown"
        missing_skills = skills_match.group(1).strip() if skills_match else "None"
        
        row_copy = row.copy()
        row_copy['ai_analysis'] = result
        row_copy['numeric_score'] = score
        row_copy['rejection_category'] = category
        row_copy['missing_skills'] = missing_skills
        
        if score > 0:
            print(f"🎯 AI Final Match ({score}%): {row['title']} at {row['company']}")
            matched_jobs.append(row_copy)
        else:
            print(f"⏭️ AI Skipped ({category} - Missing: {missing_skills}): {row['title']} at {row['company']}")
            skipped_jobs.append(row_copy)
            
        time.sleep(2)

    # 5. Advanced Sorting
    # Sort the matches by how much the AI liked them (highest AI score first)
    df_matches = pd.DataFrame(matched_jobs)
    if not df_matches.empty:
        df_matches = df_matches.sort_values(by='numeric_score', ascending=False)
        
    # Sort the skips by how much the ML model liked them (highest ML score first)
    df_skipped = pd.DataFrame(skipped_jobs)
    if not df_skipped.empty:
        df_skipped = df_skipped.sort_values(by='ml_similarity_score', ascending=False)

    # Combine them: Matches on top, Skips on the bottom
    df_final = pd.concat([df_matches, df_skipped], ignore_index=True)
    
    if df_final.empty:
        print("🤷‍♂️ Something went wrong. No data to save.")
        return

    columns_to_keep = ['site', 'title', 'company', 'location', 'job_type', 'job_url', 'date_posted', 'ml_similarity_score', 'numeric_score', 'rejection_category', 'missing_skills', 'ai_analysis']
    df_clean = df_final[columns_to_keep]
    
    if os.path.exists(OUTPUT_FILE):
        df_clean.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    else:
        df_clean.to_csv(OUTPUT_FILE, index=False)
        
    print(f"\n🎉 Success! Appended {len(df_matches)} matches and {len(df_skipped)} skipped jobs with reasoning to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    run_pipeline()