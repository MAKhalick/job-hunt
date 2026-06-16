# 🎯 AI-Powered Job Hunt Automation Pipeline

A fully automated, serverless Data Engineering pipeline that sources, evaluates, and tracks high-quality job applications while aggressively filtering out spam and ghost postings. 

## 🧠 The Problem & The Solution

**The Problem:** The modern job hunt is incredibly inefficient. Previously, I was spending hours manually applying different filters and search queries across multiple job boards just to aggregate a list of roles. From there, I had to manually read through each description to determine if the job actually aligned with my specific data engineering skills and interests. Furthermore, the market is flooded with ghost postings, staffing agencies and companies that never respond, meaning hours are wasted applying into a void.

**The Solution:** I built this automation pipeline to handle the entire discovery and evaluation phase for me. The script casts a wide net across LinkedIn, Indeed and Glassdoor, and then uses local machine learning to mathematically rank hundreds of roles against my hardcoded resume. 

To solve the issue of fake or unresponsive companies, I engineered a **"Human-in-the-Loop" Company Directory**. Every time the script finds jobs, it cross-references a master database in Google Sheets. If I have flagged a specific company or recruiting agency as "Spam," the pipeline instantly drops their postings. This ensures that every morning, I wake up to a highly curated, AI-evaluated list of jobs from verified, direct companies.

## 💸 Zero-Cost Infrastructure & Quota Optimization

This entire pipeline is designed to operate at **zero cost** by exclusively leveraging free-tier cloud services (GitHub Actions, Google Cloud, and the Gemini API). 

To protect strict daily API limits and prevent token wastage, the architecture uses a two-stage funnel. Instead of sending every scraped job to the LLM, the system uses a lightweight, local NLP model to pre-filter and score hundreds of jobs offline. Only the absolute top 15 highest-probability matches are sent to the cloud AI, ensuring maximum efficiency and zero wasted API requests.

## 🛠️ Tech Stack

I built this project using a mix of traditional data engineering libraries and modern AI tools:

* **Core Logic:** Python 3.12
* **Data Extraction & Manipulation:** `python-jobspy` (for multi-portal scraping) and `pandas` (for cross-referencing, deduplication, and data cleaning).
* **Machine Learning (Semantic Search):** `sentence-transformers` (`all-MiniLM-L6-v2`) via Hugging Face. Used locally to convert my resume and job descriptions into vector embeddings to calculate cosine similarity.
* **LLM Analysis:** Google GenAI SDK (Gemini 2.5 Flash) for deep, qualitative evaluation of job descriptions, skill gap analysis, and cover letter generation.
* **Database & Cloud Integration:** Google Sheets API (`gspread` & `google-auth`) functioning as a serverless database.
* **CI/CD & Orchestration:** GitHub Actions running on a daily cron schedule with built-in retry logic and exponential backoff to handle temporary API rate limits.

## 🔄 System Architecture: How It Works

The pipeline executes daily in six distinct phases:

1. **Database Initialization:** Authenticates securely with Google Cloud and pulls down my historical application data and Master Company Directory.
2. **Scraping:** Fetches up to 60 recent job postings across Canada matching my target keywords.
3. **Cleaning & Filtering:** Drops cross-platform duplicates, previously analyzed jobs and instantly filters out any roles from companies I have flagged as "Spam" in my directory.
4. **Semantic Pre-Filtering:** Runs the surviving jobs through a local NLP model to rank them by semantic relevance against my resume, isolating the Top 15 roles to protect API quotas.
5. **AI Evaluation:** Sends the Top 15 jobs to Google Gemini. Gemini acts as an expert recruiter, strictly enforcing hard rules (e.g., rejecting bilingual-only roles), assigning a 0-100% fit score, identifying missing skills, classifying the company sector and drafting a tailored cold-email opening.
6. **Database Sync:** Pushes the final evaluated jobs to my Google Sheet tracker and appends any brand-new companies it discovered to the "Pending" tab of my directory for my review.

## 🚀 Setup & Deployment

### 1. Prerequisites
* Google Cloud Platform (GCP) Service Account (with Sheets & Drive API enabled)
* Google Gemini API Key
* A Google Sheet with two specific tabs

### 2. Google Sheets Configuration
The Google Sheet must contain two worksheets:
* **Job Postings:** Tracks the AI-analyzed jobs.
* **Company Directory:** Must have headers: `Company Name` | `Status` | `Sector`. (Configure Data Validation on the `Status` column for: `Pending`, `Approved`, `Spam`).

### 3. Repository Secrets
Navigate to your GitHub Repository -> **Settings** -> **Secrets and variables** -> **Actions** and add:
* `GCP_CREDENTIALS`: The raw JSON string from your Google Cloud Service Account.
* `GEMINI_API_KEY`: Your personal Gemini API key.

### 4. Daily Operation
Once pushed, the GitHub Action runs completely serverless every day at 11:05 AM EDT. My only daily task is to open the spreadsheet, apply to the highly-scored jobs, and flip any garbage companies in the directory from `Pending` to `Spam`.