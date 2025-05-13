import requests
import pandas as pd
import spacy
import re
from difflib import get_close_matches
from sklearn.preprocessing import MultiLabelBinarizer
import json
import matplotlib.pyplot as plt

"""
Skillworth: Job Matching Tool
Scrapes job listings from JSearch API and analyzes skill match using NLP, regex, and custom scoring.
For personal use, educational purposes, and portfolio demonstration only.
"""

nlp = spacy.load("en_core_web_sm")

query = ""
location = ""
API_KEY = ""

my_skills = {
    # Insert your skills here. Redacted for public
}


# Here are some options. Use whichever you need.
SKILL_KEYWORDS = {
    # Programming & Scripting
    "python", "r", "sql", "java", "scala", "bash", "javascript", "typescript", "html", "css", "c++", "c#", "sas", "stata",

    # Data Analysis & Statistics
    "data analysis", "data modeling", "statistics", "hypothesis testing", "a/b testing", "experiment design",
    "regression analysis", "time series", "probability", "anomaly detection", "trend analysis",

    # Machine Learning & AI
    "machine learning", "deep learning", "natural language processing", "nlp", "classification", "clustering",
    "recommendation systems", "model tuning", "xgboost", "random forest", "pca", "feature engineering",

    # BI Tools
    "tableau", "power bi", "looker", "superset", "qlik", "d3.js", "metabase", "excel", "dashboards", "reporting",

    # Data Engineering & Warehousing
    "data pipelines", "etl", "elt", "airflow", "dbt", "bigquery", "snowflake", "databricks", "redshift", "sql server",
    "postgres", "mysql", "nosql", "mongodb", "data warehousing", "data lakes", "cloud storage",

    # Cloud & DevOps
    "aws", "azure", "gcp", "cloud computing", "lambda", "s3", "docker", "kubernetes", "ci/cd", "terraform",

    # Collaboration & Stakeholder
    "stakeholder communication", "cross-functional collaboration", "teamwork", "mentoring", "stakeholder management",
    "agile", "scrum", "kanban", "product owner", "jira", "confluence",

    # Product & Business
    "product metrics", "product strategy", "customer segmentation", "lifetime value", "churn prediction",
    "conversion optimization", "pricing analysis", "forecasting", "kpis", "business analysis", "roi",

    # Visualization & UX
    "data visualization", "storytelling", "ux research", "user research", "heatmaps", "figma", "wireframing",

    # Communication & Writing
    "presentation skills", "technical writing", "documentation", "public speaking", "data storytelling",

    # Tools & Libraries
    "pandas", "numpy", "sklearn", "matplotlib", "seaborn", "spacy", "nltk", "plotly", "streamlit", "dash",

    # Soft Skills
    "problem solving", "strategic thinking", "attention to detail", "creativity", "adaptability", "initiative",
    "critical thinking", "curiosity", "growth mindset", "resilience",

    # Marketing & CRM
    "crm", "hubspot", "salesforce", "campaign analysis", "email automation", "retention strategy",

    # Finance & Economics
    "financial modeling", "cost analysis", "budget forecasting", "valuation", "unit economics", "p&l"
}

SKILL_GROUPS = {
    "programming": {"python", "r", "sql", "java", "scala", "bash", "c++", "c#", "javascript", "typescript"},
    "analysis": {"data analysis", "statistics", "regression analysis", "hypothesis testing", "trend analysis"},
    "ml_ai": {"machine learning", "deep learning", "natural language processing", "clustering", "classification"},
    "visualization": {"tableau", "power bi", "looker", "reporting", "dashboards", "data visualization", "storytelling"},
    "bi_tools": {"excel", "qlik", "metabase", "superset", "d3.js"},
    "data_eng": {"etl", "elt", "airflow", "dbt", "data pipelines", "data lakes", "data warehousing"},
    "cloud_devops": {"aws", "azure", "gcp", "docker", "kubernetes", "terraform", "lambda"},
    "databases": {"bigquery", "snowflake", "redshift", "postgres", "mysql", "mongodb", "sql server"},
    "product": {"product strategy", "product metrics", "conversion optimization", "lifetime value"},
    "collaboration": {"stakeholder communication", "cross-functional collaboration", "mentoring", "agile", "scrum"},
    "soft_skills": {"problem solving", "adaptability", "curiosity", "attention to detail", "strategic thinking"},
    "tools": {"pandas", "numpy", "sklearn", "matplotlib", "seaborn", "streamlit", "plotly"},
    "research_ux": {"user research", "ux research", "heatmaps", "figma", "wireframing"},
    "communication": {"presentation skills", "technical writing", "documentation", "public speaking"},
    "marketing": {"crm", "hubspot", "salesforce", "campaign analysis"},
    "finance": {"financial modeling", "valuation", "p&l", "cost analysis", "budget forecasting"},
}

ALIASES = {
    "sql scripting": "sql",
    "dashboards": "reporting",
    "data": "data analysis",
    "etl process": "etl",
    "communication skills": "stakeholder communication",
    "excel": "microsoft excel",
    "looker": "business intelligence",
    "bi tools": "business intelligence",
    "cloud data warehouse": "data warehousing",
    "user acceptance testing": "testing",
    "extract transform load": "etl",
    "crm tools": "crm",
    "etl pipeline": "data pipelines",
    "data storytelling": "storytelling",
    "data viz": "data visualization",
    "docs": "documentation",
    "cost forecasting": "budget forecasting"
}

pattern = r'''
    \$?\s?              # dollar sign and space
    (?:CAD\s?)?         # 'CAD' prefix
    \d{1,3}(?:,\d{3})*  #  number with commas
    (?:\.\d+)?          #  decimal
    (?:\s?(?:-|–|to)\s?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?)?  # in case of range, second value
    (?:\s?(?:/|per)\s?(hour|hr|day|week|month|year))?    # time unit
'''

def extract_skills_and_required(text, keywords=SKILL_KEYWORDS, threshold=0.75):
    if not isinstance(text, str):
        return {"skills": [], "must_have_skills": []}

    found = set()
    lower_text = text.lower()
    words = re.findall(r'\b\w+\b', lower_text)

    # Full-text fuzzy scan
    for skill in keywords:
        skill_lower = skill.lower()
        if skill_lower in lower_text:
            found.add(skill)
        else:
            close = get_close_matches(skill_lower, words, n=1, cutoff=0.75)
            
            if close:
                found.add(skill)

    # Normalize aliases
    all_skills = {ALIASES.get(skill, skill) for skill in found}

    # Look for "must-have" sections
    must_phrases = re.findall(
        r"(must have|required|qualifications include|requirements include|you (?:should|need|must) (?:to )?have|you must be able to|ideal candidate (?:will|should)?(?: have)?)"
        r"[^.]{0,200}",  
        lower_text,
        re.IGNORECASE,
    )

    must_found = set()
    for phrase in must_phrases:
        for skill in all_skills:
            if skill in phrase:
                must_found.add(skill)

    return {
        "skills": sorted(all_skills),
        "must_have_skills": sorted(must_found)
    }

def extract_experience(text):
    if not isinstance(text, str):
        return {"experience_text": "", "is_senior": False}

    match = re.search(r'\b(?:minimum\s*)?\d{1,2}\s*(?:\+?\s*-\s*|\s*to\s*|\+?\s*)?\d{0,2}\s*(?:\+)?\s*years?', text.lower())
    experience_text = match.group() if match else ""

    # Extract number and classify as senior
    num_match = re.search(r'\d+', experience_text)
    is_senior = int(num_match.group()) >= 5 if num_match else False

    return {
        "experience_text": experience_text,
        "is_senior": is_senior
    }

def has_bachelor_degree(text):
    if not isinstance(text, str):
        return False
    return bool(re.search(r"(bachelor'?s|undergraduate)\s+(degree|education)?", text.lower()))

def is_remote_fallback(job):
    text = " ".join([
        job.get("job_title", ""),
        job.get("job_location", ""),
        job.get("job_description", "")
    ]).lower()
    return "remote" in text or "work from home" in text

def extract_salary_from_text(text):
    if not isinstance(text, str):
        return ""
    matches = re.findall(pattern, text, re.IGNORECASE | re.VERBOSE)
    if not matches:
        return ""
    
    # Return only the most relevant looking match (e.g., has $ and a time unit)
    for m in matches:
        if re.search(r"\$", m) and re.search(r"(hour|year|month|week|day)", m.lower()):
            return m
    # Otherwise return the first match (fallback)
    return matches[0]

def get_jobs_jsearch(query, location, api_key, pages=1):
    all_jobs = []

    for page in range(pages):
        url = "https://jsearch.p.rapidapi.com/search"

        params = {
            "query": f"{query} in {location}",
            "page": str(page + 1),
        }

        headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        print(json.dumps(data['data'][0], indent=2))
        print(f"Status Code: {response.status_code}")

        jobs = data.get("data", [])
        if not jobs:
            print(f"Page {page+1}: 🚫 No results or limit reached.")
            break

        for job in jobs:
            all_jobs.append({
                "title": job.get("job_title"),
                "company": job.get("employer_name"),
                "location": job.get("job_city"),
                "country": job.get("job_country"),
                "remote": job.get("job_is_remote", False) or is_remote_fallback(job),
                "posted": job.get("job_posted_at"),
                "post_date": job.get("job_posted_at_datetime_utc"),
                "salary_text": job.get("job_salary", ""),
                "min_salary": job.get("job_min_salary", ""),
                "max_salary": job.get("job_max_salary", ""),
                "salary_period": job.get("job_salary_period", ""),
                "currency": job.get("job_salary_currency", ""),
                "description": job.get("job_description"),
                "link": job.get("job_google_link", ""),
                "highlights": job.get("job_highlights", "")
            })
    return pd.DataFrame(all_jobs)

def normalize_skills(skills, aliases):
    return {aliases.get(skill, skill) for skill in skills}

def categorize_skills(skills, skill_groups):
    categorized = {group: set() for group in skill_groups}
    for skill in skills:
        for group, keywords in skill_groups.items():
            if skill in keywords:
                categorized[group].add(skill)
    return categorized

def score_job_percent_match(job_skills, must_have_skills, my_skills, aliases, skill_groups, must_weight=1.5):
    if not job_skills:
        return {"percent_match": 0, "matched_skills": [], "by_group": {}, "must_have_matched": []}

    job_skills_norm = normalize_skills(set(job_skills), aliases)
    must_have_norm = normalize_skills(set(must_have_skills), aliases)
    my_skills_norm = normalize_skills(set(my_skills), aliases)

    matched = job_skills_norm & my_skills_norm
    must_matched = must_have_norm & my_skills_norm

    # Weight boost for must-have skills
    base_score = len(matched) / len(my_skills_norm)
    bonus = (must_weight - 1) * len(must_matched) / len(my_skills_norm)
    percent = round((base_score + bonus) * 100, 2)

    group_match = categorize_skills(matched, skill_groups)

    return {
        "percent_match": percent,
        "matched_skills": sorted(matched),
        "must_have_matched": sorted(must_matched),
        "by_group": {k: sorted(v) for k, v in group_match.items() if v}
    }

def main():
    df = get_jobs_jsearch("Data Analyst", "Toronto", API_KEY, pages=5)
    df = df[df['description'].notna()]
    exp_info = df['description'].apply(extract_experience)
    df['experience_required'] = exp_info.apply(lambda x: x['experience_text'])
    df['senior'] = exp_info.apply(lambda x: x['is_senior'])
    df = df[~df['senior']]  
    skill_results = df['description'].apply(extract_skills_and_required)
    df['skills'] = skill_results.apply(lambda x: x['skills'])
    df['must_have_skills'] = skill_results.apply(lambda x: x['must_have_skills'])
    df['bachelor_required'] = df['description'].apply(has_bachelor_degree)
    df['salary_guess'] = df['description'].apply(extract_salary_from_text)
    df.drop(columns=['description'], inplace=True)
    
    df['score_data'] = df.apply(
        lambda row: score_job_percent_match(
            row['skills'], row['must_have_skills'], my_skills, ALIASES, SKILL_GROUPS
        ),
        axis=1
    )

    df['percent_match'] = df['score_data'].apply(lambda x: x['percent_match'])
    df['matched_skills'] = df['score_data'].apply(lambda x: x['matched_skills'])
    df['must_have_matched'] = df['score_data'].apply(lambda x: x['must_have_matched'])
    df['match_by_group'] = df['score_data'].apply(lambda x: x['by_group'])
    df['match_by_group'] = df['match_by_group'].apply(json.dumps)
    df.drop(columns=['score_data'], inplace=True)  # optional cleanup

    mlb = MultiLabelBinarizer()
    skills_encoded = mlb.fit_transform(df['skills'])
    skills_df = pd.DataFrame(skills_encoded, columns=mlb.classes_, index=df.index)
    df_encoded = pd.concat([df.drop(columns=['skills']), skills_df], axis=1)
    
    skill_counts = skills_df.sum().sort_values(ascending=False)
    missing_skills = set(skill_counts.index) - my_skills
    print("Top Missing Skills You're Not Listing:")
    print(sorted(missing_skills & set(skill_counts.head(20).index)))
    
    df_encoded['skill_score'] = skills_encoded.sum(axis=1)
    df_encoded.sort_values(by="percent_match", ascending=False, inplace=True)
    df_encoded.to_csv("Skillworth_Jobs.csv", index=False)
    print(df_encoded.head())
    
    top_skills = skill_counts.head(10)
    plt.figure(figsize=(10, 5))
    top_skills.plot(kind='bar')
    plt.title("Top 10 Most In-Demand Skills")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
