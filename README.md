# Skillworth

**Skillworth** is a job scraping and skill matching tool that analyzes job postings to assess fit based on your existing skills. It uses NLP, regex, and fuzzy matching to extract skills, classify them, and score alignment.

## Features

- Scrapes job listings using the JSearch API  
- Extracts skills, must-haves, experience, education, and salary from descriptions  
- Scores job match based on personal skillset  
- Categorizes matched skills by functional group  
- Outputs a scored dataset with top skills, missing skills, and job-level insights  
- Optional visualization of most in-demand skills

## Usage

1. Add your skillset to `my_skills`  
2. Provide a JSearch API key  
3. Run `main()` to scrape and analyze job postings  
4. Outputs:
   - `Skillworth_Jobs.csv`  
   - Bar chart of top 10 in-demand skills

## Technologies

- Python: `pandas`, `scikit-learn`, `matplotlib`  
- NLP: `spaCy`, `regex`, fuzzy matching  
- API: JSearch via RapidAPI

## Note

This tool is shared for demonstration and skill development purposes. API usage may be rate-limited depending on your plan.
