from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

JOBS = [
    {
        "title": "QA Automation Engineer",
        "company": "T-Mobile",
        "location": "Remote",
        "skills": ["Selenium", "Java", "TestNG", "Cucumber"],
        "url": "simplify.jobs/tmobile-qa"
    },
    {
        "title": "SDET",
        "company": "Capital One",
        "location": "Remote",
        "skills": ["Playwright", "Python", "RestAssured", "Java"],
        "url": "simplify.jobs/capitalone-sdet"
    },
    {
        "title": "Backend Java Developer",
        "company": "Visa",
        "location": "Austin, TX",
        "skills": ["Java", "Spring Boot", "Microservices", "PostgreSQL"],
        "url": "simplify.jobs/visa-java"
    },
    {
        "title": "QA Engineer",
        "company": "JPMorgan Chase",
        "location": "Remote",
        "skills": ["Selenium", "Java", "BDD", "Cucumber"],
        "url": "simplify.jobs/jpmorgan-qa"
    },
    {
        "title": "Software Engineer",
        "company": "EPAM",
        "location": "Remote",
        "skills": ["Java", "Spring Boot"],
        "url": "simplify.jobs/epam-swe"
    },
    {
        "title": "Software Engineer",
        "company": "Leidos",
        "location": "Remote",
        "skills": ["Java", "Security Clearance"],
        "url": "simplify.jobs/leidos-swe"
    }
]

# Blocked companies
BLOCKED = ["epam", "leidos", "saic", "booz allen", "caci",
           "peraton", "general dynamics", "amentum"]

# Candidate profile
CANDIDATE = """
Name: Erdenesuren Shirmen
Education: CS Graduate — Missouri State University (2026)
Skills: Java, Spring Boot, Selenium WebDriver, Playwright,
        RestAssured, Cucumber/BDD, TestNG, JUnit 5, PostgreSQL,
        Docker, AWS, Python, Kotlin
Projects:
  - Food Rescue Optimizer (Java, Spring Boot, ML, AWS ECS)
  - Ecommerce API (Spring Boot, PostgreSQL, JWT, Redis, Docker)
  - QA Portfolio (76 automated tests across Selenium + Playwright + RestAssured + Cucumber)
Status: OPT (STEM eligible — 3 years)
Targeting: QA Automation Engineer / SDET / Backend Java Developer
"""

def get_llm():
    """Initialize LLM lazily — only when needed."""
    api_key = os.environ.get("GROQ_API_KEY")  
    print(f"DEBUG: API key found: {api_key is not None}")  # ← add this

    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set!")
    return ChatGroq(
        api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
def search_jobs(keywords: list) -> list:
    """Search jobs by keywords."""
    results = []
    for job in JOBS:
        matches = any(
            keyword.lower() in job["title"].lower() or
            any(keyword.lower() in skill.lower() for skill in job["skills"])
            for keyword in keywords
        )
        if matches:
            results.append(job)
    return results

def filter_blocked(jobs: list) -> tuple:
    """Filter out blocked companies."""
    allowed = []
    blocked = []
    for job in jobs:
        if any(b in job["company"].lower() for b in BLOCKED):
            blocked.append(job)
        else:
            allowed.append(job)
    return allowed, blocked

def generate_cover_letter(job: dict) -> str:
    """Generate tailored cover letter using Groq."""
    llm = get_llm()
    response = llm.invoke(
        f"""Write a short professional cover letter for this job:

Job Title: {job['title']}
Company: {job['company']}
Location: {job['location']}
Required Skills: {', '.join(job['skills'])}

Candidate Profile:
{CANDIDATE}

Instructions:
- Keep it under 200 words
- Professional but personable tone
- Highlight most relevant skills for this specific role
- Mention OPT status briefly
- End with enthusiasm
"""
    )
    return response.content

def rank_jobs(jobs: list) -> list:
    """Rank jobs by relevance using AI."""
    if not jobs:
        return []

    llm = get_llm()

    jobs_text = "\n".join([
        f"{i+1}. {j['title']} at {j['company']} — Skills: {', '.join(j['skills'])}"
        for i, j in enumerate(jobs)
    ])

    response = llm.invoke(
        f"""Given this candidate profile:
{CANDIDATE}

Rank these jobs from best to worst fit (return just the numbers in order, comma separated):
{jobs_text}

Return only numbers like: 2,1,3"""
    )

    try:
        order = [int(x.strip()) - 1 for x in response.content.strip().split(",")]
        return [jobs[i] for i in order if i < len(jobs)]
    except:
        return jobs

def run_job_search(query: str):
    """Main job search agent function."""
    print(f"\n🔍 Searching for: {query}")
    print("=" * 50)

    keywords = query.split()
    found_jobs = search_jobs(keywords)
    print(f"✅ Found {len(found_jobs)} matching jobs")

    allowed_jobs, blocked_jobs = filter_blocked(found_jobs)
    if blocked_jobs:
        print(f"🚫 Filtered out {len(blocked_jobs)} blocked companies: "
              f"{', '.join(j['company'] for j in blocked_jobs)}")

    if not allowed_jobs:
        print("❌ No jobs remaining after filtering!")
        return

    print(f"\n🤖 Ranking {len(allowed_jobs)} jobs by fit...")
    ranked_jobs = rank_jobs(allowed_jobs)

    print(f"\n📋 Top Jobs For You:")
    print("=" * 50)
    for i, job in enumerate(ranked_jobs, 1):
        print(f"\n#{i} {job['title']} at {job['company']}")
        print(f"   📍 {job['location']}")
        print(f"   🛠  {', '.join(job['skills'])}")
        print(f"   🔗 {job['url']}")

    if ranked_jobs:
        top_job = ranked_jobs[0]
        print(f"\n✉️  Generating cover letter for top match...")
        print("=" * 50)
        cover_letter = generate_cover_letter(top_job)
        print(cover_letter)

if __name__ == "__main__":
    print("🤖 Job Search AI Agent")
    print("=" * 50)
    run_job_search("QA Automation Engineer Selenium Java")