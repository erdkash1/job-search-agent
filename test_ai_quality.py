import pytest
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric
)
from deepeval.test_case import LLMTestCase
from agent import generate_cover_letter, rank_jobs, search_jobs, filter_blocked


from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq
import os

class GroqEvaluator(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )
    
    def get_model_name(self):
        return "groq-llama-3.3-70b"
    
    def load_model(self):
        return self.model
    
    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

def test_cover_letter_relevancy():
    job = {
        "title": "QA Automation Engineer",
        "company": "T-Mobile",
        "location": "Remote",
        "skills": ["Selenium", "Java", "TestNG", "Cucumber"]
    }
    
    cover_letter = generate_cover_letter(job)
    
    test_case = LLMTestCase(
        input="Generate cover letter for QA Automation Engineer at T-Mobile",
        actual_output=cover_letter,
        expected_output="Cover letter mentioning Selenium, Java, QA, T-Mobile"
    )
    
    groq_evaluator = GroqEvaluator()
    metric = AnswerRelevancyMetric(threshold=0.7, model=groq_evaluator)
    metric.measure(test_case)
    
    print(f"\n✅ Relevancy Score: {metric.score}")
    print(f"   Reason: {metric.reason}")
    assert metric.score >= 0.7

def test_job_search_returns_results():
    results = search_jobs(["QA", "Selenium", "Java"])
    assert len(results) > 0, "Job search returned no results!"
    print(f"\n✅ Found {len(results)} jobs")

def test_blocked_companies_filtered():
    all_jobs, _ = filter_blocked(search_jobs(["Java"]))
    company_names = [job["company"].lower() for job in all_jobs]
    
    blocked = ["epam", "leidos", "saic"]
    for company in blocked:
        assert company not in company_names, \
            f"Blocked company {company} was not filtered!"
    
    print(f"\n✅ All blocked companies filtered correctly")

def test_ranking_returns_all_jobs():
    jobs = search_jobs(["QA", "Java"])
    allowed, _ = filter_blocked(jobs)
    ranked = rank_jobs(allowed)
    
    assert len(ranked) == len(allowed), \
        "Ranking lost some jobs!"
    print(f"\n✅ Ranking preserved all {len(ranked)} jobs")

def test_cover_letter_mentions_company():
    job = {
        "title": "SDET",
        "company": "Capital One",
        "location": "Remote",
        "skills": ["Playwright", "Python", "Java"]
    }
    
    cover_letter = generate_cover_letter(job)
    
    assert "Capital One" in cover_letter, \
        "Cover letter doesn't mention the company!"
    print(f"\n✅ Cover letter mentions Capital One")

if __name__ == "__main__":
    print("🧪 Running AI Quality Tests...")
    pytest.main([__file__, "-v"])