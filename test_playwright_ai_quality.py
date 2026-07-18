import pytest
import os
from playwright.sync_api import sync_playwright
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

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
        return self.model.invoke(prompt).content
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

@pytest.fixture
def page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        yield page
        browser.close()

# Test 1 
def test_error_message_quality(page):
    page.goto("https://www.saucedemo.com")
    page.fill("[data-test='username']", "wrong_user")
    page.fill("[data-test='password']", "wrong_pass")
    page.click("[data-test='login-button']")
    
    error_text = page.locator("[data-test='error']").text_content()
    print(f"\n📋 Error: {error_text}")
    
    test_case = LLMTestCase(
        input="What went wrong during login?",
        actual_output=error_text,
        expected_output="Clear error message explaining login failed"
    )
    
    metric = AnswerRelevancyMetric(threshold=0.5, model=GroqEvaluator())
    metric.measure(test_case)
    
    print(f"✅ Score: {metric.score} — {metric.reason}")
    assert metric.score >= 0.5

# Test 2 
def test_product_descriptions_quality(page):
    page.goto("https://www.saucedemo.com")
    page.fill("[data-test='username']", "standard_user")
    page.fill("[data-test='password']", "secret_sauce")
    page.click("[data-test='login-button']")
    
    descriptions = page.locator("[data-test='inventory-item-desc']").all()
    desc_texts = [d.text_content() for d in descriptions]
    combined = " | ".join(desc_texts[:3])  # first 3
    
    print(f"\n📋 Descriptions: {combined[:100]}...")
    
    test_case = LLMTestCase(
        input="Are these good e-commerce product descriptions?",
        actual_output=combined,
        expected_output="Clear, helpful product descriptions for an online store"
    )
    
    metric = AnswerRelevancyMetric(threshold=0.5, model=GroqEvaluator())
    metric.measure(test_case)
    
    print(f"✅ Score: {metric.score} — {metric.reason}")
    assert metric.score >= 0.5

# Test 3 
def test_cart_confirmation_quality(page):
    page.goto("https://www.saucedemo.com")
    page.fill("[data-test='username']", "standard_user")
    page.fill("[data-test='password']", "secret_sauce")
    page.click("[data-test='login-button']")
    
    # Add item and go to cart
    page.click("[data-test='add-to-cart-sauce-labs-backpack']")
    page.click("[data-test='shopping-cart-link']")
    
    cart_title = page.locator("[data-test='title']").text_content()
    item_name = page.locator("[data-test='inventory-item-name']").text_content()
    
    cart_content = f"Page: {cart_title}, Item: {item_name}"
    print(f"\n📋 Cart: {cart_content}")
    
    test_case = LLMTestCase(
        input="Is this a clear shopping cart page?",
        actual_output=cart_content,
        expected_output="Cart page showing items ready for checkout"
    )
    
    metric = AnswerRelevancyMetric(threshold=0.5, model=GroqEvaluator())
    metric.measure(test_case)
    
    print(f"✅ Score: {metric.score} — {metric.reason}")
    assert metric.score >= 0.5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])