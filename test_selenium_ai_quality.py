import pytest
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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
def driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=options)
    yield driver
    driver.quit()

# Test 1 
def test_error_message_quality(driver):
    driver.get("https://www.saucedemo.com")
    
    # Use Selenium to trigger error
    driver.find_element(By.CSS_SELECTOR, "[data-test='username']").send_keys("wrong_user")
    driver.find_element(By.CSS_SELECTOR, "[data-test='password']").send_keys("wrong_pass")
    driver.find_element(By.CSS_SELECTOR, "[data-test='login-button']").click()
    
    wait = WebDriverWait(driver, 10)
    error = wait.until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "[data-test='error']")))
    error_text = error.text
    
    print(f"\n📋 Error message: {error_text}")
    
    test_case = LLMTestCase(
        input="What went wrong during login?",
        actual_output=error_text,
        expected_output="A clear error message explaining login failed"
    )
    
    metric = AnswerRelevancyMetric(threshold=0.5, model=GroqEvaluator())
    metric.measure(test_case)
    
    print(f"✅ Error message quality score: {metric.score}")
    print(f"   Reason: {metric.reason}")
    assert metric.score >= 0.5, f"Error message quality too low: {metric.score}"

# Test 2 
def test_product_names_quality(driver):
    driver.get("https://www.saucedemo.com")
    
    # Login with Selenium
    driver.find_element(By.CSS_SELECTOR, "[data-test='username']").send_keys("standard_user")
    driver.find_element(By.CSS_SELECTOR, "[data-test='password']").send_keys("secret_sauce")
    driver.find_element(By.CSS_SELECTOR, "[data-test='login-button']").click()
    
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "[data-test='inventory-item']")))
    
    products = driver.find_elements(By.CSS_SELECTOR, "[data-test='inventory-item-name']")
    product_names = [p.text for p in products]
    products_text = ", ".join(product_names)
    
    print(f"\n📋 Products found: {products_text}")
    
    # Use DeepEval to verify product names are real product names
    test_case = LLMTestCase(
        input="Are these valid e-commerce product names?",
        actual_output=products_text,
        expected_output="Real product names that make sense for an e-commerce store"
    )
    
    metric = AnswerRelevancyMetric(threshold=0.5, model=GroqEvaluator())
    metric.measure(test_case)
    
    print(f"✅ Product names quality score: {metric.score}")
    assert metric.score >= 0.5

# Test 3 
def test_page_title_quality(driver):
    driver.get("https://www.saucedemo.com")
    
    # Login
    driver.find_element(By.CSS_SELECTOR, "[data-test='username']").send_keys("standard_user")
    driver.find_element(By.CSS_SELECTOR, "[data-test='password']").send_keys("secret_sauce")
    driver.find_element(By.CSS_SELECTOR, "[data-test='login-button']").click()
    
    wait = WebDriverWait(driver, 10)
    title = wait.until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "[data-test='title']")))
    title_text = title.text
    
    print(f"\n📋 Page title: {title_text}")
    
    test_case = LLMTestCase(
        input="Is this a good page title for a products listing page?",
        actual_output=title_text,
        expected_output="A clear title indicating this is a products page"
    )
    
    metric = AnswerRelevancyMetric(threshold=0.5, model=GroqEvaluator())
    metric.measure(test_case)
    
    print(f"✅ Page title quality score: {metric.score}")
    assert metric.score >= 0.5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])