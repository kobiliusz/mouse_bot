#!/usr/bin/env python3
"""
Usage:
  python script.py --task "promote mice ownership" --iterations 5
"""

import argparse
import configparser
import time

from loguru import logger
from selenium import webdriver
from selenium.webdriver.common.by import By

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from pydantic import BaseModel, Field

KEY_INI_TEMPLATE = '''[KEYS]
openai = [YOUR API KEY]'''


class SeleniumAction(BaseModel):
    action: str = Field(description="Allowed actions: navigate, click, input, extract")
    selector: str = Field(default="", description="CSS (or XPath) selector")
    value: str = Field(default="", description="Value to use, e.g. URL or input text")


parser = PydanticOutputParser(pydantic_object=SeleniumAction)


def execute_selenium_action(action_data: SeleniumAction, driver) -> str:
    """Perform the requested Selenium action and return the result or error message."""
    try:
        if action_data.action == "navigate":
            driver.get(action_data.value)
            time.sleep(2)
            return "Navigation successful"
        elif action_data.action == "click":
            element = driver.find_element(By.CSS_SELECTOR, action_data.selector)
            element.click()
            time.sleep(1)
            return "Click successful"
        elif action_data.action == "input":
            element = driver.find_element(By.CSS_SELECTOR, action_data.selector)
            element.send_keys(action_data.value)
            time.sleep(1)
            return "Input successful"
        elif action_data.action == "extract":
            element = driver.find_element(By.CSS_SELECTOR, action_data.selector)
            # Truncate text if it's too large
            return element.text[:2000]
        elif action_data.action == "sleep":
            secs = int(action_data.value)
            time.sleep(secs)
            return "{secs} seconds have elapsed"
        else:
            return f"Unknown action: {action_data.action}"
    except Exception as e:
        return f"Error executing Selenium action: {e}"


def main():
    # --------------------------
    # Parse CLI arguments
    # --------------------------
    arg_parser = argparse.ArgumentParser(description="Autonomous Selenium Assistant.")
    arg_parser.add_argument("-t", "--task", required=True, help="Final goal/task for the assistant.")
    arg_parser.add_argument("-i", "--iterations", type=int, default=5, help="Number of LLM iterations allowed.")
    args = arg_parser.parse_args()

    # --------------------------
    # Load OpenAI API key
    # --------------------------
    config = configparser.ConfigParser()
    config.read('key.ini')
    try:
        openai_key = config['KEYS']['openai']
    except KeyError:
        logger.error('OpenAI API key missing! Enter your key in key.ini')
        with open('key.ini', 'w') as f:
            f.write(KEY_INI_TEMPLATE)
        exit(-1)

    # --------------------------
    # Initialize Selenium WebDriver
    # --------------------------
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    logger.info("Headless Chrome WebDriver started.")

    # --------------------------
    # Create a short-memory buffer to limit token usage
    # --------------------------
    memory = ConversationBufferWindowMemory(
        k=3,  # keep last 3 exchanges in memory
        return_messages=True
    )

    # --------------------------
    # Build a system prompt
    # --------------------------
    system_template = """
You are a Selenium browser automation assistant. Your final goal: {task}.
You have up to {iterations} steps (iterations) to achieve this goal.

Respond ONLY in valid JSON matching this schema:
{
  "action": "navigate|click|input|extract|sleep",
  "selector": "<CSS selector>",
  "value": "<value, e.g. URL or text or number of seconds>"
}

No explanations or extra text outside the JSON.
Example:
{"action": "navigate", "value": "https://google.com"}
""".strip()

    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_prompt = HumanMessagePromptTemplate.from_template("{human_input}")

    # We insert a placeholder for conversation memory in between system & human prompts
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_prompt, *memory.prompt_messages, human_prompt]
    )

    # Format the system message with our arguments
    system_prompt_content = system_prompt.format(task=args.task, iterations=args.iterations)
    logger.info(f"System Prompt for LLM:\n{system_prompt_content}")

    # --------------------------
    # Initialize the LLM chain with memory
    # --------------------------
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, openai_api_key=openai_key)
    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        memory=memory,
        verbose=False
    )

    # We "prime" the memory by manually adding the system prompt
    memory.chat_memory.add_message({"role": "system", "content": system_prompt_content})

    # No Selenium result yet
    last_selenium_result = "No actions performed yet."

    # --------------------------
    # Autonomous loop
    # --------------------------
    try:
        for step in range(1, args.iterations + 1):
            logger.info(f"--- Iteration {step}/{args.iterations} ---")

            # Our "human" input to the chain:
            # We tell the LLM to provide the next command,
            # and also show the result from the last iteration.
            human_input_text = (
                f"Please provide the next command.\n"
                f"Last Selenium result: {last_selenium_result}"
            )

            # Query the LLM chain
            llm_output = chain.run(human_input=human_input_text)
            logger.info(f"LLM Output: {llm_output}")

            # Parse JSON from LLM output
            try:
                action_data = parser.parse(llm_output)
            except Exception as e:
                logger.error(f"Error parsing LLM output as JSON: {e}")
                break

            # Execute Selenium action
            last_selenium_result = execute_selenium_action(action_data, driver)
            logger.info(f"Selenium result: {last_selenium_result}")

            # Optional: Decide if we have reached the goal or if we should break early.
            # For now, we just continue until we exhaust all iterations.

        logger.info("All iterations completed.")
    finally:
        driver.quit()
        logger.info("Browser session ended.")


if __name__ == "__main__":
    main()
