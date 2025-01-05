#!/usr/bin/env python3
"""
Usage:
  python script.py --task "promote mice ownership" --iterations 5
"""

import argparse
import configparser
import re
import time
import html2text

from loguru import logger
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

try:
    from langchain_core.messages import (
        SystemMessage,
        HumanMessage,
        AIMessage,
        trim_messages
    )
except ImportError:
    logger.error("You need a newer version of langchain that includes langchain_core.messages.")
    raise

from pydantic import BaseModel, Field
from langchain_openai.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

KEY_INI_TEMPLATE = '''[KEYS]
openai = [YOUR API KEY]'''


class SeleniumAction(BaseModel):
    action: str = Field(description="Allowed actions: navigate, click, input, extract")
    selector: str = Field(default="", description="CSS (or XPath) selector")
    value: str = Field(default="", description="Value to use, e.g. URL or input text")


parser = PydanticOutputParser(pydantic_object=SeleniumAction)


def extract_text(driver):
    html_parser = html2text.HTML2Text()
    html_parser.single_line_break = True
    html_parser.images_to_alt = True
    html_parser.inline_links = False

    html_source = driver.page_source
    return html_parser.handle(html_source)


def find_clickable_elements(driver):
    """
    Return a list of descriptive strings for clickable elements on the current page.
    Each string includes tag name, class, id, and text (where applicable).
    """
    # Common selectors for potentially clickable elements
    clickable_selector = (
        "a[href], "
        "button, "
        "[role='button'], "
        "input[type='button'], "
        "input[type='submit'], "
        "input[type='image'], "
        "[onclick]"
    )

    # Find candidate elements
    candidates = driver.find_elements(By.CSS_SELECTOR, clickable_selector)

    # Build descriptions for visible & enabled elements
    descriptions = []
    for element in candidates:
        if element.is_displayed() and element.is_enabled():
            tag_name = element.tag_name or "(no tag)"
            class_attr = element.get_attribute("class") or "(no class)"
            id_attr = element.get_attribute("id") or "(no id)"
            text_attr = element.text.strip() or "(no text)"

            description = f"tag={tag_name}, class={class_attr}, id={id_attr}, text={text_attr}"
            descriptions.append(description)

    return 'Found clickable elements:\n' + '\n'.join(descriptions)


def find_input_elements(driver):
    """
    Return a list of descriptive strings for input-like elements on the current page.
    Each string includes tag name, class, id, and type/placeholder (where applicable).
    Only visible and enabled elements are included.
    """
    # Common selectors for input elements
    input_selector = (
        "input, "  # <input>
        "textarea, "  # <textarea>
        "select"  # <select>
    )

    # Find candidate elements
    candidates = driver.find_elements(By.CSS_SELECTOR, input_selector)

    # Build descriptions for visible & enabled elements
    descriptions = []
    for element in candidates:
        if element.is_displayed() and element.is_enabled():
            tag_name = element.tag_name or "(no tag)"
            class_attr = element.get_attribute("class") or "(no class)"
            id_attr = element.get_attribute("id") or "(no id)"

            # For <input> tags, we often want the 'type' attribute
            type_attr = element.get_attribute("type") or "(no type)"

            # For <textarea> or <select>, 'type' doesn't apply, but we might show placeholder or name
            # Just as an example, let's also capture the placeholder if present
            placeholder_attr = element.get_attribute("placeholder") or "(no placeholder)"

            # Build a unified description
            # If it's an <input> tag, we show the type
            # If it's not <input>, 'type' often won't matter, but we'll show it anyway for consistency
            description = (
                f"tag={tag_name}, "
                f"class={class_attr}, "
                f"id={id_attr}, "
                f"type={type_attr}, "
                f"placeholder={placeholder_attr}"
            )

            descriptions.append(description)

    return 'Found input elements:\n' + '\n'.join(descriptions)


def press_enter(driver, css_selector):
    """
    Znajduje element za pomocą selektora CSS i wysyła mu klawisz Enter.
    """
    element = driver.find_element(By.CSS_SELECTOR, css_selector)
    element.send_keys(Keys.ENTER)


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
            return f"{secs} seconds have elapsed"
        elif action_data.action == "find_clickable":
            return find_clickable_elements(driver)
        elif action_data.action == "find_input":
            return find_input_elements(driver)
        elif action_data.action == "press_enter":
            element = driver.find_element(By.CSS_SELECTOR, action_data.selector)
            element.click()
            time.sleep(1)
            return "Press enter successful"
        elif action_data.action == "extract_text":
            return extract_text(driver)
        else:
            return f"Unknown action: {action_data.action}"
    except Exception as e:
        # Convert to string
        error_text = str(e)
        # If the string contains "Stacktrace:", chop it off
        error_text = re.sub(r"Stacktrace:.*", "", error_text, flags=re.DOTALL).strip()

        return f"Error executing Selenium action: {error_text}"


def main():
    # A) Parse CLI arguments
    arg_parser = argparse.ArgumentParser(description="Autonomous Selenium Assistant (new trim_messages approach)")
    arg_parser.add_argument("-t", "--task", required=True, help="Final goal/task for the assistant.")
    arg_parser.add_argument("-i", "--iterations", type=int, default=5, help="Number of LLM iterations allowed.")
    args = arg_parser.parse_args()

    custom_user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/102.0.5005.115 Safari/537.36"
    )

    # B) Initialize headless Selenium WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument(f"--user-agent={custom_user_agent}")
    # Remove standard Selenium 'AutomationControlled' flag
    # to reduce detection (optional).
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=options)
    logger.info("Headless Chrome WebDriver started.")

    # PS) Initialize api key
    config = configparser.ConfigParser()
    config.read('key.ini')
    try:
        openai_key = config['KEYS']['openai']
    except KeyError:
        logger.error('No API key! Add api key to key.ini')
        with open('key.ini', 'w') as f:
            f.write(KEY_INI_TEMPLATE)
        exit(-1)

    # C) Build an initial conversation history (list of messages)
    #    We'll keep the system instructions as the first item.
    conversation_history = [
        SystemMessage(
            content=(
                "You are a Selenium browser automation assistant.\n"
                f"Your final goal: {args.task}\n\n"
                "You must respond ONLY in valid JSON (no extra text) with the schema:\n"
                "{\n"
                "  \"action\": \"navigate|extract_text|find_clickable|find_input|press_enter|click|input|extract|sleep\",\n"
                "  \"selector\": \"<CSS selector>\",\n"
                "  \"value\": \"<URL/text/seconds if needed>\"\n"
                "}\n"
                "Example:\n"
                "{\"action\": \"navigate\", \"value\": \"https://duckduckgo.com/\"}\n"
                "Do not use the following websites as selenium can't handle them:\n"
                "google.com"
            )
        )
    ]

    # We'll define a simpler function to pass trimmed history to the LLM.
    def call_llm_with_trimmed_history(llm, conversation_history, user_text, max_tokens=3000):
        """
        1) Append a HumanMessage(user_text) to conversation_history.
        2) Trim the entire conversation with `trim_messages` so we don't blow up token usage.
        3) Call the LLM with the trimmed conversation.
        4) Return the LLM's text output as a string.
        5) Also append the LLM's response as an AIMessage to conversation_history.
        """
        # Step 1: Append user message
        conversation_history.append(HumanMessage(content=user_text))

        # Step 2: Trim the conversation
        # This is a simplified approach: we count messages with `len()`.
        # For real usage, pass a token_counter that uses model to measure tokens or a more advanced approach.
        trimmed = trim_messages(
            conversation_history,
            token_counter=len,
            max_tokens=10,        # e.g. keep 10 messages total
            strategy="last",      # keep last messages
            start_on="human",     # ensure we don't break the sequence in the middle
            include_system=True,  # we want to keep the system prompt if possible
            allow_partial=False,
        )

        # Step 3: Actually call the model
        # We'll use `ChatOpenAI.invoke` or `.predict_messages`.
        # We'll pass `trimmed` as a list of messages.
        response = llm.invoke(trimmed)
        llm_text = response.content

        # Step 4: Return the LLM's text
        # Step 5: Append the LLM's message to conversation_history
        conversation_history.append(AIMessage(content=llm_text))

        return llm_text

    # D) Initialize our LLM
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.0, openai_api_key=openai_key)  # or any other model

    # E) Let's track the last Selenium result to show LLM
    last_selenium_result = "No actions performed yet."

    # F) Main loop
    try:
        for step in range(1, args.iterations + 1):
            logger.info(f"--- Iteration {step}/{args.iterations} ---")

            # We'll ask the LLM: "Give me next command. (Here is the last result...)"
            user_message_text = (
                f"Please provide your next Selenium command.\n"
                f"Last Selenium result: {last_selenium_result}"
            )

            # 1) Call LLM with trimmed conversation
            llm_output = call_llm_with_trimmed_history(llm, conversation_history, user_message_text)
            logger.info("LLM Output:\n{}", llm_output)

            # 2) Parse JSON from the LLM
            try:
                action_data = parser.parse(llm_output)
            except Exception as e:
                logger.error(f"Error: LLM response is not valid JSON: {e}")
                break

            # 3) Execute the Selenium action
            last_selenium_result = execute_selenium_action(action_data, driver)
            logger.info("Selenium Result:\n{}", last_selenium_result)

            # If there's a condition to stop early (e.g. if the LLM says "stop"), you can break here.
            # e.g. if action_data.action == "stop": break

        logger.info("Loop completed normally.")
    finally:
        # G) Cleanup
        driver.quit()
        logger.info("Browser session ended.")


if __name__ == "__main__":
    main()
