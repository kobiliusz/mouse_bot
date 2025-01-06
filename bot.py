#!/usr/bin/env python3
"""
Usage:
  python script.py --task "promote mice ownership" --iterations 100
"""

import argparse
import configparser
import re
import time
import boilerpy3
import tiktoken
import requests
import urllib
import json

from loguru import logger
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from pydantic import BaseModel, Field
from langchain_openai.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

MODEL_NAME = "gpt-4o"

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


KEY_INI_TEMPLATE = '''[KEYS]
openai = [YOUR API KEY]
serpstack = [YOUR SERPSTACK KEY]'''


# Initialize api key
config = configparser.ConfigParser()
config.read('key.ini')
try:
    OPENAI_KEY = config['KEYS']['openai']
    SERPSTACK_KEY = config['KEYS']['serpstack']
except KeyError:
    logger.error('No API key! Add api key to key.ini')
    with open('key.ini', 'w') as f:
        f.write(KEY_INI_TEMPLATE)
    exit(-1)


class SeleniumAction(BaseModel):
    action: str = Field(description="Allowed actions: navigate, click, input, extract")
    selector: str = Field(default="", description="CSS (or XPath) selector")
    value: str = Field(default="", description="Value to use, e.g. URL or input text")


parser = PydanticOutputParser(pydantic_object=SeleniumAction)


def search_google(keywords):
    api_result = json.loads(requests.get(
        f'http://api.serpstack.com/search?access_key={SERPSTACK_KEY}&query={urllib.parse.quote_plus(keywords)}').text)
    organic_results = api_result['organic_results']
    processed_results = [{k: r[k] for k in {'title', 'url'}} for r in organic_results]
    return json.dumps(processed_results)


def extract_text(driver):
    html_source = driver.page_source
    extractor = boilerpy3.extractors.ArticleExtractor()
    return extractor.get_content(html_source)


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
    candidates = driver.find_elements(By.XPATH, clickable_selector)

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
    candidates = driver.find_elements(By.XPATH, input_selector)

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
    element = driver.find_element(By.XPATH, css_selector)
    element.send_keys(Keys.ENTER)


def execute_selenium_action(action_data: SeleniumAction, driver, default_sleep: int, max_tokens: int) -> str:
    """Perform the requested Selenium action and return the result or error message."""
    try:
        if action_data.action == "navigate":
            driver.get(action_data.value)
            return "Navigation successful"
        elif action_data.action == "click":
            element = driver.find_element(By.XPATH, action_data.selector)
            element.click()
            return "Click successful"
        elif action_data.action == "input":
            element = driver.find_element(By.XPATH, action_data.selector)
            element.send_keys(action_data.value)
            return "Input successful"
        elif action_data.action == "extract":
            element = driver.find_element(By.XPATH, action_data.selector)
            # Truncate text if it's too large
            return truncate_text_to_n_tokens(element.text, max_tokens, MODEL_NAME, 0)
        elif action_data.action == "sleep":
            secs = int(action_data.value)
            logger.info(f'Sleeping for {secs} seconds.')
            time.sleep(secs)
            return f"{secs + default_sleep} seconds have elapsed"
        elif action_data.action == "find_clickable":
            return truncate_text_to_n_tokens(find_clickable_elements(driver), max_tokens, MODEL_NAME, 0)
        elif action_data.action == "find_input":
            return truncate_text_to_n_tokens(find_input_elements(driver), max_tokens, MODEL_NAME, 0)
        elif action_data.action == "web_search":
            return truncate_text_to_n_tokens(search_google(action_data.value), max_tokens, MODEL_NAME, 0)
        elif action_data.action == "press_enter":
            element = driver.find_element(By.XPATH, action_data.selector)
            element.click()
            return "Press enter successful"
        elif action_data.action == "extract_article_text":
            omit = int(action_data.value)
            extracted_text = extract_text(driver)
            extracted_text = truncate_text_to_n_tokens(extracted_text, max_tokens, MODEL_NAME, omit)
            return extracted_text
        else:
            return f"Unknown action: {action_data.action}"
    except Exception as e:
        error_text = process_error(e)
        return f"Error executing Selenium action: {error_text}"


def process_error(e):
    # Convert to string
    error_text = str(e)
    # If the string contains "Stacktrace:", chop it off
    error_text = re.sub(r"Stacktrace:.*", "", error_text, flags=re.DOTALL).strip()
    return error_text


def truncate_text_to_n_tokens(text: str, n: int, model_name: str, omit_tokens: int) -> str:
    """
    Przycina tekst tak, aby mieścił się w maksymalnie n tokenach (dla zadanego modelu).
    Zwraca przycięty tekst.
    """
    tokenizer = tiktoken.encoding_for_model(model_name)
    tokens = tokenizer.encode(text)
    truncated_tokens = tokens[omit_tokens:n]
    truncated_text = tokenizer.decode(truncated_tokens)
    return truncated_text


def main():
    # A) Parse CLI arguments
    arg_parser = argparse.ArgumentParser(description="Autonomous Selenium Assistant (new trim_messages approach)")
    arg_parser.add_argument("-t", "--task", required=True, help="Final goal/task for the assistant.")
    arg_parser.add_argument("-i", "--iterations", type=int, default=100, help="Number of LLM iterations allowed.")
    arg_parser.add_argument("-p", "--pause", type=int, default=60, help="Pause between requests.")
    arg_parser.add_argument("-m", "--max-tokens", type=int, default=2000, help="Maximum tokens in a single response.")
    arg_parser.add_argument("-r", "--remembered", type=int, default=8, help="Chat memory capacity in messages.")
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

    # C) Build an initial conversation history (list of messages)
    #    We'll keep the system instructions as the first item.
    conversation_history = [
        SystemMessage(
            content=(
                "You are an autonomous Selenium browser automation assistant.\n"
                f"Your final goal is to {args.task}.\n"
                f"It's important that this task is achieved in the actual internet, not just conversation logs.\n"

                "## Response Format:\n"
                "You must respond ONLY in valid JSON (no extra text!) with the following schema:\n"
                "{\n"
                "  \"action\": \"web_search|navigate|extract_article_text|find_clickable|find_input|press_enter|click|input|extract|stop|sleep\",\n"
                "  \"selector\": \"<CSS selector>\",\n"
                "  \"value\": \"<URL/text/number_of_seconds/number_of_tokens if needed>\"\n"
                "}\n\n"

                "### Action Descriptions:\n"
                "- **web_search**: Use ONLY if you do not know where to find required information (e.g., no known URL). Please do not overuse it. (put the search terms in \"value\")\n"
                "- **navigate**: Go to a given page (put the URL in \"value\").\n"
                "- **find_input**: Locate all texts field/inputs on the page (no css selector needed).\n"
                "- **input**: Type text into the input identified by XPath selector. (provide the text in \"value\").\n"
                "- **find_clickable**: Locate all clickable elements such as a button/link (no css selector needed).\n"
                "- **click**: Click the element identified by XPath selector.\n"
                "- **press_enter**: Press the Enter key in input identified by CSS selector.\n"
                "- **extract_article_text**: Extract the main text of the page (with the number of tokens to skip in \"value\").\n"
                "- **extract**: Extract plain text from the chosen element (XPath in \"selector\").\n"
                "- **stop**: If you're convinced that you already achieved your final goal, use this command to stop the conversation.\n"
                "- **sleep**: Wait for the specified number of seconds (put that in \"value\").\n\n"

                "## Execution Rules:\n"
                "1) Start by navigating to the relevant page if you know the URL. If you do not know it, only then use \"web_search\".\n"
                "2) Fill form fields with \"find_input\" and \"input\". Click buttons or links with \"find_clickable\" and \"click\".\n"
                "3) Continue these steps (only as many as needed) until you have **actually completed**: {args.task}.\n"
                "4) **Do not** remain stuck at searching: once you have enough info or a URL, proceed with the subsequent actions.\n"
                "5) When the task is finished, **end** the sequence. Do not add extra unnecessary steps.\n\n"

                "## Additional Constraints:\n"
                f"The response will be truncated to {args.max_tokens} tokens.\n"
                f"There will be an additional pause between requests of {args.pause} seconds.\n"
                f"Your conversation memory fits {args.remembered} messages.\n"
                "Do NOT use the following websites (Selenium can't handle them):\n"
                "google.com\n"
            )
        )
    ]

    # We'll define a simpler function to pass trimmed history to the LLM.
    def call_llm_with_trimmed_history(llm, conversation_history, user_text, memory_size):
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
            max_tokens=memory_size,        # e.g. keep 10 messages total
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
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.0, openai_api_key=OPENAI_KEY)  # or any other model

    # E) Let's track the last Selenium result to show LLM
    last_selenium_result = "No actions performed yet."

    # F) Main loop
    try:
        for step in range(1, args.iterations + 1):
            logger.info(f"--- Iteration {step}/{args.iterations} ---")

            # We'll ask the LLM: "Give me next command. (Here is the last result...)"
            user_message_text = (
                f'Iteration {step} of {args.iterations}\n'
                f"Please provide your next Selenium command.\n"
                f"Last Selenium result: {last_selenium_result}"
            )

            # 1) Call LLM with trimmed conversation
            llm_output = call_llm_with_trimmed_history(llm, conversation_history, user_message_text, args.remembered)
            logger.info("LLM Output:\n{}", llm_output)

            # 2) Parse JSON from the LLM
            try:
                action_data = parser.parse(llm_output)
                # 3) Execute the Selenium action
                if action_data.action == "stop":
                    logger.info("The LLM decided to stop execution.")
                    return
                last_selenium_result = execute_selenium_action(action_data, driver, args.pause, args.max_tokens)
            except Exception as e:
                logger.error(f"Error: LLM response is not valid JSON: {e}")
                last_selenium_result = process_error(e)

            logger.info("Truncated Selenium Result:\n{}", last_selenium_result)

            # sleep
            logger.info(f'Sleeping for {args.pause} seconds.')
            time.sleep(args.pause)

            # If there's a condition to stop early (e.g. if the LLM says "stop"), you can break here.
            # e.g. if action_data.action == "stop": break

        logger.info("Loop completed normally.")
    finally:
        # G) Cleanup
        driver.quit()
        logger.info("Browser session ended.")


if __name__ == "__main__":
    main()
