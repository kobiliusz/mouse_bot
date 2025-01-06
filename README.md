# Mouse bot - Autonomous Selenium Assistant

This repository contains a Python script (`bot.py`) that uses an LLM (Large Language Model) to autonomously control a headless Chrome browser via Selenium. It can search the web, navigate to pages, fill out forms, click buttons, and parse text—all guided by the model’s responses in JSON format.

## Key Features

- **Headless Selenium**: Interact with a Chrome browser invisibly (no GUI).
- **LLM-Powered Automation**: The script prompts a Large Language Model to decide which actions to take (e.g., navigating, clicking, or extracting text).
- **Configurable**: Multiple CLI arguments let you set the task, iteration limits, pause durations, etc.
- **Flexible Integration**: The code is modular; you can adapt it to your own workflows or data-collection needs.

## Requirements

1. **Python 3.7+**
2. **Chrome / Chromium Browser** (installed locally)
3. **ChromeDriver** matching your local Chrome/Chromium version.
4. Python libraries (install via `pip install -r requirements.txt` if you create one):
   - `argparse`
   - `boilerpy3`
   - `configparser`
   - `loguru`
   - `requests`
   - `selenium`
   - `tiktoken`
   - `tqdm`
   - `pydantic`
   - `langchain-openai` (and its dependencies)
   - (Optional) `langchain` with a recent version that supports `langchain_core.messages`.

## Setup

1. **Install ChromeDriver**  
   - Make sure `chromedriver` is in your `PATH`, or specify the path to the executable as needed.

2. **Create `key.ini`**  
   - The script reads your OpenAI API key and SERPStack key from a `key.ini` file.
   - A template will be generated if `key.ini` is missing. It looks like this:
     ```ini
     [KEYS]
     openai = [YOUR API KEY]
     serpstack = [YOUR SERPSTACK KEY]
     ```
   - Replace the placeholders with your **actual** API keys.

3. **Install Python Dependencies**  
   - (Optional) Create a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install required libraries:
     ```bash
     pip install -r requirements.txt
     ```
     or manually:
     ```bash
     pip install boilerpy3 configparser loguru requests selenium tiktoken tqdm pydantic langchain-openai
     ```
   - If you see errors about `langchain_core.messages`, you may need a newer `langchain` version.

4. **Run the Script**  
   - Once everything is set up, you can run:
     ```bash
     python script.py --task "promote mice ownership" --iterations 100
     ```
   - Adjust arguments as needed (see **Usage** below).

## Usage

From the help text inside `bot.py`:

```bash
Usage:
  python script.py --task "promote mice ownership" --iterations 100

Optional arguments:
  -t, --task            The final goal or task for the assistant.
  -i, --iterations      Number of LLM iterations allowed (default: 100).
  -p, --pause           Pause (in seconds) between Selenium actions (default: 60).
  -m, --max-tokens      Maximum tokens the LLM can use per response (default: 2000).
  -r, --remembered      Number of messages to keep in conversation memory (default: 8).
```

### Example

```bash
python bot.py \
    --task "automate blog login and comment" \
    --iterations 10 \
    --pause 30 \
    --max-tokens 1500 \
    --remembered 5
```

## How It Works

1. **Initialization**  
   - Loads keys from `key.ini`.  
   - Sets up a headless Chrome browser using Selenium.  

2. **LLM Interaction**  
   - The script keeps a conversation history with a **System Prompt** (explaining how to respond in JSON and what actions are available).
   - Each iteration:
     1. **Sends** the last Selenium result (or “No actions performed yet.” on first run) to the LLM, prompting for the next action.
     2. **Parses** the LLM’s JSON response (e.g., `{"action": "click", "selector": "...", "value": ""}`).
     3. **Executes** that action in the headless browser.
     4. **Pauses** for a specified time (`--pause`).
   - This cycle repeats up to `--iterations` times or until the LLM sends `{"action": "stop"}`.

3. **Actions**  
   - The LLM can request:
     - `navigate`: go to a new URL.
     - `click`: click a specific selector on the page.
     - `input`: type text into an element.
     - `web_search`: call a SERP API to find links (SERPStack).
     - `extract`: get text from a specific selector.
     - `extract_article_text`: parse “main article” text from the current page with [boilerpy3](https://github.com/kohlschutter/boilerpipe).
     - `find_clickable` and `find_input`: list clickable or input-like elements on the page.
     - `stop`: end the session.

4. **Truncation**  
   - The script uses `tiktoken` to keep output within the `--max-tokens` limit.
   - Also uses a custom `trim_messages` function to keep the conversation memory from blowing up.

## Troubleshooting

- **Missing `langchain_core.messages`:**  
  Update `langchain`. E.g. `pip install --upgrade langchain`.
- **ChromeDriver version mismatch**:  
  Make sure your local Chrome/Chromium version matches your ChromeDriver version.
- **API Key Issues**:  
  Double-check that `OPENAI_KEY` and `SERPSTACK_KEY` in `key.ini` are set correctly.
- **Invalid JSON**:  
  Sometimes the model might produce invalid JSON. The script will log an error. Consider tweaking prompts or retrying.

## Contributing

- Feel free to open issues or pull requests if you encounter bugs or have improvements to share.
- This script is a minimal working example—further enhancements (like better error handling, more actions, etc.) are welcome.

---

Enjoy exploring your own “autonomous Selenium assistant” driven by a Large Language Model!