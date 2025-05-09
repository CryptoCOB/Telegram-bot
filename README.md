# SulaGPT Telegram Bot

SulaGPT is an advanced Telegram bot designed to assist users with cryptocurrency, blockchain technology, AI, and prompt engineering. It leverages a local/remote Large Language Model (LLM) for intelligent responses, features wallet address collection, data export, scheduled messages, and more. The bot is built with `python-telegram-bot` and `httpx` for robust asynchronous operations.

A key feature is its unique LLM interaction model, "Multi-Resolution Adaptive Path (MRAP)," which uses three distinct AI agent personas (Captain Current, Navigator Nettle, Critique Coral) to provide comprehensive and structured answers.

## Key Features

*   **LLM Integration:** Connects to a local or remote LLM (e.g., via LM Studio, Ollama) for generating AI-powered responses.
*   **Multi-Agent Persona:** Uses a detailed system prompt with three AI agent personas for structured and insightful answers.
*   **Wallet Address Collection:** Detects and stores Ethereum, Solana, and Bitcoin wallet addresses sent by users.
*   **Data Export:** Allows exporting collected wallet addresses to a CSV file.
*   **Command Handling:** Supports a variety of commands (see "Bot Commands" below).
*   **Interactive Buttons:** Provides inline keyboards for quick actions.
*   **Scheduled Messages:** Users can schedule a message to be sent back to them.
*   **Token Price Fetching:** `/price` command to fetch the price of a configured token (e.g., SULA).
*   **Asynchronous Operations:** Built with `async` and `await` for non-blocking performance.
*   **Robust API Calls:** Uses `httpx` with timeouts, connection pooling, and `tenacity` for retries on LLM API calls.
*   **Text Chunking:** Splits large prompts into smaller chunks for LLM processing.
*   **Detailed Logging:** Configurable logging to both file (`bot.log`) and console.
*   **Error Handling:** Graceful error handling for bot operations and LLM interactions.
*   **Environment Variable Configuration:** Easy configuration using a `.env` file.
*   **Optional Group Message History:** Can fetch and process past messages from a specified group.

## Prerequisites

*   Python 3.8+
*   Pip (Python package installer)
*   Access to a local or remote LLM server compatible with the OpenAI API format (e.g., LM Studio, Ollama with an OpenAI-compatible endpoint).
*   A Telegram Bot Token.

## Setup and Installation

1.  **Clone the repository (or download the script):**
    ```bash
    # If you have it in a git repo
    # git clone <your-repo-url>
    # cd <your-repo-name>

    # Or just place the Python script in a directory
    mkdir sulagpt-bot
    cd sulagpt-bot
    # Save the script as e.g., bot.py in this directory
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    python-dotenv
    httpx
    pandas
    python-telegram-bot
    tenacity
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory of the project (next to your Python script).
    Copy the contents of `.env.example` (see below) into `.env` and fill in your actual values.

## Environment Variables Configuration

Create a `.env` file with the following variables:

```ini
# .env file
SULAGPT_KEY="YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_GROUP_ID="YOUR_OPTIONAL_TELEGRAM_GROUP_ID" # Optional: for fetching past messages
SULA_PRICE_API_URL="YOUR_SULA_TOKEN_PRICE_API_URL" # e.g., "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd" (replace 'solana' with your token)
Use code with caution.
Markdown
SULAGPT_KEY: Required. Your Telegram Bot Token obtained from BotFather.
TELEGRAM_GROUP_ID: Optional. If you want the bot to fetch and process historical messages from a specific group, provide the group ID here. The bot must be a member of this group with appropriate permissions.
SULA_PRICE_API_URL: Optional, but required for the /price command. The URL should point to an API endpoint that returns the token price in JSON format (e.g., {"your_token_id": {"usd": 0.123}}). The script expects a structure like data.get("price") or needs adjustment for different API responses. The current placeholder https://api.example.com/sula/price will not work; replace it with a real one.
LLM Configuration
The LLM settings are primarily configured within the script:
LLM_API_BASE_URL:
Default: "http://192.168.2.182:1234/v1"
Action: Change this to the base URL of your LLM API server (e.g., LM Studio, Ollama if it's serving an OpenAI-compatible API). This typically ends in /v1.
MODEL_NAME:
Default: "second-state/Llava-v1.5-7B-GGUF/llava-v1.5-7b-Q4_0.gguf"
Action: Change this to the specific model identifier your LLM server uses. For some servers, you might not need the full path, just the model name loaded in the server. Check your LLM server's documentation.
LLM_SETTINGS:
Contains timeout, connection limits, and prompt chunk size. Adjust these based on your LLM server's performance and your needs.
Important: Ensure your LLM server is running and accessible from where you run the bot script. The model specified by MODEL_NAME must be loaded and available on the LLM server.
Running the Bot
Once you have set up the environment variables and (if necessary) installed dependencies:
Ensure your LLM server is running and correctly configured in the script.
Activate your virtual environment (if you created one):
source venv/bin/activate # On Windows: venv\Scripts\activate
Use code with caution.
Bash
Run the Python script (e.g., if you named it bot.py):
python bot.py
Use code with caution.
Bash
The bot will start polling for updates from Telegram.
Bot Commands
/start - Greet the user and optionally fetch past group messages if TELEGRAM_GROUP_ID is set.
/help - Show all available commands.
/echo <text> - Echo back the provided text.
/buttons - Show interactive inline buttons for quick actions (Check SULA Price, Get AI Response, Cancel).
/schedule <seconds> - Schedule a message to be sent to the chat after the specified number of seconds.
/export - Export all collected wallet addresses to a CSV file and sends it to the user.
/price - Fetch the current SULA token price (uses SULA_PRICE_API_URL).
/ai <prompt> - Get an LLM-based response to your prompt.
/history - View all collected wallet addresses.
Non-Command Interactions:
Wallet Address Submission: Simply sending a message containing a valid Ethereum, Solana, or Bitcoin wallet address will trigger its collection.
General Text: Sending any other text (that isn't a command or a wallet address) will be sent to the LLM for a response.
Logging
The bot implements comprehensive logging:
File Logging: DEBUG level and above messages are logged to bot.log. This file rotates when it reaches 5MB, keeping up to 2 backup files.
Console Logging: ERROR level and above messages are printed to the console.
This setup ensures detailed logs for debugging while keeping the console output clean for critical errors.
Error Handling
LLM API Calls: Uses tenacity for automatic retries with exponential backoff for httpx.TimeoutException and httpx.HTTPError.
Telegram API Calls: Uses AIORateLimiter for handling rate limits. The safe_send_message function also provides retries.
General Exceptions: A global error handler (error_handler) logs exceptions and attempts to notify the user about unexpected errors.
Project Structure (Single File)
The provided code is a single Python script. For larger projects, consider breaking it down into modules (e.g., llm_handler.py, bot_commands.py, utils.py).
.
├── bot.py                # The main Python script for the bot
├── .env                  # Your environment variables (GITIGNORED!)
├── .env.example          # Example environment variables file
├── requirements.txt      # Python dependencies
└── bot.log               # Log file (GITIGNORED!)
Use code with caution.
This README should provide a good starting point for anyone looking to understand, set up, and run your SulaGPT bot. Remember to create the .env.example and requirements.txt files as mentioned.
