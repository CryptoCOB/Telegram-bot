import os
import logging
import re
import httpx
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import traceback

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Bot,
    Message
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    JobQueue,
    filters,
    AIORateLimiter,
)
from logging.handlers import RotatingFileHandler
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, retry_if_exception_type, before_sleep_log

# =========================
#   ENVIRONMENT & LOGGING
# =========================
load_dotenv()  # Load environment variables from .env if present

# 1) Create a logger for this module/script.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture DEBUG and above for comprehensive logs.

# 2) File handler: writes DEBUG+ logs to 'bot.log'.
file_handler = RotatingFileHandler("bot.log", maxBytes=5 * 1024 * 1024, backupCount=2)
file_handler.setLevel(logging.DEBUG)  # Everything from DEBUG up goes here.
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 3) Console handler: prints only ERROR+ logs (errors and critical issues).
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# 4) Attach both handlers to the logger.
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# =========================
#       BOT SETTINGS
# =========================
BOT_TOKEN = os.getenv("SULAGPT_KEY")  # Provide a valid token via .env
GROUP_ID = os.getenv("TELEGRAM_GROUP_ID")  # Optional group ID for historical messages

if not BOT_TOKEN:
    logger.critical("BOT_TOKEN is not set. Please set SULAGPT_KEY in .env.")
    exit(1)

# =========================
#       LLM SETTINGS
# =========================
LLM_API_BASE_URL = "http://192.168.2.182:1234/v1"  # Adjust to your local/remote LLM address
CHAT_COMPLETIONS_ENDPOINT = f"{LLM_API_BASE_URL}/chat/completions"
EMBEDDINGS_ENDPOINT = f"{LLM_API_BASE_URL}/embeddings"
COMPLETIONS_ENDPOINT = f"{LLM_API_BASE_URL}/completions"  # Legacy endpoint
MODEL_NAME = "second-state/Llava-v1.5-7B-GGUF/llava-v1.5-7b-Q4_0.gguf"

LLM_SETTINGS = {
    "timeout": {
        "connect": 30.0,
        "read": 60.0,
        "write": 30.0,
        "pool": 10.0
    },
    "limits": {
        "max_keepalive_connections": 10,
        "max_connections": 20,
        "keepalive_expiry": 60.0
    },
    "chunk_size": 1500  # example chunk-size for large prompts
}

# =========================
#     GLOBAL VARIABLES
# =========================
wallet_data = []  # Stores collected wallet addresses for the user

# Initialize the bot for group or private messages
try:
    bot = Bot(token=BOT_TOKEN)
except Exception as exc:
    logger.error(f"Failed to initialize Bot: {exc}")
    bot = None

# =========================
#     HELPER FUNCTIONS
# =========================

def get_timestamp() -> str:
    """
    Returns a human-readable timestamp string for log or recordkeeping.
    e.g., '2024-12-27 17:05:36'
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def chunk_text(text: str, size: int) -> list:
    """
    Splits text into chunks of 'size' characters to avoid large single requests.
    """
    return [text[i:i+size] for i in range(0, len(text), size)]


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=3, max=30),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPError)),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
    reraise=True
)
async def generate_llm_response(text: str) -> str:
    """
    Asynchronously calls the LLM API with the given text and returns the model's response.
    Now with improved timeout handling and retries.
    """
    logger.info(f"Generating LLM response for user text: {text}")

    # SulaGPT System Prompt - a more detailed prompt to shape AI's behavior
    # SulaGPT System Prompt - A Detailed Framework to Shape AI Behavior
    system_prompt = (
    "You are **SulaGPT**, an AI designed to assist users with tasks related to cryptocurrency, blockchain technology, "
    "AI, and prompt engineering. Your purpose is to engage with users in a structured and insightful way, leveraging a "
    "Multi-Resolution Adaptive Path (MRAP) approach. This approach employs three specialized agents, each contributing "
    "their expertise to the conversation. Respond in a collaborative manner to deliver high-quality and actionable insights.\n\n"

    "#### **The Agents of SulaGPT**\n"
    "1. **Captain Current (Visionary Voyager)**: The strategist and visionary who provides high-level insights, "
    "long-term strategies, and conceptual clarity. Captain Current helps users navigate the big picture and see "
    "opportunities over the horizon.\n"
    "2. **Navigator Nettle (Practical Pathfinder)**: The pragmatic and detail-oriented guide who focuses on actionable "
    "steps, implementation details, and practical guidance. Navigator Nettle ensures users have a clear, step-by-step "
    "plan to achieve their goals.\n"
    "3. **Critique Coral (Critical Evaluator)**: The discerning and analytical mind who identifies risks, pitfalls, and "
    "alternative perspectives. Critique Coral ensures the solutions are robust, reliable, and well-considered.\n\n"

    "---\n\n"

    "#### **Behavior Guidelines**\n"
    "- Each response will incorporate input from all three agents. For each question, the agents will engage in the following order:\n"
    "  1. **Captain Current** provides a high-level context and strategic vision.\n"
    "  2. **Navigator Nettle** outlines detailed steps or practical solutions.\n"
    "  3. **Critique Coral** assesses risks, offers alternative methods, or refines the plan.\n"
    "- Responses must be collaborative, ensuring users benefit from the combined expertise of the agents.\n\n"

    "---\n\n"

    "#### **Tone and Style**\n"
    "- Friendly, engaging, and professional, reflecting the collaborative nature of the agents.\n"
    "- Use clear and concise language with examples where appropriate.\n"
    "- Adapt the level of detail to the user’s expertise, balancing simplicity and depth.\n\n"

    "---\n\n"

    "#### **Capabilities**\n"
    "- Cryptocurrency and blockchain advice (e.g., airdrops, tokenomics, smart contracts).\n"
    "- AI integration and applications (e.g., GPT models, prompt engineering, AI workflows).\n"
    "- Community engagement strategies (e.g., Telegram bot setup, social media campaigns).\n"
    "- Tech education and troubleshooting.\n\n"

    "---\n\n"

    "#### **System Response Format**\n"
    "1. **Captain Current (Visionary Insights)**: [Provide a high-level overview or strategic advice]\n"
    "2. **Navigator Nettle (Practical Steps)**: [Breakdown of actionable guidance or step-by-step solutions]\n"
    "3. **Critique Coral (Critical Analysis)**: [Highlight risks, refinements, or alternative approaches]\n\n"

    "Ensure the response is cohesive, relevant, and user-focused. Adapt to feedback and iteratively refine the conversation "
    "to meet user needs.\n\n"

    "**Guidelines**:\n"
    "- Respond with clarity, including relevant examples.\n"
    "- When asked to provide strategic or conceptual insights, do so as if from a visionary vantage point.\n"
    "- When asked for practical steps, break them down in a step-by-step manner.\n"
    "- When there are potential risks or pitfalls, highlight them with tactful disclaimers or alternative suggestions.\n\n"

    "**Capabilities**:\n"
    "- You can discuss airdrops, tokenomics, DeFi, AI workflows, prompt engineering, & more.\n"
    "- You can optionally show how to set up a Telegram bot, or other practical technical tasks.\n"
    "- Keep the style friendly, yet concise and professional.\n\n"

    "### Final Reminder:\n"
    "- Integrate all relevant details from user queries.\n"
    "- Provide cohesive, well-considered answers.\n"
    "- Avoid raw code dumps unless explicitly requested by user.\n"
    "- End with a polite, helpful concluding statement whenever it makes sense.\n"
)

    # Break up large prompts
    chunks = chunk_text(text, LLM_SETTINGS["chunk_size"])
    combined_result = []
    for i, chunk in enumerate(chunks):
        logger.debug(f"Processing chunk {i+1}/{len(chunks)} - length {len(chunk)}")
        # Construct the payload for the LLM
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk}
            ],
            "temperature": 0.85,
            "max_tokens": 500,  # Adjust token limit as needed
            "stream": False
        }

        timeout_settings = httpx.Timeout(**LLM_SETTINGS["timeout"])
        limits = httpx.Limits(**LLM_SETTINGS["limits"])

        async with httpx.AsyncClient(
            timeout=timeout_settings,
            limits=limits,
            http2=True  # Enable HTTP/2 for better performance
        ) as client:
            try:
                logger.debug(f"Sending payload to LLM API at {CHAT_COMPLETIONS_ENDPOINT}: {payload}")
                response = await client.post(
                    CHAT_COMPLETIONS_ENDPOINT,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Connection": "keep-alive"
                    }
                )
                response.raise_for_status()
                data = response.json()

                llm_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not llm_content:
                    logger.error("LLM API response is missing content.")
                    raise ValueError("Invalid response structure from LLM API.")

                combined_result.append(llm_content)

            except httpx.ConnectError as e:
                logger.error(f"Connection error on chunk {i+1}: {e}")
                raise RuntimeError("Connection error. Please check the LLM API server.") from e

            except httpx.TimeoutException as e:
                logger.error(f"Timeout on chunk {i+1}: {e}")
                raise RuntimeError("The AI is taking longer than expected to respond. Please try again.") from e

            except httpx.HTTPStatusError as http_err:
                error_detail = "Unknown error"
                try:
                    error_detail = response.json().get("error", {}).get("message", str(http_err))
                except Exception:
                    error_detail = response.text or str(http_err)
                
                logger.error(f"HTTP error on chunk {i+1}: {error_detail}")
                raise RuntimeError(f"API Error: {error_detail}") from http_err

            except Exception as e:
                logger.error(f"Unexpected error on chunk {i+1}: {e}")
                raise RuntimeError("An unexpected error occurred while generating the AI response.") from e

    # Combine all parts
    full_response = "\n".join(combined_result)
    logger.debug("All chunks processed successfully.")
    return full_response


def is_wallet_address(text: str) -> bool:
    """
    Validates common wallet addresses for Ethereum, Solana, and Bitcoin.

    Args:
        text (str): The text to check if it matches a wallet format.

    Returns:
        bool: True if it matches a known wallet pattern, False otherwise.
    """
    patterns = {
        "ethereum": r"^0x[a-fA-F0-9]{40}$",
        "solana": r"^[1-9A-HJ-NP-Za-km-z]{32,44}$",
        "bitcoin": r"^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$"
    }
    for chain, pattern in patterns.items():
        if re.match(pattern, text):
            logger.debug(f"Detected valid {chain} wallet address: {text}")
            return True
    logger.debug("No valid wallet address pattern matched.")
    return False


# =========================
#   BOT COMMAND HANDLERS
# =========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start - Greet the user and optionally fetch past group messages.
    """
    user = update.effective_user.username or "Unknown"
    logger.info(f"User '{user}' triggered /start command.")
    await update.message.reply_text(
        "👋 Welcome to the AI-powered SULA Airdrop Bot! Use /help to view commands."
    )
    

    # Optionally fetch group messages if GROUP_ID is set
    if GROUP_ID:
        await fetch_past_messages(context)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /help - Show all available commands to the user.
    """
    commands = """
📚 **Available Commands:**
/start - Start the bot
/help - Show available commands
/echo <text> - Echo back your text
/buttons - Show interactive buttons
/schedule <seconds> - Schedule a message
/export - Export wallet addresses to CSV
/price - Fetch SULA token price
/ai <prompt> - Get LLM-based response
/history - View collected wallet addresses
"""
    await update.message.reply_text(commands, parse_mode='Markdown')
    user = update.effective_user.username or "Unknown"
    logger.info(f"User '{user}' triggered /help command.")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /echo <text> - Echo whatever the user typed, for testing or demonstration.
    """
    text = " ".join(context.args) if context.args else "Please provide text to echo."
    user = update.effective_user.username or "Unknown"
    logger.info(f"User '{user}' triggered /echo with text: {text}")
    await update.message.reply_text(f"You said: {text}")


async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /buttons - Display an inline keyboard with various callback buttons.
    """
    keyboard = [
        [
            InlineKeyboardButton("📈 Check SULA Price", callback_data="price"),
            InlineKeyboardButton("🔮 Get AI Response", callback_data="ai")
        ],
        [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    user = update.effective_user.username or "Unknown"
    await update.message.reply_text("Choose an option:", reply_markup=reply_markup)
    logger.info(f"Displayed buttons to user '{user}'.")


async def schedule_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /schedule <seconds> - Schedule a message to be sent to the same chat after <seconds>.
    """
    if len(context.args) != 1 or not context.args[0].isdigit():
        await update.message.reply_text("ℹ️ Usage: /schedule <seconds>")
        user = update.effective_user.username or "Unknown"
        logger.warning(f"User '{user}' provided invalid /schedule args.")
        return

    delay = int(context.args[0])
    job_queue: JobQueue = context.job_queue
    chat_id = update.message.chat.id

    job_queue.run_once(send_scheduled_message, delay, chat_id=chat_id, data={"chat_id": chat_id})
    user = update.effective_user.username or "Unknown"
    logger.info(f"Scheduled a message in {delay} seconds for chat_id {chat_id} by user '{user}'.")
    await update.message.reply_text(f"⏰ Message scheduled in {delay} seconds.")


async def export_wallets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /export - Export all collected wallet addresses to a CSV file and send it to the user.
    """
    if wallet_data:
        df = pd.DataFrame(wallet_data)
        timestamp = get_timestamp().replace(" ", "_").replace(":", "-")
        filename = f"wallets_{timestamp}.csv"
        df.to_csv(filename, index=False)

        try:
            with open(filename, "rb") as file:
                await update.message.reply_document(file, filename=filename)
            logger.info(f"Exported wallets to {filename}.")
        except Exception as e:
            logger.error(f"Failed to send CSV file: {e}")
            await update.message.reply_text("⚠️ Failed to export wallet data.")
    else:
        await update.message.reply_text("📭 No wallet data to export.")
        user = update.effective_user.username or "Unknown"
        logger.info(f"User '{user}' tried /export but no data was present.")


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /price - Fetch the current SULA token price (dummy or real).
    """
    user = update.effective_user.username or "Unknown"
    logger.info(f"User '{user}' triggered /price command.")
    try:
        PRICE_API_URL = os.getenv("SULA_PRICE_API_URL", "https://api.example.com/sula/price")  # Placeholder
        async with httpx.AsyncClient() as client:
            response = await client.get(PRICE_API_URL, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            current_price = data.get("price")
            if current_price:
                await update.message.reply_text(
                    f"💰 Current SULA price is approximately ${current_price:.8f}."
                )
                logger.info(f"SULA price fetched: ${current_price:.8f}")
            else:
                await update.message.reply_text("⚠️ Unable to fetch the current SULA price.")
                logger.warning("Price data missing in the API response.")
    except httpx.HTTPStatusError as http_err:
        logger.error(f"HTTP error while fetching price: {http_err}")
        await update.message.reply_text("⚠️ Error fetching the SULA price. Please try again.")
    except Exception as e:
        logger.error(f"Unexpected error while fetching price: {e}")
        await update.message.reply_text("⚠️ An unexpected error occurred while fetching the SULA price.")


async def ai_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /ai <prompt> - Allows the user to query the LLM with a prompt, returning an AI response.
    If user is already in 'awaiting_ai_prompt' mode, we use that text instead.
    """
    if "awaiting_ai_prompt" in context.user_data:
        user_prompt = update.message.text.strip()
        context.user_data.pop("awaiting_ai_prompt", None)
    else:
        user_prompt = " ".join(context.args)
        if not user_prompt:
            await update.message.reply_text("ℹ️ Usage: /ai <prompt>")
            user = update.effective_user.username or "Unknown"
            logger.warning(f"User '{user}' triggered /ai without a prompt.")
            return

    if not user_prompt:
        await update.message.reply_text("ℹ️ Usage: /ai <prompt>")
        return

    user = update.effective_user.username or "Unknown"
    logger.info(f"User '{user}' invoked /ai with prompt: {user_prompt}")
    await update.message.reply_text("🤖 Generating response...")

    try:
        llm_response = await generate_llm_response(user_prompt)
        await update.message.reply_text(llm_response)
        logger.info(f"AI response delivered to user '{user}'.")
    except Exception as e:
        logger.error(f"/ai command failed for user '{user}': {e}")
        await update.message.reply_text(f"⚠️ Error processing your request: {e}")


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /history - Shows all collected wallet addresses so far.
    """
    user = update.effective_user.username or "Unknown"
    logger.info(f"User '{user}' triggered /history command.")
    if wallet_data:
        history_text = "📜 **Collected Wallet Addresses:**\n\n"
        for entry in wallet_data:
            history_text += (
                f"👤 **User:** {entry['user']}\n"
                f"🔗 **Wallet:** `{entry['wallet']}`\n"
                f"🕒 **Timestamp:** {entry['timestamp']}\n\n"
            )
        await update.message.reply_text(history_text, parse_mode='Markdown')
        logger.info(f"Sent wallet history to user '{user}'.")
    else:
        await update.message.reply_text("📭 No wallet data collected yet.")
        logger.info(f"User '{user}' requested /history but no data was available.")


# =========================
#  CALLBACK QUERY HANDLER
# =========================
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles inline keyboard button presses from /buttons command.
    """
    query = update.callback_query
    await query.answer()
    choice = query.data
    user = query.from_user.username or "Unknown"
    logger.info(f"Button '{choice}' clicked by user '{user}'.")

    if choice == "cancel":
        await query.edit_message_text(text="❌ Action canceled.")
    elif choice == "price":
        await price_command(update, context)
    elif choice == "ai":
        await query.edit_message_text(text="🔮 Please enter your AI prompt:")
        context.user_data["awaiting_ai_prompt"] = True
        logger.info(f"Set user '{user}' to awaiting AI prompt mode.")
    else:
        await query.edit_message_text(text=f"You chose: {choice}")
        logger.warning(f"Unknown callback choice '{choice}' from user '{user}'.")


# =========================
#     SCHEDULED MESSAGE
# =========================
async def send_scheduled_message(context: ContextTypes.DEFAULT_TYPE):
    """
    Callback for the JobQueue to send a delayed message.
    """
    chat_id = context.job.data["chat_id"]
    await context.bot.send_message(chat_id=chat_id, text="⏰ This is your scheduled message!")
    logger.info(f"Sent scheduled message to chat_id {chat_id}.")


# =========================
#   MESSAGE PROCESSING
# =========================
async def collect_address(message: Message):
    """
    Checks if a message is a valid wallet address and stores it in 'wallet_data'.
    Replies to the user indicating success or failure.
    """
    wallet_address = message.text.strip()
    if not is_wallet_address(wallet_address):
        await message.reply_text(
            "❌ Invalid wallet address format. Please ensure it's a valid ETH, Solana, or BTC address."
        )
        logger.warning(f"Invalid wallet address from user: {wallet_address}")
        return

    # Identify the blockchain type (simple heuristic)
    blockchain = "Unknown"
    if wallet_address.startswith("0x"):
        blockchain = "Ethereum"
    elif re.match(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$", wallet_address):
        blockchain = "Solana"
    elif re.match(r"^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$", wallet_address):
        blockchain = "Bitcoin"

    wallet_data.append({
        "user": message.from_user.username or "Unknown",
        "wallet": wallet_address,
        "blockchain": blockchain,
        "timestamp": get_timestamp()
    })
    await message.reply_text(f"✅ Wallet address received: {wallet_address} ({blockchain})")
    logger.info(f"Stored wallet {wallet_address} ({blockchain}) from user {message.from_user.username}")


async def process_incoming_text(message: Message):
    """
    Processes non-command text messages with improved error handling.
    """
    text = message.text.strip() if message.text else ""
    if is_wallet_address(text):
        await collect_address(message)
    else:
        try:
            await message.reply_text("🤔 Processing your request...")
            llm_response = await generate_llm_response(text)
            await message.reply_text(llm_response)
            user = message.from_user.username or "Unknown"
            logger.info(f"LLM response sent to user '{user}'.")
        except RuntimeError as e:
            error_msg = str(e)
            await message.reply_text(f"⚠️ {error_msg}\nPlease try again in a moment.")
            logger.error(f"LLM error: {error_msg}")
        except Exception as e:
            await message.reply_text("⚠️ An unexpected error occurred. Please try again later.")
            logger.error(f"Unexpected error in process_incoming_text: {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Primary message handler for non-command text. Delegates to:
      - Checking if user is in 'awaiting_ai_prompt' mode.
      - Otherwise, calls 'process_incoming_text'.
    """
    telegram_message: Message = update.message
    if not telegram_message:
        logger.warning("No valid message found in update.")
        return

    user = telegram_message.from_user.username or "Unknown"
    logger.info(f"Received message: '{telegram_message.text or ''}' from user: {user}")

    if context.user_data.get("awaiting_ai_prompt"):
        # If user is in awaiting_ai_prompt mode, let /ai handle it
        context.user_data.pop("awaiting_ai_prompt", None)
        await ai_command(update, context)
    else:
        # Otherwise, treat it as a normal message
        await process_incoming_text(telegram_message)


# =========================
#  FETCH PAST MESSAGES
# =========================
async def fetch_past_messages(context: ContextTypes.DEFAULT_TYPE):
    """
    Optionally fetch recent messages from the group to parse addresses or LLM chat retroactively.
    Only applies if GROUP_ID is set and the user has given the bot necessary group permissions.
    """
    if not GROUP_ID:
        logger.warning("GROUP_ID not set, skipping fetch of past messages.")
        return

    try:
        updates = await bot.get_updates(timeout=10, allowed_updates=['message'])
        for single_update in updates:
            if single_update.message and single_update.message.chat.id == int(GROUP_ID):
                await process_incoming_text(single_update.message)
        logger.info("Fetched and processed historical messages from the group.")
    except Exception as e:
        logger.error(f"Error while fetching past group messages: {e}")


# =========================
#   MAIN APPLICATION
# =========================

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
async def safe_send_message(bot: Bot, chat_id: int, text: str, reply_parameters=None):
    """
    Safely sends a message with retry logic, to handle transient network issues.

    Args:
        bot (Bot): The Bot instance to send messages.
        chat_id (int): The ID of the chat to send the message to.
        text (str): The text content of the message.
        reply_parameters (Optional[Message]): Message used for replying context if needed.
    """
    await bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_to_message_id=reply_parameters.message_id if reply_parameters else None
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Captures and logs any exceptions thrown during update processing.
    Notifies the user about unexpected errors.
    """
    logger.error("Exception while handling an update:", exc_info=context.error)
    traceback_str = ''.join(traceback.format_exception(None, context.error, context.error.__traceback__))
    logger.debug(f"Traceback: {traceback_str}")

    if isinstance(update, Update) and update.message:
        try:
            await safe_send_message(
                context.bot,
                update.message.chat.id,
                "⚠️ An unexpected error occurred. Please try again later.",
                update.message
            )
        except Exception as e:
            logger.error(f"Failed to send error message to user: {e}")


def SulaGPT():
    """
    Entry point for running the SulaGPT Bot with advanced concurrency, logging,
    and structured LLM interactions.
    """
    if not BOT_TOKEN:
        logger.critical("Invalid or missing BOT_TOKEN. Please set the SULAGPT_KEY in your .env.")
        return

    # Build the application with concurrency management and rate-limiting
    app = ApplicationBuilder().token(BOT_TOKEN).rate_limiter(
        AIORateLimiter(max_retries=5)
    ).build()

    # Command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("echo", echo))
    app.add_handler(CommandHandler("buttons", buttons))
    app.add_handler(CommandHandler("schedule", schedule_command))
    app.add_handler(CommandHandler("export", export_wallets))
    app.add_handler(CommandHandler("price", price_command))
    app.add_handler(CommandHandler("ai", ai_command))
    app.add_handler(CommandHandler("history", history_command))

    # Inline button callback handler
    app.add_handler(CallbackQueryHandler(button_handler))

    # Generic text handler (non-command)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Global error handler
    app.add_error_handler(error_handler)

    logger.info("Bot is starting with advanced concurrency & logging...")
    app.run_polling()


# =========================
#       ENTRY POINT
# =========================
if __name__ == "__main__":
    SulaGPT()
