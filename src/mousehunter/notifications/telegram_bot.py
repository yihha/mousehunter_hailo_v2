"""
Telegram Bot - Async notification and control interface

Provides:
- Push notifications with images to Telegram
- Command handlers (/status, /unlock, /scream, /photo)
- Inline buttons for quick actions
- Retry logic for network resilience
"""

import asyncio
import logging
import os
from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, Callable

from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    before_sleep_log,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Conditional import for development
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application,
        CommandHandler,
        CallbackQueryHandler,
        ContextTypes,
    )
    from telegram.error import TimedOut, NetworkError, TelegramError

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not available - notifications will be simulated")

if TYPE_CHECKING:
    from telegram.ext import Application


# Global reference to main event loop (set by main.py)
MAIN_LOOP: asyncio.AbstractEventLoop | None = None


class MockTelegramApp:
    """Mock Telegram application for development."""

    def __init__(self, token: str):
        self.token = token
        logger.info(f"[MOCK] Telegram bot initialized")

    async def initialize(self):
        logger.info("[MOCK] Bot initialized")

    async def start(self):
        logger.info("[MOCK] Bot started")

    async def stop(self):
        logger.info("[MOCK] Bot stopped")

    def add_handler(self, handler):
        logger.debug(f"[MOCK] Handler added: {handler}")

    def add_error_handler(self, handler):
        logger.debug("[MOCK] Error handler added")

    class bot:
        @staticmethod
        async def send_message(chat_id, text, **kwargs):
            logger.info(f"[MOCK] Message to {chat_id}: {text}")

        @staticmethod
        async def send_photo(chat_id, photo, caption=None, **kwargs):
            logger.info(f"[MOCK] Photo to {chat_id}: {caption}")

    class updater:
        @staticmethod
        async def start_polling(**kwargs):
            logger.info("[MOCK] Polling started")

        @staticmethod
        async def stop():
            logger.info("[MOCK] Polling stopped")


class TelegramBot:
    """
    Async Telegram bot for MouseHunter notifications and control.

    Commands:
    - /status: System health check
    - /unlock: Manually unlock cat flap
    - /lock: Manually lock cat flap
    - /scream: Trigger audio deterrent
    - /photo: Capture current camera frame
    - /help: Show available commands
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str | int,
        enabled: bool = True,
    ):
        """
        Initialize the Telegram bot.

        Args:
            bot_token: Bot token from @BotFather
            chat_id: Chat/group ID for notifications
            enabled: Whether bot is enabled
        """
        self.bot_token = bot_token
        self.chat_id = int(chat_id) if chat_id else 0
        self.enabled = enabled

        self._app: "Application" = None
        self._started = False

        # External component references (set by main controller)
        self._jammer = None
        self._audio = None
        self._camera = None

        if not bot_token:
            logger.warning("Telegram bot token not configured")
            self.enabled = False

        if not chat_id:
            logger.warning("Telegram chat ID not configured")
            self.enabled = False

        if self.enabled:
            if TELEGRAM_AVAILABLE:
                self._app = Application.builder().token(bot_token).build()
            else:
                self._app = MockTelegramApp(bot_token)

        logger.info(f"TelegramBot initialized: enabled={enabled}, chat_id={chat_id}")

    def set_components(self, jammer=None, audio=None, camera=None) -> None:
        """Set references to hardware components."""
        self._jammer = jammer
        self._audio = audio
        self._camera = camera

    async def start(self) -> None:
        """Initialize and start the bot."""
        if not self.enabled:
            logger.info("Telegram bot disabled, skipping start")
            return

        if self._started:
            logger.warning("Bot already started")
            return

        # Register handlers
        self._register_handlers()

        # Initialize
        await self._app.initialize()
        await self._app.start()

        # Start polling
        if TELEGRAM_AVAILABLE:
            await self._app.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
            )

        self._started = True
        logger.info("Telegram bot started and polling")

        # Send startup message
        await self._send_startup_message()

    async def stop(self) -> None:
        """Stop the bot."""
        if not self._started:
            return

        if TELEGRAM_AVAILABLE:
            await self._app.updater.stop()
        await self._app.stop()
        self._started = False
        logger.info("Telegram bot stopped")

    def _register_handlers(self) -> None:
        """Register command handlers."""
        if not TELEGRAM_AVAILABLE:
            return

        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("lock", self._cmd_lock))
        self._app.add_handler(CommandHandler("unlock", self._cmd_unlock))
        self._app.add_handler(CommandHandler("scream", self._cmd_scream))
        self._app.add_handler(CommandHandler("photo", self._cmd_photo))

        # Inline button callbacks
        self._app.add_handler(CallbackQueryHandler(self._handle_callback))

        # Error handler
        self._app.add_error_handler(self._error_handler)

        logger.info("Telegram handlers registered")

    async def _send_startup_message(self) -> None:
        """Send startup notification."""
        try:
            await self.send_message(
                "MouseHunter Online\n"
                "Prey detection system active.\n"
                "Use /help for commands."
            )
        except Exception as e:
            logger.error(f"Failed to send startup message: {e}")

    # ==================== Command Handlers ====================

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        await update.message.reply_text(
            "Welcome to MouseHunter!\n\n"
            "I monitor the cat flap for prey and block entry when detected.\n"
            "Use /help to see available commands."
        )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        help_text = (
            "MouseHunter Commands:\n\n"
            "/status - System health check\n"
            "/photo - Capture camera snapshot\n"
            "/lock - Lock cat flap (5 min)\n"
            "/unlock - Unlock cat flap\n"
            "/scream - Trigger audio deterrent\n"
            "/help - Show this message"
        )
        await update.message.reply_text(help_text)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        try:
            import psutil

            cpu_temp = self._get_cpu_temp()
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            # Jammer status
            jammer_status = "Unknown"
            if self._jammer:
                js = self._jammer.get_status()
                if js["is_active"]:
                    jammer_status = f"LOCKED ({js['time_remaining_seconds']:.0f}s remaining)"
                else:
                    jammer_status = "UNLOCKED"

            status = (
                "System Status\n\n"
                f"Cat Flap: {jammer_status}\n"
                f"CPU: {cpu_percent:.1f}%\n"
                f"Memory: {memory.percent:.1f}%\n"
                f"Temperature: {cpu_temp}C\n"
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            )

            await update.message.reply_text(status)

        except Exception as e:
            logger.error(f"Status command error: {e}")
            await update.message.reply_text(f"Error getting status: {e}")

    async def _cmd_lock(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /lock command."""
        if not self._jammer:
            await update.message.reply_text("Jammer not available")
            return

        if self._jammer.activate():
            remaining = self._jammer.time_remaining
            await update.message.reply_text(
                f"Cat flap LOCKED\n"
                f"Duration: {remaining:.0f} seconds"
            )
        else:
            remaining = self._jammer.time_remaining
            await update.message.reply_text(
                f"Cat flap already locked\n"
                f"Remaining: {remaining:.0f} seconds"
            )

    async def _cmd_unlock(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /unlock command."""
        if not self._jammer:
            await update.message.reply_text("Jammer not available")
            return

        if self._jammer.deactivate("Manual unlock via Telegram"):
            await update.message.reply_text("Cat flap UNLOCKED")
        else:
            await update.message.reply_text("Cat flap was not locked")

    async def _cmd_scream(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /scream command."""
        if not self._audio:
            await update.message.reply_text("Audio system not available")
            return

        if self._audio.play():
            await update.message.reply_text("SCREAM triggered!")
        else:
            await update.message.reply_text("Audio failed or already playing")

    async def _cmd_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /photo command."""
        if not self._camera:
            await update.message.reply_text("Camera not available")
            return

        try:
            image_bytes = self._camera.capture_snapshot_bytes(quality=90)
            if image_bytes:
                photo = BytesIO(image_bytes)
                photo.name = "snapshot.jpg"
                await update.message.reply_photo(
                    photo=photo,
                    caption=f"Live capture - {datetime.now().strftime('%H:%M:%S')}"
                )
            else:
                await update.message.reply_text("Failed to capture image")
        except Exception as e:
            logger.error(f"Photo command error: {e}")
            await update.message.reply_text(f"Error: {e}")

    # ==================== Inline Buttons ====================

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle inline button callbacks."""
        query = update.callback_query
        await query.answer()

        action = query.data

        if action == "unlock":
            if self._jammer:
                self._jammer.deactivate("Telegram button")
                await query.edit_message_text("Cat flap unlocked!")
            else:
                await query.edit_message_text("Jammer not available")

        elif action == "scream":
            if self._audio:
                self._audio.play()
                await query.edit_message_text("SCREAM triggered!")
            else:
                await query.edit_message_text("Audio not available")

        elif action == "ignore":
            await query.edit_message_text("Alert ignored")

    def _get_action_keyboard(self) -> "InlineKeyboardMarkup":
        """Create inline keyboard for quick actions."""
        if not TELEGRAM_AVAILABLE:
            return None

        keyboard = [
            [
                InlineKeyboardButton("Unlock", callback_data="unlock"),
                InlineKeyboardButton("Scream", callback_data="scream"),
                InlineKeyboardButton("Ignore", callback_data="ignore"),
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    # ==================== Error Handling ====================

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors."""
        logger.error(f"Telegram error: {context.error}", exc_info=context.error)

    # ==================== Notification Methods ====================

    # Build retry exceptions based on availability
    # When TELEGRAM_AVAILABLE, retry on network errors; otherwise no retry for mock mode
    _RETRY_EXCEPTIONS = (
        (TimedOut, NetworkError, TelegramError, ConnectionError, TimeoutError)
        if TELEGRAM_AVAILABLE
        else (type(None),)  # Never matches - effectively disables retry for mock
    )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def send_message(self, text: str) -> None:
        """Send a text message with retry on network errors."""
        if not self.enabled or not self._app:
            logger.debug(f"[DISABLED] Would send: {text}")
            return

        await self._app.bot.send_message(chat_id=self.chat_id, text=text)
        logger.debug(f"Message sent: {text[:50]}...")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def send_alert(
        self,
        text: str,
        image_bytes: bytes | None = None,
        include_buttons: bool = True,
    ) -> None:
        """
        Send an alert notification with optional image and action buttons.

        Args:
            text: Alert message
            image_bytes: JPEG image data
            include_buttons: Include action buttons
        """
        if not self.enabled or not self._app:
            logger.debug(f"[DISABLED] Would send alert: {text}")
            return

        keyboard = self._get_action_keyboard() if include_buttons else None

        if image_bytes:
            photo = BytesIO(image_bytes)
            photo.name = "detection.jpg"
            await self._app.bot.send_photo(
                chat_id=self.chat_id,
                photo=photo,
                caption=text,
                reply_markup=keyboard,
            )
        else:
            await self._app.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                reply_markup=keyboard,
            )

        logger.info(f"Alert sent: {text}")

    def _get_cpu_temp(self) -> str:
        """Get CPU temperature."""
        try:
            # Try Pi-specific method
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                temp = int(f.read()) / 1000
                return f"{temp:.1f}"
        except Exception:
            return "N/A"


# ==================== Global Functions ====================

# Global bot instance
_bot_instance: TelegramBot | None = None


def _create_default_bot() -> TelegramBot:
    """Create bot from config."""
    try:
        from mousehunter.config import telegram_config

        return TelegramBot(
            bot_token=telegram_config.bot_token,
            chat_id=telegram_config.chat_id,
            enabled=telegram_config.enabled,
        )
    except ImportError:
        logger.warning("Config not available")
        return TelegramBot(
            bot_token=os.getenv("MOUSEHUNTER_TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("MOUSEHUNTER_TELEGRAM_CHAT_ID", ""),
        )


def get_telegram_bot() -> TelegramBot:
    """Get or create the global Telegram bot."""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = _create_default_bot()
    return _bot_instance


async def notify_async(
    text: str,
    image_bytes: bytes | None = None,
    include_buttons: bool = True,
) -> None:
    """Send notification from async context."""
    bot = get_telegram_bot()
    await bot.send_alert(text, image_bytes, include_buttons)


def notify_sync(
    text: str,
    image_bytes: bytes | None = None,
    include_buttons: bool = True,
) -> None:
    """Send notification from sync context (detection thread)."""
    global MAIN_LOOP

    if MAIN_LOOP is None:
        logger.error("Cannot send notification: MAIN_LOOP not initialized")
        return

    try:
        asyncio.run_coroutine_threadsafe(
            notify_async(text, image_bytes, include_buttons),
            MAIN_LOOP,
        )
        logger.debug(f"Notification scheduled from sync context")
    except Exception as e:
        logger.error(f"Failed to schedule notification: {e}")


async def test_bot() -> None:
    """Test the Telegram bot."""
    logging.basicConfig(level=logging.INFO)
    print("=== Telegram Bot Test ===")
    print(f"Telegram Available: {TELEGRAM_AVAILABLE}")

    bot = TelegramBot(
        bot_token=os.getenv("MOUSEHUNTER_TELEGRAM_BOT_TOKEN", "test_token"),
        chat_id=os.getenv("MOUSEHUNTER_TELEGRAM_CHAT_ID", "12345"),
        enabled=True,
    )

    await bot.start()

    print("Bot running... Press Ctrl+C to stop")
    try:
        await asyncio.sleep(30)
    except KeyboardInterrupt:
        pass

    await bot.stop()
    print("Test complete")


if __name__ == "__main__":
    asyncio.run(test_bot())
