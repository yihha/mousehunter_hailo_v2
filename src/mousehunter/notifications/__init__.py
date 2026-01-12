"""
Notifications module for MouseHunter.

Provides:
- TelegramBot: Async Telegram bot for notifications and control
"""

from .telegram_bot import TelegramBot, get_telegram_bot, notify_async, notify_sync

__all__ = ["TelegramBot", "get_telegram_bot", "notify_async", "notify_sync"]
