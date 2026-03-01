"""sluice — priority-queue LLM router for Apple Silicon."""

__version__ = "0.1.0"

from sluice.client import SluiceClient
from sluice.config import P_CRITICAL, P_HIGH, P_MEDIUM, P_LOW, P_BG

# Module-level singleton — import and use directly
sluice = SluiceClient()
