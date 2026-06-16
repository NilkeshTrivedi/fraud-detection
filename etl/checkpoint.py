"""
Checkpoint module — tracks ETL pipeline completion state
to avoid redundant reprocessing.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = Path("data/.pipeline_checkpoint.json")


def load_checkpoint() -> dict:
    """
    Load existing checkpoint state.

    Returns:
        Dictionary with pipeline state.
    """
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(stage: str, metadata: dict = {}) -> None:
    """
    Save completed stage to checkpoint file.

    Args:
        stage: Pipeline stage name.
        metadata: Optional metadata to store.
    """
    checkpoint = load_checkpoint()
    checkpoint[stage] = {
        "completed": True,
        "timestamp": datetime.now().isoformat(),
        **metadata,
    }
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)

    logger.info(f"Checkpoint saved: '{stage}' ✅")


def is_completed(stage: str) -> bool:
    """
    Check if a pipeline stage is already completed.

    Args:
        stage: Pipeline stage name.

    Returns:
        True if stage already completed.
    """
    checkpoint = load_checkpoint()
    return checkpoint.get(stage, {}).get("completed", False)


def reset_checkpoint(stage: str = None) -> None:
    """
    Reset checkpoint — either a specific stage or all.

    Args:
        stage: Stage to reset. If None, resets everything.
    """
    if stage is None:
        CHECKPOINT_FILE.unlink(missing_ok=True)
        logger.info("All checkpoints reset ✅")
    else:
        checkpoint = load_checkpoint()
        checkpoint.pop(stage, None)
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Checkpoint reset for stage: '{stage}' ✅")