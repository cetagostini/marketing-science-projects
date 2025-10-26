"""Simple JSON-backed persistence for user experiments."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import date, datetime, time

logger = logging.getLogger(__name__)

try:  # Optional dependency; available when pandas is installed.
    import pandas as pd  # type: ignore
    PANDAS_TIMESTAMP = (pd.Timestamp,)  # type: ignore
except Exception:  # pragma: no cover - pandas not always present
    PANDAS_TIMESTAMP = ()


def _store_path() -> Path:
    """Resolve the experiments store path, creating directories as needed."""

    path_str = os.getenv("EXPERIMENT_STORE_PATH")
    if path_str:
        path = Path(path_str).expanduser()
    else:
        default_dir = Path(__file__).resolve().parent / "data"
        default_dir.mkdir(parents=True, exist_ok=True)
        path = default_dir / "experiments_store.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _planning_store_path() -> Path:
    """Resolve the planning sessions store path, creating directories as needed."""

    path_str = os.getenv("PLANNING_STORE_PATH")
    if path_str:
        path = Path(path_str).expanduser()
    else:
        default_dir = Path(__file__).resolve().parent / "data"
        default_dir.mkdir(parents=True, exist_ok=True)
        path = default_dir / "planning_store.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_all() -> Dict[str, List[Dict[str, Any]]]:
    path = _store_path()
    if not path.exists():
        return {}

    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    if not content.strip():
        return {}

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {}

    if isinstance(data, dict):
        normalised: Dict[str, List[Dict[str, Any]]] = {}
        for key, value in data.items():
            if isinstance(value, list):
                normalised[key] = [item for item in value if isinstance(item, dict)]
        return normalised

    return {}


def load_experiments(user_id: str) -> List[Dict[str, Any]]:
    """Return stored experiments for the given user id."""
    
    experiments = _load_all().get(user_id, [])
    logger.info(f"Loaded {len(experiments)} experiments for user '{user_id}' from storage")
    return experiments


def save_experiments(user_id: str, experiments: List[Dict[str, Any]]) -> None:
    """Persist experiments for the given user id."""

    path = _store_path()
    logger.info(f"Saving {len(experiments)} experiments for user '{user_id}' to {path}")
    
    data = _load_all()
    data[user_id] = experiments

    safe_payload = _ensure_json_safe(data)

    try:
        path.write_text(json.dumps(safe_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Successfully saved experiments to {path}")
    except OSError as e:
        logger.error(f"Failed to save experiments to {path}: {e}")


def _load_all_planning() -> Dict[str, List[Dict[str, Any]]]:
    """Load all planning sessions from the planning store."""
    path = _planning_store_path()
    if not path.exists():
        return {}

    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    if not content.strip():
        return {}

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {}

    if isinstance(data, dict):
        normalised: Dict[str, List[Dict[str, Any]]] = {}
        for key, value in data.items():
            if isinstance(value, list):
                normalised[key] = [item for item in value if isinstance(item, dict)]
        return normalised

    return {}


def load_planning_sessions(user_id: str) -> List[Dict[str, Any]]:
    """Return stored planning sessions for the given user id."""
    
    sessions = _load_all_planning().get(user_id, [])
    logger.info(f"Loaded {len(sessions)} planning sessions for user '{user_id}' from storage")
    return sessions


def save_planning_sessions(user_id: str, sessions: List[Dict[str, Any]]) -> None:
    """Persist planning sessions for the given user id."""

    path = _planning_store_path()
    logger.info(f"Saving {len(sessions)} planning sessions for user '{user_id}' to {path}")
    
    data = _load_all_planning()
    data[user_id] = sessions

    safe_payload = _ensure_json_safe(data)

    try:
        path.write_text(json.dumps(safe_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Successfully saved planning sessions to {path}")
    except OSError as e:
        logger.error(f"Failed to save planning sessions to {path}: {e}")


def _ensure_json_safe(value: Any) -> Any:
    """Recursively convert values into JSON-serialisable structures."""

    if isinstance(value, dict):
        return {key: _ensure_json_safe(val) for key, val in value.items()}

    if isinstance(value, list):
        return [_ensure_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_ensure_json_safe(item) for item in value]

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    if PANDAS_TIMESTAMP and isinstance(value, PANDAS_TIMESTAMP):  # type: ignore[arg-type]
        return value.to_pydatetime().isoformat()

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            return str(value)

    if hasattr(value, "tolist"):
        try:
            return _ensure_json_safe(value.tolist())
        except Exception:
            return str(value)

    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value

