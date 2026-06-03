import logging
import logging.config
import os
import traceback
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml
from pythonjsonlogger import jsonlogger

try:
    from importlib.metadata import version, metadata as pkg_metadata
    APP_NAME = pkg_metadata("DocProcessorPro").get("Name", "DocProcessorPro")
    APP_VERSION = version("DocProcessorPro")
except Exception:
    APP_NAME = "DocProcessorPro"
    APP_VERSION = "unknown"


def _get_log_dir() -> Path:
    # LOCALAPPDATA (non-roaming) matches what update_codebase.py uses and is
    # writable without admin rights under a standard Inno Setup installation.
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        return base / APP_NAME / "logs"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / APP_NAME
    else:
        return Path.home() / ".local" / "share" / APP_NAME / "logs"


_DEFAULT_LOG_PATH = str(_get_log_dir() / "docprocessorpro.log")


class GlobalContextFilter(logging.Filter):
    def __init__(self, name="", **fields):
        super().__init__(name)
        self._fields = {"app_name": APP_NAME, "version": APP_VERSION, **fields}

    def filter(self, record: logging.LogRecord):
        for key, value in self._fields.items():
            setattr(record, key, value)
        return True


class DefaultJSONFormatter(jsonlogger.JsonFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(json_indent=2, *args, **kwargs)

    def add_fields(self, log_data, record, message_dict):
        super().add_fields(log_data, record, message_dict)

        if record.exc_info:
            exc_type, exc_value, exc_traceback = record.exc_info
            log_data["exception"] = {
                "exception_type": exc_type.__name__,
                "exception_value": str(exc_value),
                "traceback": traceback.format_exception(
                    exc_type, exc_value, exc_traceback
                ),
            }
            log_data.pop("exc_info", None)
            log_data.pop("exc_text", None)


class DefaultConsoleFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def formatException(self, ei) -> str:
        return "".join(traceback.format_exception(*ei)).rstrip()


class JSONFileHandler(RotatingFileHandler):
    def __init__(self, filename=_DEFAULT_LOG_PATH, maxBytes=500000, backupCount=5, mode='a', delay=True):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(filename, maxBytes=maxBytes, backupCount=backupCount, mode=mode, delay=delay)
        self.setFormatter(DefaultJSONFormatter())
        self.terminator = "\n\n"
        sys.excepthook = self._handle_exception

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger(__name__).critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )


def setup_logging():
    config_path = Path(__file__).parent / "logging_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)
    logging.getLogger(__name__).debug("Log file: %s", _DEFAULT_LOG_PATH)
