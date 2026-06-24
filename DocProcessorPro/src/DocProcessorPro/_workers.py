from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal


class _ScanWorker(QThread):
    """Runs scan_directory() off the main thread."""

    progress = Signal(str)
    finished = Signal(
        int, int, int
    )  # (pdfs_processed, total_matches, total_exclusions)
    error = Signal(str)

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        min_hits: float,
        page_buffer: int,
        require_anchor: bool = False,
    ) -> None:
        super().__init__()
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._min_hits = min_hits
        self._page_buffer = page_buffer
        self._require_anchor = require_anchor

    def run(self) -> None:
        try:
            from DocProcessorPro.dpp_scripts.keyword_scanner_scripts.keyword_scanner_codebase import (
                DEFAULT_CATEGORIES,
                scan_directory,
            )

            results = scan_directory(
                self._input_dir,
                self._output_dir,
                DEFAULT_CATEGORIES,
                min_hits=self._min_hits,
                page_buffer=self._page_buffer,
                require_anchor=self._require_anchor,
                progress_callback=self.progress.emit,
            )
            total_matches = sum(len(r.matches) for r in results.values())
            total_exclusions = sum(len(r.exclusions) for r in results.values())
            self.finished.emit(len(results), total_matches, total_exclusions)
        except Exception as exc:
            self.error.emit(str(exc))


class _UpdateChecker(QThread):
    """Fetches version.json and emits update_available if a newer version exists."""

    update_available = Signal(str, str)  # (new_version, download_url)

    def run(self) -> None:
        try:
            from DocProcessorPro.dpp_scripts.update_scripts.update_codebase import (
                fetch_remote_version,
                is_update_available,
            )

            remote = fetch_remote_version()
            if remote and is_update_available(remote):
                self.update_available.emit(remote["version"], remote["download_url"])
        except Exception:
            pass  # silently ignore — update check is best-effort


class _UpdateDownloader(QThread):
    """Downloads the installer to the system temp directory."""

    finished = Signal(str)  # absolute path to downloaded installer
    error = Signal(str)

    def __init__(self, download_url: str) -> None:
        super().__init__()
        self._url = download_url

    def run(self) -> None:
        import tempfile

        try:
            from DocProcessorPro.dpp_scripts.update_scripts.update_codebase import (
                download_installer,
            )

            path = download_installer(self._url, Path(tempfile.gettempdir()))
            if path:
                self.finished.emit(str(path))
            else:
                self.error.emit("Download failed — please update manually.")
        except Exception as exc:
            self.error.emit(str(exc))


class _FeedbackWorker(QThread):
    """Runs apply_feedback() or apply_matched_feedback() off the main thread."""

    progress = Signal(str)
    finished = Signal(
        int, int, int
    )  # (pages_approved_or_kept, pages_rejected, pages_skipped)
    error = Signal(str)

    def __init__(
        self,
        output_dir: str,
        feedback_path: str,
        mode: str = "unmatched",
    ) -> None:
        super().__init__()
        self._output_dir = output_dir
        self._feedback_path = feedback_path
        self._mode = mode

    def run(self) -> None:
        try:
            if self._mode == "matched":
                from DocProcessorPro.dpp_scripts.keyword_scanner_scripts.keyword_scanner_codebase import (
                    apply_matched_feedback,
                )

                kept, removed, skipped = apply_matched_feedback(
                    self._output_dir,
                    self._feedback_path,
                    progress_callback=self.progress.emit,
                )
                self.finished.emit(kept, removed, skipped)
            elif self._mode == "combined":
                from DocProcessorPro.dpp_scripts.keyword_scanner_scripts.keyword_scanner_codebase import (
                    apply_combined_feedback,
                )

                approved, unapproved, skipped = apply_combined_feedback(
                    self._output_dir,
                    self._feedback_path,
                    progress_callback=self.progress.emit,
                )
                self.finished.emit(approved, unapproved, skipped)
            else:
                from DocProcessorPro.dpp_scripts.keyword_scanner_scripts.keyword_scanner_codebase import (
                    apply_feedback,
                )

                approved, rejected, skipped = apply_feedback(
                    self._output_dir,
                    self._feedback_path,
                    progress_callback=self.progress.emit,
                )
                self.finished.emit(approved, rejected, skipped)
        except Exception as exc:
            self.error.emit(str(exc))
