from DocProcessorPro.dpp_scripts.keyword_scanner_scripts.keyword_scanner_codebase import (
    DEFAULT_CATEGORIES,
    scan_directory,
)

# FALLBACK SOURCE CODE ENTRY POINT IF GUI DISABLED

INPUT_DIR = r"C:/path/to/input"  # EDIT BEFORE RUNNING
OUTPUT_DIR = r"C:/path/to/output"  # EDIT BEFORE RUNNING

if __name__ == "__main__":
    scan_directory(INPUT_DIR, OUTPUT_DIR, DEFAULT_CATEGORIES, min_hits=3.0)
