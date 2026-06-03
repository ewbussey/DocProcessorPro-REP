import logging
from pathlib import Path

log = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
DICOM_EXTENSIONS = {".dcm"}
DOCX_EXTENSIONS = {".docx"}
PDF_EXTENSIONS = {".pdf"}


def collect_files(input_dir: Path, extensions: set[str]) -> list[Path]:
    """Return a sorted list of files in input_dir whose suffix matches extensions."""
    files = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )
    log.info("Found %d file(s) in %s matching %s", len(files), input_dir, extensions)
    return files


def collect_text_files(input_dir: Path) -> list[Path]:
    """Return sorted list of .txt files in input_dir."""
    return collect_files(input_dir, TEXT_EXTENSIONS)


def collect_image_files(input_dir: Path) -> list[Path]:
    """Return sorted list of image files in input_dir."""
    return collect_files(input_dir, IMAGE_EXTENSIONS)


def collect_dicom_files(input_dir: Path) -> list[Path]:
    """Return sorted list of .dcm files in input_dir."""
    return collect_files(input_dir, DICOM_EXTENSIONS)


def collect_docx_files(input_dir: Path) -> list[Path]:
    """Return sorted list of .docx files in input_dir."""
    return collect_files(input_dir, DOCX_EXTENSIONS)


def collect_pdf_files(input_dir: Path) -> list[Path]:
    """Return sorted list of .pdf files in input_dir."""
    return collect_files(input_dir, PDF_EXTENSIONS)


def resolve_output_path(input_file: Path, output_dir: Path) -> Path:
    """Return the output path for input_file mirrored into output_dir."""
    return output_dir / input_file.name


def ensure_output_dir(output_dir: Path) -> None:
    """Create output_dir and any missing parents if they don't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
