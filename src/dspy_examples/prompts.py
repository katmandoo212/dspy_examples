"""Prompt loading and saving utilities."""

from pathlib import Path


def load_prompt(path: str | Path) -> str:
    """Load a prompt from a file.

    Args:
        path: Path to the prompt file.

    Returns:
        The prompt content as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def save_prompt(
    content: str,
    path: str | Path,
    version: int | None = None,
) -> Path:
    """Save an optimized prompt to a file.

    Args:
        content: The prompt content to save.
        path: Base path for the output file.
        version: Optional version number. If provided, appends _XX to filename.

    Returns:
        The actual path where the file was saved.
    """
    path = Path(path)

    if version is not None:
        # Insert version number before extension
        stem = path.stem
        suffix = path.suffix
        version_str = f"_{version:02d}"
        path = path.with_name(f"{stem}{version_str}{suffix}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path