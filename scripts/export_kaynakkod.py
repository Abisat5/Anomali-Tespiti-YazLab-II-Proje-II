"""Proje kaynak kodlarini yorum satirlari temizlenerek tek txt dosyasina aktarir."""

import ast
import io
import re
import tokenize
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = Path(r"c:\Users\oguzhan\Desktop\30_yazlab2_kaynakkod.txt")

SOURCE_PATTERNS = [
    "main.py",
    "run.bat",
    "requirements.txt",
    "configs/config.json",
    "src/**/*.py",
    "tests/**/*.py",
    "scripts/generate_sample_data.py",
    "scripts/check_data.py",
]

EXCLUDE_FILES = {
    "scripts/md_to_pdf_rapor.py",
    "scripts/export_kaynakkod.py",
}


def collect_source_files() -> list[Path]:
    files: list[Path] = []
    for pattern in SOURCE_PATTERNS:
        for path in sorted(PROJECT_ROOT.glob(pattern)):
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            if rel in EXCLUDE_FILES:
                continue
            if any(part in {"venv", "__pycache__", "logs", ".vs"} for part in path.parts):
                continue
            if path.is_file():
                files.append(path)
    unique = sorted(set(files), key=lambda p: p.relative_to(PROJECT_ROOT).as_posix())
    return unique


def remove_docstrings(source: str) -> str:
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                node.body.pop(0)

    return ast.unparse(tree)


def remove_python_comments(source: str) -> str:
    out: list[str] = []
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0

    reader = io.StringIO(source).readline
    for toktype, toktext, start, end, _ in tokenize.generate_tokens(reader):
        sline, scol = start
        eline, ecol = end

        if sline > last_lineno:
            last_col = 0

        if scol > last_col:
            out.append(" " * (scol - last_col))

        if toktype == tokenize.COMMENT:
            pass
        elif toktype in (tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING):
            out.append(toktext)
        else:
            out.append(toktext)

        prev_toktype = toktype
        last_col = ecol
        last_lineno = eline

    text = "".join(out)
    lines = []
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped:
            lines.append(stripped)
        elif lines and lines[-1] != "":
            lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def strip_python_source(source: str) -> str:
    no_doc = remove_docstrings(source)
    return remove_python_comments(no_doc)


def strip_batch_source(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        upper = stripped.upper()
        if upper.startswith("REM ") or upper == "REM":
            continue
        if stripped.startswith("::"):
            continue
        lines.append(line.rstrip())
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def strip_requirements_source(source: str) -> str:
    lines = []
    for line in source.splitlines():
        cleaned = re.sub(r"\s*#.*$", "", line).rstrip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines) + ("\n" if lines else "")


def strip_file_content(path: Path, content: str) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix == ".py":
        return strip_python_source(content)
    if suffix == ".bat":
        return strip_batch_source(content)
    if name == "requirements.txt":
        return strip_requirements_source(content)
    return content if content.endswith("\n") else content + "\n"


def build_output(files: list[Path]) -> str:
    parts = []
    separator = "=" * 80

    for path in files:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        raw = path.read_text(encoding="utf-8")
        cleaned = strip_file_content(path, raw)
        parts.append(separator)
        parts.append(f"DOSYA: {rel}")
        parts.append(separator)
        parts.append(cleaned.rstrip())
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    files = collect_source_files()
    output = build_output(files)
    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print(f"Kaynak kod dosyasi olusturuldu: {OUTPUT_FILE}")
    print(f"Dosya sayisi: {len(files)}")


if __name__ == "__main__":
    main()
