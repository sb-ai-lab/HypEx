#!/usr/bin/env python3
"""Collect project files into a single output file for LLM consumption.

Default behavior excludes standard ignores and test files. No truncation, 
file-count limits, or character limits are applied unless explicitly requested.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, TextIO

SEPARATOR: str = "=" * 70

DEFAULT_EXTENSIONS: frozenset[str] = frozenset({".py"})

DEFAULT_EXCLUDE_NAMES: frozenset[str] = frozenset({
    "__pycache__",
    ".git",
    ".github",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".vscode",
    ".idea",
    ".DS_Store",
    "dist",
    "build",
    "Thumbs.db",
    "collect_code.py",
    "docs",
    "examples"
})

DEFAULT_EXCLUDE_PATTERNS: frozenset[str] = frozenset({
    "*.egg-info",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.exe",
})

# Test-related files and directories excluded by default.
DEFAULT_TEST_EXCLUDE_NAMES: frozenset[str] = frozenset({
    "tests",
    "test",
    "testing",
    "unitests",
    "conftest.py",
    "tests.py",
    "test.py",
})

DEFAULT_TEST_EXCLUDE_PATTERNS: frozenset[str] = frozenset({
    "test_*.py",
    "*_test.py",
})

MARKDOWN_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".md": "markdown",
    ".markdown": "markdown",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".txt": "text",
    ".rst": "rst",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".sh": "bash",
    ".bash": "bash",
}

TEST_IMPORT_PREFIXES: tuple[str, ...] = (
    "pytest",
    "unittest",
)


@dataclass(frozen=True)
class OutputOptions:
    """Options controlling how collected files are written."""

    output_format: str
    include_toc: bool
    include_tree: bool
    line_numbers: bool
    list_only: bool
    include_timestamp: bool
    preferred_encoding: str
    max_file_chars: int | None
    max_total_chars: int | None
    quiet: bool
    include_tests: bool


def current_timestamp() -> str:
    """Return the current local timestamp as a stable string.

    Returns:
        Timestamp string in YYYY-MM-DD HH:MM:SS format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def positive_int_or_none(value: int | None) -> int | None:
    """Return the value if it is a positive integer, otherwise return None.

    Args:
        value: Integer value or None.

    Returns:
        Positive integer or None.
    """
    return value if value is not None and value > 0 else None


def normalize_pattern(value: str) -> str:
    """Normalize an exclude pattern to a POSIX-like representation.

    Args:
        value: Raw pattern supplied by the user.

    Returns:
        Normalized pattern string.
    """
    item = value.strip().replace("\\", "/")
    while item.startswith("./"):
        item = item[2:]
    return item


def parse_excludes(
    user_excludes: list[str],
    use_defaults: bool = True,
    include_tests: bool = False,
) -> tuple[set[str], set[str], set[str]]:
    """Parse CLI exclude values into exact names, exact paths, and glob patterns.

    Args:
        user_excludes: Raw exclude values from argparse.
        use_defaults: Whether default excludes should be included.
        include_tests: Whether test files should be included (skips test defaults).

    Returns:
        A tuple containing:
        - exact file/directory names,
        - exact relative POSIX paths,
        - glob patterns.
    """
    exact_names: set[str] = set()
    exact_rel_paths: set[str] = set()
    glob_patterns: set[str] = set()

    if use_defaults:
        exact_names.update(DEFAULT_EXCLUDE_NAMES)
        glob_patterns.update(DEFAULT_EXCLUDE_PATTERNS)
        
        if not include_tests:
            exact_names.update(DEFAULT_TEST_EXCLUDE_NAMES)
            glob_patterns.update(DEFAULT_TEST_EXCLUDE_PATTERNS)

    for raw_item in user_excludes:
        item = normalize_pattern(raw_item)
        if not item:
            continue

        if any(ch in item for ch in "*?["):
            glob_patterns.add(item)
        elif "/" in item:
            exact_rel_paths.add(item)
        else:
            exact_names.add(item)

    return exact_names, exact_rel_paths, glob_patterns


def is_excluded(
    rel_path: Path,
    exact_names: set[str],
    exact_rel_paths: set[str],
    glob_patterns: set[str],
) -> bool:
    """Check whether a project-relative path should be excluded.

    Args:
        rel_path: Path relative to the project root.
        exact_names: Exact file/directory names to exclude.
        exact_rel_paths: Exact relative POSIX paths to exclude.
        glob_patterns: Glob patterns to exclude.

    Returns:
        True if the path should be excluded, False otherwise.
    """
    rel = rel_path.as_posix()
    name = rel_path.name

    if name in exact_names or rel in exact_rel_paths:
        return True

    for pattern in glob_patterns:
        if fnmatch(name, pattern) or fnmatch(rel, pattern):
            return True

    return False


def is_test_related_module(module_name: str) -> bool:
    """Return True if a module name looks like a test-framework module.

    Args:
        module_name: Imported module name.

    Returns:
        True if the module is related to pytest or unittest.
    """
    return any(module_name.startswith(prefix) for prefix in TEST_IMPORT_PREFIXES)


def looks_like_test_content(content: str) -> bool:
    """Return True if Python content imports pytest or unittest.

    Args:
        content: Python file content.

    Returns:
        True if the file imports pytest or unittest.
    """
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        # If we can't parse it, we can't reliably detect test imports.
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if is_test_related_module(alias.name):
                    return True

        elif isinstance(node, ast.ImportFrom):
            if node.module and is_test_related_module(node.module):
                return True

    return False


def parse_extensions(raw: str) -> frozenset[str]:
    """Parse comma-separated extensions into normalized suffixes.

    Args:
        raw: Raw CLI value, for example: "py,md,toml".

    Returns:
        Frozen set of normalized extensions like {".py", ".md", ".toml"}.
    """
    extensions: set[str] = set()

    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue

        if not part.startswith("."):
            part = f".{part}"

        extensions.add(part)

    if not extensions:
        return DEFAULT_EXTENSIONS

    return frozenset(extensions)


def collect_files(
    project_root: Path,
    exact_names: set[str],
    exact_rel_paths: set[str],
    glob_patterns: set[str],
    extensions: frozenset[str],
    output_file: Path,
) -> tuple[list[Path], list[str]]:
    """Collect matching files from the project root.

    Args:
        project_root: Absolute path to the project root.
        exact_names: Exact file/directory names to exclude.
        exact_rel_paths: Exact relative POSIX paths to exclude.
        glob_patterns: Glob patterns to exclude.
        extensions: File suffixes to include.
        output_file: Output file path, excluded from collection if present.

    Returns:
        A tuple containing:
        - sorted list of collected file paths,
        - list of non-fatal walk warnings.
    """
    files: list[Path] = []
    warnings: list[str] = []

    def on_walk_error(error: OSError) -> None:
        """Collect directory walk errors without stopping the whole process.

        Args:
            error: OS error produced by os.walk().
        """
        warnings.append(str(error))

    for root, dirs, filenames in os.walk(project_root, topdown=True, onerror=on_walk_error):
        rel_root = Path(root).relative_to(project_root)

        # Filter directories in-place so os.walk() does not descend into them.
        dirs[:] = sorted(
            d
            for d in dirs
            if not is_excluded(rel_root / d, exact_names, exact_rel_paths, glob_patterns)
        )

        for filename in filenames:
            file_path = Path(root) / filename

            # Prevent including a previous output file if it matches extensions.
            if file_path == output_file:
                continue

            if Path(filename).suffix.lower() not in extensions:
                continue

            rel_file = rel_root / filename
            if is_excluded(rel_file, exact_names, exact_rel_paths, glob_patterns):
                continue

            files.append(file_path)

    files.sort(key=lambda path: path.relative_to(project_root).as_posix().lower())
    return files, warnings


def build_project_tree(
    project_root: Path,
    exact_names: set[str],
    exact_rel_paths: set[str],
    glob_patterns: set[str],
    output_file: Path | None = None,
) -> str:
    """Build a visual directory tree of the project, respecting directory exclusions.

    Shows all non-excluded files and directories, regardless of file extensions,
    to give the LLM full context of the project layout.

    Args:
        project_root: Absolute path to the project root.
        exact_names: Exact file/directory names to exclude.
        exact_rel_paths: Exact relative POSIX paths to exclude.
        glob_patterns: Glob patterns to exclude.
        output_file: Output file path, excluded from the tree if present.

    Returns:
        Formatted string representing the directory tree.
    """
    tree: dict[str, Any] = {}

    for root, dirs, filenames in os.walk(project_root, topdown=True):
        rel_root = Path(root).relative_to(project_root)

        # Filter directories in-place
        dirs[:] = sorted(
            d for d in dirs
            if not is_excluded(rel_root / d, exact_names, exact_rel_paths, glob_patterns)
        )

        # Filter files
        filtered_filenames = sorted(
            f for f in filenames
            if not is_excluded(rel_root / f, exact_names, exact_rel_paths, glob_patterns)
            and (output_file is None or (Path(root) / f) != output_file)
        )

        node = tree
        if rel_root != Path("."):
            for part in rel_root.parts:
                node = node.setdefault(part, {})

        for f in filtered_filenames:
            node[f] = None

    root_name = project_root.name or "."
    lines: list[str] = [f"{root_name}/"]

    def walk(node: dict[str, Any], prefix: str) -> None:
        # Sort: directories first (subtree is dict), then files (subtree is None).
        # Secondary sort: case-insensitive alphabetical.
        items = sorted(node.items(), key=lambda x: (x[1] is None, x[0].lower()))

        for i, (name, subtree) in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "

            if subtree is None:
                lines.append(f"{prefix}{connector}{name}")
            else:
                lines.append(f"{prefix}{connector}{name}/")
                extension = "    " if is_last else "│   "
                walk(subtree, prefix + extension)

    walk(tree, "")
    return "\n".join(lines)


def write_project_tree(
    out: TextIO,
    project_root: Path,
    exact_names: set[str],
    exact_rel_paths: set[str],
    glob_patterns: set[str],
    output_file: Path,
    output_format: str,
) -> None:
    """Write the project directory tree to the output stream.

    Args:
        out: Output stream.
        project_root: Project root.
        exact_names: Exact file/directory names to exclude.
        exact_rel_paths: Exact relative POSIX paths to exclude.
        glob_patterns: Glob patterns to exclude.
        output_file: Output file path.
        output_format: Output format, either "text" or "md".
    """
    tree_str = build_project_tree(
        project_root,
        exact_names,
        exact_rel_paths,
        glob_patterns,
        output_file,
    )

    if output_format == "md":
        out.write("## Project structure\n\n")
        out.write("```text\n")
        out.write(tree_str)
        if not tree_str.endswith("\n"):
            out.write("\n")
        out.write("```\n\n")
    else:
        out.write("Project structure:\n")
        out.write(tree_str)
        out.write("\n\n")


def read_text_smart(file_path: Path, preferred_encoding: str) -> tuple[str, str]:
    """Read a text file with encoding fallbacks.

    Args:
        file_path: File path to read.
        preferred_encoding: Preferred encoding supplied by the user.

    Returns:
        A tuple containing:
        - decoded file content,
        - encoding that was actually used.
    """
    preferred = (preferred_encoding or "utf-8").strip()
    candidates: list[str] = []

    # For UTF-8, try utf-8-sig first to strip BOM if present.
    if preferred.lower() in {"utf-8", "utf8", "utf_8"}:
        candidates.extend(["utf-8-sig", "utf-8"])
    else:
        candidates.append(preferred)
        candidates.extend(["utf-8-sig", "utf-8"])

    # Common fallbacks for legacy or mixed projects.
    candidates.extend(["cp1251", "latin-1"])

    seen: set[str] = set()
    unique_candidates: list[str] = []

    for encoding in candidates:
        key = encoding.lower()
        if key not in seen:
            seen.add(key)
            unique_candidates.append(encoding)

    for encoding in unique_candidates:
        try:
            return file_path.read_text(encoding=encoding), encoding
        except (UnicodeDecodeError, LookupError):
            continue

    # Last resort: replace invalid bytes instead of failing.
    return file_path.read_text(encoding="utf-8", errors="replace"), "utf-8-replace"


def add_line_numbers(content: str) -> str:
    """Prefix each line with a human-readable line number.

    Args:
        content: Original text content.

    Returns:
        Content with line numbers.
    """
    return "\n".join(
        f"{line_number:4} | {line}"
        for line_number, line in enumerate(content.splitlines(), start=1)
    )


def is_probably_binary(content: str) -> bool:
    """Heuristically detect binary content.

    Args:
        content: Decoded content.

    Returns:
        True if the content appears to be binary.
    """
    return "\x00" in content[:8192]


def choose_fence(content: str) -> str:
    """Choose a Markdown code fence that does not appear inside content.

    Args:
        content: File content.

    Returns:
        Markdown fence string.
    """
    fence = "```"
    while fence in content:
        fence += "`"
    return fence


def markdown_language(file_path: Path) -> str:
    """Return a Markdown language hint for a file path.

    Args:
        file_path: File path.

    Returns:
        Language hint string, possibly empty.
    """
    if file_path.name.lower() == "dockerfile":
        return "docker"

    return MARKDOWN_LANGUAGES.get(file_path.suffix.lower(), "")


def write_file_list(
    out: TextIO,
    files: list[Path],
    project_root: Path,
    output_format: str,
    title: str,
) -> None:
    """Write a list of files as plain text or Markdown.

    Args:
        out: Output stream.
        files: Files to list.
        project_root: Project root used to compute relative paths.
        output_format: Output format, either "text" or "md".
        title: Section title.
    """
    is_md = output_format == "md"

    if is_md:
        out.write(f"## {title}\n\n")
    else:
        out.write(f"{title}:\n")

    for file_path in files:
        rel_path = file_path.relative_to(project_root).as_posix()
        if is_md:
            out.write(f"- `{rel_path}`\n")
        else:
            out.write(f"- {rel_path}\n")

    out.write("\n")


def write_header(
    out: TextIO,
    project_root: Path,
    files_found: int,
    files_selected: int,
    options: OutputOptions,
) -> None:
    """Write the output header.

    Args:
        out: Output stream.
        project_root: Project root.
        files_found: Total number of files found before optional limits.
        files_selected: Number of files selected for output.
        options: Output options.
    """
    if options.output_format == "md":
        out.write("# Project code collection\n\n")
        out.write(f"- Project root: `{project_root.resolve().as_posix()}`\n")
        out.write(f"- Files found: {files_found}\n")
        out.write(f"- Files selected: {files_selected}\n")

        if options.include_timestamp:
            out.write(f"- Generated at: {current_timestamp()}\n")

        out.write(f"- Output format: `{options.output_format}`\n")
        out.write("\n---\n\n")
        return

    out.write(f"Project code collection: {project_root.resolve().as_posix()}\n")
    out.write(f"Files found: {files_found}\n")
    out.write(f"Files selected: {files_selected}\n")

    if options.include_timestamp:
        out.write(f"Generated at: {current_timestamp()}\n")

    out.write(f"Output format: {options.output_format}\n\n")
    out.write(f"{SEPARATOR}\n\n")


def write_footer(
    out: TextIO,
    output_format: str,
    files_found: int,
    files_included: int,
    total_lines: int,
    total_chars: int,
    skipped: list[tuple[str, str]],
    failed: list[tuple[str, str]],
) -> None:
    """Write final statistics and problem lists.

    Args:
        out: Output stream.
        output_format: Output format, either "text" or "md".
        files_found: Total number of files found before optional limits.
        files_included: Number of files actually included in the output.
        total_lines: Total number of included lines.
        total_chars: Total number of included characters.
        skipped: Skipped file paths and reasons.
        failed: Failed file paths and error messages.
    """
    approximate_tokens = total_chars // 4

    if output_format == "md":
        out.write("---\n\n")
        out.write("## Final statistics\n\n")
        out.write(f"- Files found: {files_found}\n")
        out.write(f"- Files included: {files_included}\n")
        out.write(f"- Lines included: {total_lines}\n")
        out.write(f"- Characters included: {total_chars}\n")
        out.write(f"- Approximate tokens: {approximate_tokens}\n")

        if skipped:
            out.write("\n### Skipped files\n\n")
            for rel_path, reason in skipped:
                out.write(f"- `{rel_path}`: {reason}\n")

        if failed:
            out.write("\n### Failed files\n\n")
            for rel_path, reason in failed:
                out.write(f"- `{rel_path}`: {reason}\n")

        out.write("\n")
        return

    out.write(f"{SEPARATOR}\nFINAL STATISTICS\n{SEPARATOR}\n")
    out.write(f"Files found: {files_found}\n")
    out.write(f"Files included: {files_included}\n")
    out.write(f"Lines included: {total_lines}\n")
    out.write(f"Characters included: {total_chars}\n")
    out.write(f"Approximate tokens: {approximate_tokens}\n")

    if skipped:
        out.write(f"\nSkipped files ({len(skipped)}):\n")
        for rel_path, reason in skipped:
            out.write(f"  - {rel_path}: {reason}\n")

    if failed:
        out.write(f"\nFailed files ({len(failed)}):\n")
        for rel_path, reason in failed:
            out.write(f"  - {rel_path}: {reason}\n")


def print_summary(
    options: OutputOptions,
    output_file: Path,
    files_found: int,
    files_included: int,
    total_lines: int,
    total_chars: int,
    skipped_count: int,
    failed_count: int,
) -> None:
    """Print a short summary to stdout unless quiet mode is enabled.

    Args:
        options: Output options.
        output_file: Output file path.
        files_found: Total number of files found before optional limits.
        files_included: Number of files actually included or listed.
        total_lines: Total number of included lines.
        total_chars: Total number of included characters.
        skipped_count: Number of skipped files.
        failed_count: Number of failed files.
    """
    if options.quiet:
        return

    print("✓ Collection finished")
    print(f"  Output file: {output_file.resolve().as_posix()}")
    print(f"  Files found: {files_found}")
    print(f"  Files included: {files_included}")

    if not options.list_only:
        print(f"  Lines included: {total_lines}")
        print(f"  Characters included: {total_chars}")
        print(f"  Approximate tokens: {total_chars // 4}")

    if skipped_count:
        print(f"  Skipped: {skipped_count}")

    if failed_count:
        print(f"  Failed: {failed_count}")


def write_output(
    files: list[Path],
    project_root: Path,
    output_file: Path,
    options: OutputOptions,
    exact_names: set[str],
    exact_rel_paths: set[str],
    glob_patterns: set[str],
    pre_skipped: list[tuple[Path, str]] | None = None,
) -> None:
    """Write collected files to the output file.

    Args:
        files: Selected files to write.
        project_root: Project root.
        output_file: Output file path.
        options: Output options.
        exact_names: Exact file/directory names to exclude.
        exact_rel_paths: Exact relative POSIX paths to exclude.
        glob_patterns: Glob patterns to exclude.
        pre_skipped: Files skipped before writing, for example due to max-files limit.
    """
    skipped: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    for file_path, reason in pre_skipped or []:
        skipped.append((file_path.relative_to(project_root).as_posix(), reason))

    files_found = len(files) + len(pre_skipped or [])

    try:
        with output_file.open("w", encoding="utf-8", newline="\n") as out:
            if options.list_only:
                if options.output_format == "jsonl":
                    for file_path in files:
                        rel_path = file_path.relative_to(project_root).as_posix()
                        out.write(json.dumps({"path": rel_path}, ensure_ascii=False) + "\n")

                    meta: dict[str, object] = {
                        "project_root": project_root.resolve().as_posix(),
                        "files_found": files_found,
                        "files_listed": len(files),
                        "skipped": [
                            {"path": rel_path, "reason": reason}
                            for rel_path, reason in skipped
                        ],
                        "failed": [
                            {"path": rel_path, "reason": reason}
                            for rel_path, reason in failed
                        ],
                    }

                    if options.include_timestamp:
                        meta["generated_at"] = current_timestamp()

                    out.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
                else:
                    write_header(out, project_root, files_found, len(files), options)
                    
                    if options.include_tree:
                        write_project_tree(
                            out, project_root, exact_names, exact_rel_paths, 
                            glob_patterns, output_file, options.output_format
                        )
                        
                    write_file_list(out, files, project_root, options.output_format, "File list")
                    write_footer(
                        out,
                        options.output_format,
                        files_found,
                        len(files),
                        0,
                        0,
                        skipped,
                        failed,
                    )

                print_summary(
                    options=options,
                    output_file=output_file,
                    files_found=files_found,
                    files_included=len(files),
                    total_lines=0,
                    total_chars=0,
                    skipped_count=len(skipped),
                    failed_count=len(failed),
                )
                return

            if options.output_format != "jsonl":
                write_header(out, project_root, files_found, len(files), options)

                if options.include_tree:
                    write_project_tree(
                        out, project_root, exact_names, exact_rel_paths, 
                        glob_patterns, output_file, options.output_format
                    )

                if options.include_toc:
                    write_file_list(
                        out,
                        files,
                        project_root,
                        options.output_format,
                        "Table of contents",
                    )

            total_lines = 0
            total_chars = 0
            processed = 0

            normal_encodings = {
                options.preferred_encoding.lower(),
                "utf-8",
                "utf-8-sig",
                "utf8",
            }

            for file_path in files:
                rel_path = file_path.relative_to(project_root)
                rel_posix = rel_path.as_posix()

                try:
                    file_size = file_path.stat().st_size
                except OSError as exc:
                    failed.append((rel_posix, str(exc)))
                    continue

                if options.max_file_chars is not None and file_size > options.max_file_chars:
                    skipped.append(
                        (
                            rel_posix,
                            f"file size is larger than max file chars "
                            f"(approximate: {file_size} bytes)",
                        )
                    )
                    continue

                if options.max_total_chars is not None:
                    remaining = options.max_total_chars - total_chars
                    if remaining <= 0:
                        skipped.append((rel_posix, "total character limit reached"))
                        continue

                    if file_size > remaining:
                        skipped.append(
                            (
                                rel_posix,
                                "file size is larger than remaining total character limit "
                                "(approximate)",
                            )
                        )
                        continue

                try:
                    content, used_encoding = read_text_smart(
                        file_path,
                        options.preferred_encoding,
                    )
                except Exception as exc:  # noqa: BLE001 - report and continue.
                    failed.append((rel_posix, str(exc)))
                    continue

                if is_probably_binary(content):
                    skipped.append((rel_posix, "probably binary file"))
                    continue

                # AST-based test detection for Python files.
                if (
                    not options.include_tests 
                    and file_path.suffix.lower() == ".py"
                    and looks_like_test_content(content)
                ):
                    skipped.append((rel_posix, "detected as test file by content (pytest/unittest)"))
                    continue

                line_count = len(content.splitlines())

                if options.line_numbers:
                    content = add_line_numbers(content)

                char_count = len(content)

                if options.max_file_chars is not None and char_count > options.max_file_chars:
                    skipped.append(
                        (
                            rel_posix,
                            "formatted content exceeds max file chars",
                        )
                    )
                    continue

                if options.max_total_chars is not None:
                    if total_chars + char_count > options.max_total_chars:
                        skipped.append(
                            (
                                rel_posix,
                                "formatted content exceeds remaining total chars",
                            )
                        )
                        continue

                encoding_is_expected = used_encoding.lower() in normal_encodings

                if options.output_format == "jsonl":
                    record: dict[str, object] = {
                        "path": rel_posix,
                        "content": content,
                    }

                    if not encoding_is_expected:
                        record["encoding_warning"] = used_encoding

                    out.write(json.dumps(record, ensure_ascii=False) + "\n")

                elif options.output_format == "md":
                    fence = choose_fence(content)
                    language = markdown_language(file_path)

                    out.write(f"## `{rel_posix}`\n\n")

                    if not encoding_is_expected:
                        out.write(f"> [WARNING] File was read using `{used_encoding}`.\n\n")

                    out.write(f"{fence}{language}\n")
                    out.write(content)

                    if content and not content.endswith("\n"):
                        out.write("\n")

                    out.write(f"{fence}\n\n")

                else:
                    out.write(f"File: {rel_posix}\n{SEPARATOR}\n\n")

                    if not encoding_is_expected:
                        out.write(f"# [WARNING: file was read using {used_encoding}]\n")

                    out.write(content)

                    if content and not content.endswith("\n"):
                        out.write("\n")

                    out.write("\n\n")

                processed += 1
                total_lines += line_count
                total_chars += char_count

            if options.output_format == "jsonl":
                meta = {
                    "project_root": project_root.resolve().as_posix(),
                    "files_found": files_found,
                    "files_selected": len(files),
                    "files_included": processed,
                    "lines_included": total_lines,
                    "characters_included": total_chars,
                    "approximate_tokens": total_chars // 4,
                    "skipped": [
                        {"path": rel_path, "reason": reason}
                        for rel_path, reason in skipped
                    ],
                    "failed": [
                        {"path": rel_path, "reason": reason}
                        for rel_path, reason in failed
                    ],
                }
                
                if options.include_tree:
                    meta["project_tree"] = build_project_tree(
                        project_root, exact_names, exact_rel_paths, glob_patterns, output_file
                    )

                if options.include_timestamp:
                    meta["generated_at"] = current_timestamp()

                out.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
            else:
                write_footer(
                    out,
                    options.output_format,
                    files_found,
                    processed,
                    total_lines,
                    total_chars,
                    skipped,
                    failed,
                )

            print_summary(
                options=options,
                output_file=output_file,
                files_found=files_found,
                files_included=processed,
                total_lines=total_lines,
                total_chars=total_chars,
                skipped_count=len(skipped),
                failed_count=len(failed),
            )

    except Exception as exc:  # noqa: BLE001 - final fatal write error.
        print(f"❌ Failed to write output file {output_file}: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Collect project files into a single output file for LLM usage."
    )

    parser.add_argument(
        "project_path",
        nargs="?",
        default=".",
        help="Project root directory. Defaults to the current directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="project_code.txt",
        help="Output file path. Defaults to project_code.txt.",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        action="append",
        default=[],
        help="Additional exclude pattern. Can be used multiple times.",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Disable default exclude patterns.",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test files and directories (they are excluded by default).",
    )
    parser.add_argument(
        "--no-tree",
        action="store_true",
        help="Do not include the project directory tree in the output.",
    )
    parser.add_argument(
        "--extensions",
        default="py",
        help="Comma-separated file extensions to include. Defaults to py.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "md", "markdown", "jsonl"),
        default="text",
        help="Output format. Defaults to text.",
    )
    parser.add_argument(
        "--toc",
        action="store_true",
        help="Include a table of contents before file contents.",
    )
    parser.add_argument(
        "--line-numbers",
        action="store_true",
        help="Prefix each line with its line number.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Write only the list of selected files, not their contents.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to include. Default: no limit.",
    )
    parser.add_argument(
        "--max-file-chars",
        type=int,
        default=None,
        help="Skip files larger than this number of characters. Default: no limit.",
    )
    parser.add_argument(
        "--max-total-chars",
        type=int,
        default=None,
        help="Stop adding content after this number of characters. Default: no limit.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Preferred encoding for reading files. Defaults to utf-8.",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Do not include a timestamp in the output header.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable summary output.",
    )

    args = parser.parse_args()

    project_root = Path(args.project_path).resolve()

    if not project_root.exists():
        print(f"❌ Path does not exist: {project_root}", file=sys.stderr)
        sys.exit(1)

    if not project_root.is_dir():
        print(f"❌ Specified path is not a directory: {project_root}", file=sys.stderr)
        sys.exit(1)

    output_file = Path(args.output).resolve()

    if output_file.is_dir():
        print(f"❌ Output path is a directory: {output_file}", file=sys.stderr)
        sys.exit(1)

    extensions = parse_extensions(args.extensions)
    exact_names, exact_rel_paths, glob_patterns = parse_excludes(
        args.exclude,
        use_defaults=not args.no_default_excludes,
        include_tests=args.include_tests,
    )

    if not args.quiet:
        print(f"🔍 Searching files in: {project_root.as_posix()}")
        print(f"📄 Extensions: {', '.join(sorted(extensions))}")

        all_excludes = sorted(exact_names | exact_rel_paths | glob_patterns)
        if all_excludes:
            print(f"⛔ Excludes: {', '.join(all_excludes)}")

    files, walk_warnings = collect_files(
        project_root=project_root,
        exact_names=exact_names,
        exact_rel_paths=exact_rel_paths,
        glob_patterns=glob_patterns,
        extensions=extensions,
        output_file=output_file,
    )

    for warning in walk_warnings:
        print(f"⚠ Walk warning: {warning}", file=sys.stderr)

    if not files:
        print("⚠ No matching files found", file=sys.stderr)
        sys.exit(0)

    max_files = positive_int_or_none(args.max_files)
    pre_skipped: list[tuple[Path, str]] = []

    if max_files is not None and len(files) > max_files:
        pre_skipped.extend(
            (file_path, "max files limit")
            for file_path in files[max_files:]
        )
        files = files[:max_files]

    options = OutputOptions(
        output_format="md" if args.format in {"md", "markdown"} else args.format,
        include_toc=args.toc,
        include_tree=not args.no_tree,
        line_numbers=args.line_numbers,
        list_only=args.list_only,
        include_timestamp=not args.no_timestamp,
        preferred_encoding=args.encoding,
        max_file_chars=positive_int_or_none(args.max_file_chars),
        max_total_chars=positive_int_or_none(args.max_total_chars),
        quiet=args.quiet,
        include_tests=args.include_tests,
    )

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001 - fatal configuration error.
        print(f"❌ Cannot create output directory for {output_file}: {exc}", file=sys.stderr)
        sys.exit(1)

    write_output(
        files=files,
        project_root=project_root,
        output_file=output_file,
        options=options,
        exact_names=exact_names,
        exact_rel_paths=exact_rel_paths,
        glob_patterns=glob_patterns,
        pre_skipped=pre_skipped,
    )


if __name__ == "__main__":
    main()