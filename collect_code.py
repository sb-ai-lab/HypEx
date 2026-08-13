#!/usr/bin/env python3
"""
Скрипт для сбора всех .py файлов проекта в один текстовый файл.
Требует Python 3.9+ (встроенные дженерики list, set, tuple).
"""
from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

SEPARATOR: str = "=" * 70

DEFAULT_EXCLUDE_NAMES: frozenset[str] = frozenset({
    '__pycache__', '.git', '.venv', 'venv', 'env', '.env',
    'node_modules', '.mypy_cache', '.pytest_cache', '.vscode',
    '.idea', '.DS_Store', 'dist', 'build', 'Thumbs.db'
})

DEFAULT_EXCLUDE_PATTERNS: frozenset[str] = frozenset({
    '*.egg-info', '*.pyc', '*.pyo', '*.pyd', '*.so', '*.dll', '*.exe'
})


def _parse_excludes(
    user_excludes: list[str],
    use_defaults: bool = True
) -> tuple[set[str], set[str]]:
    """Разделяет шаблоны исключений на точные имена и glob-паттерны."""
    exact_names: set[str] = set()
    glob_patterns: set[str] = set()

    if use_defaults:
        exact_names.update(DEFAULT_EXCLUDE_NAMES)
        glob_patterns.update(DEFAULT_EXCLUDE_PATTERNS)

    for item in user_excludes:
        if '*' in item:
            glob_patterns.add(item)
        else:
            exact_names.add(item)

    return exact_names, glob_patterns


def _is_excluded(
    path: Path,
    base_path: Path,
    exact_names: set[str],
    glob_patterns: set[str]
) -> bool:
    """Проверяет, должен ли путь быть исключён из обхода."""
    if path.name in exact_names:
        return True
    
    # Исправлено: path.match(p) вместо path.name.match(p)
    # pathlib.Path.match() корректно работает с glob-паттернами
    return any(path.match(p) for p in glob_patterns)


def collect_python_files(
    project_root: Path,
    exact_names: set[str],
    glob_patterns: set[str]
) -> list[Path]:
    """Рекурсивно собирает все .py файлы, исключая указанные директории и файлы."""
    python_files: list[Path] = []

    try:
        for root, dirs, files in os.walk(project_root, topdown=True):
            # Фильтрация директорий "на лету"
            dirs[:] = [
                d for d in dirs
                if d not in exact_names and not any(Path(d).match(p) for p in glob_patterns)
            ]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if not _is_excluded(file_path, project_root, exact_names, glob_patterns):
                        python_files.append(file_path)

    except PermissionError as e:
        print(f"⚠ Ошибка доступа при обходе {project_root}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"❌ Критическая ошибка при обходе директории: {e}", file=sys.stderr)
        sys.exit(1)

    python_files.sort(key=lambda p: str(p.relative_to(project_root)))
    return python_files


def write_files_to_txt(
    files: list[Path],
    project_root: Path,
    output_file: Path,
    encoding: str = 'utf-8'
) -> None:
    """Записывает содержимое файлов в текстовый файл с разделителями и статистикой."""
    total_lines: int = 0
    failed_files: list[tuple[Path, Exception]] = []
    now_str: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(output_file, 'w', encoding=encoding) as out_f:
            out_f.write(f"Сборка Python-кода проекта: {project_root.resolve()}\n")
            out_f.write(f"Всего файлов: {len(files)}\n")
            out_f.write(f"Дата сборки: {now_str}\n\n")
            out_f.write(f"{SEPARATOR}\n\n")

            for file_path in files:
                rel_path = file_path.relative_to(project_root)
                out_f.write(f"Файл: {rel_path}\n{SEPARATOR}\n\n")

                try:
                    content = file_path.read_text(encoding=encoding)
                    out_f.write(content)
                    total_lines += len(content.splitlines())
                except UnicodeDecodeError:
                    try:
                        content = file_path.read_text(encoding='latin-1')
                        out_f.write(f"# [ПРЕДУПРЕЖДЕНИЕ: файл прочитан в кодировке latin-1]\n{content}")
                        total_lines += len(content.splitlines())
                    except Exception as e:
                        failed_files.append((file_path, e))
                        out_f.write(f"# [ОШИБКА ЧТЕНИЯ: {e}]\n")
                except Exception as e:
                    failed_files.append((file_path, e))
                    out_f.write(f"# [ОШИБКА ЧТЕНИЯ: {e}]\n")

                out_f.write("\n\n")

            out_f.write(f"{SEPARATOR}\nИТОГОВАЯ СТАТИСТИКА\n{SEPARATOR}\n")
            out_f.write(f"Всего файлов: {len(files)}\n")
            out_f.write(f"Всего строк кода: {total_lines}\n")

            if failed_files:
                out_f.write(f"\nОшибки чтения ({len(failed_files)} файлов):\n")
                for fp, err in failed_files:
                    out_f.write(f"  - {fp.relative_to(project_root)}: {err}\n")

        print("✓ Сборка завершена успешно!")
        print(f"  Файлов обработано: {len(files)}")
        print(f"  Строк кода: {total_lines}")
        if failed_files:
            print(f"  Ошибок чтения: {len(failed_files)}")
        print(f"  Результат сохранён в: {output_file.resolve()}")

    except Exception as e:
        print(f"❌ Ошибка записи в файл {output_file}: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сбор всех .py файлов проекта в один текстовый файл"
    )
    parser.add_argument(
        'project_path',
        nargs='?',
        default='.',
        help='Путь к корню проекта (по умолчанию: текущая директория)'
    )
    parser.add_argument(
        '-o', '--output',
        default='project_code.txt',
        help='Имя выходного файла (по умолчанию: project_code.txt)'
    )
    parser.add_argument(
        '-e', '--exclude',
        action='append',
        default=[],
        help='Дополнительные шаблоны для исключения (можно указывать несколько раз)'
    )
    parser.add_argument(
        '--no-default-excludes',
        action='store_true',
        help='Отключить стандартные исключения (.git, __pycache__ и т.д.)'
    )

    args = parser.parse_args()

    project_root = Path(args.project_path).resolve()
    if not project_root.exists():
        print(f"❌ Путь не существует: {project_root}", file=sys.stderr)
        sys.exit(1)
    if not project_root.is_dir():
        print(f"❌ Указанный путь не является директорией: {project_root}", file=sys.stderr)
        sys.exit(1)

    exact_names, glob_patterns = _parse_excludes(args.exclude, not args.no_default_excludes)

    print(f"🔍 Поиск .py файлов в: {project_root}")
    if exact_names or glob_patterns:
        all_excludes = sorted(exact_names | glob_patterns)
        print(f"⛔ Исключения: {', '.join(all_excludes)}")

    python_files = collect_python_files(project_root, exact_names, glob_patterns)

    if not python_files:
        print("⚠ Внимание: не найдено ни одного .py файла!", file=sys.stderr)
        sys.exit(0)

    output_file = Path(args.output).resolve()
    write_files_to_txt(python_files, project_root, output_file)


if __name__ == '__main__':
    main()