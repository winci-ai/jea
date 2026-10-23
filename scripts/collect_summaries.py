#!/usr/bin/env python3
"""Collect SOURCE/model/task summaries into OUTPUT/model/task/summary.csv.

Examples (run from the repository root):
    python scripts/collect_summaries.py --source out --output summary
    python scripts/collect_summaries.py --source out --output summary --dry-run
    python scripts/collect_summaries.py --source out --check-only
    python scripts/collect_summaries.py --source out --check-only --models model_a \
        --expected-tasks segmentation_all optical_flow_all

An existing task-level summary is authoritative and copied unchanged. Otherwise,
collect the nearest summary in each subtree, stopping below each summary found,
and concatenate rows with a union of columns. No metrics are averaged. Merged
files include a source path relative to the task directory. Source files are
never modified. Only the Python standard library is required (Python >= 3.7).

Completeness is checked against the union of discovered *_all directories and
directories containing summaries, unless --expected-tasks is supplied. A result
must have a valid CSV header and at least one nonblank data row in every selected
summary. This checks task presence, not individual checkpoints, datasets or
metric values. Exit code 1 means missing/empty/invalid results or no summaries;
2 means invalid arguments. Valid results are still collected when others fail.
"""

import argparse
import csv
import os
from pathlib import Path
import shutil
import sys
import tempfile


def is_within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def find_summaries(task_dir):
    """Prefer each subtree's total over its per-run summaries."""
    found = []

    def on_error(error):
        raise error

    for directory, children, files in os.walk(str(task_dir), onerror=on_error):
        children[:] = sorted(
            name for name in children
            if not name.startswith('.') and not (Path(directory) / name).is_symlink()
        )
        if 'summary.csv' in files:
            path = Path(directory) / 'summary.csv'
            if path.is_symlink():
                raise ValueError('Refusing symbolic link: {}'.format(path))
            found.append(path)
            children[:] = []
    return sorted(found)


def read_table(path):
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, strict=True)
        columns = reader.fieldnames
        if not columns or any(not c.strip() for c in columns):
            raise ValueError('Missing or blank CSV header: {}'.format(path))
        if len(columns) != len(set(columns)):
            raise ValueError('Duplicate CSV columns: {}'.format(path))
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Invalid column count: {}:{}'.format(path, reader.line_num))
            rows.append(row)
    return columns, rows


def merge_tables(paths, task_dir):
    columns, records = [], []
    for path in paths:
        fields, rows = read_table(path)
        for field in fields:
            if field not in columns:
                columns.append(field)
        records.append((path.relative_to(task_dir).as_posix(), rows))

    # Preserve an existing source column by choosing an unused metadata name.
    source_column = 'source_summary'
    while source_column in columns:
        source_column = '_' + source_column
    merged = []
    for source, rows in records:
        for row in rows:
            row[source_column] = source
            merged.append(row)
    return [source_column] + columns, merged


def write_summary(destination, paths, task_dir):
    # Validate before creating output; use a temporary file to avoid partial CSVs.
    table = merge_tables(paths, task_dir) if len(paths) > 1 else None
    if table is None:
        read_table(paths[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix='.summary-', suffix='.tmp', dir=str(destination.parent))
    os.close(handle)
    temporary = Path(name)
    try:
        if table is None:
            shutil.copyfile(str(paths[0]), str(temporary))
        else:
            columns, rows = table
            with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def check_result(task_dir, paths):
    if not task_dir.is_dir():
        return 'MISSING_TASK', 'task directory does not exist'
    if not paths:
        return 'MISSING_SUMMARY', 'no summary.csv'
    problems = []
    invalid = False
    for path in paths:
        try:
            _, rows = read_table(path)
            if not any(any(value.strip() for value in row.values()) for row in rows):
                problems.append('{}: no data rows'.format(path.relative_to(task_dir)))
        except (OSError, ValueError, csv.Error) as error:
            invalid = True
            problems.append(str(error))
    if problems:
        return ('INVALID_SUMMARY' if invalid else 'EMPTY_SUMMARY'), '; '.join(problems)
    return 'OK', '{} summary file(s)'.format(len(paths))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=Path('out'), help='Root containing model directories (default: out)')
    parser.add_argument('--output', type=Path, default=Path('summary'), help='New output directory (default: summary)')
    parser.add_argument('--dry-run', action='store_true', help='Show selected files without writing')
    parser.add_argument('--check-only', action='store_true', help='Check completeness only; do not create or modify output')
    parser.add_argument('--models', nargs='+', help='Only process these model directory names; baseline still uses all models')
    parser.add_argument('--expected-tasks', nargs='+', help='Exact required task directory names; overrides automatic inference')
    args = parser.parse_args(argv)
    for name in (args.models or []) + (args.expected_tasks or []):
        if name in ('.', '..') or '/' in name or '\\' in name or ':' in name or not name.strip():
            parser.error('Use directory names, not paths: {}'.format(name))
    writing = not (args.dry_run or args.check_only)
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_dir():
        parser.error('Source directory does not exist: {}'.format(source))
    if not args.check_only and is_within(source, output):
        parser.error('Output must not equal or contain the source directory')
    # A fresh destination guarantees that tasks contain only one summary.csv.
    if writing and output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('Output must be absent or empty; choose a new --output directory')

    models, inferred = {}, set()
    for model in sorted(source.iterdir()):
        if not model.is_dir() or model.is_symlink() or model.name.startswith('.') or model.resolve() == output:
            continue
        if not args.check_only and is_within(output, model.resolve()):
            parser.error('Output may be under source, but not inside a model directory')
        models[model.name] = {}
        for task in sorted(model.iterdir()):
            if not task.is_dir() or task.is_symlink() or task.name.startswith('.'):
                continue
            paths = find_summaries(task)
            models[model.name][task.name] = paths
            if paths or task.name.endswith('_all'):
                inferred.add(task.name)

    selected = sorted(set(args.models) if args.models else models)
    unknown = set(selected) - set(models)
    if unknown:
        parser.error('Model directories not found: {}'.format(', '.join(sorted(unknown))))
    expected = sorted(set(args.expected_tasks) if args.expected_tasks else inferred)
    print('Expected tasks ({}): {}'.format(
        'explicit' if args.expected_tasks else 'inferred', ', '.join(expected) or '(none)'))
    if not args.expected_tasks:
        print('Tasks absent from every model cannot be inferred; use --expected-tasks for a fixed checklist.')

    plan, failures = [], 0
    for model_name in selected:
        tasks = models[model_name]
        # Explicit expectations control validation, not which existing results get copied.
        checked = sorted(set(expected) | {name for name, paths in tasks.items() if paths})
        statuses = {}
        for task_name in checked:
            task = source / model_name / task_name
            paths = tasks.get(task_name, [])
            status, detail = check_result(task, paths)
            statuses[task_name] = status
            if status != 'OK':
                failures += 1
                print('[{}] {}/{}: {}'.format(status, model_name, task_name, detail))
            elif paths:
                plan.append((task, output / model_name / task_name / 'summary.csv', paths))
        complete = sum(statuses[name] == 'OK' for name in expected)
        print('[CHECK] {}: {}/{} expected tasks have results'.format(model_name, complete, len(expected)))

    if args.check_only:
        print('Check only; no files written. {} task(s) with problems.'.format(failures))
        return 1 if failures or not plan else 0

    for task, destination, paths in plan:
        action = 'MERGE' if len(paths) > 1 else 'COPY'
        print('[{}] {} -> {}'.format(action, task.relative_to(source), destination))
        for path in paths:
            print('    {}'.format(path.relative_to(source)))
        if writing:
            write_summary(destination, paths, task)
    print('{} tasks collected/planned; {} task(s) with problems.{}'.format(
        len(plan), failures, ' Dry run; no files written.' if args.dry_run else ''))
    if not plan:
        print('No summaries found. --source must contain the model directories.', file=sys.stderr)
        return 1
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, csv.Error) as error:
        print('ERROR: {}'.format(error), file=sys.stderr)
        sys.exit(1)
