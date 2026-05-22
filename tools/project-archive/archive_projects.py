#!/usr/bin/env python3
"""Archive local projects into private per-project GitHub repositories."""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


OWNER = "kylin-feng"
CENTRAL_REPO = Path("/Users/shixianping/Projects/02-AI教育/秋招作品/kylin-feng")
INDEX_PATH = CENTRAL_REPO / "tools/project-archive/archive-index.json"
WORK_ROOT = Path("/Users/shixianping/Documents/Codex/2026-05-22/github/project-archive-work")
SCAN_ROOTS = [
    Path("/Users/shixianping/Projects"),
    Path("/Users/shixianping/Desktop"),
    Path("/Users/shixianping/Documents"),
    Path("/Users/shixianping/Downloads"),
]
SKIP_ROOTS = [
    Path("/Users/shixianping/Documents/Codex"),
    Path("/Users/shixianping/Projects/01-AI开发者/学习测试/flutter"),
]

MARKER_FILES = {
    "package.json",
    "pubspec.yaml",
    "requirements.txt",
    "pyproject.toml",
    "project.config.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "README.md",
}
MARKER_DIRS = {".git", "ios", "android", "lib", "src"}
PRUNE_NAMES = {
    ".Trash",
    ".cache",
    ".codex",
    ".dart_tool",
    ".git",
    ".gradle",
    ".hg",
    ".idea",
    ".next",
    ".pytest_cache",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "DerivedData",
    "dist",
    "Library",
    "node_modules",
    "Pods",
    "target",
    "venv",
}
RSYNC_EXCLUDES = [
    ".git/",
    ".DS_Store",
    ".cache/",
    ".dart_tool/",
    ".gradle/",
    ".idea/",
    ".next/",
    ".venv/",
    "__pycache__/",
    "build/",
    "DerivedData/",
    "dist/",
    "node_modules/",
    "Pods/",
    "target/",
    "venv/",
    "data/input/",
    "data/output/",
    "*.db",
    "*.key",
    "*.log",
    "*.p12",
    "*.pem",
    "*.pkl",
    "*.pyc",
    "*.sqlite",
    "*.sqlite3",
    ".env",
    ".env.*",
    ".streamlit/secrets.toml",
    "local.properties",
    "static/demo/*qr*",
    "static/demo/wechat-group-qr.*",
]
HIGH_RISK_SECRET_RE = re.compile(
    r"sk-[A-Za-z0-9][A-Za-z0-9_-]{10,}|"
    r"ghp_[A-Za-z0-9]+|"
    r"github_pat_[A-Za-z0-9_]+|"
    r"AKIA[0-9A-Z]{16}|"
    r"pat_[A-Za-z0-9]{20,}|"
    r"AKID[A-Za-z0-9]+|"
    r"security-token|"
    r"COZE_API_KEY=.*pat_",
    re.IGNORECASE,
)
REDACTIONS = [
    (re.compile(r"sk-[A-Za-z0-9][A-Za-z0-9_-]{10,}"), "YOUR_API_KEY"),
    (re.compile(r"ghp_[A-Za-z0-9]+"), "YOUR_GITHUB_TOKEN"),
    (re.compile(r"github_pat_[A-Za-z0-9_]+"), "YOUR_GITHUB_TOKEN"),
    (re.compile(r"pat_[A-Za-z0-9]{20,}"), "YOUR_COZE_PAT"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "YOUR_AWS_ACCESS_KEY_ID"),
    (re.compile(r"AKID[A-Za-z0-9]+"), "YOUR_ACCESS_KEY_ID"),
    (re.compile(r"X-Security-Token=[^\"&\n\r\s]+", re.IGNORECASE), "X-Security-Token=YOUR_SECURITY_TOKEN"),
    (re.compile(r"(?m)^(SECRET_KEY\s*=\s*)[\"'][^\"']+[\"']"), r'\1"YOUR_SECRET_KEY"'),
    (re.compile(r"(?m)^([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET)[A-Z0-9_]*\s*=\s*)[\"']?[^\"'\n#]+[\"']?"), r"\1YOUR_SECRET"),
]
TEXT_SUFFIXES = {
    ".bat",
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".css",
    ".dart",
    ".env",
    ".go",
    ".h",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".lock",
    ".m",
    ".md",
    ".mm",
    ".php",
    ".plist",
    ".properties",
    ".py",
    ".rb",
    ".rs",
    ".scss",
    ".sh",
    ".sol",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed\n{result.stdout}\n{result.stderr}")
    return result


def load_index() -> dict[str, Any]:
    with INDEX_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_index(index: dict[str, Any]) -> None:
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return slug or "project"


def central_backup_slugs() -> set[str]:
    backup_dir = CENTRAL_REPO / "project-backups"
    if not backup_dir.exists():
        return set()
    return {slugify(path.name) for path in backup_dir.iterdir() if path.is_dir()}


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
    except OSError:
        return False


def read_probe(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            fh.read(4096)
        return True
    except OSError:
        return False


def has_project_marker(path: Path) -> bool:
    try:
        names = {child.name for child in path.iterdir()}
    except OSError:
        return False
    if names & MARKER_FILES:
        return True
    if ".git" in names:
        return True
    if path.suffix in {".xcodeproj", ".xcworkspace"}:
        return True
    flutter_shape = {"pubspec.yaml", "lib"} <= names and ({"ios", "android"} & names)
    return flutter_shape


def discover_candidates() -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for current, dirnames, filenames in os.walk(root):
            path = Path(current)
            if is_under(path, CENTRAL_REPO) or is_under(path, WORK_ROOT) or any(is_under(path, skip) for skip in SKIP_ROOTS):
                dirnames[:] = []
                continue
            is_scan_root = path.resolve() in {root.resolve() for root in SCAN_ROOTS}
            depth = len(path.relative_to(root).parts)
            if depth > 7:
                dirnames[:] = []
                continue
            names = set(filenames) | set(dirnames)
            is_project = bool(names & MARKER_FILES) or ".git" in names
            is_project = is_project or ({"pubspec.yaml", "lib"} <= names and ({"ios", "android"} & names))
            if is_project and not is_scan_root:
                resolved = path.resolve()
                if resolved not in seen:
                    found.append(path)
                    seen.add(resolved)
                dirnames[:] = []
                continue
            dirnames[:] = [name for name in dirnames if name not in PRUNE_NAMES and not name.startswith(".Trash")]
    return found


def git_remote_and_dirty(path: Path) -> tuple[str, bool]:
    if not (path / ".git").exists():
        return "", False
    remote = run(["git", "remote", "-v"], cwd=path, check=False).stdout.strip()
    dirty = bool(run(["git", "status", "--short"], cwd=path, check=False).stdout.strip())
    return remote, dirty


def classify_candidate(path: Path, index: dict[str, Any], backup_slugs: set[str]) -> tuple[bool, str]:
    archived_paths = {entry.get("source_path") for entry in index.get("archived", [])}
    if any(path.resolve() == root.resolve() for root in SCAN_ROOTS):
        return False, "scan_root_not_project"
    if str(path) in archived_paths:
        return False, "already_archived_in_index"
    if slugify(path.name) in backup_slugs:
        return False, "already_archived_in_central_project_backups"
    if any(part in PRUNE_NAMES for part in path.parts):
        return False, "inside_ignored_directory"
    try:
        child_projects = [child for child in path.iterdir() if child.is_dir() and has_project_marker(child)]
        own_markers = {name for name in MARKER_FILES if (path / name).exists()}
        if child_projects and not (path / ".git").exists() and own_markers <= {"README.md"}:
            return False, "container_directory_with_child_projects"
    except OSError:
        return False, "unreadable_or_icloud_placeholder"
    marker = next((path / name for name in MARKER_FILES if (path / name).exists()), None)
    if marker and not read_probe(marker):
        return False, "unreadable_or_icloud_placeholder"
    remote, dirty = git_remote_and_dirty(path)
    if remote and not dirty:
        if "github.com:kylin-feng/" in remote or "github.com/kylin-feng/" in remote:
            return False, "already_has_kylin_feng_remote"
        return False, "clean_third_party_clone"
    return True, "candidate"


def copy_clean_source(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a"]
    for pattern in RSYNC_EXCLUDES:
        cmd += ["--exclude", pattern]
    cmd += [str(source) + "/", str(dest) + "/"]
    run(cmd)


def is_probably_text(path: Path) -> bool:
    if path.stat().st_size > 2_000_000:
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    return path.name in {".gitignore", ".npmrc", "Dockerfile", "Makefile"}


def redact_tree(root: Path) -> int:
    changed = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            if not is_probably_text(path):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            new_text = text
            for pattern, replacement in REDACTIONS:
                new_text = pattern.sub(replacement, new_text)
            if new_text != text:
                path.write_text(new_text, encoding="utf-8")
                changed += 1
        except OSError:
            continue
    return changed


def secret_hits(root: Path) -> list[str]:
    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            if not is_probably_text(path):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if HIGH_RISK_SECRET_RE.search(text):
            hits.append(str(path.relative_to(root)))
            if len(hits) >= 20:
                break
    return hits


def tree_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def unique_repo_slug(base: str) -> str:
    base = slugify(base)
    for idx in range(0, 100):
        slug = base if idx == 0 else f"{base}-{idx + 1}"
        exists = run(["gh", "repo", "view", f"{OWNER}/{slug}", "--json", "name"], check=False)
        if exists.returncode != 0:
            return slug
    raise RuntimeError(f"Could not find available repo slug for {base}")


def git_commit_and_create_repo(stage: Path, slug: str, source: Path) -> tuple[str, str]:
    run(["git", "init"], cwd=stage)
    run(["git", "config", "user.name", "Codex Project Archiver"], cwd=stage)
    run(["git", "config", "user.email", "codex-project-archiver@users.noreply.github.com"], cwd=stage)
    run(["git", "add", "."], cwd=stage)
    run(["git", "commit", "-m", f"Archive {source.name}"], cwd=stage)
    run(["gh", "repo", "create", f"{OWNER}/{slug}", "--private", "--source", str(stage), "--remote", "origin", "--push"], cwd=stage)
    sha = run(["git", "rev-parse", "HEAD"], cwd=stage).stdout.strip()
    url = f"https://github.com/{OWNER}/{slug}"
    return sha, url


def commit_central_index(message: str) -> None:
    run(["git", "config", "user.name", "Codex Project Archiver"], cwd=CENTRAL_REPO)
    run(["git", "config", "user.email", "codex-project-archiver@users.noreply.github.com"], cwd=CENTRAL_REPO)
    run(["git", "add", str(INDEX_PATH.relative_to(CENTRAL_REPO))], cwd=CENTRAL_REPO)
    if run(["git", "diff", "--cached", "--quiet"], cwd=CENTRAL_REPO, check=False).returncode == 0:
        return
    run(["git", "commit", "-m", message], cwd=CENTRAL_REPO)
    run(["git", "push"], cwd=CENTRAL_REPO)


def ensure_gh_ready(dry_run: bool) -> bool:
    if dry_run:
        return True
    gh = shutil.which("gh")
    if not gh:
        print("BLOCKED: gh is not on PATH. Add ~/.local/bin to PATH or install GitHub CLI.")
        return False
    auth = run(["gh", "auth", "status"], check=False)
    if auth.returncode != 0:
        print("BLOCKED: gh is installed but not logged in. Run: gh auth login --hostname github.com --git-protocol ssh --web")
        print(auth.stdout.strip() or auth.stderr.strip())
        return False
    ssh = run(["ssh", "-T", "git@github.com"], check=False)
    ok = "successfully authenticated" in (ssh.stdout + ssh.stderr)
    if not ok:
        print("BLOCKED: SSH authentication to git@github.com did not succeed.")
        print((ssh.stdout + ssh.stderr).strip())
    return ok


def archive_once(dry_run: bool, max_projects: int) -> int:
    index = load_index()
    run_record: dict[str, Any] = {"started_at": now_iso(), "dry_run": dry_run, "archived": [], "skipped": []}

    if not ensure_gh_ready(dry_run):
        run_record["status"] = "blocked_gh_auth"
        index.setdefault("runs", []).append(run_record)
        index["runs"] = index["runs"][-30:]
        if not dry_run:
            save_index(index)
            commit_central_index("Record project archive auth blocker")
        return 2

    backup_slugs = central_backup_slugs()
    candidates = discover_candidates()
    archived_count = 0
    for source in candidates:
        should_archive, reason = classify_candidate(source, index, backup_slugs)
        if not should_archive:
            run_record["skipped"].append({"source_path": str(source), "reason": reason})
            continue
        if archived_count >= max_projects:
            run_record["skipped"].append({"source_path": str(source), "reason": "deferred_to_next_run"})
            continue

        slug = slugify(source.name) if dry_run else unique_repo_slug(source.name)
        stage = WORK_ROOT / slug
        try:
            copy_clean_source(source, stage)
            redacted_files = redact_tree(stage)
            hits = secret_hits(stage)
            size = tree_size(stage)
            if hits:
                run_record["skipped"].append({"source_path": str(source), "reason": "unresolved_secret_hits", "hits": hits})
                shutil.rmtree(stage, ignore_errors=True)
                continue
            if size > 80_000_000:
                run_record["skipped"].append({"source_path": str(source), "reason": "too_large_after_cleanup", "bytes": size})
                shutil.rmtree(stage, ignore_errors=True)
                continue
            if dry_run:
                run_record["archived"].append({"source_path": str(source), "target_repo": f"{OWNER}/{slug}", "dry_run": True})
                archived_count += 1
                shutil.rmtree(stage, ignore_errors=True)
                continue
            sha, url = git_commit_and_create_repo(stage, slug, source)
            entry = {
                "source_path": str(source),
                "target_repo": f"{OWNER}/{slug}",
                "target_url": url,
                "commit_sha": sha,
                "archived_at": now_iso(),
                "redacted_files": redacted_files,
                "bytes": size,
            }
            index.setdefault("archived", []).append(entry)
            run_record["archived"].append(entry)
            archived_count += 1
        except Exception as exc:
            run_record["skipped"].append({"source_path": str(source), "reason": "archive_error", "error": str(exc)[:500]})
            shutil.rmtree(stage, ignore_errors=True)

    run_record["finished_at"] = now_iso()
    run_record["status"] = "complete_no_candidates" if archived_count == 0 else "archived_projects"
    index.setdefault("runs", []).append(run_record)
    index.setdefault("skipped", []).extend(run_record["skipped"])
    index["runs"] = index["runs"][-30:]
    index["skipped"] = index["skipped"][-500:]

    print(json.dumps(run_record, ensure_ascii=False, indent=2))
    if not dry_run:
        save_index(index)
        commit_central_index("Update project archive index")
    return 0 if archived_count > 0 else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one archive pass.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and stage decisions without creating repos.")
    parser.add_argument("--max-projects", type=int, default=int(os.environ.get("PROJECT_ARCHIVE_MAX_PROJECTS", "2")))
    args = parser.parse_args()
    if not args.once and not args.dry_run:
        parser.error("Use --once or --dry-run")
    return archive_once(dry_run=args.dry_run, max_projects=max(1, args.max_projects))


if __name__ == "__main__":
    sys.exit(main())
