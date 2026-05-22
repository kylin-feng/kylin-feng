# Project Archive Automation

This tool scans local project directories, creates cleaned per-project private GitHub repositories, and records progress in `archive-index.json`.

Run once:

```bash
export PATH="$HOME/.local/bin:$PATH"
python3 tools/project-archive/archive_projects.py --once
```

Dry run:

```bash
python3 tools/project-archive/archive_projects.py --dry-run
```

It intentionally skips dependency folders, build outputs, local secrets, runtime data, unreadable iCloud placeholders, and clean third-party clones.
