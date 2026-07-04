Chinese version: [`README_zh.md`](./README_zh.md)

# MagiCompiler Documentation Guide

This guide explains how to build, preview, and contribute to the MagiCompiler documentation. No prior Sphinx experience is required.

## Prerequisites

- Python 3.10+
- `pip` (comes with Python)
- A terminal (bash, zsh, PowerShell, etc.)
- A text editor (VS Code, Vim, etc.)

## Quick Start (5 minutes)

```bash
# 1. Enter the docs directory
cd docs

# 2. Install dependencies (one-time)
pip install -r requirements.txt

# 3. Build the docs
make html

# 4. Open in browser
open build/html/index.html        # macOS
xdg-open build/html/index.html    # Linux
start build/html/index.html       # Windows (Git Bash)
```

That's it! You should see the documentation in your browser.

> **Tip:** Run `make clean && make html` to force a full rebuild if you see stale content.

---

## Directory Structure

```
docs/
├── source/                  # All documentation source files live here
│   ├── index.md             # Homepage
│   ├── conf.py              # Sphinx configuration (rarely need to edit)
│   ├── _static/             # Custom CSS, images referenced by docs
│   ├── _templates/          # HTML templates (e.g. language switcher)
│   ├── user_guide/          # User-facing guides
│   │   ├── toc.md           # Table of contents for user guide section
│   │   ├── install.md       # Installation guide
│   │   └── quickstart.md    # Quick start guide
│   └── blog/                # Technical blog posts
│       ├── toc.md           # Table of contents for blog section
│       └── refs/            # BibTeX citation files (one per blog post)
├── locale/                  # Chinese translation files (.po)
│   └── zh_CN/LC_MESSAGES/
├── build/                   # Generated output (git-ignored)
├── requirements.txt         # Python dependencies
├── Makefile                 # Build commands (Linux/macOS)
└── make.bat                 # Build commands (Windows)
```

---

## Build Commands

| Command | Description |
|---------|-------------|
| `make html` | Build the English docs (default) |
| `make html-en` | Build the English docs into `build/html` (site root) |
| `make html-zh` | Build the Chinese docs into `build/html/zh_CN` |
| `make html-multilang` | Build both English and Chinese docs |
| `make update-po` | Regenerate `.po` translation templates |
| `make clean` | Remove the `build/` directory |

---

## How to Write Documentation

All documentation is written in **Markdown** using [MyST syntax](https://myst-parser.readthedocs.io/en/latest/), which extends standard Markdown with Sphinx-specific features.

### Adding a New User Guide Page

1. Create a new `.md` file in `source/user_guide/`.
2. Register its filename (without `.md`) in `source/user_guide/toc.md`.
3. Build and preview with `make clean && make html`.

### Adding a New Blog Post

1. Create `source/blog/my_topic.md` with a YAML frontmatter header
   (`blogpost: true`, `date`, `author`, ...).
2. If it has citations, create `source/blog/refs/my_topic.bib` and add the
   title to the `blog_titles` list in `source/conf.py`.
3. Register it in `source/blog/toc.md`.
4. Build and preview.

---

## Bilingual Docs (English + Chinese)

All documentation uses **English as the single source of truth**. Chinese
translations are managed via `.po` files following the standard Sphinx
internationalization (i18n) workflow.

```
source/*.md (English source, "single source of truth")
     │
     ▼  make update-po
locale/zh_CN/LC_MESSAGES/*.po (Chinese translations)
     │
     ▼  make html-multilang
build/html/        (English site, at the site root)
build/html/zh_CN/  (Chinese site, with a language switcher)
```

1. Edit the English `.md` sources under `source/` as usual.
2. Run `make update-po` to extract/refresh translation strings.
3. Translate the `msgstr` entries in `locale/zh_CN/LC_MESSAGES/*.po`
   (leave the `msgid` untouched).
4. Run `make html-multilang` to build both languages.
