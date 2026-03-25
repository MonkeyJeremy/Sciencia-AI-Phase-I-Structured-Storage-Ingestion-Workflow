# Contributing to Sciencia AI — Phase I

Thanks for contributing. This document covers the conventions used in this repository so that the codebase stays consistent as the team grows.

---

## Getting started

```bash
# 1. Clone the repo
git clone <repo-url>
cd <repo-name>

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Branching strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, production-ready code only |
| `dev` | Integration branch — merge feature branches here first |
| `feature/<short-description>` | New features or components |
| `fix/<short-description>` | Bug fixes |
| `chore/<short-description>` | Refactoring, docs, config changes |

Always branch off `dev`, never directly off `main`.

```bash
git checkout dev
git pull origin dev
git checkout -b feature/apple-scraper
```

---

## Commit messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <short summary>

[optional body]
```

**Types:** `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `perf`

**Examples:**

```
feat(scraper): add Apple App Store pipeline
fix(ingestion): handle null review_id without crashing
docs(readme): add database schema diagram
chore(deps): bump pandas to 2.1.0
```

Keep the subject line under 72 characters. Use the body to explain *why*, not *what*.

---

## Pull requests

1. Open PRs against `dev`, not `main`.
2. Give the PR a title that follows the commit convention above.
3. Fill in the PR description: what changed, why, and how to test it.
4. At least one reviewer must approve before merging.
5. Squash-merge into `dev`; rebase-merge into `main` for releases.

---

## Code style

- **Python version:** 3.11+
- **Formatter:** `black` (line length 100)
- **Linter:** `ruff`
- Follow existing naming patterns — `snake_case` for variables and functions, `UPPER_SNAKE` for module-level constants.
- Add a module-level docstring to every new `.py` file explaining its purpose, inputs, and outputs (see `db_ingestion.py` as a reference).
- Log meaningful progress at `INFO` level; use `DEBUG` for row-level detail.

---

## Adding a new scraper

New platform scrapers must:

1. Output a CSV with **at minimum** these columns:

   | Column | Type | Notes |
   |---|---|---|
   | `review_id` | string | Unique source-system ID |
   | `app_name` | string | Human-readable name |
   | `user` | string | Username or `anonymous` |
   | `star_rating` | int (1–5) | Null if unavailable |
   | `date` | YYYY-MM-DD | Review date |
   | `review` | string | Review text |
   | `review_length` | int | Character count |
   | `sentiment` | string | `positive`/`neutral`/`negative`/`unrated` |

2. Apply the same cleaning steps as `Google_scraper.py` (missing date, empty review, too short, non-English, duplicate ID).
3. Be compatible with `db_ingestion.py` — test with `--dry-run` before committing.

---

## Data & secrets

- **Never commit CSV files, `.db` files, or log files.** They are excluded in `.gitignore`.
- **Never commit API keys, credentials, or `.env` files.** Use `.env.example` to document required variables without values.
- Large datasets should be stored in the agreed shared data store and referenced by path, not checked in.

---

## Questions

Open an issue or ping the team in the project channel.
