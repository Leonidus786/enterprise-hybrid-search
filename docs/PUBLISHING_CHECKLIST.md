# Enterprise Hybrid Search — Maintenance and Release Checklist

This checklist is for maintaining the existing public repository and GitHub Pages site. It does not create repositories, push commits, or publish posts automatically. Review every change locally before committing or pushing.

## Project links

- Repository: https://github.com/Leonidus786/enterprise-hybrid-search
- GitHub Pages: https://Leonidus786.github.io/enterprise-hybrid-search/
- Default branch: `main`
- Pages source: `main` branch, `/docs` folder
- Baseline tag: `v0.3-baseline` (local tag; push it separately if you choose to publish it)

The GitHub Pages site is a static project overview with a small illustrative RRF explainer. It is not a hosted Python search API or a live search engine.

## 1. Check the working tree before starting

From `E:\enterprise-hybrid-search`, run:

```bat
git status --short
git branch --show-current
git remote -v
git log -1 --oneline
git tag --list v0.3-baseline
```

Confirm that the `origin` remote points to the intended repository. Do not overwrite project files with a ZIP or another workspace without reviewing differences first.

## 2. Keep sample data separate from benchmarks

- Toy sample corpus: `data/sample/corpus.jsonl`
- Toy sample queries: `data/sample/queries.json`
- The sample corpus has 14 documents and 10 labeled queries. It is useful for smoke testing, not for general retrieval-quality claims.
- Keep official benchmark data and generated results in clearly named, ignored directories. Do not commit downloaded model caches, private data, credentials, or large generated files unless intentionally required and reviewed.
- For a standard document-level SciFact comparison, index one item per original corpus document. Do not silently apply the project's chunking pipeline to a benchmark intended to rank original documents.
- Preserve original query IDs, document IDs, relevance judgments, and any relevance grades when preparing benchmark data.

## 3. Required local checks before a code commit

Run from the project root:

```bat
uv run python -m pytest -q
uv run python scripts/run_eval.py --fake
git diff --check
git status --short
```

The full test suite must pass before the implementation is considered complete. During deliberate test-driven development, newly added tests may temporarily fail because their implementations have not been written yet; do not commit a partially implemented feature as complete.

Fake-model evaluation is a smoke test only. Do not use its metrics as evidence of search relevance or production performance.

For real-model or benchmark runs, save the dataset/split, configuration, model identifiers and revisions, code commit, environment, per-query rankings, metrics, and latency measurements. Clearly distinguish sample-data results from standardized benchmark results.

## 4. Review changes before committing

Review only the intended files:

```bat
git diff --check
git status --short
git --no-pager diff --stat
git --no-pager diff
```

Then stage specific files rather than blindly adding everything:

```bat
git add <intended-file-1> <intended-file-2>
git diff --cached --check
git diff --cached --stat
git status --short
```

Before committing, check for `.env` files, tokens, private corpora, model caches, generated result files, and unrelated edits. `.gitignore` does not remove files already tracked by Git and does not erase secrets from Git history. If a secret was committed, revoke or rotate it.

Commit a single logical change at a time, with a message that describes it. Do not push until you have reviewed the commit and confirmed the remote and branch.

## 5. Benchmark reporting requirements

For an external benchmark, record the exact protocol and configuration. Do not assume a difference from a published BM25 number automatically means there is a bug; first match corpus fields, preprocessing, tokenization/analyzer, BM25 parameters, ranking unit, query split, relevance judgments, and metric implementation as closely as possible.

Report cutoffs correctly. For example, Recall@100 can use the top 100 results, while Precision@10, MRR@10, and nDCG@10 must be computed using their top-10 protocol. Keep per-query results so paired comparisons can be analysed.

When comparing systems, report per-query paired metric differences and confidence intervals for those differences, along with effect size and win/loss counts. Record latency environment, warm-up policy, and number of measured runs. Avoid performance claims that the dataset or measurements cannot support.

## 6. Maintain the GitHub Pages site

- Keep `docs/index.html` claims consistent with the latest verified project state.
- If a real-model run has only been performed on the small sample corpus, say that representative benchmark validation is still pending.
- The RRF explainer uses made-up rankings. Keep it clearly labeled as illustrative and not connected to the Python retrieval pipeline.
- GitHub Pages publishes the `docs/` directory, so any image used by the page must also be inside `docs/`. For the social card, keep a copy at `docs/assets/enterprise-hybrid-search-social-card.png` and reference it from `docs/index.html` as `assets/enterprise-hybrid-search-social-card.png`.
- Check that the GitHub repository and Pages links use the correct username: `Leonidus786`. Do not replace it with a placeholder.
- After you push a reviewed commit yourself, wait for the Pages deployment to finish and verify the site, links, images, and mobile layout in a browser.

## 7. Before sharing a project update

- Confirm the repository is public and the repository link works while signed out.
- Confirm the GitHub Pages site loads and all images resolve.
- Link to reproducible benchmark results only after the protocol and data have been verified.
- State limitations plainly. Keep fake-model smoke-test results separate from real-model quality results.
- Do not claim that a hosted search API, production deployment, authentication, persistent vector database, or full browser search UI exists until it has actually been implemented and verified.
- No `LICENSE` file was present at the baseline checkpoint. Choose and add a license before claiming that others may freely reuse or redistribute the project.

## 8. Suggested project milestones

1. Finish and review the sample-data path and documentation cleanup.
2. Implement and test the BEIR adapter and embedding cache.
3. Add correct multi-cutoff evaluation and save reproducible benchmark outputs.
4. Run SciFact evaluation and analyse errors and paired metric differences.
5. Build and label an enterprise-domain evaluation set; tune on development data and report held-out test results.
6. Build the search API and a polished comparison UI after the evaluation contract is stable.
7. Add Docker, CI validation, setup documentation, and a live demo when each is verified.
