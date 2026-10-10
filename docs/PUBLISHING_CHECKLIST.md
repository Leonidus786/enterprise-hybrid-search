# GitHub and GitHub Pages publishing checklist

The source project is ready to publish, but the repository and Pages URL depend on your GitHub account. This checklist does not push anything automatically; it keeps the target and credentials under your control.

## 1. Choose the repository URL

Create a **public** GitHub repository named `enterprise-hybrid-search`. If you are pushing an existing local repository, first check whether `origin` already points to the right place:

```bat
git remote -v
git status --short
git branch --show-current
```

If you are creating a new GitHub repository for an existing local project, create it empty (do not add a second README or `.gitignore` on GitHub). Do not replace an existing remote until you have checked what it points to.

## 2. Replace the username placeholder

Before publishing, search for `Leonidus786` in `README.md`, `docs/index.html`, and `docs/SOCIAL_POSTS.md`. Replace it with your exact GitHub username. In VS Code, use **Ctrl+Shift+H** with `Leonidus786` as the search text.

## 3. Copy the launch files into your existing project

Copy these files from the launch kit into the matching locations in your local `E:\enterprise-hybrid-search` project:

- `README.md`
- `.gitignore`
- `docs/index.html`
- `docs/SOCIAL_POSTS.md`
- `docs/PUBLISHING_CHECKLIST.md`
- `assets/enterprise-hybrid-search-social-card.png`

The full GitHub-ready ZIP also contains the project source and tests. If your local project has any changes that are not in that ZIP, **do not overwrite the whole folder**; copy only the files listed above.

## 4. Run checks before pushing

From your project root, run:

```bat
uv run python -m pytest -q
uv run python scripts/run_eval.py --fake
git diff --check
git status --short
```

Review `git status` and the staged diff before committing. Never commit `.env` files, credentials, private corpora, model caches, or customer data. The `.gitignore` prevents many common local files from being added, but it does not remove files already tracked by Git or erase secrets from Git history. If a secret was ever committed, rotate/revoke it and clean history before publishing.

## 5. Commit and push

Stage only the launch changes first:

```bat
git add README.md .gitignore docs/index.html docs/SOCIAL_POSTS.md docs/PUBLISHING_CHECKLIST.md assets/enterprise-hybrid-search-social-card.png
git diff --cached --check
git diff --cached --stat
git status --short
```

If the staged files look right, commit and push. If your branch is `main` and `origin` is already correct:

```bat
git commit -m "docs: prepare hybrid search project launch"
git push -u origin main
```

If no remote exists yet, add the actual URL first (replace the username):

```bat
git remote add origin https://github.com/Leonidus786/enterprise-hybrid-search.git
git push -u origin main
```

If your current branch is not `main`, use the branch shown by `git branch --show-current` and verify the target before pushing.

## 6. Publish the project overview page

After the repository files are pushed:

1. Open the repository on GitHub.
2. Go to **Settings → Pages**.
3. Under build and deployment, select **Deploy from a branch**.
4. Choose branch `main`, folder `/docs`, then save.
5. Wait for the Pages deployment to finish. GitHub will show the published URL.

The expected project-page URL is `https://Leonidus786.github.io/enterprise-hybrid-search/` after replacing the username. It is a static overview page with a small illustrative RRF ranking demo. **It is not a hosted endpoint for the Python retrieval engine.** A real search UI/API can be built and deployed as a later step.

## 7. Before sharing on LinkedIn or X

- Confirm the repository is public and the GitHub URL opens while signed out.
- Confirm the GitHub Pages URL loads.
- Attach `assets/enterprise-hybrid-search-social-card.png` to the post.
- Use the drafts in `docs/SOCIAL_POSTS.md` and replace the repository placeholder.
- Keep the caveat that fake evaluation is only a smoke test; do not present those numbers as real retrieval performance.
- The repository currently has no `LICENSE` file. Decide on a license before telling people the project is open source or granting reuse rights.
