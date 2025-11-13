# Sentinex: Project Workflow

This document outlines the standard workflow for contributing to the Sentinex project. All new features, bug fixes, and documentation changes should follow this process.

## The Core Philosophy

1.  **The `main` branch is sacred.** It should always be stable, functional, and "deployable." No one ever commits directly to `main`.
2.  **All work happens in branches.** Every new task (feature, bug fix, docs) gets its own branch.
3.  **Work is "atomic."** We create one branch for one *thing*. We don't mix "adding examples" with "fixing a bug" and "updating docs."
4.  **All work is reviewed.** All branches are merged into `main` via a Pull Request (PR). This is your own "code review" and a great way to track your work.

## The 5-Step Workflow

### Step 1: The Issue (The "Why")

* **What it is:** An "Issue" (on GitHub) is the starting point. It's the task to be done.
* **Action:** Before you write any code, create a new Issue.
* **Issue Title:** A clear, descriptive title.
    * **Good:** `Add a simple linear regression example`
    * **Good:** `Unit tests for nn.Dense layer are missing`
    * **Bad:** `code is broken`

### Step 2: The Branch (The "Where")

* **What it is:** A branch is a "copy" of the project where you can do your work in a safe sandbox without breaking `main`.
* **Action:** From your `main` branch, create a new branch.
    * `git checkout -b <branch-name>`
* **Branch Naming Convention:** `type/issue-number/description`
    * **`type`**:
        * `feature/`: For a new feature (e.g., adding examples, a new layer).
        * `fix/`: For a bug fix.
        * `docs/`: For updating the `README.md` or other documentation.
        * `chore/`: For "housekeeping" (e.g., linting, updating `.gitignore`).
    * **`issue-number`**: The number of the Issue you just created (e.g., `1-`).
    * **`description`**: A short, kebab-case (dash-separated) summary.

* **Examples:**
    * `feature/1/add-regression-example`
    * `fix/2/dense-layer-shape-error`
    * `docs/3/update-readme-install-guide`

### Step 3: The Work (The "How")

* **What it is:** Writing the code and committing your changes.
* **Action:** As you work, make small, "atomic" commits. A commit should be one logical change.
* **Commit Message Convention:** We use **Conventional Commits**.
    * **Format:** `type(scope): subject`
    * **`type`**:
        * `feat`: A new feature (e.g., `feat(examples): ...`)
        * `fix`: A bug fix (e.g., `fix(nn): ...`)
        * `docs`: Documentation only (e.g., `docs(readme): ...`)
        * `test`: Adding missing tests (e.g., `test(nn): ...`)
        * `chore`: Build process, setup (e.g., `chore(setup): ...`)
        * `refactor`: A code change that neither fixes a bug nor adds a feature.
    * **`(scope)`**: The part of the project you're changing (e.g., `(nn)`, `(examples)`, `(module)`).
    * **`subject`**: A short description of the change.

* **Examples:**
    * `git commit -m "feat(examples): Add simple regression script"`
    * `git commit -m "docs(readme): Update install-from-source instructions"`
    * `git commit -m "test(nn): Add unit test for Dense layer output shape"`

### Step 4: The Pull Request (The "Review")

* **What it is:** A "Pull Request" (PR) is a formal request to merge your branch (e.g., `feature/1/add-regression-example`) into the `main` branch.
* **Action:**
    1.  Push your branch: `git push -u origin feature/1/add-regression-example`
    2.  Go to GitHub and create a Pull Request.
* **PR Title:** Use the Conventional Commit format. This is the most important title!
    * **Example:** `feat(examples): Add simple linear regression and MNIST examples`
* **PR Description:**
    * Link the issue it solves. Write: `Closes #1`. (GitHub will auto-close the issue when the PR is merged).
    * Write a brief summary of *what* you did.

### Step 5: The Merge (The "Done")

* **What it is:** You (as the project owner) review the PR, see that it's good, and click the "Merge Pull Request" button.
* **Action:** Merge your PR. GitHub will then give you an option to delete the branch. Do it. Your branch's work is done and safely in `main`.