# Style Guide

## Table of Contents

- [Style Guide](#style-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
  - [2. Baseline Coding Standards](#2-baseline-coding-standards)
    - [2.1 Overview of PEP 8](#21-overview-of-pep-8)
    - [2.2 Overview of Google's Python Style Guide](#22-overview-of-googles-python-style-guide)
    - [2.3 Key Differences and When to Follow](#23-key-differences-and-when-to-follow)
    - [2.4 Linting and Autoformatting](#24-linting-and-autoformatting)
    - [2.5 Markdown Formatting](#25-markdown-formatting)
  - [3. Naming Conventions](#3-naming-conventions)
    - [3.1 Functions and Variables](#31-functions-and-variables)
    - [3.2 Folders and Files](#32-folders-and-files)
  - [4. Docstrings and Comments](#4-docstrings-and-comments)
    - [4.1 Docstrings](#41-docstrings)
    - [4.2 Comments](#42-comments)
  - [5. Importing and Dependency Management](#5-importing-and-dependency-management)
    - [5.1 Importing Packages or Modules](#51-importing-packages-or-modules)
    - [5.2 Managing Dependencies](#52-managing-dependencies)
  - [6. Testing](#6-testing)
    - [6.1 Unit Testing](#61-unit-testing)
  - [7. Error Handling](#7-error-handling)
  - [8. Logging](#8-logging)
  - [9. Directory Organization](#9-directory-organization)
  - [10. Version Control and Git](#10-version-control-and-git)
    - [10.1 Branch Naming and Management](#101-branch-naming-and-management)
    - [10.2 Commits](#102-commits)
    - [10.3 Pull Requests](#103-pull-requests)
    - [10.4 Security](#104-security)

## 1. Introduction

Welcome to our team's style guide! You can find the Table of Contents right above the introduction to open any topic that you would like to reach. As a note, these are guidelines and not strict rules that needs to be followed for the project.

## 2. Baseline Coding Standards

### 2.1 Overview of PEP 8

[PEP 8](https://peps.python.org/pep-0008/) is the official style guide for Python coding. It outlines coding conventions, including things like indentations, layout, when to use comments and documentation, naming conventions, and more.

### 2.2 Overview of Google's Python Style Guide

[Google's Python Syle Guide](https://google.github.io/styleguide/pyguide.html) is how the company Google writes their code with Python. It addresses some issues in more depth than the PEP 8 style guide.

### 2.3 Key Differences and When to Follow

The main differences between the two style guides are that certain conventions are different. For example, PEP 8 recommends that coders use 4 spaces for indentation, but Google recommends that coders use 2 spaces for indentation.

As PEP 8 is the official style guide for Python code, we will look there for references. In general, we will use the PEP 8 style, but if something in PEP 8 is contradicted by what is in this style guide, use this style guide. If there is something that PEP 8 doesn't address, we will default to the Google style guide. This style guide serves as the first reference for how to write Python code.

### 2.4 Linting and Autoformatting

Use Python Black to autoformat our code. For linting, we will use Pylint with default settings.

### 2.5 Markdown Formatting

We will use the [VSCode markdownlint plugin](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint) by David Anson. You can
see the [`Rules.md` here](https://github.com/DavidAnson/markdownlint/blob/v0.38.0/doc/Rules.md). There will be no deviations

## 3. Naming Conventions

### 3.1 Functions and Variables

Our function names will be verbs, and our variable names will be nouns.

The function and variable names will accurately and succinctly describe what the functions and variables do.

We won't write a comment to explain a function or variable name. If we need to do that, we will change the name so it better explains what the function or variable is.

We won't include the container type in a name (for example, we won't have the name years_dict).

The length of a name should correspond to its scope. So, if something is only used for only a few lines, it's ok for it to be short. However, if something is used throughout the entire code base, it should be longer and much more descriptive.

We will try to start variable names with something general (such as "index", "year", etc.) and then get more specific. This allows us to quickly search for variables while coding.

### 3.2 Folders and Files

Folders (directories) should be all lowercase and short (ideally one word, avoiding underscore). Abbreviations or shorthand is okay. For example, the source code
directory can be named `src/` and utilities `utils/`.
File names should be in short and in snake case like variables. They should also be descriptive so we know what they do at a glance. If a file contains a single
class, then the file should be the analogous snake case of that class (ex. `ClassName` becomes `class_name.py`).

We also have test naming schemes (here)(#testing).

## 4. Docstrings and Comments

### 4.1 Docstrings

Docstrings should be included below all function declarations and follow the following pattern.

```python
"""
Short description of what the function does.

Args:
    first_arg (arg_type): Short description of what this argument represents.
    second_arg (arg_type): Short description of what this argument represents.

Returns:
    type: What the output is.

Preconditions:
    Any preconditions here.
"""
```

As a more concrete example, we can write something like the following.

```python
def my_function(x : int, xs : List[str]) -> bool:
"""
Checks if the string representation of x is in xs.

Args:
    x (int): The integer we are trying to find.
    xs (List[str]): A list of integers represented as type strings.

Returns:
    (bool): True if x is in xs, otherwise False.

Preconditions:
    xs is not empty.

"""
# function implementation
```

### 4.2 Comments

Comments should be used sparingly. We'll adhere to the "why, not what" convention. Only use a comment to explain why
you are doing something, don't use it to describe what is happening. If you find yourself needing to do the latter,
we might need to refactor the code to make it easier to read.

## 5. Importing and Dependency Management

### 5.1 Importing Packages or Modules

Use import statements for packages and modules only, not for individual types, classes, or functions.

- Use import x for importing packages and modules.
- Use from x import y where x is the package prefix and y is the module name with no prefix.
- Use from x import y as z in any of the following circumstances:
  - Two modules named y are to be imported.
  - y conflicts with a top-level name defined in the current module.
  - y conflicts with a common parameter name that is part of the public API (e.g., features).
  - y is an inconveniently long name.
  - y is too generic in the context of your code (e.g., from storage.file_system import options as fs_options).
- Use import y as z only when z is a standard abbreviation (e.g., import numpy as np).

Do not use relative names in imports. Even if the module is in the same package, use the full package name. This helps prevent unintentionally importing a package twice.

[^5.1.1] Source: Google Python Style Guide

### 5.2 Managing Dependencies

We'll be using pyproject.toml to keep track of modules. Any packages we create should have a subdirectory under the `src` directory. [Find the naming scheme here](#32-folders-and-files). Each of these packages should include a `__init__.py` file to denote that it is in fact a package.

We'll also be using pip to install said packages and Python virtual environments to manage packages. The `.venv` folder should be included in your
`.gitignore` (as in do not commit it).

## 6. Testing

### 6.1 Unit Testing

While doing unit testing, we will make sure that the unit tests cover all possible cases. We will test our software on a very large variety of things: from large to small videos, from very clear to very blurry videos, and from long to short videos. We will do all these kinds of tests on all the video types that the lab has. These types of videos are:

   1. The nuclei are dyed
   2. The cytoplasms are dyed
   3. Nothing in the cell is dyed

We will use Pytest to provide a framework for our unit tests.

## 7. Error Handling

We will use try: except for error handling within our code. When moving into the except branch, we will make sure to log what's happening.poetry install specific 

## 8. Logging

To understand what the software is doing, we'll use the Python logging module. The purpose of these logs are to give
clear, actionable changes. So, we shouldn't be using the logging module to log every iteration of a loop, but rather
to give us checkpoints during the execution of the program.

It is important to use the appropriate signals provided by the module.

- debug: Information to help programmers debug the program.
- info: High level information such as the programming starting or shutting down.
- warning: An error occurred, but the program can continue.
- error: An error occurred, and the program cannot continue (ie. it is going to crash).
- critical: An error occurred that corrupts the state of either the program or (in our case) the videos we are analyzing.

When writing an actual message, try to keep it short and use decorators. We'll be logging everything onto a rotating file
handler.

Here's an example of our setup.

```python
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    "app.log",
    maxBytes=1_000_000,
    backupCount=5
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[handler]
)

logger = logging.getLogger(__name__)

logger.info("Program starts with %d free GB.", gb_num)
```

## 9. Directory Organization

Our root directory is organized as follows.

```text
Cell-Tracking/
    |-- datasets/      # example videos or processed videos, mainly for testing
    |-- docs/          # documentation including style guide, user and contributor docs
    |-- scripts/       # helper scripts for development and debugging
    |-- src/           # main application code (frontend + backend)
    |-- tests/         # unit and integration test scripts
    |-- .env
    |-- .gitignore
    |-- pyproject.toml # project metadata, dependencies, and tool configurations
    |-- README.md      # project overview, usage, and contribution instructions
```

Here's a short explanation of each file or directory.

1. `datasets/`:
    - Stores `.tiff` video files or processed video arrays.
    - Mainly used for testing.
    - Large datasets should be kept out using `.gitignore` and smaller ones uploaded via git large file sharing.
2. `docs/`
    - Documentation, including style guide and contributing docs.
3. `scripts/`
    - Holds bash scripts to automate tasks.
4. `src/`
    - Holds runtime code.
    - Organized into packages. [See more information here](#5-importing-and-dependency-management).
5. `tests/`
    - Contains tests.
    - [More information here](#6-testing).
6. `.env`:
    - Holds environment variables.
    - Should **not** be committed to git.
7. `.gitignore`:
    - Specifies files to not include in commits.
    - Do not include `.venv`, `.env`, `datasets/`, or any auto-generated machine code.
8. `pyproject.toml`
    - Includes project information like dependencies.
    - Also has configs for linters, formatters, and other tools.
9. `README.md`
    - Landing page for the project.
    - Gives an overview of Cell-Tracking.

## 10. Version Control and Git

### 10.1 Branch Naming and Management

Conventions around branch naming are designed to keep the version control history clean and easy to read. A well named
branch should follow these basic rules[^10.1.1]:

1. All lowercase and hyphen separated: For example, it would be bad practice to write 'Feature/Foo Bar'. Instead, write
`feature/foo-bar`.
2. Alphanumeric characters only: Only use `A-Z`, `a-z`, `0-9`, and hyphens.
3. No double or continuous hyphens
4. Descriptive: Your branch name should be short, but give enough information so that anyone reading it knows exactly what the purpose of the branch is.

Branches should generally follow the 'prefix/descriptor' pattern. While the descriptor depends on what the branch is being made for, the prefixes are well defined below.[^10.1.2]

1. `feature`: For when you are creating a new feature.
2. `bugfix`: For when you are fixing an existing feature that is either causing the program to crash or otherwise
unoptimized. Note that this is only for pre-release branches.
3. `release`: To prepare the codebase for release and manage the release process.
4. `hotfix`: Fix issues that exist in the release branch.

Once we are done merging a branch with main, we'll delete it from the remote repository to keep the version history and branches clean.

[^10.1.1]: These basic rules are inspired by [this Medium article](https://medium.com/@abhay.pixolo/naming-conventions-for-git-branches-a-cheatsheet-8549feca2534) by Abhay Amin. \
[^10.1.2]: These prefixes were inspired by [this Graphite guide](https://graphite.dev/guides/git-branch-naming-conventions) by Greg Foster.

### 10.2 Commits

We'll keep commits simple. Our primary philosophy is that commits should **wrap related changes**. We'll mostly follow
Github's best practices for commits.[^10.2.1] To summarize them,

1. Commit Related Changes: Commits are a wrapper for related changes because it makes them easier to read and track.
2. Commit Often: This helps keeps commits small, which will allow us to rollback changes if something breaks.
3. Commit Working Code: Make sure to test your code before committing and don't commit half-written code.

On commit messages, we'll try to keep them below 50 characters, but if a commit requires it, we can write more. This is
not a hard and fast rule because readability is far more important. Commit messages should be descriptive, so we know
exactly what changed in a given commit.
For example, the commit message "Fixed bug" is not a good message. Instead, make it more descriptive like
"Fixed syntax error in heatmap_visualization".
Commit messages should also be imperative and in the past tense with proper capitalization. For example, "Created README.md" and not "creating README.md".

If you need to write a very detailed commit, then use bullet points (using a hyphen as the point) with a hanging indent. For instance, the example given in the Github's best practices is:

```text
Refactor libvirt create calls

 - Minimize duplicated code for create

 - Make wait_for_destroy happen on shutdown instead of undefine

 - Allow for destruction of an instance while leaving the domain
```

[^10.2.1] Based on [this Github guide](https://gist.github.com/luismts/495d982e8c5b1a0ced4a57cf3d93cf60)

### 10.3 Pull Requests

To be added. We will address this once discussed in class.

### 10.4 Security

To be added. We will address this once discussed in class.
