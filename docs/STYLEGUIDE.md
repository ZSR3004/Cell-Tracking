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
    - [9.1 Root Directory](#91-root-directory)
    - [9.2 Src](#92-src)
    - [9.3 Tests](#93-tests)
    - [9.4 Docs](#94-docs)
  - [10. Version Control and Git](#10-version-control-and-git)
    - [10.1 Branch Naming and Management](#101-branch-naming-and-management)
    - [10.2 Commits](#102-commits)
    - [10.3 Pull Requests](#103-pull-requests)
    - [10.4 Security](#104-security)

## 1. Introduction

## 2. Baseline Coding Standards

### 2.1 Overview of PEP 8

### 2.2 Overview of Google's Python Style Guide

### 2.3 Key Differences and When to Follow

### 2.4 Linting and Autoformatting

Use Python Black to autoformat our code. For linting, we will use Pylint with default settings.

## 3. Naming Conventions

### 3.1 Functions and Variables

Our function names will be verbs, and our variable names will be nouns.

The function and variable names will accurately and succinctly describe what the functions and variables do. 

We won't write a comment to explain a function or variable name. If we need to do that, we will change the name so it better explains what the function or variable is.

We won't include the container type in a name (for example, we won't have the name years_dict).

The length of a name should correspond to its scope. So, if something is only used for only a few lines, it's ok for it to be short. However, if something is used throughout the entire code base, it should be longer and much more descriptive.

We will try to start variable names with something general (such as "index", "year", etc.) and then get more specific. This allows us to quickly search for variables while coding.

### 3.2 Folders and Files

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
    Any preconditons here.
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

We should also be including comments at the top of every Python file to describe what the file does. For example,
```python
"""
File: file_name.py
Description: This file handles things that this file was made for.
"""
```

## 5. Importing and Dependency Management

### 5.1 Importing Packages or Modules

### 5.2 Managing Dependencies

## 6. Testing

### 6.1 Unit Testing

While doing unit testing, we will make sure that the unit tests cover all possible cases. We will test our software on a very large variety of things: from large to small videos, from very clear to very blurry videos, and from long to short videos. We will do all these kinds of tests on all the video types that the lab has. These types of videos are:

   1. The nuclei are dyed
   2. The cytoplasms are dyed
   3. Nothing in the cell is dyed

We will use Pytest to provide a framework for our unit tests.

## 7. Error Handling

We will use try: except for error handling within our code. When moving into the except branch, we will make sure to log what's happening.

## 8. Logging

## 9. Directory Organization

### 9.1 Root Directory

### 9.2 Src

### 9.3 Tests

### 9.4 Docs

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

If
you need to write a very detailed commit, then use bullet points (using a hyphen as the point) with a hanging indent. For instance, the example given in the
Github's best practices is:

```text
Refactor libvirt create calls

 - Minimize duplicated code for create

 - Make wait_for_destroy happen on shutdown instead of undefine

 - Allow for destruction of an instance while leaving the domain
```

[^10.2.1] Based on [this Github guide](https://gist.github.com/luismts/495d982e8c5b1a0ced4a57cf3d93cf60)

### 10.3 Pull Requests

### 10.4 Security
