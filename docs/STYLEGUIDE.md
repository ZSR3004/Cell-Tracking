# Style Guide

## Table of Contents

1. Introduction
2. Baseline Coding Standards \
    2.1  Overview of PEP 8 \
    2.2 Overview of Google's Python Style Guide \
    2.3 Key Differences and When to Follow \
    2.4 Linting and Autoformatting
3. Naming Conventions \
    3.1 Functions and Variables \
    3.2 Folders and Files
4. Docstrings and Comments \
    4.1 Docstrings \
    4.2 Comments
5. Importing and Dependency Management \
    5.1 Importing Packages or Modules \
    5.2 Managing Dependencies
6. Testing \
    5.1 Unit Testing
7. Error Handling
8. Logging
9. Directory Organization
    9.1 Root Directory \
    9.2 Src \
    9.3 Tests \
    9.4 Docs
10. Version Control and Git
    10.1 Branch Naming \
    10.2 Commits \
    10.3 Pull Requests \
    10.4 Security

## Version Control and Git

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

### 10.3 Pull Requests

### 10.4 Security
