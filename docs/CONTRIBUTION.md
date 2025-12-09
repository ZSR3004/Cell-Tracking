# Contribution Guide

Cell-Tracking was designed for the Mitchel Lab. The intention is for the original software group to design a program that the Mitchel lab members can build upon, namely
adding new types of analysis or tweaking existing analysis. As such, it is important that these future developers are able to install our source code and make changes
that others can view and access.

This guide assumes you largely know how to program, but maybe haven't used Github before. Namely, you should be able to create functions in Python and run Python files,
but we'll step through the process of downloading code to your computer and uploading changes to the cloud (Github).

The primary maintainers for the project are Ziyad Rahman, Caroline Tracy, Jamie Sloves, and Shital Oli. These will likely be the ones to approve any pull requests or 
changes to repository.

## Table of Contents

- [Contribution Guide](#contribution-guide)
  - [Table of Contents](#table-of-contents)
  - [Downloading the Source Code and Making Changes](#downloading-the-source-code-and-making-changes)
    - [Prerequisite](#prerequisite)
    - [Dependencies](#dependencies)
      - [Cloning](#cloning)
  - [Github Issues](#github-issues)
    - [Writing Good Issues](#writing-good-issues)
    - [Closing Issues](#closing-issues)
  - [Pull Requests](#pull-requests)
  - [Testing the Software](#testing-the-software)
    - [How To Get Cell-Tracking's Path](#how-to-get-cell-trackings-path)


## Downloading the Source Code and Making Changes

### Prerequisite

We use our Github repository to track changes across devices. If you're familiar with the process, all you need to do is clone this repository, and you're ready to start
programming. If not, here are the basic dependencies (in other words programs you need to install). I highly recommend downloading all of these things via a terminal
package manager rather than off the internet. It just makes it easier to maintain and update everything.

### Dependencies

- A text editor like VSode (you should Google how to configure this for Python)
- Python 3.13 (If not yet installed, you can install it from the official Python website: python.org)
- Pip
- Git
- A Github account.
- A "True-Color" Terminal like PowerShell, Alacritty, or Kitty (You'll need this to see the colors the command line interface produces, but really its optional).

Follow any online instructions to setup Git and Github.

#### Cloning 

Now that you have everything setup, go to where you want your code to be. Now, just type in the following command which will make a copy of the code on your computer and
sync your code with the cloud.

```bash
git clone https://github.com/ZSR3004/Cell-Tracking.git
git remote add upstream <URL_of_the_original_repository>
```

You'll see a file called `gitignore-template`. This file will ignore machine code that becomes annoying to keep committing. To use this template, all you have to do
as use the following commands.

```bash
cd Cell-Tracking
mv gitignore-template .gitignore
```

Now, you can setup your development environment. We've used the library Poetry to make it easy to install all the dependencies. We recommend creating a Python virtual
environment. 

```bash
python3.13 -m venv .venv       # create a virtual environment with Python version 3.13
source .venv/bin/activate   # use the virtual environment
pip install poetry
poetry install
```

Now, you'll have all your dependencies installed and you can start programming!

## Github Issues

If you navigate to our issues page, at [https://github.com/ZSR3004/Cell-Tracking/issues](https://github.com/ZSR3004/Cell-Tracking/issues), you can find a list of issues.
Issues can be used to note any bugs or any features you want to be added. For developers, it is exactly that, it gives you an idea of what needs to be fixed or added.

### Writing Good Issues

Titles should be complete sentences and usually an imperative statement like "Create Tiff Class". You're welcome to keep them somewhat broad at first. Once people start
working on it, we should break them down into more tasks like "Write functions for reading Tiff files" and "Write Tiff pre-processing functions". Make sure to link these
tasks to the original issue so that we can keep track of what are sub tasks of what. The one caveat is regarding tests. Do not make a separate issue for tests, these 
should be included every time you make a new function.

For the description of each issue, you should outline what exactly needs to be done to consider this issue complete or not. For instance, if you are writing a class,
outline every function that you think should be included and the attributes the function should have. If the issue is about a function, detail what exactly the function
needs to do.

### Closing Issues

If an issue has been completed, you should have another person look at the issue and close it for you. When reviewing an issue, you should check that everything works,
everything that has been said needs to be completed has been completed, and that tests are cohesive and all-encompassing.

## Pull Requests

Pull requests should be approved by any of the core maintainers/admin. If you are a core maintainer, you should have a different maintainer look at your code. Make sure
that every new function has tests and that it works cohesively with what already exists.

## Testing the Software

1. Clone our repository using `Git`. Double-check that you are in the "Cell-Tracking" directory.
   
    - In the command line, run ```pwd``` (for Mac users) or ```cwd``` (for Windows users).
      
    - You should get a file path ending in ```/Cell-Tracking```. For example, you should get something that looks like this: ```/Users/carolinetracy/Desktop/Cell-Tracking```.
      
      - If you don't get a file path ending in ```/Cell-Tracking```, you must find the path for the Cell-Tracking folder. Follow the **How To Get Cell-Tracking's Path** steps below. The path you just copied will be referred to as CELL-TRACKING-PATH. Then, run ```cd CELL-TRACKING-PATH```
        
2. Run ```python3 -m venv venv```
   
   - Note: if you have python (not python3) installed, run ```python -m venv venv``` instead.
     
3. For Mac users: run ```source venv/bin/activate```. For Windows users: run ```venv\Scripts\activate```.
   
4. Run ```pip install poetry``` then ```poetry install```.
   
   - Note that this will take a second to install.
     
5. Find the file whose tests you want to run. This file will be referred to as TEST_FILE. (Make sure TEST_FILE includes ".py" at the end. For example, TEST_FILE could be tiffclass_test.py).
   
6. Figure out which folder TEST_FILE lies in. Note that this folder is a folder within the "tests" folder. This folder will be referred to as TEST_FOLDER. Make sure you are
   running the test from the main directory.
     
8. Run ```pytest TEST_FILE```
<br><br>

Here's an example work flow. We'll assume you're starting from scratch because you just want to check the tests.
```python
git clone https://github.com/ZSR3004/Cell-Tracking.git     # clone our repository
cd Cell-Tracking                                           # go into it
python3 -m venv .venv                                      # create a virtual environment
source .venv/bin/activate                                  # activate it. this may be different on Windows computers
pip install poetry                                         # install poetry
poetry install                                             # have poetry install all the packages
pytest                                                     # run all tests
```

If you're planning on contributing, we ask that you skim the [pytest documentation](https://docs.pytest.org/en/stable/index.html) so you know how to 
create tests and run them. If you're just interesetd in running the tests, there are plenty of tutorials online that may be easier to understand.

### How To Get Cell-Tracking's Path

Use your computer's search bar (the one that searches the contents of your computer), and search "Cell-Tracking". Click on the folder called "Cell-Tracking".

   - For Mac users: A window should have opened, and at the bottom of this window there should be a folder icon that says "Cell-Tracking". Right-click this folder icon. Then, click the button that says "Copy Cell-Tracking as Pathname".
     
   - For Windows users: Click the address bar (it should turn into Cell-Tracking's full path). Type CTRL+C to copy this path.
<br><br>
