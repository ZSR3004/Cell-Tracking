# Cell-Tracking as a Command-Line Interface Instructions
One of the ways you can interface with the Cell-Tracking app is via a command line client. This guide serves to explain how to install, configure, and use the 
Cell-Tracking as a command-line interface (CLI).

## Table of Contents

 

## Requirements


## Installation

You'll be installing the binary code for this application.

1. Navigate to the [releases page for Cell-Tracking](https://github.com/ZSR3004/Cell-Tracking/releases) and find the latest release.
2. Install the binary file for your operating system (Mac, Windows, or Linux).
3. 


## Configuration File

After initializing the program for the first time, you'll find the Cell-Tracking folder at your specified location. In the top-level of that directory you'll find a 
file called `config.yaml`. This file will holds on the parameters and options for the program. The program generates a YAML file with default configurations. If any 
fields of the YAML file are empty, Cell-Tracking will use the default parameter instead. You can find the default configurations and explanations for what each
parameter does in the `example_conf.yaml` file found in the `docs/cli/` directory of the Cell-Tracking Github.

## Basic Arguments & Flags

The Cell-Tracking CLI takes a few arguments. We've conveniently laid these in a table with their shorthand and a very brief description of what they do. 

|   Command             |   Shorthand   |   Description                                                         |   Default Argument    | 
|   path                |   `p`         |   Specify the path to the tiff file.                                  |   None                | 
|   type                |   `t`         |   The type of the cells: nuclei labeled or cytoplasm/phase contrast.  |   Nuclei Labeled      | 
|   out                 |   `o`         |   What visualization to output: a heat-map, a kymograph, or nothing.  |   Nothing             | 
|   save-type           |   `s`         |   The output types: `.npy`, `.mat`, `.xyz`.                           |   All file types      |

To see more about each of these commands you can just type the following into the CLI,

```bash
cct help <command>
```

For example,

```bash
cct help type
```

## Interactive Mode

You can also use the CLI in interactive mode. Just type

```bash
cct interactive
```
Or,

```bash
cct i
```

Then, follow the prompts as they're given to you. They'll walk you through each step of the process from inputting the file path to selecting what outputs you want.
Just read the instructions and input the corresponding number at each step.

