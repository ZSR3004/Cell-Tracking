# Using the Command Line Interface 

One of the ways you can run our Cell_Tracking software is through the Command Line Interface (CLI). The CLI guides you through selecting your input file, choosing output intems, and configuring how the software processes your data. This document will walk you through how to run the software using the CLI from start to finish. 

## Table of Contents
  1. Prerequisites
  
  2. Locating the Cell-Tracking Folder
  
  3. Launching the CLI
  
  4. Initializing the Cell-Tracking Folder
  
  5. Entering Folder and File Paths 
  
  6. Selecting Output items
  
  7. Specifying Video Type
  
  8. Combined Flow Option 
  
  9. Processing and Output 

### Prerequisites

Before using the CLI, ensure that you have the following installed:
1. Python 3 is installed 
2. A Command Line Interface, such as: 
    * Terminal (Mac)
    * Command Prompt or Powershell (Windows)
    * Integrated Terminal in VS Code

Note: Before moving on, please note that you should also have already cloned and opened the Cell-tracking GitHub repository, or have a folder with all of the code found in the repository. You can find the instruction to cloning a repository [here](http://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

If you are unsure whether Python is installed, type: 

```bash
python3 --version
```
If it installed, proceed to the next step, if it is not installed we recommend opening an internet browser and downloading the latest python version, 3.14.0 into your computer.


### Locating the Cell-Tracking Folder 

You will need the fill path to the Cell-Tracking folder. 

Mac Users: 
1. Open Spotlight Search 
2. Type Cell-Tracking 
3. Click the folder named Cell-Tracking 
4. A Finder window will open 
5. At the bottom of this window you should see a smaller folder icon with the name "Cell-Tracking'
6. Right click the folder icon
7. You will get a box with many options that you can click. Select "Copy Cell-Tracking as Pathname" (This copies the full path to your clipboard).
   
Windows Users: 
1. Use File Explorer to search for Cell-Tracking 
2. Click on the folder
3. Click inside the address bar, which should expand to show the entire path
4. Press CTRL + C to copy it (You will paste this path when running the CLI)

### Launching the CLI 

1. Open your terminal window 
2. Then navigate into the Cell-Tracking folder using the following command: 
```bash
cd CELL-TRACKING-PATH
```
Replace "CELL-TRACKING-PATH" with the path that you copied earlier. 

Example: 

```bash
cd Users/shitaloli/Documents/Cell-Tracking
```

3. Once inside the directory, run the CLI tool: 
```bash
python3 cli/main_cli.py
```
When you run this command inside the Cell-Tracking repository, it launches the main CLI scrip tthat controls the dull user-interactive workflow. 

### Initializing the Cell-Tracking Folder

The program will first ask: 
```bash
Have you intialized your Cell-Tracking? [y/n]:
```
  - Type "y" if you have run this program before and already created the ouput directory 
  
  - Type "n" if this is your first time. (The software will automatically create new folders for saving your outputs.)

Press "Enter" or "Return" on your keyboard after typing your response 

### Entering Folder and File Paths

1. Enter the parent folder where your TIFF file exists. 

You will see: 
```bash
Type the directory where your folder is saved "(type ~/folder_name or folder_name)":
```

Examples:
- Documents 
- Downloads 
- Desktop 
- Custom Folder 

1. Enter the full path to your TIFF file when it prompts for your file name 

```bash
Enter your file path name:
```
   Paste the full path to your TIFF file here (Make sure that you TIFF file is on the folder that you are working with).

Example: 

```bash
Enter your file path name:/Users/shitaloli/Downloads/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif
```

Note: If the file does not exist, you will see: 
```bash
Error: File 'FILE_PATH' does not exist. 
```
At this point, double check: 
    * File name (letterings)
    * File ends in .tif or .tiff
    * File is in the correct folder

Re-try the path until the program is able to run. 

### Selecting Output Items 

You will now be presented with an interactive selection menu: 
```bash
Select output items to generate: 
    "Optical Flow",
    "Heatmap",
    "Kymograph",
    "Raw Data"
```

Once you have chosen the outputs that you want to receive,
1. Use the arrows keys to move the highlight 
2. Press the space bar to select or de-select an item 
3. Press enter to confirm your selections 

After you are done, the menu will show: 
```bash
done (4 selections)
```
Note: The menu will indicate how many you have selected

### Specifying Video Type

The program will now ask:
```bash
Type in the type of video you input (n for nuclei dyed, p for phase contrast):
```

Type: 
    1. n for nucleus-dyes images
    2. p for phase-contrast or cytoplasm labeled videos 
    3. Once you have typed your response, press Enter or Return on your keyboard

    * This step determines which preprocessing and optical flow model will be used

### Combined Flow Option 

If your TIFF file contains multiple channels, the CLI will ask: 
```bash
Do you want to calculate the combined flows of channels one and two? [y/n]
```
Type: 
  - y if you want the software to compute optcial flow across both channels simultaneously 
  - n if you only want flow from one channel

### Processing and Input 

Once all the inputs are confirmed: 
1. The CLI will begin reading your TIFF file 
2. Preprocessing steps will run depending on the defaults of your video type
3. Optical flow computation will begin 
4. Output files will be saved automatically into the directed sub-folders created inside your Cell-Tracking directory

Note: Depending on the file size and selections, processing may take several minutes. 

Once complete, your files that you chose will be available on your folder. 

