# Cell-Tracking

## Table of Contents

1. [Creators](#creators)
2. [Style Guide](#style-guide)


## Creators

[slovesjamie: Jamie Sloves](https://github.com/slovesjamie) \
[ZSR3004: Ziyad Rahman](https://github.com/ZSR3004) \
[CarolineTracy: Caroline Tracy](https://github.com/CarolineTracy) \
[Shital-Olee: Shital Oli](https://github.com/Shital-Olee)


## Style Guide

You can find our style guide [here](https://github.com/ZSR3004/Cell-Tracking/blob/main/docs/STYLEGUIDE.md). We wrote our style guide as a markdown file located in the `docs/` directory in this repository.


## User Guide (How to Upload Your Tiff Video for Desired Output)

### Introduction

In this user guide, there are step by step instructions on how to upload your Tiff file to our software and receive the desired output (kymograph, heatmap, vector field, sparse-like optical flow visualization, and raw data [xyz file, arrays]). This user guide is intended for the Mitchel lab and its members, as well as any other lab that would like to utilize this software. We will describe how to download our app, upload the video, choose the desired outputs, and save the results on your device/email the output to others.


### Equipment and Supplies

1. Computer with storage space available
2. Completed Tiff video
3. Internet access
   

### How To Use Our Software

This next section will go over the sub-tasks that the user can follow to run our software.

#### 1. Download the app

1. Open Google Chrome or a supporting internet browser.

2. Follow this [link](https://github.com/ZSR3004/Cell-Tracking/)
    - Note: After step 2, you should have arrived at the GitHub homepage (this page will contain all of the files you need to download.) You will see a screen like figure [FIGUREX], which displays the GitHub homepage.

3. Click on the newest release (which looks like figure [FIGUREX] below), and download the attached package by clicking the "Download" button as seen on figrue [FIGUREX].

    - Note: Software should now be downloaded and ready to run. You'll likely find it within your downloads folder.

#### 2. Upload video to our software

1. Open the downloaded application.

2. Run the installer by following the prompts within the installer window. Figure [FIGUREX] displays the first screen that initiates the installer window. Once completed, you should arricve at a screen that says "Finish". Click "Finish".

3. Open the application (that you downloaded in the previous step) by double clicking on the icon on your desktop or by searching for the application "Cell-Tracker" in your Finder app (for Mac) or File Explorer (for Windows 10 and Windows 11).

    - Note: You should have arrived at the main page of the application as shown in figure [FIGUREX].

4. On the main page, select "Upload Tiff Video." You'll now see a new window with a button saying "Upload file". You have two options to upload files.

  - Click the "Upload file" button. This will open your Finder or File Explorer. You can navigate to wherever your TIFF file is located, double-click it, and then press "Open" in the bottom right corner of this window. You should see a progress bar and the uploading will have begun.
  - If you already have the file open (say on your desktop or in Finder/File Explorer), you can select the file and drag it into this window. You will see a progress bar and uploading will have begun.

    - Note: If the file you attempt to upload is not a TIFF file (specficailly, a file with extension .tiff, .tif, .tiff.ome, or .tif.ome), then the program will tell you this is an invalid file type.

5. Once the upload is complete, you will see a screen like figure [FIGUREX] with the file name in the middle of the page. Confirm this file name matches the file you intended to upload and click "Next". If you accidentally uploaded the wrong file, click the "Remove" button on the same screen and try step 3 again.


#### 3. Select desired output
Now, you will need to tell Cell Tracker what type of video you have uploaded into the system. Namely, if it's a nuclei or cytoplasm labeled file. You will also tell the program what information or visualizations you would like the program to output. The current program output types are as follows:
  - Heatmaps
  - Kymographs
  - An array representing the optical flow
  - An XYZ file representing the optical flow

1. Select the dropdown menu. 

    - Note: A dropdown menu, as shown in figure [FIGUREX] should pop up, with a list of outputs.

2. Select the outputs you want by clicking the checkbox next to each one.

3. Click the "Next" button from the same page.

    - Note: A new screen, as shown in figure [FIGUREX], will pop up listing the outputs you want to receive. Check and make sure that everything you want is there.

4. From the same screen, click "Get results".


#### 4. Access the output from your computer

Our software has two different ways that the user can utilize to access the output: Save the output to your computer or through an email. First we will list the steps to download the output to your computer, then we will go over how to receive the output through an email. 

   - Note: At this point, ensure that the desired output is ready to be saved to your computer.

**Saving the output on your computer:**

1. Once you are satisfied with the output, click on the "Save" button. 

2. A file explorer window will open, as seen on figure [FIGUREX]. Navigate to the folder where you would like to save the result.

3. Enter a descriptive name for your output file. 

4. Like previously, from the explorer window, choose the file format that you like to save the video in. 

5. After naming and selecting the file format, click "Save" to finalize the process to save the output to your computer. 

**Receiving an email of the output:**

1. Once you have clicked the "save" button, click the "Email Output" button.

2. On the email space, as seen on figure [FIGUREX], enter the email addresses of the recepients and add a message (if desired).

3. Click "Send".


## Testing Our Software

### Introduction

Below, you will find steps on how to test our software using the various tests we wrote (such as unit tests). Users from the Mitchel lab will likely not need to do these steps. Please disregard these steps if you only intend to use our software (as opposed to testing it). Note that we will be using pytest to test our software. 


### How To Clone And Open Our Repository

Before you test our software, you must clone and open our GitHub repository. The steps below will explain how to do this.

1. At the top right of this page, click on the green "Code" button.
   
2. Make sure that "HTTPS" is selected.
   
   - If "HTTPS" isn't selected, select it now.
     
3. Copy the URL (which is located above the words "Clone using the web URL.").
   
4. Open a Command Line Interface (CLI), such as VSCode.
   
5. In your CLI's terminal, run ```git clone COPIED_URL``` (where COPIED_URL is the URL you just copied from our GitHub).
   
6. Now, open the repository that you just cloned.
    
   - First, follow the **How To Get Cell-Tracking's Path** steps below. The path you just copied will be referred to as CELL-TRACKING-PATH.
     
   - Run ```cd CELL-TRACKING-PATH```


### How To Test Our Software

1. Double-check that you are in the "Cell-Tracking" directory.
   
    - In the command line, run ```pwd``` (for Mac users) or ```cwd``` (for Windows users).
      
    - You should get a file path ending in ```/Cell-Tracking```. For example, you should get something that looks like this: ```/Users/carolinetracy/Desktop/Cell-Tracking```.
      
      - If you don't get a file path ending in ```/Cell-Tracking```, you must find the path for the Cell-Tracking folder. Follow the **How To Get Cell-Tracking's Path** steps below. The path you just copied will be referred to as CELL-TRACKING-PATH. Then, run ```cd CELL-TRACKING-PATH```
        
2. Run ```python3 -m venv venv```
   
   - Note: if you have python (not python3) installed, run ```python -m venv venv``` instead.
     
3. For Mac users: run ```source venv/bin/activate```. For Windows users: run ```venv\Scripts\activate```.
   
4. Run ```pip install pytest```
   
   - Note that this will take a second to install.
     
5. Find the file whose tests you want to run. This file will be referred to as TEST_FILE. (Make sure TEST_FILE includes ".py" at the end. For example, TEST_FILE could be tiffclass_test.py).
   
6. Figure out which folder TEST_FILE lies in. Note that this folder is a folder within the "tests" folder. This folder will be referred to as TEST_FOLDER.
   
   - You can do this by navigating to the "Cell-Tracking" folder on your computer and looking through the folders that are inside the "tests" folder.
     
7. Run ```pytest tests/TEST_FOLDER/TEST_FILE```

### How To Get Cell-Tracking's Path

Use your computer's search bar (the one that searches the contents of your computer), and search "Cell-Tracking". Click on the folder called "Cell-Tracking".

   - For Mac users: A window should have opened, and at the bottom of this window there should be a folder icon that says "Cell-Tracking". Right-click this folder icon. Then, click the button that says "Copy Cell-Tracking as Pathname".
     
   - For Windows users: Click the address bar (it should turn into Cell-Tracking's full path). Type CTRL+C to copy this path.


## Additional Resources: 

Contact Support: If you encounter any technical issues not addressed, please contact: 

[CarolineTracy: Caroline Tracy](https://github.com/CarolineTracy) 

[slovesjamie: Jamie Sloves](https://github.com/slovesjamie) 

[Shital-Olee: Shital Oli](https://github.com/Shital-Olee)

[ZSR3004: Ziyad Rahman](https://github.com/ZSR3004)


