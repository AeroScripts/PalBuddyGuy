# Pal Buddy Guy: The anipal's best friend
This is a small script to improve upon the tracking capabilities of the Vive Pro Eye and facial tracker. You can create custom expressions by making the expression and calibrating on that parameter.


# SYSTEM REQUIREMENTS
Currently this requires a CUDA-capable (nvidia) GPU with at least 4gb vram. It is possible to support AMD GPUs, but this will take some additional development work. Also, the current example script requires both the eye and face tracker. However, it would be simple to adapt it to work with only eye or only face.


# Installation
You must first replace the tvm_runtime and opencl DLLs inside SRanipal.
Copy the two .DLL files from the "tvm runtime" folder into "C:\Program Files\VIVE\SRanipal" replacing the existing files. You should back up your old files incase you want to revert later.

You then need to install Pytorch with gpu support. The easiest way to do so is using [anaconda](https://www.anaconda.com/products/individual).
To install the runtime with anaconda, launch anaconda by searching "Anaconda prompt" in the start menu. Once open, run the following commands:
```
conda install cudatoolkit cudnn pip
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install tqdm opencv-python numpy
```

Edit the .bat file to repository's directory for running scripts.</br>
Edit the path in script.py "DATASET_FOLDER" to output datasets. Datasets recording is 408mb so you need a decent amount of storage space free</br>

# Running

### Steps:
<ol>1. Start Anaconda Prompt and run script.py</ol>
<ol>2. Start another instance of Anaconda Prompt to run tvm_proxy.py</ol>
<ol>3. Start SRaniple</ol>
<ol>4. Launch VRC with https://github.com/benaclejames/VRCFaceTracking or Neos that can trigger facetracking to be active (Green, not orange)</ol>

### Script usage

** output swapping **
Before running any other commands, ensure the output window shows eye cameras on top, and face cameras below. If its reversed, run the comamand "swap" to swap them first.
This will be handled automatically in a later release.

** recording **
To run this, you must first record some "calibration" data for the expressions you want.
This *must* always include a "neutral" face recording. This is explained in more detail below.
When recording you sould try to make movements during the 20-30 seconds that you are calibrating, just make sure the target expression you are calibrating for is the most predominant (this also includes like adjusting your headset and stuff while making the expression)

the idea is to capture some diverse data where the primary consistent point is the target expression. Once you record one for each expression you want (both face and eyes are recorded at the same time) I can explain the next bit



** training **
Once you have recorded some datasets, edit script.py to include the filenames in the table at the top of the file.
Run the script, and enter the "train" command. Once it finishes, make sure to run "save" to save the results. 
Loss/Avg should be below 0.001 by the end. if not, something is wrong.

** inference **
Run the script and enter "infer". This is what you will run when actually using the parameters

# Tips
For neutral face recordings, this shouldn't nesisarily be truly neutral face, but any faces that you aren't trying to track. I keep it mostly neutral but also do some taking, and make sure to look around/blink with the eye tracker (unless one of your parameters is related to that)
This is basically to give the AI something to say "we aren't trying to look for this" so it doesnt have false positives.









