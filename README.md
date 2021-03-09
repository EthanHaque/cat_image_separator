# cat_image_separator

Python Tensorflow model that can separate out images of my cat from all my pictures.
## Purpose
I have several hundred images of my cat and several thousand more images of things that are not my cat. 
This project sorts those images out with near perfect accuracy (~98-99% accuracy)
## How it Works
I created and trained an optimized version of Inception ResNet V2 for classifying cats and 
wrote created a command-line utility for the project.
## How to Download it
Download the latest release, and the tensorflow model available here: [Shared Drive Link](https://drive.google.com/drive/folders/169dPptMA6AdbJPRpZTgIagdIugSIXwMG?usp=sharing).
<br>
Wherever you downloaded the code to, place the model in directory called `inceptionResNetV2_optimized_h5` and place that directory inside a directory named
`models`. You should end up with a path that looks like: `models/inceptionResNetV2_optimized_h5/inceptionResNetV2_optimized.h5` insde the directory that contains the source code.
<br>
<br>
Tensorflow 2.0 and numpy are the only dependencies you might have to install that don't come
standard with a python distro (i.e. argparse, os, and shutil).
## How to Use it
If you are using Anaconda, activate your conda environment that contains tensorflow and numpy. Otherwise, navigate to the 
directory that contains the main.py file and run 
<br>
* `python main.py -h` to see a list of the available functionality. 
<br>
<br>
[//]: # (end)
### Flags
* `-d` (or) `--directory`: performs predictions on the entire directory of images specified folder path
* `-dr` (or) `--directory_recursive`: performs predictions on all the sub-directories of the specified folder path
* `-i` (or) `--image`: performs predictions on a single image with the specified path
* `-r` (or) `--reorganize`: reorganizes the files into a directory with the positive class name and negative class name
* `-p` (or) `--positive`: the positive class name
* `-n` (or) `--negative`: the negative class name
* `-w` (or) `--write`: writes the output of the predictions to a specified file
* `-v` (or) `--verbose`: prints out the predictions to the command line
[//]: # (end)
### Examples
* `python main.py -v -i path/to/file`
* `python main.py -w path/to/output/file -d path/to/directory`
* `python main.py -d path/to/directory -r`
* `python main.py -dr path/to/root/directory -r -v -w path/to/output/file`




