# Self-Driving-Car (Ongoing Project)

### Dependencies
<ol>
  <li>Python 3.5</li>
  <li>numpy</li>
  <li>opencv</li>
  <li>pyautogui (optional)</li>
</ol>

### Installation
This project is built and tested on Windows 10 in an Anaconda Virtual Environment. The yml files for setting up the Anaconda virtual environment can be downloaded <a href="superdatascience.com/wp-content/uploads/2017/09/Installations.zip">here</a>.<br>

To install the virtual environment, navigate to the folder for the yml file of your respective OS and execute the following command:<br>
<code>conda env create -f virtual_platform_windows.yml</code><br>Be sure to use the yml file for you respective OS.<br>

To clone the repository, use the following command:<br>
<code>git clone https://github.com/soumik12345/Self-Driving-Car</code>

### Run
First navigate to the project directory cloned from github.<br>

Activate the virtual environmennt using the command: <code>activate virtual_platform</code><br>
On Linux, use <code>source activate virtual_platform</code><br>

To run, use the command: <code>python bot.py</code><br>
On Linux, use <code>python3 bot.py</code>

### Overview
This is an implementation of a simple self driving car. The <a href="https://github.com/udacity/self-driving-car-sim">Udacity Self Driving Car Simulator</a> is used as a testing environment.
