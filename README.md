# Swayam - The Self Driving Car

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/soumik12345/Swayam-Self-Driving-Car/master)
[![HitCount](http://hits.dwyl.com/soumik12345/Swayam-Self-Driving-Car.svg)](http://hits.dwyl.com/soumik12345/Swayam-Self-Driving-Car)

### Overview
This is an implementation of various algorithms and techniques required to build a simple Self Driving Car. A modified versions of the <a href="https://github.com/udacity/self-driving-car-sim">Udacity Self Driving Car Simulator</a> is used as a testing environment.

<table style="width:100%">
  <tr>
    <td>
      <p align="center">
        <img src="./P1_Lane_Finder/Results/output_video.gif" alt="Overview" width="30%" height="30%">
        <br>Project 1: Lane Finding
        <br>Status: Completed<br>
        <br>
        <a href="./P1_Lane_Finder" name="p1_code">
          (code)
        </a>
      </p>
    </td>
    <td>
      <p align="center">
        <img src="./P2_Advanced_Lane_Finder/Results/real_world_footage.gif" alt="Overview" width="30%" height="30%">
        <br>Project 2: Advanced Lane Finding
        <br>Status: Completed<br>
        <br>
        <a href="./P2_Advanced_Lane_Finder" name="p2_code">   (code)
        </a>
      </p>
    </td>
  </tr>
  <tr>
    <td>
      <p align="center">
        <img src="./P3_Car_Finder/Images/car_cutout_1.jpg" alt="Overview" width="30%" height="30%">
        <br>Project 3: Deep Road Finder
        <br>Status: Ongoing<br>
        <br>
        <a href="./Deep_Road_Finder" name="p3_code">
          (code)
        </a>
      </p>
    </td>
  </tr>
</table>

### Dependencies

Main Dependencies are -
<ol>
  <li>Python 3.5</li>
  <li>numpy</li>
  <li>opencv</li>
  <li>pyautogui (optional)</li>
</ol>

### Project 1: Lane Finding
Detection of Lane Lines on road from both Real World Footage as well as Simulator Footage using basic Image Processing techniques including colorspace shifting, thresholding, edge detection and Line Detection.

<img src="./P1_Lane_Finder/Results/output_video.gif" alt="Overview" width="60%" height="60%">

### Project 2: Advanced Lane Finding
Detection of Lane Lines on road from both Real World Footage as well as Simulator Footage using advanced Image Processing techniques including Distortion Fixing, Perspective Transformation, Binarization and Polynomial Curve Fitting.

<img src="./P2_Advanced_Lane_Finder/Results/real_world_footage.gif" alt="Overview" width="60%" height="60%">
