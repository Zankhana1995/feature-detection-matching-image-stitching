# feature-detection-and-matching (COMP 6341 Computer Vision)
Feature detection and feature matching with Python (Individual Assignments and Project)

## Assignment 1
The goal of this assignment is to gain an understanding of demosaicing: converting the Bayer pixel pattern into a RGB representation where each pixel has red, green and blue color channels.

## Assignment 2
The goal of this assignment is to write code to detect discriminating features in an image and find the best matching features in other images. Because features should be reasonably invariant to translation, rotation (plus illumination and scale if you do the extra credit).  
This involves three steps: feature detection, feature description, and feature matching. 

In the Project, you will apply your features to automatically stitch images into a panorama.

## Final Project (Image stitching)
In this project, the goal is to write software which stitches images together to form panoramas. The software will detect useful features in the images, find the best matching features in other images, align the photographs, then warp and blend the photos to create a seamless panorama.
This Project can be thought of as two major components:
1. Feature Detection and Matching [Assignment #2]
2. Panorama Mosaic Stitching [This Project]. 
All the step required are as follow and find details in project folder.
1. Compute the Harris corner detector.  
2. Matching the interest points between two images.  
3. Compute the homography between the images using RANSAC.  
4. Stitch the images together using the computed homography.   
