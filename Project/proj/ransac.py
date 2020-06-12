import math
from random import randint
import cv2 as cv
import numpy as np


# This is a project function which will return projected points from given homography.
def project(x1, y1, hom):
    if hom is None:
        return
    image1Matrix = np.ones((3, 1), dtype=np.float)
    image1Matrix[0, 0] = x1
    image1Matrix[1, 0] = y1
    Image2Matrix = np.array(hom).dot(np.array(image1Matrix))
    u = Image2Matrix[0, 0]
    v = Image2Matrix[1, 0]
    w = Image2Matrix[2, 0]
    if w == 0:
        return 0, 0
    if np.isnan(u) or np.isnan(v):
        return 0, 0
    x2 = float(u / w)
    y2 = float(v / w)
    return x2, y2


# this is a main ransac function which will return best homography and inverse homography
def ransac(matches, numOfMatches, noOfIterations, inliersThreshold, keyPoints1, keyPoints2):
    noOfInliers = 0
    homography = None
    for i in range(0, noOfIterations):
        random = np.random.choice(numOfMatches, 4, replace=False)
        sourcePoint = np.float32([keyPoints1[matches[random[j]].queryIdx].pt for j in range(4)])
        targetPoint = np.float32([keyPoints2[matches[random[j]].trainIdx].pt for j in range(4)])
        hom, mask = cv.findHomography(sourcePoint, targetPoint, 0)
        if hom is None:
            continue
        inliersCount = computeInlierCount(hom, matches, numOfMatches, inliersThreshold, keyPoints1, keyPoints2)
        if inliersCount > noOfInliers:
            noOfInliers = inliersCount
            homography = hom

    bestMatches = findInliers(homography, matches, inliersThreshold, keyPoints1, keyPoints2)
    bestKP1 = np.float32([keyPoints1[bestMatches[p].queryIdx].pt for p in range(len(bestMatches))])
    bestKP2 = np.float32([keyPoints2[bestMatches[p].trainIdx].pt for p in range(len(bestMatches))])
    hom, mask = cv.findHomography(bestKP1, bestKP2, 0)
    homInv = cv.invert(hom)[1]
    return hom, homInv


# this function will calculate distance between points
def calculateDistance(x1, y1, x2, y2):
    return math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))


# This function will calculate inliers count from given parameters
def computeInlierCount(hom, matches, numOfMatches, inliersThreshold, keyPoints1, keyPoints2):
    count = 0
    if hom is None:
        return
    else:
        for i in range(0, numOfMatches):
            left = keyPoints1[matches[i].queryIdx]
            right = keyPoints2[matches[i].trainIdx]
            x1 = left.pt[0]
            y1 = left.pt[1]
            x2, y2 = project(x1, y1, hom)
            distance = calculateDistance(right.pt[0], right.pt[1], x2, y2)
            if distance < inliersThreshold:
                count += 1
    return count


# this function will find inliers from given homography and matches
def findInliers(homography, matches, inliersThreshold, keyPoints1, keyPoints2):
    ransacMatches = []
    noOfMatches = len(matches)
    for i in range(0, noOfMatches):
        firstImagePoints = keyPoints1[matches[i].queryIdx]
        secondImagePoints = keyPoints2[matches[i].trainIdx]
        x1 = firstImagePoints.pt[0]
        y1 = firstImagePoints.pt[1]
        x2, y2 = project(x1, y1, homography)
        distance = calculateDistance(secondImagePoints.pt[0], secondImagePoints.pt[1], x2, y2)
        if distance < inliersThreshold:
            ransacMatches.append(matches[i])
    return ransacMatches
