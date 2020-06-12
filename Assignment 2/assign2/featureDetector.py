import math

import cv2 as cv
import numpy as np

maxHarrisFeatures = 500
harrisThreshold = 50000000
thresholdMatching = 0.3
windowOfPixel = 5
# for adaptive maximum suppression
cRobust = 0.95


# to select a fixed number of interest points from each image here 500 according to MOPS paper
def adaptiveMaximumSuppression(keypoints):
    finalKeyPoints = []
    i = 0

    for x, y, c in keypoints:
        i += 1
        minRadius = 999999999

        for refX, refY, refC in keypoints:
            if x == refX and y == refY:
                continue

            # below condition  ensures that a neighbour must have significantly higher
            # strength for suppression to take place
            if c < (cRobust * refC):
                distance = squareDistance(x, refX, y, refY)
                if distance < minRadius:
                    minRadius = distance

        finalKeyPoints.append((x, y, minRadius))

    totalKeyPoints = len(finalKeyPoints)
    print(str(totalKeyPoints) + " interest points found after non-maximum suppression!")
    # sort with min radius
    finalKeyPoints.sort(key=lambda a: a[2])
    return finalKeyPoints[:maxHarrisFeatures]


# to find local maximum in 3x3 neighborhood with non-maximum suppression
def findLocalMaximumWithNonMaxSuppression(intermediate, window):
    offset = floorFunction(window)
    h, w = intermediate.shape
    keyPoints = []
    for i in np.arange(offset, h - offset):
        for j in np.arange(offset, w - offset):

            neighbourhood = intermediate[i - offset:i + offset + 1, j - offset: j + offset + 1]
            x = np.where(neighbourhood == np.max(neighbourhood))
            val = neighbourhood[x][0]

            if val <= harrisThreshold:
                continue

            pointY, pointX = x
            pointY, pointX = pointY[0], pointX[0]

            pointX = j - offset + pointX
            pointY = i - offset + pointY

            intermediate[i - offset:i + offset + 1, j - offset: j + offset + 1] = 0
            intermediate[pointY, pointX] = val

            keyPoints.append((pointX, pointY, val))

    keyPoints = list(set(keyPoints))
    keyPoints = adaptiveMaximumSuppression(keyPoints)
    return keyPoints


# this method will find gradient and derivative of Ix and Iy
def findGradientAndGaussianSmooth(grayscale):
    i_x = cv.Sobel(grayscale, cv.CV_64F, 1, 0, ksize=5)
    i_y = cv.Sobel(grayscale, cv.CV_64F, 0, 1, ksize=5)

    i_x2 = i_x ** 2
    i_y2 = i_y ** 2
    i_xy = i_x * i_y

    i_x2 = findGaussianBlur(i_x2, (3, 3), 1)
    i_y2 = findGaussianBlur(i_y2, (3, 3), 1)
    i_xy = findGaussianBlur(i_xy, (3, 3), 1)
    return i_x2, i_y2, i_xy


# find harris corner from the image
def findHarrisCorner(grayscale, offset, h, w):
    noOfKeyPoints = 0
    intermediate = np.zeros((h, w), np.uint)
    i_x2, i_y2, i_xy = findGradientAndGaussianSmooth(grayscale)
    for y in np.arange(offset, h - offset):
        for x in np.arange(offset, w - offset):
            r_x2 = i_x2[y - offset:y + offset + 1, x - offset:x + offset + 1]
            r_xy = i_xy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            r_y2 = i_y2[y - offset:y + offset + 1, x - offset:x + offset + 1]

            s_x2 = r_x2.sum()
            s_xy = r_xy.sum()
            s_y2 = r_y2.sum()

            trace = s_x2 + s_y2
            det = s_x2 * s_y2 - s_xy ** 2

            if trace == 0:
                continue

            # corner response ( c or R)
            cornerStrength = math.floor(det / trace)

            if cornerStrength > harrisThreshold:
                noOfKeyPoints += 1
                intermediate[y, x] = cornerStrength

    print("A total of", noOfKeyPoints, " is the number of KeyPoints that are possible!")

    # find local maximum in 3 x 3 neighbourhood
    keyPoints = findLocalMaximumWithNonMaxSuppression(intermediate, 3)
    return keyPoints


# find angle, magnitude for given image
def calculateMagnitudeAngle(image):
    magnitude = np.zeros(image.shape, np.float)
    angle = np.zeros(image.shape, np.float)
    height, weight = image.shape[:2]
    for y in np.arange(1, height - 1):
        for x in np.arange(1, weight - 1):
            dx = int(image[y, x + 1]) - int(image[y, x - 1])
            dy = int(image[y + 1, x]) - int(image[y - 1, x])
            magnitude[y, x] = math.sqrt((dx ** 2) + (dy ** 2))
            angle[y, x] = math.atan2(dy, dx)

    return angle, magnitude


# rotation invariance for SIFT
def rotationInvariance(wAngle):
    tempAngle = np.array(wAngle)
    keyPointAngle = tempAngle[8, 8]
    for y in np.arange(0, 16):
        for x in np.arange(0, 16):
            if x == 8 and y == 8:
                continue
            angle = tempAngle[y, x]
            tempAngle[y, x] = angle - keyPointAngle
    return tempAngle


# This function will prepare histogram from given data
def prepareHistogram(magnitude, angle, histogram):
    for y in range(0, 4):
        for x in range(0, 4):
            a = math.degrees(angle[y, x])
            if a < 0:
                a = a + 360
            else:
                a = a
            index = int(math.floor(a / 45))
            histogram[index] += magnitude[y, x]

    # for i in histogram:
    #     if i >= 0.2:
    #         histogram = i
    #     else:
    #         histogram = 0.2
    histogram = [x if x < 0.2 else 0.2 for x in histogram]
    return histogram


# find SIFT descriptor of given image and interest points
def siftDescriptor(gray, keypoints, h, w):
    finalPoints = []
    iteratorList = [0, 4, 8, 12]
    image = cv.GaussianBlur(gray, (0, 0), 1.5)
    angle, magnitude = calculateMagnitudeAngle(image)

    for x, y, value in keypoints:
        if x - 8 < 0 or x + 8 > w or y - 8 < 0 or y + 8 > h:
            continue

        wMagnitude = magnitude[y - 8:y + 8, x - 8:x + 8]
        wMagnitude = cv.normalize(wMagnitude, None, norm_type=cv.NORM_L2)

        wAngle = angle[y - 8:y + 8, x - 8:x + 8]
        wAngle = rotationInvariance(wAngle)

        descList = []
        for i in iteratorList:
            for j in iteratorList:
                histogram = [0, 0, 0, 0, 0, 0, 0, 0]
                histogram = prepareHistogram(wMagnitude[i:i + 4, j:j + 4], wAngle[i:i + 4, j:j + 4], histogram)
                descList.append(histogram)

        descList = np.array(descList).reshape(-1)

        # normalize the descriptor to make it contrast invariant
        value = cv.normalize(descList, None, norm_type=cv.NORM_L2)
        finalPoints.append((x, y, value))
    return finalPoints


# This function will match feature between features from image 1 and image 2
def featureMatching(interestPoints1, interestPoints2):
    a = len(interestPoints1)
    b = len(interestPoints2)
    totalMatches = []
    ssdRatio = []
    for first in range(0, a):
        value1 = interestPoints1[first][2]
        secondBestIndex = 0
        bestDistance = 10
        secondBestDistance = 10

        for second in range(0, b):
            value2 = interestPoints2[second][2]
            d = value1 - value2
            d = d ** 2
            tempSum = d.sum()

            if tempSum < bestDistance:
                secondBestDistance = bestDistance
                bestDistance = tempSum
                secondBestIndex = second

        if bestDistance < thresholdMatching:
            ssdRatio.append((bestDistance, secondBestDistance, bestDistance / secondBestDistance))
            match = cv.DMatch(first, secondBestIndex, bestDistance)
            totalMatches.append(match)

    return totalMatches, ssdRatio


# this function will improve matches by Ratio Test
def improvedMatches(matches, ssdRatio):
    newMatches = []
    for i in np.arange(0, len(matches)):
        if ssdRatio[i][2] < 0.8:
            newMatches.append(matches[i])
    return newMatches


# calculate distance between given points
def squareDistance(x, x_r, y, y_r):
    a = (x_r - x) ** 2
    b = (y_r - y) ** 2
    return math.sqrt(a + b)


# print the ssd ratio
def printSSDRatio(ssd_ratio):
    for first, second, ratio in ssd_ratio:
        print("first, ", first, "second, ", second, "ratio", ratio)


# floor function
def floorFunction(window):
    padding = math.floor(window / 2)
    return padding


# find gray scale image
def findCVTcolor(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return gray


# return height and width of given image
def findShape(image):
    (height, width) = image.shape[:2]
    return height, width


# draw the matches found in both images
def drawMatchesFunction(image1, kp1, image2, kp2, matches):
    final = cv.drawMatches(image1, kp1, image2, kp2, matches, None)
    return final


# find gaussian blur from the image
def findGaussianBlur(i, matrix, depth):
    i_x2 = cv.GaussianBlur(i, matrix, depth)
    return i_x2
