from proj.featureDetector import *
from proj.stitching import *
import cv2 as cv

# the main method
def findHarris(image):
    gray = findCVTcolor(image)
    offset = floorFunction(windowOfPixel)
    (h, w) = findShape(image)
    interestPoints = findHarrisCorner(gray, offset, h, w)
    return interestPoints


# convert to keyPoints
def convertToKeyPoints(interestPoints):
    keyPoints = []
    for x, y, value in interestPoints:
        keyPoints.append(cv.KeyPoint(x, y, 1))
    return keyPoints


# find matches from 2 images
def findMatches(image1, image2, interestPoints1, interestPoints2):
    (h1, w1) = findShape(image1)
    gray1 = findCVTcolor(image1)
    (h2, w2) = findShape(image2)
    gray2 = findCVTcolor(image2)
    interestPoints1 = siftDescriptor(gray1, interestPoints1, h1, w1)
    interestPoints2 = siftDescriptor(gray2, interestPoints2, h2, w2)
    keyPoints1 = convertToKeyPoints(interestPoints1)
    keyPoints2 = convertToKeyPoints(interestPoints2)
    matches, ssdRatio = featureMatching(interestPoints1, interestPoints2)
    threshold = 0.35 * math.sqrt((h1 ** 2) + (w1 ** 2))
    newMatches = improvedMatches(matches, ssdRatio, threshold, interestPoints1, interestPoints2)
    return keyPoints1, keyPoints2, newMatches


# main function
def main():
    image1 = cv.imread("images/Rainier1.png")
    image2 = cv.imread("images/Rainier2.png")

    # find harris corner on boxes image from Harris detector and SIFT descriptor created in assignment 2
    # from featureDetector.py file
    a1 = cv.imread("images/Boxes.png")
    keyPointsa1 = []
    interestPointsa1 = findHarris(a1)
    (h, w) = findShape(a1)
    graya1 = findCVTcolor(a1)
    interestPointsa1 = siftDescriptor(graya1, interestPointsa1, h, w)
    for x, y, value in interestPointsa1:
        keyPointsa1.append(cv.KeyPoint(x, y, 1))
    a1 = cv.drawKeypoints(a1, keyPointsa1, None)
    cv.imwrite("results/1a.png", a1)
    print("1a.png done!")

    # find harris corner on Rainier1 image from Harris detector and SIFT descriptor created in assignment 2
    # from featureDetector.py file
    kp1 = []
    interest1 = findHarris(image1)
    (h1, w1) = findShape(image1)
    gray1 = findCVTcolor(image1)
    interest1 = siftDescriptor(gray1, interest1, h1, w1)
    for x, y, value in interest1:
        kp1.append(cv.KeyPoint(x, y, 1))
    b1 = cv.drawKeypoints(image1, kp1, None)
    cv.imwrite("results/1b.png", b1)
    print("1b.png done!")

    # find harris corner on Rainier2 image from Harris detector and SIFT descriptor created in assignment 2
    # from featureDetector.py file
    kp2 = []
    interest2 = findHarris(image2)
    (h2, w2) = findShape(image2)
    gray2 = findCVTcolor(image2)
    interest2 = siftDescriptor(gray2, interest2, h2, w2)
    for x, y, value in interest2:
        kp2.append(cv.KeyPoint(x, y, 1))
    c1 = cv.drawKeypoints(image2, kp2, None)
    cv.imwrite("results/1c.png", c1)
    print("1c.png done!")

    matches, _ = featureMatching(interest1, interest2)

    # draw matches which are found from assignment 2 code from featureDetector.py
    matchedImage = drawMatchesFunction(image1, kp1, image2, kp2, matches)
    cv.imwrite("results/2.png", matchedImage)
    print("2.png done!")

    numOfMatches = len(matches)
    noOfIterations = 2000
    inliersThreshold = 2
    homography, homInv = ransac(matches, numOfMatches, noOfIterations, inliersThreshold, kp1,
                                kp2)
    ransacMatches = findInliers(homography, matches, inliersThreshold, kp1, kp2)
    ransacImage = drawMatchesFunction(image1, kp1, image2, kp2, ransacMatches)
    cv.imwrite("results/3.png", ransacImage)
    print("3.png done!")

    stitchedImage = stitch(image1, image2, homography, homInv)
    cv.imwrite("results/4.png", stitchedImage)
    print("4.png done!")


# entry point for the project part 1 and 2
if __name__ == '__main__':
    main()
