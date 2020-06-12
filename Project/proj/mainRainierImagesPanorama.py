from proj.featureDetector import *
from proj.stitching import *


# The use of inbuilt SIFT function
def findHarrisAndSift(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray, None)
    interestPoints = []
    for i in range(len(kp1)):
        x, y, d = int(kp1[i].pt[0]), int(kp1[i].pt[1]), des1[i]
        d = cv.normalize(d, None, norm_type=cv.NORM_L2)
        interestPoints.append((x, y, d))
    return kp1, interestPoints


# This function will improve matches with ratio test
def improveMatching(matches, ssd):
    newMatches = []
    for i in np.arange(0, len(matches)):
        if ssd[i][2] < 0.75:
            newMatches.append(matches[i])
    return newMatches


# this function will find stitch image by calling different functions
def findStitchedImage(image1, image2):
    kp1, interest1 = findHarrisAndSift(image1)
    kp2, interest2 = findHarrisAndSift(image2)
    matches, ssd = featureMatching(interest1, interest2)
    matches = improveMatching(matches, ssd)
    numOfMatches = len(matches)
    noOfIterations = 2000
    inliersThreshold = 2
    homography, homInv = ransac(matches, numOfMatches, noOfIterations, inliersThreshold, kp1,
                                kp2)
    # ransacMatches = findInliers(homography, matches, inliersThreshold, kp1, kp2)
    # ransacImage = drawMatchesFunction(image1, kp1, image2, kp2, ransacMatches)
    # cv.imshow("matches", ransacImage)
    # cv.waitKey(0)
    stitchedImage = stitch(image1, image2, homography, homInv)
    return stitchedImage


# this function will load Rainier Images
def loadRainierImages():
    print("For 6 Rainier Images")
    images = []
    image1 = cv.imread("images/Rainier1.png")
    images.append(image1)
    image2 = cv.imread("images/Rainier2.png")
    images.append(image2)
    image3 = cv.imread("images/Rainier3.png")
    images.append(image3)
    image4 = cv.imread("images/Rainier4.png")
    images.append(image4)
    image5 = cv.imread("images/Rainier5.png")
    images.append(image5)
    image6 = cv.imread("images/Rainier6.png")
    images.append(image6)
    # pass list of all images and the name of result Image
    findFinalImage(images, "finalSixImages")


# This function will use loaded image and find stitched image by calling appropriate functions
def findFinalImage(images, imageName):
    while len(images) != 1:
        stitchedImage = findStitchedImage(images[0], images[1])
        images.pop(0)
        images.pop(0)
        cv.imwrite("./results/" + imageName + ".png", stitchedImage)
        stitchedImage = cv.imread("results/" + imageName + ".png")
        images.insert(0, stitchedImage)


# the main function
def main():
    loadRainierImages()


if __name__ == '__main__':
    main()
