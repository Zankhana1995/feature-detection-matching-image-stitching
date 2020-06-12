from assign2.featureDetector import *


# the main method
def main():
    image1 = cv.imread("images/Yosemite1.jpg")
    image2 = cv.imread("images/Yosemite2.jpg")

    # make grayscale image
    gray1 = findCVTcolor(image1)
    gray2 = findCVTcolor(image2)

    offset = floorFunction(windowOfPixel)

    # find height and width of both image
    (h1, w1) = findShape(image1)
    (h2, w2) = findShape(image2)
    keyPoints1 = []
    keyPoints2 = []
    print("For Image 1")
    interestPoints1 = findHarrisCorner(gray1, offset, h1, w1)
    print(len(interestPoints1), " interest points after applying maximum harris features for Image 1")
    interestPoints1 = siftDescriptor(gray1, interestPoints1, h1, w1)

    for x, y, value in interestPoints1:
        keyPoints1.append(cv.KeyPoint(x, y, 1))
    print()
    print("For Image 2")
    interestPoints2 = findHarrisCorner(gray2, offset, h2, w2)
    print(len(interestPoints2), " interest points after applying maximum harris features for Image 2")
    interestPoints2 = siftDescriptor(gray2, interestPoints2, h2, w2)

    for x, y, value in interestPoints2:
        keyPoints2.append(cv.KeyPoint(x, y, 1))
    print()
    print(len(interestPoints1), " Key points for Matching (Image 1)")
    print(len(interestPoints2), " Key points for Matching (Image 2)")
    print()
    matches, ssdRatio = featureMatching(interestPoints1, interestPoints2)
    print(len(matches), "Matches Found!")
    newMatches = improvedMatches(matches, ssdRatio)
    print(len(newMatches), "improved matches found!")
    print()
    print("SSD Ratio")
    # print the ssd ratio
    printSSDRatio(ssdRatio)

    result = drawMatchesFunction(image1, keyPoints1, image2, keyPoints2, matches)
    improvedResult = drawMatchesFunction(image1, keyPoints1, image2, keyPoints2, newMatches)

    cv.imshow("Result", result)
    cv.waitKey(0)
    cv.imshow("Improved Result", improvedResult)
    cv.waitKey(0)


# entry point for the assignment 2
if __name__ == '__main__':
    main()
