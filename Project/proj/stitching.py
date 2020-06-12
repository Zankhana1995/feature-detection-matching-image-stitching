from proj.blend import *


# this function will find corner of image 2 from given homInv
def cornersOfImage2(image2, homInv):
    image2Corner = [0] * 4
    image1CornerProjected = [0] * 4
    image2Corner[0] = [0, 0]
    image2Corner[1] = [image2.shape[1], 0]
    image2Corner[2] = [image2.shape[1], image2.shape[0]]
    image2Corner[3] = [0, image2.shape[0]]
    for i in range(4):
        x1 = image2Corner[i][0]
        y1 = image2Corner[i][1]
        x2, y2 = project(x1, y1, homInv)
        image1CornerProjected[i] = [x2, y2]
    return image1CornerProjected, image2Corner


# This function will stitch 2 images from homography and inverse homography
def findMinMax(image1CornerProjected, minimumX, minimumY, maximumX, maximumY):
    for i in range(4):
        if image1CornerProjected[i][0] < minimumX:
            minimumX = image1CornerProjected[i][0]
        if image1CornerProjected[i][0] > maximumX:
            maximumX = image1CornerProjected[i][0]
    if minimumX < 0:
        minimumX = abs(minimumX)

    for j in range(4):
        if image1CornerProjected[j][1] < minimumY:
            minimumY = image1CornerProjected[j][1]
        if image1CornerProjected[j][1] > maximumY:
            maximumY = image1CornerProjected[j][1]
    if minimumY < 0:
        minimumY = abs(minimumY)
    return minimumX, minimumY, maximumX, maximumY


# this function will stitch image 1 and image 2 with given hom and homInv
def stitch(image1, image2, hom, homInv):
    image1CornerProjected, image2Corner = cornersOfImage2(image2, homInv)
    print("Inside Stitching images")
    minimumX = 0
    minimumY = 0
    maximumX = image1.shape[1]
    maximumY = image1.shape[0]
    minimumX, minimumY, maximumX, maximumY = findMinMax(image1CornerProjected, minimumX, minimumY, maximumX, maximumY)

    stitchedRow = int(maximumY) + int(minimumY)
    stitchedCol = int(maximumX) + int(minimumX)

    stitchedImage = np.zeros((stitchedRow, stitchedCol, 3))
    r, corner = int(minimumY), int(minimumX)
    h = image1.shape[0]
    w = image1.shape[1]
    r1 = r + h
    c1 = corner + w
    stitchedImage[r:r1, corner:c1, :] = image1

    stitchedImage = blend(image2, hom, minimumX, minimumY, stitchedImage)
    return stitchedImage
