from proj.mainRainierImagesPanorama import *
import cv2 as cv


# This function will resize given image
def resizeImage(image):
    image = cv.resize(image, (600, 400))
    return image


# This function will load camera images and call other function to stitch images
def loadCameraImages():
    images = []
    print("For 4 Camera Images")
    image1 = cv.imread("images/camera1.jpg")
    image2 = cv.imread("images/camera2.jpg")
    image3 = cv.imread("images/camera3.jpg")
    image4 = cv.imread("images/camera4.jpg")
    image1 = resizeImage(image1)
    image2 = resizeImage(image2)
    image3 = resizeImage(image3)
    image4 = resizeImage(image4)
    images.append(image1)
    images.append(image2)
    images.append(image3)
    images.append(image4)
    findFinalImage(images, "finalCameraImages")


# the main function
def main():
    loadCameraImages()


if __name__ == '__main__':
    main()
