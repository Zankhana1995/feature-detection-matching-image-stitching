from proj.ransac import *


# this function will blend images
def blend(image2, homography, minX, minY, stitchedImage):
    h = stitchedImage.shape[0]
    w = stitchedImage.shape[1]
    for i in range(h):
        for j in range(w):
            x1 = j - minX
            y1 = i - minY
            x2, y2 = project(x1, y1, homography)
            x2 = int(x2)
            y2 = int(y2)
            if x2 >= 0 and x2 < image2.shape[1] and y2 >= 0 and y2 < image2.shape[0]:
                if stitchedImage[i][j][0] == 0 or stitchedImage[i][j][1] == 0 or stitchedImage[i][j][2] == 0:
                    stitchedImage[i][j] = image2[y2][x2]
                else:
                    stitchedImage[i][j] = 0.5 * stitchedImage[i][j] + 0.5 * image2[y2][x2]
    print("blending done")
    return stitchedImage
