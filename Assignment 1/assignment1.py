import cv2 as cv
import numpy as np


# This function will split image in three channels
def split_function(mosaic_image):
    blue, green, red = cv.split(mosaic_image)
    return blue, green, red


# This function will return height and width of the image
def get_shape(mosaic_image):
    width = mosaic_image.shape[0]
    height = mosaic_image.shape[1]
    return width, height


# interpolation of blue channel
def get_blue_mask(shape):
    blue_channel_mask = np.zeros(shape, dtype=np.uint8)
    blue_channel_mask[:, ::2] = 1
    blue_channel_mask[1::2] = 0
    return blue_channel_mask


# interpolation of green channel
def get_green_mask(shape):
    green_channel_mask = np.zeros(shape, dtype=np.uint8)
    green_channel_mask[1::2, 1::2] = 1
    return green_channel_mask


# interpolation of red channel
def get_red_mask(shape):
    red_channel_mask = np.zeros(shape, dtype=np.uint8)
    red_channel_mask[:, 1::2] = 1
    red_channel_mask[1::2] = 0
    red_channel_mask[1::2, ::2] = 1
    return red_channel_mask


# interpolation of all channels
def get_mask(shape):
    blue_channel_mask = get_blue_mask(shape)
    green_channel_mask = get_green_mask(shape)
    red_channel_mask = get_red_mask(shape)
    return blue_channel_mask, green_channel_mask, red_channel_mask


# kernel (filter) for part 1 for each channel
def kernel_part1():
    green_filtered = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
    red_filtered = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4
    blue_filtered = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
    return blue_filtered, green_filtered, red_filtered


# converts the Bayer arrangement to RGB components for each pixel
def interpolation_channel(index, bayer_channel, shape):
    # Calculates the per-element bit-wise conjunction of two arrays or an array and a scalar.
    bitwise_channel = cv.bitwise_and(bayer_channel, bayer_channel, mask=get_mask(shape)[index])
    # filter2D
    interpolated_channel = cv.filter2D(bitwise_channel, -1, kernel_part1()[index])
    return interpolated_channel


# merge of channels
def merge_function(blue, green, red):
    merged_image = cv.merge((blue, green, red))
    return merged_image


# difference of images
def subtract_function(original_image, merged_image):
    difference_image = cv.subtract(original_image, merged_image)
    return difference_image


# main method
def main():
    original_image = cv.imread("pencils.jpg")
    mosaic_image = cv.imread("pencils_mosaic.bmp")

    # split the mosaic image onto 3 different color channel
    blue_bayer, green_bayer, red_bayer = split_function(mosaic_image)
    shape = get_shape(mosaic_image)

    # linear interpolation
    blue = interpolation_channel(0, blue_bayer, shape)
    green = interpolation_channel(1, green_bayer, shape)
    red = interpolation_channel(2, red_bayer, shape)
    final_img1 = merge_function(blue, green, red)

    # difference of original_image and demosaic_image (final_image1)
    diff_part1 = subtract_function(original_image, final_img1)

    image_stack_part1 = np.concatenate((original_image, final_img1, diff_part1), axis=1)
    cv.imshow("Part 1", image_stack_part1)
    cv.waitKey()

    # for part II
    g_minus_r = green_bayer - red_bayer
    b_minus_r = blue_bayer - red_bayer
    temp_g = cv.medianBlur(g_minus_r, 3)
    temp_b = cv.medianBlur(b_minus_r, 3)
    final_g = temp_g + red_bayer
    final_b = temp_b + red_bayer
    final_mosaic_2 = merge_function(final_b, final_g, red_bayer)

    # split the mosaic image onto 3 different color channel
    blue_bayer_2, green_bayer_2, red_bayer_2 = split_function(final_mosaic_2)
    shape2 = get_shape(final_mosaic_2)

    # linear interpolation
    blue_2 = interpolation_channel(0, blue_bayer_2, shape2)
    green_2 = interpolation_channel(1, green_bayer_2, shape2)
    red_2 = interpolation_channel(2, red_bayer_2, shape2)
    final_img3 = merge_function(blue_2, green_2, red_2)

    # difference of original_image and demosaic_image (final_image1)
    diff_part3 = subtract_function(original_image, final_img3)

    image_stack_part2 = np.concatenate((original_image, final_img3, diff_part3), axis=1)
    cv.imshow("Part 2", image_stack_part2)
    cv.waitKey()

    # # gaussian kernel_g for part II (second method)
    # kernel_g = np.zeros((7, 7), np.float32)
    # var_x = 2
    # var_y = 2
    # center_x = round(kernel_g.shape[0] / 2) - 1
    # center_y = round(kernel_g.shape[1] / 2) - 1
    #
    # for i in range(kernel_g.shape[0]):
    #     for j in range(kernel_g.shape[1]):
    #         kernel_g[i, j] = np.exp(
    #             -((i - center_x) ** 2 / (2 * (var_x ** 2)) + (j - center_y) ** 2 / (2 * (var_y ** 2))))
    #
    # kernel_g = kernel_g / np.sum(kernel_g)
    # temp_g1 = cv.filter2D(g_minus_r, -1, kernel_g)
    # temp_b1 = cv.filter2D(b_minus_r, -1, kernel_g)
    # final_gaussian_g = temp_g1 + red_bayer
    # final_gaussian_b = temp_b1 + red_bayer
    # final_image_gaussian = merge_function(final_gaussian_b, final_gaussian_g, red_bayer)
    # diff_part_gaussian = subtract_function(original_image, final_image_gaussian)
    #
    # image_stack_part_2_gaussian = np.concatenate((original_image, final_image_gaussian, diff_part_gaussian), axis=1)
    # cv.imshow("Part 2 with Gaussian", image_stack_part_2_gaussian)
    # cv.waitKey()
    cv.destroyAllWindows()


# entry point for the assignment
if __name__ == '__main__':
    main()
