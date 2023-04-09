import cv2
import numpy as np

def contour_cutter(img_b, img_p):
    # Read in an image
    img_p = cv2.imread('P.png', cv2.IMREAD_COLOR)
    # img_p = cv2.resize(img_p, (256 * 2, 256 * 2))
    img_b = cv2.imread('B.png', cv2.IMREAD_COLOR)
    # img_b = cv2.resize(img_b, (256 * 2, 256 * 2))
    g_img = cv2.imread('B.png', cv2.IMREAD_GRAYSCALE)
    # g_img = cv2.resize(g_img, (256 * 2, 256 * 2))

    # Set threshold value
    threshold_value = 50

    # Create threshold mask
    _, threshold_mask = cv2.threshold(g_img, threshold_value, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image for the mask
    mask = np.zeros_like(g_img)

    cv2.drawContours(mask, contours, -1, (255, 255, 255), 2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(g_img)

    areas = [cv2.contourArea(cnt) for cnt in contours]
    max_index = np.argmax(areas)

    cv2.drawContours(mask, contours, max_index, (255, 255, 255), -1)

    result_B = cv2.bitwise_and(img_b, img_b, mask=mask)
    result_P = cv2.bitwise_and(img_p, img_p, mask=mask)

    cv2.imwrite("B_out.png", result_B)
    cv2.imwrite("P_out.png", result_P)

contour_cutter(1, 2)