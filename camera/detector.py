import cv2
import numpy as np
from numba import jit


class Detector:
    def __init__(self):
        self.blur_kernel_size = (3, 3)
        self.sobelX = np.array([[1, 0, -1], [2.4, 0, -2.4], [1, 0, -1]])
        self.sobelY = self.sobelX.T
        pass

    def convert2gray(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = cv2.addWeighted(img, 2.5, img, 0, 0)
        return out

    def blur(self, frame):
        img = cv2.GaussianBlur(frame, self.blur_kernel_size, 0)
        return cv2.bilateralFilter(img, 5, 75, 75)

    def sobel(self, frame):
        ximg = cv2.filter2D(src=frame, ddepth=cv2.CV_16S, kernel=self.sobelX)
        yimg = cv2.filter2D(src=frame, ddepth=cv2.CV_16S, kernel=self.sobelY)
        gxabs = cv2.convertScaleAbs(ximg)
        gyabs = cv2.convertScaleAbs(yimg)
        grad = cv2.addWeighted(gxabs, 0.5, gyabs, 0.5, 0)
        return grad

    def canny(self, frame):
        return cv2.Canny(image=frame, threshold1=50, threshold2=150)

    @jit
    def clean(self, img):
        for x in range(img.shape[1]):
            for y in range(img.shape[0]):
                if img[y, x] < 50:
                    img[y, x] = 255
                else:
                    break
        for x in range(img.shape[1]):
            for y in range(img.shape[0] - 1, 0, -1):
                if img[y, x] < 50:
                    img[y, x] = 255
                else:
                    break

        for x in range(img.shape[1]):
            for y in range(img.shape[0]):
                if img[y, x] < 255:
                    img[y, x] = 0
        x = img.shape[1]
        segment_left = img[:, 0:int(x / 4)].sum()
        segment_right = img[:, int(3 * x / 4):-1].sum()
        segment_center = img[:, int(x / 4):int(3 * x / 4)].sum()
        direction = np.argmax([segment_left, segment_center * 0.5, segment_right])
        return img, ['left', 'forward', 'right'][direction]

    def detect(self, frame):
        img = self.canny(self.blur(self.convert2gray(frame)))
        _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        img, direction = self.clean(img)
        img = cv2.merge([img, img, img])
        y, x, _ = img.shape
        arrow_point1 = (int(3 / 8 * x), y - 50)
        arrow_point2 = (int(5 / 8 * x), y - 50)
        if direction == 'left':
            img = cv2.arrowedLine(img, arrow_point2, arrow_point1, (0, 0, 255), 10)
        elif direction == 'right':
            img = cv2.arrowedLine(img, arrow_point1, arrow_point2, (0, 0, 255), 10)
        else:
            img = cv2.arrowedLine(img, (int(4 / 8 * x), y - 50), (int(4 / 8 * x), y - 150), (0, 0, 255), 10)
        return img
