import cv2

if __name__ == "__main__":
    img = cv2.imread("D:/VOCdevkit/VOC2012/JPEGImages/2008_000112.jpg")
    cv2.imshow("ImageRestoration",img)
    cv2.waitKey()