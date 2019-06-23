import cv2
import numpy as np
from matplotlib import pyplot as plt
from numba import jit

class AdaptiveMedianBlur:
    @classmethod
    def remove_noise(cls, img: np.ndarray)->np.ndarray:
        width, height = img.shape[:2]
        img_ret = img.copy()

        for i in range(width):
            for j in range(height):
                img_ret[i, j] = cls().__find_not_pulse(img, (i, j))

        return img_ret


    @classmethod
    def __find_not_pulse(cls, img: np.ndarray, point: tuple, max_feature_size: tuple = (20, 20))->int:
        s_size:tuple = (3, 3)
        width, height = img.shape[:2]

        ret_value = img[point]
        while s_size[0] <= max_feature_size[0] and  s_size[1] <= max_feature_size[1]:
            left = int(max(point[0] - s_size[0] // 2, 0))
            right = int(min(point[0] + s_size[0] // 2 + 1, width))
            bottom = int(max(point[1] - s_size[1] // 2, 0))
            top = int(min(point[1] + s_size[1] // 2 + 1, height))

            s:np.ndarray = img[left: right, bottom: top]

            z_max, z_min, z_med = np.max(s), np.min(s), np.median(s)

            a1 = z_med - z_min
            a2 = z_med - z_max

            if a2 < 0 < a1:
                ret_value:int = cls().__check_point_pulse(img, point, z_max, z_min, z_med)
                break
            else:
                s_size = (s_size[0] + 1, s_size[1] + 1)

        return ret_value


    @classmethod
    def __check_point_pulse(cls, img: np.ndarray, point: tuple, z_max: int, z_min: int, z_med: np.float64)->int:
        if img[point] - z_max< 0 < img[point] - z_min:
            return img[point]
        else:
            return z_med


def add_noise(img: np.ndarray, noise_point_num: int = 10000)->np.ndarray:
    img_ret = img.copy()
    width, height = img.shape[:2]
    for _ in range(noise_point_num):
        i = np.random.randint(0, width)
        j = np.random.randint(0, height)

        img_ret[i, j] = np.random.choice([0, 255])

    return img_ret


def main():
    img = cv2.imread("D:/VOCdevkit/VOC2012/JPEGImages/2008_000112.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    width, height = img.shape[:2]
    img_noise = add_noise(img, int(width*height*0.25))

    img_noise_remove_plain = cv2.medianBlur(img_noise, 5)
    img_noise_remove = AdaptiveMedianBlur.remove_noise(img_noise)

    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(img, cmap="gray")
    plt.subplot(2, 2, 2)
    plt.title("Add Noise Pa=Pb=0.25")
    plt.imshow(img_noise, cmap="gray")
    plt.subplot(2, 2, 3)
    plt.title("MedianBlur kSize=5")
    plt.imshow(img_noise_remove_plain, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.title("AdaptiveMedianBlur")
    plt.imshow(img_noise_remove, cmap="gray")

    plt.show()


if __name__ == "__main__":
    main()