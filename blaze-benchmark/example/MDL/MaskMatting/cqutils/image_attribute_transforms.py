import skimage
from PIL import Image
import cv2
import random
import numpy as np

class Compose(object):
    def __init__(self, transforms, prob):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            for t in self.transforms:
                img = t(img)
        return img

class PIL2OpenCV(object):
    def __call__(self,img):
        return cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

class OpenCV2PIL(object):
    def __call__(self,img):
        return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

# input is a cv2 image
class RandomGaussianNoise(object):
    def __init__(self, noise_prob, meta_std):
        self.noise_prob = noise_prob
        self.meta_std = meta_std
    
    def __call__(self, img):
        mean = 0
        if random.random() < self.noise_prob:
            actual_std = random.random() * self.meta_std
            noise = np.random.normal(mean, actual_std, img.shape)
            out = img + noise
            out = np.clip(out, 0, 255)
            out = np.uint8(out)
            return out
        return img

# input is a cv2 image
class RandomGaussianBlur(object):
    def __init__(self, blur_prob):
        self.blur_prob = blur_prob

    def __call__(self, img):
        if random.random() < self.blur_prob:
            sigma = 0.2 + random.random() * 1.0
            #print(sigma)
            blurred_img = cv2.GaussianBlur(img, (3,3), sigma)
            return blurred_img
        return img

if __name__ == "__main__":
    randomGaussianBlur = RandomGaussianBlur(0.5)
    randomGaussianNoise = RandomGaussianNoise(0.5,10)
    pIL2OpenCV = PIL2OpenCV()
    openCV2PIL = OpenCV2PIL()

    filename = "/Users/jimmy.mjm/img_examples/选项示例/有品牌LOGO/TB25dXKvtcnBKNjSZR0XXcFqFXa_!!592879663.jpg"
    img = Image.open(filename)
    img = img.resize((224,224))
    img = pIL2OpenCV(img)
    img = randomGaussianNoise(img)
    img = randomGaussianBlur(img)
    img = openCV2PIL(img)
    img.show()
