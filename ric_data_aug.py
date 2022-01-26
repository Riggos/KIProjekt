
import cv2
import os
import numpy as np
from scipy.spatial import distance
import copy
import ricimg_to_table as ritt


class data_aug(ritt.imgorga):
    def __init__(self,location) -> None:
        super().__init__(location)
        self.img = None
        self.save_img = None
        self.imgdict ={}
        self.name = "_"

    def setimg(self,img_path):
        img = cv2.imread(img_path)
        resize = cv2.resize(img, (100, 100))
        self.img = copy.deepcopy(resize)
        self.save_img = copy.deepcopy(resize)
        return 1

    def gray(self):
        self.save_img = cv2.cvtColor(self.save_img, cv2.COLOR_BGR2GRAY)
        
    def blur(self):
        self.save_img = cv2.GaussianBlur(self.save_img, (5, 5), 0)
        return 1

    def rotate_img(self):
        angle = np.random.randint(0,360)
        h, w = self.img.shape[:2]
        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
        self.save_img = cv2.warpAffine(self.save_img, M, (w, h))
        return 1

    def flip_img(self):
        VH = np.random.randint(0,1)
        self.save_img = cv2.flip(self.save_img, VH)

    def add_noise(self, noise_type="gauss"):
        if noise_type == "gauss":
            mean=0
            st=0.5
            gauss = np.random.normal(mean,st,self.save_img.shape)
            gauss = gauss.astype('uint8')
            self.save_img = cv2.add(self.save_img,gauss)
        
        elif noise_type == "sp":
            prob = 0.05
            if len(self.save_img.shape) == 2:
                black = 0
                white = 255            
            else:
                colorspace = self.save_img.shape[2]
                if colorspace == 3:  # RGB
                    black = np.array([0, 0, 0], dtype='uint8')
                    white = np.array([255, 255, 255], dtype='uint8')
                else:  # RGBA
                    black = np.array([0, 0, 0, 255], dtype='uint8')
                    white = np.array([255, 255, 255, 255], dtype='uint8')
            probs = np.random.random(self.save_img.shape[:2])
            self.save_img[probs < (prob / 2)] = black
            self.save_img[probs > 1 - (prob / 2)] = white

    def do_random_aug(self):
        choice = np.random.randint(2, size=5)
        if choice[0]:
            self.blur()
            self.name += "B"
        else:
            self.add_noise()
            self.name += "N"
        if choice[1]:
            self.name += "U"
        else:
            self.name += "0"
        if choice[2]:
            self.flip_img()
            self.name += "F"
        else:
            self.name += "0"
        if choice[3]:
            self.rotate_img()
            self.name += "R"
        else:
            self.name += "0"
        if choice[4]:
            self.gray()
            self.name += "G"
        else:
            self.name += "0"

    def save_imgs(self,new_location):
        os.makedirs(new_location, exist_ok=True)
        for idx,label in enumerate(self.labels):
            path_with_label = os.path.join(new_location,label)
            os.makedirs(path_with_label, exist_ok=True)
            for img_path in self.image_paths[idx]:
                filename = os.path.basename(img_path)
                filetype = os.path.splitext(img_path)[1]
                self.setimg(img_path)
                self.do_random_aug()
                save1 = os.path.join(path_with_label, filename + filetype)
                str
                save2 = os.path.join(path_with_label, filename+ self.name + filetype)
                cv2.imwrite(save1,self.img)
                cv2.imwrite(save2,self.save_img)
                self.name = "_"
        return 1





