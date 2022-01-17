from turtle import left
import cv2
import numpy as np
from scipy.spatial import distance


class preimg:
    def __init__(self) -> None:

        self.img = None
        self.draw_img = None
        self.imgdict ={}

    def setimg(self,pimg):
        self.img = pimg
        self.draw_img = pimg
        print("Hi")
        return 1

    def prepreprocess(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9,9), 0)
        return blur

    def calc_canny_cnts(self):

        image = self.prepreprocess()
        height, width = image.shape
        mpkt= (int(width/2),int(height/2)) # (x,y)
        Groesse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

        canny_img1= cv2.Canny(image=image, threshold1=8,threshold2=16)
        canny_img2= cv2.Canny(image=image, threshold1=8,threshold2=194)

        dil_img1 = cv2.dilate(canny_img1,Groesse,iterations = 6)
        dil_img2 = cv2.dilate(canny_img2,Groesse,iterations = 6)
        
        cnts1 = cv2.findContours(dil_img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts1 = cnts1[0] if len(cnts1) == 2 else cnts1[1]
        c1 = max(cnts1, key=cv2.contourArea)
        cnts2 = cv2.findContours(dil_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = cnts2[0] if len(cnts2) == 2 else cnts2[1]
        c2 = max(cnts2, key=cv2.contourArea)

        left1 = tuple(c1[c1[:, :, 0].argmin()][0])
        right1 = tuple(c1[c1[:, :, 0].argmax()][0])
        top1 = tuple(c1[c1[:, :, 1].argmin()][0])
        bottom1 = tuple(c1[c1[:, :, 1].argmax()][0])
        mpkt_conts1 = ( left1[0]+int((right1[0]-left1[0])/2),top1[1] + int((bottom1[1] - top1[1])/2))

        left2 = tuple(c2[c2[:, :, 0].argmin()][0])
        right2 = tuple(c2[c2[:, :, 0].argmax()][0])
        top2 = tuple(c2[c2[:, :, 1].argmin()][0])
        bottom2 = tuple(c2[c2[:, :, 1].argmax()][0])
        mpkt_conts2 = ( left2[0]+int((right2[0]-left2[0])/2),top2[1] + int((bottom2[1] - top2[1])/2))

        if 0.1*width > left1[0] or right1[0] > 0.9*width or 0.1*height > top1[1] or bottom1[1] > 0.9*height:
            if np.sum(np.abs(np.subtract(mpkt_conts1,mpkt))) < np.sum(np.abs(np.subtract(mpkt_conts2,mpkt))):
                self.imgdict = {
                    "cannyimg": canny_img1,
                    "image": image,
                    "height": height,
                    "width": width,
                    "mpkt": mpkt,
                    "cnts": c1,
                    "left": left1,
                    "right": right1,
                    "top": top1,
                    "bottom": bottom1,
                    "mpkt_conts": mpkt_conts2,
                    }
                return 1
            
        self.imgdict = {
            "cannyimg": canny_img2,
            "image": image,
            "height": height,
            "width": width,
            "mpkt": mpkt,
            "cnts": c2,
            "left": left2,
            "right": right2,
            "top": top2,
            "bottom": bottom2,
            "mpkt_conts": mpkt_conts2,
            }
        return 1
    
    def calc_rel_breitgroß(self):
        """Rechnet das Größe/ Breite Verhälnis aus"""

        left, right = self.imgdict["left"],self.imgdict["right"]
        top, bottom = self.imgdict["top"],self.imgdict["bottom"]

        d_breit = left[0] - right[0]
        d_hoch =  bottom[1] -top[1]

        return abs(d_breit/d_hoch)

    def calc_rel_spitze(self):

        left, right = self.imgdict["left"],self.imgdict["right"]
        top, bottom = self.imgdict["top"],self.imgdict["bottom"]
        mpkt_conts = self.imgdict["mpkt_conts"]
        cnts = self.imgdict["cnts"]

        #d_breit = distance.euclidean(left, right)
        #d_hoch = distance.euclidean(top, bottom)

        d_breit = left[0] - right[0]
        d_hoch =  bottom[1] - top[1]

        if d_breit < d_hoch:
            thresh_oben =  top[1] + 0.3*d_hoch
            thresh_unten = top[1] + 0.7*d_hoch
            sleftO = mpkt_conts[0]
            srightO = mpkt_conts[0]
            sleftU = mpkt_conts[0]
            srightU = mpkt_conts[0]
            for idx,y_wert in enumerate(cnts[:,0,1]):
                if y_wert < thresh_oben:
                    if cnts[idx,0,0] < sleftO:
                        sleftO = cnts[idx,0,0]
                        slpktO = cnts[idx][0]
                    if cnts[idx,0,0] > srightO:
                        srightO = cnts[idx,0,0]
                        srpktO = cnts[idx][0]
                if y_wert > thresh_unten:
                    if cnts[idx,0,0] < sleftU:
                        sleftU = cnts[idx,0,0]
                        slpktU = cnts[idx][0]
                    if cnts[idx,0,0] > srightU:
                        srightU = cnts[idx,0,0]
                        srpktU = cnts[idx][0]

            s_breitO = distance.euclidean(sleftO, srightO)
            s_breitU = distance.euclidean(sleftU, srightU)
            self.imgdict["spitze_pktO"] = [slpktO,srpktO]
            self.imgdict["spitze_pktU"] = [slpktU,srpktU]
            return abs(s_breitO/d_breit), abs(s_breitU/d_breit)

        else:
            thresh_links =  left[0] + 0.2*int(d_breit)
            thresh_rechts = left[0] + 0.8*int(d_breit)
            sobenO = mpkt_conts[1]
            suntenO = mpkt_conts[1]
            sobenU = mpkt_conts[1]
            suntenU = mpkt_conts[1]

            for idx,x_wert in enumerate(cnts[:,0,0]):
                if x_wert < thresh_links:
                    if cnts[idx,0,1] < sobenO:
                        sobenO= cnts[idx,0,1]
                    if cnts[idx,0,1] > suntenO:
                        suntenO = cnts[idx,0,1]
                if x_wert > thresh_rechts:
                    if cnts[idx,0,1] < sobenU:
                        sobenU = cnts[idx,0,1]
                    if cnts[idx,0,1] > suntenU:
                        suntenU = cnts[idx,0,1]

            s_hochO = distance.euclidean(sobenO, suntenO)
            s_hochU = distance.euclidean(sobenU, suntenU)
            return abs(s_hochO/d_hoch), abs(s_hochU/d_hoch)

    def calc_canny_lines(self):
        cannyimg = self.imgdict["cannyimg"]
        lines = cv2.HoughLinesP(cannyimg,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = 30,maxLineGap = 2)
        
        if lines is not None:
            return len(lines)
        return 0


    def calc_circles(self):
        img = self.img
        img = cv2.medianBlur(img,5)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(img,method= cv2.HOUGH_GRADIENT_ALT,dp = 1.5,minDist= 250,
                                    param1=140,param2=0.9,minRadius=20,maxRadius=200)
        
        if circles is not None:
            return len(circles)
        return 0
            
    def draw_image(self):

        cnts = self.imgdict["cnts"]
        left, right = self.imgdict["left"],self.imgdict["right"]
        top, bottom = self.imgdict["top"],self.imgdict["bottom"]

        slpktO,srpktO = self.imgdict["spitze_pktO"][0], self.imgdict["spitze_pktO"][1]
        slpktU,srpktU = self.imgdict["spitze_pktU"][0], self.imgdict["spitze_pktU"][1]


        cv2.drawContours(self.draw_img, [cnts], -1, (36, 255, 12), 2)
        cv2.circle(self.draw_img, left, 8, (0, 50, 255), -1)
        cv2.circle(self.draw_img, right, 8, (0, 255, 255), -1)
        cv2.circle(self.draw_img, top, 8, (255, 50, 0), -1)
        cv2.circle(self.draw_img, bottom, 8, (255, 255, 0), -1)

        cv2.circle(self.draw_img, slpktO, 4, (255, 234, 255), -1)
        cv2.circle(self.draw_img, srpktO, 4, (255, 234, 255), -1)
        cv2.circle(self.draw_img, slpktU, 4, (255, 234, 255), -1)
        cv2.circle(self.draw_img, srpktU, 4, (255, 234, 255), -1)

        return 1








