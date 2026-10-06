import numpy as np
import cv2

import math

if __name__ == "__main__":
    import sys
    sys.path.append('.')

from src.organells.location import *
from src.container.spline import *
from src.container.subclass import *
from settings import PARAM, DEBUG_MODE, uniform_int, normal_randint

class Vesicles(Location):
    def __init__(self):
        super().__init__()
        self.type = "Vesicles"

        self.sizeVesiculeMin = 10
        self.sizeVesiculeMax = 25

        self.listSizeVesiculs = []
        self.listAngleVesiculs = []

        color = normal_randint(
            PARAM['vesicles_shell_color_mean'],
            PARAM['vesicles_shell_color_std'])
        self.color = (color, color, color)       
        self.nowPen = Pen(self.color, np.random.randint(PARAM['vesicles_border_w_min'], PARAM['vesicles_border_w_max']+1))
        backcolor = normal_randint(
            PARAM['vesicles_back_color_mean'],
            PARAM['vesicles_back_color_std'])
        #print(color, backcolor)
        self.addColor = (backcolor, backcolor, backcolor)

        self.nowBrush = Brush(self.addColor)

        # вероятность полностью заполненных везикул
        self.varibleInputFilling = 0

        # 0 - normal mode, 1 - many small vesicles
        self.typeGen = 0            

        self.Create()

    def __copy__(self):
        new_vesic = Vesicles()

        new_vesic.sizeVesiculeMin = self.sizeVesiculeMin
        new_vesic.sizeVesiculeMax = self.sizeVesiculeMax
        new_vesic.listSizeVesiculs = self.listSizeVesiculs
        new_vesic.listAngleVesiculs = self.listAngleVesiculs

        new_vesic.color = self.color
        new_vesic.nowPen  = self.nowPen.copy()
        new_vesic.addColor = self.addColor

        new_vesic.typeGen = self.typeGen
        new_vesic.varibleInputFilling = self.varibleInputFilling

        # copy location data
        new_vesic.centerPoint = self.centerPoint.copy()
        new_vesic.Points = self.Points.copy()
        new_vesic.angle = self.angle
        new_vesic.numberPoints = self.numberPoints

        new_vesic.setDrawParam()
        new_vesic.setRandomAngle(0,0)
        return new_vesic

    def copy(self):
        return self.__copy__()

    def Create(self):

        #self.typeGen = np.random.randint(0, 2)

        if self.typeGen == 0:
            mainVesiculesSize = np.random.randint(6,9)
            self.varibleInputFilling = np.random.random() * 0.05  
        else:
            mainVesiculesSize = np.random.randint(4,6+1)
            self.varibleInputFilling = np.random.random() * 0.4                            

        self.sizeVesiculeMin = mainVesiculesSize - 1
        self.sizeVesiculeMax = mainVesiculesSize + 1

        # radiuses of vesicules area (ellipse)
        radius_x = np.random.randint(70, 140)
        radius_y = np.random.randint(70, 140)

        square = radius_y * radius_x

        sqrt_int_square = int(round(square / mainVesiculesSize**2))
        print(f"{self.typeGen=}, {sqrt_int_square=}")
        if self.typeGen == 0:
            self.numberPoints = np.random.randint(sqrt_int_square//4, int(sqrt_int_square*1.25)//3+1)
        else:
            self.numberPoints = np.random.randint(2*sqrt_int_square//9, int((sqrt_int_square*1.25)//2+1))

        MAX_ITERATION = 800

        fail_counter = 0

        self.Points = []
        self.listAngleVesiculs=[]

        for i in range(self.numberPoints):
            counter = 0
            while True:
                now_point = self.getNewCoordVesicules(radius_x, radius_y)
                nowSizeCycle = np.random.randint(self.sizeVesiculeMin, self.sizeVesiculeMax+1)
                counter += 1
                #print(f"{nowSizeCycle=}, {self.sizeVesiculeMin=} {self.sizeVesiculeMax=}")
                if not self.CheckOverlap(self.Points, now_point, nowSizeCycle, 0.8):
                    # if we found the place for new vesicula
                    # добавление небольших искривлений, так как не всегда везикулы выглядят круглыми
                    if np.random.random() < 0.05:
                        self.listSizeVesiculs.append(np.random.randint(nowSizeCycle-2, nowSizeCycle+1, 2))
                    else:
                        self.listSizeVesiculs.append((nowSizeCycle, nowSizeCycle))
                    self.Points.append(now_point)
                    self.listAngleVesiculs.append(np.random.randint(0, 90))               
                    break
                
                if counter >= MAX_ITERATION:
                    fail_counter += 1
                    break

        if DEBUG_MODE:
            if (fail_counter != 0):
                print(f"The number of vesicles that could not be generated at a unique position: {fail_counter} out of {self.numberPoints}")

        self.numberPoints = len(self.Points)

        self.setRandomAngle(0, 90)

    def getNewCoordVesicules(self, radius_x, radius_y):

        #if self.typeForm == -1:
        #    now_point =  [np.random.randint(min_max_r_x[0], min_max_r_x[1]), np.random.randint(min_max_r_y[0], min_max_r_y[1])]

        #else:

        ## генерация эллипсом
        stop_gen = False
        while stop_gen == False:
            #x = np.random.randint(-radius_x, radius_x+1)
            #y = np.random.randint(-radius_y, radius_y+1)
            x, y = np.random.normal(loc=0.0, scale=0.2, size=2)

            x = int(round(x * radius_x))
            y = int(round(y * radius_y))

            if ((x**2) / (radius_x**2) + (y**2) / (radius_y**2)) <= 1:
                stop_gen = True
        now_point = [x,y]


        return now_point

    def CheckOverlap(self, Points, point, sizeVesicule, proportion):
        '''
        

        Parameters
        ----------
        Points : TYPE
            DESCRIPTION.
        point : TYPE
            DESCRIPTION.
        sizeVesicule : TYPE
            DESCRIPTION.
        proportion : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            True if new vesicula overlap old vesicules. False otherwise.

        '''
        for i in range(len(Points)):

            x_delt = abs(Points[i][0] - point[0])
            y_delt = abs(Points[i][1] - point[1])

            len_new_now = math.sqrt(x_delt**2 + y_delt**2)

            if len_new_now < max(self.listSizeVesiculs[i]) + sizeVesicule + self.nowPen.sizePen + proportion:
                return True

        return False

    def Draw(self, image, use_rand_color = False, minus_radius_val = 0):    #minus_radius_val для коррекции разметки, так как из-за размытия она страдает
        # Основная рисующая фукция
        draw_image = image.copy()

        for i in range(self.numberPoints):
            if use_rand_color:
                color = normal_randint(
                    PARAM['vesicles_shell_color_mean'],
                    PARAM['vesicles_shell_color_std'])
                self.color = (color, color, color)       
                backcolor = normal_randint(
                    PARAM['vesicles_back_color_mean'],
                    PARAM['vesicles_back_color_std'])
                self.addColor = (backcolor, backcolor, backcolor)
                #print("Colors:", color, "\n\n",backcolor)
                self.setDrawParam()

            radius_ves = (max(self.listSizeVesiculs[i][0], 3),
                          max(self.listSizeVesiculs[i][1], 3))           
            boarder_thickness = self.nowPen.sizePen + (radius_ves[0]+radius_ves[1])//2 - self.sizeVesiculeMin - minus_radius_val

            angle_ves = (self.listAngleVesiculs[i]+self.angle)%360

            if np.random.random() < self.varibleInputFilling:
                Brush(self.nowPen.color).FullBrushEllipse(draw_image, self.PointsWithOffset[i], radius_ves, angle_ves)
            else:
                self.nowBrush.FullBrushEllipse(draw_image, self.PointsWithOffset[i], radius_ves, angle_ves)

            cv2.ellipse(img = draw_image,
                       center = self.PointsWithOffset[i],
                       axes = radius_ves,
                       color = self.nowPen.color,
                       thickness = boarder_thickness,
                       angle = angle_ves,
                       startAngle = 0,
                       endAngle = 360)

        return draw_image

    def DrawLayer(self, image):
        return self.Draw(image, use_rand_color=True)

    def setDrawParam(self):
        self.nowPen.color = self.color
        self.nowBrush.brush = self.addColor

    def setMaskParam(self):
        self.nowPen.color = (255,255,255)
        self.nowBrush.brush = (255,255,255)

    def DrawMask(self, image):
        #Смена цветов для рисования маски
        self.setMaskParam()
        mask = self.Draw(image, minus_radius_val=1)
        self.setDrawParam()

        return mask

    def DrawUniqueArea(self, image, small_mode = False):
        # Функция создающая маску с немного большим отступом для алгорима случайного размещения новых органнел без пересечения
        ret_image = image.copy()
        ret_image = ret_image.astype(int)

        draw_image = np.zeros(image.shape, np.uint8)

        hull = cv2.convexHull(np.array(self.PointsWithOffset).astype(int))
        hull = np.squeeze(hull)

        #for point in hull:
        #    point[1] += 5

        draw_image = cv2.drawContours(draw_image, [hull], -1, (255,255,255), -1)

        if small_mode == False:
            kernel = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=np.uint8)

            draw_image = cv2.dilate(draw_image, kernel, iterations = 6)
        else:
            kernel = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=np.uint8)

            draw_image = cv2.dilate(draw_image, kernel, iterations = 3)


        ret_image = ret_image + draw_image
        ret_image[ret_image[:,:,:] > 255] = 255
        ret_image = ret_image.astype(np.uint8)

        return ret_image

def testVesicles():
    q = None

    print("Press button 'Q' or 'q' to exit")
    while q != ord('q') and q != ord("Q"):

        color = np.random.randint(182, 200)

        img = np.full((512,512,3), (color,color,color), np.uint8)

        for i in range(0, 512, 32):
            img[i:i+15,:,:] += 30

        vesicles = Vesicles()

        vesicles.NewPosition(256,256)

        print("centerPoint", vesicles.centerPoint)
        print("Points", vesicles.Points)
        print("PointsWithOffset", vesicles.PointsWithOffset)

        img1 = vesicles.Draw(img)

        mask = np.zeros((512,512,3), np.uint8)

        tecnicalMask = np.zeros((512,512,3), np.uint8)

        mask = vesicles.DrawMask(mask)
        img2 = vesicles.Draw(img)
        tecnicalMask = vesicles.DrawUniqueArea(tecnicalMask)

        smalltecnicalMask = np.zeros((512,512,3), np.uint8)

        smalltecnicalMask = vesicles.DrawUniqueArea(smalltecnicalMask, True)
        #
        # print(vesicles.typeGen)
        # if vesicles.typeGen==1:
        #    print(vesicles.varibleInputFilling)

        mask = cv2.resize(mask, (256, 256))
        img2 = cv2.resize(img2, (256, 256))
        img1 = cv2.resize(img1, (256, 256))

        cv2.imshow("img", img1)
        #cv2.imshow("mask", mask)
        #cv2.imshow("img2", img2)
        #cv2.imshow("tecnicalMask", tecnicalMask)

        #v2.imshow("tecnicalMaskZove", tecnicalMask - mask)

        #cv2.imshow("tecnicalMaskSmall", smalltecnicalMask)
        #cv2.imshow("tecnicalMaskZoveSmall", smalltecnicalMask - mask)

        q = cv2.waitKey()

if __name__ == "__main__":
    testVesicles()
