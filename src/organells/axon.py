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

class Axon(Location):
    def __init__(self):
        super().__init__()
        self.type = "Axon"

        self.nowInnerBrush = None
        self.nowBubbleBrush = None
        self.innerTexture = None
        self.bubbleTexture = None

        color = normal_randint(
            PARAM['axon_shell_color_mean'],
            PARAM['axon_shell_color_std'])
        self.shell_color = (color, color, color)
        self.use_stretch = np.random.uniform(0, 1) >= PARAM.get("axon_use_stretch_p", 1)

        self.sizePen = np.random.randint(PARAM.get('axon_shell_min_size', 14), PARAM.get('axon_shell_max_size', 18))
        self.sizeAddPen = np.random.randint(self.sizePen, PARAM.get('axon_shell_max_size') or 20)

        self.input_radius = None
        self.sublistNumber = 0

        self.typeGen = 0 # 0 - only output boarder, 1 - with input boarder

        self.nowPen = Pen(self.shell_color, self.sizePen)
        self.nowAddPen = Pen(self.shell_color, self.sizeAddPen)
        self.Create()

    def __copy__(self):
        new_axon = Axon()

        if self.innerTexture is not None:
            new_axon.nowInnerBrush = self.nowInnerBrush.copy()
            new_axon.nowBubbleBrush= self.nowBubbleBrush.copy()
            new_axon.innerTexture  = self.innerTexture.copy()
            new_axon.bubbleTexture = self.bubbleTexture.copy()

        new_axon.shell_color = self.shell_color
        new_axon.sizePen = self.sizePen
        self.sizeAddPen = self.sizeAddPen
        new_axon.input_radius = self.input_radius
        new_axon.sublistNumber = self.sublistNumber
        new_axon.typeGen = self.typeGen

        new_axon.nowPen = self.nowPen.copy()
        new_axon.nowAddPen = self.nowAddPen.copy()
        
        # copy location data
        new_axon.centerPoint = self.centerPoint.copy()
        new_axon.Points = self.Points.copy()
        new_axon.angle = self.angle
        new_axon.numberPoints = self.numberPoints

        new_axon.setDrawParam()
        new_axon.setRandomAngle(0,0)
        return new_axon

    def copy(self):
        return self.__copy__()

    def Create(self):

        min_r = 0; PARAM.get("axon_min_r", 0)
        max_r = 0; PARAM.get("axon_max_r", 0)

        if (PARAM.get('axon_shell_type') == 0) or np.random.randint(0, 5) == 0:
            self.typeGen = 0
        else:
            self.typeGen = 1

        self.Points = []

        self.numberPoints = np.random.randint(
            PARAM.get("axon_points_min", 14),
            PARAM.get("axon_points_max", 19) + 1
        )


        if self.typeGen == 0:
            if min_r == 0:
                min_r = np.random.randint(
                    PARAM.get("axon_min_radius_min", 15),
                    PARAM.get("axon_min_radius_max", 36)
                )

            if max_r == 0:
                max_r = min_r + np.random.randint(
                    PARAM.get("axon_radius_growth_min", 5),
                    PARAM.get("axon_radius_growth_max", max(5, 41 - min_r + 5))
                )

        elif self.typeGen == 1:
            if (min_r == 0):
                min_r = np.random.randint(40, 71)

            if (max_r == 0):
                max_r = min_r + np.random.randint(4, 104 - min_r + 4)                                # лучше поменять параметры

        max_increase_len = max_r - min_r

        # 2 типа генерации, если маленький просто оболочка, если большой, то с внутренней частью

        step_angle = 2.0 * math.pi / self.numberPoints

        stretch = np.random.uniform(
            PARAM.get("axon_stretch_min", 1),
            PARAM.get("axon_stretch_max", 1) if self.use_stretch else 2
        )
        r = (min_r + max_r) / 2

        a = r * stretch
        b = r / stretch

        print(a, b, r, min_r, max_r, stretch)
        for i in range(self.numberPoints):
            now_angle = step_angle * i

            noise = np.random.randint(1, max_increase_len)

            x = (a + noise) * math.sin(now_angle)
            y = (b + noise) * math.cos(now_angle)

            now_point = [int(round(x)), int(round(y))]
            self.Points.append(now_point)

        if self.typeGen == 1:
            self.input_radius = np.random.randint(min_r/2, min_r-2, 2)
            self.angle = np.random.randint(0, 360)

        self.ChangePositionPoints()
        self.sublistNumber = np.random.randint(3, len(self.PointsWithOffset) - 3)

    def CreateNewTexture(self, image):
        self.innerTexture = image

        color = normal_randint(
            PARAM['main_color_mean'],
            PARAM['main_color_std'])
        # backgroundСolor = (color, color, color) # выбор цвета фона

        self.bubbleTexture = np.full(image.shape, color, np.uint8)

        count = 600
        xs = np.random.randint(0, image.shape[0], count)
        ys = np.random.randint(0, image.shape[1], count)
        xd = np.random.randint(-5, 5, count)
        yd = np.random.randint(-5, 5, count)
        xe = np.minimum(np.ones(count) * image.shape[0], np.maximum(np.zeros(count), xs + xd)).astype(np.int32)
        ye = np.minimum(np.ones(count) * image.shape[1], np.maximum(np.zeros(count), ys + yd)).astype(np.int32)
        w = np.random.randint(1, 5, count)
        for i in range(0, count):
            c = np.random.randint(100, 150)
            self.bubbleTexture = cv2.line(self.bubbleTexture, [xs[i], ys[i]], [xe[i], ye[i]], (c,c,c) , w[i])

        r = 5
        G = (2 * r + 1) / 3
        self.bubbleTexture = cv2.GaussianBlur(self.bubbleTexture,(r*2+1,r*2+1), G)

        t = normal_randint(
            PARAM['axon_back_color_diff_mean'],
            PARAM['axon_back_color_diff_std'])

        self.bubbleTexture[self.bubbleTexture[:,:,0] > t] -= t

        # создание пятен  (spot)
        now_x = 10

        spot_color1 = 168
        spot_color2 = 100

        while now_x < image.shape[1] - 10:
            now_y = 10
            while now_y < image.shape[1] - 10:

                radius = np.random.randint(11,14,2)

                Brush((spot_color1,spot_color1,spot_color1)).FullBrushEllipse(self.bubbleTexture, (now_x, now_y), radius)

                cv2.ellipse(img = self.bubbleTexture,
                           center = (now_x, now_y),
                           axes = radius,
                           color = (spot_color2,spot_color2,spot_color2),
                           thickness = 4,
                           angle = 0,
                           startAngle = 0,
                           endAngle = 360)
                #step y
                step_y = np.random.randint(160, 201)
                now_y = now_y + step_y

            #step x
            step_x = np.random.randint(160,201)
            now_x += step_x

        self.nowInnerBrush = Brush(self.innerTexture, typeFull = "texture")
        self.nowBubbleBrush = Brush(self.bubbleTexture, typeFull = "texture")

    #def SheathTear(self, size):

    def Draw(self, image, layer_drawing = True):
        # Основная рисующая фукция
        draw_image = image.copy()

        if layer_drawing == True:
            self.CreateNewTexture(image)

        spline_line(draw_image, self.PointsWithOffset, self.nowPen.color, self.nowPen.sizePen)

        # для 2 типа генерации
        if (self.input_radius is not None):
            # дополнительное утолщение части границы
            sublist = self.PointsWithOffset[1: self.sublistNumber+1+1]

            small_spline_line(draw_image, sublist, self.nowAddPen.color, self.nowAddPen.sizePen)

            #заполнение внутренности
            self.nowBubbleBrush.FullBrush(draw_image, self.PointsWithOffset)

            #очистить центральную внутренность
            self.nowInnerBrush.FullBrushEllipse(draw_image, self.centerPoint, self.input_radius, self.angle)

            #внутренняя оболочка
            cv2.ellipse(img = draw_image,
                       center = (self.centerPoint[0], self.centerPoint[1]),
                       axes = self.input_radius,
                       color = self.nowPen.color,
                       thickness = 3,
                       angle = self.angle,
                       startAngle = 0,
                       endAngle = 360)

        else:
             #очистить центральную внутренность (и сделать границу тоньше в 2 раза)
             self.nowInnerBrush.FullBrush(draw_image, self.PointsWithOffset)

        return draw_image

    def DrawLayer(self, image):
        return self.Draw(image)

    def DrawMask(self, image):
        #Смена цветов для рисования маски
        self.setMaskParam()
        mask = self.Draw(image, layer_drawing = False)
        self.setDrawParam()

        return mask

    def DrawUniqueArea(self, image, small_mode = False):
        # Функция создающая маску с немного большим отступом для алгорима случайного размещения новых органнел без пересечения
        ret_image = image.copy()
        ret_image = ret_image.astype(int)

        draw_image = np.zeros(image.shape, np.uint8)

        Brush((255,255,255)).FullBrush(draw_image, self.PointsWithOffset)
        spline_line(draw_image, self.PointsWithOffset, (255,255,255), self.nowPen.sizePen)

        sublist = self.PointsWithOffset[1: self.sublistNumber+1+1]
        spline_line(draw_image, sublist, (255,255,255), self.nowAddPen.sizePen, is_closed = False)

        kernel = np.array([[0, 1, 1, 1, 0],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [0, 1, 1, 1, 0]], dtype=np.uint8)

        if small_mode:
            draw_image = cv2.dilate(draw_image,kernel,iterations = 2)
        else:
            draw_image = cv2.dilate(draw_image,kernel,iterations = 4)

        ret_image = ret_image + draw_image
        ret_image[ret_image[:,:,:] > 255] = 255
        ret_image = ret_image.astype(np.uint8)

        return ret_image


    def setMaskParam(self):
        self.nowInnerBrush = Brush(brush = (0,0,0), typeFull = "full")
        self.nowBubbleBrush = Brush(brush = (255,255,255), typeFull = "full")

        self.nowPen.color = (255,255,255)
        self.nowAddPen.color = (255,255,255)

    def setDrawParam(self):
        self.nowInnerBrush = Brush(brush = self.innerTexture, typeFull = "texture")
        self.nowBubbleBrush = Brush(brush = self.bubbleTexture, typeFull = "texture")

        self.nowPen.color = self.shell_color
        self.nowAddPen.color = self.shell_color


def testAxon():

    q = None


    print("Press button 'Q' or 'q' to exit")
    while q != ord('q') and q != ord("Q"):

        color = np.random.randint(182, 200)

        img = np.full((512,512,3), (color,color,color), np.uint8
                      )

        for i in range(0, 512, 32):
            img[i:i+15,:,:] += 30

        axon = Axon()

        axon.NewPosition(256,256)

        print("centerPoint", axon.centerPoint)
        print("Points", axon.Points)
        print("PointsWithOffset", axon.PointsWithOffset)
        print("sublistNumber", axon.sublistNumber)

        img1 = axon.Draw(img)

        mask = np.zeros((512,512,3), np.uint8)

        tecnicalMask = np.zeros((512,512,3), np.uint8)

        mask = axon.DrawMask(mask)
        img2 = axon.Draw(img)
        mask = cv2.resize(mask, (256, 256))
        img2 = cv2.resize(img2, (256, 256))
        img1 = cv2.resize(img1, (256, 256))
        tecnicalMask = axon.DrawUniqueArea(tecnicalMask)

        cv2.imshow("img", img1)
        cv2.imshow("mask", mask)
        cv2.imshow("img2", img2)
        cv2.imshow("tecnicalMask", tecnicalMask)

        q = cv2.waitKey()

if __name__ == "__main__":
    testAxon()
