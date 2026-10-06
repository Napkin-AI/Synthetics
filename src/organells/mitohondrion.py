import numpy as np
import cv2
import random

import math

if __name__ == "__main__":
    import sys
    sys.path.append('.')

from src.organells.location import *
from src.container.spline import *
from src.container.subclass import *
from src.organells.brushes import *
from settings import PARAM, DEBUG_MODE, uniform_int, normal_randint

class Mitohondrion(Location):
    def __init__(self):
        super().__init__()
        self.type = "Mitohondrion"

        # COLORS
        color = normal_randint(
            PARAM['mitohondrion_shell_color_mean'],
            PARAM['mitohondrion_shell_color_std'])
        self.color = (color, color, color)
        self.nowPen = Pen(self.color, np.random.randint(PARAM['mitohondrion_border_w_min'], PARAM['mitohondrion_border_w_max']))    
        self.nowBrush = Brush(self.color)
        self.texture = None

        # GEOMETRY
        self.main_len = 0
        self.inputType = 0             # 0 - обычная митохондрия, 1 - только оболочка

        self.direction = [0,0]         # вектор смещения границы, для имитации косого среза
        self.dopSizeLine = 0

        self.addPoints = []            # точки с смещением для имитации косого среза
        self.addPointsWithOffset = []

        self.Create()

    def __copy__(self):
        new_mito = Mitohondrion()

        new_mito.color = self.color
        new_mito.nowPen  = self.nowPen.copy()

        if self.texture is not None:
            new_mito.nowBrush  = self.nowBrush.copy()
            new_mito.texture  = self.texture.copy()
            
        new_mito.inputType = self.inputType

        new_mito.main_len = self.main_len

        new_mito.direction  = self.direction.copy()
        new_mito.addPoints  = self.addPoints.copy()

        new_mito.dopSizeLine = self.dopSizeLine

        # copy location data
        new_mito.centerPoint = self.centerPoint.copy()
        new_mito.Points = self.Points.copy()
        new_mito.angle = self.angle
        new_mito.numberPoints = self.numberPoints

        new_mito.setDrawParam()
        new_mito.setRandomAngle(0,0)

        return new_mito

    def copy(self):
        return self.__copy__()

    def Create(self):
        if np.random.random() < 0.00:
            self.inputType = 1
            min_r = np.random.randint(10, 15+1)
            max_r = np.random.randint(16, 22+1)
            main_len = np.random.randint(PARAM["mit_len_min"], 50+1)
        else:
            self.inputType = 0
            min_r = np.random.randint(18, 25+1) * 1.7
            max_r = np.random.randint(28, 43+1) * 2
            main_len = np.random.randint(PARAM["mit_len_min"], PARAM["mit_len_max"])

        self.main_len = main_len
        #print(self.main_len)
        # минимум 4, максимум 10
        self.numberPoints = np.random.randint(4, 10)

        half_side_num_points = (self.numberPoints - 2)//2
        step = main_len/(half_side_num_points+1)

        # во избежании создания крайне извилистой или вытянутой митохондрии вводится ограничение на изменение крутизны её извилитости
        min_change_y = min(min_r, main_len//2)
        max_change_y = min(max_r, main_len//2)

        max_change_y_iter = (min_r+max_r)//2

        tPoints = []
        # Добавление первой главной точки
        startMPoint = [-main_len//2, 0]
        tPoints.append(startMPoint.copy())

        start_y_value = np.random.randint(min_change_y, max_change_y + 1)
        # заполнение первой половины митохондрии
        for i in range(half_side_num_points):
           addPointX = startMPoint[0] + step * (i+1)
           addPointY = start_y_value
           tPoints.append([addPointX, addPointY])

           start_y_value = np.clip(start_y_value + np.random.randint(-max_change_y_iter, max_change_y_iter + 1), min_r, max_r)
        # Добавление второй главной точки
        endMPoint = [main_len//2, 0]
        tPoints.append(endMPoint.copy())

        # заполнение второй половины митохондрии
        end_y_value = np.random.randint(min_change_y, max_change_y + 1)
        for j in range(half_side_num_points):
           addPointX = endMPoint[0] - step * (j+1)
           addPointY = -end_y_value
           tPoints.append([addPointX, addPointY])

           end_y_value = np.clip(end_y_value + np.random.randint(-max_change_y_iter, max_change_y_iter + 1), min_r, max_r)

        self.Points = tPoints
    
        '''
        этот код в основном генерирует круглые митохондрии большого размера - плохая генерация !
         
        # заполнение иксов - поделили отрезок на half_len + 2 отрезков
        xes = np.linspace(-main_len//2, main_len//2, num = half_len + 2)[1:-1]
        # нам нужны эти точки дважды, для верхней и для нижней половин митохондрии
        xes = np.concatenate((xes, xes[::-1]))


        # во избежание создания крайне извилистой или вытянутой митохондрии 
        # вводится ограничение на изменение крутизны её извилистости
        step = main_len / (half_len+1)
        
        # игреки для верхней половины митохондрии
        start_y_value = np.random.randint(min(max(step, min_r),max_r)//2 , min(max_r, main_len//2)+1)
        yes = np.random.randint(-max_r//2, max_r//2 + 1, half_len)
        cumulative_sum = np.clip(np.cumsum(yes, axis = 0) + start_y_value, min_r, max_r)
        yes = np.array([start_y_value]) + cumulative_sum
        
        # игреки для нижней половины митохондрии
        start_y_value = np.random.randint(min(max(step, min_r),max_r)//2 , min(max_r, main_len//2)+1)
        yes2 = np.random.randint(-max_r//2, max_r//2 + 1, half_len)
        cumulative_sum =  np.clip(np.cumsum(yes2, axis = 0) + start_y_value, min_r, max_r)
        yes2 = np.array([start_y_value]) + cumulative_sum
        
        yes = np.concatenate((yes , (-1) * yes2[::-1]))


        list1 = xes.tolist()
        list2 = yes.tolist()
        self.Points = [[x, y] for x, y in zip(list1, list2)]
        '''

        # поворот митохондрии
        self.angle = np.random.randint(0, 90)

        #добавление доп. точек
        if np.random.random() < 0.5: 
            self.dopSizeLine = np.random.randint(3, 4+1)

            direction = np.random.randint(-4, 4+1, 2)

            len_dir = math.sqrt(direction[0]**2 + direction[1]**2)

            if len_dir != 0:
                self.direction = [direction[0]/len_dir, direction[1]/len_dir]

        self.setRandomAngle(0, 0)
        
    def AddShellBreaks(self, layer_drawing, boarder_mask):
        break_zones = np.zeros(boarder_mask.shape, np.uint8)
        
        # поиск ограничивающего прямоугольника
        contours, _ = cv2.findContours(boarder_mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)

        for i in range(20):
            center = (np.random.randint(x, x+w+1), np.random.randint(y, y+h+1)) 
            radius = (np.random.randint(4, 5+1, 2))
        
            cv2.ellipse(img = break_zones,
                        center = center,
                        axes = radius,
                        color = 255,
                        thickness = -1,
                        angle = np.random.randint(0,90+1),
                        startAngle = 0,
                        endAngle = 360)

        change_mask = boarder_mask&break_zones
        
        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.uint8)

        change_mask = cv2.erode(change_mask,kernel,iterations = np.random.randint(self.nowPen.sizePen+self.dopSizeLine-2, 2*self.nowPen.sizePen+self.dopSizeLine-2))
        
        breaks_color =  self.color[0] + np.random.randint(5, 40+1)
        layer_drawing[change_mask==255] = (breaks_color, breaks_color, breaks_color)
        
        #cv2.imshow("dop", change_mask)
        
        return layer_drawing

    def Draw(self, image, create_texture=False):
        # Основная рисующая фукция
        if create_texture and self.inputType == 0:
            self.texture, self.nowBrush = CreateTexture(image, self.main_len, self.angle)

        draw_image = image.copy()

        self.nowBrush.FullBrush(draw_image, self.PointsWithOffset)

        boarder_mask = np.zeros(image.shape[:2], np.uint8)
        if self.dopSizeLine != 0 and  self.direction[0] != 0 and self.direction[1] != 0:
            for step in range(-self.dopSizeLine, self.dopSizeLine+1, self.nowPen.sizePen-1):

                temp_PointsWithOffset_for_boarder = []

                for point in self.PointsWithOffset:
                    temp_PointsWithOffset_for_boarder.append([point[0]+int(round(step*self.direction[0])), point[1] + int(round(step*self.direction[1]))])

                spline_line(draw_image, temp_PointsWithOffset_for_boarder, self.nowPen.color, self.nowPen.sizePen)
                spline_line(boarder_mask, temp_PointsWithOffset_for_boarder, 255, self.nowPen.sizePen)


        # if len(self.addPointsWithOffset) != 0:
        #    fill_texture_2_poligons(draw_image, self.addPointsWithOffset[0:3], self.addPointsWithOffset[3:6], self.nowPen.color, self.nowPen.sizePen+1)
        #    fill_texture_2_poligons(draw_image, self.addPointsWithOffset[6:9], self.addPointsWithOffset[9:12], self.nowPen.color, self.nowPen.sizePen+1)
        else:
            spline_line(draw_image, self.PointsWithOffset, self.nowPen.color, self.nowPen.sizePen)
            spline_line(boarder_mask, self.PointsWithOffset, 255, self.nowPen.sizePen)

        if create_texture and self.inputType == 0:
            draw_image = self.AddShellBreaks(draw_image, boarder_mask)

        return draw_image

    def DrawLayer(self, image):
        return self.Draw(image, create_texture=True)

    def setDrawParam(self):
        self.nowPen.color = self.color
        if self.inputType == 0:
            self.nowBrush = Brush(brush = self.texture, typeFull = "texture")
        else:
            self.nowBrush = Brush(self.color)

    def setMaskParam(self):
        self.nowPen.color = (255,255,255)
        self.nowBrush = Brush((255,255,255))

    def setMaskBoarderParam(self):
        self.nowPen.color = (255,255,255)
        if self.inputType == 0:
            self.nowBrush =  Brush((0,0,0))
        else:
            self.nowBrush =  Brush((255,255,255))

    def DrawMask(self, image):
        #Смена цветов для рисования маски
        self.setMaskParam()
        mask = self.Draw(image)
        self.setDrawParam()

        return mask

    def DrawMaskBoarder(self, image):
        #Смена цветов для рисования маски
        self.setMaskBoarderParam()
        mask = self.Draw(image)
        self.setDrawParam()

        return mask

    def DrawUniqueArea(self, image, small_mode = False):
        # Функция создающая маску с немного большим отступом для алгорима случайного размещения новых органнел без пересечения
        ret_image = image.copy()
        ret_image = ret_image.astype(int)
        # import matplotlib.pyplot as plt
        # plt.imshow(ret_image)
        # plt.show()

        draw_image = np.zeros(image.shape, np.uint8)
        draw_image = self.DrawMask(draw_image)
        # plt.imshow(draw_image)
        # plt.show()

        # размещать митохондрии в упор друг к другу и если они ближе 4 пикселей, то регион будет общим (в том числе и с другими классами)
        if small_mode == True:
            kernel = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=np.uint8)

            draw_image = cv2.dilate(draw_image,kernel,iterations = 2)

        ret_image = ret_image + draw_image
        ret_image[ret_image[:,:,:] > 255] = 255
        ret_image = ret_image.astype(np.uint8)

        return ret_image


def testMitohondrion():
    q = None

    print("Press button 'Q' or 'q' to exit")
    while q != ord('q') and q != ord("Q"):

        color = np.random.randint(182, 200)

        img = np.full((512,512,3), (color,color,color), np.uint8)

        for i in range(0, 512, 32):
            img[i:i+15,:,:] += 30

        mitohondrion = Mitohondrion()

        mitohondrion.NewPosition(256,256)

        print("centerPoint", mitohondrion.centerPoint)
        print("Points", mitohondrion.Points)
        print("PointsWithOffset", mitohondrion.PointsWithOffset)

        print("addPoints", mitohondrion.addPoints)
        print("addPointsWithOffset", mitohondrion.addPointsWithOffset)

        img1 = mitohondrion.DrawLayer(img)

        mask = np.zeros((512,512,3), np.uint8)

        tecnicalMask = np.zeros((512,512,3), np.uint8)

        maskMito = mitohondrion.DrawMask(mask.copy())
        img2 = mitohondrion.DrawLayer(img)
        tecnicalMask = mitohondrion.DrawUniqueArea(tecnicalMask)

        for point in mitohondrion.PointsWithOffset:
            cv2.circle(img1, point, 3, (0, 0, 255), 2)

        for point in mitohondrion.addPointsWithOffset:
            cv2.circle(img2, point, 3, (0, 255, 0), 2)

        Boarder = mitohondrion.DrawMaskBoarder(mask)
        mask = cv2.resize(mask, (256, 256))
        img2 = cv2.resize(img2, (256, 256))
        img1 = cv2.resize(img1, (256, 256))
        Boarder = cv2.resize(Boarder, (256, 256))
        cv2.imshow("img", img1)
        cv2.imshow("mask", maskMito)
        cv2.imshow("img2", img2)
        cv2.imshow("tecnicalMask", tecnicalMask)
        cv2.imshow("Boarder", Boarder)

        q = cv2.waitKey()

if __name__ == "__main__":
    testMitohondrion()
