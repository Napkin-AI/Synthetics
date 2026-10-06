import numpy as np
import cv2

import math

if __name__ == "__main__":
    import sys
    sys.path.append('.')

from src.organells.location import *
from src.container.spline import *
from src.container.subclass import *
from settings import PARAM, DEBUG_MODE, uniform_float, uniform_int, normal_randint

WHITE = (255,255,255)

class PSD(Location):
    def __init__(self, ThreePoints = None):
        super().__init__()
        self.type = "PSD"

        """
        Генерация PSD в 3 линии
        u - верхняя
        i - внутрення
        d - нижняя

        Форма по слоям:
            uid
           uid
           uid
           uid
           uid
            uid
        """

        color = normal_randint(
            PARAM['psd_back_color_mean'],
            PARAM['psd_back_color_std'])

        addcolor = normal_randint(
            PARAM['psd_addcolor_mean'],
            PARAM['psd_addcolor_std'])

        centerline_color = normal_randint(
            PARAM['psd_centerline_color_mean'],
            PARAM['psd_centerline_color_std'])

        self.colors = {'sandwich' : (color, color, color),
                       'center line' : (centerline_color, centerline_color, centerline_color),
                       'pallium': (addcolor, addcolor, addcolor)}

        self.pens = {
                     'sandwich' : Pen(self.colors['sandwich']),    # карандаш для темной линии PSD
                     'center'   : Pen(self.colors['center line']), # карандаш для центральной линии
                     'brush'    : Brush(self.colors['pallium'])    # кисть для заливки затемненной области (паллиума)
                     }

        self.length = {'hmin': 15, 'hmax': 40}

        self.lenPSD = 0
        self.typeGen = 0       # 0 - with center line, 1 - full fill PSD

        if np.random.randint(0,10) == 0:
            self.typeGen = 1
            self.colors['pallium'] = self.colors['sandwich']
            self.pens['center'].color = self.colors['pallium']

        if ThreePoints is None:
            self.Create()
        else:
            self.CreateThreePoints(ThreePoints)

    def Create(self):
        lenPSD_05 = np.random.randint(self.length['hmin'], self.length['hmax'] + 1)
        normal_y = abs(np.random.normal(loc=0.0, scale=0.5)) * self.length['hmin']//2

        # Coздание первой основной точки
        point1 = [-lenPSD_05,0]
        # Coздание точки на нормали
        normal = [0, normal_y]
        # Coздание третьей основной точки
        point2 = [lenPSD_05,0]

        self.CreateThreePoints([point1, normal, point2])

        self.setRandomAngle(0,360)

    def __copy__(self):
        new_psd = PSD()
        new_psd.colors = self.colors.copy()
        new_psd.lineSize = self.lineSize.copy()

        new_psd.lenPSD = self.lenPSD
        new_psd.typeGen = self.typeGen

        # copy location data
        new_psd.centerPoint = self.centerPoint.copy()
        new_psd.Points = self.Points.copy()
        new_psd.angle = self.angle
        new_psd.numberPoints = self.numberPoints

        new_psd.setDrawParam()
        new_psd.setRandomAngle(0,0)
        return new_psd

    def copy(self):
        return self.__copy__()

    def CreateThreePoints(self, ThreePoints):
        # в 1 из 10 случаев полностью черная, в 9/10 с полосой внутри
        if self.typeGen == 1:
            self.lineSize = {'top': 2, 'center': np.random.randint(2, 3+1), 'bottom': 2}
        else:
            self.lineSize = {'top': np.random.randint(2, 4+1), 'center': np.random.randint(1, 2+1), 'bottom': 2}

        point1 = np.array(ThreePoints[0])
        normal = np.array(ThreePoints[1])
        point2 = np.array(ThreePoints[2])

        # приведение входных точек в горизонтальный вид и в начало координат, с запоминанием исходного местоположения

        # центр - точка между краями
        centerPoint = (point1 + point2)//2
        lenPSD = np.linalg.norm(point1 - point2)

        self.centerPoint = centerPoint
        self.lenPSD = lenPSD

        vec = point2 - centerPoint
        vec = 2 * vec / lenPSD

        if vec[0] != 0:
            #print("\t\ttype 1")
            self.angle = math.degrees(math.atan(vec[1]/vec[0]))
        else:
            #print("\t\ttype 2")
            self.angle = math.degrees(math.pi/2 - math.atan(vec[0]/vec[1]))

        normal_y = np.linalg.norm(centerPoint - normal)

        ########### 0-2 main-line (цвет PSD или цвет межклеточный)
        # Добавление первой основной точки
        self.Points.append([-self.lenPSD//2,0])
        # Добавление точки на нормали
        self.Points.append([0, normal_y])
        # Добавление третьей основной точки
        self.Points.append([self.lenPSD//2,0])
        ############


        ########### 3-7 PSD-line (цвет PSD под выпуклостью)
        # смещение 1 дополнительной полосы в выпуклую(внешнюю) сторону (+ значение) и в внутренюю сторону (- значение)
        sizeOffsetY_1 = - (self.lineSize['bottom'] + self.lineSize['center']) + 1
        #главная темная линия PSD
        p1_1 = point1 + (0, sizeOffsetY_1 + 1)
        c_1  = normal + (0, sizeOffsetY_1)
        p2_1 = point2 + (0, sizeOffsetY_1 + 1)

        self.Points = self.Points + [p1_1.tolist(), c_1.tolist(), p2_1.tolist()]

        ########### 8-12 PSD-line (цвет PSD над выпуклостью)
        # смещение 2 дополнительной полосы в выпуклую(внешнюю) сторону (+ значение) и в внутренюю сторону (- значение)
        sizeOffsetY_2 = (self.lineSize['top'] + self.lineSize['center'])
        #главная темная линия PSD
        p1_2 = point1 + (0, sizeOffsetY_2 - 1)
        c_2  = normal + (0, sizeOffsetY_2)
        p2_2 = point2 + (0, sizeOffsetY_2 - 1)

        self.Points = self.Points + [p1_2.tolist(), c_2.tolist(), p2_2.tolist()]

        # Отрисовка замкнутой области для затемнения
        sizeOffset = 10
        sizeOffset2 = 5
        self.Points = self.Points + [self.Points[3], self.Points[4], self.Points[5]]
        self.Points.append((p2_2 + (0, sizeOffset2)).tolist())
        self.Points.append(( c_2 + (0, sizeOffset )).tolist())
        self.Points.append((p1_2 + (0, sizeOffset2)).tolist())

        self.numberPoints = len(self.Points)

        self.setAngle(self.angle)

    def setDrawParam(self):
        self.pens = {'sandwich' : Pen(self.colors['sandwich']), 'center' : Pen(self.colors['center line']), 'brush' : Brush(self.colors['pallium'])}

    def setMaskParam(self):
        self.pens = {'sandwich': Pen(WHITE), 'center' : Pen(WHITE), 'brush' : Brush(WHITE) }

    def addedBlur(self, draw_image, rangeList):
        arr = np.asarray(rangeList)
        offset = 20
        minxy = np.amin(arr, axis=0)
        minxy = np.maximum(np.zeros(2), minxy - offset).astype(int)
        maxxy = np.amax(arr, axis=0)
        maxxy = np.minimum(np.asarray([draw_image.shape[0], draw_image.shape[1]]),maxxy + offset).astype(int)
        add = np.random.randint(2,3)
        r = PARAM["main_radius_gausse_blur"] + add
        G = PARAM["main_sigma_gausse_blur"] + add + 0.5
        patch = cv2.GaussianBlur(draw_image[minxy[1]:maxxy[1], minxy[0]:maxxy[0]],(r*2+1,r*2+1), G)
        original = draw_image[minxy[1]:maxxy[1], minxy[0]:maxxy[0]]

        self.pens['brush'].color = WHITE
        mask = np.zeros(draw_image.shape)
        self.pens['brush'].FullBrush(mask, rangeList)
        self.pens['brush'].brush = self.colors['pallium']
        mask = mask[minxy[1]:maxxy[1], minxy[0]:maxxy[0],0]

        kernel = np.ones((r, r), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        original = np.where(mask > 0, patch, original)
        draw_image[minxy[1]:maxxy[1], minxy[0]:maxxy[0]] = original

    def Draw(self, image):
        # Основная рисующая фукция

        draw_image = image.copy()

        # основной алгоритм - нарисовать пятно, нарисовать линии PSD (над и под), нарисовать внутреннюю линию

        # пятно
        rangeList = self.PointsWithOffset[9:9+6]
        self.pens['brush'].FullBrush(draw_image, rangeList)

        #if draw_layer:
        self.addedBlur(draw_image, rangeList)

        # рисование нижней линии 
        rangeList_up = self.PointsWithOffset[3: 3+3]
        small_spline_line(draw_image, rangeList_up, self.pens['sandwich'].color, self.lineSize['bottom'])
        # рисование верхней линии
        rangeList_down = self.PointsWithOffset[6: 6+3]
        small_spline_line(draw_image, rangeList_down, self.pens['sandwich'].color, self.lineSize['top'])
        # рисование внутренней линии
        rangeList_input = self.PointsWithOffset[0: 0+3]
        small_spline_line(draw_image, rangeList_input, self.pens['center'].color, self.lineSize['center'])

        return draw_image

    def DrawLayer(self, image):
        draw_image = self.Draw(image)
        return draw_image

    def DrawUniqueArea(self, image, small_mode = False):
        # Функция создающая маску с немного большим отступом для алгорима случайного размещения новых органнел без пересечения
        ret_image = image.copy()

        ret_image = ret_image.astype(int)

        draw_image = np.zeros(image.shape, np.uint8)

        draw_image = self.DrawMask(draw_image)

        if small_mode == False:
            kernel = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=np.uint8)
            draw_image = cv2.dilate(draw_image,kernel,iterations = 30)
        else:
            kernel = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=np.uint8)
            draw_image = cv2.dilate(draw_image,kernel,iterations = 15)

            #cv2.imshow("sdad", draw_image)
            #cv2.waitKey()

        ret_image = ret_image + draw_image
        ret_image[ret_image[:,:,:] > 255] = 255
        ret_image = ret_image.astype(np.uint8)

        return ret_image

    def DrawMask(self, image):
        #Смена цветов для рисования маски
        self.setMaskParam()
        mask = self.Draw(image)
        mask[mask < 192] = 0
        mask[mask > 192] = 255

        self.setDrawParam()

        return mask

def testPSD():
    q = None

    print("1/2 Press button 'Q' or 'q' to exit")
    while q != ord('q') and q != ord("Q"):

        color = np.random.randint(182, 200)

        img = np.full((512,512,3), (color,color,color), np.uint8)

        for i in range(0, 512, 32):
            img[i:i+15,:,:] += 30

        psd = PSD()

        psd.NewPosition(256,256)

        #psd.setAngle(-psd.angle)

        print("centerPoint", psd.centerPoint)
        print("Points", psd.Points)
        print("PointsWithOffset", psd.PointsWithOffset)

        print("typeGen", psd.typeGen)

        img1 = psd.DrawLayer(img)

        mask = np.zeros((512,512,3), np.uint8)

        tecnicalMask = np.zeros((512,512,3), np.uint8)

        mask = psd.DrawMask(mask)
        img2 = psd.DrawLayer(img)
        tecnicalMask = psd.DrawUniqueArea(tecnicalMask)
        mask = cv2.resize(mask, (256, 256))
        img2 = cv2.resize(img2, (256, 256))
        img1 = cv2.resize(img1, (256, 256))

        cv2.imshow("img", img1)
        # cv2.imshow("mask", mask)
        # cv2.imshow("img2", img2)
        # cv2.imshow("tecnicalMask", tecnicalMask)

        r = PARAM["main_radius_gausse_blur"]
        G = PARAM["main_sigma_gausse_blur"]
        Img = cv2.GaussianBlur(img1,(r*2+1,r*2+1), G)

        noisy = np.ones(img1.shape[:2], np.uint8)
        noisy = np.random.poisson(noisy)*PARAM['poisson_noise'] - PARAM['poisson_noise']/2

        Img = Img + cv2.merge([noisy, noisy, noisy])
        Img[Img < 0] = 0
        Img[Img > 255] = 255
        Img = Img.astype(np.uint8)

        cv2.imshow("pallium noise and blur", Img)

        q = cv2.waitKey()

    q = None

    print("2/2 Press button 'Q' or 'q' to exit")
    while q != ord('q') and q != ord("Q"):

        color = np.random.randint(182, 200)

        img = np.full((512,512,3), (color,color,color), np.uint8)

        for i in range(0, 512, 32):
            img[i:i+15,:,:] += 30

        Three_gen_temp = [np.random.randint(128,384,2) for i in range(2)]
        center = [(Three_gen_temp[0][0] + Three_gen_temp[1][0])//2, (Three_gen_temp[0][1] + Three_gen_temp[1][1])//2]

        dir = (Three_gen_temp[0][0] - center[0], Three_gen_temp[0][1] - center[1])

        if (dir[0] == 0):
            normal_x = 1
            normal_y = 0
        else:
            normal_y = 1
            normal_x = -dir[1]/dir[0]

            len_normal = math.sqrt(normal_y + normal_x**2)

            normal_y /= len_normal
            normal_x /= len_normal

        size_change = np.random.randint(0,128)
        norm_point = [int(round(center[0] + normal_x * size_change)), int(round(center[1] + normal_y * size_change))]

        Three_gen = [Three_gen_temp[0], norm_point, Three_gen_temp[1]]

        print(Three_gen)
        psd = PSD(Three_gen)

        print("centerPoint", psd.centerPoint)
        print("Points", psd.Points)
        print("PointsWithOffset", psd.PointsWithOffset)

        print("typeGen", psd.typeGen)
        print("angle", psd.angle)

        img1 = psd.DrawLayer(img)

        for point in Three_gen:
            cv2.circle(img1, point, 3, (0, 0, 255), 2)

        cv2.circle(img1, center, 3, (0, 255, 0), 2)



        mask = np.zeros((512,512,3), np.uint8)

        tecnicalMask = np.zeros((512,512,3), np.uint8)

        mask = psd.DrawMask(mask)
        img2 = psd.DrawLayer(img)
        tecnicalMask = psd.DrawUniqueArea(tecnicalMask)
        mask = cv2.resize(mask, (256, 256))
        img2 = cv2.resize(img2, (256, 256))
        img1 = cv2.resize(img1, (256, 256))
        cv2.imshow("img", img1)
        # cv2.imshow("mask", mask)
        # cv2.imshow("img2", img2)
        # cv2.imshow("tecnicalMask", tecnicalMask)
        #
        # r = PARAM["main_radius_gausse_blur"]
        # G = PARAM["main_sigma_gausse_blur"]
        # Img = cv2.GaussianBlur(img1,(r*2+1,r*2+1), G)
        #
        # noisy = np.ones(img1.shape[:2], np.uint8)
        # noisy = np.random.poisson(noisy)*PARAM['poisson_noise'] - PARAM['poisson_noise']/2
        #
        # Img = Img + cv2.merge([noisy, noisy, noisy])
        # Img[Img < 0] = 0
        # Img[Img > 255] = 255
        # Img = Img.astype(np.uint8)
        #
        # cv2.imshow("add noise and blur", Img)

        q = cv2.waitKey()

if __name__ == "__main__":
    testPSD()
