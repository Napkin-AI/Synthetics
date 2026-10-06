import numpy as np
import cv2

def uniform_float(mean, std):
    return random.uniform(mean - std, mean + std)


def uniform_int(mean, std):
    return int(random.uniform(mean - std, mean + std) + 0.5)


def normal_randint(mean, std):
    vals = int(random.gauss(mean, std/3)+0.495)
    return 0 if vals < 0 else 255 if vals > 255 else vals

import random
import json

with open("D:\course_work\data\synth_rep\Synthetics\kasthuri_indexed.json", 'r', encoding="utf-8") as f:
    PARAM = json.load(f)

DEBUG_MODE = False

import math

if __name__ == "__main__":
    import sys
    sys.path.append('.')

from src.organells.axon import Axon
from src.organells.PSD import PSD

from src.container.spline import *
from src.container.subclass import *

import random


class Membrane:
    def __init__(self, sizeImage, compartmentsList = None):
        self.type = "Membrane"

        self.color = None
        self.nowColor = None
        self.labels = None

        self.sheathTearMap = None
        self.thickeningOfMembraneShellMap = None

        self.typeLine = 0 # 0 -full boarder, 1 - clear center, 2 - clear point in boarder   # 1 type отключен, так как выглядит не естественно 
        self.sizeInputLine = 0 # size input line

        self.Points = []

        #для копий мембран
        self.StopList = None
        #для того, чтобы помниит сколько первых компонентов попало в генерацию, а сколько последних нет
        self.componentsList = compartmentsList     # список хранит только ссылки на те классы, которые используются для построения мембран
        self.imageSize = sizeImage

        self.SetStartValue()
        if compartmentsList is None:
            self.CreateWithNoneList(self.imageSize)
        else:
            self.Create(self.componentsList, self.imageSize)

        if DEBUG_MODE:
            print("Type membrane generation:", self.typeLine)

    def copy_main_param(self, main_membran):
        self.color = main_membran.color
        self.nowColor = main_membran.nowColor
        self.typeLine = main_membran.typeLine
        self.sizeInputLine = main_membran.sizeInputLine
        self.sizeLine = main_membran.sizeLine
        self.StopList = main_membran.StopList
        self.imageSize = main_membran.imageSize
        self.Create(self.componentsList, self.imageSize)        # перестроить всё со списком остановки

    def CreateWithNoneList(self, sizeImage):

        number_region = np.random.randint(3, 7)


        number_of_associations = np.random.randint(0, number_region//2)

        if DEBUG_MODE:
            print("number_region", number_region)
            print("number of associations", number_of_associations)
        list_center_region = []

        union_radius = 40

        # create unique center
        for i in range(number_region):
            x = np.random.randint(0, sizeImage[1])
            y = np.random.randint(0, sizeImage[0])

            counter = 0
            max_iteration = 100000
            while self.CheckOverlap(list_center_region, (x,y), union_radius) and counter < max_iteration:
                x = np.random.randint(0, sizeImage[1])
                y = np.random.randint(0, sizeImage[0])
                counter += 1

            if counter != max_iteration:
                list_center_region.append([x,y])

        label_map = np.zeros(sizeImage, np.uint8)

        # print center label
        for i, point in enumerate(list_center_region):
            cv2.circle(label_map, point, union_radius-1, 255, -1)

        # create union zone
        for j in range(number_of_associations):
            point1 = random.choice(list_center_region)
            list_center_region.remove(point1)

            point2 = random.choice(list_center_region)
            list_center_region.remove(point2)

            if np.random.random() < 0.5:
                list_center_region.append(point1)
            else:
                list_center_region.append(point2)


            cv2.line(label_map, point1, point2, 255, 1)

        connectivity = 4
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(label_map[:,:], connectivity, cv2.CV_32S)

        #cv2.imshow("union zone", (labels[:,:] * 25).astype(np.uint8))
        #cv2.waitKey()
        #print("count_reg", retval)

        labels = self.RegionExpansion(labels)

        #cv2.imshow("RegionExpansion", ((labels[:,:]+1) * 25).astype(np.uint8))
        #cv2.waitKey()

        self.labels = labels

    def CheckOverlap(self, Points, point, union_radius):
        for i in range(len(Points)):

            x_delt = abs(Points[i][0] - point[0])
            y_delt = abs(Points[i][1] - point[1])

            len_new_now = math.sqrt(x_delt**2 + y_delt**2)

            if len_new_now < union_radius * 2:
                return True

        return False

    def SetStartValue(self):
        #self.typeLine = np.random.randint(0,3)

        choice = np.random.randint(0,10)

        if choice < 8:
            self.typeLine = 0
        elif choice < 9:
            self.typeLine = 2
        else:
            #self.typeLine = 1           # первый тип линии сильно растворяется, поэтому его вероятность появления снижена
            self.typeLine = 2           # отключен, так как выглядит не естественно 
        #self.typeLine = 1

        self.sizeInputLine = np.random.randint(0, PARAM['membrane_thickness_mean'])
        
        self.sizeLine = uniform_int(
            PARAM['membrane_thickness_mean'],
            PARAM['membrane_thickness_std'])

        self.color = normal_randint(PARAM['membrane_color_mean'],
                                    PARAM['membrane_color_std'])

        if self.typeLine == 1:
            self.sizeInputLine = self.sizeInputLine + 1

        if self.typeLine == 2:
            self.sizeLine -= np.random.randint(0,2)

        self.setDrawParam()

    def Create(self, compartments, sizeImg):


        checkMask = np.zeros((*sizeImg,3), np.uint8)
  
        # Создание маски регионов и заполнение её
        for compartment in compartments:
            checkMask = compartment.DrawUniqueArea(checkMask, small_mode = True)

        # Сохранение общей маски
        #cv2.imwrite("Sintetic_generation/original/fileMask.png", checkMask)

        # Создание и инициализация 0 матрицы меток (или матрицы изображения меток)
        #labelImage = np.zeros(sizeImg, int)

        connectivity = 4
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(checkMask[:,:,0], connectivity, cv2.CV_32S)

        # retval - возвращает колличество меток (с 0 меткой для фона) кол-во объектов retval-1 [1:retval-1]
        #print(retval)
        #print(labels)
        #print(labels.max())

        # Сохранение маски меток
        #cv2.imwrite("Sintetic_generation/original/fileMaskLabels.png", labels.astype(np.uint8))

        labels = self.RegionExpansion(labels)

        # Сохранение маски разросшихся регионов
        #cv2.imwrite("Sintetic_generation/original/fileMaskLabelsEnd.png", (labels[:,:] + 1).astype(np.uint8))

        labels = self.ExpansionPSD(labels, compartments)

        #cv2.imwrite("Sintetic_generation/original/fileMaskLabelsEndWithPSD.png", (self.labels[:,:] + 2).astype(np.uint8))

        self.labels = self.CreateMembraneAxon(labels, compartments)

    def RegionExpansion(self, input_labels, num_labels = 0, p_stop = 0.01, min_count_iter = 15):
        """
        Функция разрастания регионов итеративно расширяет границу
        Также эта функия добавляет граничные пиксели в self.Points
        """
        labels = input_labels.copy()

        if num_labels == 0:
            num_labels = input_labels.max()

        #print("num_labels", num_labels)

        # инициализация цикла
        Work_regions = []

        for i in range(num_labels):
            Work_regions.append([i,[]])

        for y in range(labels.shape[0]):
            for x in range(labels.shape[1]):
                if labels[y,x] != 0:
                    add_point = False
                    if x > 0 and labels[y,x-1] == 0:
                        add_point = True
                    if x < labels.shape[1]-1 and labels[y,x+1] == 0:
                        add_point = True
                    if y > 0 and labels[y-1,x] == 0:
                        add_point = True
                    if y < labels.shape[0]-1 and labels[y+1,x] == 0:
                        add_point = True

                    if add_point == True:
                        #print(labels[y,x]-1, len(Work_regions))
                        Work_regions[labels[y,x]-1][1].append([x,y])


        # случайный порядок выбора региона для расширения
        np.random.shuffle(Work_regions)

        Work_regions_number = []
        for i in range(num_labels):
            Work_regions_number.append(len(Work_regions[i][1]))

        #print("iter 0", len(Work_points))

        # количество прекращений разростания в случае если он резко перестал расширяться в отношении к прошлой итерации (для закругления углов)
        statictic_break_procent = 0
        # аналогично предыдущему, но завершается если расширяющихся точек в регионе меньше порогового значения (чтобы не было, например, тонких линий в одну сторону)
        statictic_break_number = 0

        proccent_work = 0.5

        # переключение в режим использования списка остановки
        flag_use_stop_list = False
        if not self.StopList is None:
            flag_use_stop_list = True
        else:
            self.StopList = np.zeros(len(Work_regions), np.uint64)

        counter = 0
        while len(Work_regions) != 0:
            Work_regions_iteration = []

            for i, [index_componenst, region] in enumerate(Work_regions):

                # не используется список остановки
                if not flag_use_stop_list:
                    # Останавливать разростание региона на случайной итерации
                    if counter > min_count_iter and np.random.random() < p_stop:
                        self.StopList[index_componenst] = counter
                        for (x,y) in region:
                            labels[y,x] = -1
                            self.Points.append([x,y])
                        continue

                # используется список остановки
                else:
                    if counter == self.StopList[index_componenst] and self.StopList[index_componenst] > 0:
                        for (x,y) in region:
                            labels[y,x] = -1
                            self.Points.append([x,y])
                        continue

                # основной алгоритм разрастания
                Work_points_iteration = []
                for (x,y) in region:
                    boarder = 0

                    repit = False

                    if x > 0:
                        if labels[y,x-1] == 0:
                            if  np.random.random() < proccent_work:
                                labels[y,x-1] = labels[y,x]
                                Work_points_iteration.append([x-1,y])
                            else:
                                repit = True

                        elif labels[y,x-1] != labels[y,x]:                      # граница в 2 пикселя
                            boarder += 1

                    if x < labels.shape[1]-1:
                        if labels[y,x+1] == 0:
                            if  np.random.random() < proccent_work:
                                labels[y,x+1] = labels[y,x]
                                Work_points_iteration.append([x+1,y])
                            else:
                                repit = True
                        elif labels[y,x+1] != labels[y,x]:
                            boarder += 1

                    if y > 0:
                        if labels[y-1,x] == 0:
                            if  np.random.random() < proccent_work:
                                labels[y-1,x] = labels[y,x]
                                Work_points_iteration.append([x,y-1])
                            else:
                                repit = True

                        elif labels[y-1,x] != labels[y,x]:
                            boarder += 1

                    if y < labels.shape[0]-1:
                        if labels[y+1,x] == 0:
                            if  np.random.random() < proccent_work:
                                labels[y+1,x] = labels[y,x]
                                Work_points_iteration.append([x,y+1])
                            else:
                                repit = True
                        elif labels[y+1,x] != labels[y,x]:
                            boarder += 1

                    if repit == True:
                        Work_points_iteration.append([x,y])
                    # Случайность заполнения исправляет образование очень толстых границ
                    if boarder != 0:
                        if np.random.random() < 0.5:
                            labels[y,x] = -1                                       # Значение -1 обозначает границу
                            self.Points.append([x,y])
                        else:
                            Work_points_iteration.append([x,y])

                #print(counter)
                # продолжать развитие региона если он не перестал резко расширяться и точек больше некоторого количества
                if Work_regions_number[i] * 0.2 < len(Work_points_iteration) and len(Work_points_iteration) > 50:
                    Work_regions_iteration.append([index_componenst, Work_points_iteration])
                else:
                    if Work_regions_number[i] * 0.2 < len(Work_points_iteration):
                        statictic_break_procent += 1
                    else:
                        statictic_break_number += 1

                    for (x,y) in Work_points_iteration:
                        labels[y,x] = -1
                        self.Points.append([x,y])

            # случайный порядок выбора региона для расширения
            np.random.shuffle(Work_regions_iteration)

            Work_regions = Work_regions_iteration
            Work_regions_number = []
            for i in range(len(Work_regions)):
                Work_regions_number.append(len(Work_regions[i][1]))

            counter += 1

        if DEBUG_MODE:
            print(f"Number region iteration: {counter}")

            print(f"Number break expansion for procent: {statictic_break_procent}")
            print(f"Number break expansion for number: {statictic_break_number}")

        return labels

    def ExpansionPSD(self, input_labels, compartments):
        labels = input_labels.copy()

        for compartment in compartments:
            if compartment.type == "PSD" or compartment.type == "Vesicles_and_PSD":
                if compartment.type == "Vesicles_and_PSD":
                    compartment = compartment.PSD

                center = compartment.centerPoint
                startPoint = compartment.PointsWithOffset[0]
                normalPoint = compartment.PointsWithOffset[1]
                endPoint = compartment.PointsWithOffset[2]

                # смещение разметки мембран так как в реальных данных PSD больше внутри 1 мембраны клетки
                if center[0] == normalPoint[0] and center[1] == normalPoint[1]:
                    # вычисление нормали к вектору из end to start
                    if (startPoint[1] - endPoint[1]) != 0:
                        eDirX = 1
                        eDirY = -((startPoint[0] - endPoint[0]) / (startPoint[1] - endPoint[1]))
                    else:
                        eDirX = -((startPoint[1] - endPoint[1]) / (startPoint[0] - endPoint[0]))
                        eDirY = 1

                else:
                    eDirX = normalPoint[0] - center[0];
                    eDirY = normalPoint[1] - center[1];

                lenNormal = math.sqrt(eDirX * eDirX + eDirY * eDirY)
                eDirX /= lenNormal
                eDirY /= lenNormal

                addedX = int(round(eDirX * 2)) # 2 = (6 - 2) /2 - размер пера PSD минус размер пера мембран деленое на 2
                addedY = int(round(eDirY * 2)) # 2 = (6 - 2) /2 - размер пера PSD минус размер пера мембран деленое на 2

                # добавление смещения
                #offset_main_normal = 3
                #offset_start_normal = -1
                offset_main_normal = 1
                offset_start_normal = 0

                self.AddingDirection(labels, [normalPoint[0] + offset_main_normal * addedX, normalPoint[1] + offset_main_normal * addedY],
                                                [startPoint[0] + addedX * offset_start_normal, startPoint[1] + addedY * offset_start_normal], input_dir_membr = normalPoint)

                self.AddingDirection(labels, [normalPoint[0] + offset_main_normal * addedX, normalPoint[1] + offset_main_normal * addedY],
                                                [endPoint[0] + addedX * offset_start_normal, endPoint[1] + addedY * offset_start_normal], input_dir_membr = normalPoint)

        return labels

    def CreateMembranePSD(self, compartments):
        self.labels = self.ExpansionPSD(self.labels, compartments)

    def AddingDirection(self, labels, antiDir, startPoint, labelPrint = -2, input_dir_membr = None):
        #основная идея расширяться в направлении конца до тех пор, пока не упрёмся в границу или выйдем за поле

        # если край за полем видимости, то выходим
        height, width = labels.shape

        if (startPoint[0] < 0 or startPoint[0] >= width):
            return

        if (startPoint[1] < 0 or startPoint[1] >= height):
            return

        # вычисление направления движения
        directionX = startPoint[0] - antiDir[0]
        directionY = startPoint[1] - antiDir[1]

        lenDir = math.sqrt(directionX * directionX + directionY * directionY)
        directionX /= lenDir
        directionY /= lenDir

        #print(f"stPoint {startPoint}, antiDir {antiDir},  dir : {directionX}, {directionY}")

        nowPos = startPoint.copy()
        # Двигаемся, помечая границей labelPrint пока не выйдем за поле или рядом будет граница

        counter = 1
        repit_dir = True

        directionX_work = directionX
        directionY_work = directionY

        rotate_coef = 4

        if input_dir_membr is not None:
            normal_dir_x = input_dir_membr[0] - antiDir[0]
            normal_dir_y = input_dir_membr[1] - antiDir[1]

            lenDirNormal = math.sqrt(normal_dir_x ** 2 + normal_dir_y ** 2)

            normal_dir_x /= lenDirNormal
            normal_dir_y /= lenDirNormal

            sign = -math.copysign(1, normal_dir_x*directionY - normal_dir_y*directionX)


            rotate = (-1+ (-1+sign)/10 * rotate_coef, 1 + (1+sign)/10 * rotate_coef)
        else:
            rotate = (-rotate_coef, rotate_coef)

        last_Pos = nowPos.copy()
        float_Pos = nowPos.copy()

        while repit_dir:
            if  nowPos[0] == last_Pos[0] and nowPos[1] == last_Pos[1]:
                float_Pos = [float_Pos[0]+directionX_work, float_Pos[1]+directionY_work]

                nowPos = [int(round(float_Pos[0])), int(round(float_Pos[1]))]

            else:
                last_Pos = nowPos.copy()
                #print(nowPos)
                flag_fing_boarder = 0 # flag нахождения в окрестности границы

                if nowPos[0] > 0:
                    if labels[nowPos[1], nowPos[0] - 1] < 0:
                        flag_fing_boarder += 1

                if nowPos[0] < width - 1:
                    if labels[nowPos[1], nowPos[0] + 1] < 0:
                        flag_fing_boarder += 1

                if nowPos[1] > 0:
                    if labels[nowPos[1] - 1, nowPos[0]] < 0:
                        flag_fing_boarder += 1

                if nowPos[1] < height - 1:
                    if labels[nowPos[1] + 1, nowPos[0]] < 0:
                        flag_fing_boarder += 1

                # предыдущее значение не считается
                if flag_fing_boarder < 3:
                    # Добавление граничного пикселя
                    labels[nowPos[1], nowPos[0]] = labelPrint
                    self.Points.append(nowPos.copy())

                    # вычисление следующего шага
                    float_Pos = [float_Pos[0]+directionX_work, float_Pos[1]+directionY_work]
                    nowPos = [int(round(float_Pos[0])), int(round(float_Pos[1]))]

                    counter += 1

                else:
                    labels[nowPos[1], nowPos[0]] = labelPrint
                    self.Points.append(nowPos.copy())
                    repit_dir = False

                change_angle = np.random.uniform(rotate[0], rotate[1]+0.00001)
                work_angle = change_angle * (math.pi/180)

                directionX_work = directionX_work * math.cos(work_angle) - directionY_work * math.sin(work_angle)
                directionY_work = directionX_work * math.sin(work_angle) + directionY_work * math.cos(work_angle)


            # проверка останова
            if nowPos[0] < 0 or nowPos[0] >= width:
                repit_dir = False

            elif nowPos[1] < 0 or nowPos[1] >= height:
                repit_dir = False

            elif labels[nowPos[1], nowPos[0]] < 0 and (last_Pos[0] !=  nowPos[0] or last_Pos[1] !=  nowPos[1]):
                repit_dir = False

        #raise Exception("my error")

    def CreateMembraneAxon(self, input_labels, compartments):
        #основная идея оградить axon по периметру и из вершин выпустить лучи

        labels = input_labels.copy()

        labelPrint = -3
        labelPrintOreol = -4

        for compartment in compartments:
            if compartment.type == "Axon":

                center = compartment.centerPoint

                Points = compartment.PointsWithOffset

                for point in Points:
                    self.AddingDirection(labels, center, point, labelPrint = labelPrint)

                mask = np.zeros((*labels.shape, 3), np.uint8)

                mask_axon = compartment.DrawMask(mask)

                Brush((255,255,255)).FullBrush(mask_axon, Points)

                kernel = np.array([[0, 1, 0],
                                   [1, 1, 1],
                                   [0, 1, 0]], dtype=np.uint8)

                mask_axon_with_membrane = cv2.dilate(mask_axon,kernel,iterations = 2)

                mask_erode = cv2.erode(mask_axon_with_membrane,kernel,iterations = 2)

                oreol_membrane_axon = mask_axon_with_membrane - mask_erode

                labels[mask_axon_with_membrane[:,:,0] == 255] = labels[center[1], center[0]]
                labels[oreol_membrane_axon[:,:,0] == 255] = labelPrintOreol

        return labels

    def CreareShellsMap(self, shapeImage):
        self.sheathTearMap = np.zeros(shapeImage[:2], np.uint8)

        sizeShape = shapeImage[0]

        number_zone = np.random.randint(sizeShape//20, sizeShape//10+1)
        min_r = 5
        max_r = 50

        for i in range(number_zone):
            x = np.random.randint(shapeImage[1])
            y = np.random.randint(shapeImage[0])

            cv2.ellipse(img = self.sheathTearMap,
                center = [x, y],
                axes = np.random.randint(min_r, max_r + 1, 2),
                color = 255,
                thickness = -1,
                angle = np.random.randint(0, 180),
                startAngle = 0,
                endAngle = 360)

        self.thickeningOfMembraneShellMap = np.zeros(shapeImage[:2], np.uint8)
        number_thickening_zone = np.random.randint(0, 3+1)
        min_r_t = 30
        max_r_t = 60

        iter = 0
        max_iteration = 100000
        count = 0
        while count < number_thickening_zone and iter < max_iteration:
            temp_mask = np.zeros(shapeImage[:2], np.uint8)

            x_t = np.random.randint(shapeImage[1])
            y_t = np.random.randint(shapeImage[0])

            axis_1 = np.random.randint(min_r_t, max_r_t + 1)

            cv2.ellipse(img = temp_mask,
                center = [x_t, y_t],
                axes = (axis_1, max_r_t + min_r_t - axis_1),
                color = 255,
                thickness = -1,
                angle = np.random.randint(0, 180),
                startAngle = 0,
                endAngle = 360)

            if all(self.sheathTearMap[temp_mask[:,:] == 255] == 0):
                self.thickeningOfMembraneShellMap += temp_mask
                count += 1

            iter += 1

        if DEBUG_MODE:
            if iter == max_iteration:
                print("thickeningOfMembraneShellMap no added:", number_thickening_zone - count, "zones")

    # добавляет небольшие области с расслоением мембран
    def SheathTear(self, mask):
        if self.sheathTearMap is None:
            self.CreareShellsMap(mask.shape)

        mask_sheath_tear = mask & self.sheathTearMap

        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.uint8)

        mask_sheath_tear_deleted = cv2.erode(mask_sheath_tear,kernel,iterations = 2)
        mask_sheath_tear_add = cv2.dilate(mask_sheath_tear,kernel,iterations = 2)
        return (mask | mask_sheath_tear_add) - mask_sheath_tear_deleted
        
        

    def ThickeningOfMembraneShell(self, mask):
        if self.thickeningOfMembraneShellMap is None:
            self.CreareShellsMap(mask.shape)

        mask_thickening_shell = mask & self.thickeningOfMembraneShellMap

        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.uint8)

        self.mask_sheath_tear_add = cv2.dilate(mask_thickening_shell,kernel,iterations = np.random.randint(5, 7+1))

        #cv2.imshow("self.thickeningOfMembraneShellMap", self.thickeningOfMembraneShellMap)
        #cv2.imshow("self.sheathTearMap", self.sheathTearMap)

        #cv2.imshow("mask", mask)
        #cv2.imshow("mask_thickening_shell", mask_thickening_shell)
        #cv2.imshow("mask_sheath_tear_add", mask_sheath_tear_add)
        #cv2.imshow("result", (mask | mask_sheath_tear_add))

        #cv2.waitKey()

        return self.mask_sheath_tear_add

    def Draw(self, image, use_rand_color = False):
        # Основная рисующая фукция

        draw_image = image.copy()

        mask = np.zeros(draw_image.shape[0:2], np.uint8)
        mask[self.labels[:,:] == -1] = 255

        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.uint8)

        mask_dilate = cv2.dilate(mask,kernel,iterations = self.sizeLine)

        maskPSD = np.zeros(draw_image.shape[0:2], np.uint8)
        maskPSD[self.labels[:,:] == -2] = 255
        maskPSD_dilate = cv2.dilate(maskPSD,kernel,iterations = 2)

        maskAxon = np.zeros(draw_image.shape[0:2], np.uint8)
        maskAxon[self.labels[:,:] == -3] = 255
        maskAxon_dilate = cv2.dilate(maskAxon,kernel,iterations = 2)

        maskAxonOreol = np.zeros(draw_image.shape[0:2], np.uint8)
        maskAxonOreol[self.labels[:,:] == -4] = 255

        if self.typeLine == 0:
            mask_erode = cv2.erode(mask_dilate,kernel,iterations = 30)
            mask_dilate = mask_dilate - mask_erode

        if self.typeLine == 1:
            union_mask = mask + maskPSD + maskAxon

            union_mask_dilate = mask_dilate + maskPSD_dilate + maskAxon_dilate
            mask_dilate = cv2.dilate(union_mask_dilate, kernel,iterations = 2)

            maskPSD_dilate = maskPSD = np.zeros(draw_image.shape[0:2], np.uint8)
            maskAxon_dilate = maskAxon = np.zeros(draw_image.shape[0:2], np.uint8)


            if self.sizeInputLine != 0:
                mask_dilate = mask_dilate - cv2.dilate(union_mask,kernel,iterations = self.sizeInputLine)
            else:
                mask_dilate = mask_dilate - union_mask

        elif self.typeLine == 2:
            mask_erode = cv2.erode(mask_dilate, kernel, iterations = 7 - self.sizeInputLine)
            mask_dilate = mask_dilate - mask_erode

            mask_erode_PSD = cv2.erode(maskPSD_dilate, kernel, iterations = 7 - self.sizeInputLine)
            maskPSD_dilate = maskPSD_dilate - mask_erode_PSD

            mask_erode_Axon = cv2.erode(maskAxon_dilate, kernel, iterations = 7 - self.sizeInputLine)
            maskAxon_dilate = maskAxon_dilate - mask_erode_Axon

        elif self.typeLine == 0:
            mask_dilate = self.SheathTear(mask_dilate)
            self.ThickeningOfMembraneShell(mask_dilate)

        # Добавление к изображению (маске) основных клеточных мембран
        draw_image[mask_dilate[:,:] == 255] = self.nowColor
        # Добавление к изображению (маске) мембран от PSD
        draw_image[maskPSD_dilate[:,:] == 255] = self.nowColor
        # Добавление к изображению (маске) мембран от Axon
        draw_image[maskAxon_dilate[:,:] == 255] = self.nowColor
        draw_image[maskAxonOreol[:,:] == 255] = self.nowColor

        if use_rand_color:
            mask_membrane = mask_dilate|maskPSD_dilate
            if self.typeLine == 0:
                mask_membrane = mask_membrane

            for i in range(500):
                temp_mask = np.zeros(draw_image.shape[0:2], np.uint8)
                rx = 2 * np.random.randint(0, 10) + 1
                ry = 2 * np.random.randint(0, 10) + 1
                y = np.random.randint(draw_image.shape[0])
                x = np.random.randint(draw_image.shape[1])

                cv2.ellipse(img = temp_mask,
                            center = [x, y],
                            axes = (rx, ry),
                            color = 255,
                            thickness = -1,
                            angle = np.random.randint(0, 180),
                            startAngle = 0,
                            endAngle = 360)

                change_color_mask = temp_mask&mask_membrane

                #cv2.imshow("mask", change_color_mask)
                #cv2.waitKey()

                min_color = PARAM['membrane_color_mean'] - PARAM['membrane_color_std']
                max_color = PARAM['membrane_color_mean'] + PARAM['membrane_color_std']

                # чтобы не было границ пятнами нужно ограничение от базового цвета
                change_color_val_max = 15
                change_color = min(max(self.nowColor[0] + int(random.gauss(0, change_color_val_max/3)), min_color), max_color)

                draw_image[change_color_mask==255]= (change_color, change_color, change_color)

        # в конце добавлять размазанные регионы
        if self.typeLine == 0:
            change_color = self.nowColor[0]+self.changeColorVal
            if use_rand_color:
                # расширение размазать, а маску немного подправить
                mask_sheath_tear_add_read = cv2.erode(self.mask_sheath_tear_add, kernel, iterations=2)
                draw_image[mask_sheath_tear_add_read==255] = (change_color, change_color, change_color)
                r = np.random.randint(3 ,4+1)*2+1
                blur_image = cv2.GaussianBlur(draw_image, (r,r), random.uniform(4, 6.5))
                
                draw_image[self.mask_sheath_tear_add==255] = blur_image[self.mask_sheath_tear_add==255]
            else:
                mark_mask_sheath_tear_add = cv2.erode(self.mask_sheath_tear_add, kernel, iterations=2)
                draw_image[mark_mask_sheath_tear_add==255] = (change_color, change_color, change_color)

        return draw_image

    def setDrawParam(self):
        self.nowColor = (self.color, self.color, self.color)
        self.changeColorVal = np.random.randint(-10, 20+1)

    def setMaskParam(self):
        self.nowColor = (255,255,255)
        self.changeColorVal = 0

    def DrawLayer(self, image):
        self.setDrawParam()
        return self.Draw(image, use_rand_color=True)

    def DrawMask(self, image):
        self.setMaskParam()
        mask = self.Draw(image)

        #kernel = np.array([[0, 0, 0],
        #                   [1, 1, 0],
        #                   [0, 1, 0]], dtype=np.uint8)

        #mask = cv2.dilate(mask, kernel)

        #kernel2 = np.array([[0, 1, 0],
        #                   [0, 1, 1],
        #                   [0, 0, 0]], dtype=np.uint8)

        #mask = cv2.erode(mask, kernel2)

        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def DrawUniqueArea(self, image):
        # Функция создающая маску с немного большим отступом для алгорима случайного размещения новых органнел без пересечения

        draw_image = image.copy()

        draw_image[self.labels[:,:] == -1] = (255, 255,255)
        draw_image = self.DrawMask(draw_image)

        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.uint8)

        draw_image = cv2.dilate(draw_image,kernel,iterations = 2)
        draw_image = cv2.erode(draw_image,kernel,iterations = 2)

        return draw_image

def testMembrane():
    q = None

    print("1/3 type. Press button 'Q' or 'q' to exit")
    while q != ord('q') and q != ord("Q"):

        color = np.random.randint(182, 200)

        img = np.full((512,512,3), (color,color,color), np.uint8)

        for i in range(0, 512, 32):
            img[i:i+15,:,:] += 30

        membrane = Membrane(img.shape[:2])

        img1 = membrane.Draw(img)

        mask = np.zeros((512,512,3), np.uint8)

        tecnicalMask = np.zeros((512,512,3), np.uint8)

        mask = membrane.DrawMask(mask)
        img2 = membrane.Draw(img)
        tecnicalMask = membrane.DrawUniqueArea(tecnicalMask)
        mask = cv2.resize(mask, (256, 256))
        img2 = cv2.resize(img2, (256, 256))
        img1 = cv2.resize(img1, (256, 256))
        cv2.imshow("img", img1)
        cv2.imshow("mask", mask)
        cv2.imshow("img2", img2)
        cv2.imshow("tecnicalMask", tecnicalMask)

        r = PARAM["main_radius_gausse_blur"]
        G = PARAM["main_sigma_gausse_blur"]
        Img = cv2.GaussianBlur(img1,(r*2+1,r*2+1), G)

        noisy = np.ones(img1.shape[:2], np.uint8)
        noisy = np.random.poisson(noisy)*PARAM['poisson_noise'] - PARAM['poisson_noise']/2

        Img = Img + cv2.merge([noisy, noisy, noisy])
        Img[Img < 0] = 0
        Img[Img > 255] = 255
        Img = Img.astype(np.uint8)

        cv2.imshow("add noise and blur", Img)

        q = cv2.waitKey()
    # return
    print("2/3 type. Press button 'Q' or 'q' to exit")
    q = 's'
    while q != ord('q') and q != ord("Q"):

        color = np.random.randint(182, 200)

        img = np.full((512,512,3), (color,color,color), np.uint8)

        for i in range(0, 512, 32):
            img[i:i+15,:,:] += 30


        axon = Axon()
        axon.NewPosition(256,256)


        membrane = Membrane(img.shape[:2], [axon])

        img1 = membrane.Draw(img)
        img1 = axon.Draw(img1)

        mask = np.zeros((512,512,3), np.uint8)

        tecnicalMask = np.zeros((512,512,3), np.uint8)

        mask = membrane.DrawMask(mask)
        img2 = membrane.Draw(img)
        img2 = axon.Draw(img2)

        tecnicalMask = membrane.DrawUniqueArea(tecnicalMask)

        cv2.imshow("img", img1)
        cv2.imshow("mask", mask)
        cv2.imshow("img2", img2)
        cv2.imshow("tecnicalMask", tecnicalMask)

        r = PARAM["main_radius_gausse_blur"]
        G = PARAM["main_sigma_gausse_blur"]
        Img = cv2.GaussianBlur(img1,(r*2+1,r*2+1), G)

        noisy = np.ones(img1.shape[:2], np.uint8)
        noisy = np.random.poisson(noisy)*PARAM['poisson_noise'] - PARAM['poisson_noise']/2

        Img = Img + cv2.merge([noisy, noisy, noisy])
        Img[Img < 0] = 0
        Img[Img > 255] = 255
        Img = Img.astype(np.uint8)

        cv2.imshow("add noise and blur", Img)

        q = cv2.waitKey()

    print("3/3 type. Press button 'Q' or 'q' to exit")
    q = 's'
    while q != ord('q') and q != ord("Q"):

        color = np.random.randint(182, 200)

        img = np.full((512,512,3), (color,color,color), np.uint8)

        for i in range(0, 512, 32):
            img[i:i+15,:,:] += 30


        PSD_elem = PSD()
        PSD_elem.NewPosition(256,128)


        PSD_elem2 = PSD()
        PSD_elem2.NewPosition(256,384)

        membrane = Membrane(img.shape[:2], [PSD_elem, PSD_elem2])

        img1 = membrane.Draw(img)
        img1 = PSD_elem.Draw(img1)

        mask = np.zeros((512,512,3), np.uint8)

        tecnicalMask = np.zeros((512,512,3), np.uint8)

        mask = membrane.DrawMask(mask)
        img2 = membrane.Draw(img)
        img2 = PSD_elem.Draw(img2)

        tecnicalMask = membrane.DrawUniqueArea(tecnicalMask)

        cv2.imshow("img", img1)
        cv2.imshow("mask", mask)
        cv2.imshow("img2", img2)
        cv2.imshow("tecnicalMask", tecnicalMask)

        r = PARAM["main_radius_gausse_blur"]
        G = PARAM["main_sigma_gausse_blur"]
        Img = cv2.GaussianBlur(img1,(r*2+1,r*2+1), G)

        noisy = np.ones(img1.shape[:2], np.uint8)
        noisy = np.random.poisson(noisy)*PARAM['pearson_noise'] - PARAM['pearson_noise']/2

        Img = Img + cv2.merge([noisy, noisy, noisy])
        Img[Img < 0] = 0
        Img[Img > 255] = 255
        Img = Img.astype(np.uint8)

        cv2.imshow("add noise and blur", Img)

        q = cv2.waitKey()

if __name__ == "__main__":
    testMembrane()
