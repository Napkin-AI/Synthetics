# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 00:55:41 2023

@author: Alexandra
"""
import numpy as np
import cv2
from settings import PARAM, uniform_int, normal_randint
import math

class PointsNoise:
    def __init__(self, size = (512,512)):
        # self.Points = []
        # self.PointsWithOffset = []
        self.sizeImage = size

    def Draw(self, image):
        draw_image = image.copy()
        # число линий
        count = np.random.randint(200, 500+1)
        # максимальная длина линий
        maxdist = np.random.randint(5, 45, 4)

        #  разбиваем область на 4 части в каждой будет свое направление линий
        xs = np.random.randint(0, self.sizeImage[0]/2, count)
        ys = np.random.randint(0, self.sizeImage[1]/2, count)

        xs[int(count/4):int(count/4) + int(count/4)] = np.random.randint(int(self.sizeImage[0]//2), self.sizeImage[0], int(count/4))
        xs[int(count*3/4):int(count*3/4) + int(count/4)] = np.random.randint(int(self.sizeImage[0]//2), self.sizeImage[0], int(count/4))
        ys[int(count/2): int(count/2) + int(count/2)] = np.random.randint(int(self.sizeImage[1]//2), self.sizeImage[1], int(count/2))

        #  направление линий
        vecx = np.random.rand(4)
        vecy = np.random.rand(4)*2 - 1

        #  забиваем дистанцию у каждой четверти свой максимум
        #  забиваем направления
        xd = np.ones(count)
        yd = np.ones(count)
        dist = np.ones(count)*2
        j = 0
        for i in range(0, int(count//4)*4, int(count//4)):
            a = np.asarray([vecx[j], vecy[j]])
            a = a/np.sqrt(a.dot(a))
            xd[i:i+int(count//4)] = a[0]
            yd[i:i+int(count//4)] = a[1]
            dist[i:i+int(count//4)] = np.random.randint(3, maxdist[j], int(count//4))
            j = j + 1

        xe = xs + xd * dist
        ye = ys + yd * dist

        xe = xe.astype(np.int32)
        ye = ye.astype(np.int32)

        xe[xe < 0] = 0
        ye[ye < 0] = 0

        xe[xe > self.sizeImage[0]] = self.sizeImage[0]
        ye[ye > self.sizeImage[1]] = self.sizeImage[1]

        #  толщина линий
        w = np.random.randint(1, PARAM['main_noise_w'], count)

        for i in range(0, count):    
            # c = np.random.randint(95, 140)
            c = normal_randint(
                PARAM['main_noise_color_mean'],
                PARAM['main_noise_color_std'])
            draw_image = cv2.line(draw_image, [xs[i], ys[i]], [xe[i], ye[i]], (c,c,c) , w[i])

        count = np.random.randint(300, 1000+1)
        # максимальная длина линий
        xs = np.random.randint(0, self.sizeImage[0], count)
        ys = np.random.randint(0, self.sizeImage[1], count)
        xd = np.random.randint(-5, 5, count)
        yd = np.random.randint(-5, 5, count)
        xe = np.minimum(np.ones(count) * self.sizeImage[0], np.maximum(np.zeros(count), xs + xd)).astype(np.int32)
        ye = np.minimum(np.ones(count) * self.sizeImage[1], np.maximum(np.zeros(count),ys + yd)).astype(np.int32)
        w = np.random.randint(1, PARAM['main_noise_w'], count)
        for i in range(0, count):    
            c = normal_randint(
                PARAM['main_noise_color_mean'],
                PARAM['main_noise_color_std'])
            draw_image = cv2.line(draw_image, [xs[i], ys[i]], [xe[i], ye[i]], (c,c,c) , w[i])

        return draw_image

def AddGaussianNoise(image, noisePower = 16):
    mean = 0

    gaussian = np.random.normal(mean, noisePower, image.shape[0:2])

    noise_image = np.zeros(image.shape, np.float32)

    if len(image.shape) == 2:
        noise_image = image + gaussian
    else:
        noise_image[:, :, 0] = image[:, :, 0] + gaussian
        noise_image[:, :, 1] = image[:, :, 1] + gaussian
        noise_image[:, :, 2] = image[:, :, 2] + gaussian

    noise_image[noise_image[:,:,:] < 0] = 0
    noise_image[noise_image[:,:,:] > 255] = 255

    noise_image = noise_image.astype(np.uint8)

    return noise_image

class SpamComponents:
    def __init__(self, compartmentsList, numberSpam = 10):
        '''
        I say "hello"
        This class adds spam or junk to the input image. At the same time, garbage is not superimposed on classes.

        sizeOverlap - offset from masks
        numberSpam - amount of added spam

        возможно стоит сделать параметризованный по типам класс (типо точка, полосочка, и т.д. по типу органнел) 
        чтобы на соседних слоях была та же картина
        '''
        self.type = "SpamComponents"
        self.sizeOverlap = 1
        self.numberSpam = numberSpam
        self.compartmentsList = compartmentsList
        
        self.mask_spam = None

    def CreateLine(self, len_line, max_y = 10, len_segment_min = 10, len_segment_max = 20):
        now_pos = -len_line//2
        y_s =  np.random.randint(-max_y, max_y+1)
        # задание первой точки линии
        points_segment = [[now_pos, y_s]]

        # добавление промежуточных участков
        len_segment = np.random.randint(len_segment_min, len_segment_max + 1)
        while now_pos+len_segment_min+len_segment_max < len_line//2:
            x_s = now_pos + len_segment
            y_s =  np.random.randint(-max_y, max_y+1)
            points_segment.append([x_s, y_s])

            now_pos = x_s
            len_segment = np.random.randint(len_segment_min, len_segment_max + 1)

        # добавление конечной точки
        x_s = len_line//2 - now_pos
        y_s =  np.random.randint(-max_y, max_y+1)
        points_segment.append([x_s, y_s])

        # поворот вокруг центральной точки
        change_angle = math.radians(np.random.randint(0, 360))
        work_points = []
        for point in points_segment:
            x_s = int(round(point[0] * math.cos(change_angle) - point[1] * math.sin(change_angle)))
            y_s = int(round(point[0] * math.sin(change_angle) + point[1] * math.cos(change_angle)))
            work_points.append([x_s,y_s])

        return work_points

    def DrawType1(self, image, mask, color, size_line):
        # выбор центра для паттерна
        x = np.random.randint(15, image.shape[1] - 15)
        y = np.random.randint(15, image.shape[0] - 15)

        angle = np.random.randint(0, 180)

        radius = np.random.randint(7, 22, 2)

        cv2.ellipse(img = image,
                    center = (x, y),
                    axes = radius,
                    color = (color, color, color),
                    thickness = size_line,
                    angle = angle,
                    startAngle = 0,
                    endAngle = 360)

        cv2.ellipse(img = mask,
                    center = (x, y),
                    axes = radius,
                    color = (255, 255,255),
                    thickness = size_line,
                    angle = angle,
                    startAngle = 0,
                    endAngle = 360)

        cv2.ellipse(img = mask,
                    center = (x, y),
                    axes = radius,
                    color = (255, 255,255),
                    thickness = -1,
                    angle = angle,
                    startAngle = 0,
                    endAngle = 360)
        return mask

    def DrawType2(self, image, mask, color, size_line):
        # выбор центра для паттерна
        x = np.random.randint(45, image.shape[1] - 45)
        y = np.random.randint(45, image.shape[0] - 45)

        # длина линии
        len_line = np.random.randint(20, 180 + 1)

        max_y = 10

        len_segment_max = 20
        len_segment_min = 10

        work_points = self.CreateLine(len_line, max_y, len_segment_min, len_segment_max)

        height_line = np.random.randint(2, 10+1)

        # смещение в центр паттерна
        out_list = np.array([[int(round(work_points[i][0] + x)), int(round(work_points[i][1] + y))] for i in range(len(work_points))])

        cv2.polylines(mask, [out_list], isClosed = False, color = (255,255,255), thickness = height_line + size_line, lineType = cv2.LINE_AA)
        cv2.polylines(mask, [out_list], isClosed = False, color = (0,0,0), thickness = height_line, lineType = cv2.LINE_AA)

        mask[mask[:,:,0] > 0] = (255,255,255)
        image[mask[:,:,0] == 255] = (color, color, color)

        #kernel = np.array([[0, 1, 0],
        #                  [1, 1, 1],
        #                   [0, 1, 0]], dtype=np.uint8)
        #mask = cv2.dilate(mask, kernel, iteration)
        return mask

    def DrawType3(self, image, mask):
        # выбор центра для паттерна
        x = np.random.randint(35, image.shape[1] - 35)
        y = np.random.randint(35, image.shape[0] - 35)

        angle = np.random.randint(0, 180)

        radius = np.random.randint(3, 14+1)

        color = np.random.randint(44,70+1)

        cv2.ellipse(img = image,
                    center = (x, y),
                    axes = (radius,radius),
                    color = (color, color, color),
                    thickness = -1,
                    angle = angle,
                    startAngle = 0,
                    endAngle = 360)

        cv2.ellipse(img = mask,
                    center = (x, y),
                    axes = (radius,radius),
                    color = (255, 255,255),
                    thickness = -1,
                    angle = angle,
                    startAngle = 0,
                    endAngle = 360)

        return mask

    def DrawType4(self, image, mask, size_line):
        # выбор центра для паттерна
        x = np.random.randint(50, image.shape[1] - 50)
        y = np.random.randint(50, image.shape[0] - 50)

        color = np.random.randint(44,70+1)
        # длина линии
        len_line = np.random.randint(50, 100 + 1)

        max_y = 5

        len_segment_max = 20
        len_segment_min = 10

        work_points = self.CreateLine(len_line, max_y, len_segment_min, len_segment_max)

        # смещение в центр паттерна
        out_list = np.array([[int(round(work_points[i][0] + x)), int(round(work_points[i][1] + y))] for i in range(len(work_points))])

        cv2.polylines(mask, [out_list], isClosed = False, color = (255,255,255), thickness = size_line+1, lineType = cv2.LINE_AA)

        mask[mask[:,:,0] > 0] = (255,255,255)
        image[mask[:,:,0] == 255] = (color, color, color)

        #kernel = np.array([[0, 1, 0],
        #                   [1, 1, 1],
        #                   [0, 1, 0]], dtype=np.uint8)
        #mask = cv2.dilate(mask, kernel, iterations = size_line+1)

        return mask

    def DrawAny(self, image, color, size_line):
        # маска для проверки на пересечение
        temp_mask = np.zeros(image.shape, np.uint8)

        random_val = np.random.randint(0, 25)
        # наиболее частые - это небольшие кружочки и полые полосочки
        # темные кружочки и линии - очень редкие
        if random_val < 5:
            type_pattern = 1    # type1 - small ellipse
        elif random_val < 22:
            type_pattern = 2    # type2 - line
        elif random_val == 22:
            type_pattern = 3    # type3 - black point
        else:
            type_pattern = 4    # type4 - black line

        # type1 - small ellipse
        if type_pattern == 1:
            temp_mask = self.DrawType1(image, temp_mask, color, size_line)
        # type2 - line
        elif type_pattern == 2:
            temp_mask = self.DrawType2(image, temp_mask, color, size_line)
        # type3 - black point
        elif type_pattern == 3:
            temp_mask = self.DrawType3(image, temp_mask)
        # type4 - black line
        elif type_pattern == 4:
            temp_mask = self.DrawType4(image, temp_mask, size_line)
        else:
            raise Exception("What's up in DrawAny from SpamComponents?")

        return image, temp_mask

    def DrawLayer(self, image):
        # создание маски для всех классов
        masks_classes = np.zeros(image.shape, np.uint8)

        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.uint8)
        # заполнение маски всеми классами
        for component in self.compartmentsList:
            if component.type != "SpamComponents" and\
               component.type != "Vesicles" and\
               component.type != "Axon" and\
               component.type != "PSD" and\
               component.type != "Membrane":
                masks_classes = component.DrawMask(masks_classes)

            elif component.type == "Vesicles" or component.type == "PSD":
                masks_classes_temp = np.zeros(image.shape, np.uint8)
                masks_classes_temp = component.DrawUniqueArea(masks_classes_temp) # не рисовать рядом (small_mode = False)
                masks_classes_temp = cv2.dilate(masks_classes_temp, kernel, iterations = 30) # вообще даже близко к ним не рисовать
                masks_classes = masks_classes | masks_classes_temp

            elif component.type == "Axon":
                masks_classes = component.DrawUniqueArea(masks_classes, small_mode = True)    # не рисовать внутри, но можно рядом

            elif component.type == "Membrane":
                masks_classes = component.DrawUniqueArea(masks_classes)
                sizeLine = component.sizeLine

        # отступ от маски
        masks_classes = cv2.dilate(masks_classes, kernel, self.sizeOverlap)

        iter = 0
        max_iter = 10000
        count = 0
        
        self.mask_spam = np.zeros(image.shape, np.uint8)

        while count < self.numberSpam and iter < max_iter:
            image_clone = image.copy()

            drawColor = normal_randint(PARAM['spam_color_mean'],
                                       PARAM['spam_color_std'])

            temp_image, temp_mask = self.DrawAny(image_clone, drawColor, sizeLine)

            if all(masks_classes[temp_mask[:,:,0] == 255, 0] == 0):
                # добавляем расширенную маску в общую
                masks_classes = masks_classes | cv2.dilate(temp_mask, kernel, iterations = 3)
                image = temp_image
                self.mask_spam = self.mask_spam|temp_mask
                count += 1

            iter += 1

        if iter == max_iter:
            print("I can't create spam : only", count, "is",  self.numberSpam)

        #cv2.imshow("spam_mask", self.mask_spam)
        #cv2.imshow("image", image)
        #cv2.waitKey()

        return image
        
    def DrawMask(self, image):
        if self.mask_spam:
            return self.mask_spam
        else:
            return
        
def CreateNoise(size_image, poisson_noise):
    noisy = np.ones((*size_image, 3), np.uint8)
    # apply Poisson noise to the array and scale it by the noise parameter
    noisy = np.random.poisson(noisy)*poisson_noise - poisson_noise
    # apply a 5x5 blur to the noisy image to smooth out high-frequency noise
    noisy = cv2.blur(noisy,(5,5))
    # create a second array of ones with the same shape and data type as the first one
    noisy2 = np.random.poisson(np.ones((*size_image, 3), np.uint8))*poisson_noise - poisson_noise
    # add the two noisy arrays together
    noisy = noisy + noisy2
    return noisy
