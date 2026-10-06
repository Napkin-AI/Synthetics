import numpy as np
import cv2
import os
import datetime
import math
from settings import PARAM, DEBUG_MODE, uniform_int, normal_randint
import skimage
import random

import albumentations as albu

if __name__ == "__main__":
    import sys

    sys.path.append('.')

from src.organells.location import *
from src.organells.axon import Axon
from src.organells.PSD import PSD
from src.organells.vesicles import Vesicles
from src.organells.mitohondrion import Mitohondrion
from src.organells.membrane import Membrane
from src.organells.union_organels import Vesicles_and_PSD

from src.container.spline import *
from src.container.subclass import *

from src.container.output import SaveGeneration

from src.container.noises import PointsNoise, SpamComponents, CreateNoise


class Form:
    def __init__(self, size=(512, 512)):

        color = normal_randint(
            PARAM['main_color_mean'],
            PARAM['main_color_std'])
        self.backgroundСolor = (color, color, color)  # выбор цвета фона

        self.nowViewType = 'layer'
        self.sizeImage = size

        # self.currView = np.zeros((*self.sizeImage, 3), np.uint8)
        # self.backgroundImg = np.zeros((*self.sizeImage, 3), np.uint8)

        # self.technicalMackAllcompanenst = np.zeros((*self.sizeImage, 3), np.uint8)

        self.compartmentsList = []

    def CheckOverlapNewElement(self, all_mask, check_mask):
        if any(all_mask[check_mask[:, :, 0] != 0, 0] != 0):
            return True
        return False

    def GetNewPosition(self, x_limit, y_limit, indexes_choise=None):
        start_x, end_x = x_limit
        start_y, end_y = y_limit

        if indexes_choise is None:
            x = np.random.randint(start_x, end_x)
            y = np.random.randint(start_y, end_y)

        else:
            index_сhoise = random.choice(range(len(indexes_choise[0])))
            x, y = indexes_choise[1][index_сhoise], indexes_choise[0][index_сhoise]

        return x, y

    def AddNewElementWithoutOverlapWithMask(self, technicalMackAllcompanenst, newComponent):
        # максимальное количество попыток добавить
        MAX_ITER = 150

        # small items should be not near the edge
        offset = 32 if newComponent.type == "PSD" or newComponent.type == "Vesicles_and_PSD" else 5

        lim_x_pos = self.sizeImage[1] - offset
        lim_y_pos = self.sizeImage[0] - offset

        # если возможно, то пытаемся брать незанятые области с отступом от маски
        mask_overlap = 16
        choise_pixel = np.full(technicalMackAllcompanenst.shape, 255, np.uint8) - technicalMackAllcompanenst
        choise_pixel[:offset, :] = 0
        choise_pixel[lim_y_pos:, :] = 0
        choise_pixel[:, :offset] = 0
        choise_pixel[:, lim_x_pos:] = 0

        kernel = np.ones((3, 3), 'uint8')
        choise_pixel = cv2.erode(choise_pixel, kernel, iterations=mask_overlap)
        indexes_choise = np.where(choise_pixel == 255)

        counter = 0
        while True:
            # рисование маски нового элемента. Выбирается позиция, угол и рисуется.
            x, y = self.GetNewPosition((offset, lim_x_pos), (offset, lim_y_pos), indexes_choise)
            newComponent.NewPosition(x, y)
            newComponent.setRandomAngle(0, 90)
            checkNewImage = newComponent.DrawUniqueArea(np.zeros((*self.sizeImage, 3), np.uint8))

            counter += 1
            if counter >= MAX_ITER:
                break
            # проверка условия для продолжения цикла
            if not self.CheckOverlapNewElement(technicalMackAllcompanenst, checkNewImage):
                break

        # Добавлено исключение чтобы элементы не накладывались друг на друга
        if counter == MAX_ITER:
            raise Exception("Can't add unique position new element")

    def addNewElementIntoImage(self, compartmentsList, newComponent, technicalMackAllcompanenst=None):
        # на всякий случай проверяем newComponent
        if newComponent is not None:
            # иключение на случай если не получилось добавить без пересечения или
            # что-то пошло не так при попытке рисования масок
            try:
                # если не дали тех.маску текущих компонентов, то сделаем из списка компонентов
                if technicalMackAllcompanenst is None:
                    checkImage = np.zeros((*self.sizeImage, 3), np.uint8)
                    for component in compartmentsList:
                        checkImage = component.DrawUniqueArea(checkImage)

                # иначе используем тех.маску
                else:
                    checkImage = technicalMackAllcompanenst

                # нет смысла проверять мембраны, так как они касаются PSD и будет перекрытие
                if newComponent.type != "Membrane":
                    self.AddNewElementWithoutOverlapWithMask(checkImage, newComponent);

                compartmentsList.append(newComponent)

                checkImage = newComponent.DrawUniqueArea(checkImage)  # рисуется новый элемент на тех.маску
                newComponent = None

                return checkImage

            except Exception as ex:
                print(ex)
                # print(f"{type(ex).__name__} at line {ex.__traceback__.tb_lineno} of {__file__}: {ex}")

        return technicalMackAllcompanenst

    def get_count(self, max_count):
        return np.random.randint(1, max_count + 1) if max_count > 0 else 0

    class_constructors = {
        "Axon": Axon,
        "Mitohondrion": Mitohondrion,
        "Vesicles": Vesicles,
        "PSD": PSD,
        "Vesicles_and_PSD": Vesicles_and_PSD
    }

    def generate_lists(self, element_dict, split=True):
        '''
        Generate list or lists of compartment which will be used as order to adding
        this compartment into layer

        Parameters
        ----------
        element_dict : dictionary
            key: compartment type as string
            value: number of compartment in layer.
        split : bool
            Need we 2 lists or one. The default is True.

        Returns
        -------
        list
            List of lists of compartment order.
            For example: [["PSD","Axon", "PSD", "Mitohondrion", "Vesicles"], ["Mitohondrion", "Vesicles"]]
            "PSD" can be only in the first list
        '''

        if split:
            element_dict1 = {}
            element_dict2 = {}
            for key in element_dict:
                if key != "PSD":
                    # добавить PSD и рядом с везикулой
                    if key == "Vesicles" and element_dict["Vesicles"] > 0 and element_dict["PSD"] > 0:
                        num_union_elements = np.random.randint(min(element_dict["Vesicles"], element_dict["PSD"]) // 2,
                                                               min(element_dict["Vesicles"], element_dict["PSD"]) + 1)

                        element_dict["Vesicles"] -= num_union_elements
                        element_dict["PSD"] -= num_union_elements

                        element_dict1["Vesicles_and_PSD"] = num_union_elements

                    element_dict1[key] = np.random.randint(0, element_dict[key] + 1)
                    element_dict2[key] = element_dict[key] - element_dict1[key]

            element_dict1["PSD"] = element_dict["PSD"]

            element_list1 = []
            for key in element_dict1:
                element_list1.extend([key] * element_dict1[key])
            element_list2 = []
            for key in element_dict2:
                element_list2.extend([key] * element_dict2[key])
            random.shuffle(element_list1)
            random.shuffle(element_list2)
            return [element_list1, element_list2]
        else:
            element_list = []
            for key in element_dict:
                element_list.extend([key] * element_dict[key])
            random.shuffle(element_list)
            return [element_list]

    def createListGeneration(self, max_count_PSD=0, max_count_Axon=0, max_count_Vesicles=0, max_count_Mitohondrion=0,
                             max_count_spam=5):
        RetList = []

        count_PSD = self.get_count(max_count_PSD)
        count_Axon = self.get_count(max_count_Axon)
        count_Vesicles = self.get_count(max_count_Vesicles)
        count_Mitohondrion = self.get_count(max_count_Mitohondrion)

        element_dict = {"Axon": count_Axon, "Mitohondrion": count_Mitohondrion,
                        "Vesicles": count_Vesicles, "PSD": count_PSD}

        element_list = self.generate_lists(element_dict)

        # тех.маска уже добавленных элементов
        UnionMask = np.zeros((*self.sizeImage, 3), np.uint8)

        # защита на случай если не существует класса element
        try:
            # добавление элементов перед добавлением мембран
            for element in element_list[0]:
                if element in self.class_constructors:
                    newElement = self.class_constructors[element]()
                    # функция, добавляющая новый элемент без пересечения с имеющимися по общей маске (также рисует его в общую маску)
                    UnionMask = self.addNewElementIntoImage(RetList, newElement, UnionMask)
                else:
                    raise Exception("No such class exist: " + element)

            # cv2.imshow("testMask", UnionMask)
            # cv2.waitKey()

            # DRAW MEMBRANES
            # для мембран просто добавление в список и отрисовка общей маски
            UnionMask = self.addNewElementIntoImage(RetList, Membrane(self.sizeImage, RetList.copy()), UnionMask)

            # добавление элементов после добавления мембран для более плотного расположения
            for element in element_list[1]:
                if element in self.class_constructors:
                    newElement = self.class_constructors[element]()
                    # функция, добавляющая новый элемент без пересечения с имеющимися по общей маске (также рисует его в общую маску)
                    UnionMask = self.addNewElementIntoImage(RetList, newElement, UnionMask)
                else:
                    print("No such class exist: " + element)
                    raise Exception("No such class exist: " + element)

            RetList.append(SpamComponents(RetList.copy(), max_count_spam))

        except Exception as ex:
            import traceback
            print(ex)
            print(f"{type(ex).__name__}\nat line {ex.__traceback__.tb_lineno}\nof {__file__}:\n{ex}")
            print(''.join(traceback.TracebackException.from_exception(ex).format()))

        RetListWithoutUnionClasses = []
        for companent in RetList:
            if companent.type == "Vesicles_and_PSD":
                RetListWithoutUnionClasses.append(companent.PSD)
                RetListWithoutUnionClasses.append(companent.vesicles)
            else:
                RetListWithoutUnionClasses.append(companent)

        return RetListWithoutUnionClasses

    def createListGenerationWithStartMembrane(self, max_count_PSD=0, max_count_Axon=0, max_count_Vesicles=0,
                                              max_count_Mitohondrion=0, max_count_spam=5):

        RetList = []

        count_PSD = self.get_count(max_count_PSD)
        count_Axon = self.get_count(max_count_Axon)
        count_Vesicles = self.get_count(max_count_Vesicles)
        count_Mitohondrion = self.get_count(max_count_Mitohondrion)

        element_dict = {"Axon": count_Axon, "Mitohondrion": count_Mitohondrion,
                        "Vesicles": count_Vesicles, "PSD": count_PSD}

        element_list = self.generate_lists(element_dict, False)

        # DRAW MEMBRANES
        UnionMask = self.addNewElementIntoImage(RetList, Membrane(self.sizeImage))

        try:
            for element in element_list[0]:
                if element in self.class_constructors:
                    newElement = self.class_constructors[element]()
                    UnionMask = self.addNewElementIntoImage(RetList, newElement)
                else:
                    raise Exception("No such class exist: " + element)

            RetList.append(SpamComponents(RetList.copy(), max_count_spam))
        except Exception as ex:
            print(ex)

        return RetList

    def createBackground(self, size_image, backgroundСolor):
        layer = np.full((*size_image, 3), backgroundСolor, np.uint8)
        point_noise = PointsNoise(size_image)
        layer = point_noise.Draw(layer)
        return layer

    @staticmethod
    def GaussianBlurWithOutMask(layer, list_masks):
        r = PARAM["background_radius_gausse_blur"]
        G = PARAM["background_sigma_gausse_blur"]
        kernel = (r * 2 + 1, r * 2 + 1)
        maskPSD, maskAxon, maskMembrans, maskMito, maskMitoBoarder, maskVesicules = list_masks
        all_masks = maskPSD | maskAxon | maskMembrans | maskMito | maskMitoBoarder | maskVesicules

        # размытия
        blur_main_layer = cv2.GaussianBlur(layer, kernel, G)
        blur_main_layer[all_masks == 255] = layer[all_masks == 255]
        return blur_main_layer

    @staticmethod
    def GaussianBlurOneCompanent(layer, component):
        # индивидуальное размытие внутренности, границы и области вокгуг каждой компоненты
        boarder_mask = None

        ############################################## добавить рандому на значения ###########################################
        input_r = PARAM["main_input_radius_gausse_blur"] + np.random.randint(-1, 1 + 1)
        input_G = PARAM["main_input_sigma_gausse_blur"] + random.uniform(-1, 1.00000001)

        output_r = PARAM["main_output_radius_gausse_blur"] + np.random.randint(-1, 1 + 1)
        output_G = PARAM["main_output_sigma_gausse_blur"] + random.uniform(-1, 1.00000001)

        big_output_r = PARAM["main_big_output_radius_gausse_blur"] + np.random.randint(-1, 1 + 1)
        big_output_G = PARAM["main_big_output_sigma_gausse_blur"] + random.uniform(-1, 1.00000001)

        boarder_r = 2  # возможно лучше не рандомить
        boarder_G = 1  # возможно лучше не рандомить

        ######################################### возможно внести в параметры ################################
        center_boarder_overlap = np.random.randint(1, 2 + 1)
        output_add_zone = np.random.randint(2, 4 + 1)
        big_output_add_zone = output_add_zone + np.random.randint(1, 3 + 1)

        if component.type == "Axon":
            boarder_thickness = np.random.randint(2, 3 + 1)

        # сильнее размыть вокруг PSD
        elif component.type == "PSD":
            boarder_thickness = np.random.randint(2, 3 + 1)
            output_add_zone += np.random.randint(1, 3 + 1)
            big_output_add_zone += np.random.randint(2, 4 + 1)

            output_G += random.uniform(0.75, 1.25)
            big_output_G += random.uniform(0.75, 1.75)

            output_r += np.random.randint(1, 2 + 1)
            big_output_r += np.random.randint(3, 5 + 1)


        elif component.type == "Membrane":
            boarder_thickness = np.random.randint(0, 2 + 1)

            if random.random() < 0.05:
                output_G += random.uniform(0, 1)
                big_output_G += random.uniform(0, 2)
                boarder_G += random.uniform(0, 1)
                input_G += random.uniform(0, 2)
            else:
                output_G -= random.uniform(0, 1.00000001)
                big_output_G -= random.uniform(0, 1.00000001)
                boarder_G -= random.uniform(0, 0.5)
                input_G -= random.uniform(0, 1.00000001)

            boarder_r *= 2
            big_output_r = int(big_output_r * 1.5)
        elif component.type == "Vesicles":
            # сильнее размыть везикулы
            boarder_thickness = np.random.randint(1, 2 + 1)
            output_add_zone += np.random.randint(1, 3 + 1)
            big_output_add_zone += np.random.randint(2, 4 + 1)
            center_boarder_overlap = np.random.randint(1, 6 + 1)

            output_G += random.uniform(0.5, 1.0)  # TODO change blur here
            big_output_G += random.uniform(0.75, 1.75)
            input_G += random.uniform(-0.5, 0.2)

            output_r += np.random.randint(1, 2 + 1)
            big_output_r += np.random.randint(3, 5 + 1)

            boarder_r *= 3
            big_output_r *= 2

        elif component.type == "Mitohondrion":
            center_boarder_overlap = 10  # off mask
            input_G += random.uniform(0, 1.5)
            boarder_G += random.uniform(-0.3, 0.1)

            boarder_mask = component.DrawMaskBoarder(np.zeros((*(layer.shape[:2]), 3), np.uint8))

        elif component.type == "SpamComponents":
            return layer, None
        else:
            print(f"ERROR: no type {component.type}")

        # check
        input_G = max(input_G, 0.5)
        boarder_G = max(boarder_G, 0.5)
        output_G = max(output_G, 0.5)
        big_output_G = max(big_output_G, 0.5)

        # размытие
        # input_r *= 3
        # output_r *= 3
        # big_output_r *= 3
        # boarder_r *= 3

        input_kernel = (input_r * 2 + 1, input_r * 2 + 1)
        output_kernel = (output_r * 2 + 1, output_r * 2 + 1)
        big_output_kernel = (big_output_r * 2 + 1, big_output_r * 2 + 1)
        boarder_kernel = (boarder_r * 2 + 1, boarder_r * 2 + 1)

        input_blur_layer = cv2.GaussianBlur(layer, input_kernel, input_G)
        output_blur_layer = cv2.GaussianBlur(layer, output_kernel,
                                             output_G)  # сглаживание по интенсивности преграничной области
        big_output_blur_layer = cv2.GaussianBlur(layer, big_output_kernel,
                                                 big_output_G)  # сильное сглаживание по интенсивности около компоненты
        boarder_blur_layer = cv2.GaussianBlur(layer, boarder_kernel, boarder_G)

        # маски
        input_mask_kernel = np.array([[0, 1, 0],
                                      [1, 1, 1],
                                      [0, 1, 0]], dtype=np.uint8)
        mask = component.DrawMask(np.zeros((*(layer.shape[:2]), 3), np.uint8))

        if boarder_mask is None:
            input_mask = cv2.erode(mask, input_mask_kernel, iterations=boarder_thickness)
            boarder_mask = mask - input_mask
        else:
            input_mask = mask - boarder_mask

        center_boundaries = cv2.erode(boarder_mask, input_mask_kernel, iterations=center_boarder_overlap)

        output_mask_with_component = cv2.dilate(mask, input_mask_kernel, iterations=output_add_zone)
        output_mask = output_mask_with_component - mask
        big_output_mask = cv2.dilate(output_mask_with_component, input_mask_kernel,
                                     iterations=big_output_add_zone) - output_mask_with_component

        # paint
        result_layer = layer.copy()
        result_layer[input_mask == 255] = input_blur_layer[input_mask == 255]
        result_layer[output_mask == 255] = output_blur_layer[output_mask == 255]
        result_layer[big_output_mask == 255] = big_output_blur_layer[big_output_mask == 255]
        result_layer[boarder_mask == 255] = boarder_blur_layer[boarder_mask == 255]
        result_layer[center_boundaries == 255] = layer[center_boundaries == 255]
        return result_layer, mask

    def fake_3_layers(self, ListGeneration, counter, dir_save, startIndex, size_overlap=5):
        # список для хранения списков компонетнов предыдущего, текущего и следующего слоёв
        # нужен для того, чтобы 1 и тот же компонент был сдвинут по прямой

        ListListGeneration = []
        for i in range(-1, 2):
            ListListGeneration.append([])

        for k, component in enumerate(ListGeneration):
            # создание единичного случайного вектора для смещения
            x = np.random.random() * 2 - 1  # to -1:1
            y = np.random.random() * 2 - 1  # to -1:1
            len_direction = math.sqrt(x ** 2 + y ** 2)
            direction = [x / len_direction, y / len_direction]

            for index, i in enumerate(range(-1, 2)):  # 1-индексация по массиву, 2-взвешанное направление по прямой
                if component.type != "Membrane" and component.type != "SpamComponents":
                    copy_component = component.copy()
                    copy_component.NewPosition(component.centerPoint[0] + int(round(i * direction[0] * size_overlap)),
                                               component.centerPoint[1] + int(round(i * direction[1] * size_overlap)))
                    ListListGeneration[index].append(copy_component)
                elif component.type == "Membrane":
                    new_membrane = Membrane(self.sizeImage, ListListGeneration[index].copy())
                    new_membrane.copy_main_param(component)
                    ListListGeneration[index].append(new_membrane)
                else:
                    new_spam = SpamComponents(ListListGeneration[index].copy(), component.numberSpam)
                    ListListGeneration[index].append(new_spam)

        for j, ListGenerationCopy in enumerate(ListListGeneration):
            fake_suffix = str(j)  # 0, 1, 2...

            draws_result = self.DrawsLayerAndMask(ListGenerationCopy)

            Img, MackPSD, MackAxon, MackMembrans, MackMito, MackMitoBoarder, MackVesicules = draws_result

            SaveGeneration(Img, MackPSD, MackAxon, MackMembrans, MackMito, MackMitoBoarder, MackVesicules, counter,
                           dir_save, startIndex, fake_suffix)

    def DrawsLayerAndMask(self, ListComponents):
        # рисование слоя и фона к нему
        layer = self.createBackground(self.sizeImage, self.backgroundСolor)

        # выделяем память под маски
        maskAxon = np.zeros((*self.sizeImage, 3), np.uint8)
        maskPSD = np.zeros((*self.sizeImage, 3), np.uint8)
        maskMito = np.zeros((*self.sizeImage, 3), np.uint8)
        maskMitoBoarder = np.zeros((*self.sizeImage, 3), np.uint8)
        maskMembrans = np.zeros((*self.sizeImage, 3), np.uint8)
        maskVesicules = np.zeros((*self.sizeImage, 3), np.uint8)

        # рисовка в соответствующее изображение
        '''
        for component in ListComponents:
            layer = component.DrawLayer(layer)

            if component.type == "PSD":
                maskPSD = component.DrawMask(maskPSD)

            elif component.type == "Axon":
                maskAxon = component.DrawMask(maskAxon)

            elif component.type == "Membrane":
                maskMembrans = component.DrawMask(maskMembrans)

            elif component.type == "Mitohondrion":
                maskMito = component.DrawMask(maskMito)
                maskMitoBoarder = component.DrawMaskBoarder(maskMitoBoarder)

            elif component.type ==  "Vesicles":
                maskVesicules = component.DrawMask(maskVesicules)

            elif component.type == "SpamComponents":
                pass
            else:
                print(f"ERROR: no type {component.type}")
        '''

        for component in ListComponents:
            layer = component.DrawLayer(layer)
            layer, mask_component = self.GaussianBlurOneCompanent(layer, component)

            if component.type == "PSD":
                maskPSD = maskPSD | mask_component
            elif component.type == "Axon":
                maskAxon = maskAxon | mask_component
            elif component.type == "Membrane":
                maskMembrans = maskMembrans | mask_component
            elif component.type == "Mitohondrion":
                maskMito = maskMito | mask_component
                maskMitoBoarder = component.DrawMaskBoarder(maskMitoBoarder)
            elif component.type == "Vesicles":
                maskVesicules = maskVesicules | mask_component
            elif component.type == "SpamComponents":
                pass
            else:
                print(f"ERROR: no type {component.type}")

        # добавление общего размытия
        layer = self.GaussianBlurWithOutMask(layer=layer,
                                             list_masks=[maskPSD,
                                                         maskAxon,
                                                         maskMembrans,
                                                         maskMito,
                                                         maskMitoBoarder,
                                                         maskVesicules]
                                             )
        # добавление шума
        noisy = CreateNoise(self.sizeImage, PARAM['poisson_noise'])
        # add the noise to the image layer
        layer = layer + noisy
        layer = np.clip(layer, 0, 255)
        layer = layer.astype(np.uint8)

        return layer, maskPSD, maskAxon, maskMembrans, maskMito, maskMitoBoarder, maskVesicules

    def StartGeneration(self, count_img=100, count_PSD=3, count_Axon=1, count_Vesicles=3, count_Mitohondrion=3,
                        dir_save=None, startIndex=0, max_count_spam=5):
        # Цикличная генерация
        ArrLayers = []

        for counter in range(count_img):
            print(f"{counter + 1} generation img for {count_img}")

            # создаю новый список для каждой генерации
            ListGeneration = self.createListGeneration(count_PSD, count_Axon, count_Vesicles, count_Mitohondrion,
                                                       max_count_spam=max_count_spam)

            # выбор цвета фона
            color = normal_randint(
                PARAM['main_color_mean'],
                PARAM['main_color_std'])
            self.backgroundСolor = (color, color, color)

            layer, maskPSD, maskAxon, maskMembrans, maskMito, maskMitoBoarder, maskVesicules = self.DrawsLayerAndMask(
                ListGeneration)

            # ArrLayers.append([layer, maskPSD, maskAxon, maskMembrans, maskMito, maskMitoBoarder, maskVesicules])

            SaveGeneration(layer, maskPSD, maskAxon, maskMembrans, maskMito, maskMitoBoarder, maskVesicules, counter,
                           dir_save, startIndex)

        return ArrLayers

    def StartFake3LayerGeneration(self, count_img=100, count_PSD=3, count_Axon=1, count_Vesicles=3,
                                  count_Mitohondrion=3, dir_save=None, startIndex=0, max_count_spam=5):
        # Цикличная генерация
        ArrLayers = []

        for counter_ in range(count_img):
            counter = counter_ + 325

            print(f"{counter + 1} generation img for {count_img}")

            # создаю новый список для каждой генерации
            ListGeneration = self.createListGeneration(count_PSD, count_Axon, count_Vesicles, count_Mitohondrion,
                                                       max_count_spam=max_count_spam)

            # выбор цвета фона для 1 пачки
            color = normal_randint(
                PARAM['main_color_mean'],
                PARAM['main_color_std'])
            self.backgroundСolor = (color, color, color)

            self.fake_3_layers(ListGeneration, counter, dir_save, startIndex, 5)

        return ArrLayers
