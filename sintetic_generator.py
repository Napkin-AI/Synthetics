import numpy as np
import cv2

from src.organells.axon import Axon, testAxon
from src.organells.PSD import PSD, testPSD
from src.organells.vesicles import Vesicles, testVesicles
from src.organells.mitohondrion import Mitohondrion, testMitohondrion
from src.organells.membrane import Membrane, testMembrane

from src.organells.union_organels import testUnion_PSD_ves

from src.container.main_field import Form


#testAxon()
#testPSD()
#testVesicles()
#testMitohondrion()
#testMembrane()

#testUnion_PSD_ves()


#size_image = (256, 256)
size_image = (512, 512)
#size_image = (128, 128)
forma = Form(size_image)

# количество элементов на изображении
max_count_PSD = 5
max_count_Axon = 1
max_count_Vesicles = 3
max_count_Mitohondrion = 4

number_generation = 200
ArrLayers = forma.StartGeneration(number_generation, max_count_PSD, max_count_Axon, max_count_Vesicles, max_count_Mitohondrion,\
                                  #dir_save = "dataset/synthetic_dataset10/", startIndex = 0, max_count_spam = 5)
                                  dir_save = "dataset/test_dataset_no_noise_kasthuri_512_v2_300/", startIndex = 100, max_count_spam = 50)
#ArrLayers = [[Img, MackPSD, MackAxon, MackMembrans, MackMito, MackMitoBoarder, MackVesicules], ...]

#number_generation = 20
#ArrLayers = forma.StartFake3LayerGeneration(number_generation, max_count_PSD, max_count_Axon, max_count_Vesicles, max_count_Mitohondrion,\
#                                  dir_save = "dataset/test_gen_3", startIndex = 0)
