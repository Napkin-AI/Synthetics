# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 18:39:53 2023

@author: Alexandra
"""
import numpy as np
import cv2
import os
import datetime

def SaveGeneration(Img, MackPSD, MackAxon, MackMembrans, MackMito, MackMitoBoarder, MackVesicules, counter, dir_save = None, startIndex = 0, suffix_name = ""):
     if dir_save is not None:
         # Сохранение слоя и масок
         if not os.path.isdir(dir_save):
             print(f"create dir_save: {dir_save}")
             os.mkdir(dir_save)

         if not os.path.isdir(os.path.join(dir_save, "original")):
             print(f"create original: {os.path.join(dir_save, 'original')}")
             os.mkdir(os.path.join(dir_save, "original"))

         if not os.path.isdir(os.path.join(dir_save, "PSD")):
             print(f"create PSD: {os.path.join(dir_save, 'PSD')}")
             os.mkdir(os.path.join(dir_save, "PSD"))

         if not os.path.isdir(os.path.join(dir_save, "axon")):
             print(f"create axon: {os.path.join(dir_save, 'axon')}")
             os.mkdir(os.path.join(dir_save, "axon"))

         if not os.path.isdir(os.path.join(dir_save, "boundaries")):
             print(f"create boundaries: {os.path.join(dir_save, 'boundaries')}")
             os.mkdir(os.path.join(dir_save, "boundaries"))

         if not os.path.isdir(os.path.join(dir_save, "mitochondria")):
             print(f"create mitochondria: {os.path.join(dir_save, 'mitochondria')}")
             os.mkdir(os.path.join(dir_save, "mitochondria"))

         if not os.path.isdir(os.path.join(dir_save, "mitochondrial_boundaries")):
             print(f"create mitochondrial_boundaries: {os.path.join(dir_save, 'mitochondrial_boundaries')}")
             os.mkdir(os.path.join(dir_save, "mitochondrial_boundaries"))

         if not os.path.isdir(os.path.join(dir_save, "vesicles")):
             print(f"create vesicles: {os.path.join(dir_save, 'vesicles')}")
             os.mkdir(os.path.join(dir_save, "vesicles"))

         date = datetime.datetime.now().strftime(r'%Y_%m_%d')
         name = str(counter + startIndex) + "_" + date

         if len(suffix_name) > 0:
             name += "_" + suffix_name

         f = lambda image: cv2.resize(image, (256, 256))
         Img = f(Img)
         MackPSD = f(MackPSD)
         MackAxon = f(MackAxon)
         MackMembrans = f (MackMembrans)
         MackVesicules = f(MackVesicules)
         MackMitoBoarder = f(MackMitoBoarder)
         MackMito = f(MackMito)
         # coхранение слоя
         cv2.imwrite(os.path.join(dir_save, "original", name + ".png"), cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY))
         # coхранение маски
         cv2.imwrite(os.path.join(dir_save, "PSD", name + ".png"), cv2.cvtColor(MackPSD, cv2.COLOR_RGB2GRAY))
         # coхранение маски
         cv2.imwrite(os.path.join(dir_save, "axon", name + ".png"), cv2.cvtColor(MackAxon, cv2.COLOR_RGB2GRAY))
         # coхранение маски
         cv2.imwrite(os.path.join(dir_save, "boundaries", name + ".png"), cv2.cvtColor(MackMembrans, cv2.COLOR_RGB2GRAY))
         # coхранение маски
         cv2.imwrite(os.path.join(dir_save, "mitochondria", name + ".png"), cv2.cvtColor(MackMito, cv2.COLOR_RGB2GRAY))
         # coхранение маски
         cv2.imwrite(os.path.join(dir_save, "vesicles", name + ".png"), cv2.cvtColor(MackVesicules, cv2.COLOR_RGB2GRAY))
         # coхранение маски
         cv2.imwrite(os.path.join(dir_save, "mitochondrial_boundaries", name + ".png"), cv2.cvtColor(MackMitoBoarder, cv2.COLOR_RGB2GRAY))