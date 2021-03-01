# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 07:05:44 2021

@author: Yitao Yu
"""
import numpy as np

W, H = 960, 540
F = 270
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])

paramsdic = {"K":K, 
          "width":960,"height": 540,
          "lowlight":False,
          "featureextract":"evenCorners",
          "maxfeatures": 2000,"n":4,
          "maxdepth":500
          }