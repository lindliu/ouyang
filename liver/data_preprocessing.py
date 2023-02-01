#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:49:39 2023

@author: dliu
"""

import pandas as pd
import matplotlib.pyplot as plt

path = '../data/胰腺癌术后肝转移191例.xlsx'
data = pd.read_excel(path)