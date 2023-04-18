#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 01:09:27 2022

@author: dliu
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols



#### 
def anova_subgroup(group1, group2):
    t_pos_1, t_neg_1, ut_pos_1, ut_neg_1 = group1
    t_pos_2, t_neg_2, ut_pos_2, ut_neg_2 = group2
    
    
    num = sum([*group1, *group2])
    cum = np.cumsum([*group1, *group2])
    target = np.ones(num)
    target[:cum[0]] = 1
    target[cum[0]:cum[1]] = 0
    target[cum[1]:cum[2]] = 1
    target[cum[2]:cum[3]] = 0
    target[cum[3]:cum[4]] = 1
    target[cum[4]:cum[5]] = 0
    target[cum[5]:cum[6]] = 1
    target[cum[6]:cum[7]] = 0
    
    df = pd.DataFrame({'AA': np.r_[np.repeat(['trans'],sum([t_pos_1, t_neg_1])),np.repeat(['untrans'],sum([ut_pos_1, ut_neg_1])),\
                                   np.repeat(['trans'],sum([t_pos_2, t_neg_2])),np.repeat(['untrans'],sum([ut_pos_2, ut_neg_2]))],
                       'BB': np.r_[np.repeat(['group1'],sum([t_pos_1, t_neg_1, ut_pos_1, ut_neg_1])),\
                                   np.repeat(['group2'],sum([t_pos_2, t_neg_2, ut_pos_2, ut_neg_2]))],
                       'Target': target})
    
    # model = ols('Target ~ C(AA) + C(BB) + C(AA):C(BB)', data=df).fit()
    model = ols('Target ~ AA + BB + AA*BB', data=df).fit()
    
    return print(sm.stats.anova_lm(model, typ=2))


age1 = [63, 4, 177, 19]
age2 = [21, 2, 71, 3]
anova_subgroup(age1, age2)

sex1 = [30, 3, 90, 13]
sex2 = [54, 3, 158, 9]
anova_subgroup(sex1, sex2)

KPS1 = [51, 4, 169, 12]
KPS2 = [33, 2, 79, 10]
anova_subgroup(KPS1, KPS2)

PM1 = [21, 0, 68, 6]
PM2 = [63, 6, 180, 16]
anova_subgroup(PM1, PM2)

BM1 = [65, 4, 229, 22]
BM2 = [19, 2, 19, 0]
anova_subgroup(BM1, BM2)

PTL1 = [25, 0, 62, 6]
PTL2 = [59, 6, 186, 16]
anova_subgroup(PTL1, PTL2)

PTS1 = [43, 2, 118, 10]
PTS2 = [30, 4, 100, 11]
anova_subgroup(PTS1, PTS2)

RLN1 = [60, 6, 179, 17]
RLN2 = [24, 0, 69, 5]
anova_subgroup(RLN1, RLN2)

ELI1 = [60, 3, 182, 19]
ELI2 = [24, 3, 66, 3]
anova_subgroup(ELI1, ELI2)

Asc1 = [18, 1, 68, 6]
Asc2 = [66, 5, 180, 16]
anova_subgroup(Asc1, Asc2)

WLM1 = [30, 4, 80, 8]
WLM2 = [54, 2, 168, 14]
anova_subgroup(WLM1, WLM2)

LD1 = [54, 4, 180, 14]
LD2 = [21, 2, 53, 3]
anova_subgroup(LD1, LD2)

AL1 = [76, 6, 220, 20]
AL2 = [8, 0, 28, 2]
anova_subgroup(AL1, AL2)

SCT1 = [33, 1, 102, 5]
SCT2 = [51, 5, 146, 17]
anova_subgroup(SCT1, SCT2)







def anova_subgroup_3(group1, group2, group3):
    t_pos_1, t_neg_1, ut_pos_1, ut_neg_1 = group1
    t_pos_2, t_neg_2, ut_pos_2, ut_neg_2 = group2
    t_pos_3, t_neg_3, ut_pos_3, ut_neg_3 = group3

    
    num = sum([*group1, *group2, *group3])
    cum = np.cumsum([*group1, *group2, *group3])
    target = np.ones(num)
    target[:cum[0]] = 1
    target[cum[0]:cum[1]] = 0
    target[cum[1]:cum[2]] = 1
    target[cum[2]:cum[3]] = 0
    target[cum[3]:cum[4]] = 1
    target[cum[4]:cum[5]] = 0
    target[cum[5]:cum[6]] = 1
    target[cum[6]:cum[7]] = 0
    target[cum[7]:cum[8]] = 1
    target[cum[8]:cum[9]] = 0
    target[cum[9]:cum[10]] = 1
    target[cum[10]:cum[11]] = 0
    
    df = pd.DataFrame({'AA': np.r_[np.repeat(['trans'],sum([t_pos_1, t_neg_1])),np.repeat(['untrans'],sum([ut_pos_1, ut_neg_1])),\
                                   np.repeat(['trans'],sum([t_pos_2, t_neg_2])),np.repeat(['untrans'],sum([ut_pos_2, ut_neg_2])),\
                                   np.repeat(['trans'],sum([t_pos_3, t_neg_3])),np.repeat(['untrans'],sum([ut_pos_3, ut_neg_3]))],
                       'BB': np.r_[np.repeat(['group1'],sum([t_pos_1, t_neg_1, ut_pos_1, ut_neg_1])),\
                                   np.repeat(['group2'],sum([t_pos_2, t_neg_2, ut_pos_2, ut_neg_2])),\
                                   np.repeat(['group3'],sum([t_pos_3, t_neg_3, ut_pos_3, ut_neg_3]))],
                       'Target': target})
    
    model = ols('Target ~ AA + BB + AA*BB', data=df).fit()
    
    return print(sm.stats.anova_lm(model, typ=2))


LCA1 = [13, 1, 35, 3]
LCA2 = [18, 1, 63, 3]
LCA3 = [53, 4, 150, 16]
anova_subgroup_3(LCA1, LCA2, LCA3)














Age1 = [40, 2, 86, 8]
Age2 = [13, 3, 21, 1]
anova_subgroup(Age1, Age2)

Sex1 = [15, 3, 38, 5]
Sex2 = [38, 2, 69, 4]
anova_subgroup(Sex1, Sex2)

KPS1 = [41, 3, 90, 8]
KPS2 = [12, 2, 17, 1]
anova_subgroup(KPS1, KPS2)

LM1 = [5, 1, 11, 1]
LM2 = [48, 4, 96, 8]
anova_subgroup(LM1, LM2)

PM1 = [32, 4, 104, 7]
PM2 = [21, 1, 3, 2]
anova_subgroup(PM1, PM2)

BM1 = [43, 4, 105, 8]
BM2 = [10, 1, 2, 1]
anova_subgroup(BM1, BM2)

PTL1 = [22, 2, 43, 3]
PTL2 = [31, 3, 64, 6]
anova_subgroup(PTL1, PTL2)

PTS1 = [27, 3, 63, 6]
PTS2 = [15, 2, 29, 3]
anova_subgroup(PTS1, PTS2)

RLN1 = [5, 0, 49, 4]
RLN2 = [48, 5, 58, 5]
anova_subgroup(RLN1, RLN2)

ELI1 = [35, 1, 72, 6]
ELI2 = [18, 4, 35, 3]
anova_subgroup(ELI1, ELI2)

LCA1 = [6, 2, 11, 3]
LCA2 = [18, 1, 34, 2]
LCA3 = [29, 2, 62, 4]
anova_subgroup_3(LCA1, LCA2, LCA3)

WL1 = [37, 1, 59, 5]
WL2 = [16, 4, 45, 4]
anova_subgroup(WL1, WL2)

LDH1 = [34, 2, 79, 8]
LDH2 = [16, 3, 23, 1]
anova_subgroup(LDH1, LDH2)

Albumin1 = [41, 3, 88, 12]
Albumin2 = [12, 2, 19, 1]
anova_subgroup(Albumin1, Albumin2)

SCT1 = [34, 1, 45, 8]
SCT2 = [19, 4, 62, 1]
anova_subgroup(SCT1, SCT2)