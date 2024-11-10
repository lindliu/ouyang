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


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

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
    
    print(namestr(group1, globals())[0])
    return print(sm.stats.anova_lm(model, typ=2))


age1 = [194, 23, 60, 17]
age2 = [98, 9, 27, 4]
anova_subgroup(age1, age2)

sex1 = [114, 11, 33, 9]
sex2 = [178, 21, 54, 12]
anova_subgroup(sex1, sex2)

KPS1 = [138, 7, 45, 5]
KPS2 = [154, 25, 42, 16]
anova_subgroup(KPS1, KPS2)

PT_size1 = [166, 18, 64, 18]
PT_size2 = [109, 13, 18, 2]
anova_subgroup(PT_size1, PT_size2)

RLN1 = [212, 20, 64, 14]
RLN2 = [78, 12, 23, 7]
anova_subgroup(RLN1, RLN2)

ELI1 = [248, 31, 75, 18]
ELI2 = [44, 1, 12, 3]
anova_subgroup(ELI1, ELI2)

WLM1 = [98, 12, 30, 8]
WLM2 = [194, 20, 57, 13]
anova_subgroup(WLM1, WLM2)

ASC1 = [63, 12, 28, 4]
ASC2 = [229, 30, 59, 17]
anova_subgroup(ASC1, ASC2)


CEA1 = [123, 16, 44, 12]
CEA2 = [168, 16, 43, 9]
anova_subgroup(CEA1, CEA2)

PTR1 = [11, 4, 49, 16]
PTR2 = [281, 28, 38, 5]
anova_subgroup(PTR1, PTR2)

LDT1 = [65, 7, 25, 6]
LDT2 = [227, 25, 62, 15]
anova_subgroup(LDT1, LDT2)








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
    
    print(namestr(group1, globals())[0])
    return print(sm.stats.anova_lm(model, typ=2))


LCA1 = [34, 6, 10, 8]
LCA2 = [74, 8, 24, 8]
LCA3 = [184, 18, 53, 5]
anova_subgroup_3(LCA1, LCA2, LCA3)







def anova_subgroup_5(group1, group2, group3, group4, group5):
    t_pos_1, t_neg_1, ut_pos_1, ut_neg_1 = group1
    t_pos_2, t_neg_2, ut_pos_2, ut_neg_2 = group2
    t_pos_3, t_neg_3, ut_pos_3, ut_neg_3 = group3
    t_pos_4, t_neg_4, ut_pos_4, ut_neg_4 = group4
    t_pos_5, t_neg_5, ut_pos_5, ut_neg_5 = group5

    
    num = sum([*group1, *group2, *group3, *group4, *group5])
    cum = np.cumsum([*group1, *group2, *group3, *group4, *group5])
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
    target[cum[11]:cum[12]] = 1
    target[cum[12]:cum[13]] = 0
    target[cum[13]:cum[14]] = 1
    target[cum[14]:cum[15]] = 0
    target[cum[15]:cum[16]] = 1
    target[cum[16]:cum[17]] = 0
    target[cum[17]:cum[18]] = 1
    target[cum[18]:cum[19]] = 0
    
    df = pd.DataFrame({'AA': np.r_[np.repeat(['trans'],sum([t_pos_1, t_neg_1])),np.repeat(['untrans'],sum([ut_pos_1, ut_neg_1])),\
                                   np.repeat(['trans'],sum([t_pos_2, t_neg_2])),np.repeat(['untrans'],sum([ut_pos_2, ut_neg_2])),\
                                   np.repeat(['trans'],sum([t_pos_3, t_neg_3])),np.repeat(['untrans'],sum([ut_pos_3, ut_neg_3])),\
                                   np.repeat(['trans'],sum([t_pos_4, t_neg_4])),np.repeat(['untrans'],sum([ut_pos_4, ut_neg_4])),\
                                   np.repeat(['trans'],sum([t_pos_5, t_neg_5])),np.repeat(['untrans'],sum([ut_pos_5, ut_neg_5]))],
                       'BB': np.r_[np.repeat(['group1'],sum([t_pos_1, t_neg_1, ut_pos_1, ut_neg_1])),\
                                   np.repeat(['group2'],sum([t_pos_2, t_neg_2, ut_pos_2, ut_neg_2])),\
                                   np.repeat(['group3'],sum([t_pos_3, t_neg_3, ut_pos_3, ut_neg_3])),\
                                   np.repeat(['group3'],sum([t_pos_4, t_neg_4, ut_pos_4, ut_neg_4])),\
                                   np.repeat(['group3'],sum([t_pos_5, t_neg_5, ut_pos_5, ut_neg_5]))],
                       'Target': target})
    
    model = ols('Target ~ AA + BB + AA*BB', data=df).fit()
    
    print(namestr(group1, globals())[0])
    return print(sm.stats.anova_lm(model, typ=2))


SCT1 = [117, 8, 32, 5]
SCT2 = [28, 1, 12, 2]
SCT3 = [29, 1, 9, 1]
SCT4 = [95, 19, 20, 11]
SCT5 = [23, 3, 14, 2]
anova_subgroup_5(SCT1, SCT2, SCT3, SCT4, SCT5)



