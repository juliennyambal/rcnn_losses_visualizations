# from http://blog.csdn.net/wxplol/article/details/73694657

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 14:17:47 2017

@author: julien
"""

#!/usr/bin/env python  
import os  
import sys  
import numpy as np  
import matplotlib.pyplot as plt  
import math  
import re  
import pylab  
from pylab import figure, show, legend  
from mpl_toolkits.axes_grid1 import host_subplot  
  
# read the log file  
#fp = open('2.txt', 'r')

train_rpn_loss_box = []
train_bbox_loss = []
train_iterations = []  
train_loss = []
train_rpn_cls_loss = []
train_cls_loss = []
test_iterations = []

fp = open('/home/julien/Documents/Python_tests/log/my_model_modified.log', 'r')

for ln in fp:
    
    if 'rpn_loss_bbox = ' in ln:
        print ln
        train_rpn_loss_box.append(float(ln.strip().split(' ')[-2]))
        print train_rpn_loss_box
    
    if 'bbox_loss = ' in ln:
        print ln
        train_bbox_loss.append(float(ln.strip().split(' ')[-2]))
        print train_bbox_loss
    
    if 'rpn_cls_loss = ' in ln:
        print ln
        train_rpn_cls_loss.append(float(ln.strip().split(' ')[-2]))
        print train_rpn_cls_loss
    
    if ' cls_loss = ' in ln:
        train_cls_loss.append(float(ln.strip().split(' ')[-2]))
        print train_cls_loss
    
    if '] Iteration ' in ln and 'loss = ' in ln:
        print ln
        arr = re.findall(r'ion \b\d+\b',ln)
        print arr
        train_iterations.append(int(arr[0].strip(',')[4:]))
        print train_iterations
        
        train_loss.append(float(ln.strip().split(' = ')[-1]))
        print train_loss
fp.close()

host = host_subplot(111)
plt.subplots_adjust(right=0.8) 
# ajust the right boundary of the plot window  
#par1 = host.twinx()  
# set labels  
host.set_xlabel("iterations")  
host.set_ylabel("Loss")  
#par1.set_ylabel("validation accuracy")  
  
# plot curves  
p1, = host.plot(train_iterations, train_loss, label="train loss")
p2, = host.plot(train_iterations, train_cls_loss, label="cls loss")
p3, = host.plot(train_iterations, train_rpn_cls_loss, label="rpn cls loss")
p4, = host.plot(train_iterations, train_bbox_loss, label="bbox loss")
p4, = host.plot(train_iterations, train_rpn_loss_box, label="rpn bbox loss")
#p2, = par1.plot(test_iterations, test_accuracy, label="validation accuracy")  
  
# set location of the legend,   
# 1->rightup corner, 2->leftup corner, 3->leftdown corner  
# 4->rightdown corner, 5->rightmid ...  
host.legend(loc=1)  
  
# set label color  
host.axis["left"].label.set_color(p1.get_color())  
#par1.axis["right"].label.set_color(p2.get_color())  
# set the range of x axis of host and y axis of par1  
host.set_xlim([0,500])  
host.set_ylim([0., 5])  
  
plt.draw()  
plt.show()  
