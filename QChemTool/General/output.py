# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:29:14 2016

@author: User
"""
import numpy as np

def ArrayToFile(data,filename): # pridat fornat vypisu jako vstupni paarmetr
    dim=len(np.shape(data))
    with open(filename, "wt") as f:
        if dim==1:
            for ii in range(len(data)):
                f.write("{:22.14f}\n".format(data[ii]))
        elif dim==2:
            for ii in range(np.shape(data)[0]):
                for jj in range(np.shape(data)[1]):
                    f.write("{:22.14f}".format(data[ii,jj]))
                f.write('\n')
        elif dim==3:
            for ii in range(np.shape(data)[0]):
                for jj in range(np.shape(data)[1]):
                    for kk in range(np.shape(data)[2]):
                        f.write("{:22.14f}".format(data[ii,jj,kk]))
                    f.write('\n')
        #else:
            # definovat rekurentne odstranit jednu dimenzi a zbytek vypsat
            # for ii in range(np.shape(data)[0]):
            #   ArrayToFile(data[ii,.....]) ale to by tam nesmelo byt otevreni a zavreni souboru
    f.close()
        