# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:36:48 2018

@author: Vladislav Sláma
"""

from QChemTool.Development.polarization_periodicTEST import PolarAtom
from QChemTool.Development.polarization_periodic import PolarAtom as PolarAtom2
import numpy as np
import matplotlib.pyplot as plt

atom = PolarAtom(5.0,7.0,1.5,1.0,phase = np.pi/3)
atom2 = PolarAtom2(5.0,-2.0,3,1.0,phase = np.pi/3)
Y = np.zeros(360)
X = np.arange(360)
Z = np.zeros(360)
ZZ = np.zeros(360)
for ii in range(360):
    phi = np.deg2rad(ii)
    E = np.array([np.cos(phi),np.sin(phi),0.0])
    dip = atom.get_induced_dipole(E)
    dip2 = atom2.get_induced_dipole(E)
    Y[ii] = np.linalg.norm(dip)
    Z[ii] = np.linalg.norm(dip2)
    ZZ[ii] = np.rad2deg(np.arccos(np.dot(dip2,dip)/np.linalg.norm(dip)/np.linalg.norm(dip2)))
#plt.plot(X,Y)
#plt.plot(X,Z)
plt.plot(X,ZZ)