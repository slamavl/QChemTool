# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:27:58 2018

@author: Vladislav Sláma
"""

from QChemTool.QuantumChem.read_mine import read_VMD_pdb
from QChemTool.QuantumChem.Classes.structure import Structure
from QChemTool.General.UnitsManager import position_units
import numpy as np
import matplotlib.pyplot as plt

MD=read_VMD_pdb("D:/slamav/MD/Perylene-Chloroform/05_prod.pdb")
indx1 = [38,61]
indx2 = [ [92,70], [29,2] ]

# Transform PDB atomic names to Gaussian atomic types
at_type=[]
for jj in range(MD.NAtom):
    at_type.append(''.join([i for i in MD.at_name[jj] if not i.isdigit()]))

struc_init = Structure()
coor=MD.geom[:,:,0]
with position_units("Angstrom"):
    struc_init.add_coor(coor,at_type)

Angle1 = np.zeros(MD.NStep,dtype="f8")
Angle2 = np.zeros(MD.NStep,dtype="f8")

Vec_init1 = struc_init.coor.value[indx1[1]] - struc_init.coor.value[indx1[0]]
Vec_init2 = struc_init.coor.value[indx2[1][0]] + struc_init.coor.value[indx2[1][1]] 
Vec_init2 -= (struc_init.coor.value[indx2[0][0]] + struc_init.coor.value[indx2[0][1]])
Vec_init1 = Vec_init1/np.linalg.norm(Vec_init1)
Vec_init2 = Vec_init2/np.linalg.norm(Vec_init2)

for ii in range(MD.NStep):
    new_struc=Structure()
    coor=MD.geom[:,:,ii]
    with position_units("Angstrom"):
        new_struc.add_coor(coor,at_type)
    
    Vec1 = new_struc.coor.value[indx1[1]] - new_struc.coor.value[indx1[0]]
    Vec2 = new_struc.coor.value[indx2[1][0]] + new_struc.coor.value[indx2[1][1]] 
    Vec2 -= (new_struc.coor.value[indx2[0][0]] + new_struc.coor.value[indx2[0][1]])
    Vec1 = Vec1/np.linalg.norm(Vec1)
    Vec2 = Vec2/np.linalg.norm(Vec2)
    
    Angle1[ii] = np.rad2deg( np.arccos(np.dot(Vec1,Vec_init1)) )
    Angle2[ii] = np.rad2deg( np.arccos(np.dot(Vec2,Vec_init2)) )

time = np.arange(MD.NStep) # time in ps
try:
    indx = max(plt.get_fignums())
except:
    indx = 0

fig = plt.figure(indx+2,figsize=(10,5))
fig.canvas.set_window_title('Angle')
plt.title('Angle')
plt.xlabel('Time [ps]')
plt.ylabel('Angle [deg]')
plt.plot(time,Angle1,'r-')
plt.plot(time,Angle2,'b-')
plt.legend(["Angle 1","Angle 2"])
plt.show()
