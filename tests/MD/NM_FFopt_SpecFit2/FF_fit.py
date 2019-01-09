#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 19:41:47 2018

@author: slamav
"""

from QChemTool import Structure
import numpy as np
from QChemTool.QuantumChem.Fluorographene.fluorographene import get_border_carbons_FG

# Load initial structure
struc = Structure()
struc.load_xyz("FGrph-5hex_opt_freqHP_reord.xyz") #FGrph-5hex_opt_freqHP_reord.xyz")

# assign charges
FG_charges = [-0.0522,-0.0522]
#FG_charges = [0.081250,0.081250]
border_C_indx,border_F_indx = get_border_carbons_FG(struc)
struc.esp_grnd = np.zeros(struc.nat,dtype='f8')
charges = struc.esp_grnd # pointer to structure charges
   
struc.get_FF_types() 
for ii in range(struc.nat):
    if struc.at_type[ii] == 'C':
        struc.ff_type[ii] = 'c3'
        charges[ii] = FG_charges[0]
    elif struc.at_type[ii] == 'F':
        struc.ff_type[ii] = 'f'
        charges[ii] = -FG_charges[0]
    else:
        raise Warning("Unknown atom type in structure")
charges[border_C_indx] = 2*FG_charges[1]
charges[border_F_indx] = -FG_charges[1]
for ii in range(len(border_C_indx)):
    struc.ff_type[border_C_indx[ii]] = 'cb'
for ii in range(len(border_F_indx)):
    struc.ff_type[border_F_indx[ii]] = 'fb'

VdW_param = {'c3': [1.9080, 0.1094],'cb': [1.9080, 0.1094],'ca': [1.9080, 0.1094],'f': [1.75, 0.061],'fb': [1.75, 0.061]}

AoI = 163
E_elstat = 0.0
E_vdw = 0.0

#for AoI in range(struc.nat):
#    connected = struc.get_bonded_atoms()
#    INDX=[AoI]
#    for nn in range(3):
#        for ii in range(len(INDX)):
#            TMP = connected[INDX[ii]]
#            for jj in TMP:
#                INDX.append(jj)
#        INDX = list(np.unique(INDX))
connected = struc.get_bonded_atoms()
INDX=[AoI]
for nn in range(3):
    for ii in range(len(INDX)):
        TMP = connected[INDX[ii]]
        for jj in TMP:
            INDX.append(jj)
    INDX = list(np.unique(INDX))
    
nmax=11
z_init = struc.coor.value[AoI][2]
energy=np.zeros(11)
zcoor=np.zeros(11)
for nn in range(nmax):
    struc.coor._value[AoI][2] = z_init + (nn-(nmax-1)/2) * 0.1/0.52917721067
    E_elstat = 0.0
    E_vdw = 0.0
    for ii in range(struc.nat):
        if not (ii in INDX):
            rij = np.linalg.norm(struc.coor.value[ii] - struc.coor.value[AoI]) # Cha
            E_elstat += charges[ii]*charges[AoI]/rij 
            
            R0 = (VdW_param[struc.ff_type[ii]][0] + VdW_param[struc.ff_type[AoI]][0])
            Eij = np.sqrt(VdW_param[struc.ff_type[ii]][1] * VdW_param[struc.ff_type[AoI]][1])
            E_vdw += Eij*((R0/rij/0.52917721067)**12 - 2*(R0/rij/0.52917721067)**6 ) *0.00159362 
            #E_vdw += (4*Eij*(R0/2/rij)**12 - (R0/2/rij)**6 )
    energy[nn] = E_elstat +E_vdw
    zcoor[nn] = struc.coor.value[AoI][2]
     
#print(E_elstat/2/0.00159362,E_vdw/2/0.00159362)

