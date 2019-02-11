#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 19:41:47 2018

@author: slamav
"""

from QChemTool import Structure
from QChemTool.QuantumChem.output import OutputToGauss
from QChemTool import position_units
import numpy as np
from QChemTool.QuantumChem.Fluorographene.fluorographene import get_border_carbons_FG

def load_struc(name):
    # Load initial structure
    struc = Structure()
    struc.load_xyz(name) #FGrph-5hex_opt_freqHP_reord.xyz")
    # assign charges
    FG_charges = [-0.0522,-0.0522]
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
    return struc

def angle(a,a0,E):
    res = E*(a-a0)**2
    return res

def dihedral(a,a0,E):
    phi = np.deg2rad(a)
    phi0 = np.deg2rad(a0)
    res= E*(1+np.cos(3*phi - phi0))
    return res

def E_dih(struc,indx,a0,E):
    dih = struc.get_dihedral(indx,'deg')
    res = dihedral(dih,a0,E)
    return res

def E_ang(struc,indx,a0,E):
    dih = struc.get_angle(indx,'deg')
    res = angle(dih,a0,E)
    return res

def E_vdw_elstat(struc,AoI):
    VdW_param = {'c3': [1.9080, 0.1094],'cb': [1.9080, 0.1094],'ca': [1.9080, 0.1094],'f': [1.75, 0.061],'fb': [1.75, 0.061]}
    E_elstat = 0.0
    E_vdw = 0.0
    charges = struc.esp_grnd
    connected = struc.get_bonded_atoms()
    INDX=[AoI]
    for nn in range(3):
        for ii in range(len(INDX)):
            TMP = connected[INDX[ii]]
            for jj in TMP:
                INDX.append(jj)
        INDX = list(np.unique(INDX))
        
    for ii in range(struc.nat):
        if not (ii in INDX):
            rij = np.linalg.norm(struc.coor.value[ii] - struc.coor.value[AoI]) # Cha
            E_elstat += charges[ii]*charges[AoI]/rij 
            
            R0 = (VdW_param[struc.ff_type[ii]][0] + VdW_param[struc.ff_type[AoI]][0])
            Eij = np.sqrt(VdW_param[struc.ff_type[ii]][1] * VdW_param[struc.ff_type[AoI]][1])
            E_vdw += Eij*((R0/rij/0.52917721067)**12 - 2*(R0/rij/0.52917721067)**6 ) *0.00159362 
    
    return E_elstat+E_vdw

def structure_eng(struc,params_ang,params_dih,Angle_indx,Torsion_indx):
    Energy = 0.0

    for ii in range(len(Angle_indx)):
        fftype = "-".join([struc.ff_type[Angle_indx[ii][0]],struc.ff_type[Angle_indx[ii][1]],struc.ff_type[Angle_indx[ii][2]]])
        Phi = struc.get_angle(Angle_indx[ii],'deg')
        print(fftype,Phi)
        Phi0 = params_ang[fftype][1]
        K = params_ang[fftype][0]
        Energy += E_ang(struc,Angle_indx[ii],Phi0,K) 
    
    for ii in range(len(Torsion_indx)):
        fftype = "-".join([struc.ff_type[Torsion_indx[ii][0]],struc.ff_type[Torsion_indx[ii][1]],struc.ff_type[Torsion_indx[ii][2]],struc.ff_type[Torsion_indx[ii][3]]])
        #Phi = struc.get_dihedral(Torsion_indx[ii],'deg')
        #print(fftype,Phi)
        Phi0 = params_dih[fftype][1]
        K = params_dih[fftype][0]
        Energy += E_dih(struc,Torsion_indx[ii],Phi0,K) 
    
    for ii in AoI:
        Energy += E_vdw_elstat(struc,ii)*627.503 # in kcal/mol
    
    return Energy

AoI=[77,124]
params_ang={'fb-cb-c3': [100.0,120.0],'fb-cb-cb': [100.0,120.0]}
params_dih={'fb-cb-cb-c3': [1.0,180.0],'fb-cb-c3-cb': [1.0,180.0],
            'fb-cb-c3-c3': [1.0,180.0],'fb-cb-c3-f': [1.0,180.0],
            'fb-cb-cb-fb': [1.0,180.0]}

Angle_indx=[[77 ,23, 51],[77 ,23, 47],[124 ,23, 51],[124 ,23, 47]]
Torsion_indx = [[77, 23, 47, 19],[77, 23, 47, 18],[77, 23, 47, 101],
                [124, 23, 47, 19],[124, 23, 47, 18],[124, 23, 47, 101],
                [77, 23, 51, 24],[77, 23, 51, 108],[77, 23, 51, 105],
                [124, 23, 51, 24],[124, 23, 51, 108],[124, 23, 51, 105]]

Angle2_indx=[[72 ,18, 47],[72 ,18, 42],[122 ,18, 47],[122 ,18, 42]]
Torsion2_indx =[[72, 18, 47, 101],[72, 18, 47, 19],[72, 18, 47, 23],
               [122, 18, 47, 101],[122, 18, 47, 19],[122, 18, 47, 23],
               [72, 18, 42, 96],[72, 18, 42, 12],[72, 18, 42, 13],
               [122, 18, 42, 96],[122, 18, 42, 12],[122, 18, 42, 13]]


struc = load_struc("FGrph-3hex.xyz")
Energy = structure_eng(struc,params_ang,params_dih,Angle_indx,Torsion_indx)
print(Energy)
Energy = structure_eng(struc,params_ang,params_dih,Angle2_indx,Torsion2_indx)
print(Energy)

