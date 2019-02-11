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

# Load initial structure
struc = Structure()
struc.load_xyz("FGrph-3hex.xyz") #FGrph-5hex_opt_freqHP_reord.xyz")

struc.center(23,77,124)
struc.output_to_xyz('FGrph-3hex_aligned.xyz')
struc_rot = struc.copy()

for ii in range(-3,4):
    struc_rot = struc.copy()
    struc_rot.rotate(np.deg2rad(5)*ii,0.0,0.0)
    struc_out = struc.copy()
    struc_out.coor._value[77] = struc_rot.coor._value[77]
    struc_out.coor._value[124] = struc_rot.coor._value[124]
    name = "".join(["FGrph-3hex_fb_Scan_0",str(ii),".xyz"])
    struc_out.output_to_xyz(name)
    with position_units("Angstrom"):
        coor = struc_out.coor.value
        at_type = struc_out.at_type
        filename = "".join(["FGrph-3hex_fb_Scan_0",str(ii),".gjf"])
        filename_chk = "".join(["FGrph-3hex_fb_Scan_0",str(ii),".chk"])
        OutputToGauss(coor,at_type,'sp','BLYP','LANL2DZ',filename=filename,namechk=filename_chk,Tight=False,MEM_GB=10)
    
    
for ii in range(-3,4):
    struc_rot = struc.copy()
    struc_rot.rotate(0.0,0.0,np.deg2rad(5)*ii)
    struc_out = struc.copy()
    struc_out.coor._value[77] = struc_rot.coor._value[77]
    struc_out.coor._value[124] = struc_rot.coor._value[124]
    name = "".join(["FGrph-3hex_fb_Scan_1",str(ii),".xyz"])
    struc_out.output_to_xyz(name)
    with position_units("Angstrom"):
        coor = struc_out.coor.value
        at_type = struc_out.at_type
        filename = "".join(["FGrph-3hex_fb_Scan_1",str(ii),".gjf"])
        filename_chk = "".join(["FGrph-3hex_fb_Scan_1",str(ii),".chk"])
        OutputToGauss(coor,at_type,'sp','BLYP','LANL2DZ',filename=filename,namechk=filename_chk,Tight=False,MEM_GB=10)
    
    
struc.center(23,124,77)
for ii in range(-3,4):
    struc_rot = struc.copy()
    struc_rot.rotate(0.0,0.0,np.deg2rad(5)*ii)
    struc_out = struc.copy()
    struc_out.coor._value[77] = struc_rot.coor._value[77]
    struc_out.coor._value[124] = struc_rot.coor._value[124]
    name = "".join(["FGrph-3hex_fb_Scan_2",str(ii),".xyz"])
    struc_out.output_to_xyz(name)
    with position_units("Angstrom"):
        coor = struc_out.coor.value
        at_type = struc_out.at_type
        filename = "".join(["FGrph-3hex_fb_Scan_2",str(ii),".gjf"])
        filename_chk = "".join(["FGrph-3hex_fb_Scan_2",str(ii),".chk"])
        OutputToGauss(coor,at_type,'sp','BLYP','LANL2DZ',filename=filename,namechk=filename_chk,Tight=False,MEM_GB=10)
    

struc.center(18,72,122)    
for ii in range(-3,4):
    struc_rot = struc.copy()
    struc_rot.rotate(np.deg2rad(5)*ii,0.0,0.0)
    struc_out = struc.copy()
    struc_out.coor._value[72] = struc_rot.coor._value[72]
    struc_out.coor._value[122] = struc_rot.coor._value[122]
    name = "".join(["FGrph-3hex_fb_Scan_3",str(ii),".xyz"])
    struc_out.output_to_xyz(name)
    with position_units("Angstrom"):
        coor = struc_out.coor.value
        at_type = struc_out.at_type
        filename = "".join(["FGrph-3hex_fb_Scan_3",str(ii),".gjf"])
        filename_chk = "".join(["FGrph-3hex_fb_Scan_3",str(ii),".chk"])
        OutputToGauss(coor,at_type,'sp','BLYP','LANL2DZ',filename=filename,namechk=filename_chk,Tight=False,MEM_GB=10)
    
    
for ii in range(-3,4):
    struc_rot = struc.copy()
    struc_rot.rotate(0.0,0.0,np.deg2rad(5)*ii)
    struc_out = struc.copy()
    struc_out.coor._value[72] = struc_rot.coor._value[72]
    struc_out.coor._value[122] = struc_rot.coor._value[122]
    name = "".join(["FGrph-3hex_fb_Scan_4",str(ii),".xyz"])
    struc_out.output_to_xyz(name)
    with position_units("Angstrom"):
        coor = struc_out.coor.value
        at_type = struc_out.at_type
        filename = "".join(["FGrph-3hex_fb_Scan_4",str(ii),".gjf"])
        filename_chk = "".join(["FGrph-3hex_fb_Scan_4",str(ii),".chk"])
        OutputToGauss(coor,at_type,'sp','BLYP','LANL2DZ',filename=filename,namechk=filename_chk,Tight=False,MEM_GB=10)
    
    
struc.center(18,122,72)
for ii in range(-3,4):
    struc_rot = struc.copy()
    struc_rot.rotate(0.0,0.0,np.deg2rad(5)*ii)
    struc_out = struc.copy()
    struc_out.coor._value[72] = struc_rot.coor._value[72]
    struc_out.coor._value[122] = struc_rot.coor._value[122]
    name = "".join(["FGrph-3hex_fb_Scan_5",str(ii),".xyz"])
    struc_out.output_to_xyz(name)
    with position_units("Angstrom"):
        coor = struc_out.coor.value
        at_type = struc_out.at_type
        filename = "".join(["FGrph-3hex_fb_Scan_5",str(ii),".gjf"])
        filename_chk = "".join(["FGrph-3hex_fb_Scan_5",str(ii),".chk"])
        OutputToGauss(coor,at_type,'sp','BLYP','LANL2DZ',filename=filename,namechk=filename_chk,Tight=False,MEM_GB=10)
    


## assign charges
#FG_charges = [-0.0522,-0.0522]
##FG_charges = [0.081250,0.081250]
#border_C_indx,border_F_indx = get_border_carbons_FG(struc)
#struc.esp_grnd = np.zeros(struc.nat,dtype='f8')
#charges = struc.esp_grnd # pointer to structure charges
#   
#struc.get_FF_types() 
#for ii in range(struc.nat):
#    if struc.at_type[ii] == 'C':
#        struc.ff_type[ii] = 'c3'
#        charges[ii] = FG_charges[0]
#    elif struc.at_type[ii] == 'F':
#        struc.ff_type[ii] = 'f'
#        charges[ii] = -FG_charges[0]
#    else:
#        raise Warning("Unknown atom type in structure")
#charges[border_C_indx] = 2*FG_charges[1]
#charges[border_F_indx] = -FG_charges[1]
#for ii in range(len(border_C_indx)):
#    struc.ff_type[border_C_indx[ii]] = 'cb'
#for ii in range(len(border_F_indx)):
#    struc.ff_type[border_F_indx[ii]] = 'fb'
#
#VdW_param = {'c3': [1.9080, 0.1094],'cb': [1.9080, 0.1094],'ca': [1.9080, 0.1094],'f': [1.75, 0.061],'fb': [1.75, 0.061]}
#
#AoI = 163
#E_elstat = 0.0
#E_vdw = 0.0
#
##for AoI in range(struc.nat):
##    connected = struc.get_bonded_atoms()
##    INDX=[AoI]
##    for nn in range(3):
##        for ii in range(len(INDX)):
##            TMP = connected[INDX[ii]]
##            for jj in TMP:
##                INDX.append(jj)
##        INDX = list(np.unique(INDX))
#connected = struc.get_bonded_atoms()
#INDX=[AoI]
#for nn in range(3):
#    for ii in range(len(INDX)):
#        TMP = connected[INDX[ii]]
#        for jj in TMP:
#            INDX.append(jj)
#    INDX = list(np.unique(INDX))
#    
#nmax=11
#z_init = struc.coor.value[AoI][2]
#energy=np.zeros(11)
#zcoor=np.zeros(11)
#for nn in range(nmax):
#    struc.coor._value[AoI][2] = z_init + (nn-(nmax-1)/2) * 0.1/0.52917721067
#    E_elstat = 0.0
#    E_vdw = 0.0
#    for ii in range(struc.nat):
#        if not (ii in INDX):
#            rij = np.linalg.norm(struc.coor.value[ii] - struc.coor.value[AoI]) # Cha
#            E_elstat += charges[ii]*charges[AoI]/rij 
#            
#            R0 = (VdW_param[struc.ff_type[ii]][0] + VdW_param[struc.ff_type[AoI]][0])
#            Eij = np.sqrt(VdW_param[struc.ff_type[ii]][1] * VdW_param[struc.ff_type[AoI]][1])
#            E_vdw += Eij*((R0/rij/0.52917721067)**12 - 2*(R0/rij/0.52917721067)**6 ) *0.00159362 
#            #E_vdw += (4*Eij*(R0/2/rij)**12 - (R0/2/rij)**6 )
#    energy[nn] = E_elstat +E_vdw
#    zcoor[nn] = struc.coor.value[AoI][2]
#     
##print(E_elstat/2/0.00159362,E_vdw/2/0.00159362)

