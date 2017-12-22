# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:49:56 2017

@author: Vladislav Sláma
"""

from QChemTool.QuantumChem.Classes.structure import Structure
from QChemTool.QuantumChem.Fluorographene.fluorographene import deleteFG
from QChemTool.General.UnitsManager import position_units
import QChemTool.QuantumChem.output as out
import numpy as np

dir="C:/PhD/Ab-initio-META/Fluorographane/Excited_states/Distance-dependence/"
xyz_file="".join([dir,"FGrph_2perylene3_3dist_par_TDDFT-wB97XD_geom_BLYP-landl2dz_symm_9x14.xyz"])
struc=Structure()
struc.load_xyz(xyz_file)
border=True
defect=False

def constrainsFG(struc,border=True,defect=False):
    if struc.bonds is None:
        struc.guess_bonds()
    
    # border carbons - conected to only 2 other carbons + cabons conected to these
    if border:
        constrains=np.zeros(struc.nat,dtype='bool')
        Nbonds = np.zeros(struc.nat,dtype='i8')
        connect_CC=[]
        connect_CF=[]
        for ii in range(struc.nat):
            connect_CC.append([])
            connect_CF.append([])
        for bond in struc.bonds:
            if struc.at_type[bond[0]]=='C' and struc.at_type[bond[1]]=='C':
                Nbonds[bond[0]] += 1
                Nbonds[bond[1]] += 1
                connect_CC[bond[0]].append(bond[1])
                connect_CC[bond[1]].append(bond[0])
            elif struc.at_type[bond[0]]=='C' and struc.at_type[bond[1]]=='F':
                connect_CF[bond[0]].append(bond[1])
                connect_CF[bond[1]].append(bond[0])
            elif struc.at_type[bond[0]]=='F' and struc.at_type[bond[1]]=='C':
                connect_CF[bond[0]].append(bond[1])
                connect_CF[bond[1]].append(bond[0])
            
        BorderC1=np.where(Nbonds==2)[0]
        
        for ii in BorderC1:
            constrains[ii]=True
            for jj in connect_CC[ii]:
                constrains[jj]=True
    
    # Defect carbons - carbons without fluorines
    if defect:
        for ii in range(struc.nat):
            if len(connect_CF[ii])==0:
                constrains[ii]=True
    
    constrain_indx=list(np.arange(struc.nat)[constrains])
    
    return constrain_indx
    
constrain = constrainsFG(struc,border=True,defect=True)
for ii in constrain:
    print("X",ii+1,"F")

#def constrainsFG(struc,border=True):
#    if struc.bo


## Create gjf file for H optimization
#defect_struc=deleteFG(struc,add_hydrogen=True)
#defect_struc.output_to_xyz("Defect_test.xyz")
#
#Fragments=deleteFG(struc,add_hydrogen=True,Fragmentize=True)
#for ii in range(len(Fragments)):
#    structure=Fragments[ii]
#    structure.output_to_xyz("".join(["Defect_",str(ii),".xyz"]))
#
## Greate Gaussian input file for hydrogen optimization
#constrain=np.where(np.array(defect_struc.at_type)!='H')[0]
#with position_units('Angstrom'):
#    out.OutputToGauss(defect_struc.coor.value, defect_struc.at_type, 
#                  'opt_restricted', 'BLYP', 'LANL2DZ', filename='Defects_Hopt.gjf',
#                  namechk='Defects_Hopt.chk', constrain=constrain, MEM_GB=2,
#                  Tight=True,verbose=False)