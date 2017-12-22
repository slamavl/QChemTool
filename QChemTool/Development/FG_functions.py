# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:49:56 2017

@author: Vladislav Sláma
"""

from QChemTool.QuantumChem.Classes.structure import Structure
from QChemTool.QuantumChem.Fluorographene.fluorographene import deleteFG, constrainsFG
from QChemTool.General.UnitsManager import position_units
import QChemTool.QuantumChem.output as out
import numpy as np

dir="C:/PhD/Ab-initio-META/Fluorographane/Excited_states/Distance-dependence/"
xyz_file="".join([dir,"FGrph_2perylene3_3dist_par_TDDFT-wB97XD_geom_BLYP-landl2dz_symm_9x14.xyz"])
gjf_file="".join([dir,"FGrph_1bisanthrene_5dist_ser_symm_18x9.gjf"])
struc=Structure()
struc.load_xyz(xyz_file)
#struc.load_gjf(gjf_file)
    
#constrain = constrainsFG(struc,border=True,defect=True)
#for ii in constrain:
#    print("X",ii+1,"F")

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