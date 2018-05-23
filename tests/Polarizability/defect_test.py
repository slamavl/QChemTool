# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:28:04 2018

@author: Vladislav Sláma
"""

from QChemTool.Development.defect import Defect,initialize_defect_database,defects_database
from QChemTool import position_units
from QChemTool import Structure
from QChemTool.QuantumChem.read_mine import read_TrEsp_charges
from QChemTool.QuantumChem.Fluorographene.fluorographene import constrainsFG

import numpy as np

coor,tr_charge,at_type = read_TrEsp_charges("Perylene_transition_TDDFT_fitted_charges_NoH.out")
coor,gr_charge,at_type = read_TrEsp_charges("Perylene_ground_TDDFT_fitted_charges_NoH.out")
coor,ex_charge,at_type = read_TrEsp_charges("Perylene_excited_TDDFT_fitted_charges_NoH.out")

with position_units("Angstrom"):
    struc = {"coor": coor, "at_type": at_type}
    charges =  {"ground": gr_charge, "excited": ex_charge,"transition": tr_charge}
    def1 = Defect(struc=struc, charges=charges)


system = Structure()
system.load_xyz("FGrph_1perylene_2dist_ser_TDDFT-wB97XD_geom_BLYP-landl2dz_symm.xyz")

# Get set of defects from structure
indx_FG = constrainsFG(system,border=False,defect=True)
indx_FG = np.array(indx_FG,dtype="i8")
coor = system.coor.value[indx_FG]
at_type = []
for ii in indx_FG:
    at_type.append( system.at_type[ii] )

FGdefects = Structure()
FGdefects.add_coor(coor,at_type)
indx_def = FGdefects.count_fragments()
Ndef = len(indx_def)
defects = []
for ii in range(Ndef):
    at_type = []
    for jj in indx_def[ii]:
        at_type.append( FGdefects.at_type[jj] )
    struc = {"coor": FGdefects.coor.value[indx_def[ii]], "at_type": at_type}
    index = list(indx_FG[ indx_def[ii] ])
    defct = Defect(struc=struc, index=index)
    defects.append(defct)
print(defects[0].index)

# identify defect
def2 = defects[0]
#index_corr,RSMD = def2.identify_defect(def1)
#def1.output_to_pdb("def1.pdb")
#def2.coor.value = def2.coor.value[index_corr]
#def2.output_to_pdb("def2_reord.pdb")

def2.load_charges_from_defect(def1)



coor,tr_charge,at_type = read_TrEsp_charges("Perylene_transition_TDDFT_fitted_charges_NoH.out")
coor,gr_charge,at_type = read_TrEsp_charges("Perylene_ground_TDDFT_fitted_charges_NoH.out")
coor,ex_charge,at_type = read_TrEsp_charges("Perylene_excited_TDDFT_fitted_charges_NoH.out")
with position_units("Angstrom"):
    struc = {"coor": coor, "at_type": at_type}
    charges =  {"ground": gr_charge, "excited": ex_charge,"transition": tr_charge}
    def1 = Defect(struc=struc, charges=charges)
    def1.coor.value
    
defects_database = initialize_defect_database("QC")
print(defects_database.keys())

# FG system automaticaly identify defects - parameters: transition energy type
# Then manually call function assign defects to allow some changes in the database

