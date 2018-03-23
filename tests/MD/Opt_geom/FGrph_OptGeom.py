# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:27:58 2018

@author: Vladislav Sláma
"""
import numpy as np
from QChemTool.General.units import conversion_facs_position as conv_pos
from QChemTool.QuantumChem.Classes.structure import Structure
from QChemTool.QuantumChem.Fluorographene.fluorographene import Optimize_MD_AMBER_structure,get_border_carbons_FG

charges = 'Hirshfeld'
state='Ground'

# Set FG charges
if charges == 'Hirshfeld':
    FG_charges = [0.08125,0.08125]

# Load initial structure
struc = Structure()
struc.load_xyz("FGrph-5hex_opt_freqHP_reord.xyz")
frcmod_filename = "FGrph_reord.frcmod"

# assign charges
border_C_indx,border_F_indx = get_border_carbons_FG(struc)
if state=='Ground':
    struc.esp_grnd = np.zeros(struc.nat,dtype='f8')
    charges = struc.esp_grnd
elif state=='Excited':
    struc.esp_exct = np.zeros(struc.nat,dtype='f8')
    charges = struc.esp_exct
elif state=='Transition':
    struc.esp_trans = np.zeros(struc.nat,dtype='f8')
    charges = struc.esp_trans
    
for ii in range(struc.nat):
    if struc.at_type[ii] == 'C':
        charges[ii] = FG_charges[0]
    elif struc.at_type[ii] == 'F':
        charges[ii] = -FG_charges[0]
    else:
        raise Warning("Unknown atom type in structure")
charges[border_C_indx] = 2*FG_charges[1]
charges[border_F_indx] = -FG_charges[1]


# optimize structure
RMSD,struc_opt,struc_prepc = Optimize_MD_AMBER_structure(frcmod_filename,struc,state=state,gen_input=True,struc_out=True)
print("RMSD:",RMSD*conv_pos["Angstrom"],"Angstroms")
struc_opt.output_to_xyz("struc_opt.xyz")
struc_prepc.output_to_xyz("struc_prepc.xyz")