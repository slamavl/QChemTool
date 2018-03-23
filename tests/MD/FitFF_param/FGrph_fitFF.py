# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:27:58 2018

@author: Vladislav Sláma
"""
import numpy as np
from scipy.optimize import minimize

from QChemTool.General.units import conversion_facs_position as conv_pos
from QChemTool.QuantumChem.Classes.structure import Structure
from QChemTool.QuantumChem.Fluorographene.fluorographene import Optimize_MD_AMBER_structure,get_border_carbons_FG

global frcmod_filename,struc,state,FF_param

optimize = False
charges = 'Hirshfeld'
state='Ground'
FF_param = {'equilibrium': {}, 'force': {}}
FF_param['equilibrium']['c3-c3'] = 1.50398  #1.5350
FF_param['equilibrium']['cb-c3'] = 1.58027  #1.5350
FF_param['equilibrium']['cb-cb'] = 1.58349  #1.5350
FF_param['equilibrium']['c3-f']  = 1.45975
FF_param['equilibrium']['c3-cb-c3'] =  99.781
FF_param['equilibrium']['cb-cb-c3'] =  93.821
FF_param['equilibrium']['c3-c3-c3'] = 112.575

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


RMSD = Optimize_MD_AMBER_structure(frcmod_filename,struc,state=state,gen_input=True,struc_out=False,**FF_param)
print("RMSD init:",RMSD*conv_pos["Angstrom"],"Angstroms")

def OptimizeRMSD(param):
    FF_param['equilibrium']['c3-c3'] = param[0]
    FF_param['equilibrium']['c3-f'] = param[1]
    FF_param['equilibrium']['cb-c3'] = param[2]
    FF_param['equilibrium']['cb-cb'] = param[3]
    FF_param['equilibrium']['c3-c3-c3'] = 110.6 + param[4] * 200
    FF_param['equilibrium']['c3-cb-c3'] = 100.16 + param[5] * 200
    FF_param['equilibrium']['cb-cb-c3'] = 94.18 + param[6] * 200
    print(FF_param)
    RMSD = Optimize_MD_AMBER_structure(frcmod_filename,struc,state=state,gen_input=False,struc_out=False,**FF_param)
    RMSD = RMSD*conv_pos["Angstrom"]
    print(RMSD)
    return RMSD

if optimize:
    min_method='SLSQP'
    options={'eps': 0.001} 
    #res = minimize(OptimizeRMSD,(FF_param['equilibrium']['c3-c3'],FF_param['equilibrium']['c3-f'],FF_param['equilibrium']['cb-c3'],FF_param['equilibrium']['c3-cb-c3-c3']),method=min_method,options=options)
    res = minimize(OptimizeRMSD,(FF_param['equilibrium']['c3-c3'],FF_param['equilibrium']['c3-f'],FF_param['equilibrium']['cb-c3'],FF_param['equilibrium']['cb-cb'],0.0,0.0,0.0),method=min_method,options=options)
    print(res)

# optimize structure
RMSD,struc_opt,struc_prepc = Optimize_MD_AMBER_structure(frcmod_filename,struc,state=state,gen_input=False,struc_out=True,**FF_param)
print("RMSD final:",RMSD*conv_pos["Angstrom"],"Angstroms")
struc_opt.output_to_xyz("struc_opt.xyz")
struc_prepc.output_to_xyz("struc_prepc.xyz")