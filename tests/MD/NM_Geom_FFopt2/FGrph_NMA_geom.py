# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:27:58 2018

@author: Vladislav Sláma
"""
import numpy as np
from scipy.optimize import minimize

from QChemTool.General.units import conversion_facs_position as conv_pos
from QChemTool.General.UnitsManager import position_units
from QChemTool.QuantumChem.Classes.structure import Structure
from QChemTool.QuantumChem.Fluorographene.fluorographene import get_AMBER_MD_normal_modes,get_border_carbons_FG,Optimize_MD_AMBER_structure
from QChemTool.QuantumChem.vibration import NormalModes_mol

global frcmod_filename,struc,state,FF_param

optimize = True
compare_w_gauss = True
charges = 'Hirshfeld'
state='Ground'
FF_param = {'equilibrium': {}, 'force': {}}

# Parameters obtained from fitting geometry
FF_param['equilibrium']['c3-c3'] = 1.46066  #1.5350
FF_param['equilibrium']['cb-c3'] = 1.53008  #1.5350
FF_param['equilibrium']['cb-cb'] = 1.57402  #1.5350
FF_param['equilibrium']['c3-f']  = 1.45404
FF_param['equilibrium']['c3-cb-c3'] =  95.525
FF_param['equilibrium']['cb-cb-c3'] =  88.017
FF_param['equilibrium']['c3-c3-c3'] = 107.153

# parameters obtained from normal mode calculation
FF_param['force']['c3-c3'] = 303.1   #166.584      # 303.1
FF_param['force']['cb-c3'] = 303.1   #147.702      # 303.1
FF_param['force']['cb-cb'] = 303.1   #261.708      # 303.1
FF_param['force']['c3-f']  = 363.8   #275.709      # 363.8
FF_param['force']['cb-fb']  = FF_param['force']['c3-f']
FF_param['force']['c3-c3-c3'] = 63.21 #38.1812   # 63.21
FF_param['force']['cb-c3-c3'] = FF_param['force']['c3-c3-c3']   # 63.21
FF_param['force']['c3-cb-c3'] = FF_param['force']['c3-c3-c3']   # 63.21
FF_param['force']['cb-cb-c3'] = FF_param['force']['c3-c3-c3']   # 63.21
FF_param['force']['cb-c3-cb'] = FF_param['force']['c3-c3-c3']   # 63.21
FF_param['force']['cb-cb-cb'] = FF_param['force']['c3-c3-c3']   # 63.21
FF_param['force']['c3-c3-f'] = 66.22 #23.0185   # 66.22
FF_param['force']['cb-cb-fb'] = FF_param['force']['c3-c3-f']    # 66.22
FF_param['force']['c3-cb-fb'] = FF_param['force']['cb-cb-fb']   # 66.22



# 'cb-cb-c3': [63.21,110.63],
# 'cb-c3-cb': [63.21,110.63], 'c3-cb-c3': [63.21,110.63],
# 'cb-cb-cb': [63.21,110.63], 'cb-cb-fb': [66.22,109.41],
# 'cb-c3-f': [66.22,109.41], 'c3-cb-fb': [66.22,109.41], 
# 'fb-cb-fb': [71.260,120.0], 'c3-c3-f': [66.22,109.41],
# 'f-c3-f': [71.260,210.0]

# Set FG charges
if charges == 'Hirshfeld':
    FG_charges = [0.08125,0.08125]

if compare_w_gauss or optimize:
    print("Reading gaussian input files... ")
    # read normal mode information from gaussian freq calculation
    log_filename = "/mnt/sda2/PhD/Ab-initio-META/Fluorographane/Freq/FGrph-5hex_opt_freqHP_reord.log"
    fchk_filename = "/mnt/sda2/PhD/Ab-initio-META/Fluorographane/Freq/FGrph-5hex_opt_freqHP_reord.fchk"
    
    from QChemTool.QuantumChem.Classes.molecule import Molecule
    mol_gauss = Molecule("Frequency calculation")
    mol_gauss.load_Gaussian_fchk(fchk_filename)
    mol_gauss.load_Gaussian_log(log_filename)
    freq_gauss = mol_gauss.vib_spec['Frequency']
    print("Finished reading gaussian input files. ")

# Load initial structure
struc = Structure()
struc.load_xyz("FGrph-5hex_opt_freqHP_reord.xyz")

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


NM_info, indx_orig2new_atoms = get_AMBER_MD_normal_modes(struc,state=state,gen_input=True,**FF_param)
RMSD = Optimize_MD_AMBER_structure("nab_input.frcmod",struc,state=state,gen_input=True,struc_out=False,**FF_param)
RMSD = RMSD*conv_pos["Angstrom"]
print(RMSD)
# print(indx_orig2new)
indx_orig2new = np.zeros((struc.nat,3),dtype='i8')
for ii in range(3):
    indx_orig2new[:,ii] = indx_orig2new_atoms*3+ii
indx_orig2new = indx_orig2new.reshape(3*struc.nat)
#print(indx_orig2new_atoms)
#print(indx_orig2new)


def relative_NM_difference(NM_info,int2cart_gauss):
    AngleMat = np.rad2deg( np.arccos( np.dot(NM_info['int2cart'].T,int2cart_gauss)))
    AngleMat = np.abs(AngleMat)
    indx_corr = np.zeros(AngleMat.shape[0],dtype='i8')
    angle = np.zeros(AngleMat.shape[0],dtype='i8')
    for ii in range(AngleMat.shape[0]):
        minmax = [np.argmin(AngleMat[ii,:]),np.argmax(AngleMat[ii,:])]
        if AngleMat[ii,minmax[0]] > np.abs(180-AngleMat[ii,minmax[1]]):
            index = minmax[1]
            angle[ii] = 180-AngleMat[ii,index]
        else:
            index = minmax[0]
            angle[ii] = AngleMat[ii,index]
        indx_corr[ii] = index
        
        
        # check if unique
        
    dist = np.sum( np.abs( NM_info['freq'] - freq_gauss[indx_corr] )/freq_gauss[indx_corr])

    return dist,indx_corr,angle


# reorder gaussian frequencies
if compare_w_gauss or optimize:
    print("Calculating normal modes from gaussian hessian... ")
    Freqcm1,RedMass,ForcesCm1Agstrom2,InternalToCartesian,CartesianToInternal,Units = NormalModes_mol(mol_gauss)
    int2cart_gauss = InternalToCartesian[indx_orig2new,:]
    print("Finished calculating normal modes. ")

    if 0:
        dif_vec = np.sum(np.linalg.norm(NM_info['int2cart'] - int2cart_gauss, axis=0) )    # sum of diferences
    else:
        dif_vec = np.sum( np.arccos( np.sum( NM_info['int2cart'] * int2cart_gauss ,axis=0)  ) ) # sum of angles (in radians)
        
    dist, indx_corr, angle = relative_NM_difference(NM_info,int2cart_gauss)
#    print(indx_corr)
#    print(dist)
#    print(angle)


def compare_NM(param):
    print(param)
    FF_param['force']['c3-c3'] = abs(param[0])
    FF_param['force']['cb-c3'] = abs(param[1])
    FF_param['force']['cb-cb'] = abs(param[2])
    FF_param['force']['c3-f']  = abs(param[3])
    FF_param['force']['cb-fb']  = FF_param['force']['c3-f']
    FF_param['force']['c3-c3-c3'] = abs(param[4])
    FF_param['force']['cb-c3-c3'] = FF_param['force']['c3-c3-c3']
    FF_param['force']['c3-cb-c3'] = FF_param['force']['c3-c3-c3']
    FF_param['force']['cb-cb-c3'] = FF_param['force']['c3-c3-c3']
    FF_param['force']['cb-c3-cb'] = FF_param['force']['c3-c3-c3']
    FF_param['force']['cb-cb-cb'] = FF_param['force']['c3-c3-c3']
    FF_param['force']['c3-c3-f'] = abs(param[5])
    FF_param['force']['cb-cb-fb'] = FF_param['force']['c3-c3-f']
    FF_param['force']['c3-cb-fb'] = FF_param['force']['cb-cb-fb']
    print(FF_param)
    NM_info, indx_orig2new = get_AMBER_MD_normal_modes(struc,state=state,gen_input=False,**FF_param)
    
    diff_freq = np.sum(np.abs( freq_gauss - NM_info['freq'] )/ freq_gauss)
    if 1:
        dif_vec = np.sum(np.linalg.norm(NM_info['int2cart'] - int2cart_gauss, axis=0) )    # sum of diferences
    else:
        dif_vec = np.sum( np.arccos( np.sum( NM_info['int2cart'] * int2cart_gauss ,axis=0)  ) ) # sum of angles (in radians)
    
    dist, indx_corr, angle = relative_NM_difference(NM_info,int2cart_gauss)
    # dist += np.sum(angle) / 180.0
    
    print('\n')
    print(diff_freq)
    print(dist)
    print('\n')
    return dist

def OptimizeRMSD(param):
    FF_param['equilibrium']['c3-c3'] = param[0]
    FF_param['equilibrium']['c3-f'] = param[1]
    FF_param['equilibrium']['cb-c3'] = param[2]
    FF_param['equilibrium']['cb-cb'] = param[3]
    FF_param['equilibrium']['c3-c3-c3'] = 110.6 + param[4] * 200
    FF_param['equilibrium']['c3-cb-c3'] = 100.16 + param[5] * 200
    FF_param['equilibrium']['cb-cb-c3'] = 94.18 + param[6] * 200
    print(FF_param)
    RMSD = Optimize_MD_AMBER_structure("nab_input.frcmod",struc,state=state,gen_input=False,struc_out=False,**FF_param)
    RMSD = RMSD*conv_pos["Angstrom"]
    print(RMSD)
    return RMSD

if optimize:
    for ii in range(15):
        min_method='SLSQP'
        
        options={'eps': 0.1, "maxiter": 300} 
        res = minimize(compare_NM,(FF_param['force']['c3-c3'],FF_param['force']['cb-c3'],FF_param['force']['cb-cb'],FF_param['force']['c3-f'],FF_param['force']['c3-c3-c3'],FF_param['force']['c3-c3-f']),method=min_method,options=options)
        print(res)
        
        options={'eps': 0.001}
        res = minimize(OptimizeRMSD,(FF_param['equilibrium']['c3-c3'],FF_param['equilibrium']['c3-f'],FF_param['equilibrium']['cb-c3'],FF_param['equilibrium']['cb-cb'],0.0,0.0,0.0),method=min_method,options=options)
        print(res)
        

    NM_info, indx_orig2new = get_AMBER_MD_normal_modes(struc,state=state,gen_input=False,**FF_param)

# plot histogram 
import matplotlib.pyplot as plt
step = 50
bins = np.arange(0,max(NM_info['freq'][-1],freq_gauss[-1]),50.0)
plt.hist(NM_info['freq'], alpha=0.5, normed=False, bins=bins, label='AMBER MD')
if compare_w_gauss or optimize:
    plt.hist(freq_gauss, alpha=0.5, normed=False, bins=bins, label='Gaussian09')
plt.xlabel('Frequency');
plt.xlabel('Count');
plt.show()


#    NM_info["int2cart"] = InternalToCartesian
#    NM_info["cart2int"] = CartesianToInternal
#    NM_info["freq"] = Freqcm1
#    NM_info["RedMass"] = RedMass
#    NM_info['force'] = ForcesCm1Agstrom2
#    NM_info['units'] = {"freq": "1/cm", "RedMass": "AMU(atomic mass units)",
#           "force": "1/(cm * Angstrom^2)", "int2cart": "dimensionles",
#           'cart2int': "dimensionles"}
# {'equilibrium': {'cb-cb-c3': 93.821, 'c3-c3-c3': 112.575, 'c3-cb-c3': 99.781, 'c3-f': 1.45975, 'c3-c3': 1.50398, 'cb-cb': 1.58349, 'cb-c3': 1.58027}, 'force': {'cb-cb-c3': 46.161964772446098, 'c3-cb-c3': 46.161964772446098, 'cb-cb-cb': 46.161964772446098, 'c3-c3-f': 38.338517296715168, 'c3-c3-c3': 46.161964772446098, 'c3-c3': 149.92644957647627, 'cb-cb-fb': 38.338517296715168, 'c3-f': 281.97958014222456, 'cb-c3-c3': 46.161964772446098, 'cb-fb': 281.97958014222456, 'c3-cb-fb': 38.338517296715168, 'cb-cb': 281.64140482308568, 'cb-c3-cb': 46.161964772446098, 'cb-c3': 133.4506671439552}}

# {'equilibrium': {'cb-cb-c3': 88.017241498952885, 'c3-c3-c3': 107.15294657539577, 'c3-cb-c3': 95.525131045009203, 'c3-f': 1.4540369721579582, 'c3-c3': 1.4606626390987845, 'cb-cb': 1.5740194652571133, 'cb-c3': 1.5300819088244113}, 'force': {'cb-cb-c3': 38.181232448725083, 'c3-cb-fb': 23.018474159608967, 'c3-cb-c3': 38.181232448725083, 'cb-cb': 261.70794675559847, 'c3-c3-f': 23.018474159608967, 'c3-c3-c3': 38.181232448725083, 'c3-c3': 166.58376517157259, 'cb-cb-fb': 23.018474159608967, 'c3-f': 275.70889394317834, 'cb-c3-c3': 38.181232448725083, 'cb-fb': 275.70889394317834, 'cb-c3-cb': 38.181232448725083, 'cb-cb-cb': 38.181232448725083, 'cb-c3': 147.70243577927974}}


# {'equilibrium': {'cb-cb-c3': 83.002735221304803, 'c3-cb-c3': 91.953627226648081, 'c3-f': 1.4600792858235836, 'cb-cb': 1.5830393950969697, 'c3-c3-c3': 111.52980555393397, 'c3-c3': 1.4966281114555786, 'cb-c3': 1.5824458486038846}, 'force': {'cb-cb-c3': 37.607329454133364, 'c3-cb-c3': 37.607329454133364, 'c3-f': 333.61967877205979, 'c3-c3-c3': 37.607329454133364, 'cb-c3-cb': 37.607329454133364, 'cb-c3': 322.57965539078702, 'c3-c3-f': 56.164723495359347, 'cb-cb-cb': 37.607329454133364, 'cb-fb': 333.61967877205979, 'cb-cb': 314.764473231419, 'c3-cb-fb': 56.164723495359347, 'c3-c3': 263.10858646606357, 'cb-cb-fb': 56.164723495359347, 'cb-c3-c3': 37.607329454133364}}
# {'equilibrium': {'cb-cb-c3': 83.404720470460035, 'c3-cb-c3': 92.207727759399972, 'c3-f': 1.4598453348432738, 'cb-cb': 1.5827041906234309, 'c3-c3-c3': 111.40233200658345, 'c3-c3': 1.4965158732230384, 'cb-c3': 1.5826003612640041}, 'force': {'cb-cb-c3': 38.142332310822596, 'c3-cb-c3': 38.142332310822596, 'c3-f': 334.05827604837214, 'c3-c3-c3': 38.142332310822596, 'cb-c3-cb': 38.142332310822596, 'cb-c3': 321.8363136727844, 'c3-c3-f': 55.486818374753867, 'cb-cb-cb': 38.142332310822596, 'cb-fb': 334.05827604837214, 'cb-cb': 317.39659827235744, 'c3-cb-fb': 55.486818374753867, 'c3-c3': 262.36090055711145, 'cb-cb-fb': 55.486818374753867, 'cb-c3-c3': 38.142332310822596}}



