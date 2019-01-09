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
from QChemTool.QuantumChem.Fluorographene.fluorographene import get_AMBER_MD_normal_modes,get_border_carbons_FG,Optimize_MD_AMBER_structure,constrainsFG
from QChemTool.QuantumChem.vibration import NormalModes_mol

from QChemTool.QuantumChem.Fluorographene.fluorographene import orientFG
from QChemTool.Development.polarizablesytem_periodic import PolarizableSystem
from QChemTool.Spectroscopy.spectraldensity import SpectralDensity
from QChemTool import FrequencyAxis
from QChemTool.General.UnitsManager import frequency_units
from QChemTool.QuantumChem.positioningTools import project_on_plane,fit_plane

global frcmod_filename,struc,state,FF_param
import matplotlib.pyplot as plt

optimize = False
optimize_type = 'Geom'  # 'Geom', 'Freq', 'Geom+Freq'
compare_w_gauss = False
IsotropicPolar = True
charges = 'ESPfit'
state='Ground'
FF_param = {'equilibrium': {}, 'force': {}}

# =============================================================================
# Specify parameters with different then default FF value
# =============================================================================

# Parameters obtained from fitting geometry
FF_param['equilibrium']['c3-c3'] = 1.49667 #1.49652  #1.5350
FF_param['equilibrium']['cb-c3'] = 1.58288 #1.58260  #1.5350
FF_param['equilibrium']['cb-cb'] = 1.58356 #1.58270  #1.5350
FF_param['equilibrium']['c3-f']  = 1.46121 #1.45985
FF_param['equilibrium']['c3-cb-c3'] =  63.0473 #92.2077
FF_param['equilibrium']['cb-cb-c3'] =  44.4324 #83.4047
FF_param['equilibrium']['c3-c3-c3'] = 106.0817 #111.4023
FF_param['equilibrium']['ca-ca']  = 1.43 #1.45985 
FF_param['equilibrium']['ca-c3']  = 1.54 #1.45985
FF_param['equilibrium']['ca-ca-c3']  = 130.71 # 120.63
FF_param['equilibrium']['ca-c3-c3']  = 131.63 # 112.09

# parameters obtained from normal mode calculation
FF_param['force']['c3-c3'] = 259.68 #262.36   #166.584      # 303.1
FF_param['force']['cb-c3'] = 327.07 #321.84   #147.702      # 303.1
FF_param['force']['cb-cb'] = 304.61 #317.40  #261.708      # 303.1
FF_param['force']['c3-f']  = 335.36 #334.06   #275.709      # 363.8
FF_param['force']['cb-fb']  = FF_param['force']['c3-f']
FF_param['force']['c3-c3-c3'] = 13.89 #38.14 #38.1812   # 63.21
FF_param['force']['cb-c3-c3'] = FF_param['force']['c3-c3-c3']   # 63.21
FF_param['force']['c3-cb-c3'] = FF_param['force']['c3-c3-c3']   # 63.21
FF_param['force']['cb-cb-c3'] = FF_param['force']['c3-c3-c3']   # 63.21
FF_param['force']['cb-c3-cb'] = FF_param['force']['c3-c3-c3']   # 63.21
FF_param['force']['cb-cb-cb'] = FF_param['force']['c3-c3-c3']   # 63.21
FF_param['force']['c3-c3-f'] = 40.97 #55.49 #23.0185   # 66.22
FF_param['force']['cb-cb-fb'] = FF_param['force']['c3-c3-f']    # 66.22
FF_param['force']['c3-cb-fb'] = FF_param['force']['cb-cb-fb']   # 66.22

# =============================================================================
# 
# =============================================================================

# Set FG charges
if charges == 'Hirshfeld':
    FG_charges = [0.08125,0.08125]  # FG_charges[0]=inner carbon charge, FG_charges[0]= - border fluorine charge
elif charges == "ESPfit":
    FG_charges = [-0.0522,-0.0522]


if compare_w_gauss or optimize:
    print("Reading gaussian input files... ")
    # read normal mode information from gaussian freq calculation
    DIR = "/mnt/sda2/PhD/Ab-initio-META/Fluorographane/Freq"
    fchk_filename = "".join([DIR,"/FGrph_perylene_symm_9x6_opt_freq.fchk"])
    log_filename = "".join([DIR,"/FGrph_perylene_symm_9x6_opt_freq.log"])
    
    from QChemTool.QuantumChem.Classes.molecule import Molecule
    mol_gauss = Molecule("Frequency calculation")
    mol_gauss.load_Gaussian_fchk(fchk_filename)
    mol_gauss.load_Gaussian_log(log_filename)
    freq_gauss = mol_gauss.vib_spec['Frequency']
    print("Finished reading gaussian input files. ")

# Load initial structure
struc = Structure()
struc.load_xyz("FG_per_small.xyz")

# assign charges
border_C_indx,border_F_indx = get_border_carbons_FG(struc)
if state=='Ground':
    struc.esp_grnd = np.zeros(struc.nat,dtype='f8')
    charges = struc.esp_grnd # pointer to structure charges
elif state=='Excited':
    struc.esp_exct = np.zeros(struc.nat,dtype='f8')
    charges = struc.esp_exct # pointer to structure charges
elif state=='Transition':
    struc.esp_trans = np.zeros(struc.nat,dtype='f8')
    charges = struc.esp_trans # pointer to structure charges
    
for ii in range(struc.nat):
    if struc.at_type[ii] == 'C':
        charges[ii] = FG_charges[0]
    elif struc.at_type[ii] == 'F':
        charges[ii] = -FG_charges[0]
    else:
        raise Warning("Unknown atom type in structure")
charges[border_C_indx] = 2*FG_charges[1]
charges[border_F_indx] = -FG_charges[1]

def_indx = constrainsFG(struc,border=False,defect=True) 
charges[def_indx] = 0.0

# Flatten the defect
nvec,origin = fit_plane(struc.coor._value[def_indx,:] )
struc.coor._value[def_indx,:] = project_on_plane(struc.coor._value[def_indx,:], nvec,origin) 

# get defect indexes
use_VinterFG = False
FG_charges = 'ESPfit'
CoarseGrain="plane"
params_polar={"VinterFG": use_VinterFG,"coarse_grain": CoarseGrain,
              "charge_type": FG_charges,"approximation": 1.1,"symm": IsotropicPolar}
elstat = {"structure": struc,"charge": FG_charges}
diel  = {"structure": struc,"polar": params_polar}
params = {"energy_type": "QC","permivity": 1.0,"order": 2}
system = PolarizableSystem(diel = diel, elstat = elstat, params = params)

# identify defects - separated because now changes can be made to the database
system.identify_defects()

# set ground state charges on defects
for defect in system.defects:
    charges[defect.index] = defect.get_charge(state='ground')
    print(charges[defect.index])

# =============================================================================
# CALCULATION
# =============================================================================

# Optimize structure
NM_info, indx_orig2new_atoms, indx_new2orig_atoms = get_AMBER_MD_normal_modes(struc,state=state,gen_input=True,**FF_param)
#RMSD,struc,struc_old = Optimize_MD_AMBER_structure("nab_input.frcmod",struc,state=state,gen_input=True,struc_out=True,**FF_param)
struc = NM_info['struc']
#RMSD = RMSD*conv_pos["Angstrom"]
#print(RMSD)

indx_orig2new = np.zeros((struc.nat,3),dtype='i8')
indx_new2orig = np.zeros((struc.nat,3),dtype='i8')
for ii in range(3):
    indx_orig2new[:,ii] = indx_orig2new_atoms*3+ii
    indx_new2orig[:,ii] = indx_new2orig_atoms*3+ii
indx_orig2new = indx_orig2new.reshape(3*struc.nat)
indx_new2orig = indx_new2orig.reshape(3*struc.nat)

# =============================================================================
# Calculate difference between MM and QC NMA
# =============================================================================
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
    #int2cart_gauss = InternalToCartesian[indx_orig2new,:]
    int2cart_gauss = InternalToCartesian
#    NM_info['int2cart'] = NM_info['int2cart'][indx_new2orig,:]
    print("Finished calculating normal modes. ")

    if 0:
        dif_vec = np.sum(np.linalg.norm(NM_info['int2cart'] - int2cart_gauss, axis=0) )    # sum of diferences
    else:
        dif_vec = np.sum( np.arccos( np.sum( NM_info['int2cart'] * int2cart_gauss ,axis=0)  ) ) # sum of angles (in radians)
        
    dist, indx_corr, angle = relative_NM_difference(NM_info,int2cart_gauss)

# =============================================================================
# Plot results
# =============================================================================
# reorder gaussian frequencies
if compare_w_gauss or optimize:
    # plot histogram 
    plt.figure(2,figsize=(12,8))
    step = 50
    bins = np.arange(0,max(NM_info['freq'][-1],freq_gauss[-1]),50.0)
    plt.hist(NM_info['freq'], alpha=0.5, normed=False, bins=bins, label='AMBER MD')
    if compare_w_gauss or optimize:
        plt.hist(freq_gauss, alpha=0.5, normed=False, bins=bins, label='Gaussian09')
        plt.legend(["MD results","QC results"])
    else:
        plt.legend(["MD results"])
    plt.xlabel('Frequency');
    plt.xlabel('Count');
    plt.show()


print("Reorientation molecule...")
#struc = orientFG(struc)
# reorder gaussian frequencies
if compare_w_gauss or optimize:
    mol_gauss.rotate(0.0,np.pi/2.0,0.0)
    mol_gauss.output_to_xyz("struc_reorient_gauss.xyz")
struc.output_to_xyz("struc_reorient_FG.xyz")
print("Reorientation molecule DONE")

# Initialize the system
use_VinterFG = False
FG_charges = 'ESPfit'
CoarseGrain="plane"
params_polar={"VinterFG": use_VinterFG,"coarse_grain": CoarseGrain,
              "charge_type": FG_charges,"approximation": 1.1,"symm": IsotropicPolar}
elstat = {"structure": struc,"charge": FG_charges}
diel  = {"structure": struc,"polar": params_polar}
params = {"energy_type": "QC","permivity": 1.0,"order": 2}
system = PolarizableSystem(diel = diel, elstat = elstat, params = params)

# identify defects - separated because now changes can be made to the database
system.identify_defects()

# correct for the negative frequency
for ii in range(len(NM_info["freq"])):
    MASK = np.where(NM_info["freq"]>0.0)

print("Calculating system bth coupling...")
# Calculate coupling MM
Int2Cart = NM_info["int2cart"][:,MASK]
Freq = NM_info["freq"][MASK]
RedMass = NM_info["RedMass"][MASK]
print(system.get_SingleDefectProperties(0)[0].value)
g00 = system.get_gmm(0,Int2Cart,Freq,RedMass)
print("Calculating system bth coupling DONE")

# reorder gaussian frequencies
if compare_w_gauss or optimize:
    # Initialize the system
    use_VinterFG = False
    FG_charges = 'ESPfit'
    CoarseGrain="plane"
    params_polar={"VinterFG": use_VinterFG,"coarse_grain": CoarseGrain,
                  "charge_type": FG_charges,"approximation": 1.1,"symm": IsotropicPolar}
    elstat = {"structure": mol_gauss.struc,"charge": FG_charges}
    diel  = {"structure": mol_gauss.struc,"polar": params_polar}
    params = {"energy_type": "QC","permivity": 1.0,"order": 2}
    system_gauss = PolarizableSystem(diel = diel, elstat = elstat, params = params)
    
    # identify defects - separated because now changes can be made to the database
    system_gauss.identify_defects()
    
    print("Calculating system bth coupling Gauss...")
    # correct for the negative frequency
    for ii in range(len(Freqcm1)):
        MASK = np.where(Freqcm1>0.0)
        
    # Calculate coupling Gaussinan
    Int2Cart_gauss = mol_gauss.vib_spec['NMinCart'][:,MASK]
    Freq_gauss = mol_gauss.vib_spec['Frequency'][MASK]
    RedMass_gauss = mol_gauss.vib_spec['RedMass'][MASK]
    print(system_gauss.get_SingleDefectProperties(0)[0].value)
    g00_gauss = system_gauss.get_gmm(0,Int2Cart_gauss,Freq_gauss,RedMass_gauss)
    print("Calculating system bth coupling Gauss DONE")
    
    params_specdens_gauss={"sheer": 25.0, "coupling": g00_gauss, "freq": Freq_gauss, "ftype": "DiscreteModes"}

params_specdens={"sheer": 25.0, "coupling": g00, "freq": Freq, "ftype": "DiscreteModes"}


start=-5.0
step=0.0005
length=20000

plt.figure(3,figsize=(12,8))
freq_axis = FrequencyAxis(start=start,length=length,step=step,time_start=0.0)
with frequency_units("1/cm"):
    specdens = SpectralDensity(axis = freq_axis, params = params_specdens)
    specdens.plot()
    # reorder gaussian frequencies
    if compare_w_gauss or optimize:
        specdens_gauss = SpectralDensity(axis = freq_axis, params = params_specdens_gauss)
        specdens_gauss.plot()
    plt.legend(["MD calculation","QC calculation"])

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



#{'equilibrium': {'c3-c3': 1.4966680911551919, 'c3-cb-c3': 63.047340462610443, 'cb-c3': 1.5828756492868352, 'cb-cb': 1.5835608546924587, 'cb-cb-c3': 44.432428791147352, 'c3-f': 1.4612061033677017, 'c3-c3-c3': 106.08166760218445}, 'force': {'c3-c3-f': 40.973122024504754, 'c3-c3': 259.68257919280018, 'cb-c3-cb': 13.889921204782778, 'cb-c3': 327.07344313991069, 'cb-cb-c3': 13.889921204782778, 'c3-cb-c3': 13.889921204782778, 'cb-c3-c3': 13.889921204782778, 'cb-fb': 335.36014666185804, 'cb-cb-cb': 13.889921204782778, 'cb-cb': 304.61438860703259, 'c3-cb-fb': 40.973122024504754, 'c3-f': 335.36014666185804, 'cb-cb-fb': 40.973122024504754, 'c3-c3-c3': 13.889921204782778}}
#112.865058446
#128.382625987