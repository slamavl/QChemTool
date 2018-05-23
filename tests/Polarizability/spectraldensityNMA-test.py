# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:42:06 2018

@author: Vladislav Sláma
"""
import numpy as np
from QChemTool import Molecule
from QChemTool.QuantumChem.vibration import NormalModes_mol
from QChemTool.QuantumChem.Fluorographene.fluorographene import orientFG
from QChemTool.Development.polarizablesytem import PolarizableSystem

ElstatType='Hirshfeld'


DIR = "C:/PhD/Ab-initio-META/Fluorographane/Freq"
filename_fchk = "".join([DIR,"/FGrph_perylene_symm_9x6_opt_freq.fchk"])
filename_log = "".join([DIR,"/FGrph_perylene_symm_9x6_opt_freq.log"])

# load structure and vibrational information about molecule
print("Reading molecule info...")
mol = Molecule("perylene")
mol.load_Gaussian_fchk(filename_fchk)
print("Reading molecule info DONE")
# mol.load_Gaussian_log(filename_log)

# Calculate vibration normal modes
print("Calculating vibrational normal modes...")
[Freqcm1,RedMass,ForcesCm1Agstrom2,Int2Cart,Cart2Int,Units] = NormalModes_mol(mol)
mol.vib_spec['NMinCart']=Int2Cart
mol.vib_spec['CartInNM']=Cart2Int
mol.vib_spec['Frequency']=Freqcm1
mol.vib_spec['RedMass']=RedMass
mol.vib_spec['ForceConst']=ForcesCm1Agstrom2
mol.vib_spec['Nmodes']=len(Freqcm1)
print("Calculating vibrational normal modes DONE")

# Get oriented structure 
#      - reorientation of whole molecule is long (reorient only needed properties of make it faster)
print("Reorientation molecule...")
#print("Normal mode before rotation:")
#print(mol.vib_spec['NMinCart'][0:10,0])
#mol = orientFG(mol)
mol.rotate(0.0,np.pi/2.0,0.0)
#print("Normal mode after rotation:")
#print(mol.vib_spec['NMinCart'][0:10,0])
print("Reorientation molecule DONE")
mol.output_to_xyz("struc_reorient.xyz")

# Initialize the system
FG_charges = "ESPfit"
params_polar={"VinterFG": True,"coarse_grain": "plane",
              "charge_type": FG_charges,"approximation": 1.1,"symm": True}
elstat = {"structure": mol.struc,"charge": FG_charges}
diel  = {"structure": mol.struc,"polar": params_polar}
params = {"energy_type": "QC","permivity": 1.0,"order": 2}
system = PolarizableSystem(diel = diel, elstat = elstat, params = params)

# identify defects - separated because now changes can be made to the database
system.identify_defects()

print("Calculating system bth coupling...")
g00 = system.get_gmm(0,mol.vib_spec['NMinCart'],mol.vib_spec['Frequency'],mol.vib_spec['RedMass'])
print("Calculating system bth coupling DONE")

print("Frequencies:")
print(mol.vib_spec['Frequency'][0:5])

