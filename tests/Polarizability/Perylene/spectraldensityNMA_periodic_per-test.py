# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:42:06 2018

@author: Vladislav Sláma
"""
import numpy as np
from QChemTool import Molecule, Structure
from QChemTool.QuantumChem.vibration import NormalModes_mol
from QChemTool.QuantumChem.Fluorographene.fluorographene import orientFG
from QChemTool.Development.polarizablesytem_periodic import PolarizableSystem
from QChemTool.Spectroscopy.spectraldensity import SpectralDensity
from QChemTool import FrequencyAxis
from QChemTool.General.UnitsManager import frequency_units
from QChemTool.General.units import conversion_facs_energy,conversion_facs_mass

DIR = "C:/PhD/Ab-initio-META/Fluorographane/Freq"
filename_fchk = "".join([DIR,"/Perylene_freq.fchk"])
filename_log = "".join([DIR,"/Perylene_freq.log"])



# load structure and vibrational information about molecule
print("Reading molecule info...")
mol = Molecule("perylene")
mol.load_Gaussian_fchk(filename_fchk)
print("Reading molecule info DONE")
# mol.load_Gaussian_log(filename_log)

# Calculate vibration normal modes
print("Calculating vibrational normal modes...")
[Freqcm1,RedMass,ForcesCm1Agstrom2,Int2Cart,Cart2Int,Units] = NormalModes_mol(mol)

# correct for the negative frequency
for ii in range(len(Freqcm1)):
    MASK = np.where(Freqcm1>0.0)

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
mol.rotate(0.0,np.pi/2.0,0.0)
print("Reorientation molecule DONE")
mol.output_to_xyz("Perylene_reorient.xyz")

struc=Structure()
struc.load_xyz("Perylene_NoH.xyz")

# Initialize the system
FG_charges = "ESPfit"
IsotropicPolar = False
use_VinterFG = False
CoarseGrain="plane"
params_polar={"VinterFG": use_VinterFG,"coarse_grain": CoarseGrain,
              "charge_type": FG_charges,"approximation": 1.1,"symm": IsotropicPolar}
elstat = {"structure": struc,"charge": FG_charges}
diel  = {"structure": struc,"polar": params_polar}
params = {"energy_type": "QC","permivity": 1.0,"order": 2}
system = PolarizableSystem(diel = diel, elstat = elstat, params = params)

# identify defects - separated because now changes can be made to the database
system.identify_defects()

print("Calculating system bth coupling...")
Int2Cart = mol.vib_spec['NMinCart'][0:60,MASK]
Freq = mol.vib_spec['Frequency'][MASK]
RedMass = mol.vib_spec['RedMass'][MASK]
g00 = system.get_gmm(0,Int2Cart,Freq,RedMass)
print("Calculating system bth coupling DONE")

omega_au = Freq/conversion_facs_energy["1/cm"]
RedMass_au = RedMass/conversion_facs_mass["AMU"]

# Energy difference along normal mode 1
NMI = 7
dq = 0.1
dE_Ha = g00[NMI]*np.sqrt(2*RedMass_au[NMI]*omega_au[NMI]**3)*dq
print('Energy difference should be:',dE_Ha*conversion_facs_energy["1/cm"],'cm-1')

# Generate structure
dR = np.reshape(mol.vib_spec['NMinCart'][:,NMI]*dq,(mol.struc.nat,3))
struc_shifted = mol.struc.copy()
struc_shifted.coor._value += dR
struc_shifted.output_to_xyz('Perylene_shifted.xyz')