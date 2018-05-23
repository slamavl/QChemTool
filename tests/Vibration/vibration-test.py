# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 18:57:27 2018

@author: Vladislav Sláma
"""

''' Normal modes - read'''
from QChemTool.QuantumChem.Classes.molecule import Molecule
from QChemTool.QuantumChem.vibration import NormalModes_mol
from QChemTool.General.units import conversion_facs_position,conversion_facs_energy
from sys import platform
from math import isclose
import numpy

mol=Molecule('TEST')
#molecule.QchMolecule.vib_spec[]
if (platform=='cygwin' or platform=="linux" or platform == "linux2"):
    MolDir='/mnt/sda2/Dropbox/PhD/Programy/Python/Test/'
elif platform=='win32':
    MolDir='C:/Dropbox/PhD/Programy/Python/Test/'
mol.load_Gaussian_fchk("".join([MolDir,'ethen_freq.fchk']))
mol.load_Gaussian_log("".join([MolDir,'ethen_freq.log']))
Freq=[832.2234,961.4514,976.8042,1069.4514,1241.7703,1388.7119,1483.4161,
      1715.6638,3144.5898,3160.3419,3220.2977,3246.2149]    
RedMass=[1.0427,1.5202,1.1607,1.0078,1.5261,1.214,1.112,3.1882,1.0477,
         1.0748,1.1146,1.1177]
Forces=[0.4255,0.8279,0.6525,0.6791,1.3865,1.3794,1.4417,5.5292,6.104,
        6.3247,6.8102,6.9394]
Hessian=[[6.38411997e-01, 2.75141057e-05,-5.37681873e-13,-2.66270576e-01,
          1.21243801e-01, 1.07762585e-12,-2.66282120e-01,-1.21273001e-01,
          -8.60166016e-13,-1.13450589e-01,-1.02575193e-06,-1.03742764e-12,
          3.79460942e-03, 2.99655336e-02, 1.33045136e-12, 3.79667933e-03,
          -2.99628217e-02,-5.05861479e-14],
          [2.75141057e-05,8.78790767e-01, 9.97851160e-13, 1.25022780e-01,
           -1.31318626e-01,-3.07467951e-13,-1.25051703e-01,-1.31342208e-01,
           -1.22702053e-12,-6.29420112e-07,-5.90037931e-01, 2.64148199e-12,
           -2.06887340e-03,-1.30465096e-02,-1.63484391e-12, 2.07091150e-03,
           -1.30454932e-02,-3.17165323e-13]]
test=True
for ii in range(len(Freq)):
    if (not isclose(mol.vib_spec['Frequency'][ii],Freq[ii],abs_tol=1e-4)) or \
       (not isclose(mol.vib_spec['RedMass'][ii],RedMass[ii],abs_tol=1e-4)) or \
       (not isclose(mol.vib_spec['ForceConst'][ii],Forces[ii],abs_tol=1e-4)):
        print(ii,mol.vib_spec['Frequency'][ii],Freq[ii],mol.vib_spec['RedMass'][ii],RedMass[ii],mol.vib_spec['ForceConst'][ii],Forces[ii])
        test=False 
for ii in range(2):
    for jj in range(len(Hessian[ii])):
        if (not isclose(mol.vib_spec['Hessian'][ii,jj],Hessian[ii][jj],abs_tol=1e-7)):
            test=False 
if mol.vib_spec['Nmodes']!=len(Freq):
    test=False
if test:
    print('Normal mode read     ...    OK')
else:
    print('Normal mode read     ...    Error')
    
''' Normal modes - molecule'''
Freqcm1,RedMass,ForcesCm1Agstrom2,InternalToCartesian,CartesianToInternal,Units=NormalModes_mol(mol)
test=True
for ii in range(len(Freq)):
    if (not isclose(mol.vib_spec['Frequency'][ii],Freq[ii],abs_tol=1e-4)) or \
       (not isclose(mol.vib_spec['RedMass'][ii],RedMass[ii],abs_tol=1e-4)):
           test=False

IdentMat=numpy.identity(mol.vib_spec['Nmodes'])
IdentMat2=numpy.dot(CartesianToInternal,InternalToCartesian)
for i in range(mol.vib_spec['Nmodes']):
    for j in range(mol.vib_spec['Nmodes']):
        if (not isclose(IdentMat[i,j],IdentMat2[i,j],abs_tol=1e-6)):
            test=False
        
if test:
    print('NormalModes_mol      ...    OK')
else:
    print('NormalModes_mol      ...    Error')
    
#print(' ')
#print(RedMass[0]*(Freqcm1[0]**2)*const.Freqcm_1ToForcecmA_2,RedMass[1]*(Freqcm1[1]**2)*const.Freqcm_1ToForcecmA_2)
#print(ForcesCm1Agstrom2[0],ForcesCm1Agstrom2[1])
    
# TODO: check some small normal mode displacement along first normal mode coordinate
struc_new = mol.struc.copy()
struc_new.coor._value += 0.01 * numpy.reshape(InternalToCartesian[:,0],(struc_new.nat,3))
struc_new.output_to_xyz("ethen_NormalMode1_0.01au.xyz")
E_dx = -78.5938062
E = -78.5938076

# energy difference for one angstrom
E_diff = (E_dx-E)/(0.01*conversion_facs_position["Angstrom"])**2*conversion_facs_energy["1/cm"]
print(E_diff,ForcesCm1Agstrom2[0]/2)

omega_au = Freqcm1/conversion_facs_energy["1/cm"]
RedMass_au = RedMass*1822.8886154
force_au = omega_au**2*RedMass_au
Energy_au = 1/2 * force_au[0] * 0.01**2
print(E_dx-E,Energy_au)

ForcesHaBohr2 = ForcesCm1Agstrom2/conversion_facs_energy["1/cm"]*(conversion_facs_position["Angstrom"]**2)
print(Freqcm1)
print( numpy.sqrt(ForcesHaBohr2/RedMass/1822.8886154)*conversion_facs_energy["1/cm"] )