# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 17:52:32 2018

@author: Vladislav Sláma
"""
import numpy as np

from QChemTool.QuantumChem.read_mine import read_qchem_esp, read_gaussian_esp
from QChemTool.General.UnitsManager import position_units
from QChemTool.QuantumChem.esp import RESP_fit_charges,ESP_get_quadrupole
from QChemTool.QuantumChem.Classes.structure import Structure
import matplotlib.pyplot as plt

# load structure for fitting the charges
struc = Structure()
struc.load_xyz("perylene_opt_TD-DFT_wB97XD_lanl2dz.xyz")

# load AMBER results:
results = Structure()
results.load_mol2("perylene_wB97XD.mol2")

# load electrostatic potential
Points,ESP,Coor,Charges = read_gaussian_esp("perylene_opt_esp_wB97XD-ini.esp",output_charge=True)

# fit charges
res = RESP_fit_charges(struc,Points,ESP)

charge = res["charge"]
coor = res["coor"]
dipole = res["dipole"]
quad_traceles = res["quadrupole_traceless"]
quad = res["quadrupole"]
quad_traceles = np.array(quad_traceles) #* 2.541765 * 0.52917725
quad = np.array(quad) #* 2.541765 * 0.52917725
steps=np.arange(res["steps"])+1
print("Dipole:",dipole*2.541765)
print("Quadrupole traceless:")
print(quad_traceles[0],quad_traceles[1],quad_traceles[2])
print(quad_traceles[1],quad_traceles[3],quad_traceles[4])
print(quad_traceles[2],quad_traceles[4],quad_traceles[5])

print("Quadrupole:")
print(quad[0],quad[1],quad[2])
print(quad[1],quad[3],quad[4])
print(quad[2],quad[4],quad[5])

# print results
for ii in range(struc.nat):
    print(results.esp_grnd[ii] - charge[ii])
print("Sum of all charges:",np.sum(charge))
with position_units("Angstrom"):
    for ii in range(struc.nat):
        print(coor.value[ii],results.coor.value[ii])


plt.plot(steps,res["RRMS"])
plt.title("Relative root mean square error")
plt.show()
plt.plot(steps,res["Chi_square"])
plt.title("Chi square")
plt.show()
plt.plot(steps,res["q_change"])
plt.title("Mean charge change")
plt.show()

print("Result from ESP fiting function:")
quad_test,quad = ESP_get_quadrupole(Coor,Charges)
quad_test = np.array(quad_test) * 2.541765 * 0.52917725
print("   XX=",quad_test[0],"   YY=",quad_test[3],"   ZZ=",quad_test[5])
print("   XY=",quad_test[1],"   XZ=",quad_test[2],"   YZ=",quad_test[4])

print("Result from AMBER RESP fiting:")
print("   XX=",-55.04654,"   YY=",27.70042,"   ZZ=",27.34612)

print("Result from Gaussian ESP fiting:")
print("   XX=",-14.790405,"   YY=",7.4598493,"   ZZ=",7.3305554)
print("   XY=",1.0840891e-13,"   XZ=",-6.4562661e-14,"   YZ=",-1.4252488e-12)

print("Result from Gaussian calculation:")
print("   XX=",-19.8936,"   YY=",10.0338,"   ZZ=",9.8598)
print("   XY=",0.0,"   XZ=",0.0,"   YZ=",0.0)

# TRACELESS QUADRUPOLE MOMENT:
#   XX= -0.14790405D+02   YY=  0.74598493D+01   ZZ=  0.73305554D+01
#   XY=  0.10840891D-14   XZ= -0.64562661D-15   YZ= -0.14252488D-13

# GAUSSIAN result
# Traceless Quadrupole moment (field-independent basis, Debye-Ang):
#   XX=            -19.8936   YY=             10.0338   ZZ=              9.8598
#   XY=              0.0000   XZ=              0.0000   YZ=              0.0000

# AMBER OUTPUT
# Quadrupole (Debye*Angst.):
# Qxx = -55.04654     QYY =  27.70042     QZZ =  27.34612
# Qxy = -55.04654     QXZ = -55.04654     QYZ = -55.04654