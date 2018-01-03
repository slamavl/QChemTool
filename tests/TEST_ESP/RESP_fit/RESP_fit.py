# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 17:52:32 2018

@author: Vladislav Sláma
"""
import numpy as np

from QChemTool.QuantumChem.read_mine import read_qchem_esp, read_gaussian_esp
from QChemTool.General.UnitsManager import position_units
from QChemTool.QuantumChem.esp import RESP_fit_charges
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
steps=np.arange(res["steps"])+1

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