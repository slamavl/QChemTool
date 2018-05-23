# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 15:50:17 2018

@author: Vladislav Sláma
"""


import scipy.spatial.distance.pdist as pdist
from QChemTool.QuantumChem.Classes.structure import Structure
import numpy as np


MolDirPer='C:/PhD/Ab-initio-META/Fluorographane/ESP_Charges/Perylene/Qchem/OPTgeom/RESP/Transdens/'
MolDir='C:/PhD/Ab-initio-META/Fluorographane/Excited_states/Distance-dependence/'
TrChargeDir='C:/PhD/Ab-initio-META/Fluorographane/Excited_states/Perylene_sph/'
charge_filename_xyz = "".join([TrChargeDir,'perylene_exct_wB97XD_LANL2DZ_geom_BLYP_LANL2DZ_sph.xyz'])
struc_filename_xyz = "".join([MolDir,'FGrph_1perylene_',str(1),'dist_par_TDDFT-wB97XD_geom_BLYP-landl2dz_symm2.xyz'])

struc = Structure()
struc.load_xyz(struc_filename_xyz)
dist_matrix = pdist(struc.coor._value)
mean_dist = np.mean(dist_matrix,axis=0)
var_dist = np.var(dist_matrix,axis=0)

indx_mean = np.argsort(mean_dist)
indx_var = np.argsort(var_dist)


defect = Structure()
defect.load_xyz(charge_filename_xyz)
def_dist_matrix = pdist(defect.coor._value)
def_mean_dist = np.mean(def_dist_matrix,axis=0)
def_var_dist = np.var(def_dist_matrix,axis=0)

indx_def_mean = np.argsort(def_mean_dist)
indx_def_var = np.argsort(def_var_dist)

