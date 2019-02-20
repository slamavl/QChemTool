# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:45:29 2019

@author: Vladislav Sláma
"""

import numpy as np
from ..QuantumChem.Classes.general import Energy as EnergyClass
from ..General.UnitsManager import energy_units

def get_Elstat_energy(struc,state="Ground",charge=None):
    """ Calculate total electrostatic energy of the system (contributions from
    1-4 bonded interactions are excluded as in standard forcefields) 
    
    Parameters
    ----------
    struc : Structure class
        Structural data of the system 
    state : string (optional init = "Ground")
        Specifies state of the system for which electrostatic energy should be 
        calculated. Allowed types are ``"Ground"`` and ``"Excited"``.
    charge : numpy array of real (optional, dimension Natom, init = None)
        Atomic charges for every atom in elemetary charges.
    
    Returns
    ----------
    elstat : Energy class
        Total electrostatic energy of the system
    """
    
    bonded_14 = struc.get_14_bonded_atoms()
    R,RR = struc.get_distance_matrixes()
    RR_1 = np.divide(1, RR, out=np.zeros_like(RR), where=RR!=0) # This should produce zeros when dividing by zero 
    
    for ii in range(struc.nat):
        RR_1[ii,bonded_14[ii]] = 0.0
    
    if charge is not None:
        if len(charge) != struc.nat:
            raise IOError("For calculation of Electrostatic energy charges for"+
                          " every atom hve to be defined.")
        else:
            #Q=np.meshgrid(charge,charge)[0]
            elstat = np.dot(np.dot(charge,RR_1),charge)
    elif state == "Ground":
        #Q=np.meshgrid(struc.esp_grnd,struc.esp_grnd)[0]
        elstat = np.dot(np.dot(struc.esp_grnd,RR_1),struc.esp_grnd)
    elif state == "Excited":
        #Q=np.meshgrid(struc.esp_exct,struc.esp_exct)[0]
        elstat = np.dot(np.dot(struc.esp_exct,RR_1),struc.esp_exct)
    
    with energy_units("AU"):
        elstat = EnergyClass(elstat)
    
    return elstat

def get_vdW_energy(struc):
    """ Calculate total vdW energy of the system (contributions from
    1-4 bonded interactions are excluded as in standard forcefields) 
    
    Parameters
    ----------
    struc : Structure class
        Structural data of the system 
    
    Returns
    ----------
    vdW_energy : Energy class
        Total electrostatic energy of the system
    """
    
    if struc.vdw_rad is None:
        struc.get_FF_rad()
    
    bonded_14 = struc.get_14_bonded_atoms()
    R,RR = struc.get_distance_matrixes()
    RR_1 = np.divide(1, RR, out=np.zeros_like(RR), where=RR!=0) # This should produce zeros when dividing by zero 
    for ii in range(struc.nat):
        RR_1[ii,bonded_14[ii]] = 0.0
    
    # calculate energy and optimal distance matrix
    e0 = np.meshgrid(struc._vdw_eng,struc._vdw_eng)[0]
    e0 = np.sqrt(e0)    # e0_ij = sqrt(e0_i * e0_j) 
    r0 = np.tile(struc._vdw_rad,(struc.nat,1))
    r0 = r0 + r0.T      # r0_ij = r0_i + r0_j 
    
    # calculate vdW energy
    vdW_energy = 4*e0 * (np.power(r0*RR_1,12)/4 - np.power(r0*RR_1,6)/2)
    vdW_energy = np.sum(vdW_energy)
    with energy_units("AU"):
        vdW_energy = EnergyClass(vdW_energy)
    
    return vdW_energy