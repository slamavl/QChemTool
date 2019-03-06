# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:45:29 2019

@author: Vladislav Sláma
"""

import numpy as np
from ..QuantumChem.Classes.general import Energy as EnergyClass
from ..General.UnitsManager import energy_units

def get_Elstat_energy(struc,state="Ground",charge=None,scaling_14 = 1.2):
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
    scaling_14 : float (optional init = 1.2)
        1-4 electrostatic interactions will are scaled by factor 1/scaling_14
        (taken from AMBER implementation)
    
    Returns
    ----------
    elstat : Energy class
        Total electrostatic energy of the system
    """
    
    excluded, bonded_14 = struc.get_14_bonded_atoms()
    R,RR = struc.get_distance_matrixes()
    RR_1 = np.divide(1, RR, out=np.zeros_like(RR), where=RR!=0) # This should produce zeros when dividing by zero 
    RR_14 = RR_1.copy()
    
    mask = np.zeros(RR_1.shape, np.bool)
    mask_14 = np.ones(RR_1.shape, np.bool)
    for ii in range(struc.nat):
        mask[ii,excluded[ii]] = 1
        mask_14[ii,bonded_14[ii]] = 0
    
    RR_1[mask] = 0.0
    RR_14[mask_14] = 0.0
    
    if charge is not None:
        if len(charge) != struc.nat:
            raise IOError("For calculation of Electrostatic energy charges for"+
                          " every atom hve to be defined.")
        else:
            #Q=np.meshgrid(charge,charge)[0]
            elstat = np.dot(np.dot(charge,RR_1),charge)/2.0
            elstat14 = np.dot(np.dot(charge,RR_14),charge)/2.0
    elif state == "Ground":
        #Q=np.meshgrid(struc.esp_grnd,struc.esp_grnd)[0]
        elstat = np.dot(np.dot(struc.esp_grnd,RR_1),struc.esp_grnd)/2.0
        elstat14 = np.dot(np.dot(struc.esp_grnd,RR_14),struc.esp_grnd)/2.0
    elif state == "Excited":
        #Q=np.meshgrid(struc.esp_exct,struc.esp_exct)[0]
        elstat = np.dot(np.dot(struc.esp_exct,RR_1),struc.esp_exct)/2.0
        elstat14 = np.dot(np.dot(struc.esp_exct,RR_14),struc.esp_exct)/2.0
    
    with energy_units("AU"):
        elstat = EnergyClass(elstat)
        elstat14 = EnergyClass(elstat14/scaling_14)
    
    return elstat, elstat14

def get_vdW_energy(struc, scaling_14 = 2.0):
    """ Calculate total vdW energy of the system (contributions from
    1-4 bonded interactions are excluded as in standard forcefields) 
    
    Parameters
    ----------
    struc : Structure class
        Structural data of the system 
    scaling_14 : float (optional init = 2.0)
        1-4 vdW interactions will are scaled by factor 1/scaling_14
        (taken from AMBER implementation)
    
    Returns
    ----------
    vdW_energy : Energy class
        Total electrostatic energy of the system
    """
    
    if struc.vdw_rad is None:
        struc.get_FF_rad()
    
    excluded, bonded_14 = struc.get_14_bonded_atoms()
    R,RR = struc.get_distance_matrixes()
    RR_1 = np.divide(1, RR, out=np.zeros_like(RR), where=RR!=0) # This should produce zeros when dividing by zero 
    RR_14 = RR_1.copy()
    
    mask = np.zeros(RR_1.shape, np.bool)
    mask_14 = np.ones(RR_1.shape, np.bool)
    for ii in range(struc.nat):
        mask[ii,excluded[ii]] = 1
        mask_14[ii,bonded_14[ii]] = 0
        
    RR_1[mask] = 0.0
    RR_14[mask_14] = 0.0
    
    # calculate energy and optimal distance matrix
    vdw_eng = struc._vdw_eng.reshape(-1, 1)
    e0 = np.dot(vdw_eng,vdw_eng.T)
    e0 = np.sqrt(e0)    # e0_ij = sqrt(e0_i * e0_j) 
    r0 = np.tile(struc._vdw_rad,(struc.nat,1))
    r0 = r0 + r0.T      # r0_ij = r0_i + r0_j 
    
    # calculate vdW energy
    vdW_energy = e0 * (np.power(r0*RR_1,12) - 2*np.power(r0*RR_1,6))
    vdW_energy_14 = e0 * (np.power(r0*RR_14,12) - 2*np.power(r0*RR_14,6))
    vdW_energy = np.sum(vdW_energy)/2
    vdW_energy_14 = np.sum(vdW_energy_14)/scaling_14/2
    with energy_units("AU"):
        vdW_energy = EnergyClass(vdW_energy)
        vdW_energy_14 = EnergyClass(vdW_energy_14)
    
    return vdW_energy, vdW_energy_14