# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:31:50 2018

@author: Vladislav Sláma
"""
from QChemTool import Structure
from QChemTool.Development.polarizablesytem import PolarizableSystem
from QChemTool import energy_units
from QChemTool.QuantumChem.Fluorographene.fluorographene import orientFG
import numpy as np

parameters_type_manual = False
system = "2perylene" # "anthanthrene", "perylene", "2perylene"

if not parameters_type_manual: # Automatic definition of parameters
    # Set parameters of the system
    FG_charges = "ESPfit"
    params_polar={"VinterFG": False,"coarse_grain": "plane", "charge_type": FG_charges,"approximation": 1.1,"symm": False} 
    
    # Load FG structure
    struc = Structure()
    if system == "perylene":
        struc.load_xyz("FGrph_1perylene_2dist_ser_TDDFT-wB97XD_geom_BLYP-landl2dz_symm.xyz")
        # For practical calculation also reorient sheet in propper direction (plane) and carbons has to be before fluorines
        #struc.center(72,73,86)
        struc = orientFG(struc)
    elif  system == "anthanthrene":
        struc.load_xyz("FGrph_1anthranthrene_1dist_par_TDDFT-wB97XD_geom_BLYP-landl2dz_symm_7x11.xyz")
        # For practical calculation also reorient sheet in propper direction (plane) and carbons has to be before fluorines
        # struc.center(41,43,133)
        struc = orientFG(struc)
    elif system == "2perylene":
        struc.load_xyz("FGrph_2perylene_1dist_par_TDDFT-wB97XD_geom_BLYP-landl2dz_symm_9x12.xyz")
        # For practical calculation also reorient sheet in propper direction (plane) and carbons has to be before fluorines
        # struc.center(58,57,83)
        struc = orientFG(struc)
        struc.output_to_xyz("FGrph_2perylene_1dist_par_reorient.xyz")
    
    # Initialize the system
    elstat = {"structure": struc,"charge": FG_charges}
    diel  = {"structure": struc,"polar": params_polar}
    params = {"energy_type": "QC","permivity": 1.0,"order": 2}
    system = PolarizableSystem(diel = diel, elstat = elstat, params = params)
    
    Elfield=np.array([1.0,0.0,0.0])
    nn=170
    induced_dipoles = np.dot(system.diel.polar["AlphaE"][nn],Elfield)
    print(induced_dipoles)
    induced_dipoles = np.dot(system.diel.polar["Alpha_E"][nn],Elfield)
    print(induced_dipoles)
    induced_dipoles = np.dot(system.diel.polar["BetaEE"][nn],Elfield)
    print(induced_dipoles)
    induced_dipoles = np.dot(system.diel.polar["Alpha_st"][nn],Elfield)
    print(induced_dipoles)
    Elfield=np.array([0.0,1.0,0.0])
    induced_dipoles = np.dot(system.diel.polar["AlphaE"][nn],Elfield)
    print(induced_dipoles)
    induced_dipoles = np.dot(system.diel.polar["Alpha_E"][nn],Elfield)
    print(induced_dipoles)
    induced_dipoles = np.dot(system.diel.polar["BetaEE"][nn],Elfield)
    print(induced_dipoles)
    induced_dipoles = np.dot(system.diel.polar["Alpha_st"][nn],Elfield)
    print(induced_dipoles)
    Elfield=np.array([1.0,1.0,0.0])
    induced_dipoles = np.dot(system.diel.polar["AlphaE"][nn],Elfield)
    print(induced_dipoles)
    induced_dipoles = np.dot(system.diel.polar["Alpha_E"][nn],Elfield)
    print(induced_dipoles)
    induced_dipoles = np.dot(system.diel.polar["BetaEE"][nn],Elfield)
    print(induced_dipoles)
    induced_dipoles = np.dot(system.diel.polar["Alpha_st"][nn],Elfield)
    print(induced_dipoles)
    
    # identify defects - separated because now changes can be made to the database
    system.identify_defects()
    
    # Calculate energies in the system
    Ndef = len(system.defects)
    HH = np.zeros((Ndef,Ndef),dtype='f8')
    for ii in range(Ndef):
        dAVA = system.get_elstat_energy(ii,"excited-ground")
        Eshift, res_Energy, TrDip = system.get_SingleDefectProperties(ii)
        E01_vacuum = system.defects[ii].get_transition_energy()
        HH[ii,ii] = E01_vacuum._value + Eshift._value
        with energy_units("1/cm"):
            # print(system.defects[0].name,dAVA.value)
            print(system.defects[ii].name,ii+1,"energy shift:",Eshift.value)
            print(system.defects[ii].name,ii+1,"transition dipole:",TrDip)
    
    for ii in range(Ndef):
        for jj in range(ii+1,Ndef):
            J_inter, res = system.get_HeterodimerProperties(ii, jj, EngA = HH[ii,ii], EngB = HH[jj,jj], approx=1.1)
            #J_inter, res = system.get_HeterodimerProperties(ii, jj, approx=1.1)
            HH[ii,jj] = J_inter._value
            HH[jj,ii] = HH[ii,jj]
            with energy_units("1/cm"):
                print(system.defects[ii].name,ii+1,"-",system.defects[jj].name,jj+1,"interaction E:",J_inter.value)    
    
else:
    # Set fluorographene charges
    # manual definition
    FG_charges={'CF': 0.08125,'CF2': 0.171217,'CD': 0.0,'C': 0.0}
    FG_charges['FC'] = -FG_charges['CF']
    FG_charges['F2C'] = -FG_charges["CF2"]/2.0

    # set fluorographene atomic polarizabilities
    # manual definition
    #------------------------------------------------------------------------------
    AlphaE = np.zeros((3,3),dtype='f8')
    AlphaE[0,0] = 8.37209731  
    AlphaE[1,1] = 7.20970163  
    AlphaE[2,2] = 3.7048051
    BetaE = np.zeros((3,3),dtype='f8')
    BetaE[0,0] = 0.11924151
    BetaE[1,1] = 0.06845612  
    BetaE[2,2] = 0.23965548
    Alpha_E = np.zeros((3,3),dtype='f8')
    Alpha_E[0,0] = 0.00127885831
    Alpha_E[1,1] = 1.17300541
    Alpha_st = np.zeros((3,3),dtype='f8')
    Alpha_st[0,0] = 5.02626835/2.0     # 5.02626835/2.0
    Alpha_st[1,1] = 4.79760997/2.0     # 4.79760997/2.0
    Alpha_st[2,2] = 3.24912643/2.0     # 3.24912643/2.0
    VinterFG=0.0
    # Parameters for F atoms:
    FAlphaE = np.zeros((3,3),dtype='f8')
    FAlphaE[0,0] = 0.05755886  
    FAlphaE[1,1] = 1.11166446 
    FAlphaE[2,2] = 0.00932181
    FBetaE = np.zeros((3,3),dtype='f8')
    FBetaE[0,0] = 0.95405415  
    FBetaE[1,1] = 0.96775002  
    FBetaE[2,2] = 0.81886238 
    FAlpha_E = np.zeros((3,3),dtype='f8')
    FAlpha_E[0,0] = 0.0
    FAlpha_E[1,1] = 0.0
    AlphaF_st = np.zeros((3,3),dtype='f8')
    AlphaF_st[0,0] = 0.03369416/2.0    # 0.03369416/2.0
    AlphaF_st[1,1] = 0.32536893/2.0    # 0.32536893/2.0
    AlphaF_st[2,2] = 6.62686969/2.0    # 6.62686969/2.0
    ZeroM = np.zeros((3,3),dtype='f8')
    polar = { 'CF': [AlphaE,Alpha_E,BetaE,Alpha_st], 'C': [ZeroM,ZeroM,ZeroM,ZeroM]}
    polar['FC'] = [FAlphaE,FAlpha_E,FBetaE,AlphaF_st]
    params_polar={"VinterFG": 0.0,"coarse_grain": "all_atom", "polarizability": polar} 
    #------------------------------------------------------------------------------
    # Automatic definition
    
    
    # Load FG structure
    struc = Structure()
    #struc.load_xyz("FGrph_1perylene_2dist_ser_TDDFT-wB97XD_geom_BLYP-landl2dz_symm.xyz")
    struc.load_xyz("FGrph_1anthranthrene_1dist_par_TDDFT-wB97XD_geom_BLYP-landl2dz_symm_7x11.xyz")
    
    # For practical calculation also reorient sheet in propper direction (plane) and carbons has to be before fluorines
    
    # Initialize the system
    elstat = {"structure": struc,"charge": FG_charges}
    diel  = {"structure": struc,"polar": params_polar}
    params = {"energy_type": "QC","permivity": 1.0,"order": 2}
    system = PolarizableSystem(diel = diel, elstat = elstat, params = params)
    
    # identify defects - separated because now changes can be made to the database
    system.identify_defects()
    
    # Calculate energies in the system
    dAVA = system.get_elstat_energy(0,"excited-ground")
    with energy_units("1/cm"):
        print(system.defects[0].name,dAVA.value)
