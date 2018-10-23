# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:31:50 2018

@author: Vladislav Sláma
"""
from QChemTool import Structure
from QChemTool.Development.polarizablesytem_periodic import PolarizableSystem
from QChemTool import energy_units
from QChemTool.QuantumChem.Fluorographene.fluorographene import orientFG
import numpy as np

parameters_type_manual = False
system = "2perylene" # "anthanthrene", "perylene", "2perylene"

if not parameters_type_manual: # Automatic definition of parameters
    # Set parameters of the system
    FG_charges = "ESPfit"
    params_polar={"VinterFG": True,"coarse_grain": "plane", "charge_type": FG_charges,"approximation": 1.1, "symm": True} 
    
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
    CF_charge = -0.0522
    CF2_charge = 2*CF_charge
    FG_charges={'CF': CF_charge,'CF2': CF2_charge,'CD': 0.0,'C': 0.0}
    FG_charges['FC'] = -FG_charges['CF']
    FG_charges['F2C'] = -FG_charges["CF2"]/2.0

    # set fluorographene atomic polarizabilities
    # manual definition
    #------------------------------------------------------------------------------
    #                 polxy       polz      amp     per  phase
#    CF_AE_params = [7.53538330517, 0.0000, 1.0326577124,   2, 0.0]
#    CF_A_E_params = [0.505521019116, 0.000000, 0.4981493,  2, np.pi/2]
#    CF_BE_params = [0.129161747387, 0.0000, 0.05876077,    2, 0.0]
#    CF_Ast_params = [2.30828107, 0.0000000, 0.08196599,    2, 0.0] #[2.30828107, 0.00000, 0.081966, 2]
    
    CF_AE_params =  [8.94690348, 4.50738195, 1.65097606, 3, 0.0]
    CF_A_E_params = [0.39013017, 2.09784509, 0.59003868, 3, 0.0]
    CF_BE_params =  [0.57543444, 3.98822098, 0.63754235, 3, 0.0]
    CF_Ast_params = [5.17064221/2, 4.99791421/2, 0.25093473/2,  3, 0.0] 
    VinterFG = 0.0
    
    C_params     = [0.00000000, 0.0000000, 0.0,       0, 0.0]
    FC_AE_params = [0.00000000, 0.0000000, 0.0,       0, 0.0]
    FC_A_E_params = [0.0000000, 0.0000000, 0.0,       0, 0.0]
    FC_BE_params = [0.00000000, 0.0000000, 0.0,       0, 0.0]
    FC_Ast_params = [0.0000000, 0.0000000, 0.0,       0, 0.0]
    
    polar = {'AlphaE': {"CF": CF_AE_params, "FC": FC_AE_params, "C": C_params}}
    polar['Alpha_E'] = {"CF": CF_A_E_params, "FC": FC_A_E_params, "C": C_params}
    polar['BetaEE'] = {"CF": CF_BE_params, "FC": FC_BE_params, "C": C_params}
    polar['Alpha_st'] = {"CF": CF_Ast_params, "FC": FC_Ast_params, "C": C_params}

    params_polar={"VinterFG": 0.0,"coarse_grain": "C", "polarizability": polar,"approximation": 1.1}     
    
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
    