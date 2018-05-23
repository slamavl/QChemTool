# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:09:34 2018

@author: Vladislav Sláma
"""

import numpy as np
from QChemTool.Polarizable_atoms.Electrostatics_module import Electrostatics

'''----------------------- TEST PART --------------------------------'''
if __name__=="__main__":

    print('                TESTS')
    print('-----------------------------------------')    
    
    ''' Test derivation of energy d/dR ApB '''
    coor = [[0.0,0.0,0.0],[0.0,0.0,1.0],[3.0,0.0,0.0],[3.0,0.0,1.0]]
    coor = np.array(coor,dtype='f8')
    charge = np.ones(4,dtype='f8')
    at_type = ['CD','CD','CF','CF']
    Mol_elstat = Electrostatics(coor,charge,at_type)
    Eshift = Mol_elstat.get_EnergyShift()
    Eshift2, derivative = Mol_elstat.get_EnergyShift_and_Derivative()
    
    print(Eshift,Eshift2,Eshift2-Eshift)
    
    for kk in range(8):
        dr = 1/10**kk
        Mol_elstat_dr=np.zeros(12)
        for ii in range(3):
            for jj in range(4):
                coor_tmp = coor.copy()
                coor_tmp[jj,ii] += dr
                Mol_elstat_tmp = Electrostatics(coor_tmp,charge,at_type)
                Mol_elstat_dr[jj*3+ii] = Mol_elstat_tmp.get_EnergyShift()
        
        Mol_elstat_dr = (Mol_elstat_dr - Eshift)/dr
        
        suma = np.sum(np.abs(Mol_elstat_dr-derivative))
        print('dr=',dr,' and sum=',suma)
    
    print(derivative)