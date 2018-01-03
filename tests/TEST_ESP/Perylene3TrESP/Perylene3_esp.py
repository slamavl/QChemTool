# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 12:28:30 2016

@author: slamav
"""
import timeit
import os

from QChemTool.QuantumChem.Classes.molecule import Molecule
from QChemTool.QuantumChem.output import OutputToESP


typ='TDDFT'
state_indx=1

directory=os.getcwd()+"/"

chkfile="perylene3_exct_wB97XD-LANL2DZ_geom_BLYP-LANL2DZ.fchk"
logfile="perylene3_exct_wB97XD-LANL2DZ_geom_BLYP-LANL2DZ.log"
chkfile_trans="".join(["perylene3_exct_wB97XD-LANL2DZ_geom_BLYP-LANL2DZ.Tr",str(state_indx),".fchk"])
gridfile="ESPGrid_perylene3"

if __name__ == "__main__":
    mol=Molecule('Perylene3 ESP calculation')
    mol.load_Gaussian_fchk(chkfile)
    mol.load_Gaussian_log(logfile)
    mol.load_Gaussian_density_fchk(chkfile_trans,typ='Transition')
    print('Molecule loaded')
    print('Normalization of molecular orbitals....')
    mol.mo.normalize(mol.ao)
    print('.... Molecular orbitals normalized')

    print('Starting ESP calculation ....')
    start_time=timeit.default_timer()   
    
    potential_grnd,potential_tran,potential_exct=mol.get_ESP_grid(state_indx=state_indx,load_grid='qchem',gridfile=gridfile)   
    OutputToESP(potential_grnd,filename="".join([directory,'Vysledky/plot_DFTread_esp_ground.esp']))
    OutputToESP(potential_tran,filename="".join([directory,'Vysledky/plot_DFTread_esp_trans_0-',str(state_indx),'.esp']))
    OutputToESP(potential_exct,filename="".join([directory,'Vysledky/plot_DFTread_esp_exct.esp']))
    elapsed = timeit.default_timer() - start_time
    print('Time for calculation of ESP grid with output:',elapsed)                                                     
