# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:41:55 2018

@author: Vladislav Sláma
"""


# Two alternative ways how to analyze AMBER MD output file (both provide same results) 
if 0:
    from QChemTool.QuantumChem.read_mine import read_amber_mdout
    from QChemTool.QuantumChem.Classes.mdinfo import output_MD_statictic
    
    res, inp, avrg, rmsd = read_amber_mdout("03_Pres.out")
    output_MD_statictic(res, inp, avrg, print_input=True, plot_stat=True, size_RA=10)
    
else:
    from QChemTool.QuantumChem.Classes.mdinfo import analyze_amber_mdout
    
    #analyze_amber_mdout("03_Pres.out", size_RA=10)
    analyze_amber_mdout("D:/slamav/MD/Perylene-Chloroform/05_Prod.out", size_RA=10)
