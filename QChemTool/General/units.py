# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:54:57 2016

@author: User
"""

import scipy.constants as const

# frequency posite to energy: energy in internal / conversion_facs_energy["1/cm"] = energy in 1/cm
# in fact conversion factors are for angular frequency (1 int => omega = 1 fs^{-1} )
conversion_facs_frequency = {
    "int"    : 1.0,
    "1/fs"   : 1.0, 
    "1/cm"   : 2.0*const.pi*const.c*1.0e-13, 
    "THz"    : 2.0*const.pi*1.0e-03,
    "Hz"     : 2.0*const.pi,
    "SI"     : 2.0*const.pi,
    "nm"     : 1.0/(1.0e7*2.0*const.pi*const.c*1.0e-13)#,
#    "a.u."   : 27.21138602*1.0e-15*const.e/const.hbar,
#    "Ha"     : 27.21138602*1.0e-15*const.e/const.hbar      
    } 

# energy - Hartree to ... conversion_facs_energy["eV"]=27.211386019301617 
conversion_facs_energy = {
    "int"     : 1.0,
    "Ha"      : 1.0, 
    "AU"      : 1.0,
    "eV"      : const.h*const.c*const.Rydberg*2/const.e,  
    "meV"     : const.h*const.c*const.Rydberg*2/const.e*100,
    "1/cm"    : 2*const.Rydberg/100,
    "THz"     : const.c*const.Rydberg*2*1e-12,
    "J"       : const.h*const.c*const.Rydberg*2,
    "SI"      : const.h*const.c*const.Rydberg*2
#    "nm"     : 1.0/(1.0e7*2.0*const.pi*const.c*1.0e-13),
    } 

# length - Bohr to ... conversion_facs_position["Angstrom"] 0.52917721067
conversion_facs_position = {
    "int"      : 1.0,
    "Bohr"     : 1.0,
    "m"        : const.physical_constants["Bohr radius"][0],
    "Angstrom" : const.physical_constants["Bohr radius"][0]*1e10,
    "SI"       : const.physical_constants["Bohr radius"][0]
}


conversion_facs_time = {
    "fs" : 1.0,
    "ps" : 1000.0,
    "ns" : 1.0e6,
    "s"  : 1.0e15,
    "SI" : 1.0e15
}


conversion_facs_temperature = {
    "K" : 1.0,
    "C" : 1.0,
    "F" : 0.0
}

conversion_offs_temperature = {
    "K" : 0.0,
    "C" : 273.15,
    "F" : 0.0
}

conversion_facs_edipole = {
    "int": 1.0,
    "au" : 1.0,
    "D"  : 1.0/0.20819434,
    "Cm" : 1.0e-21/const.c,
    "SI" : 1.0e-21/const.c
}

conversion_facs_length = {
    "int": 1.0,
    "A" : 1.0,
    "Bohr"  : 0.52917721067,
    "a.u." : 0.52917721067,
    "nm" : 10.0,
    "m"  : 1.0e10,
    "SI" : 1.0e10
}


conversion_facs_mass = {
        "AU": 1.0,
        "AMU": const.physical_constants["electron mass in u"][0]
}

kB_int_freq = const.k*1.0e-15/const.hbar # Boltzmann in internal frequency units [1/fs]

#
# Conversion function
#


#def convert(val, in_units, to=None):
#    """Converts value in certain units into other units
#    
#    """
#    from .managers import energy_units
#    from .managers import Manager
#    m = Manager()
#    with energy_units(in_units):
#        e = m.convert_energy_2_internal_u(val)
#    
#    if to is None:
#        ne = m.convert_energy_2_current_u(e)
#    else:
#        with energy_units(to):
#            ne = m.convert_energy_2_current_u(e)
#        
#    return ne
#
#def in_current_units(val, in_units):
#    """Converts value in certain units into the current units
#    
#    """
#    from .managers import energy_units
#    from .managers import Manager
#
#    m = Manager()
#    with energy_units(in_units):
#        e = m.convert_energy_2_internal_u(val)
#    
#    return m.convert_energy_2_current_u(e)
