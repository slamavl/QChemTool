# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:54:57 2016

@author: User
"""

import numpy
import scipy.constants as const

HaToInvcm=219474.6313705
BohrToAngstrom=0.52917721067
AmgstromToBohr=1.88972688
kcalmol1Tocm1=349.75
kcalmol1ToHa=0.00159362
konst1=7399643.84752676 # prevzal jsem ze sveho predchoziho vypoctu force[Ha/A^2]=(1/konst1).freq^2[cm-1].m[AMU]
Kb = 0.69503476   # cm-1/K
c=299792458 # m.s^-1
fsTOcm=10**13/c    # (1/x[fs])*fsTOcm=x[cm]
InternalToInvcm=numpy.sqrt(konst1)/BohrToAngstrom
evTOcm=8065.73
AuToDebye=2.541746
hbar_eVfs=0.6582119514
kB_intK=Kb=const.c*1.0e-13
MEAD2Invcm=116063

# Internal Units:
#     For quantum chemistry calculation Atomic Units
#     For spectroscopy: t = fs f = 1/fs omega = 2pi/fs energy = 2pi/fs hbar=1 (energy=omega*hbar)  energy=hbar*frequency(angular) 
#                       t .... time
#                       f .... frequency
#                       omega ..... angular frequency
 
conversion_facs_frequency = {
    "2pi/fs" : 1.0,                 # angular frequency in internal units
    "1/fs"   : 1.0,                 # frequency in internal units
    "1/cm"   : 2*const.pi*const.c*1.0e-13,     # wavenumber in inverse centimeters
    "Thz"    : 1.0e-3               # frequency in Thz
    # same as for frequency is valid for energy
    }
    
conversion_facs_energy = {
    "int"    : 1.0,
    "2pi/fs" : 1.0, 
    "1/cm"   : const.c*1.0e-13, 
    "THz"    : 1.0e-03,
    "eV"     : 1.0/(hbar_eVfs*const.pi*2)   
}
     