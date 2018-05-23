# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:31:07 2018

@author: Vladislav Sláma
"""

from QChemTool import CorrelationFunction,TimeAxis
from QChemTool.Spectroscopy.spectraldensity import SpectralDensity
import numpy as np
from  QChemTool.Development.FourierTransform import t2f
import matplotlib.pyplot as plt
start=0.0
step=0.1
length=40000
time = TimeAxis(start=start,length=length,step=step,frequency_start=0.0)

print("OverdampedBrownian")
params = {}
params["ftype"] = "OverdampedBrownian"
params["T"] = 300
params["cortime"] = 100
params["reorg"] = 200
params["matsubara"] = 200
cfunc = CorrelationFunction(axis=time, params=params)
cfunc.plot()
comega = cfunc.get_rFourier_transform()
print(comega.data)
comega.plot()
print(np.sum(np.real(comega.data)/comega.axis.data)*(comega.axis.step))

lmbda = cfunc.measure_reorganization_energy()
print(lmbda)
sfunc = SpectralDensity(axis=time, params=params)
lmbda = sfunc.measure_reorganization_energy()
print(lmbda)
print(cfunc.get_reorganization_energy())

if 0:
    time2 = (np.arange(0,length*2)-length)*step
    cfunc2 = np.zeros(length*2,dtype=np.complex128)
    cfunc2[length:length*2] = cfunc.data
    for ii in range(length-1):
        cfunc2[length-ii-1] = np.conjugate(cfunc2[length+ii+1])
    omega,comega = t2f(time2,cfunc2)    
    print("User def ",np.sum(np.imag(comega)))
    plt.plot(omega,np.real(comega))
    plt.plot(omega,np.imag(comega))
    
    # Calculate reorganization energy
    lmbda = cfunc.measure_reorganization_energy()
    print(lmbda)
    comega = cfunc.get_Fourier_transform()
    #comega.plot()
    print(cfunc.axis.atype,np.sum(np.imag(comega.data)))

if 1:

    print()
    print("UnderdampedBrownian")
    params = {}
    params["ftype"] = "UnderdampedBrownian"
    params["T"] = 300
    params["gamma"] = 0.01
    params["freq"] = 0.1
    params["reorg"] = 100
    cfunc = CorrelationFunction(axis=time, params=params)
    cfunc.plot()
    
    # Calculate reorganization energy
    lmbda = cfunc.measure_reorganization_energy()
    print(lmbda)
    
    lmbda = cfunc.measure_reorganization_energy()
    print(lmbda)
    sfunc = SpectralDensity(axis=time, params=params)
    lmbda = sfunc.measure_reorganization_energy()
    print(lmbda)
    print(cfunc.get_reorganization_energy())
    
# TEST value defined correlation function