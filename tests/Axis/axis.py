# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 10:19:05 2018

@author: Vladislav Sláma
"""

from QChemTool import TimeAxis,FrequencyAxis
from QChemTool.General.dfunction import DFunction
import numpy as np

time = TimeAxis(start=np.float32(0.0),length=10,step=np.float32(1.0), frequency_start=np.float32(0.0))
print(time.data)
frequency = time.get_FrequencyAxis()
print(frequency.data)
N=frequency.length
print(N,(N+1)//2,frequency.data[(N-1)//2])
time2 = frequency.get_TimeAxis()
print(time2.data)
frequency2 = time2.get_FrequencyAxis()
print(frequency2.data)
print(time.frequency_start,time2.frequency_start)
print()

# Gaussian
dt = 0.01
N = 10
a = 20.0
time = np.arange(N,dtype='f8')*dt - N/2*dt #(N-1)/2*dt
time_half = time[(N//2):N]
ft = np.exp(-a*time*time)
ft[0] = 0.0
ft_half = np.exp(-a*time_half*time_half)
print(time,time_half)
print(ft,ft_half)

t = TimeAxis(time[0],N,dt,atype="complete")
th = TimeAxis(time_half[0],N//2,dt)
gauss = DFunction(t,ft)
gauss_half = DFunction(th,ft_half)
ftc = gauss.get_Fourier_transform() 
fth = gauss_half.get_Fourier_transform() 
print(fth.data)
print(fth.data)
iftc = ftc.get_inverse_Fourier_transform()
ifth = fth.get_inverse_Fourier_transform()
print(np.real(iftc.data))
print(gauss.data)
print(np.real(ifth.data))

print()
print("TEST rFFT")
# Gaussian

ft = np.exp(-a*time*time)
gauss = DFunction(t,ft)
time2 = np.arange(N+2,dtype='f8')*dt - (N+2)/2*dt #(N-1)/2*dt
time2_half = time2[(N+2)//2:N+2]
th = TimeAxis(time2_half[0],(N+2)//2,dt)

print(th.data)
print(ft)


ft_half = np.exp(-a*time2_half*time2_half)
gauss_half2 = DFunction(th,ft_half)
print(gauss_half2.data)

ftc = gauss.get_Fourier_transform()
ftc.axis.atype="upper-half"
irftc = ftc.get_inverse_rFourier_transform()
print(np.real(irftc.data))

fth2 = gauss_half2.get_rFourier_transform()
print(np.real(ftc.data))
print(fth2.data)
