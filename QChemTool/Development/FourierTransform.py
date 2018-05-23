# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:09:55 2018

@author: Vladislav Sláma
"""
import numpy as np

def _sett(om,t_min=None):
    """ This function create discrete time axis from given discrete frequency
    axis in order tu fullfil requirements of discrete fourier transform 
    dt=2pi/(N*dom). Fourier transform is defined as 
    F(om)=int{t,-inf,inf}{f(t)*exp[i*om*t]} and frequency axis is centered
    around zero """

# TODO: Check if om is an array

    dom=om[1]-om[0]    # size of omega step
    N=len(om)        # number of omega values
    
    dt=2.0*np.pi/(N*dom)   # size of step in k-space
    if t_min is None:
        tmin = -dt*np.floor(N/2)  
    else:
        tmin = t_min
    # for N even pmin=-N/2
    # for N odd pmin=-(N-1)/2
    
    # Zde se nemùe nechat napøíklad 1.5 násobek dp jeliko by pak nebyla
    # splnìna podmínka zachování periodiciti pro pøechod mezi reprezentacemi
    # Pro osy v impulzové reprezentaci se hodí kdy je centrovaná okolo nuly z
    # dùvodu symetrie problémù
    
    t=np.arange(N,dtype='f8')*dt+tmin
    return t


def _setom(t,omega_min=None):
    """ This function create discrete time axis from given discrete frequency
    axis in order tu fullfil requirements of discrete fourier transform 
    dt=2pi/(N*dom). Fourier transform is defined as 
    F(om)=int{t,-inf,inf}{f(t)*exp[i*om*t]} and frequency axis is centered
    around zero """

# TODO: Check if om is an array

    dt=t[1]-t[0]    # size of omega step
    N=len(t)        # number of omega values
    
    dom=2.0*np.pi/(N*dt)   # size of step in k-space
    
    if omega_min is None:
        om_min = -dom*np.floor(N/2)  
    else:
        om_min = omega_min
    # for N even pmin=-N/2
    # for N odd pmin=-(N-1)/2
    
    # Zde se nemùe nechat napøíklad 1.5 násobek dp jeliko by pak nebyla
    # splnìna podmínka zachování periodiciti pro pøechod mezi reprezentacemi
    # Pro osy v impulzové reprezentaci se hodí kdy je centrovaná okolo nuly z
    # dùvodu symetrie problémù
    
    omega = np.arange(N,dtype='f8')*dom+om_min
    return omega

def t2f(t,ft,omega_min=None):
    """ Fourier transform of function ft discretized on time axis t into 
    F_omega defined on frequency axis omega. Fourier transform is defined as 
    F(omega)=int{t,-inf,inf}{f(t)*exp[i*omega*t]}    
    """
    
    # set frequency axis
    omega = _setom(t,omega_min=omega_min)
    dt = t[1]-t[0]
    N = len(t)
    om_min = min(omega)
    tmin = min(t)
    
    # transform the initial function for fast fourier transform (FFT)
    gt = np.multiply(ft,np.exp(1j*om_min*t))
    # inverse FFT
    F_omega = np.fft.ifft(gt)
    # Obtain fourier transform
    F_omega = N*dt*np.exp(-1j*om_min*tmin) * np.multiply(np.exp(1j*omega*tmin),F_omega)
    
    return omega,F_omega


def f2t(omega,F_omega,t_min=None):
    """ Inverse Fourier transform of function F_omega discretized on frequenxcy  
    axis omega into ft defined on time axis t. Inverse Fourier transform is 
    defined as f(t)=1/(2*pi) * int{omega,-inf,inf}{F(omega)*exp[-i*omega*t]}    
    """
    
    # set frequency axis
    t = _sett(omega,t_min=t_min)
    dom = omega[1]-omega[0]
    om_min = min(omega)
    tmin = min(t)
    
    # transform the initial function for fast fourier transform (FFT)
    g_om = np.multiply(F_omega,np.exp(-1j*omega*tmin))
    # FFT
    ft = np.fft.fft(g_om)
    # Obtain fourier transform
    ft = dom/(2*np.pi)*np.exp(1j*om_min*tmin) * np.multiply(np.exp(-1j*om_min*t),ft)
    
    return t,ft

def _setp(x):
    """ This function create discrete k-space axis from given discrete x-space
    axis in order tu fullfil requirements of discrete fourier transform 
    dp=1/(N*dx). Fourier transform is defined as 
    F(p)=int{t,-inf,inf}{f(t)*exp[-2*Pi*i*p*x]} and k-space axis is centered
    around zero """

    # Overit jestli je x pole

    dx=x[1]-x[0]    # size of x step
    N=len(x)        # number of diferent x values
    
    dp=1.0/(N*dx)   # size of step in k-space
    pmin=-dp*np.floor(N/2)  
    # for N even pmin=-N/2
    # for N odd pmin=-(N-1)/2
    
    # Zde se nemùe nechat napøíklad 1.5 násobek dp jeliko by pak nebyla
    # splnìna podmínka zachování periodiciti pro pøechod mezi reprezentacemi
    # Pro osy v impulzové reprezentaci se hodí kdy je centrovaná okolo nuly z
    # dùvodu symetrie problémù
    
    p=np.arange(N,dtype='f8')*dp+pmin
    return p
    
def _x2p(x,p,fx):
    """ Fourier transform of function fx discretized on x into fp defined on p
    Fourier transform is defined as F(p)=int{t,-inf,inf}{f(t)*exp[-2*Pi*i*p*x]}    
    """
    
    dx=x[1]-x[0]
    gx=np.multiply(fx,np.exp(-2*np.pi*1j*min(p)*x))
    Fp=np.fft.fft(gx)
    Fp=dx*np.exp(2*np.pi*1j*min(p)*min(x))*np.multiply(np.exp(-2*np.pi*1j*p*min(x)),Fp)
    
    return Fp
    
def _p2x(x,p,Fp):
    """ Inverse Fourier transform of function Fp discretized on p into fx
    defined on x. Inverse Fourier transform is defined as
    f(x)=int{p,-inf,inf}{F(p)*exp[2*Pi*i*p*x]}    
    """
    dp = p[1]-p[0]
    N = len(p)
    gp = np.exp(-2*np.pi*1j*min(p)*min(x))*np.multiply(np.exp(2*np.pi*1j*p*min(x)),Fp)
    fx = np.fft.ifft(gp)
    fx = dp*N*np.multiply(fx,np.exp(2*np.pi*1j*min(p)*x))
    
    return fx


#==============================================================================
# TESTS
#==============================================================================
       
'''----------------------- TEST PART --------------------------------'''
if __name__=="__main__":
    print(' ')
    print('TESTS')
    print('-------')
    
    import matplotlib.pyplot as plt
    
    # Fourier transform of gaussian
    dt = 0.01
    N = 10
    a = 0.2
    t = np.arange(N,dtype='f8')*dt - N/2*dt #(N-1)/2*dt
    ft = np.exp(-a*t*t)
    print(t)
#    plt.plot(t,ft)
#    plt.show()
    
    omega,F_omega = t2f(t,ft)
    F_om_test = np.sqrt(np.pi/a) * np.exp(-omega*omega/4.0/a)
    if np.allclose(np.real(F_omega),F_om_test):
        print("Fourier transform of gaussian         ...  OK")
    else:
        print("Fourier transform of gaussian         ...  Error")
    
    t_test,ft_test = f2t(omega,F_omega)
    if np.allclose(t_test,t) and np.allclose(np.real(ft_test),ft):
        print("Inverse Fourier transform of gaussian ...  OK")
    else:
        print("Inverse Fourier transform of gaussian ...  Error")
        
    #Y = N*np.fft.fftshift(np.fft.irfft(np.fft.fftshift(ft[N//2:])))*dt
    #print(Y)
    #Y = N*np.fft.fftshift(np.fft.ihfft(np.fft.fftshift(ft[:])))*dt
    #print(np.real(Y))
    Y = N*np.fft.fftshift(np.fft.ifft(np.fft.fftshift(ft)))*dt
    #print(np.real(Y))
    if np.allclose(np.real(Y),F_omega):
        print("Numpy definition of Fourier transform ...  OK")
    else:
        print("Numpy definition of Fourier transform ...  Error")
    
    
    ft_test = N*np.fft.fftshift(np.fft.ifft(np.fft.fftshift(F_omega)))*(omega[1]-omega[0])/(np.pi*2.0)
    imag1 = np.imag(ft_test)
    ft_test = np.fft.fftshift(np.fft.fft(np.fft.fftshift(F_omega)))*(omega[1]-omega[0])/(np.pi*2.0)
    imag2 = np.imag(ft_test)
    if np.allclose(np.real(ft_test),ft):
        print("Numpy def. of inv. Fourier transform  ...  OK")
    else:
        print("Numpy def. of inv. Fourier transform  ...  Error")
    print(max(imag1))
    print(max(imag2))
#    plt.plot(omega,np.real(Y))
#    plt.plot(omega,F_om_test,'--')
#    plt.show()
    
    Y = N*np.fft.fftshift(np.fft.irfft(ft[N//2:]))*dt
    print(np.real(Y))
    print(np.real(F_omega))
   

     
    ft = np.exp(-a*np.abs(t))
    omega,F_omega = t2f(t,ft)
    F_om_test = 2*a/(a*a + omega*omega)
    t_test,ft_test = f2t(omega,F_omega)
    if np.allclose(t_test,t) and np.allclose(np.real(ft_test),ft):
        print("Fourier transform of exponential      ...  OK")
    else:
        print("Fourier transform of exponential      ...  Error")
        
    omega_test = np.fft.fftshift(
                (2.0*np.pi)*np.fft.fftfreq(N,dt))

    if np.allclose(omega_test,omega):
        print("Numpy definition of frequency axis    ...  OK")
    else:
        print("Numpy definition of frequency axis    ...  Error")
    
    dom = omega[1]-omega[0]
    times = np.fft.fftshift(
                    np.fft.fftfreq(N, dom/(2.0*np.pi)))
    if np.allclose(times,t):
        print("Numpy definition of time axis         ...  OK")
    else:
        print("Numpy definition of time axis         ...  Error")
    
    #print(omega_test)
    omega_test = np.fft.fftshift(
                (2.0*np.pi)*np.fft.fftfreq(2*10001, dt))
   # print(omega_test)
   
   
    ft = np.exp(-(a+1j*40)*t*t)*np.exp(1j*30*t)
    omega,F_omega = t2f(t,ft)
    Y = N*np.fft.fftshift(np.fft.ifft(np.fft.fftshift(ft)))*dt
    if np.allclose(Y,F_omega):
        print("Numpy definition of Fourier transform ...  OK")
    else:
        print("Numpy definition of Fourier transform ...  Error")
    
    ft_test = N*np.fft.fftshift(np.fft.ifft(np.fft.fftshift(F_omega)))*(omega[1]-omega[0])/(np.pi*2.0)
    if np.allclose(ft_test,ft):
        print("Numpy def. of inv. Fourier transform  ...  OK")
    else:
        print("Numpy def. of inv. Fourier transform  ...  Error  This fhould give Error")
    
    
    ft_test = np.fft.fftshift(np.fft.fft(np.fft.fftshift(F_omega)))*(omega[1]-omega[0])/(np.pi*2.0)
    if np.allclose(ft_test,ft):
        print("Numpy def. of inv. Fourier transform  ...  OK")
    else:
        print("Numpy def. of inv. Fourier transform  ...  Error")
    
    
#    print(max(imag1))
#    print(max(imag2))
#   
#   
#   
    dt = 0.01
    N = 10
    a = 20
    t = np.arange(N,dtype='f8')*dt - N/2*dt #(N-1)/2*dt
    ft = np.exp(-a*t*t)    
    
    # Using rfft for inverse fourier transform of real "spectral density" into complex correlation function
    print(omega[1]-omega[0])
    print(ft)
    print(dt,N)
    Y = N*np.fft.fftshift(np.fft.ifft(np.fft.fftshift(ft)))*dt 
    print(Y)
    fff = np.fft.rfft(np.fft.fftshift(np.real(Y)))*(omega[1]-omega[0])/(np.pi*2.0)
    print(np.real(fff))
    print(np.real(ft))
    
    
    if np.allclose(np.real(fff[0:N//2]),np.real(ft[N//2:])): # rfft has one more point than one sided corr. function
        print("Numpy def. of inv. rFourier transform ...  OK")
    else:
        print("Numpy def. of inv. rFourier transform ...  Error")
        
    print(np.real(Y))
    Y_test = N*np.fft.fftshift(np.fft.irfft(fff))*dt
    print(Y_test)
    if np.allclose(np.real(Y),Y_test): # rfft has one more point than one sided corr. function
        print("Numpy def. of rFourier transform      ...  OK")
    else:
        print("Numpy def. of rFourier transform      ...  Error")