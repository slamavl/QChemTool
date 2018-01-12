# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:12:49 2016

@author: uzivatel
"""

import numpy as np
# plot statistic
import matplotlib.pyplot as plt
from scipy.stats import norm



class MDinfo:
    '''Class managing information from MD simulation.
    '''
    def __init__(self):
        self.NStep = []
        self.timestep = None
        self.NAtom = []
        self.at_name = []
        self.geom = []
        
def running_average(data,size=10):
    """
    Smoothing data by averaging number by region <i-size,i+size>
    """ 
    data_avrg = data.copy()
    data_len = len(data)
    
    for ii in range(1,size+1):
        data_avrg[ii] = np.sum(data[0:2*ii+1])/(2*ii+1)
    for ii in range(size+1,data_len-size):
        data_avrg[ii] = np.sum(data[ii-size:ii+size+1])/(2*size+1)
    for ii in range(1,size+1):
        data_avrg[data_len-ii-1] = np.sum(data[-2*ii-1:])/(2*ii+1)
    
    return data_avrg

def output_MD_statictic(res, inp, avrg, print_input=False, plot_stat=True, size_RA=10):
    """ Output MD setting and outputs time evolution and statistic of most
    important quantities
    """
    
# TODO: Add possibility to read different Running average size for different quantities
    RA = {"temperature": size_RA, "pressure": 50, "density": size_RA,
          "Etotal": size_RA, "Epotential": size_RA, "Ekinetic": size_RA }
    
    if print_input:
        # Print header
        print("MD simulation settings:")
        print("-----------------------")
        print("    mdout file write frequency:",inp["ntpr"])
        print("    Restart file write frequency:",inp["ntwr"])
        print("    Trajectory write frequency:",inp["ntwx"])
        print("    Number of steps:",inp["Nsteps"])
        print("    Timestep:",inp["time_step"],"fs")
        MDlen = inp["Nsteps"]*inp["time_step"]/1000.0
        print("    Total MD length:",int(MDlen//1000),"ns and",MDlen%1000,"ps")
        print("    Temperature:",inp["temperature"],"K   ( with gama:",inp["gamma_t"],"1/ps )")
        print("    Pressure:",inp["pressure"],"bar   ( with tau:",inp["taup"],"ps )")
        print(" ")
    
    if plot_stat:
        try:
            indx = max(plt.get_fignums())
        except:
            indx = 0
        
        # Temperature statistic
        #---------------------------
        fig = plt.figure(indx+1,figsize=(10,8))
        fig.canvas.set_window_title('Temperature')
        plt.subplot(211)
        plt.title('Temperature')
        plt.xlabel('Time [ps]')
        plt.ylabel('Temperature [K]')
        plt.xlim([res["time"][0],res["time"][-1]])
        plt.plot(res["time"],res["temperature"],'k-')
        plt.plot(res["time"],[inp["temperature"]]*len(res["time"]),'r--')
        plt.plot(res["time"],[avrg["temperature"]]*len(res["time"]),'r-')
        plt.plot(res["time"],running_average(res["temperature"],size=RA["temperature"]),'b-')
        plt.legend(["MD data","Requested value","Average value","Running average MD"])
        
        plt.subplot(212)
        plt.title('Temperature distribution')
        plt.xlabel('Temperature [K]')
        plt.ylabel('Probability')
        plt.hist(res["temperature"], bins=30, normed=True)
        
        mu, std = norm.fit(res["temperature"])
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        x = np.linspace(xmin, xmax, 200)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        title = "Fit results: \n   Mean temp. = %.2f K \n   Standard dev. = %.2f K" % (mu, std)
        plt.text(xmin, ymax-(ymax-ymin)/5,title)
        plt.tight_layout()
        plt.show()
        
        # Pressure statistic
        #---------------------------
        fig = plt.figure(indx+2,figsize=(10,5))
        fig.canvas.set_window_title('Pressure')
        plt.title('Pressure')
        plt.xlabel('Time [ps]')
        plt.ylabel('Pressure [bar]')
        plt.xlim([res["time"][0],res["time"][-1]])
        plt.plot(res["time"],res["pressure"],'k-')
        plt.plot(res["time"],[inp["pressure"]]*len(res["time"]),'r--')
        plt.plot(res["time"],[avrg["pressure"]]*len(res["time"]),'r-')
        plt.plot(res["time"],running_average(res["pressure"],size=RA["pressure"]),'b-')
        plt.legend(["MD data","Requested value","Average value","Running average MD"])
        plt.show()
        
        # Density statistic
        #---------------------------
        fig = plt.figure(indx+3,figsize=(10,5))
        fig.canvas.set_window_title('Density')
        plt.title('Density')
        plt.xlabel('Time [ps]')
        plt.ylabel('Density [g/cm^3]')
        plt.xlim([res["time"][0],res["time"][-1]])
        plt.plot(res["time"],res["density"],'k-')
        plt.plot(res["time"],[avrg["density"]]*len(res["time"]),'r-')
        plt.plot(res["time"],running_average(res["density"],size=RA["density"]),'b-')
        plt.legend(["MD data","Average value","Running average MD"])
        plt.show()
        
        # Energy statistic
        #---------------------------
        fig = plt.figure(indx+4,figsize=(10,8))
        fig.canvas.set_window_title('Energy')
        ax1 = plt.subplot(311)
        plt.title('Total energy')
        #plt.xlabel('Time [ps]')
        plt.ylabel('Energy [kcal/mol]')
        plt.xlim([res["time"][0],res["time"][-1]])
        plt.plot(res["time"],res["Etotal"],'k-')
        plt.plot(res["time"],[avrg["Etotal"]]*len(res["time"]),'r-')
        plt.plot(res["time"],running_average(res["Etotal"],size=RA["Etotal"]),'b-')
        plt.legend(["MD data","Average value","Running average MD"])
        plt.setp(ax1.get_xticklabels(), visible=False)
        
        ax2 = plt.subplot(312, sharex=ax1)
        plt.title('Potential energy')
        #plt.xlabel('Time [ps]')
        plt.ylabel('Energy [kcal/mol]')
        plt.xlim([res["time"][0],res["time"][-1]])
        plt.plot(res["time"],res["Epotential"],'k-')
        plt.plot(res["time"],[avrg["Epotential"]]*len(res["time"]),'r-')
        plt.plot(res["time"],running_average(res["Epotential"],size=RA["Epotential"]),'b-')
        plt.legend(["MD data","Average value","Running average MD"])
        plt.setp(ax2.get_xticklabels(), visible=False)
        
        plt.subplot(313, sharex=ax1)
        plt.title('Kinetic energy')
        plt.xlabel('Time [ps]')
        plt.ylabel('Energy [kcal/mol]')
        plt.xlim([res["time"][0],res["time"][-1]])
        plt.plot(res["time"],res["Ekinetic"],'k-')
        plt.plot(res["time"],[avrg["Ekinetic"]]*len(res["time"]),'r-')
        plt.plot(res["time"],running_average(res["Ekinetic"],size=RA["Ekinetic"]),'b-')
        plt.legend(["MD data","Average value","Running average MD"])
        plt.tight_layout()
        plt.show()
        
# TODO: Add kinetic energy with velocity distribution with optional Maxwell-Boltzmann distribution
                
def analize_amber_mdout(filename,size_RA=10):
    from ..read_mine import read_amber_mdout
    
    res, inp, avrg, rmsd = read_amber_mdout(filename)
    
    output_MD_statictic(res, inp, avrg, print_input=True, plot_stat=True, size_RA=size_RA)
            
            