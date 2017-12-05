# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:05:08 2016

@author: User
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

rH=0.3
rC=0.4
rN=0.4
rO=0.4
bond_length=2.0
bond_lengthH=1.3

Kb = 0.69503476   # cm-1/K

def plotSphere(coor,rr,clr,ax):
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:6j, 0.0:2.0*pi:6j]
    x = rr*sin(phi)*cos(theta)+coor[0]
    y = rr*sin(phi)*sin(theta)+coor[1]
    z = rr*cos(phi)+coor[2]
    
    #Set colours and render
    ax.plot_surface(x, y, z,  rstride=1, cstride=1, color=clr, linewidth=0)


def PlotMolecule(coord,atomtype):
    fig = plt.figure()    
    ax = fig.add_subplot(111, projection='3d')  
    NAtom=len(atomtype)
    xsize=np.max(coord[:,0])-np.min(coord[:,0])
    ysize=np.max(coord[:,1])-np.min(coord[:,1])
    zsize=np.max(coord[:,2])-np.min(coord[:,2])    

    size=5500/np.max([xsize,ysize,zsize])    
    
    for ii in range(NAtom):
        if atomtype[ii]=='H':
#            plotSphere(coord[ii,:],rH,'w',ax)
            ax.scatter(coord[ii,0], coord[ii,1], coord[ii,2], s=np.pi*rH**2*size, c='w')
        if atomtype[ii]=='C':
#            plotSphere(coord[ii,:],rC,'k',ax)
            ax.scatter(coord[ii,0], coord[ii,1], coord[ii,2], s=np.pi*rC**2*size, c='k')
        if atomtype[ii]=='O':
#            plotSphere(coord[ii,:],rO,'r',ax)
            ax.scatter(coord[ii,0], coord[ii,1], coord[ii,2], s=np.pi*rO**2*size, c = 'r')
        if atomtype[ii]=='N':
#            plotSphere(coord[ii,:],rN,'b',ax)
            ax.scatter(coord[ii,0], coord[ii,1], coord[ii,2], s=np.pi*rN**2*size, c= 'b')
    #ax.gca().set_aspect('equal', adjustable='box')
    for ii in range(NAtom):
        for jj in range(ii+1,NAtom):
            rr=coord[ii,:]-coord[jj,:]
            if atomtype[ii]=='H' or atomtype[jj]=='H':
                crit_length=bond_lengthH
            else:
                crit_length=bond_length
            if np.sqrt(np.dot(rr,rr))<crit_length:
                ax.plot([coord[ii,0],coord[jj,0]],[coord[ii,1],coord[jj,1]],[coord[ii,2],coord[jj,2]], 'k-', lw=3)


    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    #ax.view_init(90, 90)
    plt.show()
    
def PlotTwoMolecules(coord1,atomtype1,coord2,atomtype2):
    fig = plt.figure()    
    ax = fig.add_subplot(111, projection='3d')
    xsize=np.max([coord1[:,0],coord2[:,0]])-np.min([coord1[:,0],coord2[:,0]])
    ysize=np.max([coord1[:,1],coord2[:,1]])-np.min([coord1[:,1],coord2[:,1]])
    zsize=np.max([coord1[:,2],coord2[:,2]])-np.min([coord1[:,2],coord2[:,2]])
    size=16000/np.max([xsize,ysize,zsize])
   
    for ii in range(len(atomtype1)):
        if atomtype1[ii]=='H':
            ax.scatter(coord1[ii,0], coord1[ii,1], coord1[ii,2], s=np.pi*rH**2*size, c='w')
        if atomtype1[ii]=='C':
            ax.scatter(coord1[ii,0], coord1[ii,1], coord1[ii,2], s=np.pi*rC**2*size, c='k')
        if atomtype1[ii]=='O':
            ax.scatter(coord1[ii,0], coord1[ii,1], coord1[ii,2], s=np.pi*rO**2*size, c = 'r')
        if atomtype1[ii]=='N':
            ax.scatter(coord1[ii,0], coord1[ii,1], coord1[ii,2], s=np.pi*rN**2*size, c = 'b')
    for ii in range(len(atomtype2)):
        if atomtype2[ii]=='H':
            ax.scatter(coord2[ii,0], coord2[ii,1], coord2[ii,2], s=np.pi*rH**2*size, c='w', alpha=0.5)
        if atomtype2[ii]=='C':
            ax.scatter(coord2[ii,0], coord2[ii,1], coord2[ii,2], s=np.pi*rC**2*size, c='k', alpha=0.5)
        if atomtype2[ii]=='O':
            ax.scatter(coord2[ii,0], coord2[ii,1], coord2[ii,2], s=np.pi*rO**2*size, c = 'r', alpha=0.5)
        if atomtype2[ii]=='N':
            ax.scatter(coord2[ii,0], coord2[ii,1], coord2[ii,2], s=np.pi*rN**2*size, c = 'b', alpha=0.5)
            
    NAtom=len(atomtype1)        
    for ii in range(NAtom):
        for jj in range(ii+1,NAtom):
            rr=coord1[ii,:]-coord1[jj,:]
            if atomtype1[ii]=='H' or atomtype1[jj]=='H':
                crit_length=bond_lengthH
            else:
                crit_length=bond_length
            if np.sqrt(np.dot(rr,rr))<crit_length:
                ax.plot([coord1[ii,0],coord1[jj,0]],[coord1[ii,1],coord1[jj,1]],[coord1[ii,2],coord1[jj,2]], 'k-', lw=3)
    NAtom=len(atomtype2)
    for ii in range(NAtom):
        for jj in range(ii+1,NAtom):
            rr=coord2[ii,:]-coord2[jj,:]
            if atomtype2[ii]=='H' or atomtype2[jj]=='H':
                crit_length=bond_lengthH
            else:
                crit_length=bond_length
            if np.sqrt(np.dot(rr,rr))<crit_length:
                ax.plot([coord2[ii,0],coord2[jj,0]],[coord2[ii,1],coord2[jj,1]],[coord2[ii,2],coord2[jj,2]], 'k-', lw=3)
    #ax.view_init(90, 90)
                
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    
def plotHistogramBuff(data,db,MinBin,MaxBin):
    Nbins=int(np.ceil((MaxBin-MinBin)/db))
    BinList=np.zeros(Nbins+1)
    for ii in range(Nbins+1):
        BinList[ii]=MinBin+ii*db
    plt.hist(data, bins=BinList, normed=False) #(array([0, 2, 1]), array([0, 1, 2, 3]), <a list of 3 Patch objects>)
    
def plotHistogramNormBuff(data,db,MinBin,MaxBin):
    Nbins=int(np.ceil((MaxBin-MinBin)/db))
    BinList=np.zeros(Nbins+1)
    for ii in range(Nbins+1):
        BinList[ii]=MinBin+ii*db
    plt.hist(data, bins=BinList, normed=True,alpha=0.5) #(array([0, 2, 1]), array([0, 1, 2, 3]), <a list of 3 Patch objects>)
    
def plotHistogramAndProbBuff(data,MDnum,db,MinBin,MaxBin,Temp,Force):
    Nbins=int(np.ceil((MaxBin-MinBin)/db))
    BinList=np.zeros(Nbins)
    Prob=np.zeros(Nbins)
    
    for ii in range(Nbins):
        BinList[ii]=MinBin+ii*db
    
    Prob=np.sqrt(Force/(2.0*np.pi*Kb*Temp))*np.exp(-Force*(np.square(BinList))/(2.0*Kb*Temp))    
    plt.hist(data, bins=BinList) #(array([0, 2, 1]), array([0, 1, 2, 3]), <a list of 3 Patch objects>)
    plt.plot(BinList,Prob*MDnum*db,'r', lw=3)    
    
def plotHistogramAndProbNormBuff(data,db,MinBin,MaxBin,Temp,Force):
    Nbins=int(np.ceil((MaxBin-MinBin)/db))
    NProb=int(np.ceil((MaxBin-MinBin)/(db/100)))
    BinList=np.zeros(Nbins)
    ProbList=np.zeros(NProb)
    Prob=np.zeros(Nbins)
    
    
    for ii in range(Nbins):
        BinList[ii]=MinBin+ii*db
    for ii in range(NProb):
        ProbList[ii]=MinBin+ii*db/100
    
    Prob=np.sqrt(Force/(2.0*np.pi*Kb*Temp))*np.exp(-Force*(np.square(ProbList))/(2.0*Kb*Temp))    
    plt.hist(data, bins=BinList, normed=True,alpha=0.5) #(array([0, 2, 1]), array([0, 1, 2, 3]), <a list of 3 Patch objects>)
    plt.plot(ProbList,Prob,'r', lw=3)    
