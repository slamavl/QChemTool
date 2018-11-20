# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:52:15 2018

@author: Vladislav Sláma
"""

import numpy as np

def get_distance_matrixes(coor_i, coor_j):
    Nat_i = len(coor_i)
    Nat_j = len(coor_j)
    R = np.tile(coor_j,(Nat_i,1,1)) - np.swapaxes(np.tile(coor_i,(Nat_j,1,1)),0,1)
    # R[ii,jj] = coor_j[jj] - coor_i[ii]
    RR=np.sqrt(np.power(R[:,:,0],2)+np.power(R[:,:,1],2)+np.power(R[:,:,2],2))
    
#        # calculation of tensors with interatomic distances
#        R=np.zeros((self.Nat,self.Nat,3),dtype='f8') # mutual distance vectors
#        for ii in range(self.Nat):
#            for jj in range(ii+1,self.Nat):
#                R[ii,jj,:]=self.coor[ii]-self.coor[jj]
#                R[jj,ii,:]=-R[ii,jj,:]
#        RR=np.sqrt(np.power(R[:,:,0],2)+np.power(R[:,:,1],2)+np.power(R[:,:,2],2))  # mutual distances
    
    return R,RR

class MMAtoms:
    """
    Class which manages numerical propagation on time (simple molecular dynamics)
    
    Parameters
    ----------
    R_coor : numpy array of reals (dimension Nat x 3)
        Actual coordinates of atoms (electrons)
    R0_coor : numpy array of reals (dimension Nat x 3)
        Equilibrium coordinates of atoms (electrons)
    vel : numpy array (dimension Nat x 3)
        Actual atomic (electronic) velocities
    mass : numpy array (dimension Nat)
        Atomic masses
    accel : numpy array (dimension Nat x 3)
        Actual atomic (electronic) accelerations
    k_force : numpy array (dimension Nat)
        Force constants of harmonic oscillators
    direct : normalized numpy array (dimension Nat x 3)
        Vectors pointing in directions in which atom is allowed to move 
        (vectors normalized to 1)
    bonds : list (dimension Nbonds x 2)
        List of bonded atoms (nearest neighbors)
    
    """
    
    def __init__(self, R_coor, R0_coor, mass, vel, k_force, direct=None, bonds=None):
        self.Nat = len(R_coor)
        self.R_coor = R_coor.copy()
        self.R0_coor = R0_coor.copy()
        self.mass = np.tile(mass,(3,1)).T
        if vel is not None:
            self.vel = vel.copy()
        else:
            self.vel = vel
        self.accel = np.zeros((self.Nat,3),dtype='f8')
        self.k_force = np.tile(k_force,(3,1)).T
        self.direct = direct.copy()
        if bonds is not None:
            self.bonds = bonds.copy()
        else:
            self.bonds = bonds
        self.dt = None
        self.time = 0.0
        self.MASK = None
        self.RR = None
        self.R = None
        
        R,RR = get_distance_matrixes(R0_coor, R0_coor)
        self.RR = RR
        self.R = R
        
        # TODO: Normalize direct
    
    def get_force(self, RR3=None, RR5=None, MASK=None):
        if RR3 is None:
            RR3 = self.RR*self.RR*self.RR
        if RR5 is None:
            RR5 = RR3*self.RR*self.RR
            
        disp = self.R_coor - self.R0_coor
        
        """ Calculation of the total force acting on each particle - in AU"""
        # Contribution from harmonic oscillator
        Force = - disp*self.k_force
        
        # Contribution from dipole-dipole interaction
        XI = np.swapaxes(np.tile(disp,(self.Nat,1,1)),0,1)
        F = np.zeros(XI.shape,dtype="f8")
        for jj in range(3):
            #F[:,:,jj] = XI[:,:,jj]/RR3 - 3*self.R[:,:,jj]*(np.sum(XI*self.R,axis=2))/RR5
            #F[:,:,jj] = np.fill_diagonal(F[:,:,jj],0.0)
            F[:,:,jj] = - np.divide(XI[:,:,jj], RR3, out=np.zeros_like(XI[:,:,jj]), where=RR3!=0)
            temp = np.sum(XI*self.R,axis=2)
            F[:,:,jj] += 3*self.R[:,:,jj] * np.divide(temp, RR5, out=np.zeros_like(temp), where=RR5!=0)
            # Keep only nearest neighbor contribution
            if MASK is not None:
                F[MASK,jj] = 0.0
        Force += np.sum(F, axis=0)
        #print(F[:,:,0])
        
        # If movement is restricted to some direction
        if self.direct is not None:
            Force_projected = np.sum(self.direct*Force,axis=1)
            Force = self.direct * np.tile(Force_projected,(3,1)).T
        
        return Force
    
    def propagate(self, dt, Nsteps, Nstep_write, MASK=None, init_vel=False):
        RR3 = self.RR*self.RR*self.RR
        RR5 = RR3*self.RR*self.RR
        
        coor_write = []
        
        if self.vel is None:
            init_vel = True

        # Init velocities at half-step        
        if init_vel:
            Force = self.get_force(RR3=RR3, RR5=RR5, MASK=MASK)
            self.accel = Force / self.mass
            self.vel = self.accel*dt/2
        
        # Propagate particles 
        for step in range(Nsteps+1):
            if step%Nstep_write == 0:
                coor_write.append(self.R_coor)
            self.R_coor = self.R_coor + self.vel*dt
            Force = self.get_force(RR3=RR3, RR5=RR5, MASK=MASK)
            self.accel = Force/ self.mass
            self.vel = self.vel + self.accel*dt
            if step == 1000:
                print(Force)
        
        return coor_write
        

    
    