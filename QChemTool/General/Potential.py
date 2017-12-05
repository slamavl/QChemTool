# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:58:41 2017

@author: Vladislav Sláma
"""

import numpy as np

# TODO: change imput to three cases: 1) imput vector dim (3) output potential - single number
# TODO:                              2) imput array dim (N,3) output potential - output vector -> If all coordinates corresponded to single point then total potential is np.sum(vector)
# TODO:                              3) imput array dim (N,M,3) output potential - array dim (N,M)

# FOR POTENTIAL WE HAVE TO IMPUT COORDINATES IN BOHR -> coor._value and not coor.value!!!!!!!

def potential_charge(q,R,eps=1):
    ''' function which calculate potential value at position R from point charge q.
    Potential is calculated in atomic units (potential*charge=Hartree)
    
    Parameters
    ----------
    R : numpy.array of real (dimension 3)
        Relative position of point for which we want to calculate potential.
        R=[x-x0,y-y0,z-z0] where x,y,z are coordiantes of point where potential 
        is calculated and x0,y0,z0 are coordinates of charge position
    q : real
        Charge in atomic units (in times of electron charge)
    eps : 
        Relative permitivity of environment 
        
    Returns
    -------
    pot : real
        Potential value at given point in atomic units [Hartree/e]
    
    Notes
    ------- 
    '''
# TODO: Maybe transform R to numpy array
    if np.ndim(R)==1:
        if len(R)!=3:
            raise IOError('Vectors in potential_charge have to be 3D')
        norm=np.sqrt(np.dot(R, R))
        pot=q/(norm*eps)
        return pot
    elif np.ndim(R)==2:
        if len(R[0])!=3:
            raise IOError('Vectors have to be in format Npoints x 3 for set of 3D vectors')
        norm=np.linalg.norm(R, axis=1)
        pot=q/(norm*eps)
# TODO: maybe if q scalar then summ potential into single value
        return pot
    elif np.ndim(R)==3:
        if len(R[0,0])!=3:
            raise IOError('Vectors have to be in format Npoints x Mpoints x 3 for set of 3D vectors')
        if len(q)!=R.shape[0]:
            raise IOError('For calculation of potential at a point you have to input all charges')
        norm=np.linalg.norm(R, axis=2)   # Norm has now dimension of NxM
        pot = 1/norm
        pot = np.dot(q, pot)
        pot = pot/eps
        # pot shoud be vector of size M which correspond po the potential at the position of M points 
        return pot
    
def potential_dipole(p,R,eps=1):
    ''' function which calculate potential value at position R from point dipole p.
    Potential is calculated in atomic units (potential*charge=Hartree)
    
    Parameters
    ----------
    R : numpy.array of real (dimension 3)
        Relative position of point for which we want to calculate potential.
        R=[x-x0,y-y0,z-z0] where x,y,z are coordiantes of point where potential 
        is calculated and x0,y0,z0 are coordinates of point dipole
    p : numpy.array of real (dimension 3)
        dipole in atomic units p=q*(r_+) - q*(r_-) where positions of positive (r_+)
        and negative (r_-) charge are in Bohrs and charges are in times of electron
        charge.
    eps : 
        Relative permitivity of environment 
        
    Returns
    -------
    pot : real
        Potential value at given point in atomic units [Hartree/e]
    
    Notes
    ------- 
    '''
    
    if np.ndim(R)==1:
        if len(R)!=3:
            raise IOError('Vectors in potential_charge have to be 3D')
        norm=np.sqrt(np.dot(R,R))
        pot=np.dot(p,R)/((norm**3)*eps)
        return pot
    
    elif np.ndim(R)==2:
        if R.shape[1]!=3:
            raise IOError('Vectors have to be in format Npoints x 3 for set of 3D vectors')
        norm=np.linalg.norm(R, axis=1)
        if np.ndim(p)==1:
            pot=np.dot(R,p.T)/((norm**3)*eps)     # R is Nx3 matrix, Norm is vector of lenght N, p is matrix of dimension Nx3 or vector of length 3
        else:
            if 1:       # both should give the same result but the first one is slightly faster (only tiny difference - you can use both)
                pot=np.einsum('ij,ij->i', R, p.T)
            else:
                pot=np.sum(R*p,axis=-1)
            pot=np.dot(pot,1/norm**3)
            pot=pot/eps
    elif np.ndim(R)==3:
        if R.shape[2]!=3:
            raise IOError('Vectors have to be in format Npoints x Mpoints x 3 for set of 3D vectors')
        if p.shape[0]!=R.shape[0] or p.shape[1]!=R.shape[2] :
            raise IOError('For calculation of potential at a point you have to input all dipoles')
        norm=np.linalg.norm(R, axis=2)   # Norm has now dimension of NxM
        
        PP = np.tile(p,(R.shape[1],1,1))
        PP = np.swapaxes(PP,0,1)   # P has now dimension NxMx3
        pot=np.sum(R*PP,axis=-1)
        pot=np.divide(pot,norm**3)
        pot=np.sum(pot,axis=0)
        pot=pot/eps
        
        return pot
            
            
def ElField_charge(q,R,eps=1):
    ''' function which calculate electric field at position R from point charge q.
    Electric field is calculated in atomic units (ElField*charge*Bohr=Hartree)
    
    Parameters
    ----------
    R : numpy.array of real (dimension 3)
        Relative position of point for which we want to calculate electric field.
        R=[x-x0,y-y0,z-z0] where x,y,z are coordiantes of point where electric 
        field is calculated and x0,y0,z0 are coordinates of charge position
    q : real
        Charge in atomic units (in times of electron charge)
    eps : 
        Relative permitivity of environment 
        
    Returns
    -------
    elf : numpy.array of real (dimension 3)
        Electric field at given point in atomic units [Hartree/(e*Bohr)]
    '''
    
    norm=np.sqrt(np.dot(R,R))
    elf=q/((norm**3)*eps)*R
    return elf
    
def ElField_dipole(p,R,eps=1):
    ''' function which calculate electric field at position R from point dipole p.
    Electric field is calculated in atomic units (ElField*charge*Bohr=Hartree)
    
    Parameters
    ----------
    R : numpy.array of real (dimension 3)
        Relative position of point for which we want to calculate electric field.
        R=[x-x0,y-y0,z-z0] where x,y,z are coordiantes of point where electric 
        field is calculated and x0,y0,z0 are coordinates of point dipole
    p : numpy.array of real (dimension 3)
        dipole in atomic units p=q*(r_+) - q*(r_-) where positions of positive (r_+)
        and negative (r_-) charge are in Bohrs and charges are in times of electron
        charge.
    eps : 
        Relative permitivity of environment 
        
    Returns
    -------
    elf : numpy.array of real (dimension 3)
        Electric field at given point in atomic units [Hartree/(e*Bohr)]
    
    Notes
    ------- 
    '''
    
    norm=np.sqrt(np.dot(R,R))
    elf=(3*np.dot(p,R)*R/(norm**5) - p/(norm**3))/eps
    return elf
    
    
        