# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 14:18:01 2018

@author: Vladislav Sláma
"""

import numpy as np
from ..General.Potential import potential_charge
from ..General.UnitsManager import position_units
from .Classes.general import Coordinate

def RESP_fit_charges(struc,ESP_coor,ESP_pot, Q_tot=0.0, restr=0.001, H_fit=True,
                     List=None, MaxInt=100, q_tol=1e-7, NoHrestr=True, 
                     constrains=[], Freeze=[], Type = "Ground", verbose=False):
    """ Function for restrained ESP fit of atomic center charges from 
    electrostatic potential
    
    Parameters
    ----------
    struc : Structure class
        Structure of the molecule for which ESP was generated and which will be
        used for charge fitting
    ESP_coor : numpy array of real (dimension Npoints x 3)
        Positions of points for where ESP was evaluated in ATOMIC UNITS (Bohr)
    ESP_pot : numpy array of real (dimension Npoints)
        Values of ESP in ATOMIC UNITS for molecule at points defined in ``ESP_coor``
    Q_tot : float (optional init = 0.0)
        Total charge if the molecule - constrain for fitting (Qtot = sum of all
        fitted charges). Default is neutral molecule.
    restr : float (optional init = 0.001)
        Restrain weight for hyperbolic restrains. Default value 0.001 AU 
        (ATOMIC UNITS) correspond to tight restrain criteria.
    H_fit : logical (optional init = True):
        Include hydrogens as centers for RESP fitting of atomic charges
    List : list of integer (optional init = None)
        Atomic indexes of atoms which are used as centers for RESP fitting of
        atomic charges. If not specified whole molecule is used (or molecule
        without hydrogens if ``H_fit=False``)
    Type : string (optional init = "Ground")
        If whole molecule is used for fitting charges, charges are also written
        to ``struc.esp_grnd`` or ``struc.esp_exct`` or ``struc.esp_trans`` which is 
        specified by ``Type= "Ground"``, ``Type= "Excited"`` and ``Type= "Transition"``
        respectively
    MaxInt : integer (optional init = 100)
        Maximal number of steps for SCF procedure to reach convergence
    q_tol : float (optional init = 1e-6)
        Convergence criteria for SCF procedure for difference between charges 
        from two following steps.
    NoHrestr : logical (optional init = True):
        If ``True`` no restrains for charge fitting will be used for hydrogen
        centers. If ``False`` same restrains will be used for all atoms including
        hydrogens
    constrains : list of list of integer (optional init = [])
        List of atoms with the same charges (e.g. for symmetry or other reasons).
        For example ``constrains=[[1,5,8,10],[2,6]]`` means that q[1]=q[5]=q[8]=q[10]
        and q[2]=q[6]. For using constrains it is recommended to use the whole
        structure for fitting because after exluding the hydrogens indexes might
        change. Indexing is from zero!
    Freeze : list (dimension N_frozen_atoms x 2) (optional)
        List of indexes and charges for atoms which should be frozen on constant
        charge. for example: ``Freeze=[[1,0.0],[5,0.25],...] means that charge
        of atom with index 1 will be frozen on 0.0, charge of atom with index 
        5 wil be frozen on 0.25 etc.
        
    Returns
    -------
    result : dictionary
        Dictionary with all possible results from charge fitting. Dictionary
        keys are:
            
          * **charge** : numpy array of float (dimension Ncharge)
            Fitted charges from RESP procedure
          * **coor** : coordinate type
            Coordinates of atomic centers for which charges were fitted
          * **steps** : integer
            Number of SCF steps until convergence was obtained
          * **Chi_square** : list of float (dimension steps)
            Sum of differences, between fited and calculated ESP, squared.
            ``Sum_i{ (ESPcalc_i - ESPfit_i)^2 }``. Calculated for every step of
            SCF procedure
          * **RRMS** : list of float
            Relative root mean square error. 
            ``RRMS=sqrt{Chi_square/sum_i{ESPcalc_i^2}}``
          * **q_change** : list of float
            Sum of diferences between charges from two following steps divided
            by number of atoms 
          * **potential_fit** : numpy array of float (dimension Npoints)
            Electrostatic potential calculated from fitted charges evaluated at 
            the same points as QC ESP which was needed as an input
          * **potential_calc** : numpy array of float (dimension Npoints)
            QC ESP potential. The same as ESP_pot which was required as an input.
          * **dipole** : numpy array of float (dimension 3)
            Electrostatic dipole moment calculated from fitted charges
          * **quadrupole_traceless** : list of float (dimension 6)
            Electrostatic traceless quadrupole moment calculated from fitted 
            charges. Ordering is following: [Qxx,Qxy,Qxz,Qyy,Qyz,Qzz]
          * **charge_harmonic** : numpy array of float (dimension Ncharge)
            Fitted charges from RESP procedure for harmonic restrains
          * **Chi_square_harmonic** : list of float (dimension steps)
            The same as ``Chi_square`` only for harmonic restrains
          * **RRMS_harmonic** : list of float
            The same as ``RRMS`` only for harmonic restrains
          * **potential_harmonic** : numpy array of float (dimension Npoints)
            The same as ``potential_fit`` only for harmonic restrains
          * **dipole_harmonic** : numpy array of float (dimension 3)
            The same as ``dipole`` only for harmonic restrains
          * **quadrupole_traceless_harmonic** : list of float (dimension 6)
            The same as ``quadrupole_traceless`` only for harmonic restrains

    Notes
    -------
    **With default settings provides the same results as AMBER RESP fitting 
    procedure.**
    
    **Quadrupole moment should be calculated after transformation of coordinate
    origin into center of mass and align axes wit main axes of inetria tensor.**
    
    Ispired by "A Well-Behaved Electrostatic Potential Based Method Using Charge 
    Restraints for Deriving Atomic Charges: The RESP Model, Christopher 
    I. Bayly, Piotr Cieplak, Wendy D. Cornell, and Peter A. Kollman, 
    J. Phys. Chem., 1993, 97, 10269-10280 "  
    """
    
    def add_hyperbolic_restrain(A,b,q,restrain,Ncharge):
        A_work = A.copy()
        b_work = b.copy()
        for ii in range(Ncharge):
            A_work[ii,ii] += restrain[ii] / np.sqrt( q[ii]**2 + 0.01 )
        return A_work,b_work
    
    def add_harmonic_restrain(A,b,q,restrain,Ncharge):
        A_work = A.copy()
        b_work = b.copy()
        for ii in range(Ncharge):
            A_work[ii,ii] += 2*restrain[ii]
            b_work[ii] += 2*restrain[ii]*q[ii]
        return A_work,b_work
    
    if List is not None:
        Ncharge = len(List)
        coor = struc.coor._value[List] # Charge centers for RESP fit in AU (Bohr)
        at_type = np.array(struc.at_type)[List]
    elif H_fit:
        Ncharge = struc.nat
        coor = struc.coor._value    # Charge centers for RESP fit in AU (Bohr)
        at_type = np.array(struc.at_type)
    else:
        MASK = (np.array(struc.at_type) != "H")
        coor = struc.coor._value[MASK] # Charge centers for RESP fit in AU (Bohr)
        at_type = np.array(struc.at_type)[MASK]
        Ncharge = len(coor)
    
    nESP = len(ESP_pot)
    q0 = np.zeros(Ncharge,dtype='f8') # Set initial charges for constrains (for harmonic constrains)
    q_rest = np.ones(Ncharge,dtype='f8')*restr
    
    # Fill vector matrix with vectors from atoms to ESP points
    R = np.tile(coor,(nESP,1,1))
    R = np.swapaxes(R,0,1)
    Resp = np.tile(ESP_coor,(Ncharge,1,1))
    R = Resp - R  # Rij is vector pointing from atom i to ESP point j = rj - Ri 
    
    # Assign lagrange constrains. So far only constrain is sum of all charges.
    C = np.ones((1,Ncharge),dtype='f8')
    d = np.zeros(1,dtype='f8')
    d[0] = Q_tot 
    for ii in range(len(constrains)):
        for jj in range(1,len(constrains[ii])):
            cc = np.zeros(Ncharge,dtype='f8')
            cc[constrains[ii][0]] = 1
            cc[constrains[ii][jj]] = -1
            C = np.vstack( (C,cc) )
            d = np.append(d,0.0)
    for ii in range(len(Freeze)):
        cc = np.zeros(Ncharge,dtype='f8')
        cc[Freeze[ii][0]] = 1
        C = np.vstack( (C,cc) )
        d = np.append(d,Freeze[ii][1])
    CT = C.T
    CT = np.vstack( (CT, np.zeros((C.shape[0],C.shape[0]),dtype="f8") ) )
    
    # Build the left side matrix of the system of linear equations Ajk
    A = np.zeros((Ncharge, Ncharge),dtype="f8")
    b = np.zeros(Ncharge,dtype="f8")
    
    if verbose:
        print("Building distance matrix")
        
    # Set distance matrix
    R2=np.power(R[:,:,0],2)+np.power(R[:,:,1],2)+np.power(R[:,:,2],2)    
    RR=1/np.sqrt(R2)
    
    if verbose:
        print("Building matrix for the system of linear equations")
    
    for ii in range(nESP):
        for kk in range(Ncharge):
            #rik = 1.0 / np.linalg.norm(coor[kk]-ESP_coor[ii])
            rik = RR[kk,ii]
            A[kk,kk] += 2*(rik*rik)
            b[kk] += 2 * ESP_pot[ii] * rik
            for jj in range(kk+1,Ncharge):
                #rij = 1.0 / np.linalg.norm(coor[jj]-ESP_coor[ii])
                rij = RR[jj,ii]
                A[jj,kk] += 2*rij*rik
    # symmetrization of A matrix
    for kk in range(Ncharge):
        for jj in range(kk):
            A[jj,kk] = A[kk,jj]
            
    # Include Lagrange constrains to the A matrix and b vector
    A = np.vstack( (A, C) )
    A = np.hstack( (A, CT) )
    b = np.append(b,d)
    
    if verbose:
        print("... all matrix build")

    # zero out restrains for hydrogens if required
    if NoHrestr:
        H_indx = np.where(at_type == "H")[0]
        q_rest[H_indx] = 0.0
    
    if verbose:
        print("Solving harmonic initial guess")
    
    # Calculate initial guess with harmonic restrains
    A_work,b_work = add_harmonic_restrain(A,b,q0,q_rest,Ncharge)
    res = np.linalg.solve(A_work,b_work)
    q = res[:Ncharge]  # initial charges for SCF procedure with hyperbolic restrains
    
    if verbose:
        print("... harmonic calculation completed")
    
    # Results for harmonic restrains
    q_harmonic = q.copy()
    pot_harmonic = potential_charge(q_harmonic,R)
    pot_diff = pot_harmonic - ESP_pot
    Chi_sqr_harmonic = np.dot(pot_diff,pot_diff)
    RRMS_harmonic = np.sqrt(Chi_sqr_harmonic/np.dot(ESP_pot,ESP_pot))
    dipole_harmonic = np.dot(q_harmonic,coor)
    quadrupole_traceless, quadrupole = ESP_get_quadrupole(coor,q_harmonic)
    result={"charge_harmonic": q_harmonic,
            "potential_harmonic": pot_harmonic,"Chi_square_harmonic": Chi_sqr_harmonic,
            "RRMS_harmonic": RRMS_harmonic,"dipole_harmonic": dipole_harmonic,
            "quadrupole_harmonic": quadrupole,
            "quadrupole_traceless_harmonic": quadrupole_traceless}
    
    # SCF calculation procedure
    condition = True
    count = 0
    Chi_sqr = []
    RRMS = []
    q_change = []
    while condition:
        q_old = q.copy()
        # Add hyperbolic restrain to A matrix and b vector
        A_work,b_work = add_hyperbolic_restrain(A,b,q,q_rest,Ncharge)
        
        # Solve the system of linear equations
        res = np.linalg.solve(A_work,b_work)
        q = res[:Ncharge]
        
        count += 1
        
        # Check convergence
        q_change_i = np.linalg.norm( q - q_old ) / Ncharge
        q_change.append(q_change_i)      
        
        # calculate RMS error:
        if True:
            pot_fit = potential_charge(q,R)
            pot_diff = pot_fit - ESP_pot
            Chi_sqr_i = np.dot(pot_diff,pot_diff)
            RRMS_i = np.sqrt(Chi_sqr_i/np.dot(ESP_pot,ESP_pot))
            RRMS.append(RRMS_i)
            Chi_sqr.append(Chi_sqr_i)
        
        if q_change_i < q_tol:    # If change in charges is small calculation converged
            condition = False
        
        if count > MaxInt:      # If more then maximum steps are required - SCF is unable to converge
            condition = False
            raise Warning("RESP SCF procedure did't converge within specified maximal number of steps")
    
    if List is None and H_fit:
        if Type=="Ground":
            struc.esp_grnd = q.copy()
        elif Type=="Excited":
            struc.esp_exct = q.copy()
        elif Type=="Transition":
            struc.esp_trans = q.copy()
    
    # Results 
    Rcom = struc.get_com()
    dipole = np.dot(q,coor)
    quadrupole_traceless, quadrupole = ESP_get_quadrupole(coor,q,origin=Rcom._value)
    
    with position_units("Bohr"):
        coor = Coordinate(coor)
    result.update({"charge": q,"coor": coor,"Chi_square": Chi_sqr,"RRMS": RRMS,
            "steps": count,"q_change": q_change,"potential_fit": pot_fit,
            "potential_calc": ESP_pot,"dipole": dipole,"quadrupole": quadrupole,
            "quadrupole_traceless": quadrupole_traceless})
    
    return result

    
def ESP_get_quadrupole(coor, charge, origin=np.zeros(3,dtype="f8") ):
    """ Calculate quadrupole moment for set of point charges. If coordinates
    are given in ATOMIC UNITS units of quadrupole will be elemental charge * Bohr^2.
    
    Parameters
    ----------
    coor : numpy array of float (dimension Ncharge x 3)
        Aoordinates of charge centers (in ATOMIC UNITS)
    charge : numpy array of float (dimension Ncharge)
        Charges for every position (atom)
    origin : numpy array of float (dmension 3) (optional)
        Coordinate system origin (in ATOMIC UNITS)
    
    Returns
    --------
    Quad_traceless : list of float (dimension 6)
        Traceles quadrupole moment ordered as [Qxx,Qxy,Qxz,Qyy,Qyz,Qzz]
    Quad : list of float (dimension 6)
        Quadrupole moment ordered as [Qxx,Qxy,Qxz,Qyy,Qyz,Qzz]
    
    Notes
    ------
    Traceless quadrupole is defined as:
    ``Qij = sum_l{ q_l*[3*r_il*rjl - |r_l|^2 * delta_ij] }``
    where i,j is in {x,y,z} and l goes through all charges
    
    """
    
    # usualy coordinate system origin in center of mass
    X = coor[:,0] - origin[0]
    Y = coor[:,1] - origin[1]
    Z = coor[:,2] - origin[2]
    XX = X**2
    YY = Y**2
    ZZ = Z**2
    
    # Calculation of traceless quadrupole defined as Qij = sum_l{ q_l*[3*r_il*rjl - |r_l|^2 * delta_ij] }    
    # Qxx
    R = 2*XX - YY - ZZ
    Qxx = np.dot(charge,R)
    # Qyy
    R = 2*YY - XX - ZZ
    Qyy = np.dot(charge,R)
    # Qzz
    R = 2*ZZ - XX - YY
    Qzz = np.dot(charge,R)
    # Q xy
    Qxy = np.dot(charge,X*Y)
    # Q xz
    Qxz = np.dot(charge,X*Z)
    # Q yz
    Qyz = np.dot(charge,Y*Z)
    
    Quad_traceless = [Qxx,Qxy,Qxz,Qyy,Qyz,Qzz]
    
    # Calculation of quadrupole tensor with nonzero trace
    Qxx = np.dot(charge,XX)
    Qyy = np.dot(charge,YY)
    Qzz = np.dot(charge,ZZ)
    
    Quad = [Qxx,Qxy,Qxz,Qyy,Qyz,Qzz]
    
    return Quad_traceless, Quad
    