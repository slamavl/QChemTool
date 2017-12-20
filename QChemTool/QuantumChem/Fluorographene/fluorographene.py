# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:55:05 2017

@author: Vladislav Sláma
"""
import numpy as np

from ...General.UnitsManager import position_units

def deleteFG(struc,add_hydrogen=False,Hdist=2.0031,Fragmentize=False):
    """ Delete fluorographene atoms from structure and keeps only defect atoms.
    
    Parameters
    ----------
    struc : Structure class
        Contains information about structure of fluorographen sheet with defects
    add_hydrogen : logical (optional init = False)
        If hydrogens should be added to replace cutted bonds between defect 
        carbons and fluorographene carbons
    Hdist : float (optional init = 2.0031 Bohr)
        Distance between hydrogen and connected carbon atom in Bohr ( C-H distance).
    Fragmentize : logical (optional init = False)
        If ``True`` output will be list of separated structures corresponding 
        to individual defects. If ``Talse`` output will be single structure
        with all defects
    
    Returns
    --------
    new_struc : Structure class or list of Structure class
        Structure or list of structures with defects cutted from fluorographene
        with exactly the same coordinates as in fluorographene
    """
    
    if struc.bonds is None:
        struc.guess_bonds()
    is_FG=np.zeros(struc.nat,dtype='bool')
    for ii in range(len(struc.bonds)):
        Atom1=struc.bonds[ii,0]
        Atom2=struc.bonds[ii,1]
        if struc.at_type[Atom1]=='C' and struc.at_type[Atom2]=='F':
            is_FG[Atom1]=True
            is_FG[Atom2]=True
        elif struc.at_type[Atom1]=='F' and struc.at_type[Atom2]=='C':
            is_FG[Atom1]=True
            is_FG[Atom2]=True
    inxed_FG=np.arange(struc.nat)
    inxed_FG=list(inxed_FG[is_FG])
    
    new_struc=struc.delete_by_indx(indx=inxed_FG)
    if add_hydrogen:
        count=np.zeros(new_struc.nat,dtype='i8')
        new_struc.guess_bonds()
        for bond in new_struc.bonds:
            count[bond[0]] += 1
            count[bond[1]] += 1
        H_missing = (count==2)
        indx2addH = np.arange(new_struc.nat)[H_missing]   # index of atoms to which hydrogens will be added
        connections=[]
        for ii in range(new_struc.nat):
            connections.append([])
        for bond in new_struc.bonds:
            if bond[0] in indx2addH or bond[1] in indx2addH:                
                connections[bond[0]].append(bond[1])
                connections[bond[1]].append(bond[0])
        for ii in range(len(indx2addH)):
            indx0=indx2addH[ii]
            indx1=connections[indx0][0]
            indx2=connections[indx0][1]
            Hvec = 2*new_struc.coor._value[indx0] - new_struc.coor._value[indx1] - new_struc.coor._value[indx2]
            Hvec = Hvec*Hdist/np.linalg.norm(Hvec)
            Hvec = Hvec + new_struc.coor._value[indx0]
            with position_units('Bohr'):
                new_struc.add_atom(Hvec,'H')
    if Fragmentize:
        return new_struc.fragmentize()
    else:
        return new_struc

def constrainsFG(struc,border=True,defect=False):
    """ Output indexes of atoms to be frozen during geometry optimization for 
    fluorographene systems
    
    Parameters
    ----------
    struc : Structure class
        Contains information about structure of fluorographen sheet with defects
    border : logical (optional init = True)
        Specify if border carbon atoms (of fluorographene sheet) should be
        frozen or not (if should be included to constrain indexes)
    defect : logical (optional init = False)
        Specify if defect carbons (the one without fluorines) should be frozen
         or not (if should be included to constrain indexes). Usefull for 
         example for geometry optimization of single defect in geometry from
         two defect structure.
         
    Returns
    -------
    constrain_indx : list of integer (dimension Nconstrain)
        Indexes of atoms which should be frozen during geometry optimization
        (starting from 0).
    
    """
    
    if struc.bonds is None:
        struc.guess_bonds()
    
    # border carbons - conected to only 2 other carbons + cabons conected to these
    if border:
        constrains=np.zeros(struc.nat,dtype='bool')
        Nbonds = np.zeros(struc.nat,dtype='i8')
        connect_CC=[]
        connect_CF=[]
        for ii in range(struc.nat):
            connect_CC.append([])
            connect_CF.append([])
        for bond in struc.bonds:
            if struc.at_type[bond[0]]=='C' and struc.at_type[bond[1]]=='C':
                Nbonds[bond[0]] += 1
                Nbonds[bond[1]] += 1
                connect_CC[bond[0]].append(bond[1])
                connect_CC[bond[1]].append(bond[0])
            elif struc.at_type[bond[0]]=='C' and struc.at_type[bond[1]]=='F':
                connect_CF[bond[0]].append(bond[1])
                connect_CF[bond[1]].append(bond[0])
            elif struc.at_type[bond[0]]=='F' and struc.at_type[bond[1]]=='C':
                connect_CF[bond[0]].append(bond[1])
                connect_CF[bond[1]].append(bond[0])
            
        BorderC1=np.where(Nbonds==2)[0]
        
        for ii in BorderC1:
            constrains[ii]=True
            for jj in connect_CC[ii]:
                constrains[jj]=True
    
    # Defect carbons - carbons without fluorines
    if defect:
        for ii in range(struc.nat):
            if len(connect_CF[ii])==0:
                constrains[ii]=True
    
    constrain_indx=list(np.arange(struc.nat)[constrains])
    
    return constrain_indx