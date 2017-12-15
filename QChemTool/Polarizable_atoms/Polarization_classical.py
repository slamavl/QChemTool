# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:33:56 2017

@author: Vladislav Sláma
"""

import numpy as np
from scipy.spatial.distance import pdist,squareform
from ..QuantumChem.calc import GuessBonds
from ..QuantumChem.Classes.molecule import Molecule
from ..QuantumChem.read_mine import read_TrEsp_charges
from ..QuantumChem.calc import identify_molecule
from ..QuantumChem.output import OutputMathematica 
from ..QuantumChem.positioningTools import CenterMolecule
from ..General.units import conversion_facs_energy

debug=False

#==============================================================================
#  Definition of class for polarizable environment
#==============================================================================
    
class Dielectric:    
    ''' Class managing dielectric properties of the material 
    
    Parameters
    ----------
    coor : numpy.array of real (dimension Nx3) where N is number of atoms
        origin of density grid
        
    polar : numpy.array or list of real (dimension N)
        Polarizabilities for every atom
        
    charge : numpy.array or list of real (dimension N)
        charges on individual atoms (initial charges)

    dipole : numpy.array of real (dimension Nx3)
        dipole on individual atoms (initial dipole)
    '''
    
    def __init__(self,Coor,Charge,Dipole,Polar):
        self.coor=np.copy(Coor)
        self.polar=np.copy(Polar)
        self.charge=np.copy(Charge)
        self.dipole=np.copy(Dipole)
        self.Nat=len(Coor)
    
    def assign_polar(self,pol_type,**kwargs):
        
        ''' For now assignment is working only for fluorographene carbons with 
        type 'CF' and defect carbons with type 'CD' 

        Parameters
        ----------
        pol_type : numpy.array or list of str (dimension N)
            Polarization atomic types for assign of polarizabilities - diferent from
            atomic types - for example group C-F will be treated as single atom and
            type will be pol_type='CF'.
        **kwargs : dict
            dictionary with three matrixes for every polarizable atom type. For
            example: kwargs['PolValues']['CF'][0] is Alpha(E) polarizability
            matrix for atom tyle 'CF'. [1] correspond to Alpha(-E) matrix and 
            [2] to Beta(E,E)
        
        Returns
        -------
        polar : numpy.array or list of real (dimension N)
            Polarizabilities for every atom. 'CF'=1.03595 and 'CD'=1.4
        '''
        ZeroM=np.zeros((3,3),dtype='f8')
        PolValues={'CF': [ZeroM,ZeroM,ZeroM],
                   'CD': [ZeroM,ZeroM,ZeroM],'C': [ZeroM,ZeroM,ZeroM],'F': [ZeroM,ZeroM,ZeroM]} 
        for key in list(kwargs.keys()):
            if key=='PolValues':
                PolValues=kwargs['PolValues']
        
        if self.Nat!=len(pol_type):
            raise IOError('Polarization type vector must have the same length as number of atoms')
        
        polar=np.zeros((len(pol_type),3,3),dtype='f8')
        for ii in range(len(pol_type)):
            polar[ii,:,:]=PolValues[pol_type[ii]][0]
        return polar
    
    def swap_atoms(self,index1,index2):
        ''' Function which exchange polarization properties between atoms defined
        by index1 and atoms defined by index 2 
        
        index1 : list or numpy.array of integer (dimension Natoms_change)
            Indexes of first set of atoms which we would like to swap
        index2 : list or numpy.array of integer (dimension Natoms_change)
            Indexes of second set of atoms which we would like to swap
            
        '''

        if len(index1)!=len(index2):
            raise IOError('You can swap values only between same number of atoms')
        
        for ii in range(len(index1)):
            # swap charges
            self.charge[index1[ii]],self.charge[index2[ii]] = self.charge[index2[ii]],self.charge[index1[ii]]
            # swap dipoles
            self.dipole[index1[ii],:],self.dipole[index2[ii],:] = self.dipole[index2[ii],:],self.dipole[index1[ii],:]
            # swap polarizabilities
            #print(np.shape(self.dipole),index1[ii])
            self.polar[index1[ii],:,:],self.polar[index2[ii],:,:] = self.polar[index2[ii],:,:],self.polar[index1[ii],:,:]
            
    
    def calc_dipoles(self,Estatic=np.zeros(3,dtype='f8'),dDip=0.0001,eps=1,verbose=False,Bonds=None,**kwargs):
        ''' Function for calculation induced dipoles of SCF procedure for interaction
        of molecule with environment. It calculates induced dipoles on individual
        atoms by static charge distribution and homogeneous electric field.
        
        Parameters
        ----------
        Estatic : numpy.array of real (dimension 3) (optional - init=np.zeros(3,dtype='f8'))
            External homogeneous electric fiel vectord (orientation and strength)
            in ATOMIC UNITS. By default there is no electric field
        dDip : real (optional - init=0.0001)
            Maximal change of total dipole between two SCF steps needed to 
            converge SCF procedure
        eps : real (optional - init=1.0)
            Relative dielectric polarizability of medium where the dipoles and 
            molecule is present ( by default vacuum with relative permitivity 1.0)
        verbose : logical (optional init=False)
            If `True` more verbose output will be produced
        **kwargs : dict (optional)
            if kwargs={'index': index} then atoms specified in index won't be 
            polarized by static charge distribution. Otherwise all atoms with 
            static charge wont be polarized by static charge distribution
            
        '''
#-------------------------------------------------------------------------------
        if 1:
            R = np.tile(self.coor,(self.Nat,1,1))
            R = (np.swapaxes(R,0,1) - R)
            RR=squareform(pdist(self.coor))
                
                
            unit=np.diag([1]*self.Nat)
            RR=RR+unit   # only for avoiding ddivision by 0 for diagonal elements     
            RR3=np.power(RR,3)
            RR5=np.power(RR,5)
        
            mask=(np.abs(self.charge)>1e-8)         
            mask=np.expand_dims(mask, axis=0)
            MASK=np.dot(mask.T,mask)
            MASK=np.tile(MASK,(3,1,1))   # np.shape(mask)=(3,N,N) True all indexes where are both non-zero charges 
            MASK=np.rollaxis(MASK,0,3)
            
            
            MASK2=np.diag(np.ones(self.Nat,dtype='bool'))
            MASK2=np.tile(MASK2,(3,1,1))
            MASK2=np.rollaxis(MASK2,0,3)
            
            if Bonds is not None:
                MASK3=np.zeros((self.Nat,self.Nat,3),dtype='bool')
                for ii in range(len(Bonds)):
                    MASK3[Bonds[ii][0],Bonds[ii][1],:]=True
                    MASK3[Bonds[ii][1],Bonds[ii][0],:]=True
#                MASK3=np.tile(MASK3,(3,1,1))
#                MASK3=np.rollaxis(MASK3,0,3)
            
            Q=np.meshgrid(self.charge,self.charge)[0]   # in columns same charges
    
            dip=np.sum(self.dipole,axis=0)
            counter=0
    
            while True:
                # point charge electric field
                ELF=(Q/RR3)*np.rollaxis(R,2)
                ELF=np.rollaxis(ELF,0,3)   # ELF[i,j,:]  is electric field at position i generated by atom j - on diagonal there are zeros 
               
                # polarization by static charges only in area without charges:
                ELF[MASK]=0.0
                
                # dipole electric field
                P=np.tile(self.dipole[:,:],(self.Nat,1,1))    # P[ii,:,:]=self.dipole[:,:]  for ii going through all atoms
                PR=np.sum(np.multiply(P,R),axis=2)
                if Bonds is not None:
                    P[MASK3]=0.0
                    PR[MASK3[:,:,0]]=0.0
                for jj in range(3):             
                    ELF[:,:,jj]+=(3*PR/RR5)*R[:,:,jj]
                    ELF[:,:,jj]-=P[:,:,jj]/RR3
                ELF[MASK2]=0.0
                elf=np.sum(ELF,axis=1)/eps
                for ii in range(self.Nat):
                    self.dipole[ii,:]=np.dot(self.polar[ii],elf[ii]+Estatic)
                
                counter+=1
                if verbose:
                    print('Step: {:>4d} Dip sum: ({:8.4f};{:8.4f};{:8.4f}) Dip change: {:10.6f}'.format(counter,np.sum(self.dipole,axis=0)[0],np.sum(self.dipole,axis=0)[1],np.sum(self.dipole,axis=0)[2],np.linalg.norm(dip-np.sum(self.dipole,axis=0))))
    
                if np.linalg.norm(dip-np.sum(self.dipole,axis=0)) < dDip:
                    break 
                else:
                    dip=np.sum(self.dipole,axis=0) 
                    
                if counter>500:
                    print('SCF procedure exceeded maximum number of allowed steps (500)')
                    break
                
                if debug:
                    print('Dipole sum:',np.sum(self.dipole,axis=0))
        
#-------------------------------------------------------------------------------
        
# TODO: ADD POSSIBILITY FOR NO NEAREST NEIGHBOUR INTERACTION
        else:
            R=np.zeros((self.Nat,self.Nat,3),dtype='f8') # mutual distance vectors
            P=np.zeros((self.Nat,self.Nat,3),dtype='f8')
            for ii in range(self.Nat):
                for jj in range(ii+1,self.Nat):
                    R[ii,jj,:]=self.coor[ii]-self.coor[jj]
                    R[jj,ii,:]=-R[ii,jj,:]
            RR=np.sqrt(np.power(R[:,:,0],2)+np.power(R[:,:,1],2)+np.power(R[:,:,2],2))  # mutual distances
            unit=np.diag([1]*self.Nat)
            RR=RR+unit   # only for avoiding ddivision by 0 for diagonal elements     
            RR3=np.power(RR,3)
            RR5=np.power(RR,5)
            
            for key in kwargs.keys():
                if key=='index':
                    mask=kwargs['index'].copy()
            if not 'index' in kwargs.keys():
                mask=[]
                for ii in range(len(self.charge)):
                    if abs(self.charge[ii])>1e-8:
                        mask.append(ii)
            
            Q=np.meshgrid(self.charge,self.charge)[0]   # in columns are the same charges
            ELF=np.zeros((self.Nat,self.Nat,3),dtype='f8') # initialization of electric field tensor
    
            dip=np.sum(self.dipole,axis=0)
            counter=0
    
            while True:
                # point charge electric field        
                for jj in range(3):
                    ELF[:,:,jj]=(Q/RR3)*R[:,:,jj]   # ELF[i,j,:]  is electric field at position i generated by atom j - on diagonal there are zeros 
    
    # TODO: Change this procedure because atoms with a charges could be polarized by all atoms with charges - but imput defect charges should be fitted accordingly with polarizable atoms             
                # polarization by static charges only in area without charges:
                for ii in mask:
                    ELF[ii,mask,:]=0.0
                
                
                # dipole electric field
                for ii in range(self.Nat):
                    P[ii,:,:]=self.dipole[:,:]
                PR=np.sum(np.multiply(P,R),axis=2)
                
                # add electric field contribution from dipoles and induced dipoles           
                for jj in range(3):
                    ELF[:,:,jj]+=(3*PR/RR5)*R[:,:,jj]
                    ELF[:,:,jj]-=P[:,:,jj]/RR3
                
                # Atoms cannot be polarized by themselves => diagonal elements are zeros
                for ii in range(self.Nat):
                    ELF[ii,ii,:]=np.zeros(3,dtype='f8')
                
                # sum electric field contribution from all atoms into single vector for every atom
                elf=np.sum(ELF,axis=1)/eps
                
                #calculate induced dipoles
                for ii in range(self.Nat):
                    self.dipole[ii,:]=np.dot(self.polar[ii],elf[ii]+Estatic)
                
                counter+=1
                if verbose:
                    print('Step: {:>4d} Dip sum: ({:8.4f};{:8.4f};{:8.4f}) Dip change: {:10.6f}'.format(counter,np.sum(self.dipole,axis=0)[0],np.sum(self.dipole,axis=0)[1],np.sum(self.dipole,axis=0)[2],np.linalg.norm(dip-np.sum(self.dipole,axis=0))))
    
    # TODO: Change this criterion because if one dipole is bigger and one smaller by the same walue then sum change is zero
                if np.linalg.norm(dip-np.sum(self.dipole,axis=0)) < dDip:
                    break 
                else:
                    dip=np.sum(self.dipole,axis=0) 
                    
                if counter>500:
                    print('SCF procedure exceeded maximum number of allowed steps (500)')
                    break
                
        
    def get_interaction_energy(self,index,SelfInt=True,**kwargs):
        ''' Function calculates interaction energy between atoms defined in index
        and the rest of the atoms 

        Parameters
        ----------
        index : list of int (dimension N)
            List of atoms where we would like to calculate potential and 
            for which we would like to calculate interaction energy with the
            rest of the system
        SelfInt : logical (optional init=True)
            Specifies whether include interaction of induced dipole with static
            charges on same defect (defined by index). If `True` this selfinteraction
            is included, if `False` only interactions with atoms outside defect
            are included
        **kwargs : dict (optional) 
            if kwargs={'charge': charge} where charge is numpy.array of real and
            dimension=Natoms_of_defect with atomic trasition charges (TrEsp charges)
            for every atom of one defect defined by `index`
        
        Returns
        -------
        InterE : real
            Interaction energies in atomic units (Hartree)
        '''
        from ..General.Potential import potential_charge,potential_dipole
        
        
        if 'charge' in kwargs.keys():
            use_orig_charges=False
            charge=kwargs['charge'].copy()
        else:
            use_orig_charges=True
            charge=np.zeros(len(index),dtype='f8')
        
        # coppy charges and assign zero charges to those in index
        AllCharge=np.copy(self.charge)
        AllDipole=np.copy(self.dipole)
        for ii in range(self.Nat):
            if ii in index:
                if use_orig_charges:
                    charge[np.where(index==ii)[0][0]]=AllCharge[ii]
                AllCharge[ii]=0.0
                if not SelfInt:
                    AllDipole[ii,:]=np.zeros(3,dtype='f8')
        
        # calculate interaction energy
        InterE=0.0
        for jj in range(len(index)):
            potential=0.0
            for ii in range(self.Nat):
                if ii!=index[jj]:
                    R=self.coor[index[jj]]-self.coor[ii]
                    potential+=potential_charge(AllCharge[ii],R)
                    potential+=potential_dipole(AllDipole[ii],R)
            InterE+=potential*charge[jj]
        
        return InterE

def prepare_molecule_1Def(filenames,indx,PolarCF,PolarC,PolarF,verbose=False):
    ''' Read all informations needed for Dielectric class and transform system
    with two same defects into this class. Useful for calculation of interaction 
    energies, transition site energy shifts and dipole changes.
    
    Parameters
    ----------
    filenames :  dictionary 
        In the dictionary there are specified all needed files which contains 
        nessesary information for transformig the system into Dielectric class.
        keys:
        `'1def_structure'`: xyz file with system geometry and atom types
        `'charge_structure'`: xyz file with defect like molecule geometry for which transition charges were calculated
        `charge_grnd`: file with ground state charges for the defect
        `'charge_exct'`: file with excited state charges for the defect
        `'charge'`: file with transition charges for the defect
    indx : list of integers (dimension 9)
        There are specified indexes neded for asignment of defect 
        atoms. First three indexes correspond to center and two main axes of 
        reference structure (structure which was used for charges calculation)
        and the remaining six indexes are corresponding atoms of the defects 
        on fluorographene system (three correspond to first defect and the last
        three to the second one).
    PolarCF : numpy.array of real (dimension 3x3)
        Atomic polarizability Alpha(E) for carbon atoms which are connected to fluorine
        atom (environment atom) C-F in ATOMIC UNITS (Bohr^3)
    PolarC : numpy.array of real (dimension 3x3)
        Atomic polarizability Alpha(E) for defect carbon atoms (with sp2 hybridization)
        in ATOMIC UNITS (Bohr^3)
    PolarF : numpy.array of real (dimension 3x3)
        Atomic polarizability for fluorine atoms (part of fluorographene environment)
        in ATOMIC UNITS (Bohr^3)
        of fluorographene atoms into a 2D plane
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    **kwargs : dictionary (optional)
        Definition of polarizabitity matrixes for defect atoms (if nonzero
        polarizability is used)
    
    Returns
    -------
    mol_polar : Dielectric class 
        Fluorographene with two defects in Dielectric class which contains all 
        information needed for calculation of energy shifts, dipole changes and
        interaction energies for defect homodimer embeded in fluorographene
    index1 : list of integer (dimension Ndefect_atoms)
        Atom indexes of first defect atoms
    index2 : list of integer (dimension Ndefect_atoms)
        Atom indexes of second defect atoms
    charge : numpy.array of real (dimension Ndefect_atoms)
        Transition charges for every defect atom. First charge correspond to atom
        defined by first index in index1 (or in index2) list and so on.    
    '''
    
    
    indx_center_test=indx[0]
    indx_x_test=indx[1]
    indx_y_test=indx[2]
    
    indx_center1=indx[3]
    indx_x1=indx[4]
    indx_y1=indx[5]
    
    # Specify files:
    xyzfile2=filenames['charge_structure']
    filenameESP=filenames['charge']
    xyzfile=filenames['1def_structure']
    
    # Read Transition charges
    if verbose:
        print('     Reading charges and format to polarization format...')
    mol_test=Molecule('Perylene-charge')
    mol_test.load_xyz(xyzfile2)
    coor,charge,at_type=read_TrEsp_charges(filenameESP,verbose=False)
    
    # load molecule - fuorographene with single defect 
    if verbose:
        print('        Loading molecule...')
    mol=Molecule('FGrph-2Perylene')
    mol.load_xyz(xyzfile)
    
    index1=identify_molecule(mol.struc,mol_test.struc,indx_center1,indx_x1,indx_y1,indx_center_test,indx_x_test,indx_y_test,onlyC=True)
    if len(index1)!=len(np.unique(index1)):
        print('index1:')
        print(index1)
        raise IOError('There are repeating elements in index file')

    # Assign pol types
    if verbose:
        print('        Asigning polarizabilities to the molecule...')
    PolType=[]
    Polcharge=[]
    PolCoor=[]
    for ii in range(mol.struc.nat):
        if mol.struc.at_type[ii]=='C' and (ii in index1):
            Polcharge.append(charge[np.where(index1==ii)[0][0]])
            PolType.append('C')
            PolCoor.append(mol.struc.coor._value[ii])
        elif mol.struc.at_type[ii]=='C':
            PolType.append('CF')
            Polcharge.append(0.0)
            PolCoor.append(mol.struc.coor._value[ii])
        elif mol.struc.at_type[ii]=='F':
            PolType.append('F')
            Polcharge.append(0.0)
            PolCoor.append(mol.struc.coor._value[ii])
            
    PolType=np.array(PolType)
    Polcharge=np.array(Polcharge,dtype='f8')
    PolCoor=np.array(PolCoor,dtype='f8')
    
    # center the system - for testing nonisotropic polarizability
    if verbose:
        print('        Centering molecule...')
    PolCoor=CenterMolecule(PolCoor,indx_center1,indx_x1,indx_y1)
    
    # create dielectric representation of molecule
    mol_polar=Dielectric(PolCoor,Polcharge,np.zeros((len(PolCoor),3),dtype='f8'),
                         np.zeros((len(PolCoor),3,3),dtype='f8'))
    
    #ZeroM=np.zeros((3,3),dtype='f8')
    mol_polar.polar=mol_polar.assign_polar(PolType,**{'PolValues': {'CF': PolarCF,
                                                                    'C': PolarC,'F': PolarF}})
                  
    return mol_polar,index1,charge


def prepare_molecule_2Def(filenames,indx,PolarCF,PolarC,PolarF,nvec=np.array([0.0,0.0,1.0],dtype='f8'),verbose=False):
    ''' Read all informations needed for Dielectric class and transform system
    with two same defects into this class. Useful for calculation of interaction 
    energies, transition site energy shifts and dipole changes.
    
    Parameters
    ----------
    filenames :  dictionary 
        In the dictionary there are specified all needed files which contains 
        nessesary information for transformig the system into Dielectric class.
        keys:
        `'2def_structure'`: xyz file with system geometry and atom types
        `'charge_structure'`: xyz file with defect like molecule geometry for which transition charges were calculated
        `charge_grnd`: file with ground state charges for the defect
        `'charge_exct'`: file with excited state charges for the defect
        `'charge'`: file with transition charges for the defect
    indx : list of integers (dimension 9)
        There are specified indexes neded for asignment of defect 
        atoms. First three indexes correspond to center and two main axes of 
        reference structure (structure which was used for charges calculation)
        and the remaining six indexes are corresponding atoms of the defects 
        on fluorographene system (three correspond to first defect and the last
        three to the second one).
    PolarCF : numpy.array of real (dimension 3x3)
        Atomic polarizability Alpha(E) for carbon atoms which are connected to fluorine
        atom (environment atom) C-F in ATOMIC UNITS (Bohr^3)
    PolarC : numpy.array of real (dimension 3x3)
        Atomic polarizability Alpha(E) for defect carbon atoms (with sp2 hybridization)
        in ATOMIC UNITS (Bohr^3)
    PolarF : numpy.array of real (dimension 3x3)
        Atomic polarizability for fluorine atoms (part of fluorographene environment)
        in ATOMIC UNITS (Bohr^3)
    nvec : numpy.array (dimension 3) (optional - init=np.array([0.0,0.0,1.0],dtype='f8'))
        Normal vector to the fluorographene plane - needed for projection
        of fluorographene atoms into a 2D plane
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    **kwargs : dictionary (optional)
        Definition of polarizabitity matrixes for defect atoms (if nonzero
        polarizability is used)
    
    Returns
    -------
    mol_polar : Dielectric class 
        Fluorographene with two defects in Dielectric class which contains all 
        information needed for calculation of energy shifts, dipole changes and
        interaction energies for defect homodimer embeded in fluorographene
    index1 : list of integer (dimension Ndefect_atoms)
        Atom indexes of first defect atoms
    index2 : list of integer (dimension Ndefect_atoms)
        Atom indexes of second defect atoms
    charge : numpy.array of real (dimension Ndefect_atoms)
        Transition charges for every defect atom. First charge correspond to atom
        defined by first index in index1 (or in index2) list and so on.    
    '''
    
    
    indx_center_test=indx[0]
    indx_x_test=indx[1]
    indx_y_test=indx[2]
    
    indx_center1=indx[3]
    indx_x1=indx[4]
    indx_y1=indx[5]
    indx_center2=indx[6]
    indx_x2=indx[7]
    indx_y2=indx[8]
    
    # Specify files:
    xyzfile2=filenames['charge_structure']
    filenameESP=filenames['charge']
    xyzfile=filenames['2def_structure']
    
    # Read Transition charges
    if verbose:
        print('     Reading charges and format to polarization format...')
    mol_test=Molecule('Perylene-charge')
    mol_test.load_xyz(xyzfile2)
    coor,charge,at_type=read_TrEsp_charges(filenameESP,verbose=False)
    
    # load molecule - fuorographene with 2 defects 
    if verbose:
        print('        Loading molecule...')
    mol=Molecule('FGrph-2Perylene')
    mol.load_xyz(xyzfile)
    
    index1=identify_molecule(mol.struc,mol_test.struc,indx_center1,indx_x1,indx_y1,indx_center_test,indx_x_test,indx_y_test,onlyC=True)
    index2=identify_molecule(mol.struc,mol_test.struc,indx_center2,indx_x2,indx_y2,indx_center_test,indx_x_test,indx_y_test,onlyC=True)
    if len(index1)!=len(np.unique(index1)) or len(index2)!=len(np.unique(index2)):
        print('index1:')
        print(index1)
        print('index2:')
        print(index2)
        raise IOError('There are repeating elements in index file')

    # Assign pol types
    if verbose:
        print('        Asigning polarizabilities to the molecule...')
    PolType=[]
    Polcharge=[]
    PolCoor=[]
    for ii in range(mol.struc.nat):
        if mol.struc.at_type[ii]=='C' and (ii in index1):
            Polcharge.append(charge[np.where(index1==ii)[0][0]])
            PolType.append('C')
            PolCoor.append(mol.struc.coor._value[ii])
        elif mol.struc.at_type[ii]=='C' and (ii in index2):
            Polcharge.append(0.0)
            #Polcharge.append(charge[np.where(index2==ii)[0][0]])
            PolType.append('C')
            PolCoor.append(mol.struc.coor._value[ii])
        elif mol.struc.at_type[ii]=='C':
            PolType.append('CF')
            Polcharge.append(0.0)
            PolCoor.append(mol.struc.coor._value[ii])
        elif mol.struc.at_type[ii]=='F':
            PolType.append('F')
            Polcharge.append(0.0)
            PolCoor.append(mol.struc.coor._value[ii])
            
    PolType=np.array(PolType)
    Polcharge=np.array(Polcharge,dtype='f8')
    PolCoor=np.array(PolCoor,dtype='f8')
    
    # center the system - for testing nonisotropic polarizability
    if verbose:
        print('        Centering molecule...')
    PolCoor=CenterMolecule(PolCoor,indx_center1,indx_x1,indx_y1)
    
    # create dielectric representation of molecule
    mol_polar=Dielectric(PolCoor,Polcharge,np.zeros((len(PolCoor),3),dtype='f8'),
                         np.zeros((len(PolCoor),3,3),dtype='f8'))
    
    #ZeroM=np.zeros((3,3),dtype='f8')
    mol_polar.polar=mol_polar.assign_polar(PolType,**{'PolValues': {'CF': PolarCF,
                                                                    'C': PolarC,'F': PolarF}})
                                        
    return mol_polar,index1,index2,charge
    
def CalculateInterE(filenames,ShortName,index_all,Energy_QCH,Energy_all,nvec_all,PolarCF,PolarC,PolarF,precision=0.0001,SelfInt=True,NearestNeighbor=True,verbose=False,MathOut=True):
    ''' Calculate interaction energies between defects embeded in polarizable atom
    environment for all systems given in filenames.
    
    Parameters
    ----------
    filenames : list of dictionary (dimension Nsystems)
        In the dictionary there are specified all needed files which contains 
        nessesary information for transformig the system into Dielectric class.
        keys:
        `'2def_structure'`: xyz file with system geometry and atom types
        `'charge_structure'`: xyz file with defect like molecule geometry for which transition charges were calculated
        `charge_grnd`: file with ground state charges for the defect
        `'charge_exct'`: file with excited state charges for the defect
        `'charge'`: file with transition charges for the defect
    ShortName : list of strings
        List of short description (name) of individual systems 
    index_all : list of integers (dimension Nsystems x 6)
        There are specified indexes neded for asignment of defect 
        atoms. First three indexes correspond to center and two main axes of 
        reference structure (structure which was used for charges calculation)
        and the remaining three indexes are corresponding atoms of the defects 
        on fluorographene system.
    Energy_QCH : list of real (dimension Nsystems)
        List of quantum chemistry values of interaction energies in INVERSE
        CENTIMETERS between defects in polarizable atom environment 
        (used for printing comparison - not used for calculation at all)
    Energy_all : list of real (dimension Nsystems)
        In this variable there will be stored interaction energies in ATOMIC UNITS
        (Hartree) calculated by polarizable atoms method for description of the 
        environment.
    PolarCF : numpy.array of real (dimension 3x3)
        Atomic polarizability Alpha(E) for carbon atoms which are connected to fluorine
        atom (environment atom) C-F in ATOMIC UNITS (Bohr^3)
    PolarC : numpy.array of real (dimension 3x3)
        Atomic polarizability Alpha(E) for defect carbon atoms (with sp2 hybridization)
        in ATOMIC UNITS (Bohr^3)
    PolarF : numpy.array of real (dimension 3x3)
        Atomic polarizability for fluorine atoms (part of fluorographene environment)
        in ATOMIC UNITS (Bohr^3)
    precision : real (optional - init=0.0001)
            Maximal change of total dipole between two SCF steps needed to 
            converge SCF procedure for calculating of induced dipoles
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    
    Notes
    ----------
    Working only for systems with two symetric defects    
    
    '''    
    
    print('#       Name              QchIter  PolarInter  TrESP')           
    
    for ii in range(len(filenames)):
        if verbose:
            print('Calculation of interaction energy for:',ShortName[ii])
        
        # read and prepare molecule
        mol_polar,index1,index2,charge=prepare_molecule_2Def(filenames[ii],index_all[ii],PolarCF,PolarC,PolarF,nvec=nvec_all[ii],verbose=verbose)
        
        Bonds=GuessBonds(mol_polar.coor,bond_length=4.0)
        if MathOut:
            # output initial structure to Mathematica:
            mat_filename="".join(['Pictures/Polar_',ShortName[ii],'_init.nb'])
            OutputMathematica(mat_filename,mol_polar.coor,Bonds,['C']*mol_polar.Nat,scaleDipole=30.0,**{'TrPointCharge': mol_polar.charge,'AtDipole': mol_polar.dipole,'rSphere_dip': 0.5,'rCylinder_dip':0.1})

        # calculate TrEsp interaction energy
        E_TrEsp=mol_polar.get_interaction_energy(index2,SelfInt=SelfInt,**{'charge': charge})

        # calculate induced dipoles:
        if NearestNeighbor:
            mol_polar.calc_dipoles(dDip=precision,verbose=verbose)
        else:
            mol_polar.calc_dipoles(dDip=precision,verbose=verbose,Bonds=Bonds)
        #mol_polar.calc_dipoles(dDip=precision,verbose=True,**{'index': index1})
        
        # calculate interaction energy       
        Einter=mol_polar.get_interaction_energy(index2,SelfInt=SelfInt,**{'charge': charge})        
        
        if verbose:
            print('        Total interaction energy:',Einter*conversion_facs_energy["1/cm"],'Quantum interaction energy:',Energy_QCH[ii])

        print('{:<24} {:8.3f} {:8.3f}  {:8.3f}'.format(ShortName[ii],Energy_QCH[ii],abs(Einter*conversion_facs_energy["1/cm"]),abs(E_TrEsp*conversion_facs_energy["1/cm"])))       
        Energy_all[ii]=abs(Einter*conversion_facs_energy["1/cm"])
        
        if MathOut:
            # output dipoles to mathematica
            mat_filename="".join(['Pictures/Polar_',ShortName[ii],'.nb'])
            OutputMathematica(mat_filename,mol_polar.coor,Bonds,['C']*mol_polar.Nat,scaleDipole=30.0,**{'TrPointCharge': mol_polar.charge,'AtDipole': mol_polar.dipole,'rSphere_dip': 0.5,'rCylinder_dip':0.1})


def CalculateTrDipShift(filenames,ShortName,index_all,Dipole_QCH,Dip_all,Eshift_QCH,Eshift_all,PolarCF,PolarC,PolarF,precision=0.0001,SelfInt=True,NearestNeighbor=True,verbose=False,MathOut=False,**kwargs):
    ''' Calculate transition dipole for defect embeded in polarizable atom environment
    for all systems given in filenames.
    
    Parameters
    ----------
    filenames : list of dictionary (dimension Nsystems)
        In the dictionary there are specified all needed files which contains 
        nessesary information for transformig the system into Dielectric class.
        keys:
        `'2def_structure'`: xyz file with system geometry and atom types
        `'charge_structure'`: xyz file with defect like molecule geometry for which transition charges were calculated
        `charge_grnd`: file with ground state charges for the defect
        `'charge_exct'`: file with excited state charges for the defect
        `'charge'`: file with transition charges for the defect
    ShortName : list of strings
        List of short description (name) of individual systems 
    index_all : list of integers (dimension Nsystems x 6)
        There are specified indexes neded for asignment of defect 
        atoms. First three indexes correspond to center and two main axes of 
        reference structure (structure which was used for charges calculation)
        and the remaining three indexes are corresponding atoms of the defects 
        on fluorographene system.
    Dipole_QCH : list of real (dimension Nsystems)
        List of quantum chemistry values of transition dipoles in ATOMIC UNITS 
        (e*Bohr) for defect in polarizable atom environment 
        (used for printing comparison - not used for calculation at all)
    Dip_all : list of real (dimension Nsystems)
        In this variable there will be stored dipoles in ATOMIC UNITS (e*Bohr)
        calculated by polarizable atoms method for description of the environment.
    PolarCF : numpy.array of real (dimension 3x3)
        Atomic polarizability Alpha(E) for carbon atoms which are connected to fluorine
        atom (environment atom) C-F in ATOMIC UNITS (Bohr^3)
    PolarC : numpy.array of real (dimension 3x3)
        Atomic polarizability Alpha(E) for defect carbon atoms (with sp2 hybridization)
        in ATOMIC UNITS (Bohr^3)
    PolarF : numpy.array of real (dimension 3x3)
        Atomic polarizability for fluorine atoms (part of fluorographene environment)
        in ATOMIC UNITS (Bohr^3)
    precision : real (optional - init=0.0001)
            Maximal change of total dipole between two SCF steps needed to 
            converge SCF procedure for calculating of induced dipoles
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    
    Notes
    ----------
    Working only for fluorographene system with single defect  
     
    '''
    
    print('#       Name              QchDip  PolarDip  TrESP')
    
    for ii in range(len(filenames)):
        if verbose:
            print('Calculation of dipoles for:',ShortName[ii])
        
        # read and prepare molecule
        mol_polar,index1,charge=prepare_molecule_1Def(filenames[ii],index_all[ii],PolarCF,PolarC,PolarF,verbose=False)

        Bonds=GuessBonds(mol_polar.coor,bond_length=4.0)

        # calculate TrEsp dipole
        TrESP_dipole=np.zeros(3,dtype='f8')
        TrESP_dipole=np.dot(mol_polar.charge,mol_polar.coor)

        # calculate induced dipoles:
        if NearestNeighbor:
            mol_polar.calc_dipoles(dDip=precision,verbose=verbose)
        else:
            mol_polar.calc_dipoles(dDip=precision,verbose=verbose,Bonds=Bonds)

        # calculate transition dipole        
        TrDip=np.zeros(3,dtype='f8')
        TrDip=np.dot(mol_polar.charge,mol_polar.coor)
        TrDip+=np.sum(mol_polar.dipole,axis=0)
        
        # Calculate transition energy shift
        E_shift=mol_polar.get_interaction_energy(index1,SelfInt=SelfInt,**{'charge': charge})/2
        
        Eshift_all[ii]=E_shift*conversion_facs_energy["1/cm"]
        
        if verbose:
            print('        Total transition dipole:',np.sqrt(np.dot(TrDip,TrDip)),'Quantum chemistry dipole:',Dipole_QCH[ii])
        print(ShortName[ii],Dipole_QCH[ii], np.linalg.norm(TrDip), np.linalg.norm(TrESP_dipole), Eshift_QCH[ii], E_shift*conversion_facs_energy["1/cm"]) 
        Dip_all[ii,:]=TrDip[:]
        
        if MathOut:
            # output dipoles to mathematica
            mat_filename="".join(['Pictures/Polar_',ShortName[ii],'.nb'])
            OutputMathematica(mat_filename,mol_polar.coor,Bonds,['C']*mol_polar.Nat,scaleDipole=30.0,**{'TrPointCharge': mol_polar.charge,'AtDipole': mol_polar.dipole,'rSphere_dip': 0.5,'rCylinder_dip':0.1})
        
    