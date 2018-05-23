# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:33:56 2017

@author: Vladislav Sláma
"""
import numpy as np
from copy import deepcopy
from scipy.spatial.distance import pdist,squareform
import os

from ..QuantumChem.Classes.structure import Structure
from ..QuantumChem.calc import identify_molecule
from ..QuantumChem.read_mine import read_TrEsp_charges
from ..QuantumChem.interaction import charge_charge
from ..QuantumChem.positioningTools import project_on_plane, CenterMolecule, fit_plane
from ..General.units import conversion_facs_energy, conversion_facs_mass
from .electrostatic import PrepareMolecule_1Def as ElStat_PrepareMolecule_1Def
from .electrostatic import PrepareMolecule_2Def as ElStat_PrepareMolecule_2Def
from ..General.Potential import potential_charge, potential_dipole
from ..QuantumChem.Classes.general import Energy as EnergyClass
from ..General.UnitsManager import energy_units
from ..QuantumChem.calc import GuessBonds

from ..QuantumChem.output import OutputMathematica

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
    
    def __init__(self,coor,pol_type,charge,dipole,AlphaE,Alpha_E,Alpha_st,BetaEE,V,CoarseGrain=None):
        self.coor=np.copy(coor)
        self.polar={}
        self.polar['AlphaE']=AlphaE
        self.polar['Alpha_E']=Alpha_E
        self.polar['BetaEE']=BetaEE
        self.polar['Alpha_st']=Alpha_st
        self.VinterFG=V
        self.charge=np.copy(charge)
        self.dipole=np.copy(dipole)
        self.at_type=pol_type
        self.coarse_grain = CoarseGrain
        self.Nat=len(coor)
    
    def assign_polar(self,**kwargs):
        
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
        PolValues={'CF': [ZeroM,ZeroM,ZeroM,ZeroM],
                   'CD': [ZeroM,ZeroM,ZeroM,ZeroM],'C': [ZeroM,ZeroM,ZeroM,ZeroM]} 
        for key in list(kwargs.keys()):
            if key=='PolValues':
                PolValues=kwargs['PolValues']
        
        #print(PolValues)
        pol_type = self.at_type
        
        if self.Nat!=len(pol_type):
            raise IOError('Polarization type vector must have the same length as number of atoms')
        
        polar={}
        polar['AlphaE']=np.zeros((self.Nat,3,3),dtype='f8')
        polar['Alpha_E']=np.zeros((self.Nat,3,3),dtype='f8')
        polar['BetaEE']=np.zeros((self.Nat,3,3),dtype='f8')
        polar['Alpha_st']=np.zeros((self.Nat,3,3),dtype='f8')
        for ii in range(len(pol_type)):
            polar['AlphaE'][ii,:,:]=PolValues[pol_type[ii]][0]
            polar['Alpha_E'][ii,:,:]=PolValues[pol_type[ii]][1]
            polar['BetaEE'][ii,:,:]=PolValues[pol_type[ii]][2]
            polar['Alpha_st'][ii,:,:]=PolValues[pol_type[ii]][3]
        return polar

    def get_distance_matrixes(self):
        
        # calculation of tensors with interatomic distances
        R=np.zeros((self.Nat,self.Nat,3),dtype='f8') # mutual distance vectors
        for ii in range(self.Nat):
            for jj in range(ii+1,self.Nat):
                R[ii,jj,:]=self.coor[ii]-self.coor[jj]
                R[jj,ii,:]=-R[ii,jj,:]
        RR=np.sqrt(np.power(R[:,:,0],2)+np.power(R[:,:,1],2)+np.power(R[:,:,2],2))  # mutual distances
        
        return R,RR
    
    def get_T_tensor(self,R=None,RR=None,RR3=None,RR5=None):
        if R is None:
            R,RR = self.get_distance_matrixes(self)
            RR=RR+np.identity(self.Nat) # only for avoiding ddivision by 0 for diagonal elements
            RR3=np.power(RR,3)
            RR5=np.power(RR,5)
        
        T=np.zeros((self.Nat,self.Nat,3,3),dtype='f8') # mutual distance vectors
        for ii in range(3):
            T[:,:,ii,ii]=1/RR3[:,:]-3*np.power(R[:,:,ii],2)/RR5
            for jj in range(ii+1,3):
                T[:,:,ii,jj] = -3*R[:,:,ii]*R[:,:,jj]/RR5
                T[:,:,jj,ii] = T[:,:,ii,jj]
        for ii in range(self.Nat):
            T[ii,ii,:,:]=0.0        # no self interaction of atom i with atom i
        
        return T
    
    def get_S_tensor(self,R=None,RR=None,RR5=None):
        if R is None:
            R,RR = self.get_distance_matrixes(self)
            RR=RR+np.identity(self.Nat) # only for avoiding ddivision by 0 for diagonal elements
            RR5=np.power(RR,5)
            
        RR7=np.power(RR,7)
        
        # definition of S tensor
        S=np.zeros((self.Nat,self.Nat,3,3,3),dtype='f8') # mutual distance vectors
        for ii in range(3):
            for jj in range(3):
                for kk in range(3):
                    S[:,:,ii,jj,kk]=-5*R[:,:,ii]*R[:,:,jj]*R[:,:,kk]/RR7
        for ii in range(3):
            for jj in range(3):
                S[:,:,ii,ii,jj]+=R[:,:,jj]/RR5
                S[:,:,ii,jj,ii]+=R[:,:,jj]/RR5
                S[:,:,jj,ii,ii]+=R[:,:,jj]/RR5
        for ii in range(self.Nat):
            S[ii,ii,:,:,:]=0.0        # no self interaction of atom i with atom i
        
        return S
        
    
    def _test_2nd_order(self,typ,Estatic=np.zeros(3,dtype='f8'),eps=1):
        ''' Function for testing of calculation with induced dipoles. Calculate
        induced dipoles in second order (by induced dipoles). Combined with 
        calc_dipoles_All(typ,NN=1) we should obtain the same dipoles as with
        calc_dipoles_All(typ,NN=2)
        
        Parameters
        ----------
        typ : str ('AlphaE','Alpha_E','BetaEE')
            Specifies which polarizability is used for calculation of induced
            atomic dipoles
        Estatic : numpy.array of real (dimension 3) (optional - init=np.zeros(3,dtype='f8'))
            External homogeneous electric fiel vectord (orientation and strength)
            in ATOMIC UNITS. By default there is no electric field
        eps : real (optional - init=1.0)
            Relative dielectric polarizability of medium where the dipoles and 
            molecule is present ( by default vacuum with relative permitivity 1.0)

        Notes
        ----------
        **OK. Definition of Tensor T is right** 
        '''
        
        debug=False
        
        R=np.zeros((self.Nat,self.Nat,3),dtype='f8') # mutual distance vectors
        P=np.zeros((self.Nat,3),dtype='f8')
        for ii in range(self.Nat):
            for jj in range(ii+1,self.Nat):
                R[ii,jj,:]=self.coor[ii]-self.coor[jj]
                R[jj,ii,:]=-R[ii,jj,:]
        RR=np.sqrt(np.power(R[:,:,0],2)+np.power(R[:,:,1],2)+np.power(R[:,:,2],2))  # mutual distances
        unit=np.diag([1]*self.Nat)
        RR=RR+unit   # only for avoiding ddivision by 0 for diagonal elements     
        RR3=np.power(RR,3)
        RR5=np.power(RR,5)
        
        # definition of T tensor
        T=np.zeros((self.Nat,self.Nat,3,3),dtype='f8') # mutual distance vectors
        for ii in range(3):
            T[:,:,ii,ii]=1/RR3[:,:]-3*np.power(R[:,:,ii],2)/RR5
            for jj in range(ii+1,3):
                T[:,:,ii,jj] = -3*R[:,:,ii]*R[:,:,jj]/RR5
                T[:,:,jj,ii] = T[:,:,ii,jj]
        for ii in range(self.Nat):
            T[ii,ii,:,:]=0.0        # no self interaction of atom i with atom i
        
        # calculating induced dipoles in second order
        Q=np.meshgrid(self.charge,self.charge)[0]   # in columns same charges
        ELF=np.zeros((self.Nat,self.Nat,3),dtype='f8')
        
        for jj in range(3):
            ELF[:,:,jj]=(Q/RR3)*R[:,:,jj]   # ELF[i,j,:]  is electric field at position i generated by atom j 
        for ii in range(self.Nat):
            ELF[ii,ii,:]=np.zeros(3,dtype='f8')
            
        ELFV=np.array(np.sum(ELF,axis=1),dtype='f8')             # ELFV[i,:]   is electric field at position of atom i
        for ii in range(self.Nat):
            P[ii,:]=np.dot(self.polar[typ][ii],ELFV[ii,:])
        
        if debug and typ=='AlphaE':
            from ..General.Potential import ElField_dipole
            # Test first order induced dipoles            
            self.dipole=np.zeros((self.Nat,3),dtype='f8')
            self._calc_dipoles_All('AlphaE',NN=1)
            if np.allclose(P,self.dipole):
                print('First order dipoles are the same.')
            else:
                print('Problem with first order induced dipoles.')
            
            # test induced electric field
            Elfield=np.zeros(3,dtype='f8')
            for ii in range(3):
                Elfield[ii]=np.dot(-T[0,1,ii,:],P[1,:])
            print('Electric field at atom 0 induced by dipole at position 1 wT:',Elfield)
            Elfield=np.zeros(3,dtype='f8')
            Elfield=ElField_dipole(P[1,:],R[0,1,:])
            print('Electric field at atom 0 induced by dipole at position 1 woT:',Elfield)
        
        ELFV=np.zeros((self.Nat,3),dtype='f8')
        for ii in range(3):
            for jj in range(3):
                ELFV[:,ii]+=np.dot(T[:,:,ii,jj],P[:,jj])
        
        for ii in range(self.Nat):
            P[ii,:]=np.dot(self.polar[typ][ii],ELFV[ii,:])
        
        # -P should be 2nd order induced dipoles 
        self.dipole+=(-P)
        if debug:
            print('Dipole sum:',np.sum(self.dipole,axis=0))
    
# TODO: Add possibility for NN = -err to calculate dipoles until convergence is reached
    def _calc_dipoles_All(self,typ,Estatic=np.zeros(3,dtype='f8'),NN=60,eps=1,debug=False):
        ''' Function for calculation induced dipoles of SCF procedure for interaction
        of molecule with environment. It calculates induced dipoles on individual
        atoms by static charge distribution and homogeneous electric field.
        
        Parameters
        ----------
        typ : str ('AlphaE','Alpha_E','BetaEE')
            Specifies which polarizability is used for calculation of induced
            atomic dipoles
        Estatic : numpy.array of real (dimension 3) (optional - init=np.zeros(3,dtype='f8'))
            External homogeneous electric fiel vectord (orientation and strength)
            in ATOMIC UNITS. By default there is no electric field
        NN : integer (optional - init=60)
            Number of SCF steps for calculation of induced dipole
        eps : real (optional - init=1.0)
            Relative dielectric polarizability of medium where the dipoles and 
            molecule is present ( by default vacuum with relative permitivity 1.0)
            
        '''
        
        if debug:
            import timeit
            time0 = timeit.default_timer()
        #R=np.zeros((self.Nat,self.Nat,3),dtype='f8') # mutual distance vectors
        #P=np.zeros((self.Nat,self.Nat,3),dtype='f8')
        #for ii in range(self.Nat):
        #    for jj in range(ii+1,self.Nat):
        #        R[ii,jj,:]=self.coor[ii]-self.coor[jj]
        #        R[jj,ii,:]=-R[ii,jj,:]
        #if debug:
        #    time01 = timeit.default_timer()
        #RR=np.sqrt(np.power(R[:,:,0],2)+np.power(R[:,:,1],2)+np.power(R[:,:,2],2))  # mutual distances
        R = np.tile(self.coor,(self.Nat,1,1))
        R = (np.swapaxes(R,0,1) - R)
        RR=squareform(pdist(self.coor))
        
        if 0:
            RR=np.sqrt(np.power(R[:,:,0],2)+np.power(R[:,:,1],2)+np.power(R[:,:,2],2))
            RR2=squareform(pdist(self.coor))
            print((RR2==RR).all())          # False
            print(np.allclose(RR2,RR))      # True
            if not (RR2==RR).all():
                print(RR[0,1])
                print(pdist(self.coor)[0])
                print(RR[0,2])
                print(pdist(self.coor)[1])
            
        
        if debug:
            time01 = timeit.default_timer()
            
        unit=np.diag([1]*self.Nat)
        RR=RR+unit   # only for avoiding ddivision by 0 for diagonal elements     
        RR3=np.power(RR,3)
        RR5=np.power(RR,5)
        
        
        #mask=[]
        #for ii in range(len(self.charge)):
        #    if abs(self.charge[ii])>1e-8:
        #        mask.append(ii)
    
        mask=(np.abs(self.charge)>1e-8)         
        mask=np.expand_dims(mask, axis=0)
        MASK=np.dot(mask.T,mask)
        MASK=np.tile(MASK,(3,1,1))   # np.shape(mask)=(3,N,N) True all indexes where are both non-zero charges 
        MASK=np.rollaxis(MASK,0,3)
        
        MASK2=np.diag(np.ones(self.Nat,dtype='bool'))
        MASK2=np.tile(MASK2,(3,1,1))
        MASK2=np.rollaxis(MASK2,0,3)
        
        Q=np.meshgrid(self.charge,self.charge)[0]   # in columns same charges
        #ELF=np.zeros((self.Nat,self.Nat,3),dtype='f8')
        #ELF_Q=(Q/RR3)*np.rollaxis(R,2)
        #ELF_Q=np.rollaxis(ELF,0,3)
        if debug:
            time1 = timeit.default_timer()
            print('Time spend on preparation of variables in calc_dipoles_All:',time1-time0,'s')


        for kk in range(NN):
            # point charge electric field
            ELF=(Q/RR3)*np.rollaxis(R,2)
            ELF=np.rollaxis(ELF,0,3)
            #for jj in range(3):
            #    ELF[:,:,jj]=(Q/RR3)*R[:,:,jj]   # ELF[i,j,:]  is electric field at position i generated by atom j - on diagonal there are zeros 

# TODO: Change this procedure because atoms with a charges could be polarized by all atoms with charges - but imput defect charges should be fitted accordingly with polarizable atoms             
            # polarization by static charges only in area without charges:
            #for ii in mask:
            #    ELF[ii,mask,:]=0.0
            ELF[MASK]=0.0
            
            # dipole electric field
            #for ii in range(self.Nat):
            #    P[ii,:,:]=self.dipole[:,:]
            P=np.tile(self.dipole[:,:],(self.Nat,1,1))    # P[ii,:,:]=self.dipole[:,:]  for ii going through all atoms
            PR=np.sum(np.multiply(P,R),axis=2)
# TODO: This takes One second - make it faster
            for jj in range(3):             
                ELF[:,:,jj]+=(3*PR/RR5)*R[:,:,jj]
                ELF[:,:,jj]-=P[:,:,jj]/RR3
            #for ii in range(self.Nat):
            #    ELF[ii,ii,:]=np.zeros(3,dtype='f8')
            ELF[MASK2]=0.0
            elf=np.sum(ELF,axis=1)/eps
# TODO: Think if this could be done in some efficient way
            for ii in range(self.Nat):
                self.dipole[ii,:]=np.dot(self.polar[typ][ii],elf[ii]+Estatic)
            if debug:
                print('Dipole sum:',np.sum(self.dipole,axis=0))
        if debug:
            time2 = timeit.default_timer()
            print('Time spend on calculation in calc_dipoles_All:',time2-time1,'s')
            print('Calculation vs preparation ratio:',(time2-time1)/(time1-time0))
            print('Time for filling coordinate matrix vs all the rest:',(time01-time0)/(time1-time01))
                
            
    def _get_interaction_energy(self,index,charge=None,debug=False):
        ''' Function calculates interaction energy between atoms defined in index
        and the rest of the atoms 

        Parameters
        ----------
        index : list of int (dimension N)
            List of atoms where we would like to calculate potential and 
            for which we would like to calculate interaction energy with the
            rest of the system
        charge : numpy.array of real (dimension Natoms_of_defect)
            Atomic trasition charges (TrEsp charges) for every atom of one defect
            defined by `index`
        
        Returns
        -------
        InterE : real
            Interaction energies in atomic units (Hartree)
        '''
        
        if isinstance(charge,np.ndarray) or isinstance(charge,list):
            use_orig_charges=False
        else:
            if charge==None:
                use_orig_charges=True
            else:
                raise IOError('Unable to determine charges')
            
        if use_orig_charges:
            charge=np.zeros(len(index),dtype='f8')
        
        # coppy charges and assign zero charges to those in index
        AllCharge=np.copy(self.charge)
        AllDipole=np.copy(self.dipole)
        for ii in range(self.Nat):
            if ii in index:
                if use_orig_charges:
                    charge[np.where(index==ii)[0][0]]=AllCharge[ii]
                AllCharge[ii]=0.0
                AllDipole[ii,:]=np.zeros(3,dtype='f8')
        
        InterE=0.0

# TODO: This distance matrix R is calculated many times - it would be faster to have it as global variable
# TODO: Check if this filling of whole matrix and then taking only small slice is not slower than two for cycles only through relevant pairs
        # Fill matrix of interatomic vectors:
        R = np.tile(self.coor,(self.Nat,1,1))
        R = (R - np.swapaxes(R,0,1))            # R[ii,jj,:]=self.coor[jj]-self.coor[ii]
        
        # Correct regions with zero distance
        if (AllCharge[index]==0.0).all():
            R[index,index,0]=1.0  # it is small distance but it will be always multiplied by zero and therefore it wont influent total potential
        else:
            R[index,index,0]=1e20 # large distance to have a small norm in order not ti influent the total potential (these atoms should be excluded) 
        
        # Take only slice of the matrix R[:,jj,:] where jj corespond to indexes 
        R=R[:,index,:]
        pot_charge=potential_charge(AllCharge,R)
        pot_dipole=potential_dipole(AllDipole,R)

# TODO: Move to test part
        if debug:
            print('Length of index list:',len(index))
            print('Shape of coor matrix:',R.shape)
            #print('Coor 0,0:',R[0,0])
            #print('Coor 0,1:',R[0,1])
            #print('Coor 0,2:',R[0,2])
            #print('Coor 2,3:',R[2,3])
            potential_charge_test=np.zeros(len(index),dtype='f8')
            potential_dipole_test=np.zeros(len(index),dtype='f8')
            #print(pot_charge)
            for jj in range(len(index)):
                for ii in range(self.Nat):
                    if ii!=index[jj]:
                        R=self.coor[index[jj]]-self.coor[ii]
                        #if jj==0 and ii==0:
                        #    print('Coor 0,0:',R)
                        #if jj==1 and ii==0:
                        #    print('Coor 0,1:',R)
                        #if jj==2 and ii==0:
                        #    print('Coor 0,2:',R)
                        #if jj==3 and ii==2:
                        #    print('Coor 2,3:',R)
                        potential_charge_test[jj]+=potential_charge(AllCharge[ii],R)
                        potential_dipole_test[jj]+=potential_dipole(AllDipole[ii],R)
            #print(potential_test)
            print(pot_dipole)
            print(potential_dipole_test)
            if np.allclose(potential_charge_test,pot_charge):
                print('Potential generated by charges is the same for old and new calculation')
            else:
                raise Warning('Potentials generated by charges are different for both methods')
            if np.allclose(potential_dipole_test,pot_dipole):
                print('Potential generated by dipoles is the same for old and new calculation')
            else:
                raise Warning('Potentials generated by dipoles are different for both methods')
        
            for jj in range(len(index)):
                potential=0.0
                for ii in range(self.Nat):
                    if ii!=index[jj]:
                        R=self.coor[index[jj]]-self.coor[ii]
                        potential+=potential_charge(AllCharge[ii],R)
                        potential+=potential_dipole(AllDipole[ii],R)
                InterE+=potential*charge[jj]
            
            if np.allclose(InterE,np.dot(charge,pot_charge+pot_dipole)):
                print('Interaction energy is calculated correctly')
            else:
                raise Warning('Interaction energy for both methods is different')
                
                
        InterE = np.dot(charge, pot_charge+pot_dipole)
        return InterE
    
    
    def _fill_Polar_matrix(self,index1,index2,typ='AlphaE',order=80,debug=False):
        """ Calculate polarization matrix representation for interaction energy
        calculation.
        
        Parameters
        ---------
        index1 : list of integer (dimension Natoms_defect1)
            Indexes of all atoms from the first defect (starting from 0)
        index2 : list of integer (dimension Natoms_defect2)
            Indexes of all atoms from the second defect (starting from 0)
        typ : string (optional init = 'AlphaE')
            Which polarizability should be used for calculation of induced 
            dipoles. Supported types are: ``'AlphaE'``, ``'Alpha_E'`` and
            ``'BetaEE'``
        order : integer (optional - init=80)
            Specify how many SCF steps shoudl be used in calculation  of induced
            dipoles - according to the used model it should be 2
        
        Returns
        -------
        PolMAT : numpy array of float (dimension 2x2)
            Polarizability matrix representation. For ``typ='AlphaE'`` or 
            ``typ='BetaEE': PolMAT[0,0] = -E(1)*induced_dipole(1),
            PolMAT[0,1] = PolMAT[1,0] = -E(1)*induced_dipole(2) and
            PolMAT[1,1] = -E(2)*induced_dipole(2). For ``typ='Alpha_E'`` 
            diagonal elements are swapped: PolMAT[0,0] = -E(2)*induced_dipole(2),
            PolMAT[0,1] = PolMAT[1,0] = -E(1)*induced_dipole(2) and
            PolMAT[1,1] = -E(1)*induced_dipole(1)
        dipolesA : numpy array of float (dimension 3)
            Total induced dipole moment in the environment by the first defect.
        dipolesB : numpy array of float (dimension 3)
            Total induced dipole moment in the environment by the second defect.
        dipoles_polA : numpy array of float (dimension Natoms x 3)
            Induced atomic dipole moments for all atoms in the environment by 
            the first defect
        
        """
        
        if typ=='BetaEE' and order>1:
            raise IOError('For calculation with beta polarization maximal order is 1')
        elif typ=='BetaEE' and order<1:
            return np.zeros((2,2),dtype='f8')
        
        defA_charge=self.charge[index1]
        defB_charge=self.charge[index2]
        defA_indx=deepcopy(index1)
        defB_indx=deepcopy(index2)
        
        PolMAT=np.zeros((2,2),dtype='f8')
        E_TrEsp=self.get_TrEsp_Eng(index1, index2)
        
        if debug:
            print(typ,order)
        
        # Polarization by molecule B
        self.charge[defA_indx]=0.0
        self._calc_dipoles_All(typ,NN=order,eps=1,debug=False)
        dipolesB=np.sum(self.dipole,axis=0)   # induced dipoles by second defect (defect B) 
        self.charge[defA_indx]=defA_charge
        PolMAT[1,1] = self._get_interaction_energy(defB_indx,charge=defB_charge,debug=False) - E_TrEsp
        PolMAT[0,1] = self._get_interaction_energy(defA_indx,charge=defA_charge,debug=False) - E_TrEsp
        PolMAT[1,0] = PolMAT[0,1]
        dipoles_polB = self.dipole.copy()
        self.dipole=np.zeros((self.Nat,3),dtype='f8')
        
        # Polarization by molecule A
        self.charge[defB_indx]=0.0
        self._calc_dipoles_All(typ,NN=order,eps=1,debug=False)
        dipolesA=np.sum(self.dipole,axis=0)
        self.charge[defB_indx]=defB_charge
        PolMAT[0,0] = self._get_interaction_energy(defA_indx,charge=defA_charge,debug=False) - E_TrEsp
        if debug:
            print(PolMAT*conversion_facs_energy["1/cm"])
            if np.isclose(self._get_interaction_energy(defB_indx,charge=defB_charge,debug=False)-E_TrEsp,PolMAT[1,0]):
                print('ApB = BpA')
            else:
                raise Warning('ApB != BpA')
        dipoles_polA = self.dipole.copy()
        self.dipole=np.zeros((self.Nat,3),dtype='f8')
        
        if typ=='AlphaE' or typ=='BetaEE' or typ=='Alpha_st':
            return PolMAT,dipolesA,dipolesB,dipoles_polA,dipoles_polB
        elif typ=='Alpha_E':
            PolMAT[[0,1],[0,1]] = PolMAT[[1,0],[1,0]]   # Swap AlphaMAT[0,0] with AlphaMAT[1,1]
            return PolMAT,dipolesA,dipolesB,dipoles_polA,dipoles_polB
    
    
    def _TEST_fill_Polar_matrix(self,index1,index2,typ='AlphaE',order=80,debug=False, out_pot=False):
        """ Calculate polarization matrix representation for interaction energy
        calculation.
        
        Parameters
        ---------
        index1 : list of integer (dimension Natoms_defect1)
            Indexes of all atoms from the first defect (starting from 0)
        index2 : list of integer (dimension Natoms_defect2)
            Indexes of all atoms from the second defect (starting from 0)
        typ : string (optional init = 'AlphaE')
            Which polarizability should be used for calculation of induced 
            dipoles. Supported types are: ``'AlphaE'``, ``'Alpha_E'`` and
            ``'BetaEE'``
        order : integer (optional - init=80)
            Specify how many SCF steps shoudl be used in calculation  of induced
            dipoles - according to the used model it should be 2
        
        Returns
        -------
        PolMAT : numpy array of float (dimension 2x2)
            Polarizability matrix representation. For ``typ='AlphaE'`` or 
            ``typ='BetaEE': PolMAT[0,0] = -E(1)*induced_dipole(1),
            PolMAT[0,1] = PolMAT[1,0] = -E(1)*induced_dipole(2) and
            PolMAT[1,1] = -E(2)*induced_dipole(2). For ``typ='Alpha_E'`` 
            diagonal elements are swapped: PolMAT[0,0] = -E(2)*induced_dipole(2),
            PolMAT[0,1] = PolMAT[1,0] = -E(1)*induced_dipole(2) and
            PolMAT[1,1] = -E(1)*induced_dipole(1)
        dipolesA : numpy array of float (dimension 3)
            Total induced dipole moment in the environment by the first defect.
        dipolesB : numpy array of float (dimension 3)
            Total induced dipole moment in the environment by the second defect.
        dipoles_polA : numpy array of float (dimension Natoms x 3)
            Induced atomic dipole moments for all atoms in the environment by 
            the first defect
        
        """
        
        if typ=='BetaEE' and order>1:
            raise IOError('For calculation with beta polarization maximal order is 1')
        elif typ=='BetaEE' and order<1:
            return np.zeros((2,2),dtype='f8')
        
        defA_charge=self.charge[index1]
        defB_charge=self.charge[index2]
        defA_indx=deepcopy(index1)
        defB_indx=deepcopy(index2)
        
        PolMAT=np.zeros((2,2),dtype='f8')
        E_TrEsp=self.get_TrEsp_Eng(index1, index2)
        
        if debug:
            print(typ,order)
        
        # Polarization by molecule B
        self.charge[defA_indx]=0.0
        self._calc_dipoles_All(typ,NN=order,eps=1,debug=False)
        dipolesB=np.sum(self.dipole,axis=0)   # induced dipoles by second defect (defect B) 
        self.charge[defA_indx]=defA_charge
        PolMAT[1,1] = self._get_interaction_energy(defB_indx,charge=defB_charge,debug=False) - E_TrEsp
        PolMAT[0,1] = self._get_interaction_energy(defA_indx,charge=defA_charge,debug=False) - E_TrEsp
        PolMAT[1,0] = PolMAT[0,1]
        self.dipole=np.zeros((self.Nat,3),dtype='f8')
        
        # Polarization by molecule A
        self.charge[defB_indx]=0.0
        self._calc_dipoles_All(typ,NN=order,eps=1,debug=False)
        dipolesA=np.sum(self.dipole,axis=0)
        self.charge[defB_indx]=defB_charge
        PolMAT[0,0] = self._get_interaction_energy(defA_indx,charge=defA_charge,debug=False) - E_TrEsp
        if debug:
            print(PolMAT*conversion_facs_energy["1/cm"])
            if np.isclose(self._get_interaction_energy(defB_indx,charge=defB_charge,debug=False)-E_TrEsp,PolMAT[1,0]):
                print('ApB = BpA')
            else:
                raise Warning('ApB != BpA')
        dipoles_polA = self.dipole.copy()
        self.dipole=np.zeros((self.Nat,3),dtype='f8')
        
        if typ=='AlphaE' or typ=='BetaEE' or typ=='Alpha_st':
            return PolMAT,dipolesA,dipolesB,dipoles_polA
        elif typ=='Alpha_E':
            PolMAT[[0,1],[0,1]] = PolMAT[[1,0],[1,0]]   # Swap AlphaMAT[0,0] with AlphaMAT[1,1]
            return PolMAT,dipolesA,dipolesB,dipoles_polA
    
    def get_TrEsp_Eng(self, index1, index2):
        """ Calculate TrEsp interaction energy for defects (defect-like 
        molecules) in vacuum.
        
        Parameters
        --------
        index1 : list of integer (dimension Natoms_defect1)
            Indexes of all atoms from the first defect (starting from 0)
        index2 : list of integer (dimension Natoms_defect2)
            Indexes of all atoms from the second defect (starting from 0)
            
        Returns
        --------
        E_TrEsp : float
            TrEsp interaction energy in ATOMIC UNITS (Hartree) between defect 
            in vacuum.
        
        """
        
        defA_coor = self.coor[index1]
        defB_coor = self.coor[index2]
        defA_charge = self.charge[index1] 
        defB_charge = self.charge[index2]
        E_TrEsp = charge_charge(defA_coor,defA_charge,defB_coor,defB_charge)[0]
        
        return E_TrEsp # in hartree
    
    def get_TrEsp_Dipole(self, index):
        """ Calculate vacuum transition dipole moment for single defect (from
        TrEsp charges).
        
        Parameters
        ----------
        index : list of integer (dimension Natoms_defect)
            Indexes of all atoms from the defect (starting from 0) of which 
            transition dipole is calculated
            
        Returns
        --------
        Dip_TrEsp : numpy array of float (dimension 3)
            Transition dipole in ATOMIC UNITS for specified defect (by index) 
            calculated from TrEsp charges
            
        """
        
        def_coor = self.coor[index]
        def_charge = self.charge[index] 
        
        Dip_TrEsp = np.dot(def_charge,def_coor)
        
        return Dip_TrEsp # in AU
    
    def _TEST_Compare_SingleDefectProperties(self, tr_charge, gr_charge, ex_charge, struc, index, dAVA=0.0, order=80, approx=1.1):
        ''' Calculate effects of environment such as transition energy shift
        and transition dipole change for single defect.
        
        Parameters
        ----------
        index : list of integer (dimension Natoms_defect)
            Indexes of all atoms from the defect (starting from 0) for which
            transition energy and transition dipole is calculated
        dAVA : float
            **dAVA = <A|V|A> - <G|V|G>** Difference in electrostatic 
            interaction energy between defect and environment for defect in 
            excited state <A|V|A> and in ground state <G|V|G>.
        order : integer (optional - init = 80)
            Specify how many SCF steps shoudl be used in calculation  of induced
            dipoles - according to the used model it should be 2
        approx : real (optional - init=1.1)
            Specifies which approximation should be used.
            
            * **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)` and 
              `Alpha(-E)`.
            * **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.
            * **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
              `Alpha(E)=Alpha(-E)`, however the second one is not condition 

        Returns
        -------
        Eshift : Energy class
            Transition energy shift for the defect due to the fluorographene
            environment calculated from structure with single defect. Units are
            energy managed
        TrDip : numpy array of real (dimension 3)
            Total transition dipole for the defect with environment effects 
            included calculated from structure with single defect (in ATOMIC 
            UNITS)
        
        **Neglecting `tilde{Beta(E)}` is not valid approximation. It shoudl be
        better to neglect Beta(E,-E) to be consistent with approximation for 
        interaction energy**
        
        Notes
        ----------
        dip = Alpha(E)*El_field_TrCharge + Alpha(-E)*El_field_TrCharge 
        Then final transition dipole of molecule with environment is calculated
        according to the approximation:
        
        **Approximation 1.1:**
            dip_fin = dip - (Vinter-DE)*Beta(E,E)*El_field_TrCharge + dip_init(1-1/4*Ind_dip_Beta(E,E)*El_field_TrCharge)
        **Approximation 1.2:**
            dip_fin = dip - (Vinter-DE)*Beta(E,E)*El_field_TrCharge + dip_init     
        **Approximation 1.3:**
            dip_fin = dip - 2*Vinter*Beta(E,E)*El_field_TrCharge + dip_init
        
        '''

        # Get TrEsp Transition dipole
        TrDip_TrEsp = np.dot(self.charge[index],self.coor[index,:]) # vacuum transition dipole for single defect
        
        # Get energy contribution from polarization by transition density
        self.charge[index] = tr_charge
        charge = self.charge[index]
        
        # Set distance matrix
        R_elst = np.tile(struc.coor._value,(self.Nat,1,1))
        R_pol = np.tile(self.coor,(struc.nat,1,1))
        R = (R_elst - np.swapaxes(R_pol,0,1))            # R[ii,jj,:]=self.coor[jj]-self.coor[ii]
        
        # Calculate polarization matrixes
        # TODO: Shift this block to separate function
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('AlphaE',NN=1,eps=1,debug=False)
        Polar1_AlphaE = self._get_interaction_energy(index,charge=charge,debug=False)
        pot1_dipole_AlphaE_tr = potential_dipole(self.dipole,R)
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('AlphaE',NN=2,eps=1,debug=False)
        Polar2_AlphaE = self._get_interaction_energy(index,charge=charge,debug=False)
        Polar2_AlphaE = Polar2_AlphaE - Polar1_AlphaE
        dip_AlphaE = np.sum(self.dipole,axis=0)
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        
        self._calc_dipoles_All('Alpha_E',NN=1,eps=1,debug=False)
        Polar1_Alpha_E = self._get_interaction_energy(index,charge=charge,debug=False)
        pot1_dipole_Alpha_E_tr = potential_dipole(self.dipole,R)
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('Alpha_E',NN=2,eps=1,debug=False)
        dip_Alpha_E = np.sum(self.dipole,axis=0)
        dip_Alpha_E = np.sum(self.dipole,axis=0)
        Polar2_Alpha_E = self._get_interaction_energy(index,charge=charge,debug=False)
        Polar2_Alpha_E = Polar2_Alpha_E - Polar1_Alpha_E
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        
        
        self._calc_dipoles_All('BetaEE',NN=1,eps=1,debug=False)
        dip_Beta = np.sum(self.dipole,axis=0)
        Polar1_Beta_EE = self._get_interaction_energy(index,charge=charge,debug=False)
        pot1_dipole_betaEE_tr = potential_dipole(self.dipole,R)
        
        self.charge[index] = ex_charge
        charge = self.charge[index]
        Polar1_Beta_EE_tr_ex = self._get_interaction_energy(index,charge=charge,debug=False)
        self.charge[index] = gr_charge
        charge = self.charge[index]
        Polar1_Beta_EE_tr_gr = self._get_interaction_energy(index,charge=charge,debug=False)
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        
        # Calculate polarization by ground state charge distribution
        self.charge[index] = gr_charge
        charge = self.charge[index]
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('Alpha_st',NN=1,eps=1,debug=False)
        Polar1_static_gr = self._get_interaction_energy(index,charge=charge,debug=False)
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('Alpha_st',NN=2,eps=1,debug=False)
        Polar2_static_gr = self._get_interaction_energy(index,charge=charge,debug=False)
        Polar2_static_gr = Polar2_static_gr - Polar1_static_gr
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('BetaEE',NN=1,eps=1,debug=False)
        Polar1_Beta_EE_gr = self._get_interaction_energy(index,charge=charge,debug=False)
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        
        # Calculate polarization by excited state charge distribution
        self.charge[index] = ex_charge
        charge = self.charge[index]
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('Alpha_st',NN=1,eps=1,debug=False)
        Polar1_static_ex = self._get_interaction_energy(index,charge=charge,debug=False)
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('Alpha_st',NN=2,eps=1,debug=False)
        Polar2_static_ex = self._get_interaction_energy(index,charge=charge,debug=False)
        Polar2_static_ex = Polar2_static_ex - Polar1_static_ex
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('BetaEE',NN=1,eps=1,debug=False)
        Polar1_Beta_EE_ex = self._get_interaction_energy(index,charge=charge,debug=False)
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        
        # Calculate indiced dipole by charge difference between ground and excited state
        self.charge[index] = ex_charge - gr_charge 
        charge = self.charge[index]
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('Alpha_st',NN=1,eps=1,debug=False)
        pot1_dipole_ex_gr = potential_dipole(self.dipole,R)
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('Alpha_st',NN=2,eps=1,debug=False)
        pot2_dipole_ex_gr = potential_dipole(self.dipole,R)
        pot2_dipole_ex_gr = pot2_dipole_ex_gr - pot1_dipole_ex_gr
        
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('BetaEE',NN=1,eps=1,debug=False)
        pot1_dipole_betaEE_ex_gr = potential_dipole(self.dipole,R)
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        
        # calculate interaction between induced dipoles by transition density with ground and excited charges of the chromophore
        self.charge[index] = tr_charge
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('Alpha_st',NN=1,eps=1,debug=False)
        pot1_dipole_static_tr = potential_dipole(self.dipole,R)
        self.charge[index] = ex_charge
        charge = self.charge[index]
        Polar1_static_tr_ex = self._get_interaction_energy(index,charge=charge,debug=False)
        self.charge[index] = gr_charge
        charge = self.charge[index]
        Polar1_static_tr_gr = self._get_interaction_energy(index,charge=charge,debug=False)
        
        self.charge[index] = tr_charge
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('AlphaE',NN=1,eps=1,debug=False)
        self.charge[index] = gr_charge
        charge = self.charge[index]
        Polar1_AlphaE_tr_gr = self._get_interaction_energy(index,charge=charge,debug=False)
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self.charge[index] = tr_charge
        self._calc_dipoles_All('Alpha_E',NN=1,eps=1,debug=False)
        self.charge[index] = ex_charge
        charge = self.charge[index]
        Polar1_Alpha_E_tr_ex = self._get_interaction_energy(index,charge=charge,debug=False)
        
        
        # Set the variables to initial state
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self.charge[index] = tr_charge
        
        if approx==1.1:
            # Calculate transition energy shift
            Eshift = dAVA + Polar1_AlphaE + Polar2_AlphaE - Polar1_Alpha_E - Polar2_Alpha_E
            Eshift -= (self.VinterFG - dAVA)*Polar1_Beta_EE
            
            # Calculate transition dipoles for every defect
            TrDip = TrDip_TrEsp*(1 + Polar1_Beta_EE/4) + dip_AlphaE + dip_Alpha_E
            TrDip -= (self.VinterFG - dAVA)*dip_Beta
            
            # Change to energy class
            with energy_units('AU'):
                Eshift = EnergyClass(Eshift)
                dAVA = EnergyClass(dAVA)
                Polar1_AlphaE = EnergyClass(Polar1_AlphaE)
                Polar2_AlphaE = EnergyClass(Polar2_AlphaE)
                Polar1_Alpha_E = EnergyClass(Polar1_Alpha_E)
                Polar2_Alpha_E = EnergyClass(Polar2_Alpha_E)
                Polar1_Beta_EE = EnergyClass(Polar1_Beta_EE)
                Polar1_static_ex_gr = EnergyClass(Polar1_static_ex - Polar1_static_gr)
                Polar2_static_ex_gr = EnergyClass(Polar2_static_ex - Polar2_static_gr)
                Polar1_Beta_EE_ex_gr = EnergyClass(Polar1_Beta_EE_ex - Polar1_Beta_EE_gr)
                Polar1_static_tr_ex = EnergyClass(Polar1_static_tr_ex)
                Polar1_static_tr_gr = EnergyClass(Polar1_static_tr_gr)
                Polar1_AlphaE_tr_gr = EnergyClass(Polar1_AlphaE_tr_gr)
                Polar1_Alpha_E_tr_ex = EnergyClass(Polar1_Alpha_E_tr_ex)
                Polar1_Beta_EE_tr_ex = EnergyClass(Polar1_Beta_EE_tr_ex)
                Polar1_Beta_EE_tr_gr = EnergyClass(Polar1_Beta_EE_tr_gr)
            
            res_Energy = {'dE_0-1': Eshift, 'dE_elstat(exct-grnd)': dAVA}
            res_Energy['E_pol1_Alpha(E)'] = Polar1_AlphaE
            res_Energy['E_pol2_Alpha(E)'] = Polar2_AlphaE
            res_Energy['E_pol1_Alpha(-E)'] = Polar1_Alpha_E
            res_Energy['E_pol2_Alpha(-E)'] = Polar2_Alpha_E
            res_Energy['E_pol1_Beta(E,E)'] = Polar1_Beta_EE
            res_Energy['E_pol1_static_(exct-grnd)'] = Polar1_static_ex_gr
            res_Energy['E_pol2_static_(exct-grnd)'] = Polar2_static_ex_gr
            res_Energy['E_pol1_Beta(E,E)_(exct-grnd)'] = Polar1_Beta_EE_ex_gr
            res_Energy['E_pol1_static_(trans)_(exct)'] = Polar1_static_tr_ex
            res_Energy['E_pol1_static_(trans)_(grnd)'] = Polar1_static_tr_gr
            res_Energy['E_pol1_Alpha(E)_(trans)_(grnd)'] = Polar1_AlphaE_tr_gr
            res_Energy['E_pol1_Alpha(-E)_(trans)_(exct)'] = Polar1_Alpha_E_tr_ex
            res_Energy['E_pol1_Beta(E,E)_(trans)_(exct)'] = Polar1_Beta_EE_tr_ex
            res_Energy['E_pol1_Beta(E,E)_(trans)_(grnd)'] = Polar1_Beta_EE_tr_gr
            
            res_Pot = {'Pol2-env_static_(exct-grnd)': pot2_dipole_ex_gr}
            res_Pot['Pol1-env_static_(exct-grnd)'] = pot1_dipole_ex_gr
            res_Pot['Pol1-env_Beta(E,E)_(exct-grnd)'] = pot1_dipole_betaEE_ex_gr
            res_Pot['Pol1-env_Beta(E,E)_(trans)'] = pot1_dipole_betaEE_tr
            res_Pot['Pol1-env_Alpha(E)_(trans)'] = pot1_dipole_AlphaE_tr
            res_Pot['Pol1-env_Alpha(-E)_(trans)'] = pot1_dipole_Alpha_E_tr
            res_Pot['Pol1-env_static_(trans)'] = pot1_dipole_static_tr
            
#            with energy_units('1/cm'):
#                print(Eshift.value,dAVA.value,Polar1_AlphaE.value,Polar2_AlphaE.value,Polar1_AlphaE.value+Polar2_AlphaE.value,Polar1_Alpha_E.value,Polar2_Alpha_E.value,Polar1_Alpha_E.value+Polar2_Alpha_E.value)
#            
            return res_Energy, res_Pot, TrDip
        else:
            raise IOError('Unsupported approximation')

            
    def _TEST_HeterodimerProperties(self, gr_charge1, ex_charge1, gr_charge2, ex_charge2, FG_charge, struc, index1, index2, Eng1, Eng2, dAVA=0.0, dBVB=0.0, order=80, approx=1.1):
        ''' Calculate effects of the environment for structure with two different
        defects such as interaction energy, site transition energy shifts and 
        changes in transition dipoles

        Parameters
        ----------
        index1 : list of integer (dimension Natoms_defect1)
            Indexes of all atoms from the first defect (starting from 0)
        index2 : list of integer (dimension Natoms_defect2)
            Indexes of all atoms from the second defect (starting from 0)
        Eng1 : float 
            Vacuum transition energy of the first defect in ATOMIC UNITS (Hartree)
        Eng2 : float 
            Vacuum transition energy of the second defect in ATOMIC UNITS (Hartree)
        dAVA : float
            **dAVA = <A|V|A> - <G|V|G>** Difference in electrostatic 
            interaction energy between first defect the and environment for the 
            defect in excited state <A|V|A> and in ground state <G|V|G>.
        dBVB : float
            **dBVB = <B|V|B> - <G|V|G>** Difference in electrostatic 
            interaction energy between second defect and the environment for the 
            defect in excited state <B|V|B> and in ground state <G|V|G>.
        order : integer (optional - init = 80)
            Specify how many SCF steps shoudl be used in calculation of induced
            dipoles - according to the used model it should be 2
        approx : real (optional - init=1.1)
            Specifies which approximation should be used.
            
            * **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)` and 
              `Alpha(-E)`.
            * **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.
            * **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
              `Alpha(E)=Alpha(-E)`, however the second one is not condition 
        
        Returns
        -------
        J_inter : Energy class
            Interaction energy with effects of environment included. Units are 
            energy managed
        Eshift1 : Energy class
            Transition energy shift for the first defect due to fluorographene
            environment calculated from heterodymer structure. Units are energy
            managed
        Eshift2 : Energy class
            Transition energy shift for the second defect due to fluorographene
            environment calculated from heterodymer structure. Units are energy
            managed
        TrDip1 : numpy array of real (dimension 3)
            Total transition dipole for the first defect with environment effects 
            included calculated from heterodimer structure (in ATOMIC UNITS)
        TrDip2 : numpy array of real (dimension 3)
            Total transition dipole for the first defect with environment effects 
            included calculated from heterodimer structure (in ATOMIC UNITS)
        AllDipAE : numpy array of float (dimension Natoms x 3)
            Induced atomic dipole moments for all atoms in the environment by 
            the first defect with Alpha(E) atomic polarizability
        AllDipA_E : numpy array of float (dimension Natoms x 3)
            Induced atomic dipole moments for all atoms in the environment by 
            the first defect with Alpha(-E) atomic polarizability
        AllDipBE : numpy array of float (dimension Natoms x 3)
            Induced atomic dipole moments for all atoms in the environment by 
            the first defect with Beta(E,E) atomic polarizability
        '''

        res = {}

        # Get TrEsp interaction energy
        E_TrEsp = self.get_TrEsp_Eng(index1, index2)
        
        # Calculate polarization matrixes (1-2)
        PolarMat1_AlphaE, dip_AlphaE1, dip_AlphaE2, AllDipAE1, AllDipAE2 = self._fill_Polar_matrix(index1,index2,typ='AlphaE',order=1)
        PolarMat1_Alpha_E, dip_Alpha_E1, dip_Alpha_E2, AllDipA_E1, AllDipA_E2 = self._fill_Polar_matrix(index1,index2,typ='Alpha_E',order=1)
        PolarMat_AlphaE, dip_AlphaE1, dip_AlphaE2, AllDipAE1, AllDipAE2 = self._fill_Polar_matrix(index1,index2,typ='AlphaE',order=2)
        PolarMat_Alpha_E, dip_Alpha_E1, dip_Alpha_E2, AllDipA_E1, AllDipA_E2 = self._fill_Polar_matrix(index1,index2,typ='Alpha_E',order=2)
        PolarMat_Beta, dip_Beta1, dip_Beta2, AllDipBE1, AllDipBE2 = self._fill_Polar_matrix(index1,index2,typ='BetaEE',order=order//2)     
        
        res["E_pol2_A(E)"] = (PolarMat_AlphaE - PolarMat1_AlphaE) * conversion_facs_energy["1/cm"]
        res["E_pol2_A(-E)"] = (PolarMat_Alpha_E - PolarMat1_Alpha_E) * conversion_facs_energy["1/cm"]
        res["E_pol2_B(E,E)"] = PolarMat_Beta
        
        
        """ Aditional first order contribution """
        # gr_charge1, ex_charge1, gr_charge2, ex_charge2
        tr_charge1 = self.charge[index1]
        tr_charge2 = self.charge[index2]
        self.charge[index1] = gr_charge1
        self.charge[index2] = ex_charge2
        PolarMat_Alpha_st_gr_ex, dip_Alpha_st1_gr, dip_Alpha_st2_ex, AllDipA_st1_gr, AllDipA_st2_ex = self._fill_Polar_matrix(index1,index2,typ='Alpha_st',order=1)
        
        self.charge[index1] = ex_charge1
        self.charge[index2] = gr_charge2
        PolarMat_Alpha_st_ex_gr, dip_Alpha_st1_ex, dip_Alpha_st2_gr, AllDipA_st1_ex, AllDipA_st2_gr = self._fill_Polar_matrix(index1,index2,typ='Alpha_st',order=1)
        
        # charges for the ground state and excited state are the same => correct
        # difference between first and second defect is in non symetrical charges - repeat the fit with symmetry constrains

        PolarMat_Alpha_st = np.zeros((2,2),dtype='f8')
        PolarMat_Alpha_st[0,0] = np.sum(PolarMat_Alpha_st_ex_gr) # PolarMat_Alpha_st_ex_gr[0,0] + PolarMat_Alpha_st_ex_gr[1,1] + 2*PolarMat_Alpha_st_ex_gr[0,1]
        PolarMat_Alpha_st[1,1] = np.sum(PolarMat_Alpha_st_gr_ex) # PolarMat_Alpha_st_gr_ex[0,0] + PolarMat_Alpha_st_gr_ex[1,1] + 2*PolarMat_Alpha_st_gr_ex[0,1]
        
        # pol1-env
        #-----------------------------------
        # Set distance matrix
        R_elst = np.tile(struc.coor._value,(self.Nat,1,1))
        R_pol = np.tile(self.coor,(struc.nat,1,1))
        R = (R_elst - np.swapaxes(R_pol,0,1))            # R[ii,jj,:]=self.coor[jj]-self.coor[ii]
        # if normaly ordered first are carbon atoms and then are fluorine atoms - for carbon atoms same indexes in pol_mol as in struc
        for ii in range(self.Nat):
            R[ii,ii,:] = 0.0   # self interaction is not permited in potential calculation
# TODO: Maybe also exclude connected fluorinesto atoms ii

        # Calculate potential of induced dipoles 
        pot1_dipole_Alpha_st1_gr = potential_dipole(AllDipA_st1_gr,R)
        pot1_dipole_Alpha_st1_ex = potential_dipole(AllDipA_st1_ex,R)
        pot1_dipole_Alpha_st2_gr = potential_dipole(AllDipA_st2_gr,R)
        pot1_dipole_Alpha_st2_ex = potential_dipole(AllDipA_st2_ex,R)
        
        # calculate interaction energies with environment 
        FG_charge_tmp = FG_charge.charge.copy()
        FG_charge_tmp[index1] = 0.0
        FG_charge_tmp[index2] = 0.0
        
        E_Pol1_env_static_gr1_FG = np.dot(FG_charge_tmp,pot1_dipole_Alpha_st1_gr)
        E_Pol1_env_static_ex1_FG = np.dot(FG_charge_tmp,pot1_dipole_Alpha_st1_ex)
        E_Pol1_env_static_gr2_FG = np.dot(FG_charge_tmp,pot1_dipole_Alpha_st2_gr)
        E_Pol1_env_static_ex2_FG = np.dot(FG_charge_tmp,pot1_dipole_Alpha_st2_ex)
        
        PolarMat_Alpha_st[0,0] = 2*( E_Pol1_env_static_ex1_FG + E_Pol1_env_static_gr2_FG )
        PolarMat_Alpha_st[1,1] = 2*( E_Pol1_env_static_gr1_FG + E_Pol1_env_static_ex2_FG )

        # return transition charges back
        self.charge[index1] = tr_charge1
        self.charge[index2] = tr_charge2
        
        """ Aditional second order contribution - Comparison of magnitudes """
        # Calculate polarization matrix A_grnd B_exct
        self.charge[index1] = gr_charge1
        self.charge[index2] = ex_charge2
        PolarMat_Beta_gr_ex, dip_Beta1_gr, dip_Beta2_ex, AllDipBE1_gr, AllDipBE2_ex = self._fill_Polar_matrix(index1,index2,typ='BetaEE',order=1)
        
        # Calculate polarization matrix A_exct B_grnd
        self.charge[index1] = ex_charge1
        self.charge[index2] = gr_charge2
        PolarMat_Beta_ex_gr, dip_Beta1_ex, dip_Beta2_gr, AllDipBE1_ex, AllDipBE2_gr = self._fill_Polar_matrix(index1,index2,typ='BetaEE',order=1)
        
        res["E_pol1_B(E,E)_(A_exct,B_grnd)"]  = PolarMat_Beta_ex_gr
        res["E_pol1_B(E,E)_(A_grnd,B_exct)"]  = PolarMat_Beta_gr_ex
        
        # calculate pol-env for previous:
        pot1A_dipole_BEE_gr = potential_dipole(AllDipBE1_gr,R)
        pot1A_dipole_BEE_ex = potential_dipole(AllDipBE1_ex,R)
        pot1B_dipole_BEE_gr = potential_dipole(AllDipBE2_gr,R)
        pot1B_dipole_BEE_ex = potential_dipole(AllDipBE2_ex,R)
        PolarMat_env_Beta_ex = np.zeros((2,2),dtype="f8")
        PolarMat_env_Beta_gr = np.zeros((2,2),dtype="f8")
        PolarMat_env_Beta_ex[0,0] = np.dot(FG_charge_tmp,pot1A_dipole_BEE_ex)
        PolarMat_env_Beta_ex[1,1] = np.dot(FG_charge_tmp,pot1B_dipole_BEE_ex)
        PolarMat_env_Beta_gr[0,0] = np.dot(FG_charge_tmp,pot1B_dipole_BEE_gr)
        PolarMat_env_Beta_gr[1,1] = np.dot(FG_charge_tmp,pot1A_dipole_BEE_gr)
        res["E_pol1-env_B(E,E)_grnd"] = PolarMat_env_Beta_gr
        res["E_pol1-env_B(E,E)_exct"] = PolarMat_env_Beta_ex
        
        # Calculate secon order contribution to the first order quantities
        self.charge[index1] = gr_charge1
        self.charge[index2] = ex_charge2
        PolarMat2_Alpha_st_gr_ex, dumm, dumm, AllDipA2_st1_gr, AllDipA2_st2_ex = self._fill_Polar_matrix(index1,index2,typ='Alpha_st',order=2)
        PolarMat2_Alpha_st_gr_ex = PolarMat2_Alpha_st_gr_ex - PolarMat_Alpha_st_gr_ex
        
        self.charge[index1] = ex_charge1
        self.charge[index2] = gr_charge2
        PolarMat2_Alpha_st_ex_gr, dumm, dumm, AllDipA2_st1_ex, AllDipA2_st2_gr = self._fill_Polar_matrix(index1,index2,typ='Alpha_st',order=2)
        PolarMat2_Alpha_st_ex_gr = PolarMat2_Alpha_st_ex_gr - PolarMat_Alpha_st_ex_gr
        res["E_pol2_st_(A_exct,B_grnd)"] = PolarMat2_Alpha_st_ex_gr * conversion_facs_energy["1/cm"]
        res["E_pol2_st_(A_grnd,B_exct)"] = PolarMat2_Alpha_st_gr_ex * conversion_facs_energy["1/cm"]
        
        pot2A_dipole_st_gr = potential_dipole(AllDipA2_st1_gr - AllDipA_st1_gr,R)
        pot2A_dipole_st_ex = potential_dipole(AllDipA2_st1_ex - AllDipA_st1_ex,R)
        pot2B_dipole_st_gr = potential_dipole(AllDipA2_st2_gr - AllDipA_st2_gr,R)
        pot2B_dipole_st_ex = potential_dipole(AllDipA2_st2_ex - AllDipA_st2_ex,R)
        PolarMat2_env_st_ex = np.zeros((2,2),dtype="f8")
        PolarMat2_env_st_gr = np.zeros((2,2),dtype="f8")
        PolarMat2_env_st_ex[0,0] = np.dot(FG_charge_tmp,pot2A_dipole_st_ex)
        PolarMat2_env_st_ex[1,1] = np.dot(FG_charge_tmp,pot2B_dipole_st_ex)
        PolarMat2_env_st_gr[0,0] = np.dot(FG_charge_tmp,pot2B_dipole_st_gr)
        PolarMat2_env_st_gr[1,1] = np.dot(FG_charge_tmp,pot2A_dipole_st_gr)
        res["E_pol2-env_st_grnd"] = PolarMat2_env_st_gr * conversion_facs_energy["1/cm"]
        res["E_pol2-env_st_exct"] = PolarMat2_env_st_ex * conversion_facs_energy["1/cm"]
        
        
        
        
        # Calculate polarization matrixes A_grnd B_0->1
        self.charge[index1] = tr_charge1
        self.charge[index2] = np.zeros(len(index2),dtype='f8')
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('AlphaE',NN=1,eps=1,debug=False)
        self.charge[index1] = np.zeros(len(index1),dtype='f8')
        E_AB_pol1_tr_gr_1 = self._get_interaction_energy(index2,charge=gr_charge2,debug=False)
        E_A_pol1_tr_gr = self._get_interaction_energy(index1,charge=gr_charge1,debug=False)
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        
        self.charge[index1] = np.zeros(len(index1),dtype='f8')
        self.charge[index2] = tr_charge2
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('AlphaE',NN=1,eps=1,debug=False)
        self.charge[index2] = np.zeros(len(index2),dtype='f8')
        E_AB_pol1_gr_tr_1 = self._get_interaction_energy(index1,charge=gr_charge1,debug=False)
        E_B_pol1_tr_gr = self._get_interaction_energy(index2,charge=gr_charge2,debug=False)
        
        self.charge[index1] = gr_charge1
        self.charge[index2] = np.zeros(len(index2),dtype='f8')
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('AlphaE',NN=1,eps=1,debug=False)
        self.charge[index1] = np.zeros(len(index1),dtype='f8')
        E_AB_pol1_gr_tr_2 = self._get_interaction_energy(index2,charge=tr_charge2,debug=False)
        
        self.charge[index1] = np.zeros(len(index1),dtype='f8')
        self.charge[index2] = gr_charge2
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('AlphaE',NN=1,eps=1,debug=False)
        self.charge[index2] = np.zeros(len(index2),dtype='f8')
        E_AB_pol1_tr_gr_2 = self._get_interaction_energy(index1,charge=tr_charge1,debug=False)
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        
        # return transition charges back
        if (gr_charge1!=gr_charge2).any() :
            raise IOError("Heterodimer should have the same ground state charges")
        # return transition charges back
        if (tr_charge1!=tr_charge2).any() :
            raise IOError("Heterodimer should have the same transition charges")
        self.charge[index1] = gr_charge1
        self.charge[index2] = tr_charge2
        PolarMat_AlphaE_gr_tr, dip_AlphaE1_gr, dip_AlphaE2_tr, AllDipAE1_gr, AllDipAE2_tr = self._fill_Polar_matrix(index1,index2,typ='AlphaE',order=1)
        E_AB_pol1_gr_tr = PolarMat_AlphaE_gr_tr[0,1]
        self.charge[index1] = tr_charge1
        self.charge[index2] = gr_charge2
        PolarMat_AlphaE_gr_tr, dip_AlphaE1_gr, dip_AlphaE2_tr, AllDipAE1_gr, AllDipAE2_tr = self._fill_Polar_matrix(index1,index2,typ='AlphaE',order=1)
        E_AB_pol1_tr_gr = PolarMat_AlphaE_gr_tr[0,1]
        
        
        res["E_pol1_B(E,E)_(tr_gr,ex)"] = np.zeros((2,2),dtype="f8")
        self.charge[index1] = tr_charge1
        self.charge[index2] = np.zeros(len(index2),dtype='f8')
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('BetaEE',NN=1,eps=1,debug=False)
        self.charge[index1] = np.zeros(len(index1),dtype='f8')
        res["E_pol1_B(E,E)_(tr_gr,ex)"][0,0] = self._get_interaction_energy(index1,charge=gr_charge1,debug=False)
        res["E_pol1_B(E,E)_(tr_gr,ex)"][0,1] = self._get_interaction_energy(index1,charge=ex_charge1,debug=False)
        self.charge[index1] = np.zeros(len(index2),dtype='f8')
        self.charge[index2] = tr_charge2
        self.dipole = np.zeros((self.Nat,3),dtype='f8')
        self._calc_dipoles_All('BetaEE',NN=1,eps=1,debug=False)
        self.charge[index2] = np.zeros(len(index2),dtype='f8')
        res["E_pol1_B(E,E)_(tr_gr,ex)"][1,0] = self._get_interaction_energy(index2,charge=gr_charge2,debug=False)
        res["E_pol1_B(E,E)_(tr_gr,ex)"][1,1] = self._get_interaction_energy(index2,charge=ex_charge2,debug=False)
        
        # return transition charges back
        self.charge[index1] = tr_charge1
        self.charge[index2] = tr_charge2
        
        # compare electrostatic energies - TEST
        VAB_0101 = self.get_TrEsp_Eng(index1, index2)
        self.charge[index1] = ex_charge1
        VAB_1101 = self.get_TrEsp_Eng(index1, index2)
        self.charge[index1] = gr_charge1
        VAB_0001 = self.get_TrEsp_Eng(index1, index2)
        self.charge[index2] = gr_charge2
        VAB_0000 = self.get_TrEsp_Eng(index1, index2)
        self.charge[index1] = ex_charge1
        self.charge[index2] = ex_charge2
        VAB_1111 = self.get_TrEsp_Eng(index1, index2)
        self.charge[index2] = gr_charge2
        VAB_1100 = self.get_TrEsp_Eng(index1, index2)
        charge_orig1 = FG_charge.charge[index1]
        charge_orig2 = FG_charge.charge[index2]
        FG_charge.charge[index1] = gr_charge1
        FG_charge.charge[index2] = 0.0
        E_grnd=FG_charge.get_EnergyShift()
        FG_charge.charge[index1] = ex_charge1
        FG_charge.charge[index2] = 0.0
        E_exct=FG_charge.get_EnergyShift()
        FG_charge.charge[index1] = tr_charge1
        FG_charge.charge[index2] = 0.0
        E_trans=FG_charge.get_EnergyShift()
        
        FG_charge.charge[index1] = charge_orig1
        FG_charge.charge[index2] = charge_orig2
        
        self.charge[index1] = tr_charge1
        self.charge[index2] = tr_charge2

        # calculate new eigenstates and energies
        HH=np.zeros((2,2),dtype='f8')
        if Eng1<Eng2:
            HH[0,0] = Eng1+dAVA
            HH[1,1] = Eng2+dBVB
        else:
            HH[1,1] = Eng1+dAVA
            HH[0,0] = Eng2+dBVB
        HH[0,1] = E_TrEsp
        HH[1,0] = HH[0,1]
        Energy,Coeff=np.linalg.eigh(HH)
        
        d_esp=np.sqrt( E_TrEsp**2 + ((Eng2-Eng1+dBVB-dAVA)/2)**2 )          # sqrt( (<A|V|B>)**2 + ((Eng2-Eng1+dBVB-dAVA)/2)**2  )
        
        
        # Calculate interaction energies
        if approx==1.1:
            # Calculate Total polarizability matrix
            PolarMat = PolarMat_AlphaE + PolarMat_Alpha_E + PolarMat_Alpha_st + PolarMat_Beta*(dAVA/2 + dBVB/2 - self.VinterFG)
            
            # Calculate interaction energies
            C1 = Coeff.T[0]
            E1 = Energy[0] + np.dot(C1, np.dot(PolarMat - d_esp*PolarMat_Beta, C1.T))
            
            C2 = Coeff.T[1]
            E2 = Energy[1] + np.dot(C2, np.dot(PolarMat + d_esp*PolarMat_Beta, C2.T))
            
            J_inter = np.sqrt( (E2 - E1)**2 - (Eng2 - Eng1)**2 )/2*np.sign(E_TrEsp)
            
            # Calculate energy shifts for every defect
            Eshift1 = dAVA + PolarMat_AlphaE[0,0] - PolarMat_Alpha_E[1,1]
            Eshift1 -= (self.VinterFG - dAVA)*PolarMat_Beta[0,0]
            
            Eshift2 = dBVB + PolarMat_AlphaE[1,1] - PolarMat_Alpha_E[0,0]
            Eshift2 -= (self.VinterFG - dBVB)*PolarMat_Beta[1,1]
            
            # Calculate transition dipoles for every defect
            TrDip1 = np.dot(self.charge[index1],self.coor[index1,:]) # vacuum transition dipole for single defect
            TrDip1 = TrDip1*(1 + PolarMat_Beta[0,0]/4) + dip_AlphaE1 + dip_Alpha_E1
            TrDip1 -= (self.VinterFG - dAVA)*dip_Beta1
            
            TrDip2 = np.dot(self.charge[index2],self.coor[index2,:]) # vacuum transition dipole for single defect
            TrDip2 = TrDip2*(1 + PolarMat_Beta[1,1]/4) + dip_AlphaE2 + dip_Alpha_E2
            TrDip2 -= (self.VinterFG - dBVB)*dip_Beta2
            
        
            # Change to energy class
            with energy_units('AU'):
                J_inter = EnergyClass(J_inter)
                Eshift1 = EnergyClass(Eshift1)
                Eshift2 = EnergyClass(Eshift2)
                E_pol_static1_ex_gr = EnergyClass(PolarMat_Alpha_st_ex_gr[0,0]-PolarMat_Alpha_st_gr_ex[0,0])
                E_pol_static2_ex_gr = EnergyClass(PolarMat_Alpha_st_gr_ex[1,1]-PolarMat_Alpha_st_ex_gr[1,1])
                E_pol_env_static1_ex_gr = EnergyClass(E_Pol1_env_static_ex1_FG - E_Pol1_env_static_gr1_FG)
                E_pol_env_static2_ex_gr = EnergyClass(E_Pol1_env_static_ex2_FG - E_Pol1_env_static_gr2_FG)
                
                VAB_0101 = EnergyClass(VAB_0101)
                VAB_1101 = EnergyClass(VAB_1101)
                VAB_0001 = EnergyClass(VAB_0001)
                VAB_0000 = EnergyClass(VAB_0000)
                VAB_1111 = EnergyClass(VAB_1111)
                VAB_1100 = EnergyClass(VAB_1100)
                E_grnd = EnergyClass(E_grnd)
                E_exct = EnergyClass(E_exct)
                E_trans = EnergyClass(E_trans)
                
                E_AB_pol1_gr_tr = EnergyClass(E_AB_pol1_gr_tr)
                E_AB_pol1_tr_gr = EnergyClass(E_AB_pol1_tr_gr)
                E_AB_pol1_gr_tr_1 = EnergyClass(E_AB_pol1_gr_tr_1)
                E_AB_pol1_tr_gr_1 = EnergyClass(E_AB_pol1_tr_gr_1)
                E_AB_pol1_gr_tr_2 = EnergyClass(E_AB_pol1_gr_tr_2)
                E_AB_pol1_tr_gr_2 = EnergyClass(E_AB_pol1_tr_gr_2)
                E_A_pol1_tr_gr = EnergyClass(E_A_pol1_tr_gr)
                E_B_pol1_tr_gr = EnergyClass(E_B_pol1_tr_gr)

            with energy_units("1/cm"):
                print("EA_pol1_s_ex_gr EA_pol1_env_s_ex_gr EAB_pol1_tr_gr EA_pol1_tr_gr")
                print("  {:9.4f}        {:9.4f}           {:9.4f}       {:9.4f}".format(
                                E_pol_static1_ex_gr.value,
                                E_pol_env_static1_ex_gr.value,
                                E_AB_pol1_tr_gr.value,
                                E_A_pol1_tr_gr.value))
                print("   VAB_0101       VAB_1101        VAB_0001       VAB_0000      VAB_1111     VAB_1100       E_grnd        E_exct        E_trans")
                print(VAB_0101.value, VAB_1101.value, VAB_0001.value, VAB_0000.value, VAB_1111.value, VAB_1100.value, E_grnd.value, E_exct.value, E_trans.value)


#            res["E_pol2_A(E)"]
#            res["E_pol2_A(-E)"]
#            res["E_pol2_B(E,E)"]
#            res["E_pol1_B(E,E)_(A_exct,B_grnd)"]
#            res["E_pol1_B(E,E)_(A_grnd,B_exct)"]
#            res["E_pol1-env_B(E,E)_grnd"]
#            res["E_pol1-env_B(E,E)_exct"]
#            res["E_pol2_st_(A_exct,B_grnd)"]
#            res["E_pol2_st_(A_grnd,B_exct)"]
#            res["E_pol2-env_st_grnd"]
#            res["E_pol2-env_st_exct"]
    

            return J_inter, Eshift1, Eshift2, TrDip1, TrDip2, AllDipAE1, AllDipA_E1, AllDipBE1, res
        else:
            raise IOError('Unsupported approximation')

#==============================================================================
# Definition of fuction for allocation of polarized molecules
#==============================================================================
def prepare_molecule_1Def(filenames,indx,AlphaE,Alpha_E,BetaE,VinterFG,verbose=False,CoarseGrain="plane",**kwargs):
    ''' Read all informations needed for Dielectric class and transform system
    with single defect into this class. Useful for calculation of interaction 
    energies, transition site energy shifts and dipole changes.
    
    Parameters
    ----------
    filenames : list of dictionary (dimension Nsystems)
        In the dictionaries there are specified all needed files which contains 
        nessesary information for transformig the system into Dielectric class.
        keys:
        `'1def_structure'`: xyz file with system geometry and atom types
        `'charge_structure'`: xyz file with defect like molecule geometry for which transition charges were calculated
        `charge_grnd`: file with ground state charges for the defect
        `'charge_exct'`: file with excited state charges for the defect
        `'charge'`: file with transition charges for the defect
    indx : list of integers (dimension Nsystems x 6)
        For every system there are specified indexes neded for asignment of defect 
        atoms. First three indexes correspond to center and two main axes of 
        reference structure (structure which was used for charges calculation)
        and the remaining three indexes are corresponding atoms of the defect
        on fluorographene system.
    AlphaE : numpy.array of real (dimension 2x2)
        Atomic polarizability Alpha(E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    Alpha_E : numpy.array of real (dimension 2x2)
        Atomic polarizability Alpha(-E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    BetaE : numpy.array of real (dimension 2x2)
        Atomic polarizability Beta(E,E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    VinterFG : real
        Difference in electrostatic interaction energy between interaction of
        excited C-F corse grained atom of fluorographene with all others
        fluorographene corse grained atoms in ground state and interaction of
        ground state C-F corse grained atom of fluorographene with all others
        fluorographene corse grained atoms in ground state. Units are ATOMIC 
        UNITS (Hartree)
    CoarseGrain : string (optional init = "plane")
        Possible values are: "plane","C","CF". Define which level of coarse 
        grained model should be used. If ``CoarseGrain="plane"`` then all atoms
        are projected on plane defined by nvec and C-F atoms re treated as single
        atom - for this case polarizabilities defined only in 2D by two numbers.
        If ``CoarseGrain="C"`` then carbon atoms are center for atomic
        polarizability tensor and again C-F are treated as a single atom. 
        If ``CoarseGrain="CF"`` then center of C-F bonds are used as center for
        atomic polarizability tensor and again C-F are treated as a single atom.
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    **kwargs : dictionary (optional)
        Definition of polarizabitity matrixes for defect atoms (if nonzero
        polarizability is used)
        
    Returns
    -------
    mol_polar : Dielectric class 
        Fluorographene with defect in Dielectric class which contains all information
        needed for calculation of energy shifts and dipole changes for defect 
        embeded in fluorographene
    index1 : list of integer (dimension Ndefect_atoms)
        Atom indexes of defect atoms
    charge : numpy.array of real (dimension Ndefect_atoms)
        Transition charges for every defect atom. First charge correspond to atom
        defined by first index in index1 list and so on.
    struc : Structure class
        Structure of the fluorographene system with single defects

    '''    
    if verbose:
        print(indx)
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
    
    if verbose:
        print('     Reading charges and format to polarization format...')
    struc_test=Structure()
    struc_test.load_xyz(xyzfile2)   # Structure of molecule used for fitting charges
    if verbose:
        print('        Loading molecule...')
    struc=Structure()
    struc.load_xyz(xyzfile)   # Fluorographene with single defect
    
    coor,charge,at_type=read_TrEsp_charges(filenameESP,verbose=False)
    if verbose:
        print('        Centering molecule...')
    struc.center(indx_center1,indx_x1,indx_y1)
    
    index1=identify_molecule(struc,struc_test,indx_center1,indx_x1,indx_y1,indx_center_test,indx_x_test,indx_y_test,onlyC=True)

    if len(index1)!=len(np.unique(index1)):
        raise IOError('There are repeating elements in index file')

    # Assign pol types and charges
    PolCoor,Polcharge,PolType = _prepare_polar_structure_1def(struc,index1,charge,CoarseGrain,verbose=False)
#    PolType=[]
#    Polcharge=[]
#    PolCoor=[]
#    for ii in range(struc.nat):
#        if struc.at_type[ii]=='C' and (ii in index1):
#            Polcharge.append(charge[np.where(index1==ii)[0][0]])
#            PolType.append('C')
#            PolCoor.append(struc.coor._value[ii])
#        elif struc.at_type[ii]=='C':
#            PolType.append('CF')
#            Polcharge.append(0.0)
#            PolCoor.append(struc.coor._value[ii])
#    PolType=np.array(PolType)
#    Polcharge=np.array(Polcharge,dtype='f8')
#    PolCoor=np.array(PolCoor,dtype='f8')
#    
#    # project molecule whole system to plane defined by defect
#    nvec=np.array([0.0,0.0,1.0],dtype='f8')
#    center=np.array([0.0,0.0,0.0],dtype='f8')
#    PolCoor=project_on_plane(PolCoor,nvec,center)
    
    polar={}
    polar['AlphaE']=np.zeros((len(PolCoor),3,3),dtype='f8')
    polar['Alpha_E']=np.zeros((len(PolCoor),3,3),dtype='f8')
    polar['BetaE']=np.zeros((len(PolCoor),3,3),dtype='f8')
    
    mol_polar=Dielectric(PolCoor,Polcharge,np.zeros((len(PolCoor),3),dtype='f8'),
                              polar['AlphaE'],polar['Alpha_E'],polar['BetaE'],VinterFG)
    
    ZeroM=np.zeros((3,3),dtype='f8')
    
    Polarizability = { 'CF': [AlphaE,Alpha_E,BetaE], 'CD': [AlphaE,Alpha_E,BetaE]}
    
    if "Alpha(E)" in kwargs.keys():
        AlphaE_def=kwargs['Alpha(E)']
        Alpha_E_def=kwargs['Alpha(-E)']
        BetaE_def=kwargs['Beta(E,E)']
        Polarizability['C'] = [AlphaE_def,Alpha_E_def,BetaE_def]
    else :
        Polarizability['C'] = [ZeroM,ZeroM,ZeroM]
        
    if "Fpolar" in kwargs.keys():
        Polarizability['FC'] =  kwargs['Fpolar']
    else:
        Polarizability['FC'] = [ZeroM,ZeroM,ZeroM]

    mol_polar.polar=mol_polar.assign_polar(PolType,**{'PolValues': Polarizability})
    
    if "Alpha_static" in kwargs.keys():
        mol_polar.polar['Alpha_st'] = np.zeros((len(PolCoor),3,3),dtype='f8')
        
        if CoarseGrain=="all_atom":
            Alpha_static=kwargs["Alpha_static"]
            AlphaF_static=kwargs["AlphaF_static"]
        else:
            Alpha_static=kwargs["Alpha_static"]
            AlphaF_static=ZeroM
        
        for ii in range(len(PolType)):
            if PolType[ii]=='CF':
                mol_polar.polar['Alpha_st'][ii]=Alpha_static
            elif PolType[ii]=='FC':
                mol_polar.polar['Alpha_st'][ii]=AlphaF_static
        
    return mol_polar,index1,charge,struc

def prepare_molecule_2Def(filenames,indx,AlphaE,Alpha_E,BetaE,VinterFG,verbose=False, def2_charge=True,CoarseGrain="plane",**kwargs):
    ''' Read all informations needed for Dielectric class and transform system
    with two same defects into this class. Useful for calculation of interaction 
    energies, transition site energy shifts and dipole changes.
    
    Parameters
    ----------
    filenames :  dictionary 
        In the dictionary there are specified all needed files which contains 
        nessesary information for transformig the system into Dielectric class.
        keys:
        
        * ``'2def_structure'``: xyz file with FG system with two defects 
          geometry and atom types
        * ``'charge1_structure'``: xyz file with defect-like molecule geometry
          for which transition charges were calculated corresponding to first
          defect
        * ``'charge1'``: file with transition charges for the first defect 
          (from TrEsp charges fitting)
        * ``'charge2_structure'``: xyz file with defect-like molecule geometry
          for which transition charges were calculated corresponding to second
          defect
        * ``'charge2'``: file with transition charges for the second defect 
          (from TrEsp charges fitting)
        
    indx : list of integers (dimension 9)
        There are specified indexes neded for asignment of defect 
        atoms. First three indexes correspond to center and two main axes of 
        reference structure (structure which was used for charges calculation)
        and the remaining six indexes are corresponding atoms of the defects 
        on fluorographene system (three correspond to first defect and the last
        three to the second one).
    AlphaE : numpy.array of real (dimension 2x2)
        Atomic polarizability Alpha(E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    Alpha_E : numpy.array of real (dimension 2x2)
        Atomic polarizability Alpha(-E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    BetaE : numpy.array of real (dimension 2x2)
        Atomic polarizability Beta(E,E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    VinterFG : real
        Difference in electrostatic interaction energy between interaction of
        excited C-F corse grained atom of fluorographene with all others
        fluorographene corse grained atoms in ground state and interaction of
        ground state C-F corse grained atom of fluorographene with all others
        fluorographene corse grained atoms in ground state. Units are ATOMIC 
        UNITS (Hartree)
    def2_charge : logical (init = True)
        Specifies if transition charges should be placed also to the second 
        defect
    CoarseGrain : string (optional init = "plane")
        Possible values are: "plane","C","CF". Define which level of coarse 
        grained model should be used. If ``CoarseGrain="plane"`` then all atoms
        are projected on plane defined by nvec and C-F atoms re treated as single
        atom - for this case polarizabilities defined only in 2D by two numbers.
        If ``CoarseGrain="C"`` then carbon atoms are center for atomic
        polarizability tensor and again C-F are treated as a single atom. 
        If ``CoarseGrain="CF"`` then center of C-F bonds are used as center for
        atomic polarizability tensor and again C-F are treated as a single atom.
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
    charge1 : numpy.array of real (dimension Ndefect1_atoms)
        Transition charges for every atom of the first defect. First charge
        correspond to atom defined by first index in index1 list and so on.
    charge2 : numpy.array of real (dimension Ndefect2_atoms)
        Transition charges for every atom of the second defect. First charge
        correspond to atom defined by first index in index2 list and so on. 
    struc : Structure class
        Structure of the fluorographene system with two defects
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
    xyzfile_chrg1=filenames['charge1_structure']
    filenameESP_chrg1=filenames['charge1']
    xyzfile_chrg2=filenames['charge2_structure']
    filenameESP_chrg2=filenames['charge2']
    xyzfile=filenames['2def_structure']
    
    # Read Transition charges
    #filenameESP="".join([MolDir,'Perylene_TDDFT_fitted_charges_NoH.out'])
    if verbose:
        print('     Reading charges and format to polarization format...')
    struc1_test=Structure()
    struc2_test=Structure()
    struc1_test.load_xyz(xyzfile_chrg1)      # Structure of molecule used for fitting charges
    struc2_test.load_xyz(xyzfile_chrg2)      # Structure of molecule used for fitting charges
    coor,charge1,at_type=read_TrEsp_charges(filenameESP_chrg1,verbose=False)
    coor,charge2,at_type=read_TrEsp_charges(filenameESP_chrg2,verbose=False)
    
    # load molecule - fuorographene with 2 defects 
    if verbose:
        print('        Loading molecule...')
    struc=Structure()
    struc.load_xyz(xyzfile)     # Fluorographene with two defects
    
    index1=identify_molecule(struc,struc1_test,indx_center1,indx_x1,indx_y1,indx_center_test,indx_x_test,indx_y_test,onlyC=True)
    index2=identify_molecule(struc,struc2_test,indx_center2,indx_x2,indx_y2,indx_center_test,indx_x_test,indx_y_test,onlyC=True)
    if len(index1)!=len(np.unique(index1)) or len(index2)!=len(np.unique(index2)):
        print('index1:')
        print(index1)
        print('index2:')
        print(index2)
        raise IOError('There are repeating elements in index file')

    # Assign pol types
    PolCoor,Polcharge,PolType = _prepare_polar_structure_2def(struc,index1,charge1,index2,charge2,CoarseGrain)
#    PolType=[]
#    Polcharge=[]
#    PolCoor=[]
#    for ii in range(struc.nat):
#        if struc.at_type[ii]=='C' and (ii in index1):
#            Polcharge.append(charge1[np.where(index1==ii)[0][0]])
#            PolType.append('C')
#            PolCoor.append(struc.coor._value[ii])
#        elif struc.at_type[ii]=='C' and (ii in index2):
#            if def2_charge:
#                Polcharge.append(charge2[np.where(index2==ii)[0][0]])
#            else:
#                Polcharge.append(0.0)
#            #Polcharge.append(charge[np.where(index2==ii)[0][0]])
#            PolType.append('C')
#            PolCoor.append(struc.coor._value[ii])
#        elif struc.at_type[ii]=='C':
#            PolType.append('CF')
#            Polcharge.append(0.0)
#            PolCoor.append(struc.coor._value[ii])
#            
#    PolType=np.array(PolType)
#    Polcharge=np.array(Polcharge,dtype='f8')
#    PolCoor=np.array(PolCoor,dtype='f8')
#    
#    # project molecule whole system to plane defined by defect
#    center=np.array([0.0,0.0,0.0],dtype='f8')
#    PolCoor=project_on_plane(PolCoor,nvec,center)
    
    # center projected molecule on plane
    if verbose:
        print('        Centering molecule...')
    PolCoor,Phi,Psi,Chi,center=CenterMolecule(PolCoor,indx_center1,[indx_center1,indx_x1,indx_center2,indx_x2],[indx_center1,indx_y1,indx_center2,indx_y2],print_angles=True)
    # Do the same transformation also with the structure
    struc.move(-center[0],-center[1],-center[2])
    struc.rotate(Phi,Psi,Chi)
    
    polar={}
    polar['AlphaE']=np.zeros((len(PolCoor),3,3),dtype='f8')
    polar['Alpha_E']=np.zeros((len(PolCoor),3,3),dtype='f8')
    polar['BetaE']=np.zeros((len(PolCoor),3,3),dtype='f8')
    
    mol_polar=Dielectric(PolCoor,Polcharge,np.zeros((len(PolCoor),3),dtype='f8'),
                         polar['AlphaE'],polar['Alpha_E'],polar['BetaE'],VinterFG)
    
    ZeroM=np.zeros((3,3),dtype='f8')
    
    Polarizability = { 'CF': [AlphaE,Alpha_E,BetaE], 'CD': [AlphaE,Alpha_E,BetaE]}
    
    if "Alpha(E)" in kwargs.keys():
        AlphaE_def=kwargs['Alpha(E)']
        Alpha_E_def=kwargs['Alpha(-E)']
        BetaE_def=kwargs['Beta(E,E)']
        Polarizability['C'] = [AlphaE_def,Alpha_E_def,BetaE_def]
    else :
        Polarizability['C'] = [ZeroM,ZeroM,ZeroM]
        
    if "Fpolar" in kwargs.keys():
        Polarizability['FC'] =  kwargs['Fpolar']
    else:
        Polarizability['FC'] = [ZeroM,ZeroM,ZeroM]

    mol_polar.polar=mol_polar.assign_polar(PolType,**{'PolValues': Polarizability})
    
    if "Alpha_static" in kwargs.keys():
        mol_polar.polar['Alpha_st'] = np.zeros((len(PolCoor),3,3),dtype='f8')
        
        if CoarseGrain=="all_atom":
            Alpha_static=ZeroM
        else:
            Alpha_static=kwargs["Alpha_static"]
        
        for ii in range(len(PolType)):
            if PolType[ii]=='CF':
                mol_polar.polar['Alpha_st'][ii]=Alpha_static
                          
    return mol_polar,index1,index2,charge1,charge2,struc

def _prepare_polar_structure_1def(struc,index1,charge1,Type,verbose=False):
    """
    Type = "plane","C","CF","all_atom"
    
    """
    if not Type in ["plane","C","CF","all_atom"]:
        raise Warning("Unsupported type of coarse graining.")
    
    if verbose:
        print(Type)
        
    # Molecule has to be centered and oriented first before this calculation is done 
        
    # Assign pol types and charges
    PolType=[]
    Polcharge=[]
    PolCoor=[]
    
    if Type == "plane" or Type == "C": 
        for ii in range(struc.nat):
            if struc.at_type[ii]=='C' and (ii in index1):
                Polcharge.append(charge1[np.where(index1==ii)[0][0]])
                PolType.append('C')
                PolCoor.append(struc.coor._value[ii])
            elif struc.at_type[ii]=='C':
                PolType.append('CF')
                Polcharge.append(0.0)
                PolCoor.append(struc.coor._value[ii])
        PolType=np.array(PolType)
        Polcharge=np.array(Polcharge,dtype='f8')
        PolCoor=np.array(PolCoor,dtype='f8')
        
        if Type == "plane":
            # project molecule whole system to plane defined by defect
            nvec_test,origin_test = fit_plane(PolCoor)
            PolCoor=project_on_plane(PolCoor,nvec_test,origin_test)
            #center=np.array([0.0,0.0,0.0],dtype='f8')
            #PolCoor=project_on_plane(PolCoor,nvec,center)
    
    elif Type == "all_atom":
        PolCoor = struc.coor._value.copy()
        for ii in range(struc.nat):
            if struc.at_type[ii]=='C' and (ii in index1):
                Polcharge.append(charge1[np.where(index1==ii)[0][0]])
                PolType.append('C')
            elif struc.at_type[ii]=='C':
                PolType.append('CF')
                Polcharge.append(0.0)
            elif struc.at_type[ii]=='F':
                PolType.append('FC')
                Polcharge.append(0.0)
        PolType=np.array(PolType)
        Polcharge=np.array(Polcharge,dtype='f8')
        PolCoor=np.array(PolCoor,dtype='f8')
    
    elif Type == "CF":
        connectivity = []
        for ii in range(struc.nat):
            connectivity.append([])
        if struc.bonds is None:
            struc.guess_bonds()
        for ii in range(len(struc.bonds)):
            indx1=struc.bonds[ii][0]
            at1=struc.at_type[indx1]
            indx2=struc.bonds[ii][1]
            at2=struc.at_type[indx2]
            if at1=="C" and at2=="F":    
                connectivity[indx1].append(indx2)
            elif at2=="C" and at1=="F":
                connectivity[indx2].append(indx1)
            
        for ii in range(struc.nat):
            if struc.at_type[ii]=='C' and (ii in index1):
                Polcharge.append(charge1[np.where(index1==ii)[0][0]])
                PolType.append('C')
                PolCoor.append(struc.coor._value[ii])

            elif struc.at_type[ii]=='C':
                PolType.append('CF')
                Polcharge.append(0.0)
                
                # polarizabiliy center will be located at center of C-F bond (or F-C-F for border carbons)
                count = 1
                position = struc.coor._value[ii]
                for jj in range(len(connectivity[ii])):
                    position += struc.coor._value[ connectivity[ii][jj] ]
                    count += 1
                position = position / count
                PolCoor.append(position)
        
        PolType=np.array(PolType)
        Polcharge=np.array(Polcharge,dtype='f8')
        PolCoor=np.array(PolCoor,dtype='f8')
# TODO: add all atom representation
    
    return PolCoor,Polcharge,PolType

def _prepare_polar_structure_2def(struc,index1,charge1,index2,charge2,Type,verbose=False):
    """
    Type = "plane","C","CF","all_atom"
    
    """
    if not Type in ["plane","C","CF","all_atom"]:
        raise Warning("Unsupported type of coarse graining.")
    
    if verbose:
        print(Type)
    
    # Assign pol types
    PolType=[]
    Polcharge=[]
    PolCoor=[]
    if Type == "plane" or Type == "C": 
        for ii in range(struc.nat):
            if struc.at_type[ii]=='C' and (ii in index1):
                Polcharge.append(charge1[np.where(index1==ii)[0][0]])
                PolType.append('C')
                PolCoor.append(struc.coor._value[ii])
            elif struc.at_type[ii]=='C' and (ii in index2):
                Polcharge.append(charge2[np.where(index2==ii)[0][0]])
                PolType.append('C')
                PolCoor.append(struc.coor._value[ii])
            elif struc.at_type[ii]=='C':
                PolType.append('CF')
                Polcharge.append(0.0)
                PolCoor.append(struc.coor._value[ii])
                
        PolType=np.array(PolType)
        Polcharge=np.array(Polcharge,dtype='f8')
        PolCoor=np.array(PolCoor,dtype='f8')
    
        if Type == "plane":
            # project molecule whole system to plane defined by defect
            nvec_test,origin_test = fit_plane(PolCoor)
            PolCoor=project_on_plane(PolCoor,nvec_test,origin_test)
            #center=np.array([0.0,0.0,0.0],dtype='f8')
            #PolCoor=project_on_plane(PolCoor,nvec,center)
            
    elif Type == "all_atom":
        PolCoor = struc.coor._value.copy()
        for ii in range(struc.nat):
            if struc.at_type[ii]=='C' and (ii in index1):
                Polcharge.append(charge1[np.where(index1==ii)[0][0]])
                PolType.append('C')
            elif struc.at_type[ii]=='C' and (ii in index2):
                Polcharge.append(charge2[np.where(index2==ii)[0][0]])
                PolType.append('C')
            elif struc.at_type[ii]=='C':
                PolType.append('CF')
                Polcharge.append(0.0)
            elif struc.at_type[ii]=='F':
                PolType.append('FC')
                Polcharge.append(0.0)
        PolType=np.array(PolType)
        Polcharge=np.array(Polcharge,dtype='f8')
        #print(len(PolCoor),len(PolType))
            
# TODO: TEST this assignment of polarizability centers
    elif Type == "CF":
        connectivity = []
        for ii in range(struc.nat):
            connectivity.append([])
        if struc.bonds is None:
            struc.guess_bonds()
        for ii in range(len(struc.bonds)):
            indx1=struc.bonds[ii][0]
            at1=struc.at_type[indx1]
            indx2=struc.bonds[ii][1]
            at2=struc.at_type[indx2]
            if at1=="C" and at2=="F":    
                connectivity[indx1].append(indx2)
            elif at2=="C" and at1=="F":
                connectivity[indx2].append(indx1)
                
        for ii in range(struc.nat):
            if struc.at_type[ii]=='C' and (ii in index1):
                Polcharge.append(charge1[np.where(index1==ii)[0][0]])
                PolType.append('C')
                PolCoor.append(struc.coor._value[ii])
            elif struc.at_type[ii]=='C' and (ii in index2):
                Polcharge.append(charge2[np.where(index2==ii)[0][0]])
                PolType.append('C')
                PolCoor.append(struc.coor._value[ii])
            elif struc.at_type[ii]=='C':
                PolType.append('CF')
                Polcharge.append(0.0)
                
                # polarizabiliy center will be located at center of C-F bond (or F-C-F for border carbons)
                count = 1
                position = struc.coor._value[ii]
                for jj in range(len(connectivity[ii])):
                    position += struc.coor._value[ connectivity[ii][jj] ]
                    count += 1
                position = position / count
                PolCoor.append(position)
                
        PolType=np.array(PolType)
        Polcharge=np.array(Polcharge,dtype='f8')
        PolCoor=np.array(PolCoor,dtype='f8')
# TODO: add all atom representation
    
    return PolCoor,Polcharge,PolType


#TODO: Get rid of ShortName
def Calc_SingleDef_FGprop(filenames,ShortName,index_all,AlphaE,Alpha_E,BetaE,VinterFG,FG_charges,ChargeType,order=80,verbose=False,approx=1.1,MathOut=False,CoarseGrain="plane",**kwargs):
    ''' Calculate energy shifts and transition dipole shifts for single defect
    embeded in fluorographene
    
    Parameters
    ----------
    filenames : dictionary
        Dictionary with information about all needed files which contains 
        nessesary information for transformig the system into Dielectric class
        and electrostatic calculations. Keys:
            
        * ``'1def_structure'``: xyz file with FG system with single defect 
          geometry and atom types
        * ``'charge_structure'``: xyz file with defect-like molecule geometry
          for which transition charges were calculated corresponding to first
          defect
        * ``'charge'``: file with transition charges for the defect 
          (from TrEsp charges fitting)
        * ``'charge_grnd'``: file with ground state charges for the defect 
          (from TrEsp charges fitting)
        * ``'charge_exct'``: file with excited state charges for the defect 
          (from TrEsp charges fitting)
          
    ShortName : string
        Short description of the system 
    index_all : list of integers (dimension 6)
        There are specified indexes neded for asignment of defect 
        atoms. First three indexes correspond to center and two main axes of 
        reference structure (structure which was used for charges calculation)
        and the last three indexes are corresponding atoms of the defect.
    AlphaE : numpy.array of real (dimension 2x2)
        Atomic polarizability Alpha(E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    Alpha_E : numpy.array of real (dimension 2x2)
        Atomic polarizability Alpha(-E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    BetaE : numpy.array of real (dimension 2x2)
        Atomic polarizability Beta(E,E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    VinterFG : real
        Difference in electrostatic interaction energy between interaction of
        excited C-F corse grained atom of fluorographene with all others
        fluorographene corse grained atoms in ground state and interaction of
        ground state C-F corse grained atom of fluorographene with all others
        fluorographene corse grained atoms in ground state. Units are ATOMIC 
        UNITS (Hartree)
    FG_charges : list of real (dimension 2)
        [charge on inner fluorographene atom, charge on borded fluorographe carbon]
    ChargeType : string
        Specifies which charges should be used for electrostatic calculations
        (ground and excited state charges) for defect atoms. Allowed types are:
        ``'qchem'``, ``'qchem_all'``, ``'AMBER'`` and ``'gaussian'``. 
        
        * ``'qchem'`` - charges calculated by fiting Q-Chem ESP on carbon
          atoms. 
        * ``'qchem_all'`` - charges calculated by fiting Q-Chem ESP on all
          atoms, only carbon charges are used and same charge is added to all 
          carbon atoms in order to have neutral molecule. 
        * ``'AMBER'`` - not yet fully implemented. 
        * ``'gaussian'`` - not yet fully implemented.
        
    order : integer (optional - init=80)
        Specify how many SCF steps shoudl be used in calculation  of induced
        dipoles - according to the used model it should be 2
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    approx : real (optional - init=1.1)
        Specifies which approximation should be used.
            
        * **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)` and 
          `Alpha(-E)`.
        * **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.
        * **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
          `Alpha(E)=Alpha(-E)`, however the second one is not condition 
    
    Returns
    --------
    Eshift : Energy class
        Transition energy shift for the defect due to the fluorographene
        environment calculated from structure with single defect. Units are
        energy managed
    TrDip : numpy array of real (dimension 3)
        Total transition dipole for the defect with environment effects 
        included calculated from structure with single defect (in ATOMIC UNITS)
        
    Notes
    --------
    By comparing QC calculations it was found that energy shift from structure 
    with two defects and with single defect is almost the same.
    '''    
    
    if verbose:
        print('Calculation of interaction energy for:',ShortName)
    
    # read and prepare molecule
    mol_polar,index1,charge,struc=prepare_molecule_1Def(filenames,index_all,AlphaE,Alpha_E,BetaE,VinterFG,verbose=False,CoarseGrain=CoarseGrain,**kwargs)

    # calculate dAVA = <A|V|A>-<G|V|G>
    AditInfo={'Structure': struc,'index1': index1}
    mol_Elstat,index,charge_grnd,charge_exct=ElStat_PrepareMolecule_1Def(filenames,index_all,FG_charges,ChargeType=ChargeType,verbose=False,**AditInfo)
    dAVA=mol_Elstat.get_EnergyShift()

    # calculate transition energy shifts and transition dipole change      
    Eshift,TrDip=mol_polar.get_SingleDefectProperties(index1,dAVA=dAVA,order=order,approx=approx)
    
    if verbose:
        with energy_units("1/cm"):
            print(ShortName,Eshift.value) 
            print("      dipole:",np.linalg.norm(TrDip))
            print("      dAVA:",dAVA*conversion_facs_energy["1/cm"],'cm-1')

    return Eshift, TrDip

#TODO: Get rid of ShortName
#TODO: Input vacuum transition energies
def Calc_Heterodimer_FGprop(filenames,ShortName,index_all,AlphaE,Alpha_E,BetaE,VinterFG,FG_charges,ChargeType,order=80,verbose=False,approx=1.1,MathOut=False,CoarseGrain="plane",**kwargs):
    ''' Calculate interaction energies between defects embeded in polarizable atom
    environment for all systems given in filenames. Possibility of calculate 
    transition energy shifts and transition dipoles.
    
    Parameters
    ----------
    filenames : dictionary
        Dictionary with information about all needed files which contains 
        nessesary information for transformig the system into Dielectric class
        and electrostatic calculations. Keys:
            
        * ``'2def_structure'``: xyz file with FG system with two defects 
          geometry and atom types
        * ``'charge1_structure'``: xyz file with defect-like molecule geometry
          for which transition charges were calculated corresponding to first
          defect
        * ``'charge1'``: file with transition charges for the first defect 
          (from TrEsp charges fitting)
        * ``'charge1_grnd'``: file with ground state charges for the first defect 
          (from TrEsp charges fitting)
        * ``'charge1_exct'``: file with excited state charges for the first defect 
          (from TrEsp charges fitting)
        * ``'charge2_structure'``: xyz file with defect-like molecule geometry
          for which transition charges were calculated corresponding to second
          defect
        * ``'charge2'``: file with transition charges for the second defect 
          (from TrEsp charges fitting)
        * ``'charge2_grnd'``: file with ground state charges for the second defect 
          (from TrEsp charges fitting)
        * ``'charge2_exct'``: file with excited state charges for the second defect 
          (from TrEsp charges fitting)
          
    ShortName : string
        Short description of the system 
    index_all : list of integers (dimension 6)
        There are specified indexes neded for asignment of defect 
        atoms. First three indexes correspond to center and two main axes of 
        reference structure (structure which was used for charges calculation)
        and the next three indexes are corresponding atoms of the first defects 
        on fluorographene system and the last three indexes are corresponding 
        atoms of the second defect.
    AlphaE : numpy.array of real (dimension 2x2)
        Atomic polarizability Alpha(E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    Alpha_E : numpy.array of real (dimension 2x2)
        Atomic polarizability Alpha(-E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    BetaE : numpy.array of real (dimension 2x2)
        Atomic polarizability Beta(E,E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    VinterFG : real
        Difference in electrostatic interaction energy between interaction of
        excited C-F corse grained atom of fluorographene with all others
        fluorographene corse grained atoms in ground state and interaction of
        ground state C-F corse grained atom of fluorographene with all others
        fluorographene corse grained atoms in ground state. Units are ATOMIC 
        UNITS (Hartree)
    FG_charges : list of real (dimension 2)
        [charge on inner fluorographene atom, charge on borded fluorographe carbon]
    ChargeType : string
        Specifies which charges should be used for electrostatic calculations
        (ground and excited state charges) for defect atoms. Allowed types are:
        ``'qchem'``, ``'qchem_all'``, ``'AMBER'`` and ``'gaussian'``. 
        
        * ``'qchem'`` - charges calculated by fiting Q-Chem ESP on carbon
          atoms. 
        * ``'qchem_all'`` - charges calculated by fiting Q-Chem ESP on all
          atoms, only carbon charges are used and same charge is added to all 
          carbon atoms in order to have neutral molecule. 
        * ``'AMBER'`` - not yet fully implemented. 
        * ``'gaussian'`` - not yet fully implemented.
        
    order : integer (optional - init=80)
        Specify how many SCF steps shoudl be used in calculation  of induced
        dipoles - according to the used model it should be 2
    CoarseGrain : string (optional init = "plane")
        Possible values are: "plane","C","CF". Define which level of coarse 
        grained model should be used. If ``CoarseGrain="plane"`` then all atoms
        are projected on plane defined by nvec and C-F atoms re treated as single
        atom - for this case polarizabilities defined only in 2D by two numbers.
        If ``CoarseGrain="C"`` then carbon atoms are center for atomic
        polarizability tensor and again C-F are treated as a single atom. 
        If ``CoarseGrain="CF"`` then center of C-F bonds are used as center for
        atomic polarizability tensor and again C-F are treated as a single atom.
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    approx : real (optional - init=1.1)
        Specifies which approximation should be used.
            
        **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)` and 
        `Alpha(-E)`.
        
        **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.

        **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
        `Alpha(E)=Alpha(-E)`, however the second one is not condition 

    Returns
    --------
    Einter : Energy class
        Interaction energy with effects of environment included. Units are 
        energy managed
    Eshift1 : Energy class
        Transition energy shift for the first defect due to fluorographene
        environment calculated from heterodymer structure. Units are energy
        managed
    Eshift2 : Energy class
        Transition energy shift for the second defect due to fluorographene
        environment calculated from heterodymer structure. Units are energy
        managed
    TrDip1 : numpy array of real (dimension 3)
        Total transition dipole for the first defect with environment effects 
        included calculated from heterodimer structure (in ATOMIC UNITS)
    TrDip2 : numpy array of real (dimension 3)
        Total transition dipole for the first defect with environment effects 
        included calculated from heterodimer structure (in ATOMIC UNITS)

    Notes
    ----------
    No far working only with two symmetric defects - for heterodimer need to
    input vacuum transition energy for every defect.    
    
    '''    
    
    if verbose:
        print('Calculation of interaction energy for:',ShortName)
    
    # read and prepare molecule
    mol_polar,index1,index2,charge1,charge2,struc=prepare_molecule_2Def(filenames,index_all,AlphaE,Alpha_E,BetaE,VinterFG,verbose=False,def2_charge=True,CoarseGrain=CoarseGrain,**kwargs)
    
    # # calculate dAVA = <A|V|A>-<G|V|G> and dBVB = <B|V|B>-<G|V|G>
    AditInfo={'Structure': struc,'index1': index1,'index2':index2}
    mol_Elstat,indx1,indx2,charge1_grnd,charge2_grnd,charge1_exct,charge2_exct=ElStat_PrepareMolecule_2Def(filenames,index_all,FG_charges,ChargeType=ChargeType,verbose=False,**AditInfo)
    dAVA=mol_Elstat.get_EnergyShift(index=index2, charge=charge2_grnd)
    dBVB=mol_Elstat.get_EnergyShift(index=index1, charge=charge1_grnd)

    # calculate interaction energy and transition energy shifts      
    Einter,Eshift1,Eshift2,TrDip1,TrDip2,dipAE,dipA_E,dipBE=mol_polar.get_HeterodimerProperties(index1,index2,0.0,0.0,dAVA=dAVA,dBVB=dBVB,order=order,approx=approx)
    
    if verbose:
        with energy_units("1/cm"):
            print('        Total interaction energy:',Einter.value)
            print(ShortName,abs(Einter.value),Eshift1.value,Eshift2.value) 
            print("dipole:",np.linalg.norm(TrDip1),np.linalg.norm(TrDip2))
            print("dAVA:",dAVA*conversion_facs_energy["1/cm"],"dBVB:",dBVB*conversion_facs_energy["1/cm"])
    
    if MathOut:
        if not os.path.exists("Pictures"):
            os.makedirs("Pictures")
        Bonds = GuessBonds(mol_polar.coor)
        
        if CoarseGrain in ["plane","C","CF"]:
            at_type = ['C']*mol_polar.Nat
        elif CoarseGrain == "all_atom":
            at_type = struc.at_type.copy()
        
        mat_filename = "".join(['Pictures/Polar_',ShortName,'_AlphaE.nb'])
        params = {'TrPointCharge': mol_polar.charge,'AtDipole': dipAE,'rSphere_dip': 0.5,'rCylinder_dip':0.1}
        OutputMathematica(mat_filename,mol_polar.coor,Bonds,at_type,scaleDipole=50.0,**params)
        mat_filename = "".join(['Pictures/Polar_',ShortName,'_Alpha_E.nb'])
        params = {'TrPointCharge': mol_polar.charge,'AtDipole': dipA_E,'rSphere_dip': 0.5,'rCylinder_dip':0.1}
        OutputMathematica(mat_filename,mol_polar.coor,Bonds,at_type,scaleDipole=50.0,**params)
        mat_filename = "".join(['Pictures/Polar_',ShortName,'_BetaE.nb'])
        params = {'TrPointCharge': mol_polar.charge,'AtDipole': dipBE,'rSphere_dip': 0.5,'rCylinder_dip':0.1}
        OutputMathematica(mat_filename,mol_polar.coor,Bonds,at_type,scaleDipole=50.0,**params)

        

    return Einter, Eshift1, Eshift2, TrDip1, TrDip2

def TEST_Calc_Heterodimer_FGprop(filenames,ShortName,index_all,AlphaE,Alpha_E,BetaE,VinterFG,FG_charges,ChargeType,order=80,verbose=False,approx=1.1,MathOut=False,CoarseGrain="plane",**kwargs):
    ''' Calculate interaction energies between defects embeded in polarizable atom
    environment for all systems given in filenames. Possibility of calculate 
    transition energy shifts and transition dipoles.
    
    Parameters
    ----------
    filenames : dictionary
        Dictionary with information about all needed files which contains 
        nessesary information for transformig the system into Dielectric class
        and electrostatic calculations. Keys:
            
        * ``'2def_structure'``: xyz file with FG system with two defects 
          geometry and atom types
        * ``'charge1_structure'``: xyz file with defect-like molecule geometry
          for which transition charges were calculated corresponding to first
          defect
        * ``'charge1'``: file with transition charges for the first defect 
          (from TrEsp charges fitting)
        * ``'charge1_grnd'``: file with ground state charges for the first defect 
          (from TrEsp charges fitting)
        * ``'charge1_exct'``: file with excited state charges for the first defect 
          (from TrEsp charges fitting)
        * ``'charge2_structure'``: xyz file with defect-like molecule geometry
          for which transition charges were calculated corresponding to second
          defect
        * ``'charge2'``: file with transition charges for the second defect 
          (from TrEsp charges fitting)
        * ``'charge2_grnd'``: file with ground state charges for the second defect 
          (from TrEsp charges fitting)
        * ``'charge2_exct'``: file with excited state charges for the second defect 
          (from TrEsp charges fitting)
          
    ShortName : string
        Short description of the system 
    index_all : list of integers (dimension 6)
        There are specified indexes neded for asignment of defect 
        atoms. First three indexes correspond to center and two main axes of 
        reference structure (structure which was used for charges calculation)
        and the next three indexes are corresponding atoms of the first defects 
        on fluorographene system and the last three indexes are corresponding 
        atoms of the second defect.
    AlphaE : numpy.array of real (dimension 2x2)
        Atomic polarizability Alpha(E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    Alpha_E : numpy.array of real (dimension 2x2)
        Atomic polarizability Alpha(-E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    BetaE : numpy.array of real (dimension 2x2)
        Atomic polarizability Beta(E,E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    VinterFG : real
        Difference in electrostatic interaction energy between interaction of
        excited C-F corse grained atom of fluorographene with all others
        fluorographene corse grained atoms in ground state and interaction of
        ground state C-F corse grained atom of fluorographene with all others
        fluorographene corse grained atoms in ground state. Units are ATOMIC 
        UNITS (Hartree)
    FG_charges : list of real (dimension 2)
        [charge on inner fluorographene atom, charge on borded fluorographe carbon]
    ChargeType : string
        Specifies which charges should be used for electrostatic calculations
        (ground and excited state charges) for defect atoms. Allowed types are:
        ``'qchem'``, ``'qchem_all'``, ``'AMBER'`` and ``'gaussian'``. 
        
        * ``'qchem'`` - charges calculated by fiting Q-Chem ESP on carbon
          atoms. 
        * ``'qchem_all'`` - charges calculated by fiting Q-Chem ESP on all
          atoms, only carbon charges are used and same charge is added to all 
          carbon atoms in order to have neutral molecule. 
        * ``'AMBER'`` - not yet fully implemented. 
        * ``'gaussian'`` - not yet fully implemented.
        
    order : integer (optional - init=80)
        Specify how many SCF steps shoudl be used in calculation  of induced
        dipoles - according to the used model it should be 2
    CoarseGrain : string (optional init = "plane")
        Possible values are: "plane","C","CF". Define which level of coarse 
        grained model should be used. If ``CoarseGrain="plane"`` then all atoms
        are projected on plane defined by nvec and C-F atoms re treated as single
        atom - for this case polarizabilities defined only in 2D by two numbers.
        If ``CoarseGrain="C"`` then carbon atoms are center for atomic
        polarizability tensor and again C-F are treated as a single atom. 
        If ``CoarseGrain="CF"`` then center of C-F bonds are used as center for
        atomic polarizability tensor and again C-F are treated as a single atom.
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    approx : real (optional - init=1.1)
        Specifies which approximation should be used.
            
        **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)` and 
        `Alpha(-E)`.
        
        **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.

        **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
        `Alpha(E)=Alpha(-E)`, however the second one is not condition 

    Returns
    --------
    Einter : Energy class
        Interaction energy with effects of environment included. Units are 
        energy managed
    Eshift1 : Energy class
        Transition energy shift for the first defect due to fluorographene
        environment calculated from heterodymer structure. Units are energy
        managed
    Eshift2 : Energy class
        Transition energy shift for the second defect due to fluorographene
        environment calculated from heterodymer structure. Units are energy
        managed
    TrDip1 : numpy array of real (dimension 3)
        Total transition dipole for the first defect with environment effects 
        included calculated from heterodimer structure (in ATOMIC UNITS)
    TrDip2 : numpy array of real (dimension 3)
        Total transition dipole for the first defect with environment effects 
        included calculated from heterodimer structure (in ATOMIC UNITS)

    Notes
    ----------
    No far working only with two symmetric defects - for heterodimer need to
    input vacuum transition energy for every defect.    
    
    '''    
    
    if verbose:
        print('Calculation of interaction energy for:',ShortName)
    
    # read and prepare molecule
    mol_polar,index1,index2,charge1,charge2,struc=prepare_molecule_2Def(filenames,index_all,AlphaE,Alpha_E,BetaE,VinterFG,verbose=False,def2_charge=True,CoarseGrain=CoarseGrain,**kwargs)
    
    if (mol_polar.charge[index1] != mol_polar.charge[index2]).any():
        raise Warning("Transition charges are not the same - after creation.")
    
    # # calculate dAVA = <A|V|A>-<G|V|G> and dBVB = <B|V|B>-<G|V|G>
    AditInfo={'Structure': struc,'index1': index1,'index2':index2}
    mol_Elstat,indx1,indx2,charge1_grnd,charge2_grnd,charge1_exct,charge2_exct=ElStat_PrepareMolecule_2Def(filenames,index_all,FG_charges,ChargeType=ChargeType,verbose=False,**AditInfo)       
    dAVA=mol_Elstat.get_EnergyShift(index=index2, charge=charge2_grnd)        
    dBVB=mol_Elstat.get_EnergyShift(index=index1, charge=charge1_grnd)
#    dAVA=mol_Elstat.get_EnergyShift(index=index2)        
#    dBVB=mol_Elstat.get_EnergyShift(index=index1)
    
    if (mol_polar.charge[index1] != mol_polar.charge[index2]).any():
        raise Warning("Transition charges are not the same - after elstat.")

    # calculate interaction energy and transition energy shifts - so far for homodimer   
    Einter,Eshift1,Eshift2,TrDip1,TrDip2,dipAE,dipA_E,dipBE,res=mol_polar._TEST_HeterodimerProperties(charge1_grnd,charge1_exct,charge2_grnd,charge2_exct,mol_Elstat,struc,index1,index2,0.0,0.0,dAVA=dAVA,dBVB=dBVB,order=order,approx=approx)
    #get_HeterodimerProperties_new(self, gr_charge1, ex_charge1, gr_charge2, ex_charge2, FG_elstat, struc, index1, index2, Eng1, Eng2, eps, dAVA=0.0, dBVB=0.0, order=2, approx=1.1)
    
#            res["E_pol2_A(E)"]
#            res["E_pol2_A(-E)"]
#            res["E_pol2_B(E,E)"]
#            res["E_pol1_B(E,E)_(A_exct,B_grnd)"]
#            res["E_pol1_B(E,E)_(A_grnd,B_exct)"]
#            res["E_pol1-env_B(E,E)_grnd"]
#            res["E_pol1-env_B(E,E)_exct"]
#            res["E_pol2_st_(A_exct,B_grnd)"]
#            res["E_pol2_st_(A_grnd,B_exct)"]
#            res["E_pol2-env_st_grnd"]
#            res["E_pol2-env_st_exct"]
#            res["E_pol1_B(E,E)_(tr_gr,ex)"]
    
    import os
    if not os.path.isfile("Temp.dat"):
        text = "                            pol2_A(E)      |       pol2_A(-E)      |  pol2_st_(A_ex,B_gr)  |  pol2_st_(A_gr,B_ex)  |   E_pol2-env_st_grnd  |   E_pol2-env_st_exct  |       pol1_BEE        |  pol1_BEE_(A_ex,B_gr) |  pol1_BEE_(A_gr,B_ex) |   pol1-env_BEE_grnd   |   pol1-env_BEE_exct   |  pol1_BEE_(tr_gr,ex)  |"
        os.system("".join(['echo "',text,'" >> Temp.dat']))
        text = "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|"
        os.system("".join(['echo "',text,'" >> Temp.dat']))
    #                              pol2_A(E)      |       pol2_A(-E)      |  pol2_st_(A_ex,B_gr)  |  pol2_st_(A_gr,B_ex)  |   E_pol2-env_st_grnd  |   E_pol2-env_st_exct  |       pol1_BEE        |  pol1_BEE_(A_ex,B_gr) |  pol1_BEE_(A_gr,B_ex) |   pol1-env_BEE_grnd   |"
    ii = 0
    text="{:21} {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.6f} {:10.6f} | {:10.6f} {:10.6f} | {:10.6f} {:10.6f} | {:10.6f} {:10.6f} | {:10.6f} {:10.6f} | {:10.6f} {:10.6f} |".format(
            ShortName,res["E_pol2_A(E)"][ii,0],res["E_pol2_A(E)"][ii,1],res["E_pol2_A(-E)"][ii,0],res["E_pol2_A(-E)"][ii,1],
            res["E_pol2_st_(A_exct,B_grnd)"][ii,0],res["E_pol2_st_(A_exct,B_grnd)"][ii,1],res["E_pol2_st_(A_grnd,B_exct)"][ii,0],
            res["E_pol2_st_(A_grnd,B_exct)"][ii,1],res["E_pol2-env_st_grnd"][ii,0],res["E_pol2-env_st_grnd"][ii,1],
            res["E_pol2-env_st_exct"][ii,0],res["E_pol2-env_st_exct"][ii,1],res["E_pol2_B(E,E)"][ii,0],res["E_pol2_B(E,E)"][ii,1],
            res["E_pol1_B(E,E)_(A_exct,B_grnd)"][ii,0],res["E_pol1_B(E,E)_(A_exct,B_grnd)"][ii,1],res["E_pol1_B(E,E)_(A_grnd,B_exct)"][ii,0],
            res["E_pol1_B(E,E)_(A_grnd,B_exct)"][ii,1],res["E_pol1-env_B(E,E)_grnd"][ii,0],res["E_pol1-env_B(E,E)_grnd"][ii,1],
            res["E_pol1-env_B(E,E)_exct"][ii,0],res["E_pol1-env_B(E,E)_exct"][ii,1],res["E_pol1_B(E,E)_(tr_gr,ex)"][ii,0],res["E_pol1_B(E,E)_(tr_gr,ex)"][ii,1])
    os.system("".join(['echo "',text,'" >> Temp.dat']))
    ii = 1
    text="{:21} {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.6f} {:10.6f} | {:10.6f} {:10.6f} | {:10.6f} {:10.6f} | {:10.6f} {:10.6f} | {:10.6f} {:10.6f} | {:10.6f} {:10.6f} |".format(
            " ",res["E_pol2_A(E)"][ii,0],res["E_pol2_A(E)"][ii,1],res["E_pol2_A(-E)"][ii,0],res["E_pol2_A(-E)"][ii,1],
            res["E_pol2_st_(A_exct,B_grnd)"][ii,0],res["E_pol2_st_(A_exct,B_grnd)"][ii,1],res["E_pol2_st_(A_grnd,B_exct)"][ii,0],
            res["E_pol2_st_(A_grnd,B_exct)"][ii,1],res["E_pol2-env_st_grnd"][ii,0],res["E_pol2-env_st_grnd"][ii,1],
            res["E_pol2-env_st_exct"][ii,0],res["E_pol2-env_st_exct"][ii,1],res["E_pol2_B(E,E)"][ii,0],res["E_pol2_B(E,E)"][ii,1],
            res["E_pol1_B(E,E)_(A_exct,B_grnd)"][ii,0],res["E_pol1_B(E,E)_(A_exct,B_grnd)"][ii,1],res["E_pol1_B(E,E)_(A_grnd,B_exct)"][ii,0],
            res["E_pol1_B(E,E)_(A_grnd,B_exct)"][ii,1],res["E_pol1-env_B(E,E)_grnd"][ii,0],res["E_pol1-env_B(E,E)_grnd"][ii,1],
            res["E_pol1-env_B(E,E)_exct"][ii,0],res["E_pol1-env_B(E,E)_exct"][ii,1],res["E_pol1_B(E,E)_(tr_gr,ex)"][ii,0],res["E_pol1_B(E,E)_(tr_gr,ex)"][ii,1])
    os.system("".join(['echo "',text,'" >> Temp.dat']))
    text = "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|"
    os.system("".join(['echo "',text,' " >> Temp.dat']))
    
#    if (mol_polar.charge[index1] != mol_polar.charge[index2]).any():
#        raise Warning("Transition charges are not the same - after polar.")
    
# TODO: For testing output structure and polarization structure - I'm getting different values for first and second defect
#    struc.output_to_xyz("".join([ShortName,"_structure.xyz"]))
#    from QChemTool.QuantumChem.output import OutputToXYZ
#    from QChemTool.General.units import conversion_facs_position
#    OutputToXYZ(mol_polar.coor*conversion_facs_position["Angstrom"],["C"]*len(mol_polar.coor),"".join([ShortName,"_pol.xyz"]))
    
    if verbose:
        with energy_units("1/cm"):
            print('        Total interaction energy:',Einter.value)
            print(ShortName,abs(Einter.value),Eshift1.value,Eshift2.value) 
            print("dipole:",np.linalg.norm(TrDip1),np.linalg.norm(TrDip2))
            print("dAVA:",dAVA*conversion_facs_energy["1/cm"],"dBVB:",dBVB*conversion_facs_energy["1/cm"])
    
    if MathOut:
        if not os.path.exists("Pictures"):
            os.makedirs("Pictures")
        Bonds = GuessBonds(mol_polar.coor)
        
        if CoarseGrain in ["plane","C","CF"]:
            at_type = ['C']*mol_polar.Nat
        elif CoarseGrain == "all_atom":
            at_type = struc.at_type.copy()
            
#        if (mol_polar.charge[index1] != mol_polar.charge[index2]).any():
#            raise Warning("Transition charges are not the same - before output.")
        
        mat_filename = "".join(['Pictures/Polar_',ShortName,'_AlphaE.nb'])
        params = {'TrPointCharge': mol_polar.charge,'AtDipole': dipAE,'rSphere_dip': 0.5,'rCylinder_dip':0.1}
        OutputMathematica(mat_filename,mol_polar.coor,Bonds,at_type,scaleDipole=50.0,**params)
        mat_filename = "".join(['Pictures/Polar_',ShortName,'_Alpha_E.nb'])
        params = {'TrPointCharge': mol_polar.charge,'AtDipole': dipA_E,'rSphere_dip': 0.5,'rCylinder_dip':0.1}
        OutputMathematica(mat_filename,mol_polar.coor,Bonds,at_type,scaleDipole=50.0,**params)
        mat_filename = "".join(['Pictures/Polar_',ShortName,'_BetaE.nb'])
        params = {'TrPointCharge': mol_polar.charge,'AtDipole': dipBE,'rSphere_dip': 0.5,'rCylinder_dip':0.1}
        OutputMathematica(mat_filename,mol_polar.coor,Bonds,at_type,scaleDipole=50.0,**params)

        

    return Einter, Eshift1, Eshift2, TrDip1, TrDip2

def Calc_Heterodimer_FGprop_new(filenames,ShortName,E1,E2,index_all,AlphaE,Alpha_E,BetaE,VinterFG,FG_charges,ChargeType,order=2,verbose=False,approx=1.1,MathOut=False,CoarseGrain="plane",**kwargs):
    ''' Calculate interaction energies between defects embeded in polarizable atom
    environment for all systems given in filenames. Possibility of calculate 
    transition energy shifts and transition dipoles.
    
    Parameters
    ----------
    filenames : dictionary
        Dictionary with information about all needed files which contains 
        nessesary information for transformig the system into Dielectric class
        and electrostatic calculations. Keys:
            
        * ``'2def_structure'``: xyz file with FG system with two defects 
          geometry and atom types
        * ``'charge1_structure'``: xyz file with defect-like molecule geometry
          for which transition charges were calculated corresponding to first
          defect
        * ``'charge1'``: file with transition charges for the first defect 
          (from TrEsp charges fitting)
        * ``'charge1_grnd'``: file with ground state charges for the first defect 
          (from TrEsp charges fitting)
        * ``'charge1_exct'``: file with excited state charges for the first defect 
          (from TrEsp charges fitting)
        * ``'charge2_structure'``: xyz file with defect-like molecule geometry
          for which transition charges were calculated corresponding to second
          defect
        * ``'charge2'``: file with transition charges for the second defect 
          (from TrEsp charges fitting)
        * ``'charge2_grnd'``: file with ground state charges for the second defect 
          (from TrEsp charges fitting)
        * ``'charge2_exct'``: file with excited state charges for the second defect 
          (from TrEsp charges fitting)
          
    ShortName : string
        Short description of the system 
    index_all : list of integers (dimension 6)
        There are specified indexes neded for asignment of defect 
        atoms. First three indexes correspond to center and two main axes of 
        reference structure (structure which was used for charges calculation)
        and the next three indexes are corresponding atoms of the first defects 
        on fluorographene system and the last three indexes are corresponding 
        atoms of the second defect.
    AlphaE : numpy.array of real (dimension 2x2)
        Atomic polarizability Alpha(E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    Alpha_E : numpy.array of real (dimension 2x2)
        Atomic polarizability Alpha(-E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    BetaE : numpy.array of real (dimension 2x2)
        Atomic polarizability Beta(E,E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    VinterFG : real
        Difference in electrostatic interaction energy between interaction of
        excited C-F corse grained atom of fluorographene with all others
        fluorographene corse grained atoms in ground state and interaction of
        ground state C-F corse grained atom of fluorographene with all others
        fluorographene corse grained atoms in ground state. Units are ATOMIC 
        UNITS (Hartree)
    FG_charges : list of real (dimension 2)
        [charge on inner fluorographene atom, charge on borded fluorographe carbon]
    ChargeType : string
        Specifies which charges should be used for electrostatic calculations
        (ground and excited state charges) for defect atoms. Allowed types are:
        ``'qchem'``, ``'qchem_all'``, ``'AMBER'`` and ``'gaussian'``. 
        
        * ``'qchem'`` - charges calculated by fiting Q-Chem ESP on carbon
          atoms. 
        * ``'qchem_all'`` - charges calculated by fiting Q-Chem ESP on all
          atoms, only carbon charges are used and same charge is added to all 
          carbon atoms in order to have neutral molecule. 
        * ``'AMBER'`` - not yet fully implemented. 
        * ``'gaussian'`` - not yet fully implemented.
        
    order : integer (optional - init=80)
        Specify how many SCF steps shoudl be used in calculation  of induced
        dipoles - according to the used model it should be 2
    CoarseGrain : string (optional init = "plane")
        Possible values are: "plane","C","CF". Define which level of coarse 
        grained model should be used. If ``CoarseGrain="plane"`` then all atoms
        are projected on plane defined by nvec and C-F atoms re treated as single
        atom - for this case polarizabilities defined only in 2D by two numbers.
        If ``CoarseGrain="C"`` then carbon atoms are center for atomic
        polarizability tensor and again C-F are treated as a single atom. 
        If ``CoarseGrain="CF"`` then center of C-F bonds are used as center for
        atomic polarizability tensor and again C-F are treated as a single atom.
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    approx : real (optional - init=1.1)
        Specifies which approximation should be used.
            
        **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)` and 
        `Alpha(-E)`.
        
        **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.

        **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
        `Alpha(E)=Alpha(-E)`, however the second one is not condition 

    Returns
    --------
    Einter : Energy class
        Interaction energy with effects of environment included. Units are 
        energy managed
    Eshift1 : Energy class
        Transition energy shift for the first defect due to fluorographene
        environment calculated from heterodymer structure. Units are energy
        managed
    Eshift2 : Energy class
        Transition energy shift for the second defect due to fluorographene
        environment calculated from heterodymer structure. Units are energy
        managed
    TrDip1 : numpy array of real (dimension 3)
        Total transition dipole for the first defect with environment effects 
        included calculated from heterodimer structure (in ATOMIC UNITS)
    TrDip2 : numpy array of real (dimension 3)
        Total transition dipole for the first defect with environment effects 
        included calculated from heterodimer structure (in ATOMIC UNITS)

    Notes
    ----------
    No far working only with two symmetric defects - for heterodimer need to
    input vacuum transition energy for every defect.    
    
    '''    
    
    if verbose:
        print('Calculation of interaction energy for:',ShortName)
    
    # read and prepare molecule
    mol_polar,index1,index2,charge1,charge2,struc=prepare_molecule_2Def(filenames,index_all,AlphaE,Alpha_E,BetaE,VinterFG,verbose=False,def2_charge=True,CoarseGrain=CoarseGrain,**kwargs)
    
    if (mol_polar.charge[index1] != mol_polar.charge[index2]).any():
        raise Warning("Transition charges are not the same - after creation.")
    
    # # calculate dAVA = <A|V|A>-<G|V|G> and dBVB = <B|V|B>-<G|V|G>
    AditInfo={'Structure': struc,'index1': index1,'index2':index2}
    mol_Elstat,indx1,indx2,charge1_grnd,charge2_grnd,charge1_exct,charge2_exct=ElStat_PrepareMolecule_2Def(filenames,index_all,FG_charges,ChargeType=ChargeType,verbose=False,**AditInfo)       
    dAVA=mol_Elstat.get_EnergyShift(index=index2, charge=charge2_grnd)        
    dBVB=mol_Elstat.get_EnergyShift(index=index1, charge=charge1_grnd)
#    dAVA=mol_Elstat.get_EnergyShift(index=index2)        
#    dBVB=mol_Elstat.get_EnergyShift(index=index1)
    
    if (mol_polar.charge[index1] != mol_polar.charge[index2]).any():
        raise Warning("Transition charges are not the same - after elstat.")

    eps = EnergyClass( (E1.value+E2.value)/2 )
    # calculate interaction energy and transition energy shifts - so far for homodimer   
    Einter,Eshift1,Eshift2,TrDip1,TrDip2,dipAE,dipA_E,dipBE,res=mol_polar.get_HeterodimerProperties_new(charge1_grnd,charge1_exct,charge2_grnd,charge2_exct,mol_Elstat,struc,index1,index2,0.0,0.0,eps,dAVA=dAVA,dBVB=dBVB,order=order,approx=approx)
    
    if verbose:
        with energy_units("1/cm"):
            print('        Total interaction energy:',Einter.value)
            print(ShortName,abs(Einter.value),Eshift1.value,Eshift2.value) 
            print("dipole:",np.linalg.norm(TrDip1),np.linalg.norm(TrDip2))
            print("dAVA:",dAVA*conversion_facs_energy["1/cm"],"dBVB:",dBVB*conversion_facs_energy["1/cm"])
    
    if MathOut:
        if not os.path.exists("Pictures"):
            os.makedirs("Pictures")
        Bonds = GuessBonds(mol_polar.coor)
        
        if CoarseGrain in ["plane","C","CF"]:
            at_type = ['C']*mol_polar.Nat
        elif CoarseGrain == "all_atom":
            at_type = struc.at_type.copy()
        
        mat_filename = "".join(['Pictures/Polar_',ShortName,'_AlphaE.nb'])
        params = {'TrPointCharge': mol_polar.charge,'AtDipole': dipAE,'rSphere_dip': 0.5,'rCylinder_dip':0.1}
        OutputMathematica(mat_filename,mol_polar.coor,Bonds,at_type,scaleDipole=50.0,**params)
        mat_filename = "".join(['Pictures/Polar_',ShortName,'_Alpha_E.nb'])
        params = {'TrPointCharge': mol_polar.charge,'AtDipole': dipA_E,'rSphere_dip': 0.5,'rCylinder_dip':0.1}
        OutputMathematica(mat_filename,mol_polar.coor,Bonds,at_type,scaleDipole=50.0,**params)
        mat_filename = "".join(['Pictures/Polar_',ShortName,'_BetaE.nb'])
        params = {'TrPointCharge': mol_polar.charge,'AtDipole': dipBE,'rSphere_dip': 0.5,'rCylinder_dip':0.1}
        OutputMathematica(mat_filename,mol_polar.coor,Bonds,at_type,scaleDipole=50.0,**params)


#            res["E_pol2_A(E)"] = PolarMat_AlphaE
#            res["E_pol2_A(-E)"] = PolarMat_Alpha_E
#            res["E_pol2_A_static"] = PolarMat_Alpha_st
#            res["E_pol2_B(E,E)"] = PolarMat_Beta
#            res["E_pol2_B(E,E)_scaled"] = PolarMat_Beta_scaled
#            res["E_pol2_A(E)_(trans,grnd)"] = PolarMat_Alpha_tr_gr
#            res["E_pol1_A_static"] = PolarMat_static_tr_gr_ex
#            res["E_elstat_1"] = ElstatMat_1
    
    if verbose:
        if not os.path.isfile("Temp.dat"):
            text = "                             pol2_A(E)      |       pol2_A(-E)      |        pol2_st        |    pol2_BEE_scaled    |   E_pol1-A(E)_tr_gr   |        E_pol1_st      |         pol1_BEE        |       sum_elstat      |"
            os.system("".join(['echo "',text,'" >> Temp.dat']))
            text = "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|"
            os.system("".join(['echo "',text,'" >> Temp.dat']))
       
        with energy_units("1/cm"):
            ii = 0
            text="{:21} {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.6f} {:10.6f} | {:10.6f} {:10.6f} |".format(
                    ShortName,res["E_pol2_A(E)"].value[ii,0],res["E_pol2_A(E)"].value[ii,1],res["E_pol2_A(-E)"].value[ii,0],res["E_pol2_A(-E)"].value[ii,1],
                    res["E_pol2_A_static"].value[ii,0],res["E_pol2_A_static"].value[ii,1],res["E_pol2_B(E,E)_scaled"].value[ii,0],
                    res["E_pol2_B(E,E)_scaled"].value[ii,1],res["E_pol2_A(E)_(trans,grnd)"].value[ii,0],res["E_pol2_A(E)_(trans,grnd)"].value[ii,1],
                    res["E_pol1_A_static"].value[ii,0],res["E_pol1_A_static"].value[ii,1],res["E_pol2_B(E,E)"].value[ii,0],res["E_pol2_B(E,E)"].value[ii,1],
                    res["E_elstat_1"].value[ii,0],res["E_elstat_1"].value[ii,1])
        
            os.system("".join(['echo "',text,'" >> Temp.dat']))
            ii = 1
            text="{:21} {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.3f} {:10.3f} | {:10.6f} {:10.6f} | {:10.6f} {:10.6f} |".format(
                    " ",res["E_pol2_A(E)"].value[ii,0],res["E_pol2_A(E)"].value[ii,1],res["E_pol2_A(-E)"].value[ii,0],res["E_pol2_A(-E)"].value[ii,1],
                    res["E_pol2_A_static"].value[ii,0],res["E_pol2_A_static"].value[ii,1],res["E_pol2_B(E,E)_scaled"].value[ii,0],
                    res["E_pol2_B(E,E)_scaled"].value[ii,1],res["E_pol2_A(E)_(trans,grnd)"].value[ii,0],res["E_pol2_A(E)_(trans,grnd)"].value[ii,1],
                    res["E_pol1_A_static"].value[ii,0],res["E_pol1_A_static"].value[ii,1],res["E_pol2_B(E,E)"].value[ii,0],res["E_pol2_B(E,E)"].value[ii,1],
                    res["E_elstat_1"].value[ii,0],res["E_elstat_1"].value[ii,1])
            
            os.system("".join(['echo "',text,'" >> Temp.dat']))
            text = "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|"
            os.system("".join(['echo "',text,' " >> Temp.dat']))

    return Einter, Eshift1, Eshift2, TrDip1, TrDip2


def TEST_Compare_SingleDef_FGprop(filenames,ShortName,index_all,AlphaE,Alpha_E,BetaE,VinterFG,FG_charges,ChargeType,order=1,verbose=False,approx=1.1,MathOut=False,CoarseGrain="plane",**kwargs):
    ''' Compare magnitude of individual terms in energy shift calculation for 
    defect in Fluorographene environment (so far only for first order of
    perturbation expansion -> order = 1)
    
    Parameters
    ----------
    filenames : dictionary
        Dictionary with information about all needed files which contains 
        nessesary information for transformig the system into Dielectric class
        and electrostatic calculations. Keys:
            
        * ``'1def_structure'``: xyz file with FG system with single defect 
          geometry and atom types
        * ``'charge_structure'``: xyz file with defect-like molecule geometry
          for which transition charges were calculated corresponding to first
          defect
        * ``'charge'``: file with transition charges for the defect 
          (from TrEsp charges fitting)
        * ``'charge_grnd'``: file with ground state charges for the defect 
          (from TrEsp charges fitting)
        * ``'charge_exct'``: file with excited state charges for the defect 
          (from TrEsp charges fitting)
          
    ShortName : string
        Short description of the system 
    index_all : list of integers (dimension 6)
        There are specified indexes neded for asignment of defect 
        atoms. First three indexes correspond to center and two main axes of 
        reference structure (structure which was used for charges calculation)
        and the last three indexes are corresponding atoms of the defect.
    AlphaE : numpy.array of real (dimension 2x2)
        Atomic polarizability Alpha(E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    Alpha_E : numpy.array of real (dimension 2x2)
        Atomic polarizability Alpha(-E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    BetaE : numpy.array of real (dimension 2x2)
        Atomic polarizability Beta(E,E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    VinterFG : real
        Difference in electrostatic interaction energy between interaction of
        excited C-F corse grained atom of fluorographene with all others
        fluorographene corse grained atoms in ground state and interaction of
        ground state C-F corse grained atom of fluorographene with all others
        fluorographene corse grained atoms in ground state. Units are ATOMIC 
        UNITS (Hartree)
    FG_charges : list of real (dimension 2)
        [charge on inner fluorographene atom, charge on borded fluorographe carbon]
    ChargeType : string
        Specifies which charges should be used for electrostatic calculations
        (ground and excited state charges) for defect atoms. Allowed types are:
        ``'qchem'``, ``'qchem_all'``, ``'AMBER'`` and ``'gaussian'``. 
        
        * ``'qchem'`` - charges calculated by fiting Q-Chem ESP on carbon
          atoms. 
        * ``'qchem_all'`` - charges calculated by fiting Q-Chem ESP on all
          atoms, only carbon charges are used and same charge is added to all 
          carbon atoms in order to have neutral molecule. 
        * ``'AMBER'`` - not yet fully implemented. 
        * ``'gaussian'`` - not yet fully implemented.
        
    order : integer (optional - init=80)
        Specify how many SCF steps shoudl be used in calculation  of induced
        dipoles - according to the used model it should be 2
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    approx : real (optional - init=1.1)
        Specifies which approximation should be used.
            
        * **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)` and 
          `Alpha(-E)`.
        * **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.
        * **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
          `Alpha(E)=Alpha(-E)`, however the second one is not condition 
    
    Returns
    --------
    Eshift : Energy class
        Transition energy shift for the defect due to the fluorographene
        environment calculated from structure with single defect. Units are
        energy managed
    TrDip : numpy array of real (dimension 3)
        Total transition dipole for the defect with environment effects 
        included calculated from structure with single defect (in ATOMIC UNITS)
        
    Notes
    --------
    By comparing QC calculations it was found that energy shift from structure 
    with two defects and with single defect is almost the same.
    '''    
    
    # read and prepare molecule
    mol_polar,index1,charge,struc=prepare_molecule_1Def(filenames,index_all,AlphaE,Alpha_E,BetaE,VinterFG,verbose=False,CoarseGrain=CoarseGrain,**kwargs)

    # calculate dAVA = <A|V|A>-<G|V|G>
    AditInfo={'Structure': struc,'index1': index1,'Output_exct': True}
    mol_Elstat,index,charge_grnd,charge_exct=ElStat_PrepareMolecule_1Def(filenames,index_all,FG_charges,ChargeType=ChargeType,verbose=False,**AditInfo)
    dAVA=mol_Elstat.get_EnergyShift()

    # Calculate interaction with ground state charges
    mol_Elstat.charge[index] = charge_grnd
    E_elst_grnd = mol_Elstat.get_EnergyShift()
    mol_Elstat.charge[index] = charge_exct - charge_grnd
    
    # Calculate interaction with excited state charges
    mol_Elstat.charge[index] = charge_exct
    E_elst_exct = mol_Elstat.get_EnergyShift()
    mol_Elstat.charge[index] = charge_exct - charge_grnd

    # Calculate interaction with transition density
    mol_Elstat.charge[index] = charge
    E_elst_trans = mol_Elstat.get_EnergyShift()
    mol_Elstat.charge[index] = charge_exct - charge_grnd

    # calculate transition energy shifts and transition dipole change      
    res_Energy, res_Pot, TrDip = mol_polar._TEST_Compare_SingleDefectProperties(charge,charge_grnd,charge_exct,struc,index1,dAVA=dAVA,order=order,approx=approx)

    charge_FG_grnd = mol_Elstat.charge.copy()
    charge_FG_grnd[index] = 0.0
    E_Pol1_env_static_ex_gr_FG = np.dot(charge_FG_grnd,res_Pot['Pol1-env_static_(exct-grnd)'])
    E_Pol2_env_static_ex_gr_FG = np.dot(charge_FG_grnd,res_Pot['Pol2-env_static_(exct-grnd)'])
    E_Pol1_env_BetaEE_ex_gr_FG = np.dot(charge_FG_grnd,res_Pot['Pol1-env_Beta(E,E)_(exct-grnd)'])
    E_Pol1_env_BetaEE_trans_FG = np.dot(charge_FG_grnd,res_Pot['Pol1-env_Beta(E,E)_(trans)'])
    E_Pol1_env_AlphaE_trans_FG = np.dot(charge_FG_grnd,res_Pot['Pol1-env_Alpha(E)_(trans)'])
    E_Pol1_env_Alpha_E_trans_FG = np.dot(charge_FG_grnd,res_Pot['Pol1-env_Alpha(-E)_(trans)'])
    E_Pol1_env_static_trans_FG = np.dot(charge_FG_grnd,res_Pot['Pol1-env_static_(trans)'])
    #E_Polar_AlphaE_gr_ex_FG = 0.0
    # pot_dipole_gr_ex = potential of induced dipoles induced by difference charges between ground and excited state (gr_charges - ex_charges)
    
    with energy_units("AU"):
        E_elst_trans = EnergyClass(E_elst_trans)
        E_elst_grnd = EnergyClass(E_elst_grnd)
        E_elst_exct = EnergyClass(E_elst_exct)
        E_Pol1_env_static_ex_gr_FG = EnergyClass(E_Pol1_env_static_ex_gr_FG)
        E_Pol2_env_static_ex_gr_FG = EnergyClass(E_Pol2_env_static_ex_gr_FG)
        E_Pol1_env_BetaEE_ex_gr_FG = EnergyClass(E_Pol1_env_BetaEE_ex_gr_FG)
        E_Pol1_env_BetaEE_trans_FG = EnergyClass(E_Pol1_env_BetaEE_trans_FG)
        E_Pol1_env_AlphaE_trans_FG = EnergyClass(E_Pol1_env_AlphaE_trans_FG)
        E_Pol1_env_Alpha_E_trans_FG = EnergyClass(E_Pol1_env_Alpha_E_trans_FG)
        E_Pol1_env_static_trans_FG = EnergyClass(E_Pol1_env_static_trans_FG)
    

    if MathOut:
        if not os.path.exists("Pictures"):
            os.makedirs("Pictures")
        Bonds = GuessBonds(mol_polar.coor)
        struc.guess_bonds()
        
        if CoarseGrain in ["plane","C","CF"]:
            at_type = ['C']*mol_polar.Nat
        elif CoarseGrain == "all_atom":
            at_type = struc.at_type.copy()
        
        mat_filename = "".join(['Pictures/Charge_',ShortName,'_Exct-Grnd.nb'])
        params = {'TrPointCharge': mol_Elstat.charge,'rSphere_dip': 0.5,'rCylinder_dip':0.1}
        OutputMathematica(mat_filename,mol_Elstat.coor,struc.bonds,struc.at_type,**params)
        
        mol_Elstat.charge[index] = charge
        mat_filename = "".join(['Pictures/Charge_',ShortName,'_Trans.nb'])
        params = {'TrPointCharge': mol_Elstat.charge,'rSphere_dip': 0.5,'rCylinder_dip':0.1}
        OutputMathematica(mat_filename,mol_Elstat.coor,struc.bonds,struc.at_type,**params)
        
#    res_Pot = {'Pol2-env_static_(exct-grnd)': pot2_dipole_ex_gr}
#    res_Pot['Pol1-env_static_(exct-grnd)'] = pot1_dipole_ex_gr
#    res_Pot['Pol1-env_Beta(E,E)_(exct-grnd)'] = pot1_dipole_betaEE_ex_gr
#    res_Pot['Pol1-env_Beta(E,E)_(trans)'] = pot1_dipole_betaEE_tr
#    res_Pot['Pol1-env_Alpha(E)_(trans)'] = pot1_dipole_AlphaE_tr
#    res_Pot['Pol1-env_Alpha(-E)_(trans)'] = pot1_dipole_Alpha_E_tr
#    res_Pot['Pol1-env_static_(trans)'] = pot1_dipole_static_tr
#  
#    res_Energy = {'dE_0-1': Eshift, 'dE_elstat(exct-grnd)': dAVA}
#    res_Energy['E_pol1_Alpha(E)'] = Polar1_AlphaE
#    res_Energy['E_pol2_Alpha(E)'] = Polar2_AlphaE
#    res_Energy['E_pol1_Alpha(-E)'] = Polar1_Alpha_E
#    res_Energy['E_pol2_Alpha(-E)'] = Polar2_Alpha_E
#    res_Energy['E_pol1_Beta(E,E)'] = Polar1_Beta_EE
#    res_Energy['E_pol1_static_(exct-grnd)'] = Polar1_static_ex_gr
#    res_Energy['E_pol2_static_(exct-grnd)'] = Polar2_static_ex_gr
#    res_Energy['E_pol1_Beta(E,E)_(exct-grnd)'] = Polar1_Beta_EE_ex_gr
#    res_Energy['E_pol1_static_(trans)_(exct)'] = Polar1_static_tr_ex
#    res_Energy['E_pol1_static_(trans)_(grnd)'] = Polar1_static_tr_gr
#    res_Energy['E_pol1_Alpha(E)_(trans)_(grnd)'] = Polar1_AlphaE_tr_gr
#    res_Energy['E_pol1_Alpha(-E)_(trans)_(exct)'] = Polar1_Alpha_E_tr_ex
#    res_Energy['E_pol1_Beta(E,E)_(trans)_(exct-grnd)'] = Polar1_Beta_EE_tr_ex_gr
#    

    res_Energy['E_elstat_trans'] = E_elst_trans
    res_Energy['E_pol1-env_static_(exct-grnd)'] = E_Pol1_env_static_ex_gr_FG
    res_Energy['E_pol2-env_static_(exct-grnd)'] = E_Pol2_env_static_ex_gr_FG
    res_Energy['E_pol1-env_Beta(E,E)_(exct-grnd)'] = E_Pol1_env_BetaEE_ex_gr_FG
    res_Energy['E_pol1-env_Beta(E,E)_(trans)'] = E_Pol1_env_BetaEE_trans_FG
    res_Energy['E_pol1-env_Alpha(E)_(trans)'] = E_Pol1_env_AlphaE_trans_FG
    res_Energy['E_pol1-env_Alpha(-E)_(trans)'] = E_Pol1_env_Alpha_E_trans_FG
    res_Energy['E_pol1-env_static_(trans)'] = E_Pol1_env_static_trans_FG
        
    # E_elst_grnd, E_elst_exct
    return res_Energy, TrDip


def Calc_SingleDef_FGprop_new(filenames,ShortName,index_all,E01,AlphaE,Alpha_E,BetaE,VinterFG,FG_charges,ChargeType,order=2,verbose=False,approx=1.1,MathOut=False,CoarseGrain="plane",**kwargs):
    ''' Compare magnitude of individual terms in energy shift calculation for 
    defect in Fluorographene environment (so far only for first order of
    perturbation expansion -> order = 1)
    
    Parameters
    ----------
    filenames : dictionary
        Dictionary with information about all needed files which contains 
        nessesary information for transformig the system into Dielectric class
        and electrostatic calculations. Keys:
            
        * ``'1def_structure'``: xyz file with FG system with single defect 
          geometry and atom types
        * ``'charge_structure'``: xyz file with defect-like molecule geometry
          for which transition charges were calculated corresponding to first
          defect
        * ``'charge'``: file with transition charges for the defect 
          (from TrEsp charges fitting)
        * ``'charge_grnd'``: file with ground state charges for the defect 
          (from TrEsp charges fitting)
        * ``'charge_exct'``: file with excited state charges for the defect 
          (from TrEsp charges fitting)
          
    ShortName : string
        Short description of the system 
    index_all : list of integers (dimension 6)
        There are specified indexes neded for asignment of defect 
        atoms. First three indexes correspond to center and two main axes of 
        reference structure (structure which was used for charges calculation)
        and the last three indexes are corresponding atoms of the defect.
    AlphaE : numpy.array of real (dimension 2x2)
        Atomic polarizability Alpha(E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    Alpha_E : numpy.array of real (dimension 2x2)
        Atomic polarizability Alpha(-E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    BetaE : numpy.array of real (dimension 2x2)
        Atomic polarizability Beta(E,E) for C-F corse grained atoms of 
        fluorographene in ATOMIC UNITS (Bohr^2 - because 2D)
    VinterFG : real
        Difference in electrostatic interaction energy between interaction of
        excited C-F corse grained atom of fluorographene with all others
        fluorographene corse grained atoms in ground state and interaction of
        ground state C-F corse grained atom of fluorographene with all others
        fluorographene corse grained atoms in ground state. Units are ATOMIC 
        UNITS (Hartree)
    FG_charges : list of real (dimension 2)
        [charge on inner fluorographene atom, charge on borded fluorographe carbon]
    ChargeType : string
        Specifies which charges should be used for electrostatic calculations
        (ground and excited state charges) for defect atoms. Allowed types are:
        ``'qchem'``, ``'qchem_all'``, ``'AMBER'`` and ``'gaussian'``. 
        
        * ``'qchem'`` - charges calculated by fiting Q-Chem ESP on carbon
          atoms. 
        * ``'qchem_all'`` - charges calculated by fiting Q-Chem ESP on all
          atoms, only carbon charges are used and same charge is added to all 
          carbon atoms in order to have neutral molecule. 
        * ``'AMBER'`` - not yet fully implemented. 
        * ``'gaussian'`` - not yet fully implemented.
        
    order : integer (optional - init=80)
        Specify how many SCF steps shoudl be used in calculation  of induced
        dipoles - according to the used model it should be 2
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    approx : real (optional - init=1.1)
        Specifies which approximation should be used.
            
        * **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)` and 
          `Alpha(-E)`.
        * **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.
        * **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
          `Alpha(E)=Alpha(-E)`, however the second one is not condition 
    
    Returns
    --------
    Eshift : Energy class
        Transition energy shift for the defect due to the fluorographene
        environment calculated from structure with single defect. Units are
        energy managed
    TrDip : numpy array of real (dimension 3)
        Total transition dipole for the defect with environment effects 
        included calculated from structure with single defect (in ATOMIC UNITS)
        
    Notes
    --------
    By comparing QC calculations it was found that energy shift from structure 
    with two defects and with single defect is almost the same.
    '''    
    
    # read and prepare molecule
    mol_polar,index1,charge,struc=prepare_molecule_1Def(filenames,index_all,AlphaE,Alpha_E,BetaE,VinterFG,verbose=False,CoarseGrain=CoarseGrain,**kwargs)

    # calculate dAVA = <A|V|A>-<G|V|G>
    AditInfo={'Structure': struc,'index1': index1,'Output_exct': True}
    mol_Elstat,index,charge_grnd,charge_exct=ElStat_PrepareMolecule_1Def(filenames,index_all,FG_charges,ChargeType=ChargeType,verbose=False,**AditInfo)
    dAVA=mol_Elstat.get_EnergyShift()
#    dAVA2, dAVA_R = mol_Elstat.get_EnergyShift_and_Derivative()
#    print(dAVA,dAVA2,dAVA-dAVA2)

    # calculate transition energy shifts and transition dipole change      
    # res_Energy, res_Pot, TrDip = mol_polar._TEST_Compare_SingleDefectProperties(charge,charge_grnd,charge_exct,struc,index1,dAVA=dAVA,order=order,approx=approx)
    Eshift,res_Energy,TrDip = mol_polar.get_SingleDefectProperties_new(charge_grnd, charge_exct, mol_Elstat, struc, index1, E01, dAVA=dAVA, order=order, approx=approx)

    if MathOut:
        if not os.path.exists("Pictures"):
            os.makedirs("Pictures")
        Bonds = GuessBonds(mol_polar.coor)
        struc.guess_bonds()
        
        if CoarseGrain in ["plane","C","CF"]:
            at_type = ['C']*mol_polar.Nat
        elif CoarseGrain == "all_atom":
            at_type = struc.at_type.copy()
        
        mat_filename = "".join(['Pictures/Charge_',ShortName,'_Exct-Grnd.nb'])
        params = {'TrPointCharge': mol_Elstat.charge,'rSphere_dip': 0.5,'rCylinder_dip':0.1}
        OutputMathematica(mat_filename,mol_Elstat.coor,struc.bonds,struc.at_type,**params)
        
        mol_Elstat.charge[index] = charge
        mat_filename = "".join(['Pictures/Charge_',ShortName,'_Trans.nb'])
        params = {'TrPointCharge': mol_Elstat.charge,'rSphere_dip': 0.5,'rCylinder_dip':0.1}
        OutputMathematica(mat_filename,mol_Elstat.coor,struc.bonds,struc.at_type,**params)

    return Eshift, TrDip


'''----------------------- TEST PART --------------------------------'''
if __name__=="__main__":

    print('                TESTS')
    print('-----------------------------------------')    
    
    ''' Test derivation of energy d/dR ApB '''

    # SETUP VERY SIMPLE SYSTEM OF TWO DEFECT ATOMS AND ONE ENVIRONMENT ATOM:
    coor=np.array([[-1.0,0.0,0.0],[0.0,0.0,0.0],[1.0,0.0,0.0]],dtype='f8')
    charge_pol=np.array([1.0,0.0,0.0],dtype='f8')
    dipole=np.zeros((len(coor),3),dtype='f8')
    AlphaE=np.array([np.zeros((3,3)),[[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,0.0]],np.zeros((3,3))],dtype='f8')
    
    pol_mol=Dielectric(coor,charge_pol,dipole,AlphaE,AlphaE,AlphaE,0.0)

    # definition of defect atoms and corresponding charges     
    charge=np.array([1.0],dtype='f8')
    index1=[0]
    index2=[2]
    
    res_general=pol_mol._dR_BpA(index1,index2,charge,'AlphaE')
    
    result=np.zeros((3,3),dtype='f8')
    result2=np.array([[-4.0,0.0,0.0],[0.0,0.0,0.0],[4.0,0.0,0.0]],dtype='f8').reshape(3*len(coor))
    R01=coor[1,:]-coor[0,:]
    RR01=np.sqrt(np.dot(R01,R01))
    R21=coor[1,:]-coor[2,:]
    RR21=np.sqrt(np.dot(R21,R21))
    dn=np.dot(AlphaE[1],R21/(RR21**3))
    result[0,:]=charge[0]*charge[0]*(3*np.dot(R01/(RR01**5),dn)*R01-1/(RR01**3)*dn)
    dn=np.dot(AlphaE[1],R01/(RR01**3))
    result[2,:]=charge[0]*charge[0]*(3*np.dot(R21/(RR21**5),dn)*R21-1/(RR21**3)*dn)

    if np.allclose(res_general,result2):
        print('Symm _dR_BpA simple system      ...    OK')
    else:
        print('Symm _dR_BpA simple system      ...    Error')
        print('     General result:   ',res_general)
        print('     Analytical result:',result2)
    
    result3=np.array([[8.0,0.0,0.0],[-8.0,0.0,0.0]],dtype='f8').reshape(6)
    pol_mol._swap_atoms(index1,index2)
    res_general=pol_mol._dR_BpA(index2,index2,charge,'AlphaE')
    if np.allclose(res_general[3:9],result3):
        print('Symm _dR_ApA simple system      ...    OK')
    else:
        print('Symm _dR_ApA simple system      ...    Error')
        print('     General result:   ',res_general)
        print('     Analytical result:',result3)
    
    
    # SETUP NON-SYMETRIC SIMPLE SYSTEM OF TWO DEFECT ATOMS AND ONE ENVIRONMENT ATOM:
    coor=np.array([[-1.0,0.0,0.0],[0.0,0.0,0.0],[1.0,2.0,0.0]],dtype='f8')
    charge_pol=np.array([1.0,0.0,0.0],dtype='f8')
    dipole=np.zeros((len(coor),3),dtype='f8')
    AlphaE=np.array([np.zeros((3,3)),[[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,0.0]],np.zeros((3,3))],dtype='f8')
    pol_mol=Dielectric(coor,charge_pol,dipole,AlphaE,AlphaE,AlphaE,0.0)

    # definition of defect atoms and corresponding charges     
    charge=np.array([1.0],dtype='f8')
    index1=[0]
    index2=[2]
    
    res_general=pol_mol._dR_BpA(index1,index2,charge,'AlphaE')
#    
#    result=np.zeros((3,3),dtype='f8')
    result2=np.array([[-4.0/np.sqrt(5)**3,4.0/np.sqrt(5)**3,0.0],
                      [6*(1/np.sqrt(5)**3-1/np.sqrt(5)**5),-4/np.sqrt(5)**3-12/np.sqrt(5)**5,0.0],
                      [6/np.sqrt(5)**5-2/np.sqrt(5)**3,12/np.sqrt(5)**5,0.0]],dtype='f8').reshape(3*len(coor))
    
    result=np.zeros((3,3),dtype='f8')
    R01=coor[1,:]-coor[0,:]
    RR01=np.sqrt(np.dot(R01,R01))
    R21=coor[1,:]-coor[2,:]
    RR21=np.sqrt(np.dot(R21,R21))
    dn=np.dot(AlphaE[1],R21/(RR21**3))
    result[0,:]=charge[0]*charge[0]*(3*np.dot(R01/(RR01**5),dn)*R01-1/(RR01**3)*dn)
    dn=np.dot(AlphaE[1],R01/(RR01**3))
    result[2,:]=charge[0]*charge[0]*(3*np.dot(R21/(RR21**5),dn)*R21-1/(RR21**3)*dn)
    #print(result2)
    #print(result)
    if np.allclose(res_general,result2):
        print('non-Symm _dR_BpA simple system  ...    OK')
    else:
        print('non-Symm _dR_BpA simple system  ...    Error')
        print('     General result:   ',res_general)
        print('     Analytical result:',result2)
    
    result3=np.array([[0.064,0.128,0.0],[-0.064,-0.128,0.0]],dtype='f8').reshape(6)
    pol_mol._swap_atoms(index1,index2)
    res_general=pol_mol._dR_BpA(index2,index2,charge,'AlphaE')
    
    if np.allclose(res_general[3:9],result3):
        print('non-Symm _dR_ApA simple system  ...    OK')
    else:
        print('non-Symm _dR_ApA simple system  ...    Error')
        print('     General result:   ',res_general)
        print('     Analytical result:',result3)
    
    # SETUP LITTLE BIT MORE COMPLICATED SYSTEM OF 2 DEFECT ATOMS AND 2ENVIRONMENT ATOMS
    for kk in range(2): 
        if kk==0:
            coor=np.array([[-2.0,0.0,0.0],[-2.0,-1.0,0.0],[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[2.0,1.0,0.0]],dtype='f8')
        else:
            coor=np.array([[-2.0,0.0,0.0],[-2.0,1.0,0.0],[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[2.0,1.0,0.0]],dtype='f8')
        charge_pol=np.array([1.0,-1.0,0.0,0.0,0.0,0.0],dtype='f8')
        dipole=np.zeros((len(coor),3),dtype='f8')
        AlphaE=np.array([np.zeros((3,3)),np.zeros((3,3)),
                         [[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,0.0]],
                         [[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,0.0]],
                         np.zeros((3,3)),np.zeros((3,3))],dtype='f8')
        pol_mol=Dielectric(coor,charge_pol,dipole,AlphaE,AlphaE,AlphaE,0.0)
    
        # definition of defect atoms and corresponding charges     
        charge=np.array([1.0,-1.0],dtype='f8')
        index1=[0,1]
        index2=[4,5]
        
        res_general=pol_mol._dR_BpA(index1,index2,charge,'AlphaE')
        
        if kk==0:
            # for coor[1]=[-2.0,-1.0,0.0]
            result2=np.array([[-0.1313271490,-0.04854981982,0.0],[0.04798957640,0.07411449339,0.0],
                              [0.0,0.0,0.0],[-0.04637925945,-0.08345754376,0.0],
                              [0.1005284061,0.08560623298,0.0],
                              [0.02918842589,-0.02771336278,0.0]],dtype='f8').reshape(3*len(coor))  
        else:
            # for coor[1]=[-2.0,1.0,0.0]
            result2=np.array([[-0.131327,-0.0485498,0.0],[0.126639,-0.0300095,0.0],
                          [0.0,0.0624526,0.0],[-0.0195464,0.138987,0.0],
                          [0.100528,-0.0856062,0.0],[-0.0762936,-0.037274,0.0]],dtype='f8').reshape(3*len(coor))
        if np.allclose(res_general,result2):
            print('non-Symm _dR_BpA system',kk+1,'      ...    OK')
        else:
            print('non-Symm _dR_BpA system',kk+1,'      ...    Error')
            print('     General result:   ',res_general)
            print('     Analytical result:',result2)

        if kk==1:
            res_general=pol_mol._dR_BpA(index1,index1,charge,'AlphaE')
            result3=np.array([[0.0759272,-0.0494062,0.0],[0.00288743,0.0479804,0.0],
                          [-0.0738948,0.0013901,0.0],[-0.00491991,0.00003574515217,0.0]],dtype='f8').reshape(12)
            if np.allclose(res_general[0:12],result3):
                print('non-Symm _dR_ApA system',kk+1,'      ...    OK')
            else:
                print('non-Symm _dR_ApA system',kk+1,'      ...    Error')
                print('     General result:   ',res_general)
                print('     Analytical result:',result3)
                
                
    ''' Test derivation of energy d/dR BppA '''
    # SETUP NON-SYMETRIC SIMPLE SYSTEM OF TWO DEFECT ATOMS AND TWO ENVIRONMENT ATOM:
    coor=np.array([[-1.0,0.0,0.0],[0.0,0.0,0.0],[0.0,1.0,0.0],[1.0,0.0,0.0]],dtype='f8')
    charge_pol=np.array([1.0,0.0,0.0,0.0],dtype='f8')
    dipole=np.zeros((len(coor),3),dtype='f8')
    AlphaE=np.array([np.zeros((3,3)),[[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,0.0]],
                     [[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,0.0]],np.zeros((3,3))],dtype='f8')
    pol_mol=Dielectric(coor,charge_pol,dipole,AlphaE,AlphaE,AlphaE,0.0)

    # definition of defect atoms and corresponding charges     
    charge=np.array([1.0],dtype='f8')
    index1=[0]
    index2=[3]
    
    res_general=pol_mol._dR_BppA(index1,index2,charge,'AlphaE')

    result2=np.array([[3.535533906,-0.7071067812,0.0],[0.0,14.14213562,0.0],
                      [0.0,-12.72792206,0.0],[-3.535533906,-0.7071067812,0.0],
                      ],dtype='f8').reshape(3*len(coor))

    if np.allclose(res_general,result2):
        print('non-Symm _dR_BppA simple system ...    OK')
    else:
        print('non-Symm _dR_BppA simple system ...    Error')
        print('     General result:   ',res_general)
        print('     Analytical result:',result2)
        
    res_general=pol_mol._dR_BppA(index1,index1,charge,'AlphaE')
    result3=np.array([[-7.071067812,-9.899494937,0.0],[-2.8284271247,-2.8284271247,0.0],
                      [9.899494937,12.72792206,0.0],
                      ],dtype='f8').reshape(9)
    if np.allclose(res_general[0:9],result3):
        print('non-Symm _dR_AppA simple system ...    OK')
    else:
        print('non-Symm _dR_AppA simple system ...    Error')
        print('     General result:   ',res_general[0:9])
        print('     Analytical result:',result3)
    