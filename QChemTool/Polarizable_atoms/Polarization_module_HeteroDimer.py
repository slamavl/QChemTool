# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:33:56 2017

@author: Vladislav Sláma
"""

import QuantumChem.classmolecule as clas
import QuantumChem.output as out
import QuantumChem.interaction as inter
import QuantumChem.positioningTools as pos
import General.constants as const
import QuantumChem.calc as calc
import numpy as np
import QuantumChem.read_mine as read
import Polarizable_atoms.Electrostatics_module as ElStatMol
from scipy.spatial.distance import pdist,squareform
from copy import deepcopy

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
    
    def __init__(self,coor,charge,dipole,AlphaE,Alpha_E,BetaEE,V):
        self.coor=np.copy(coor)
        self.polar={}
        self.polar['AlphaE']=AlphaE
        self.polar['Alpha_E']=Alpha_E
        self.polar['BetaEE']=BetaEE
        self.VinterFG=V
        self.charge=np.copy(charge)
        self.dipole=np.copy(dipole)
        self.Nat=len(coor)
    
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
                   'CD': [ZeroM,ZeroM,ZeroM],'C': [ZeroM,ZeroM,ZeroM]} 
        for key in list(kwargs.keys()):
            if key=='PolValues':
                PolValues=kwargs['PolValues']
        
        if self.Nat!=len(pol_type):
            raise IOError('Polarization type vector must have the same length as number of atoms')
        
        polar={}
        polar['AlphaE']=np.zeros((len(pol_type),3,3),dtype='f8')
        polar['Alpha_E']=np.zeros((len(pol_type),3,3),dtype='f8')
        polar['BetaEE']=np.zeros((len(pol_type),3,3),dtype='f8')
        for ii in range(len(pol_type)):
            polar['AlphaE'][ii,:,:]=PolValues[pol_type[ii]][0]
            polar['Alpha_E'][ii,:,:]=PolValues[pol_type[ii]][1]
            polar['BetaEE'][ii,:,:]=PolValues[pol_type[ii]][2]
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
            self.polar['AlphaE'][index1[ii],:,:],self.polar['AlphaE'][index2[ii],:,:] = self.polar['AlphaE'][index2[ii],:,:],self.polar['AlphaE'][index1[ii],:,:]
            self.polar['Alpha_E'][index1[ii],:,:],self.polar['Alpha_E'][index2[ii],:,:] = self.polar['Alpha_E'][index2[ii],:,:],self.polar['Alpha_E'][index1[ii],:,:]
            self.polar['BetaEE'][index1[ii],:,:],self.polar['BetaEE'][index2[ii],:,:] = self.polar['BetaEE'][index2[ii],:,:],self.polar['BetaEE'][index1[ii],:,:]
            
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
            import General.functions as func
            import General.Potential as pot
            # Test first order induced dipoles            
            self.dipole=np.zeros((self.Nat,3),dtype='f8')
            self.calc_dipoles_All('AlphaE',NN=1)
            if func.are_similar(P,self.dipole):
                print('First order dipoles are the same.')
            else:
                print('Problem with first order induced dipoles.')
            
            # test induced electric field
            Elfield=np.zeros(3,dtype='f8')
            for ii in range(3):
                Elfield[ii]=np.dot(-T[0,1,ii,:],P[1,:])
            print('Electric field at atom 0 induced by dipole at position 1 wT:',Elfield)
            Elfield=np.zeros(3,dtype='f8')
            Elfield=pot.ElField_dipole(P[1,:],R[0,1,:])
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
            
    def _dRcB_BpA(self,index,charge,typ,c,eps=1):
        ''' function which calculate derivation of interaction energy between defect
        A and defect B defined by index:
        d/dRc^{(B)}[Sum_{n} E^{(B)}(Rn).(1/2*Polarizability(n)).E^{(A)}(Rn)]
        
        Parameters
        ----------
        index : list or numpy.array of integer (dimension N_def_atoms)
            Atomic indexes of atoms which coresponds to defect B (defect with zero charges)
        charges : numpy array of real (dimension N_def_atoms)
            Vector of transition charges for every atom of defect B (listed in `index`)
        typ : str ('AlphaE','Alpha_E','BetaEE')
            Specifies which polarizability is used for calculation of induced
            atomic dipoles
        c : integer
            Atomic index specifying along which atom displacement should we calculate 
            derivation
        eps : real (optional - init=1.0)
            Relative dielectric polarizability of medium where the dipoles and 
            molecule is present ( by default vacuum with relative permitivity 1.0)

        Notes
        ----------
        '''
        
        # check if atom with index c is in defect B
        if c in index:
            c_indx=np.where(index==c)[0][0]
        else:
            raise IOError('Defined index c is not in defect B')
                
        
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
        
        # calculating derivation according to atom displacement from defect B
        Q=np.meshgrid(self.charge,self.charge)[0]   # in columns same charges
        ELF=np.zeros((self.Nat,self.Nat,3),dtype='f8')
        
        for jj in range(3):
            ELF[:,:,jj]=(Q/RR3)*R[:,:,jj]   # ELF[i,j,:]  is electric field at position i generated by atom j 
        for ii in range(self.Nat):
            ELF[ii,ii,:]=np.zeros(3,dtype='f8')
            
        ELFV=np.array(np.sum(ELF,axis=1),dtype='f8')             # ELFV[i,:]   is electric field at position of atom i
        for ii in range(self.Nat):
            P[ii,:]=np.dot(self.polar[typ][ii],ELFV[ii,:])
        
        ELFV=np.zeros((self.Nat,3),dtype='f8')
        for ii in range(3):
            for jj in range(3):
                ELFV[:,ii]+=np.dot(T[:,:,ii,jj],P[:,jj])
        
        res=charge[c_indx]*ELFV[c,:]
        
        return res
        
    def _dR_BpA(self,index1,index2,charge,typ,eps=1):
        ''' function which calculate derivation of interaction energy between defect
        A and defect B defined by index:
        d/dR[Sum_{n} E^{(B)}(Rn).(1/2*Polarizability(n)).E^{(A)}(Rn)]
        
        Parameters
        ----------
        index1 : list or numpy.array of integer (dimension N_def_atoms)
            Atomic indexes of atoms which coresponds to first defect (defect with zero charges)
        index2 : list or numpy.array of integer (dimension N_def_atoms)
            Atomic indexes of atoms which coresponds to second defect (defect with zero charges)
        charges : numpy array of real (dimension N_def_atoms)
            Vector of transition charges for every atom of defect B (listed in `index`)
        typ : str ('AlphaE','Alpha_E','BetaEE')
            Specifies which polarizability is used for calculation of induced
            atomic dipoles
        eps : real (optional - init=1.0)
            Relative dielectric polarizability of medium where the dipoles and 
            molecule is present ( by default vacuum with relative permitivity 1.0)

        Notes
        ----------
        For calculation of derivation of ApA use `_dR_BpA(index1,index1,charge,typ,eps=1)`
        where charges in molecule Dielectric class have to be nonzero for defect with
        index1. If not first swap the the defects with `swap_atoms(index1,index2)`
        and than use as described earlier. The same is true for the second defect,
        where you only replace index1 for index2.
        '''
        
        res=np.zeros((self.Nat,3),dtype='f8')
        
        # calculation of tensors with interatomic distances
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
        
        # calculating derivation according to atom displacement from defect B
        Q=np.meshgrid(self.charge,self.charge)[0]   # in columns same charges
        ELF=np.zeros((self.Nat,self.Nat,3),dtype='f8')
        
        for jj in range(3):
            ELF[:,:,jj]=(Q/RR3)*R[:,:,jj]   # ELF[i,j,:]  is electric field at position i generated by atom j 
        for ii in range(self.Nat):
            ELF[ii,ii,:]=np.zeros(3,dtype='f8')
            
        ELFV=np.array(np.sum(ELF,axis=1),dtype='f8')             # ELFV[i,:]   is electric field at position of atom i
        for ii in range(self.Nat):
            P[ii,:]=np.dot(self.polar[typ][ii],ELFV[ii,:])
        
        ELFV=np.zeros((self.Nat,3),dtype='f8')
        for ii in range(3):
            for jj in range(3):
                ELFV[:,ii]+=np.dot(T[:,:,ii,jj],P[:,jj])
        
        for ii in range(len(index2)):
            res[index2[ii],:]-=charge[ii]*ELFV[index2[ii],:]
            
        # calculating derivation with respect to displacement of environment atom
        for ii in range(self.Nat):
            if not (ii in index1 or ii in index2):
                for jj in range(len(index2)):
                    res[ii,:]+=charge[jj]*np.dot(T[index2[jj],ii,:,:],P[ii,:])
        
        # swap porarization parameters from defect A to defect B
        self.swap_atoms(index1,index2)
        
        # calculating derivation according to atom displacement from defect A
        Q=np.meshgrid(self.charge,self.charge)[0]   # in columns same charges
        ELF=np.zeros((self.Nat,self.Nat,3),dtype='f8')
        
        for jj in range(3):
            ELF[:,:,jj]=(Q/RR3)*R[:,:,jj]   # ELF[i,j,:]  is electric field at position i generated by atom j 
        for ii in range(self.Nat):
            ELF[ii,ii,:]=np.zeros(3,dtype='f8')
            
        ELFV=np.array(np.sum(ELF,axis=1),dtype='f8')             # ELFV[i,:]   is electric field at position of atom i
        for ii in range(self.Nat):
            P[ii,:]=np.dot(self.polar[typ][ii],ELFV[ii,:])
        
        ELFV=np.zeros((self.Nat,3),dtype='f8')
        for ii in range(3):
            for jj in range(3):
                ELFV[:,ii]+=np.dot(T[:,:,ii,jj],P[:,jj])
        
        for ii in range(len(index1)):
            res[index1[ii],:]-=charge[ii]*ELFV[index1[ii],:]
            
        # calculating derivation with respect to displacement of environment atom
        for ii in range(self.Nat):
            if not (ii in index1 or ii in index2):
                for jj in range(len(index1)):
                    res[ii,:]+=charge[jj]*np.dot(T[index1[jj],ii,:,:],P[ii,:])

        # swap porarization parameters back to original position
        self.swap_atoms(index1,index2)
        
        
        return res.reshape(3*self.Nat)
    
    def _dR_BppA(self,index1,index2,charge,typ,eps=1):
        ''' function which calculate derivation of second order interaction energy
        between defect A and defect B defined by index1 resp. index2:
        d/dR[Sum_{n} E^{(B)}(Rn).(1/2*Polarizability(n)). Sum_{n'} T(Rn-Rn').(1/2*Polarizability(n')).E^{(A)}(Rn)]
        
        Parameters
        ----------
        index1 : list or numpy.array of integer (dimension N_def_atoms)
            Atomic indexes of atoms which coresponds to first defect (defect with zero charges)
        index2 : list or numpy.array of integer (dimension N_def_atoms)
            Atomic indexes of atoms which coresponds to second defect (defect with zero charges)
        charges : numpy array of real (dimension N_def_atoms)
            Vector of transition charges for every atom of defect B (listed in `index`)
        typ : str ('AlphaE','Alpha_E','BetaEE')
            Specifies which polarizability is used for calculation of induced
            atomic dipoles
        eps : real (optional - init=1.0)
            Relative dielectric polarizability of medium where the dipoles and 
            molecule is present ( by default vacuum with relative permitivity 1.0)

        Notes
        ----------
        '''
        
        res=np.zeros((self.Nat,3),dtype='f8')
        
        # calculation of tensors with interatomic distances
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
        RR7=np.power(RR,7)
        
        # definition of T tensor
        T=np.zeros((self.Nat,self.Nat,3,3),dtype='f8') # mutual distance vectors
        for ii in range(3):
            T[:,:,ii,ii]=1/RR3[:,:]-3*np.power(R[:,:,ii],2)/RR5
            for jj in range(ii+1,3):
                T[:,:,ii,jj] = -3*R[:,:,ii]*R[:,:,jj]/RR5
                T[:,:,jj,ii] = T[:,:,ii,jj]
        for ii in range(self.Nat):
            T[ii,ii,:,:]=0.0        # no self interaction of atom i with atom i
        
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
        
        # calculating derivation according to atom displacement from defect B
        Q=np.meshgrid(self.charge,self.charge)[0]   # in columns same charges
        ELF=np.zeros((self.Nat,self.Nat,3),dtype='f8')
        
        for jj in range(3):
            ELF[:,:,jj]=(Q/RR3)*R[:,:,jj]   # ELF[i,j,:]  is electric field at position i generated by atom j 
        for ii in range(self.Nat):
            ELF[ii,ii,:]=np.zeros(3,dtype='f8')
            
        ELFV=np.array(np.sum(ELF,axis=1),dtype='f8')             # ELFV[i,:]   is electric field at position of atom i
        PA=np.zeros((self.Nat,3),dtype='f8')
        for ii in range(self.Nat):
            PA[ii,:]=np.dot(self.polar[typ][ii],ELFV[ii,:])


        for rep in range(2):
            P=np.zeros((self.Nat,3),dtype='f8')
            for ii in range(self.Nat):
                P[ii,:]=np.dot(self.polar[typ][ii],ELFV[ii,:])
            
            ELFV=np.zeros((self.Nat,3),dtype='f8')
            for ii in range(3):
                for jj in range(3):
                    ELFV[:,ii]+=np.dot(T[:,:,ii,jj],P[:,jj])
                
        for ii in range(len(index2)):
            res[index2[ii],:]+=charge[ii]*ELFV[index2[ii],:]
            
        # calculating derivation with respect to displacement of environment atom
        for ii in range(self.Nat):
            if not (ii in index1 or ii in index2):
                for jj in range(len(index2)):
                    res[ii,:]-=charge[jj]*np.dot(T[index2[jj],ii,:,:],P[ii,:])
        # + contribution from S tensor
        
        # swap porarization parameters from defect A to defect B
        self.swap_atoms(index1,index2)
        
        # calculating derivation according to atom displacement from defect A
        Q=np.meshgrid(self.charge,self.charge)[0]   # in columns same charges
        ELF=np.zeros((self.Nat,self.Nat,3),dtype='f8')
        
        for jj in range(3):
            ELF[:,:,jj]=(Q/RR3)*R[:,:,jj]   # ELF[i,j,:]  is electric field at position i generated by atom j 
        for ii in range(self.Nat):
            ELF[ii,ii,:]=np.zeros(3,dtype='f8')
            
        ELFV=np.array(np.sum(ELF,axis=1),dtype='f8')             # ELFV[i,:]   is electric field at position of atom i
        PB=np.zeros((self.Nat,3),dtype='f8')
        for ii in range(self.Nat):
            PB[ii,:]=np.dot(self.polar[typ][ii],ELFV[ii,:])        
        
        for rep in range(2):
            P=np.zeros((self.Nat,3),dtype='f8')
            for ii in range(self.Nat):
                P[ii,:]=np.dot(self.polar[typ][ii],ELFV[ii,:])
            
            ELFV=np.zeros((self.Nat,3),dtype='f8')
            for ii in range(3):
                for jj in range(3):
                    ELFV[:,ii]+=np.dot(T[:,:,ii,jj],P[:,jj])                
        
        for ii in range(len(index1)):
            res[index1[ii],:]+=charge[ii]*ELFV[index1[ii],:]
        
        # calculating derivation with respect to displacement of environment atom
        for ii in range(self.Nat):
            if not (ii in index1 or ii in index2):
                for jj in range(len(index1)):
                    res[ii,:]-=charge[jj]*np.dot(T[index1[jj],ii,:,:],P[ii,:])
        
        # + contribution from S tensor
        for nn in range(self.Nat):
            for ii in range(3):
                for kk in range(3):
                    res[nn,:]+=3*PB[nn,ii]*np.dot(S[nn,:,ii,:,kk].T,PA[:,kk])
                    res[nn,:]+=3*PA[nn,ii]*np.dot(S[nn,:,ii,:,kk].T,PB[:,kk])
                    
        # swap porarization parameters back to original position
        self.swap_atoms(index1,index2)
        
        
        return res.reshape(3*self.Nat)
        
    
    def calc_dipoles_All(self,typ,Estatic=np.zeros(3,dtype='f8'),NN=60,eps=1,debug=False):
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
                
            
    def get_interaction_energy(self,index,charge=None,debug=False):
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
        import Program.General.Potential as pot
        
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
        pot_charge=pot.potential_charge(AllCharge,R)
        pot_dipole=pot.potential_dipole(AllDipole,R)

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
                        potential_charge_test[jj]+=pot.potential_charge(AllCharge[ii],R)
                        potential_dipole_test[jj]+=pot.potential_dipole(AllDipole[ii],R)
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
                        potential+=pot.potential_charge(AllCharge[ii],R)
                        potential+=pot.potential_dipole(AllDipole[ii],R)
                InterE+=potential*charge[jj]
            
            if np.allclose(InterE,np.dot(charge,pot_charge+pot_dipole)):
                print('Interaction energy is calculated correctly')
            else:
                raise Warning('Interaction energy for both methods is different')
                
                
        InterE = np.dot(charge, pot_charge+pot_dipole)
        return InterE
            
    def get_selfinteraction_energy(self,debug=False):
        ''' Calculates interaction energy between induced dipoles by chromophore
        transition charges and transition charges of the same chromophore
        
        Returns
        -------
        InterE : real
            Interaction energies in atomic units (Hartree) multiplied by (-1)
            correspond to Electric_field_of_TrCharges.Induced_dipole
        
        Notes
        -------
        **By definition it is not an interaction energy but interaction energy 
        with opposite sign**
            
        
        '''
        import Program.General.Potential as pot
            
        
        # coppy charges and assign zero charges to those in index
        charge=[]
        charge_coor=[]
        dipole=[]
        dipole_coor=[]
        for ii in range(self.Nat):
            if self.charge[ii]!=0.0:
                charge.append(self.charge[ii])
                charge_coor.append(self.coor[ii])
            elif self.dipole[ii,0]!=0.0 or self.dipole[ii,1]!=0.0 or self.dipole[ii,2]!=0.0:
                dipole.append(self.dipole[ii])
                dipole_coor.append(self.coor[ii])
                
        charge=np.array(charge,dtype='f8')
        charge_coor=np.array(charge_coor,dtype='f8')
        dipole=np.array(dipole,dtype='f8')
        dipole_coor=np.array(dipole_coor,dtype='f8')
        if debug:
            print('Charges:')
            print(charge)
            print('Dipoles self-inter:')
            print(dipole)
        
        if debug:
            print('Charge coordinates')
            print(charge_coor.shape)
            print(charge_coor)
            print('Charges:')
            print(charge)
        
        if not charge.any():
            return 0.0          # If all charges are zero interaction is also zero
        if not dipole.any():
            print("All induced dipoles are zero - check if you calculating everything correctly")
            return 0.0          # If all dipoles are zero interaction is zero
        
        rr = np.tile(dipole_coor,(charge_coor.shape[0],1,1))   
        rr = np.swapaxes(rr,0,1)                               # dipole coordinate
        R = np.tile(charge_coor,(dipole_coor.shape[0],1,1))    # charge coordinate
        R = R-rr                                               # R[ii,jj,:]=charge_coor[jj]-dipole_coor[ii]
        
# TODO: There is no posibility to have charge and dipole on same atom (correct this) - so far no possibility to have zero R
        pot_dipole = pot.potential_dipole(dipole, R)
        InterE = -np.dot(charge, pot_dipole)
        
        if debug:
            #calculate interaction energy
            InterE2=0.0
            for jj in range(len(charge)):
                potential=0.0
                for ii in range(len(dipole)):
                        R=charge_coor[jj]-dipole_coor[ii]
                        potential+=pot.potential_dipole(dipole[ii],R)
                InterE2-=potential*charge[jj]   
                # minus is here because we dont want to calculate interaction energy
                # but interaction of electric field of transition charges with induced
                # dipoles and this is exactly - interaction energy between transition
                # charge and dipole
                
                if np.allclose(InterE,InterE2):
                    print('Selfinteraction energy is calculated correctly')
                else:
                    raise Warning('Selfinteraction energy for both methods is different')
            
        return InterE
    
    
    def fill_Polar_matrix(self,index1,index2,typ='AlphaE',order=80,debug=False):
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
        self.calc_dipoles_All(typ,NN=order,eps=1,debug=False)
        self.charge[defA_indx]=defA_charge
        PolMAT[1,1] = self.get_interaction_energy(defB_indx,charge=defB_charge,debug=False) - E_TrEsp
        PolMAT[0,1] = self.get_interaction_energy(defA_indx,charge=defA_charge,debug=False) - E_TrEsp
        PolMAT[1,0] = PolMAT[0,1]
        self.dipole=np.zeros((self.Nat,3),dtype='f8')
        
        # Polarization by molecule A
        self.charge[defB_indx]=0.0
        self.calc_dipoles_All(typ,NN=order,eps=1,debug=False)
        self.charge[defB_indx]=defB_charge
        PolMAT[0,0] = self.get_interaction_energy(defA_indx,charge=defA_charge,debug=False) - E_TrEsp
        if debug:
            print(PolMAT*const.HaToInvcm)
            if np.isclose(self.get_interaction_energy(defB_indx,charge=defB_charge,debug=False)-E_TrEsp,PolMAT[1,0]):
                print('ApB = BpA')
            else:
                raise Warning('ApB != BpA')
        self.dipole=np.zeros((self.Nat,3),dtype='f8')
        
        if debug:
            print(PolMAT*const.HaToInvcm)
        
        if typ=='AlphaE' or typ=='BetaEE':
            return PolMAT
        elif typ=='Alpha_E':
            PolMAT[[0,1],[0,1]] = PolMAT[[1,0],[1,0]]   # Swap AlphaMAT[0,0] with AlphaMAT[1,1]
            return PolMAT
    
    def get_TrEsp_Eng(self, index1, index2):
        defA_coor = self.coor[index1]
        defB_coor = self.coor[index2]
        defA_charge = self.charge[index1] 
        defB_charge = self.charge[index2]
        E_TrEsp = inter.charge_charge(defA_coor,defA_charge,defB_coor,defB_charge)[0]
        
        return E_TrEsp # in hartree
    
    def get_InteractionEng(self, index1, index2, Eng1, Eng2, dAVA=0.0, dBVB=0.0, order=80, approx=1.1):
        '''
        
        dAVA = <A|V|A> - <G|V|G>
        dBVB = <B|V|B> - <G|V|G>
        '''

        # Get TrEsp interaction energy
        E_TrEsp = self.get_TrEsp_Eng(index1, index2)
        
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
        

        PolarMat=np.zeros((2,2),dtype='f8')
        if approx==1.1:
            # Fill polarization matrix
            PolarMat += self.fill_Polar_matrix(index1,index2,typ='AlphaE',order=order)
            PolarMat += self.fill_Polar_matrix(index1,index2,typ='Alpha_E',order=order)
            BetaMat = self.fill_Polar_matrix(index1,index2,typ='BetaEE',order=order//2)
            PolarMat += BetaMat*(dAVA/2 + dBVB/2 - self.VinterFG)
            
            # Calculate interaction energies
            C1 = Coeff.T[0]
            E1 = Energy[0] + np.dot(C1, np.dot(PolarMat - d_esp*BetaMat, C1.T))
            C2 = Coeff.T[1]
            E2 = Energy[1] + np.dot(C2, np.dot(PolarMat + d_esp*BetaMat, C2.T))
            
            J_inter = np.sqrt( (E2 - E1)**2 - (Eng2 - Eng1)**2 )/2*np.sign(E_TrEsp)
            
            return J_inter
        else:
            raise IOError('Unsupported approximation')
        
    
    def get_TrDip(self,*args,output_dipoles=False,order=80,approx=1.1):
        ''' Function for calculation  of transition dipole moment for chromophore
        embeded in polarizable atom environment
        
        Parameters
        ----------
        *args : real (optional)
            Diference in electrostatic interaction energy between ground and 
            excited state in ATOMIC UNITS (DE). If not defined it is assumed to 
            be zero. DE=<A|V|A>-<G|V|G>
        output_dipoles : logical (optional - init=False)
            If atomic dipoles should be outputed or not. Atomic dipoles are 
            outputed as `AtDip_Alpha(E)+AtDip_Alpha(-E)-self.VinterFG*AtDip_Beta(E,E)
        order : integer (optional - init=80)
            Specify how many SCF steps shoudl be used in calculation  of induced dipoles
        approx : real (optional - init=1.2)
            Specifies which approximation should be used.
            
            **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)`.
            With this approximation diference in electrostatic interaction energy
            between ground and excited state in ATOMIC UNITS (DE) has to be imputed
            as `*args`
            
            **Approximation 1.1.2**: Approximation 1.1 + neglecting difference
            in electrostatic interaction between ground and excited state
            (imputed as approximation 1.1 but no electrostatic interaction energy
            diference - DE is defiend)
                
            **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.
            With this apprximation diference in electrostatic interaction energy
            between ground and excited state in ATOMIC UNITS (DE) has to be imputed
            as `*args`
            
            **Approximation 1.2.2**:  Approximation 1.2 + neglecting difference
            in electrostatic interaction between ground and excited state
            (imputed as approximation 1.2 but no electrostatic interaction energy
            diference - DE is defiend)
            
            **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
            `Alpha(E)=Alpha(-E)`, however the second one is not condition
            
            **Approximation MMpol**: Dipole will be calculated as a original dipole
            plus full polarization of the environmnet.

        Returns
        -------
        dipole : numpy.array of real (dimension 3)
            Transition dipole including the effects from interaction with environment
            in ATOMIC UNITS (e*Bohr)
        AtDipoles : numpy.array of real (dimension Natoms x 3) (optional)
            Induced atomic dipoles defined as: 
            `AtDip_Alpha(E)+AtDip_Alpha(-E)-self.VinterFG*AtDip_Beta(E,E)
            in ATOMIC UNITS (e*Bohr)
        
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
        **Approximation 1.1.2:**
            dip_fin = dip - Vinter*Beta(E,E)*El_field_TrCharge + dip_init(1-1/4*Ind_dip_Beta(E,E)*El_field_TrCharge)
        **Approximation 1.2:**
            dip_fin = dip - (Vinter-DE)*Beta(E,E)*El_field_TrCharge + dip_init     
        **Approximation 1.2.2:**
            dip_fin = dip - Vinter*Beta(E,E)*El_field_TrCharge + dip_init
        **Approximation 1.3:**
            dip_fin = dip - 2*Vinter*Beta(E,E)*El_field_TrCharge + dip_init
        
        '''
        
        if approx==1.3:
            if not np.array_equal(self.polar['AlphaE'],self.polar['Alpha_E']):
                raise Warning('For calculation with Approximation 1.3 Alpha(E) should be equal Alpha(-E)')
        
        if approx==1.1:
            if not np.array_equal(np.zeros((len(self.polar['Alpha_E']),3,3),dtype='f8'),self.polar['Alpha_E']):
                print('For calculation with Approximation 1.1 Alpha(-E) should be equal to zero')
        
        is_elstat=False
        if len(args)==1:
            DE=args[0]
            is_elstat=True

        use_alpha_instead_alphahalf=False
        if type(approx)==str and order==2:
            if 'MMpol' in approx:
                use_alpha_instead_alphahalf=True

        # For MMpol approximation we have to use alpha instead alpha/2 and resulting induced dipoles 
        # have to be devided by 2. This way we correct the second term in perturbation expansion
        if use_alpha_instead_alphahalf:
            self.polar['AlphaE']=self.polar['AlphaE']*2
            self.polar['Alpha_E']=self.polar['Alpha_E']*2


        # reset iduced dipoles to zero        
        self.dipole=np.zeros((self.Nat,3),dtype='f8')        
            
        # calculate induced dipoles with polarizability AlphaE for rescaled charges
        self.calc_dipoles_All('AlphaE',NN=order)
        AtDipoles1=np.copy(self.dipole)
        
        # reset iduced dipoles to zero        
        self.dipole=np.zeros((self.Nat,3),dtype='f8')

        if not (approx=='MMpol' and order>2):
            # if we calculate with MMpol procedure we use only one polarizability matrix and therefore doesn't have to be calculated            
            
            # reset iduced dipoles to zero        
            self.dipole=np.zeros((self.Nat,3),dtype='f8')
            
            # calculate induced dipoles with polarizability Alpha_E for rescaled charges
            self.calc_dipoles_All('Alpha_E',NN=order)
            AtDipoles2=np.copy(self.dipole)
            
            # reset iduced dipoles to zero        
            self.dipole=np.zeros((self.Nat,3),dtype='f8')
            
            if use_alpha_instead_alphahalf:
                self.polar['AlphaE']=self.polar['AlphaE']/2
                self.polar['Alpha_E']=self.polar['Alpha_E']/2
                AtDipoles1=AtDipoles1/2
                AtDipoles2=AtDipoles2/2
            
            # calculate induced dipoles with polarizability Beta for rescaled charges
            #self.calc_dipoles_All('BetaEE',NN=order//2)
            if order>2:
                self.calc_dipoles_All('BetaEE',NN=1)
            else:
                self.calc_dipoles_All('BetaEE',NN=order//2)
            AtDipolesBeta=np.copy(self.dipole)
        
        # calculate transition dipole:
        dipole=np.zeros(3,dtype='f8')
        for ii in range(self.Nat):
            dipole+=self.coor[ii,:]*self.charge[ii]
        dipole_tmp=np.copy(dipole)
        dipole+=np.sum(AtDipoles1,axis=0)
        if not (approx=='MMpol' and order>2):
            dipole+=np.sum(AtDipoles2,axis=0)
        
        # term with Beta polarizability
        if approx==1.1 or approx=='MMpol_1.1':
            dipole-=self.VinterFG*np.sum(AtDipolesBeta,axis=0) - dipole_tmp*self.get_selfinteraction_energy()/4
            if is_elstat:
                dipole+=DE*np.sum(AtDipolesBeta,axis=0)
        if approx==1.2 or approx=='MMpol_1.2':
            dipole-=self.VinterFG*np.sum(AtDipolesBeta,axis=0)
            if is_elstat:
                dipole+=DE*np.sum(AtDipolesBeta,axis=0)
        elif approx==1.3 or approx=='MMpol_1.3':
            dipole-=2*self.VinterFG*np.sum(AtDipolesBeta,axis=0)
        
        # reset iduced dipoles to zero        
        self.dipole=np.zeros((self.Nat,3),dtype='f8')
        
        if output_dipoles:
            if approx=='MMpol' and order>2:
                return dipole,AtDipoles1
            elif approx=='MMpol':
                return dipole,AtDipoles1+AtDipoles2
            else:
                return dipole,AtDipoles1+AtDipoles2-self.VinterFG*AtDipolesBeta
        else:
            return dipole
            
    
    def calculate_EnergyShift(self,index,charge,*args,order=80,output_dipoles=False,approx=1.1):
        ''' Function for calculation  of transition energy shift for chromophore
        embeded in polarizable atom environment
        
        Parameters
        ----------
        **index and charge** : Not used (useful only for structure with more than one defect)
        
        *args : real (optional)
            Diference in electrostatic interaction energy between ground and 
            excited state in ATOMIC UNITS (DE). If not defined it is assumed to 
            be zero. DE=<A|V|A>-<G|V|G>
        order : integer (optional - init=80)
            Specify how many SCF steps shoudl be used in calculation  of induced dipoles
        output_dipoles : logical (optional - init=False)
            If atomic dipoles should be outputed or not. Atomic dipoles are 
            outputed as `AtDip_Alpha(E)+AtDip_Alpha(-E)-self.VinterFG*AtDip_Beta(E,E)
        approx : real (optional - init=1.2)
            Specifies which approximation should be used.
            
            **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)`.
            With this approximation diference in electrostatic interaction energy
            between ground and excited state in ATOMIC UNITS (DE) has to be imputed
            as `*args`
            
            **Approximation 1.1.2**: Approximation 1.1 + neglecting difference
            in electrostatic interaction between ground and excited state
            (imputed as approximation 1.1 but no electrostatic interaction energy
            diference - DE is defiend)
            
            **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.
            With this apprximation diference in electrostatic interaction energy
            between ground and excited state in ATOMIC UNITS (DE) has to be imputed
            as `*args`
            
            **Approximation 1.2.2**:  Approximation 1.2 + neglecting difference
            in electrostatic interaction between ground and excited state
            (imputed as approximation 1.2 but no electrostatic interaction energy
            diference - DE is defiend)
            
            **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
            `Alpha(E)=Alpha(-E)`, however the second one is not condition 
            
            **Approximation MMpol**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
            `Alpha(E)=Alpha(-E)`, however the second one is not condition
        
        Returns
        -------
        Eshift : real
            Excitation energy shift in ATOMIC UNITS (Hartree) caused by the 
            interaction of molecule with polarizable atom environment
        AtDipoles : numpy.array of real (dimension Natoms x 3) (optional)
            Induced atomic dipoles defined as: 
            `AtDip_Alpha(E)+AtDip_Alpha(-E)-self.VinterFG*AtDip_Beta(E,E)`
            in ATOMIC UNITS (e*Bohr)
        
        **Neglecting `tilde{Beta(E)}` is not valid approximation. It should be
        better to neglect Beta(E,-E) to be consistent with approximation for 
        interaction energy**
        
        Notes
        ----------
        E = -Ind_dip_Alpha(E)*El_field_TrCharge + Ind_dip_Alpha(-E)*El_field_TrCharge 
        Then final energy shift E_fin of molecule embeded in environment is calculated
        according to the approximation:
        
        *Approximation 1.1:**
            Exactly the same as Approximation 1.2
        *Approximation 1.1.2:**
            Exactly the same as Approximation 1.2.2
        **Approximation 1.2:**
            E_fin = E + DE + (Vinter-DE)*Ind_dip_Beta(E,E)*El_field_TrCharge      
        **Approximation 1.2.2:**
            E_fin = E + Vinter*Ind_dip_Beta(E,E)*El_field_TrCharge
        **Approximation 1.3:**
            E_fin = E + DE*(1-2*Ind_dip_Beta(E,E)*El_field_TrCharge)
        
        '''

        if approx==1.3:
            if not np.array_equal(self.polar['AlphaE'],self.polar['Alpha_E']):
                raise Warning('For calculation with Approximation 1.3 Alpha(E) should be equal Alpha(-E)')     
        
        if approx==1.1:
            if not np.array_equal(np.zeros((len(self.polar['Alpha_E']),3,3),dtype='f8'),self.polar['Alpha_E']):
                print('For calculation with Approximation 1.1 Alpha(-E) should be equal to zero')
        
        is_elstat=False
        if len(args)==1:
            DE=args[0]
            is_elstat=True
            
        use_alpha_instead_alphahalf=False
        if type(approx)==str and order==2:
            if 'MMpol' in approx:
                use_alpha_instead_alphahalf=True

        # For MMpol approximation we have to use alpha instead alpha/2 and resulting induced dipoles 
        # have to be devided by 2. This way we correct the second term in perturbation expansion
        if use_alpha_instead_alphahalf:
            self.polar['AlphaE']=self.polar['AlphaE']*2
            self.polar['Alpha_E']=self.polar['Alpha_E']*2

        # reset iduced dipoles to zero        
        self.dipole=np.zeros((self.Nat,3),dtype='f8')            
            
        # calculate induced dipoles with polarizability AlphaE for rescaled charges
        self.calc_dipoles_All('AlphaE',NN=order)
        AtDipoles1=np.copy(self.dipole)
        if use_alpha_instead_alphahalf:
            AtDipoles1=AtDipoles1/2
            self.dipole=self.dipole/2
        #Einter=self.get_interaction_energy(index,charge=charge)
# TODO: Check if with using MMpol procedure it souldn't be 1/2 of selfinteraction energy
        Eshift=-self.get_selfinteraction_energy()
    
        if not (approx=='MMpol' and order>2):
            # reset iduced dipoles to zero        
            self.dipole=np.zeros((self.Nat,3),dtype='f8')
            
            # calculate induced dipoles with polarizability Alpha_E for rescaled charges
            self.calc_dipoles_All('Alpha_E',NN=order)
            AtDipoles2=np.copy(self.dipole)
            if use_alpha_instead_alphahalf:
                AtDipoles2=AtDipoles2/2
                self.dipole=self.dipole/2
            #Eshift=-self.get_interaction_energy(index,charge=charge)
            Eshift+=self.get_selfinteraction_energy()
            
             # reset iduced dipoles to zero        
            self.dipole=np.zeros((self.Nat,3),dtype='f8')
            
            # calculate induced dipoles with polarizability Beta for rescaled charges
            #self.calc_dipoles_All('BetaEE',NN=order//2)
            if order>2:
                self.calc_dipoles_All('BetaEE',NN=1)
            else:
                self.calc_dipoles_All('BetaEE',NN=order//2)
            AtDipolesBeta=np.copy(self.dipole)
            #Eshift=-self.get_interaction_energy(index,charge=charge)
            
            if use_alpha_instead_alphahalf:
                self.polar['AlphaE']=self.polar['AlphaE']/2
                self.polar['Alpha_E']=self.polar['Alpha_E']/2
        
        if approx==1.2 or approx==1.1 or approx=='MMpol_1.2' or approx=='MMpol_1.1':
            if is_elstat:   
                Eshift+=(self.VinterFG-DE)*self.get_selfinteraction_energy()
                Eshift+=DE
            else:
                Eshift+=self.VinterFG*self.get_selfinteraction_energy()
        elif approx==1.3 or approx=='MMpol_1.3':
            if is_elstat:
                Eshift+=DE*(1-2*self.get_selfinteraction_energy())
        elif approx=='MMpol':
            if is_elstat:
                Eshift+=DE
        if output_dipoles:
            if approx=='MMpol':
                # reset iduced dipoles to zero        
                self.dipole=np.zeros((self.Nat,3),dtype='f8')
                if order>2:
                    return Eshift,AtDipoles1
                else:
                    return Eshift,AtDipoles1+AtDipoles2
            else:
                # reset iduced dipoles to zero        
                self.dipole=np.zeros((self.Nat,3),dtype='f8')                
                
                return Eshift,AtDipoles1+AtDipoles2-self.VinterFG*AtDipolesBeta
        else:
            # reset iduced dipoles to zero        
            self.dipole=np.zeros((self.Nat,3),dtype='f8')
                
            return Eshift
    
        
    def calculate_InteractionEnergy(self,index,charge,*args,order=80,output_dipoles=False,approx=1.1):
        ''' Function for calculation  of interaction energies for chromophores
        embeded in polarizable atom environment. So far only for symetric homodimer
        
        Parameters
        ----------
        index : list of integer (dimension Natoms_of_defect)
            Specify atomic indexes of one defect. For this defect interation energy
            with induced dipoles in the environment and also other defect will 
            be calculated.
        charge : numpy.array of real (dimension Natoms_of_defect)
            Atomic trasition charges (TrEsp charges) for every atom of one defect
            defined by `index`
        *args : real (optional)
            Diference in electrostatic interaction energy between ground and 
            excited state in ATOMIC UNITS (DE). If not defined it is assumed to 
            be zero. DE=<A|V|A>-<G|V|G>
        order : integer (optional - init=80)
            Specify how many SCF steps shoudl be used in calculation  of induced dipoles
        output_dipoles : logical (optional - init=False)
            If atomic dipoles should be outputed or not. Atomic dipoles are 
            outputed as `AtDip_Alpha(E)+AtDip_Alpha(-E)-self.VinterFG*AtDip_Beta(E,E)
        approx : real (optional - init=1.2)
            Specifies which approximation should be used. **Different approximation
            than for dipole or energy shift**
            
            **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)` and 
            `Alpha(-E)`. With this apprximation diference in electrostatic interaction energy
            between ground and excited state in ATOMIC UNITS (DE) has to be imputed
            as `*args`
            
            **Approximation 1.1.2**:  Approximation 1.2 + neglecting difference
            in electrostatic interaction between ground and excited state
            (imputed as approximation 1.2 but no electrostatic interaction energy
            diference - DE is defiend)
            
            **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
            `Alpha(E)=Alpha(-E)`, however the second one is not condition 
            
            **Approximation MMpol**: Interaction energy is calculated as interaction
            without environment plus interaction of induced dipoles in the environmnet
            with electric field of the second molecule.
        
        Returns
        -------
        Einter : real
            Interaction energy in ATOMIC UNITS (Hartree) between two chormophores
            embeded in polarizable atom environment.
        AtDipoles : numpy.array of real (dimension Natoms x 3) (optional)
            Induced atomic dipoles defined as: 
            `AtDip_Alpha(E)+AtDip_Alpha(-E)-2*self.VinterFG*AtDip_Beta(E,E)`
            in ATOMIC UNITS (e*Bohr)
            
        Notes
        ----------
        E = -Ind_dip_Alpha(E)*El_field_TrCharge + Ind_dip_Alpha(-E)*El_field_TrCharge 
        Then final energy shift E_fin of molecule embeded in environment is calculated
        according to the approximation:
        
        **Approximation 1.1:**
            Einter=E_TrEsp*(1+E1Bself)+(self.VinterFG-DE)*E12B+E12AE+E12A_E      
        **Approximation 1.1.2:**
            Einter=E_TrEsp*(1+E1Bself)+self.VinterFG*E12B+E12AE+E12A_E
        **Approximation 1.3:**
            Einter=E_TrEsp+2*self.VinterFG*E12B+E12AE+E12A_E
        
        '''
        
        debug=False
        
        if approx==1.2:
            raise IOError('Approximation 1.2 for interaction energy calculation not yet supported. Look at Approximation 1.1')

        if approx==1.3:
            if not np.array_equal(self.polar['AlphaE'],self.polar['Alpha_E']):
                raise Warning('For calculation with Approximation 1.3 Alpha(E) should be equal Alpha(-E)')       
                
        if approx==1.1:
            if not np.array_equal(np.zeros((len(self.polar['Alpha_E']),3,3),dtype='f8'),self.polar['Alpha_E']):
                print('For calculation with Approximation 1.1 Alpha(-E) should be equal to zero')

        is_elstat=False
        if len(args)==1:
            DE=args[0]
            is_elstat=True
        
        use_alpha_instead_alphahalf=False
        if type(approx)==str and order==2:
            if 'MMpol' in approx:
                use_alpha_instead_alphahalf=True

        # For MMpol approximation we have to use alpha instead alpha/2 and resulting induced dipoles 
        # have to be devided by 2. This way we correct the second term in perturbation expansion
        if use_alpha_instead_alphahalf:
            self.polar['AlphaE']=self.polar['AlphaE']*2
            self.polar['Alpha_E']=self.polar['Alpha_E']*2
            
        # reset iduced dipoles to zero        
        self.dipole=np.zeros((self.Nat,3),dtype='f8')        
        
        # TrEsp interaction energy
        E_TrEsp=self.get_interaction_energy(index,charge=charge) 
        #print('TrEsp interaction:',E_TrEsp*const.HaToInvcm)
        # this will put zero charges on index atoms then calculate potential from
        # everything else and calculate interaction with charges defined by charges
        # original charges and dipoles remain unchanged
        
        # reset iduced dipoles to zero        
        self.dipole=np.zeros((self.Nat,3),dtype='f8') 
        
        if not (approx=='MMpol' and order>2):
            # calculate induced dipoles with polarizability Beta for rescaled charges
            if order>2:
                self.calc_dipoles_All('BetaEE',NN=1)
            else:
                self.calc_dipoles_All('BetaEE',NN=order//2)
            #self.calc_dipoles_All('BetaEE',NN=order//2)
            AtDipolesBeta=np.copy(self.dipole)
            E1Bself=-self.get_selfinteraction_energy()      # should be negative for all Beta
            E12B=E_TrEsp-self.get_interaction_energy(index,charge=charge) #
        
            # reset iduced dipoles to zero        
            self.dipole=np.zeros((self.Nat,3),dtype='f8')
        
        # calculate induced dipoles with polarizability AlphaE for rescaled charges
        if debug==True and order==2:
            self.calc_dipoles_All('AlphaE',NN=1)
            self._test_2nd_order('AlphaE')
        else:
            self.calc_dipoles_All('AlphaE',NN=order)
        if use_alpha_instead_alphahalf:
            self.dipole=self.dipole/2
        AtDipoles1=np.copy(self.dipole)
        E12AE=(self.get_interaction_energy(index,charge=charge)-E_TrEsp)
        
        # reset iduced dipoles to zero        
        self.dipole=np.zeros((self.Nat,3),dtype='f8')
        
        if not (approx=='MMpol' and order>2):
             # calculate induced dipoles with polarizability AlphaE for rescaled charges
            self.calc_dipoles_All('Alpha_E',NN=order)
            if use_alpha_instead_alphahalf:
                self.dipole=self.dipole/2
            AtDipoles2=np.copy(self.dipole)
            E12A_E=(self.get_interaction_energy(index,charge=charge)-E_TrEsp)
        
        if use_alpha_instead_alphahalf:
            self.polar['AlphaE']=self.polar['AlphaE']/2
            self.polar['Alpha_E']=self.polar['Alpha_E']/2
               
        
        if approx==1.1 or approx=='MMpol_1.1':
            if is_elstat:
                Einter=E_TrEsp*(1+E1Bself)+(self.VinterFG-DE)*E12B+E12AE+E12A_E
            else:
                Einter=E_TrEsp*(1+E1Bself)+self.VinterFG*E12B+E12AE+E12A_E
        elif approx==1.3 or approx=='MMpol_1.3':
            Einter=E_TrEsp+2*self.VinterFG*E12B+E12AE+E12A_E
        elif approx=='MMpol':
            Einter=E_TrEsp+E12AE
        else:
            raise IOError('Unknown type of approximation. Alowed types are: 1.1 and 1.3')
        
        if output_dipoles:
            if approx=='MMpol':
                if order>2:
                    return Einter,AtDipoles1
                else:
                    return Einter,AtDipoles1+AtDipoles2
            else:
                return Einter,AtDipoles1+AtDipoles2-2*self.VinterFG*AtDipolesBeta
        else:
            return Einter
    
    def _calculate_InteractionEnergy2(self,index,charge,order=80,output_dipoles=False):
        ''' Function for calculation  of interaction energies for chromophores
        embeded in polarizable atom environment. So far only for symetric homodimer

        Induced dipoles, needed for inteaction energy calculation, calculated 
        at every step of the SCF procedure are for output multiplied by different
        factor. First order is multiplied by factor 1, second by factor 3/2, 
        third by factor of 2, etc.
        
        **According to latest derivation the rescaling of every SCF step should 
        not be used and therefore also this function should not be used**        
        
        Notes
        ----------
        This function is kept only for maintaining backward compatibility.
        
        '''

        # reset iduced dipoles to zero        
        self.dipole=np.zeros((self.Nat,3),dtype='f8')        
        
        # TrEsp interaction energy
        E_TrEsp=self.get_interaction_energy(index,charge=charge) 
        #print('TrEsp interaction:',E_TrEsp*const.HaToInvcm)
        # this will put zero charges on index atoms then calculate potential from
        # everything else and calculate interaction with charges defined by charges
        # original charges and dipoles remain unchanged
        
        # reset iduced dipoles to zero        
        self.dipole=np.zeros((self.Nat,3),dtype='f8') 
        
        # calculate induced dipoles with polarizability Beta for rescaled charges
        if order>2:
            self.calc_dipoles_All('BetaEE',NN=1)
        else:
            self.calc_dipoles_All('BetaEE',NN=order//2)
        AtDipolesBeta=np.copy(self.dipole)
        E1Bself=-self.get_selfinteraction_energy()      # should be negative for all Beta
        E12B=E_TrEsp-self.get_interaction_energy(index,charge=charge) #
        
        # reset iduced dipoles to zero        
        self.dipole=np.zeros((self.Nat,3),dtype='f8')
        
        # calculate induced dipoles with polarizability AlphaE for rescaled charges
        #self.calc_dipoles_All('AlphaE',NN=order)
        self._calc_dipoles_all2('AlphaE',NN=order+2)
        AtDipoles1=np.copy(self.dipole)
        E12AE=2*(self.get_interaction_energy(index,charge=charge)-E_TrEsp)
        
        # reset iduced dipoles to zero        
        self.dipole=np.zeros((self.Nat,3),dtype='f8')
        
         # calculate induced dipoles with polarizability AlphaE for rescaled charges
        #self.calc_dipoles_All('Alpha_E',NN=order)
        self._calc_dipoles_all2('Alpha_E',NN=order+2)        
        AtDipoles2=np.copy(self.dipole)
        E12A_E=2*(self.get_interaction_energy(index,charge=charge)-E_TrEsp)
        
        
        Einter=E_TrEsp*(1+E1Bself)+2*self.VinterFG*E12B+E12AE+E12A_E
        
        if output_dipoles:
            return Einter,AtDipoles1+AtDipoles2-2*self.VinterFG*AtDipolesBeta
        else:
            return Einter
    


#==============================================================================
# Definition of fuction for allocation of polarized molecules
#==============================================================================
def prepare_molecule_1Def(filenames,indx,AlphaE,Alpha_E,BetaE,VinterFG,verbose=False,**kwargs):
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
    mol_test=clas.QchMolecule('Perylene-charge')
    mol_test.load_xyz(xyzfile2)
    if verbose:
        print('        Loading molecule...')
    mol=clas.QchMolecule('FGrph-1Perylene')
    mol.load_xyz(xyzfile)
    coor,charge,at_type=read.read_TrEsp_charges(filenameESP)
    if verbose:
        print('        Centering molecule...')
    mol.center_molecule(indx_center1,indx_x1,indx_y1)
    
    index1=calc.identify_molecule(mol,mol_test,indx_center1,indx_x1,indx_y1,indx_center_test,indx_x_test,indx_y_test,onlyC=True)

    if len(index1)!=len(np.unique(index1)):
        raise IOError('There are repeating elements in index file')

    # Assign pol types and charges
    PolType=[]
    Polcharge=[]
    PolCoor=[]
    for ii in range(mol.at_spec['NAtoms']):
        if mol.at_spec['AtType'][ii]=='C' and (ii in index1):
            Polcharge.append(charge[np.where(index1==ii)[0][0]])
            PolType.append('C')
            PolCoor.append(mol.at_spec['Coor'][ii])
        elif mol.at_spec['AtType'][ii]=='C':
            PolType.append('CF')
            Polcharge.append(0.0)
            PolCoor.append(mol.at_spec['Coor'][ii])
    PolType=np.array(PolType)
    Polcharge=np.array(Polcharge,dtype='f8')
    PolCoor=np.array(PolCoor,dtype='f8')
    
    # project molecule whole system to plane defined by defect
    nvec=np.array([0.0,0.0,1.0],dtype='f8')
    center=np.array([0.0,0.0,0.0],dtype='f8')
    PolCoor=pos.project_on_plane(PolCoor,nvec,center)
    
    polar={}
    polar['AlphaE']=np.zeros((len(PolCoor),3,3),dtype='f8')
    polar['Alpha_E']=np.zeros((len(PolCoor),3,3),dtype='f8')
    polar['BetaE']=np.zeros((len(PolCoor),3,3),dtype='f8')
    
    mol_polar=Dielectric(PolCoor,Polcharge,np.zeros((len(PolCoor),3),dtype='f8'),
                              polar['AlphaE'],polar['Alpha_E'],polar['BetaE'],VinterFG)
    
    ZeroM=np.zeros((3,3),dtype='f8')
    if kwargs:
        AlphaE_def=kwargs['Alpha(E)']
        Alpha_E_def=kwargs['Alpha(-E)']
        BetaE_def=kwargs['Beta(E,E)']
    if kwargs:
        mol_polar.polar=mol_polar.assign_polar(PolType,**{'PolValues': {'CF': [AlphaE,Alpha_E,BetaE],
                                                                    'CD': [AlphaE,Alpha_E,BetaE],
                                                                    'C': [AlphaE_def,Alpha_E_def,BetaE_def]}})
    else:
        mol_polar.polar=mol_polar.assign_polar(PolType,**{'PolValues': {'CF': [AlphaE,Alpha_E,BetaE],
                                                                    'CD': [AlphaE,Alpha_E,BetaE],
                                                                    'C': [ZeroM,ZeroM,ZeroM]}})
    return mol_polar,index1,charge

def prepare_molecule_2Def(filenames,indx,AlphaE,Alpha_E,BetaE,VinterFG,nvec=np.array([0.0,0.0,1.0],dtype='f8'),verbose=False, def2_charge=False,**kwargs):
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
    #filenameESP="".join([MolDir,'Perylene_TDDFT_fitted_charges_NoH.out'])
    if verbose:
        print('     Reading charges and format to polarization format...')
    mol_test=clas.QchMolecule('Perylene-charge')
    mol_test.load_xyz(xyzfile2)
    coor,charge,at_type=read.read_TrEsp_charges(filenameESP)
    
    # load molecule - fuorographene with 2 defects 
    if verbose:
        print('        Loading molecule...')
    mol=clas.QchMolecule('FGrph-2Perylene')
    mol.load_xyz(xyzfile)
    
    index1=calc.identify_molecule(mol,mol_test,indx_center1,indx_x1,indx_y1,indx_center_test,indx_x_test,indx_y_test,onlyC=True)
    index2=calc.identify_molecule(mol,mol_test,indx_center2,indx_x2,indx_y2,indx_center_test,indx_x_test,indx_y_test,onlyC=True)
    if len(index1)!=len(np.unique(index1)) or len(index2)!=len(np.unique(index2)):
        print('index1:')
        print(index1)
        print('index2:')
        print(index2)
        raise IOError('There are repeating elements in index file')

    # Assign pol types
    PolType=[]
    Polcharge=[]
    PolCoor=[]
    for ii in range(mol.at_spec['NAtoms']):
        if mol.at_spec['AtType'][ii]=='C' and (ii in index1):
            Polcharge.append(charge[np.where(index1==ii)[0][0]])
            PolType.append('C')
            PolCoor.append(mol.at_spec['Coor'][ii])
        elif mol.at_spec['AtType'][ii]=='C' and (ii in index2):
            if def2_charge:
                Polcharge.append(charge[np.where(index2==ii)[0][0]])
            else:
                Polcharge.append(0.0)
            #Polcharge.append(charge[np.where(index2==ii)[0][0]])
            PolType.append('C')
            PolCoor.append(mol.at_spec['Coor'][ii])
        elif mol.at_spec['AtType'][ii]=='C':
            PolType.append('CF')
            Polcharge.append(0.0)
            PolCoor.append(mol.at_spec['Coor'][ii])
            
    PolType=np.array(PolType)
    Polcharge=np.array(Polcharge,dtype='f8')
    PolCoor=np.array(PolCoor,dtype='f8')
    
    # project molecule whole system to plane defined by defect
    center=np.array([0.0,0.0,0.0],dtype='f8')
    PolCoor=pos.project_on_plane(PolCoor,nvec,center)
    
    # center projected molecule on plane
    if verbose:
        print('        Centering molecule...')
    PolCoor=pos.CenterMolecule(PolCoor,indx_center1,indx_x1,indx_y1)
    
    polar={}
    polar['AlphaE']=np.zeros((len(PolCoor),3,3),dtype='f8')
    polar['Alpha_E']=np.zeros((len(PolCoor),3,3),dtype='f8')
    polar['BetaE']=np.zeros((len(PolCoor),3,3),dtype='f8')
    
    mol_polar=Dielectric(PolCoor,Polcharge,np.zeros((len(PolCoor),3),dtype='f8'),
                         polar['AlphaE'],polar['Alpha_E'],polar['BetaE'],VinterFG)
    
    ZeroM=np.zeros((3,3),dtype='f8')
    if kwargs:
        AlphaE_def=kwargs['Alpha(E)']
        Alpha_E_def=kwargs['Alpha(-E)']
        BetaE_def=kwargs['Beta(E,E)']
    if kwargs:
        mol_polar.polar=mol_polar.assign_polar(PolType,**{'PolValues': {'CF': [AlphaE,Alpha_E,BetaE],
                                                                    'CD': [AlphaE,Alpha_E,BetaE],
                                                                    'C': [AlphaE_def,Alpha_E_def,BetaE_def]}})
    else:
        mol_polar.polar=mol_polar.assign_polar(PolType,**{'PolValues': {'CF': [AlphaE,Alpha_E,BetaE],
                                                                    'CD': [AlphaE,Alpha_E,BetaE],
                                                                    'C': [ZeroM,ZeroM,ZeroM]}})
                                        
    return mol_polar,index1,index2,charge

def CalculateTrDip(filenames,ShortName,index_all,Dipole_QCH,Dip_all,AlphaE,Alpha_E,BetaE,VinterFG,FG_charges,ChargeType,order=80,verbose=False,approx=1.1,MathOut=False,**kwargs):
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
        Specifies which method was used for calcultion of ground and excited state
        charges for defect atoms. Allowed types are: 'qchem','qchem_all','AMBER'
        and 'gaussian'. **'qchem'** - charges calculated by fiting Q-Chem ESP on carbon
        atoms. **'qchem_all'** - charges calculated by fiting Q-Chem ESP on all
        atoms, only carbon charges are used and same charge is added to all carbon
        atoms in order to have neutral molecule. **'AMBER'** and **'gaussian'**
        not yet fully implemented.
    order : integer (optional - init=80)
            Specify how many SCF steps shoudl be used in calculation  of induced dipoles
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    approx : real (optional - init=1.1)
            Specifies which approximation should be used.
            
            **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)` and 
            `Alpha(-E)`. With this apprximation diference in electrostatic interaction energy
            between ground and excited state in ATOMIC UNITS (DE) has to be imputed
            as `*args`
            
            **Approximation 1.1.2**:  Approximation 1.2 + neglecting difference
            in electrostatic interaction between ground and excited state
            (imputed as approximation 1.2 but no electrostatic interaction energy
            diference - DE is defiend)

            **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.
            With this apprximation diference in electrostatic interaction energy
            between ground and excited state in ATOMIC UNITS (DE) has to be imputed
            as `*args`
            
            **Approximation 1.2.2**:  Approximation 1.2 + neglecting difference
            in electrostatic interaction between ground and excited state
            (imputed as approximation 1.2 but no electrostatic interaction energy
            diference - DE is defiend)            
            
            **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
            `Alpha(E)=Alpha(-E)`, however the second one is not condition 
    **kwargs : dictionary (optional)
        Definition of polarizabitity matrixes for defect atoms (if nonzero
        polarizability is used)
    
    Notes
    ----------
    Working only for fluorographene system with single defect  
     
    '''
    
    for ii in range(len(filenames)):
        if verbose:
            print('Calculation of dipoles for:',ShortName[ii])
        
        # read and prepare molecule
        if kwargs:
            mol_polar,index1,charge=prepare_molecule_1Def(filenames[ii],index_all[ii],AlphaE,Alpha_E,BetaE,VinterFG,verbose=False,**kwargs)
        else:
            mol_polar,index1,charge=prepare_molecule_1Def(filenames[ii],index_all[ii],AlphaE,Alpha_E,BetaE,VinterFG,verbose=False)
        mol_Elstat,at_type=ElStatMol.PrepareMolecule_1Def(filenames[ii],index_all[ii],FG_charges,ChargeType=ChargeType,verbose=False)

        # calculate <A|V|A>-<G|V|G>
        DE=mol_Elstat.get_EnergyShift()
        #print('DE:',DE*const.HaToInvcm,'cm-1')

        # calculate transition dipole        
        TrDip,AtDipoles=mol_polar.get_TrDip(DE,order=order,output_dipoles=True,approx=approx)
        
        if verbose:
            print('        Total transition dipole:',np.sqrt(np.dot(TrDip,TrDip)),'Quantum chemistry dipole:',Dipole_QCH[ii])
        print(ShortName[ii],Dipole_QCH[ii],np.sqrt(np.dot(TrDip,TrDip))) 
        Dip_all[ii,:]=TrDip[:]
        
        if MathOut:
            # output dipoles to mathematica
            Bonds=inter.GuessBonds(mol_polar.coor,bond_length=4.0)
            mat_filename="".join(['Pictures/Polar_',ShortName[ii],'.nb'])
            out.OutputMathematica(mat_filename,mol_polar.coor,Bonds,['C']*mol_polar.Nat,scaleDipole=30.0,**{'TrPointCharge': mol_polar.charge,'AtDipole': AtDipoles,'rSphere_dip': 0.5,'rCylinder_dip':0.1})

def CalculateEnergyShift(filenames,ShortName,index_all,Eshift_QCH,Eshift_all,AlphaE,Alpha_E,BetaE,VinterFG,FG_charges,ChargeType,order=80,verbose=False,approx=1.1,MathOut=False,**kwargs):
    ''' Calculate transition energy shifts for defect embeded in polarizable atom
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
    Eshift_QCH : list of real (dimension Nsystems)
        List of quantum chemistry values of transition energy shifts in INVERSE
        CENTIMETERS for defect in polarizable atom environment (used for printing
        comparison - not used for calculation at all)
    Eshift_all : list of real (dimension Nsystems)
        In this variable there will be stored transition energy shifts in ATOMIC
        UNITS (Hartree) calculated by polarizable atoms method for description 
        of the environment.
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
        Specifies which method was used for calcultion of ground and excited state
        charges for defect atoms. Allowed types are: 'qchem','qchem_all','AMBER'
        and 'gaussian'. **'qchem'** - charges calculated by fiting Q-Chem ESP on carbon
        atoms. **'qchem_all'** - charges calculated by fiting Q-Chem ESP on all
        atoms, only carbon charges are used and same charge is added to all carbon
        atoms in order to have neutral molecule. **'AMBER'** and **'gaussian'**
        not yet fully implemented.
    order : integer (optional - init=80)
            Specify how many SCF steps shoudl be used in calculation  of induced dipoles
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    approx : real (optional - init=1.1)
            Specifies which approximation should be used.
            
            **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)` and 
            `Alpha(-E)`. With this apprximation diference in electrostatic interaction energy
            between ground and excited state in ATOMIC UNITS (DE) has to be imputed
            as `*args`
            
            **Approximation 1.1.2**:  Approximation 1.2 + neglecting difference
            in electrostatic interaction between ground and excited state
            (imputed as approximation 1.2 but no electrostatic interaction energy
            diference - DE is defiend)

            **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.
            With this apprximation diference in electrostatic interaction energy
            between ground and excited state in ATOMIC UNITS (DE) has to be imputed
            as `*args`
            
            **Approximation 1.2.2**:  Approximation 1.2 + neglecting difference
            in electrostatic interaction between ground and excited state
            (imputed as approximation 1.2 but no electrostatic interaction energy
            diference - DE is defiend)            
            
            **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
            `Alpha(E)=Alpha(-E)`, however the second one is not condition 
    **kwargs : dictionary (optional)
        Definition of polarizabitity matrixes for defect atoms (if nonzero
        polarizability is used)
        
    Notes
    ----------
    Working only for system with single defect  
     
    '''
    
    for ii in range(len(filenames)):
        if verbose:
            print('Calculation of excitation energy shift for:',ShortName[ii])
        
        # read and prepare molecule
        if kwargs: 
            mol_polar,index1,charge=prepare_molecule_1Def(filenames[ii],index_all[ii],AlphaE,Alpha_E,BetaE,VinterFG,verbose=False,**kwargs)
        else:
            mol_polar,index1,charge=prepare_molecule_1Def(filenames[ii],index_all[ii],AlphaE,Alpha_E,BetaE,VinterFG,verbose=False)
        mol_Elstat,at_type=ElStatMol.PrepareMolecule_1Def(filenames[ii],index_all[ii],FG_charges,ChargeType=ChargeType,verbose=False)

        # calculate <A|V|A>-<G|V|G>
        DE=mol_Elstat.get_EnergyShift()
        #print('DE:',DE*const.HaToInvcm,'cm-1')

        # calculate transition dipole
        Eshift,AtDipoles=mol_polar.calculate_EnergyShift(index1,charge,DE,order=order,output_dipoles=True,approx=approx)
        
        if verbose:
            print('        Transition enegy shift:',Eshift*const.HaToInvcm,'Quantum chemistry shift:',Eshift_QCH[ii])
        print(ShortName[ii],Eshift_QCH[ii],Eshift*const.HaToInvcm) 
        Eshift_all[ii]=Eshift*const.HaToInvcm
        
        if MathOut:
            # output dipoles to mathematica
            Bonds=inter.GuessBonds(mol_polar.coor,bond_length=4.0)
            mat_filename="".join(['Pictures/Polar_',ShortName[ii],'.nb'])
            out.OutputMathematica(mat_filename,mol_polar.coor,Bonds,['C']*mol_polar.Nat,scaleDipole=30.0,**{'TrPointCharge': mol_polar.charge,'AtDipole': AtDipoles,'rSphere_dip': 0.5,'rCylinder_dip':0.1})
            
def CalculateInterE(filenames,ShortName,index_all,Energy_QCH,Energy_all,nvec_all,AlphaE,Alpha_E,BetaE,VinterFG,FG_charges,ChargeType,order=80,verbose=False,approx=1.1,MathOut=False,**kwargs):
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
        Specifies which method was used for calcultion of ground and excited state
        charges for defect atoms. Allowed types are: 'qchem','qchem_all','AMBER'
        and 'gaussian'. **'qchem'** - charges calculated by fiting Q-Chem ESP on carbon
        atoms. **'qchem_all'** - charges calculated by fiting Q-Chem ESP on all
        atoms, only carbon charges are used and same charge is added to all carbon
        atoms in order to have neutral molecule. **'AMBER'** and **'gaussian'**
        not yet fully implemented.
    order : integer (optional - init=80)
            Specify how many SCF steps shoudl be used in calculation  of induced dipoles
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    approx : real (optional - init=1.1)
            Specifies which approximation should be used.
            
            **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)` and 
            `Alpha(-E)`. With this apprximation diference in electrostatic interaction energy
            between ground and excited state in ATOMIC UNITS (DE) has to be imputed
            as `*args`
            
            **Approximation 1.1.2**:  Approximation 1.2 + neglecting difference
            in electrostatic interaction between ground and excited state
            (imputed as approximation 1.2 but no electrostatic interaction energy
            diference - DE is defiend)

            **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.
            With this apprximation diference in electrostatic interaction energy
            between ground and excited state in ATOMIC UNITS (DE) has to be imputed
            as `*args`
            
            **Approximation 1.2.2**:  Approximation 1.2 + neglecting difference
            in electrostatic interaction between ground and excited state
            (imputed as approximation 1.2 but no electrostatic interaction energy
            diference - DE is defiend)            
            
            **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
            `Alpha(E)=Alpha(-E)`, however the second one is not condition 
    **kwargs : dictionary (optional)
        Definition of polarizabitity matrixes for defect atoms (if nonzero
        polarizability is used)
    
    Notes
    ----------
    Working only for systems with two symetric defects    
    
    '''    
    
    
    for ii in range(len(filenames)):
        if verbose:
            print('Calculation of interaction energy for:',ShortName[ii])
        
        # read and prepare molecule
        if kwargs:
            mol_polar,index1,index2,charge=prepare_molecule_2Def(filenames[ii],index_all[ii],AlphaE,Alpha_E,BetaE,VinterFG,nvec=nvec_all[ii],verbose=False,**kwargs)
        else:
            mol_polar,index1,index2,charge=prepare_molecule_2Def(filenames[ii],index_all[ii],AlphaE,Alpha_E,BetaE,VinterFG,nvec=nvec_all[ii],verbose=False)
        # calculate <A|V|A>-<G|V|G>
        mol_Elstat,at_type=ElStatMol.PrepareMolecule_2Def(filenames[ii],index_all[ii],FG_charges,ChargeType=ChargeType,verbose=False)
        DE=mol_Elstat.get_EnergyShift()
        #print('DE:',DE*const.HaToInvcm,'cm-1')

        # calculate interaction energy       
        Einter,AtDipoles=mol_polar.calculate_InteractionEnergy(index2,charge,DE,order=order,output_dipoles=True,approx=approx)
        
        if verbose:
            print('        Total interaction energy:',Einter*const.HaToInvcm,'Quantum interaction energy:',Energy_QCH[ii])

        print(ShortName[ii],Energy_QCH[ii],abs(Einter*const.HaToInvcm))        
        
        Energy_all[ii]=abs(Einter*const.HaToInvcm)
        
        if MathOut:
            # output dipoles to mathematica
            Bonds=inter.GuessBonds(mol_polar.coor,bond_length=4.0)
            mat_filename="".join(['Pictures/Polar_',ShortName[ii],'.nb'])
            out.OutputMathematica(mat_filename,mol_polar.coor,Bonds,['C']*mol_polar.Nat,scaleDipole=30.0,**{'TrPointCharge': mol_polar.charge,'AtDipole': AtDipoles,'rSphere_dip': 0.5,'rCylinder_dip':0.1})


def CalculateInterE_heteroTEST(filenames,ShortName,index_all,Energy_QCH,Energy_all,nvec_all,AlphaE,Alpha_E,BetaE,VinterFG,FG_charges,ChargeType,order=80,verbose=False,approx=1.1,MathOut=False,**kwargs):
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
        Specifies which method was used for calcultion of ground and excited state
        charges for defect atoms. Allowed types are: 'qchem','qchem_all','AMBER'
        and 'gaussian'. **'qchem'** - charges calculated by fiting Q-Chem ESP on carbon
        atoms. **'qchem_all'** - charges calculated by fiting Q-Chem ESP on all
        atoms, only carbon charges are used and same charge is added to all carbon
        atoms in order to have neutral molecule. **'AMBER'** and **'gaussian'**
        not yet fully implemented.
    order : integer (optional - init=80)
            Specify how many SCF steps shoudl be used in calculation  of induced dipoles
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    approx : real (optional - init=1.1)
            Specifies which approximation should be used.
            
            **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)` and 
            `Alpha(-E)`. With this apprximation diference in electrostatic interaction energy
            between ground and excited state in ATOMIC UNITS (DE) has to be imputed
            as `*args`
            
            **Approximation 1.1.2**:  Approximation 1.2 + neglecting difference
            in electrostatic interaction between ground and excited state
            (imputed as approximation 1.2 but no electrostatic interaction energy
            diference - DE is defiend)

            **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.
            With this apprximation diference in electrostatic interaction energy
            between ground and excited state in ATOMIC UNITS (DE) has to be imputed
            as `*args`
            
            **Approximation 1.2.2**:  Approximation 1.2 + neglecting difference
            in electrostatic interaction between ground and excited state
            (imputed as approximation 1.2 but no electrostatic interaction energy
            diference - DE is defiend)            
            
            **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
            `Alpha(E)=Alpha(-E)`, however the second one is not condition 
    **kwargs : dictionary (optional)
        Definition of polarizabitity matrixes for defect atoms (if nonzero
        polarizability is used)
    
    Notes
    ----------
    Working only for systems with two symetric defects    
    
    '''    
    
    
    for ii in range(len(filenames)):
        if verbose:
            print('Calculation of interaction energy for:',ShortName[ii])
        
        # read and prepare molecule
        if kwargs:
            mol_polar,index1,index2,charge=prepare_molecule_2Def(filenames[ii],index_all[ii],AlphaE,Alpha_E,BetaE,VinterFG,nvec=nvec_all[ii],verbose=False,def2_charge=True,**kwargs)
        else:
            mol_polar,index1,index2,charge=prepare_molecule_2Def(filenames[ii],index_all[ii],AlphaE,Alpha_E,BetaE,VinterFG,nvec=nvec_all[ii],verbose=False,def2_charge=True)
        
        # calculate <A|V|A>-<G|V|G>


        # calculate interaction energy       
        Einter=mol_polar.get_InteractionEng(index1,index2,0.0,0.0,order=order,approx=approx)
        
        if verbose:
            print('        Total interaction energy:',Einter*const.HaToInvcm,'Quantum interaction energy:',Energy_QCH[ii])

        print(ShortName[ii],Energy_QCH[ii],abs(Einter*const.HaToInvcm)) 
        
        # HOMODIMER
        if kwargs:
            mol_polar,index1,index2,charge=prepare_molecule_2Def(filenames[ii],index_all[ii],AlphaE,Alpha_E,BetaE,VinterFG,nvec=nvec_all[ii],verbose=False,**kwargs)
        else:
            mol_polar,index1,index2,charge=prepare_molecule_2Def(filenames[ii],index_all[ii],AlphaE,Alpha_E,BetaE,VinterFG,nvec=nvec_all[ii],verbose=False)
            
        Einter,AtDipoles=mol_polar.calculate_InteractionEnergy(index2,charge,0.0,order=order,output_dipoles=True,approx=approx)
        
        print(ShortName[ii],Energy_QCH[ii],abs(Einter*const.HaToInvcm))        
        
        Energy_all[ii]=abs(Einter*const.HaToInvcm)
        

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
    pol_mol.swap_atoms(index1,index2)
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
    pol_mol.swap_atoms(index1,index2)
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
    