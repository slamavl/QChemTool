# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:12:49 2016

@author: uzivatel
"""
import numpy as np
from copy import deepcopy
from .general import Energy
from .density import DensityGrid

class MO:
    ''' Class containing information about molecular orbitals.
    
    name : string
        Name of molecular orbital class
    coeff : numpy array or real (dimension N_mo x N_AO_orient)
        Usually N_mo=N_AO_orient. Matrix with expansion coefficients of molecular
        orbital into AO basis.
    nmo : integer
        Number of molecular orbitals
    energy : Energy class
        Molecular orbital energies (``energy.value`` - vector with all energies)
        - Fock single electron energies (energies are energy units managed).
    occ : numpy.array of real (dimension Nmo_orbitals)
        Occupation number of individual molecular orbitals. Number between 2.0 
        correspond to fully occupied molecular orbital and 0.0  wich correspond
        to unoccupied MO. Summ of occupation numbers is equal to total number of electrons
    symm : list of string (dimension Nmo_orbitals)
        Symmetry of molecular orbitals
    densmat_grnd : numpy array of float (dimension Nao_orient x Nao_orient)
        Total ground state electron matrix in atomic orbitals, defined as:\n 
        ``M_mat[mu,nu]=Sum_{n}{occ_n*C_n,mu*C_n,nu}``.\n 
        Where `C_n,mu` are expansion coefficients of molecular orbital `n` into
        atomic orbital `mu` (contribution of atomic orbital `mu` in molecular 
        orbital `n`).
    init : logical
        information if molecular orbitals are initialized (if some information is
        allready loaded)
 
    Functions
    -----------
    add_all : 
        Add molecular orbital including expansion coefficients into atomic 
        orbitals, energy of the molecular orbital, occupation number and symmetry
    rotate :
        Rotate the molecular orbitals and ground state electron density matrix
        by specified angles in radians in positive direction.
    rotate_1 :
        Inverse totation to rotate
    copy :
        Create 1 to 1 copy of the molecular orbitals with all classes and types.
    get_MO_norm :
        Calculate norm of secified molecular orbital
    normalize :
        Normalize molecular orbitals (if loaded from Gaussian calculation there
        are some round errors and therefore orbitals have slightly different norm
        ~0.998 in this order)
    get_mo_grid :
        Evaluate selected molecular orbital on the grid
    get_grnd_densmat :
        Calculate ground state electron density matrix
    get_MO_type : 
        Calculates if molecular orbital is sigma or pi type
    
    '''
    
    def __init__(self):
        self.coeff=np.array([],dtype='f8')
        self.nmo=0
        self.energy=Energy(None)
        self.occ=[]
        self.symm=[]
        self.name='Default'
        self.init=False
        self.densmat_grnd=None
    
    def _add_coeffs(self,coeff):
        if  not self.init:
            if np.ndim(coeff)==1:
                self.coeff=np.array([coeff],dtype='f8')
                self.nmo+=1
            else:
                self.coeff=np.array(coeff,dtype='f8')
                self.nmo+=len(coeff)
            self.init=True
        else:
            if np.shape(self.coeff[0])==np.shape(coeff):
                self.coeff=np.vstack((self.coeff,coeff))
                self.nmo+=1
            else:
                raise Warning('You are trying to read different coefficient matrix shape')
    
    def _add_energy(self,energy):
        self.energy.add_energy(energy)
    
    def _add_occ(self,occupation):
        self.occ.append(occupation)
    
    def _add_symm(self,symmetry):
        self.symm.append(symmetry)

    def add_all(self,coeff,energy,occ,symm):
        """Adds molecular orbital including all needed informations. It can add
        more molecular orbitals at once.
        
        Parameters
        ----------
        coeff : numpy array or list of floats 
            Expansion coefficients of the molecular orbital atomic orbital
            basis
        energy : float or list 
            One electron energy of loaded molecular orbitals. If single orbital
            is loaded, only single energy has to be inputed
        occ : float or list 
            Occupation number/s of the molecular orbital/s
        symm : string or list of string
            Symmetry of the molecular orbital/s
        
        """
        
        if type(energy)==list or type(energy)==np.ndarray:
            if not self.init:
                self.coeff=np.array(coeff,dtype='f8')
                self.energy=Energy(energy)
                self.occ=occ
                self.symm=symm
                self.nmo=len(coeff)
                self.init=True
            else:
                for ii in range(len(coeff)):
                    self._add_coeffs(coeff[ii])
                    self._add_energy(energy[ii])
                    self._add_occ(occ[ii])
                    self._add_symm(symm[ii])
        else:
            self._add_coeffs(coeff)
            self._add_energy(energy)
            self._add_occ(occ)
            self._add_symm(symm)
            
    def rotate(self,rotxy,rotxz,rotyz,AO):
        """"
        Rotate the orbitals around the coordinate origin by specified angles.
        Molecular orbitals are still the same but atomic orbitals rotates and
        therefore expansion coefficients into atomic orbitals has to be rotated
        too. First it rotate the structure around z axis (in xy plane), then around
        y axis and in the end around x axis in positive direction 
        (if right thumb pointing in direction of axis fingers are pointing in 
        positive rotation direction)
        
        Parameters
        --------
        rotxy,rotxz,rotxz : float
            Rotation angles in radians around z, y and x axis (in xy, xz and yz plane)
        AO : AO class
            Information about atomic orbitals. Ordering of atomic orbitals has 
            to be known to rotate molecular orbitals (because AO basis is rotated)
        
        """
        
        TransfMat=AO._get_ao_rot_mat(rotxy,rotxz,rotyz)
# TODO: check this transformation of molecular orbitals
        self.coeff=np.dot(TransfMat,self.coeff)
        if self.densmat_grnd is not None:
            self.densmat_grnd=np.dot(TransfMat,np.dot(self.densmat_grnd,TransfMat.T))
    
    def rotate_1(self,rotxy,rotxz,rotyz,AO):
        """" Inverse rotation of molecular orbitals to **rotate** function.
        First rotate the structure around x axis (in yz plane), then around
        y axis and in the end around z axis in negtive direction 
        (if left thumb pointing in direction of axis fingers are pointing in 
        negative rotation direction)
        
        Parameters
        --------
        rotxy,rotxz,rotxz : float
            Rotation angles in radians around z, y and x axis (in xy, xz and yz plane)
        AO : AO class
            Information about atomic orbitals. Ordering of atomic orbitals has 
            to be known to rotate molecular orbitals (because AO basis is rotated)
        """
        
        TransfMat=AO._get_ao_rot_mat_1(rotxy,rotxz,rotyz)
# TODO: check this transformation of molecular orbitals
        self.coeff=np.dot(TransfMat,self.coeff)
    
    def get_MO_norm(self,AO):
        """
        Calculate norm of molecular orbital.
        
        Parameters
        ----------
        AO : AO class
            Information about atomic orbitals. For calculation of norm of molecular
            orbital overlap matrix between atomic orbitals has to be known.
        
        Returns
        --------
        MOnorm : float
            Norm of molecular orbital. Should be equal to 1.0
        
        """
        
        if AO.overlap is None:
            AO.get_overlap()

        MOnorm=np.zeros(self.nmo,dtype='f8')
        for mo_i in range(self.nmo):
            MOnorm[mo_i]=np.dot(np.dot(self.coeff[mo_i],AO.overlap),self.coeff[mo_i].T)
        return MOnorm

    def normalize(self,AO):
        """
        Normalize all molecular orbitals.
        
        Parameters
        ----------
        AO : AO class
            Information about atomic orbitals. For calculation of norm of molecular
            orbital overlap matrix between atomic orbitals has to be known.
        
        """
        MOnorm=self.get_MO_norm(AO)
        for mo_i in range(self.nmo):
            self.coeff[mo_i]=self.coeff[mo_i]/np.sqrt(MOnorm[mo_i])
            
    
    def get_mo_grid(self,AO,grid,mo_i):  
        """ Evaluate molecular orbital on given grid
        
        Parameters
        ----------
        AO : AO class
            Information about atomic orbitals.
        grid : Grid class
            Information about grid on which molecular orbital is evaluated.
        mo_i : integer
            Index of molecular orbital which will be evaluated on the grid 
            (starting from 0)
        
        Returns
        --------
        mo_grid : numpy array of float (dimension Grid_Nx x Grid_Ny x Grid_Nz)
            Values of molecular orbital on all grid points
        """
        
        mo_grid=np.zeros(np.shape(grid.X))        
        if AO.grid is None:
            keep_grid=True
            counter=0
            for ii in range(AO.nao):
                new_grid=True
                if not((ii>0) and (AO.atom[ii].indx==AO.atom[ii-1].indx)):
                        new_grid=False
                for jj in range(len(AO.orient[ii])):
                        if counter==AO.nao_orient:
                            keep_grid=False
                        slater_ao_tmp=AO.get_slater_ao_grid(grid,counter,keep_grid=keep_grid,new_grid=new_grid)
                        mo_grid += np.multiply(self.coeff[mo_i][counter],slater_ao_tmp)
        else:
            for ii in range(AO.nao_orient):
                mo_grid += np.multiply(self.coeff[mo_i][ii],AO.grid[ii])
        return mo_grid


    def get_mo_cube(self,AO,grid,mo_i):
        """ Create cube density for specified molecular orbital
        
        Parameters
        ----------
        AO : AO class
            Information about atomic orbitals.
        grid : Grid class
            Information about grid on which molecular orbital is evaluated.
        mo_i : integer
            Index of molecular orbital which will be evaluated on the grid 
            (starting from 0)
        
        Returns
        --------
        mo_cube : Density class
            Cube density of specified molecular orbital.
        """

        mo_grid=self.mo.get_mo_grid(self.ao,grid,mo_i)
        step=np.zeros((3,3))
        step[0,0]=grid.delta[0]
        step[1,1]=grid.delta[1]
        step[2,2]=grid.delta[2]
        mo_cube=DensityGrid(np.array(grid.origin),np.shape(grid.X),step,mo_grid,typ='mo',mo_indx=mo_i+1,Coor=self.struc.coor._value,At_charge=self.struc.ncharge)
        return mo_cube
    
    def get_grnd_densmat(self,verbose=False):
        """ Calculate total ground state electron density matrix
        
        Definition
        ----------
        Total ground state electron matrix is defined as:\n 
        ``M_mat[mu,nu]=Sum_{n}{occ_n*C_n,mu*C_n,nu}``.\n 
        Where `C_n,mu` are expansion coefficients of molecular orbital `n` into
        atomic orbital `mu` (contribution of atomic orbital `mu` in molecular 
        orbital `n`).
        
        Notes
        ---------
        Total ground state electron matrix in atomic orbitals is stored at \n
        **self.densmat_grnd** \n
        as numpy array of dimension (Nao_orient x Nao_orient)

        """
        # This could be improved for example by calculating norm of ocuupied MO * 2 and sum should have been Nel. From this difference we can calculate scaling factor to obtain exactly number of electrons 
        Nocc=int(np.sum(self.occ))//2   # number of occupied states - for single determinant methods
        if verbose:
            print('        Number of occupied states:',Nocc) 
        M_mat=np.zeros((self.nmo,self.nmo),dtype='f8')
        for nn in range(Nocc):
            coef=2.0  # this is for extension to multireference methods - two electrons in each orbital
            mo_i=nn
            mo_j=nn
            Cj_mat,Ci_mat = np.meshgrid(self.coeff[mo_j],self.coeff[mo_i])
            M_mat+=np.multiply(coef,np.multiply(Ci_mat,Cj_mat))
        self.densmat_grnd=np.copy(M_mat)
    
    def copy(self):
        ''' Create deep copy of the all informations in the class. 
        
        Returns
        ----------
        MO_new : MO class
            MO class with exactly the same information as previous one 
            
        '''
        
        MO_new=deepcopy(self)
        return MO_new
    
    def get_MO_type(self,AO,krit=90,nvec=[0.0,0.0,1.0]):
        ''' Determines if molecular orbital of the system is 'Pi' or 'Sigma' type
        
        Parameters
        ----------
        AO : class AO 
            atomic orbital information
        krit : float (optional - init=90)
            Percentage of pi state contribution which will be considered as 
            pi-molecular orbital. accepted values are (0,100) 
        nvec : numpy array or list dimension 3 (optional - init=[0.0,0.0,1.0])
            Normal vector to pi-conjugated plane.
            
        Returns
        ----------
        MO_type : list of string (dimension Nmo)
            List with orbital types ('Pi' or 'Sigma') for each molecular orbital
        
        
        '''
        if krit>=100 or krit<0:
            raise IOError('Kriteria for determine MO type has to be in percentage of P orbilats in molecular orbital (0,100)')
        
        #normalize normal vector
        nvec=np.array(nvec,dtype='f8')
        nvec=nvec/np.linalg.norm(nvec)
        
        # Calculate projection of p atomic orbitals into specified direction
        projection=np.zeros(AO.nao_orient,dtype='f8')
        for ii in range(AO.nao_orient):
                if np.sum(np.abs(AO.indx_orient[ii][1]))==1:
                    projection[ii]=np.abs(np.dot(AO.indx_orient[ii][1],nvec))
        
        # This is not the most correct approach but it might work
        coeff2=self.coeff**2
        prob_all=np.sum(coeff2,axis=1)
        prob_pi=np.dot(projection,coeff2.T)
        prob=prob_pi/prob_all*100
        
        MO_type=[]
        for kk in range(self.nmo):
            if prob[kk]>=krit:
                MO_type.append('Pi')
            else:
                MO_type.append('Sigma')
        return MO_type
#            
