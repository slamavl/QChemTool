# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:12:49 2016

@author: uzivatel
"""
import numpy as np
import timeit
from multiprocessing import Pool, cpu_count
from functools import partial
from sys import platform
import scipy
from copy import deepcopy

from ..qch_functions import overlap_STO, dipole_STO, quadrupole_STO, norm_GTO
from ..positioningTools import fill_basis_transf_matrix
from .general import Coordinate

# nepocitat self.exct_spec[exct_i]['coeff_mat']=M_mat pokud neni nutne - zdrzuje

def _ao_over_line(Coor_lin,Coeffs_lin,Exp_lin,Orient_lin,indx):
    '''
    This Function calculates row in overlap matrix            
    '''
#    print('Start of parallel calculation witj process id:',os.getpid())
    NAO=len(Coor_lin)
    AO_over_row=np.zeros(NAO)
    for ii in range(indx,NAO):
        AO_over_row[ii]=overlap_STO(Coor_lin[indx],Coor_lin[ii],np.array(Coeffs_lin[indx]),np.array(Coeffs_lin[ii]),np.array(Exp_lin[indx]),np.array(Exp_lin[ii]),np.array(Orient_lin[indx]),np.array(Orient_lin[ii]))
#    print(os.getpid())
    return AO_over_row
    
def _dip_over_line(Coor_lin,Coeffs_lin,Exp_lin,Orient_lin,indx):
    '''
    This Function calculates row in dipole matrix            
    '''
    NAO=len(Coor_lin)
    Dip_over_row=np.zeros((NAO,3))
    for ii in range(indx,NAO):
         Dip_over_row[ii,:]=dipole_STO(Coor_lin[indx],Coor_lin[ii],np.array(Coeffs_lin[indx]),np.array(Coeffs_lin[ii]),np.array(Exp_lin[indx]),np.array(Exp_lin[ii]),np.array(Orient_lin[indx]),np.array(Orient_lin[ii]))
    return Dip_over_row
 
def _quad_over_line(Coor_lin,Coeffs_lin,Exp_lin,Orient_lin,only_triangle,indx):
    '''
    This Function calculates row in quadrupole matrix            
    '''
    NAO=len(Coor_lin)
    Quad_over_row=np.zeros((NAO,6))
    if only_triangle:
        start=indx
    else:
        start=0
    for ii in range(start,NAO):
        Rik=Coor_lin[ii]-Coor_lin[indx]
        R0=np.zeros(3)
        Quad_over_row[ii,:]=quadrupole_STO(R0,Rik,np.array(Coeffs_lin[indx]),np.array(Coeffs_lin[ii]),np.array(Exp_lin[indx]),np.array(Exp_lin[ii]),np.array(Orient_lin[indx]),np.array(Orient_lin[ii]))

    return Quad_over_row

class Atom:
    ''' Class containing atom type and index(position in structure) 
    type : string
        Atom type e.g. 'C' or 'H'...
    indx : integer
        Position of atom in structure class. !Starting from 0!
    '''
    def __init__(self,typ,indx):
        self.type=typ
        self.indx=int(indx)

class AO:
    ''' Class containing all information about atomic orbitals 
    
    name : string
        Name of the atomic basis e.g. 6-31G*,..
    coeff : list of numpy.arrays of real
        For every atomic orbital contains expansion coefficients of STO orbital
        into GTO (more expained in Notes)
    exp : list of numpy.arrays of real
        For every atomic orbital contains exponents for GTO obitals in expansion
        of STO (more explaind in Notes)
    coor : Coordinate class - position units managed
        Information about center of every atomic orbital. Units are coor.units.
        Dimension of coor.value is (dimension Norbitals x 3)
    type : list of string and integer (dimension Norbitals x 2)
        Orbital types for every orbital (for example ``'s', 'p', 'd', '5d', 'f', '7f'``)
    atom : list of Atom class (dimesion Norbitals)
        For every orbital there is a list with atom information. (atom[i].indx
        index of the atom in structure class, atom[i].type string with atom type)
    nao : integer
        Number of orbitals
    nao_orient : integer
        number of atomic orbitals with orientation (for 's' type orbital there 
        is only one orientation, for 'p' orbital there are 3 orientaions - x,y 
        and z and so on for oter orbitals) = total number of atomic orbital 
        basis functions
    orient : list (dimension Norbitals)
        For every atomic orbital type there is a list with possible atomic orbital
        spatial orientations (e.g. one possible orientation could be for f orbital
        [2,0,1] which correspond to X^2Z spatial orinetation or for d orbital [0,1,1]
        correspond to YZ spatial orientation)
    indx_orient : list (dimension Norbitals_orient)
        For every spatialy oriented orbital there is a list with a number of atomic
        orbitals to which this orientation corresponds, at the fisrt position and 
        orientation of the orbital at the second position. (e.g [2, [0, 1, 0]]
        which means orientation in y direction of third orbital (numbering from 0)
        which is p type orbital (summ of all numbers in orient. is 1)) 
    overlap : numpy.array of real (dimension N_AO_orient x N_AO_orient)
        Overlap matrix between AO: overlap[i,j]=<AO_i|AO_j>
    dipole : dictionary
        * **dipole['Dip_X']** = numpy.array of real (dimension N_AO_orient x N_AO_orient)
          with dipole x coordinate in AO basis: dipole['Dip_X'][i,j]=<AO_i|x|AO_j>
        * **dipole['Dip_Y']** = numpy.array of real (dimension N_AO_orient x N_AO_orient)
          with dipole y coordinate in AO basis: dipole['Dip_Y'][i,j]=<AO_i|y|AO_j>
        * **dipole['Dip_Z']** = numpy.array of real (dimension N_AO_orient x N_AO_orient)
          with dipole z coordinate in AO basis: dipole['Dip_Z'][i,j]=<AO_i|z|AO_j>
    grid : list of numpy arrays of float (dimension Nao x Grid_Nx x Grid_Ny x Grid_Nz)
        Slater orbital basis evaluated on given grid
        
    quadrupole : numpy.array of real (dimension 6 x N_AO_orient x N_AO_orient)
        quadrupole components in AO basis:
        
        * quadrupole[0,:,:] = quadrupole xx matrix <AO_i|x^2|AO_j>
        * quadrupole[1,:,:] = quadrupole xy matrix <AO_i|xy|AO_j>
        * quadrupole[2,:,:] = quadrupole xz matrix <AO_i|xz|AO_j>
        * quadrupole[3,:,:] = quadrupole yy matrix <AO_i|yy|AO_j>
        * quadrupole[4,:,:] = quadrupole yz matrix <AO_i|yz|AO_j>
        * quadrupole[5,:,:] = quadrupole zz matrix <AO_i|zz|AO_j>
    
    Functions
    -----------
    add_orbital : 
        Add atomic orbital including expansion coefficients into gaussian orbitals
        coordinates, type, atom on which is centered.
    rotate :
        Rotate the atomic orbitals and all aditional quantities by specified
        angles in radians in positive direction.
    rotate_1 :
        Inverse totation to rotate
    move :
        Moves the atomic orbitals and all aditional quantities along specified
        vector
    copy :
        Create 1 to 1 copy of the atomic orbitals with all classes and types.
    get_overlap : 
        Calculate overlap matrix between atomic orbitals
    import_multiwfn_overlap :
        Imports the overlap matrix from multiwfn output
    get_dipole_matrix :
        Calculate dipoles between each pair of atomic orbitals and outputs it 
        as matrix
    get_quadrupole :
        Calculate quadrupole moments between each pair of atomic orbitals and 
        outputs it as matrix
    get_slater_ao_grid :
        Evaluate slater atomic orbital on given grid
    get_all_slater_grid
        Evalueate all slater orbitals on given grid (create slater orbital basis
        on given grid)
    
    Notes
    ----------
    Expansion of STO (slater orbital) into GTO (gaussian orbital) bassis
    is defined as: STO=Sum(coef*GTO(r,exp)*NormGTO(r,exp)) where r is center of
    the orbital (position of the corresponding atom)

    '''
    
    def __init__(self):
        self.name='AO-basis'
        self.coeff=[]
        self.exp=[]
        self.coor=None
        self.type=[]
        self.atom=[]
        self.nao=0
        self.nao_orient=0
        self.orient=[]
        self.indx_orient=[]
        self.init=False
        self.overlap=None
        self.dipole=None
        self.quadrupole=None
        self.grid=None
    
    def add_orbital(self,coeffs,exps,coor,orbit_type,atom_indx,atom_type):
        """ Adds atomic orbital including all needed informations
        
        Parameters
        ----------
        coeffs : numpy array or list of floats 
            Expansion coefficients of the slater atomic orbital into gaussian 
            atomic orbitals
        exps : numpy array or list of floats 
            Exponents of gaussian orbitals in expansion of the slater atomic
            orbital.
        coor : Coordinate class
            Centers of atomic orbitals (position units managed) 
        orbit_type : string
            Type of the orbital e.g. 's','d','5d'...
        atom_indx : integer
            index of atom on which orbital is centered
        atom_type : string
            Atom type on which orbital is cenetered e.g. 'C','H',...
        
        """
        if type(atom_type)==list or type(atom_type)==np.ndarray:
            if not self.init:
                if type(coeffs)==np.ndarray:
                    self.coeff=list(coeffs)         # it should be list of numpy arrays
                elif type(coeffs)==list:
                    self.coeff=coeffs.copy()
                else:
                    raise IOError('Imput expansion coefficients of AO should be list of numpy arrays or numpy array')
                if type(exps)==np.ndarray:
                    self.exp=list(exps)         # it should be list of numpy arrays
                elif type(coeffs)==list:
                    self.exp=exps.copy()
                else:
                    raise IOError('Imput expansion coefficients of AO should be list of numpy arrays or numpy array')
                self.coor=Coordinate(coor)       # assuming that we all arbital coordinates will be imputed in Bohrs
                self.type=orbit_type
                if type(atom_indx)==list:
                    for ii in range(len(atom_indx)):
                        self.atom.append(Atom(atom_type[ii],atom_indx[ii]))
                else:
                    self.atom.append(Atom(atom_type,atom_indx))                
                self.nao=len(orbit_type)
                for ii in range(len(self.type)):
                    orient=l_orient(self.type[ii])
                    self.orient.append(orient)
                    self.nao_orient+=len(orient)
                    for jj in range(len(orient)):
                        self.indx_orient.append([ii,orient[jj]])
                self.init=True
            else:
                self.coor.add_coor(coor)
                for ii in range(len(orbit_type)):
                    self.coeff.append(np.array(coeffs[ii],dtype='f8'))
                    self.exp.append(np.array(exps[ii],dtype='f8'))
                    self.type.append(orbit_type[ii])
                    self.atom.append(Atom(atom_type[ii],int(atom_indx[ii])))
                    self.nao+=1
                    orient=l_orient(orbit_type[ii])
                    self.orient.append(orient)
                    self.nao_orient+=len(orient)
                    for jj in range(len(orient)):
                        self.indx_orient.append([self.nao-1,orient[jj]])
        else:
            if not self.init:
                self.coor=Coordinate(coor)
            else:
                self.coor.add_coor(coor)
            
            self.coeff.append(np.array(coeffs,dtype='f8'))
            self.exp.append(np.array(exps,dtype='f8'))
            self.type.append(orbit_type)
            self.atom.append(Atom(atom_type,atom_indx))
            self.nao+=1
            orient=l_orient(orbit_type[0])
            self.orient.append(orient)
            self.nao_orient+=len(orient)
            for jj in range(len(orient)):
                self.indx_orient.append([self.nao-1,orient[jj]])
            self.init=True
    
    def get_overlap(self,nt=0,verbose=False):
        """ Calculate overlap matrix between atomic orbitals
        
        Parameters
        ----------
        nt : integer (optional init = 0)
            Specifies how many cores should be used for the calculation.
            Secial cases: 
            ``nt=0`` all available cores are used for the calculation. 
            ``nt=1`` serial calculation is performed.
        verbose : logical (optional init = False)
            If ``True`` information about time needed for overlap calculation 
            will be printed
        
        Notes
        ---------
        Overlap matrix is stored in:\n
        **self.overlap** \n
        as numpy array of float dimension (Nao_orient x Nao_orient)
        
        """
        
        # Toto by chtelo urcite zefektivnit pomoci np
        if (platform=='cygwin' or platform=="linux" or platform == "linux2") and nt!=1  and nt>=0:
            typ='paralell'
        elif platform=='win32' or nt==1:
            typ='seriall'
        else:
            typ='seriall_old'
            
        if typ=='seriall' or typ=='paralell':
            SS=np.zeros((self.nao_orient,self.nao_orient),dtype='f8')
            start_time = timeit.default_timer()
    
            ''' Convert all imput parameters into matrix which has dimension Nao_orient '''      
            # prepare imput
            Coor_lin=np.zeros((self.nao_orient,3))
            Coeffs_lin=[]
            Exp_lin=[]
            Orient_lin=[]
            counter1=0
            for ii in range(self.nao):
                for jj in range(len(self.orient[ii])):
                    Coor_lin[counter1]=self.coor._value[ii]
                    Coeffs_lin.append(self.coeff[ii])
                    Exp_lin.append(self.exp[ii])
                    Orient_lin.append(self.orient[ii][jj])
                    counter1+=1
                
            ao_over_line_partial = partial(_ao_over_line, Coor_lin,Coeffs_lin,Exp_lin,Orient_lin)
            # Only parameter of this function is number of row whih is calculated
            
        elif typ=='seriall_old':
            SS=np.zeros((self.nao_orient,self.nao_orient),dtype='f8')
            counter1=0
            start_time = timeit.default_timer()
            percent=0
            for ii in range(self.nao):
                for jj in range(len(self.orient[ii])):
                    counter2=0
                    for kk in range(self.nao):
                        for ll in  range(len(self.orient[kk])):
                            if counter1>=counter2:
                                SS[counter1,counter2] = overlap_STO(self.coor._value[ii],self.coor._value[kk],self.coeff[ii],self.coeff[kk],self.exp[ii],self.exp[kk],self.orient[ii][jj],self.orient[kk][ll])
                            counter2 += 1
                    counter1 += 1
                elapsed = timeit.default_timer() - start_time
                if elapsed>60.0:
                    if percent!=(counter1*10//self.nao_orient)*10:
                        if percent==0:
                            print('Overlap matrix calculation progres:')
                        percent=(counter1*10//self.nao_orient)*10                    
                        print(percent,'% ',sep="",end="")
            if verbose:
                print('Elapsed time for serial overlap matrix allocation:',elapsed)
    
            for ii in range(self.nao_orient):
                for jj in range(ii+1,self.nao_orient):
                    SS[ii,jj]=SS[jj,ii]
            if verbose:
                print(' ')
            self.overlap=np.copy(SS)
            

        if typ=='paralell':        
            ''' Parallel part '''
#            print('Prepairing parallel calculation')
            if nt>0:
                pool = Pool(processes=nt)
            else:
                pool = Pool(processes=cpu_count())
            index_list=range(self.nao_orient)
            SS= np.array(pool.map(ao_over_line_partial,index_list))
            pool.close() # ATTENTION HERE
            pool.join()
        elif typ=='seriall':
            index_list=range(self.nao_orient)
            SS=np.zeros((self.nao_orient,self.nao_orient))
            ''' Seriall part '''
            for ii in range(self.nao_orient):
                SS[ii,:]=ao_over_line_partial(index_list[ii])            
            
        ''' Fill the lower triangle of overlap matrix'''
        for ii in range(self.nao_orient):
            for jj in range(ii):
                SS[ii,jj]=SS[jj,ii] 
        elapsed = timeit.default_timer() - start_time
        
        if verbose:
            if typ=='paralell':
                print('Elapsed time for parallel overlap matrix allocation:',elapsed)
            elif typ=='seriall':
                print('Elapsed time for seriall overlap matrix allocation:',elapsed)
        
        self.overlap=np.copy(SS)
    
# TODO: include multiwfn script for generation of overlap matrix
    def import_multiwfn_overlap(self,filename):
        """ Import overlap matrix from multiwfn output
        
        Parameters
        ----------
        filename : string
            Name of the input file including the path if needed. (output file 
            from multiwfn calculation)
        
        Notes
        ---------
        Overlap matrix is stored in:\n
        **self.overlap** \n
        as numpy array of float dimension (Nao_orient x Nao_orient)
        
        """
        
        fid    = open(filename,'r')   # Open the file
        flines = fid.readlines()      # Read the WHOLE file into RAM
        fid.close()   
    
        counter=0
        Norb=self.nao_orient
        SS_inp=np.zeros((Norb,Norb))
        for jj in range(Norb//5+1):
            for ii in range(5*jj,Norb+1):
                if ii!=5*jj:
                    line = flines[counter]
                    thisline = line.split()
                    for kk in range(5):
                        if kk+5*jj+1<=ii:
                            if 'D' in thisline[kk+1]:
                                SS_inp[ii-1,kk+5*jj]=float(thisline[kk+1].replace('D', 'e'))
                            else:
                                SS_inp[ii-1,kk+5*jj]=float(thisline[kk+1][:-4]+'e'+thisline[kk+1][-4:])
                                #print(thisline[kk+1],SS_inp[ii-1,kk+5*jj])
                    #print(5*jj,ii-1,flines[counter])
                counter+=1
        for ii in range(Norb):
            for jj in range(ii+1,Norb):
                SS_inp[ii,jj]=SS_inp[jj,ii]
        
        self.overlap=SS_inp
    
    def get_dipole_matrix(self,nt=0,verbose=False):
        """ Calculate dipole matrix between atomic orbitals
        
        Parameters
        ----------
        nt : integer (optional init = 0)
            Specifies how many cores should be used for the calculation.
            Secial cases: 
            ``nt=0`` all available cores are used for the calculation. 
            ``nt=1`` serial calculation is performed.
        verbose : logical (optional init = False)
            If ``True`` information about time needed for overlap calculation 
            will be printed
        
        Notes
        ---------
        Dipole matrix is stored in:\n
        **self.dipole** \n
        as dictionary of numpy arrays of float dimension (Nao_orient x Nao_orient).
        Dictionary has 3 keys: 'Dip_X', 'Dip_Y', 'Dip_Z'. More information can
        be found in class documentation
        
        """
        
        # select platform
        if (platform=='cygwin' or platform=="linux" or platform == "linux2") and nt!=1  and nt>=0:
            typ='paralell'
        elif platform=='win32' or nt==1:
            typ='seriall'
        else:
            typ='seriall_old'
            
        start_time = timeit.default_timer()
        SS_dipX=np.zeros((self.nao_orient,self.nao_orient),dtype='f8')
        SS_dipY=np.zeros((self.nao_orient,self.nao_orient),dtype='f8')
        SS_dipZ=np.zeros((self.nao_orient,self.nao_orient),dtype='f8')
        
        if typ=='seriall' or typ=='paralell':
            ''' Convert all imput parameters into matrix which has dimension Nao_orient '''      
            # prepare imput
            Coor_lin=np.zeros((self.nao_orient,3))
            Coeffs_lin=[]
            Exp_lin=[]
            Orient_lin=[]
            counter1=0
            for ii in range(self.nao):
                for jj in range(len(self.orient[ii])):
                    Coor_lin[counter1]=self.coor._value[ii]
                    Coeffs_lin.append(self.coeff[ii])
                    Exp_lin.append(self.exp[ii])
                    Orient_lin.append(self.orient[ii][jj])
                    counter1+=1
                
            dip_over_line_partial = partial(_dip_over_line, Coor_lin,Coeffs_lin,Exp_lin,Orient_lin)
            # Only parameter of this function is number of row whih is calculated
            
        else:
            counter1=0
            for ii in range(self.nao):
                for jj in range(len(self.orient[ii])):
                    counter2=0
                    for kk in range(self.nao):
                        for ll in  range(len(self.orient[kk])):
                            if counter1>=counter2:
                                dipole=dipole_STO(self.coor._value[ii],self.coor._value[kk],self.coeff[ii],self.coeff[kk],self.exp[ii],self.exp[kk],self.orient[ii][jj],self.orient[kk][ll])
                                SS_dipX[counter1,counter2] = dipole[0]
                                SS_dipY[counter1,counter2] = dipole[1]
                                SS_dipZ[counter1,counter2] = dipole[2]
                            counter2 += 1
                    counter1 += 1
            for ii in range(self.nao_orient):
                for jj in range(ii+1,self.nao_orient):
                    SS_dipX[ii,jj]=SS_dipX[jj,ii]
                    SS_dipY[ii,jj]=SS_dipY[jj,ii]
                    SS_dipZ[ii,jj]=SS_dipZ[jj,ii]
            elapsed = timeit.default_timer() - start_time
            if verbose:
                print('Elapsed time slater dipole matrix allocation', elapsed)
            
            self.dipole={}
            self.dipole['Dip_X']=np.copy(SS_dipX)
            self.dipole['Dip_Y']=np.copy(SS_dipY)
            self.dipole['Dip_Z']=np.copy(SS_dipZ)
        
        if typ=='paralell':        
            ''' Parallel part '''
#            print('Prepairing parallel calculation')
            if nt>0:
                pool = Pool(processes=nt)
            else:
                pool = Pool(processes=cpu_count())
            index_list=range(self.nao_orient)
            DipMat= np.array(pool.map(dip_over_line_partial,index_list))
            pool.close() # ATTENTION HERE
            pool.join()
        elif typ=='seriall':
            index_list=range(self.nao_orient)
            DipMat=np.zeros((self.nao_orient,self.nao_orient,3))
            ''' Seriall part '''
            for ii in range(self.nao_orient):
                DipMat[ii,:]=dip_over_line_partial(index_list[ii])            
            
        if typ=='seriall' or typ=='paralell':
            ''' Fill the lower triangle of overlap matrix'''
            for ii in range(self.nao_orient):
                for jj in range(ii):
                    DipMat[ii,jj,:]=DipMat[jj,ii,:] 
            elapsed = timeit.default_timer() - start_time
        
        if verbose:
            if typ=='paralell':
                print('Elapsed time for parallel slater dipole matrix allocation:',elapsed)
            elif typ=='seriall':
                print('Elapsed time for seriall slater dipole matrix allocation:',elapsed)

        if typ=='seriall' or typ=='paralell':
            self.dipole={}
            self.dipole['Dip_X']=np.zeros((self.nao_orient,self.nao_orient))
            self.dipole['Dip_Y']=np.zeros((self.nao_orient,self.nao_orient))
            self.dipole['Dip_Z']=np.zeros((self.nao_orient,self.nao_orient))
            self.dipole['Dip_X'][:,:]=np.copy(DipMat[:,:,0])
            self.dipole['Dip_Y'][:,:]=np.copy(DipMat[:,:,1])
            self.dipole['Dip_Z'][:,:]=np.copy(DipMat[:,:,2])
    
    
    def get_quadrupole(self,nt=0,verbose=False):
        """ Calculate quadrupole matrix between atomic orbitals
        
        Parameters
        ----------
        nt : integer (optional init = 0)
            Specifies how many cores should be used for the calculation.
            Secial cases: 
            ``nt=0`` all available cores are used for the calculation. 
            ``nt=1`` serial calculation is performed.
        verbose : logical (optional init = False)
            If ``True`` information about time needed for overlap calculation 
            will be printed
        
        Notes
        ---------
        Dipole matrix is stored in:\n
        **self.quadrupole** \n
        as numpy array of float dimension (6 x Nao_orient x Nao_orient) and
        ordering of quadrupole moments is: xx, xy, xz, yy, yz, zz \n \n
        quadrupoles are defined as ``Qij(mu,nu) = \int{AO_mu(r+R_mu)*ri*rj*AO_nu(r+R_mu)}``\n
        The AO_mu is shifted to zero making the quadrupoles independent to coordinate shifts
        
        """
        
        QuadMat=np.zeros((6,self.nao_orient,self.nao_orient),dtype='f8')
        start_time = timeit.default_timer()

        do_faster=False
        if (self.dipole is not None) and (self.overlap is not None):
            do_faster=True
        
        # choose platform for calculation
        if (platform=='cygwin' or platform=="linux" or platform == "linux2") and nt!=1 and nt>=0:
            typ='paralell'
        elif platform=='win32' or nt==1:
            typ='seriall'
        else:
            typ='seriall_old'
            
        if typ=='seriall' or typ=='paralell':
            ''' Convert all imput parameters into matrix which has dimension Nao_orient '''      
            # prepare imput
            Coor_lin=np.zeros((self.nao_orient,3))
            Coeffs_lin=[]
            Exp_lin=[]
            Orient_lin=[]
            counter1=0
            for ii in range(self.nao):
                for jj in range(len(self.orient[ii])):
                    Coor_lin[counter1]=self.coor._value[ii]
                    Coeffs_lin.append(self.coeff[ii])
                    Exp_lin.append(self.exp[ii])
                    Orient_lin.append(self.orient[ii][jj])
                    counter1+=1
                
            quad_over_line_partial = partial(_quad_over_line, Coor_lin,Coeffs_lin,Exp_lin,Orient_lin,do_faster)
            # Only parameter of this function is number of row whih is calculated
        else:
            counter1=0
            for ii in range(self.nao):
                for jj in range(len(self.orient[ii])):
                    counter2=0
                    for kk in range(self.nao):
                        for ll in  range(len(self.orient[kk])):
                            Rik=self.coor._value[kk]-self.coor._value[ii]
                            R0=np.zeros(3)
                            QuadMat[:,counter1,counter2]=quadrupole_STO(R0,Rik,self.coeff[ii],self.coeff[kk],self.exp[ii],self.exp[kk],self.orient[ii][jj],self.orient[kk][ll])
                            counter2 += 1
                    counter1 += 1
                    
            elapsed = timeit.default_timer() - start_time
            print('Elapsed time for slater quadrupole matrix allocation:',elapsed)
        
        if typ=='paralell':        
            ''' Parallel part '''
#            print('Prepairing parallel calculation')
            if nt>0:
                pool = Pool(processes=nt)
            else:
                pool = Pool(processes=cpu_count())
            index_list=range(self.nao_orient)
            QuadMat_tmp= np.array(pool.map(quad_over_line_partial,index_list))
            pool.close() # ATTENTION HERE
            pool.join()
        elif typ=='seriall':
            index_list=range(self.nao_orient)
            ''' Seriall part '''
            for ii in range(self.nao_orient):
                QuadMat[:,ii,:]=np.swapaxes(quad_over_line_partial(index_list[ii]),0,1)            
                
        ''' Fill the lower triangle of overlap matrix'''
        if typ=='seriall' and do_faster:
            for ii in range(self.nao_orient):
                for jj in range(ii):
                    counter=0
                    Rji=self.coor._value[self.indx_orient[ii][0]]-self.coor._value[self.indx_orient[jj][0]]
                    Rj=self.coor._value[self.indx_orient[jj][0]]
                    Dji=np.array([self.dipole['Dip_X'][jj,ii],self.dipole['Dip_Y'][jj,ii],self.dipole['Dip_Z'][jj,ii]])
                    SSji=self.overlap[jj,ii]                    
                    for kk in range(3):
                        for ll in range(kk,3):
                            QuadMat[counter,ii,jj]=QuadMat[counter,jj,ii]-Rji[kk]*Dji[ll]-Rji[ll]*Dji[kk]+(Rj[kk]*Rji[ll]+Rj[ll]*Rji[kk]+Rji[kk]*Rji[ll])*SSji
                            counter+=1
        if typ=='paralell' and do_faster:
            for ii in range(self.nao_orient):
                for jj in range(ii):
                    counter=0
                    Rji=self.coor._value[self.indx_orient[ii][0]]-self.coor._value[self.indx_orient[jj][0]]
                    Rj=self.coor._value[self.indx_orient[jj][0]]
                    Dji=np.array([self.dipole['Dip_X'][jj,ii],self.dipole['Dip_Y'][jj,ii],self.dipole['Dip_Z'][jj,ii]])
                    SSji=self.overlap[jj,ii]                    
                    for kk in range(3):
                        for ll in range(kk,3):
                            QuadMat_tmp[ii,jj,counter]=QuadMat_tmp[jj,ii,counter]-Rji[kk]*Dji[ll]-Rji[ll]*Dji[kk]+(Rj[kk]*Rji[ll]+Rj[ll]*Rji[kk]+Rji[kk]*Rji[ll])*SSji
                            counter+=1
            QuadMat=np.copy(np.swapaxes(QuadMat_tmp,0,2))
            QuadMat=np.copy(np.swapaxes(QuadMat,1,2))
                    
        elapsed = timeit.default_timer() - start_time
        
        if verbose:
            if typ=='paralell':
                print('Elapsed time for parallel slater quadrupole matrix allocation:',elapsed)
            elif typ=='seriall':
                print('Elapsed time for seriall slater quadrupoele matrix allocation:',elapsed)        
                
        self.quadrupole=np.copy(QuadMat)
        
    
#    def get_quadrupole_old(self,nt=0,verbose=False):
#        """ Calculate quadrupole matrix between atomic orbitals
#        
#        Parameters
#        ----------
#        nt : integer (optional init = 0)
#            Specifies how many cores should be used for the calculation.
#            Secial cases: 
#            ``nt=0`` all available cores are used for the calculation. 
#            ``nt=1`` serial calculation is performed.
#        verbose : logical (optional init = False)
#            If ``True`` information about time needed for overlap calculation 
#            will be printed
#        
#        Notes
#        ---------
#        Dipole matrix is stored in:\n
#        **self.quadrupole** \n
#        as numpy array of float dimension (6 x Nao_orient x Nao_orient) and
#        ordering of quadrupole moments is: xx, xy, xz, yy, yz, zz
#        
#        """
#        
#        QuadMat=np.zeros((6,self.nao_orient,self.nao_orient),dtype='f8')
#        start_time = timeit.default_timer()
#
#        do_faster=False
#        if (self.dipole is not None) and (self.overlap is not None):
#            do_faster=True
#        
#        # choose platform for calculation
#        if (platform=='cygwin' or platform=="linux" or platform == "linux2") and nt!=1 and nt>=0:
#            typ='paralell'
#        elif platform=='win32' or nt==1:
#            typ='seriall'
#        else:
#            typ='seriall_old'
#            
#        if typ=='seriall' or typ=='paralell':
#            ''' Convert all imput parameters into matrix which has dimension Nao_orient '''      
#            # prepare imput
#            Coor_lin=np.zeros((self.nao_orient,3))
#            Coeffs_lin=[]
#            Exp_lin=[]
#            Orient_lin=[]
#            counter1=0
#            for ii in range(self.nao):
#                for jj in range(len(self.orient[ii])):
#                    Coor_lin[counter1]=self.coor._value[ii]
#                    Coeffs_lin.append(self.coeff[ii])
#                    Exp_lin.append(self.exp[ii])
#                    Orient_lin.append(self.orient[ii][jj])
#                    counter1+=1
#                
#            quad_over_line_partial = partial(_quad_over_line, Coor_lin,Coeffs_lin,Exp_lin,Orient_lin,do_faster)
#            # Only parameter of this function is number of row whih is calculated
#        else:
#            counter1=0
#            for ii in range(self.nao):
#                for jj in range(len(self.orient[ii])):
#                    counter2=0
#                    for kk in range(self.nao):
#                        for ll in  range(len(self.orient[kk])):
#                            Rik=self.coor._value[kk]-self.coor._value[ii]
#                            R0=np.zeros(3)
#                            QuadMat[:,counter1,counter2]=quadrupole_STO(R0,Rik,self.coeff[ii],self.coeff[kk],self.exp[ii],self.exp[kk],self.orient[ii][jj],self.orient[kk][ll])
#                            counter2 += 1
#                    counter1 += 1
#                    
#            elapsed = timeit.default_timer() - start_time
#            print('Elapsed time for slater quadrupole matrix allocation:',elapsed)
#        
#        if typ=='paralell':        
#            ''' Parallel part '''
##            print('Prepairing parallel calculation')
#            if nt>0:
#                pool = Pool(processes=nt)
#            else:
#                pool = Pool(processes=cpu_count())
#            index_list=range(self.nao_orient)
#            QuadMat_tmp= np.array(pool.map(quad_over_line_partial,index_list))
#            pool.close() # ATTENTION HERE
#            pool.join()
#        elif typ=='seriall':
#            index_list=range(self.nao_orient)
#            ''' Seriall part '''
#            for ii in range(self.nao_orient):
#                QuadMat[:,ii,:]=np.swapaxes(quad_over_line_partial(index_list[ii]),0,1)            
#                
#        ''' Fill the lower triangle of overlap matrix'''
#        if typ=='seriall' and do_faster:
#            for ii in range(self.nao_orient):
#                for jj in range(ii):
#                    counter=0
#                    Rji=self.coor._value[self.indx_orient[ii][0]]-self.coor._value[self.indx_orient[jj][0]]
#                    Rj=self.coor._value[self.indx_orient[jj][0]]
#                    Dji=np.array([self.dipole['Dip_X'][jj,ii],self.dipole['Dip_Y'][jj,ii],self.dipole['Dip_Z'][jj,ii]])
#                    SSji=self.overlap[jj,ii]                    
#                    for kk in range(3):
#                        for ll in range(kk,3):
#                            QuadMat[counter,ii,jj]=QuadMat[counter,jj,ii]-Rji[kk]*Dji[ll]-Rji[ll]*Dji[kk]+(Rj[kk]*Rji[ll]+Rj[ll]*Rji[kk]+Rji[kk]*Rji[ll])*SSji
#                            counter+=1
#        if typ=='paralell' and do_faster:
#            for ii in range(self.nao_orient):
#                for jj in range(ii):
#                    counter=0
#                    Rji=self.coor._value[self.indx_orient[ii][0]]-self.coor._value[self.indx_orient[jj][0]]
#                    Rj=self.coor._value[self.indx_orient[jj][0]]
#                    Dji=np.array([self.dipole['Dip_X'][jj,ii],self.dipole['Dip_Y'][jj,ii],self.dipole['Dip_Z'][jj,ii]])
#                    SSji=self.overlap[jj,ii]                    
#                    for kk in range(3):
#                        for ll in range(kk,3):
#                            QuadMat_tmp[ii,jj,counter]=QuadMat_tmp[jj,ii,counter]-Rji[kk]*Dji[ll]-Rji[ll]*Dji[kk]+(Rj[kk]*Rji[ll]+Rj[ll]*Rji[kk]+Rji[kk]*Rji[ll])*SSji
#                            counter+=1
#            QuadMat=np.copy(np.swapaxes(QuadMat_tmp,0,2))
#            QuadMat=np.copy(np.swapaxes(QuadMat,1,2))
#                    
#        elapsed = timeit.default_timer() - start_time
#        
#        if verbose:
#            if typ=='paralell':
#                print('Elapsed time for parallel slater quadrupole matrix allocation:',elapsed)
#            elif typ=='seriall':
#                print('Elapsed time for seriall slater quadrupoele matrix allocation:',elapsed)        
#                
#        self.quadrupole=np.copy(QuadMat)
    
    def get_slater_ao_grid(self,grid,indx,keep_grid=False,new_grid=True):  # Jediny je spravne se spravnou normalizaci  
        """ Evaluate single slater orbital on given grid
        
        Parameters
        ----------
        grid : Grid class
            Information about grid on which slater atomic orbital is evaluated.
        indx : 
            Index of atomic orbital whic is evaluated (position in indx_orient)
        keep_grid : logical (optional init = False)
            If ``True`` local grid (dependent on orbital center) is kept as
            global internal variable in order to avoid recalculation of the
            grid for calculation of more orientations of the same orbital.
        new_grid : logical (optional init = True)
            If ``True`` local grid (dependent on orbital center) is recalculated
            and the old one is overwriten. It is needed if local grid for orbital
            with different center was previously saved.
            
        Returns
        ---------
        slater_ao_tmp : numpy array of float (dimension Grid_Nx x Grid_Ny x Grid_Nz)
            Values of slater orbital on grid points defined by grid.
        
        """
        
        
        slater_ao_tmp=np.zeros(np.shape(grid.X)) 
        ii=self.indx_orient[indx][0]    
    #    print(indx,'/',mol.ao_spec['Nao_orient'])
        if new_grid:
            global X_grid_loc,Y_grid_loc,Z_grid_loc,RR_grid_loc
            X_grid_loc=np.add(grid.X,-self.coor._value[ii][0])  # posunu grid vzdy tak abych dostal centrum orbitalu do nuly.
            Y_grid_loc=np.add(grid.Y,-self.coor._value[ii][1])
            Z_grid_loc=np.add(grid.Z,-self.coor._value[ii][2])
            RR_grid_loc=np.square(X_grid_loc)+np.square(Y_grid_loc)+np.square(Z_grid_loc)
            # Vytvorena souradnicova sit a vsechny souradnice rozlozeny na grid
        
        # Pro kazdou orientaci pouziji ruzny slateruv atomovy orbital kvuli ruzne normalizaci gaussovskych orbitalu
        # Vypocet slaterova orbitalu na gridu pro AO=ii a orientaci ao_comb[jj+index]
        if self.type[ii][0] in ['s','p','d','f','5d']:
            slater_ao=np.zeros(np.shape(grid.X))
            for kk in range(len(self.coeff[ii])):
                coef=self.coeff[ii][kk]
                exp=self.exp[ii][kk]
                
                r_ao= self.coor._value[ii]
                norm=norm_GTO(r_ao,exp,self.indx_orient[indx][1])
                c=coef*norm
                slater_ao += np.multiply(c,np.exp(np.multiply(-exp,RR_grid_loc)))
        else:
            raise IOError('Supported orbitals are so far only s,p,d,5d,f orbitals')
            
            #slateruv orbital vytvoren
        if self.type[ii][0] in ['s','p','d','f']:
            m=self.indx_orient[indx][1][0]
            n=self.indx_orient[indx][1][1]
            o=self.indx_orient[indx][1][2]            
            slater_ao_tmp=np.copy(slater_ao)
            if m!=0:
                slater_ao_tmp=np.multiply(np.power(X_grid_loc,m),slater_ao_tmp)
            if n!=0:
                slater_ao_tmp=np.multiply(np.power(Y_grid_loc,n),slater_ao_tmp)
            if o!=0:
                slater_ao_tmp=np.multiply(np.power(Z_grid_loc,o),slater_ao_tmp)
                
        elif self.type[ii][0]=='5d':                
                orient=self.indx_orient[indx][1] 
                
                if orient[0]==(-2):
                    ## 3Z^2-R^2
                    slater_ao_tmp=np.copy(slater_ao)
                    slater_ao_tmp=np.multiply(3,np.multiply(np.power(Z_grid_loc,2),slater_ao_tmp))
                    slater_ao_tmp=slater_ao_tmp-np.multiply(RR_grid_loc,slater_ao)
                elif orient[0]==(-1) and orient[2]==(-1):
                    ## XZ
                    slater_ao_tmp=np.copy(slater_ao)
                    slater_ao_tmp=np.multiply(Z_grid_loc,slater_ao_tmp)
                    slater_ao_tmp=np.multiply(X_grid_loc,slater_ao_tmp)
                elif orient[1]==(-1) and orient[2]==(-1):
                    ## YZ
                    slater_ao_tmp=np.copy(slater_ao)
                    slater_ao_tmp=np.multiply(Z_grid_loc,slater_ao_tmp)
                    slater_ao_tmp=np.multiply(Y_grid_loc,slater_ao_tmp)
                elif orient[1]==(-2):
                    #X^2-Y^2
                    slater_ao_tmp=np.copy(slater_ao)
                    slater_ao_tmp=np.multiply(np.power(X_grid_loc,2),slater_ao_tmp)
                    slater_ao_tmp=slater_ao_tmp-np.multiply(np.power(Y_grid_loc,2),slater_ao)
                elif orient[0]==(-1) and orient[1]==(-1):
                    #XY
                    slater_ao_tmp=np.copy(slater_ao)
                    slater_ao_tmp=np.multiply(Y_grid_loc,slater_ao_tmp)
                    slater_ao_tmp=np.multiply(X_grid_loc,slater_ao_tmp)
                else:
                    raise IOError('5d orbital has only 5 orientations')  

        if not keep_grid:
            del X_grid_loc
            del Y_grid_loc
            del Z_grid_loc
            del RR_grid_loc
        return np.array(slater_ao_tmp)

    def get_all_slater_grid(self,grid,nt=0):  # Jediny je spravne se spravnou normalizaci  
        """ Evaluate all slater orbitals on given grid. This way basis for 
        calculation of transition densities or other electron densities.
        
        Parameters
        ----------
        grid : Grid class
            Information about grid on which slater atomic orbital is evaluated.
        nt : integer (optional init = 0)
            Specifies how many cores should be used for the calculation.
            Secial cases: 
            ``nt=0`` all available cores are used for the calculation. 
            ``nt=1`` serial calculation is performed.
            
        Notes
        ---------
        All oriented slater atomic orbitals evaluated on given grid are stored at:\n
        **self.grid** \n
        as a list (size nao_orient) of numpy arrays of float 
        (dimension Grid_Nx x Grid_Ny x Grid_Nz)
        
        """
        
# TODO: add check of OS and decide if calculate serial or parallel
        All_slater_grid=[]
        do_parallel=False
        if nt==1:   # serial execution
            counter=0
            keep_grid=True
            for ii in range(self.nao):
                new_grid=True
                print(ii)
                if ii==0:
                    new_grid=True
                elif (ii>0) and (self.atom[ii].indx==self.atom[ii-1].indx):
                    new_grid=False
                for jj in range(len(self.orient[ii])):
                    if counter==self.nao_orient:
                        keep_grid=False
                    All_slater_grid.append(self.get_slater_ao_grid(grid,counter,keep_grid=keep_grid,new_grid=new_grid))
                    counter+=1 
        elif nt>1:
            from multiprocessing import Pool, cpu_count
            pool = Pool(processes=nt)
            do_parallel=True
        else:
            from multiprocessing import Pool, cpu_count
            pool = Pool(processes=cpu_count())
            do_parallel=True
        
        if do_parallel:
            index_list=range(self.nao_orient)
            allocate_single_slater_grid_mol_partial = partial(self.get_slater_ao_grid, grid)
            All_slater_grid_tmp= pool.map(allocate_single_slater_grid_mol_partial,index_list)
            pool.close() # ATTENTION HERE
            pool.join()
            All_slater_grid=np.copy(All_slater_grid_tmp)
        
        self.grid=All_slater_grid
        
    
    def _get_ao_rot_mat(self,rotxy,rotxz,rotyz):
            RotA={}
            RotB={}
            RotC={}
            orbit=['s','p','d','5d','f']
            for ii in orbit:
                RotA[ii],RotB[ii],RotC[ii]=fill_basis_transf_matrix(ii,rotxy,rotxz,rotyz)
            
            #TransfMat=np.dot(RotC[self.type[0][0]],np.dot(RotB[self.type[0][0]],RotA[self.type[0][0]]))
            TransfMat=np.dot(RotC[self.type[0]],np.dot(RotB[self.type[0]],RotA[self.type[0]]))
            for ii in range(1,self.nao):
                #Rot_tmp=np.dot(RotC[self.type[ii][0]],np.dot(RotB[self.type[ii][0]],RotA[self.type[ii][0]]))
                Rot_tmp=np.dot(RotC[self.type[ii]],np.dot(RotB[self.type[ii]],RotA[self.type[ii]]))
                TransfMat=scipy.linalg.block_diag(TransfMat,Rot_tmp)
            TransfMat=np.array(TransfMat)
            return TransfMat
    
    def _get_ao_rot_mat_1(self,rotxy,rotxz,rotyz):
            RotA={}
            RotB={}
            RotC={}
            orbit=['s','p','d','5d','f']
            for ii in orbit:
                RotA[ii],RotB[ii],RotC[ii]=fill_basis_transf_matrix(ii,-rotxy,-rotxz,-rotyz)
            
            TransfMat=np.dot(RotA[self.type[0][0]],np.dot(RotB[self.type[0][0]],RotC[self.type[0][0]]))
            for ii in range(1,self.nao):
                Rot_tmp=np.dot(RotA[self.type[ii][0]],np.dot(RotB[self.type[ii][0]],RotC[self.type[ii][0]]))
                TransfMat=scipy.linalg.block_diag(TransfMat,Rot_tmp)
            TransfMat=np.array(TransfMat)
            return TransfMat
        
        
    def _rotate_dipole(self,rotxy,rotxz,rotyz):

        RotA,RotB,RotC=fill_basis_transf_matrix('p',rotxy,rotxz,rotyz)
        
        Rot=np.dot(RotC,np.dot(RotB,RotA))
        
        TransfMat=self._get_ao_rot_mat(rotxy,rotxz,rotyz)
        
        # Transformation of atomic orbitals
        self.dipole['Dip_X']=np.dot(TransfMat,np.dot(self.dipole['Dip_X'],TransfMat.T))
        self.dipole['Dip_Y']=np.dot(TransfMat,np.dot(self.dipole['Dip_Y'],TransfMat.T))
        self.dipole['Dip_Z']=np.dot(TransfMat,np.dot(self.dipole['Dip_Z'],TransfMat.T))
        
        # Transformation of dipole coordinate (x,z,y)
        Dip_X_tmp=Rot[0,0]*self.dipole['Dip_X']+Rot[0,1]*self.dipole['Dip_Y']+Rot[0,2]*self.dipole['Dip_Z']
        Dip_Y_tmp=Rot[1,0]*self.dipole['Dip_X']+Rot[1,1]*self.dipole['Dip_Y']+Rot[1,2]*self.dipole['Dip_Z']
        Dip_Z_tmp=Rot[2,0]*self.dipole['Dip_X']+Rot[2,1]*self.dipole['Dip_Y']+Rot[2,2]*self.dipole['Dip_Z']
        self.dipole['Dip_X']=np.copy(Dip_X_tmp)
        self.dipole['Dip_Y']=np.copy(Dip_Y_tmp)
        self.dipole['Dip_Z']=np.copy(Dip_Z_tmp)
        
# TODO: Test this 
    def _rotate_dipole_1(self,rotxy,rotxz,rotyz):
        
        RotA,RotB,RotC=fill_basis_transf_matrix('p',-rotxy,-rotxz,-rotyz)
        Rot=np.dot(RotA,np.dot(RotB,RotC))
        TransfMat=self._get_ao_rot_mat_1(rotxy,rotxz,rotyz)
        
        # Transformation of atomic orbitals
        self.dipole['Dip_X']=np.dot(TransfMat,np.dot(self.dipole['Dip_X'],TransfMat.T))
        self.dipole['Dip_Y']=np.dot(TransfMat,np.dot(self.dipole['Dip_Y'],TransfMat.T))
        self.dipole['Dip_Z']=np.dot(TransfMat,np.dot(self.dipole['Dip_Z'],TransfMat.T))
        
        # Transformation of dipole coordinate (x,z,y)
        Dip_X_tmp=Rot[0,0]*self.dipole['Dip_X']+Rot[0,1]*self.dipole['Dip_Y']+Rot[0,2]*self.dipole['Dip_Z']
        Dip_Y_tmp=Rot[1,0]*self.dipole['Dip_X']+Rot[1,1]*self.dipole['Dip_Y']+Rot[1,2]*self.dipole['Dip_Z']
        Dip_Z_tmp=Rot[2,0]*self.dipole['Dip_X']+Rot[2,1]*self.dipole['Dip_Y']+Rot[2,2]*self.dipole['Dip_Z']
        self.dipole['Dip_X']=np.copy(Dip_X_tmp)
        self.dipole['Dip_Y']=np.copy(Dip_Y_tmp)
        self.dipole['Dip_Z']=np.copy(Dip_Z_tmp)
    
    def _rotate_quadrupole(self,rotxy,rotxz,rotyz): # This should be tested
        TransfMat=self._get_ao_rot_mat(rotxy,rotxz,rotyz)
        
        # quadrupole transformation  
        RotA,RotB,RotC=fill_basis_transf_matrix('d',rotxy,rotxz,rotyz)
        Rot=np.dot(RotC,np.dot(RotB,RotA))
        
        # first rotate orbitals and than rotate quadrupole
        for ii in range(6):
            self.quadrupole[ii]=np.dot(TransfMat,np.dot(self.quadrupole[ii],TransfMat.T))
        # and now rotate quadrupole !! d orbital definition is XX,YY,ZZ,XY,XZ,YZ  but our quadrupole definition is XX, XY, XZ, YY, YZ, ZZ
        quad_XX=Rot[0,0]*self.quadrupole[0]+Rot[0,1]*self.quadrupole[3]+Rot[0,2]*self.quadrupole[5]+Rot[0,3]*self.quadrupole[1]+Rot[0,4]*self.quadrupole[2]+Rot[0,5]*self.quadrupole[4]
        quad_YY=Rot[1,0]*self.quadrupole[0]+Rot[1,1]*self.quadrupole[3]+Rot[1,2]*self.quadrupole[5]+Rot[1,3]*self.quadrupole[1]+Rot[1,4]*self.quadrupole[2]+Rot[1,5]*self.quadrupole[4]
        quad_ZZ=Rot[2,0]*self.quadrupole[0]+Rot[2,1]*self.quadrupole[3]+Rot[2,2]*self.quadrupole[5]+Rot[2,3]*self.quadrupole[1]+Rot[2,4]*self.quadrupole[2]+Rot[2,5]*self.quadrupole[4]
        quad_XY=Rot[3,0]*self.quadrupole[0]+Rot[3,1]*self.quadrupole[3]+Rot[3,2]*self.quadrupole[5]+Rot[3,3]*self.quadrupole[1]+Rot[3,4]*self.quadrupole[2]+Rot[3,5]*self.quadrupole[4]
        quad_XZ=Rot[4,0]*self.quadrupole[0]+Rot[4,1]*self.quadrupole[3]+Rot[4,2]*self.quadrupole[5]+Rot[4,3]*self.quadrupole[1]+Rot[4,4]*self.quadrupole[2]+Rot[4,5]*self.quadrupole[4]
        quad_YZ=Rot[5,0]*self.quadrupole[0]+Rot[5,1]*self.quadrupole[3]+Rot[5,2]*self.quadrupole[5]+Rot[5,3]*self.quadrupole[1]+Rot[5,4]*self.quadrupole[2]+Rot[5,5]*self.quadrupole[4]
        self.quadrupole[0]=np.copy(quad_XX)
        self.quadrupole[1]=np.copy(quad_XY)
        self.quadrupole[2]=np.copy(quad_XZ)
        self.quadrupole[3]=np.copy(quad_YY)
        self.quadrupole[4]=np.copy(quad_YZ)
        self.quadrupole[5]=np.copy(quad_ZZ)

# TODO: test this:
    def _rotate_quadrupole_1(self,rotxy,rotxz,rotyz): # This should be tested
        TransfMat=self._get_ao_rot_mat_1(rotxy,rotxz,rotyz)
        
        # quadrupole transformation  
        RotA,RotB,RotC=fill_basis_transf_matrix('d',-rotxy,-rotxz,-rotyz)
        Rot=np.dot(RotA,np.dot(RotB,RotC))
        
        # first rotate orbitals and than rotate quadrupole
        for ii in range(6):
            self.quadrupole[ii]=np.dot(TransfMat,np.dot(self.quadrupole[ii],TransfMat.T))
        # and now rotate quadrupole !! d orbital definition is XX,YY,ZZ,XY,XZ,YZ  but our quadrupole definition is XX, XY, XZ, YY, YZ, ZZ
        quad_XX=Rot[0,0]*self.quadrupole[0]+Rot[0,1]*self.quadrupole[3]+Rot[0,2]*self.quadrupole[5]+Rot[0,3]*self.quadrupole[1]+Rot[0,4]*self.quadrupole[2]+Rot[0,5]*self.quadrupole[4]
        quad_YY=Rot[1,0]*self.quadrupole[0]+Rot[1,1]*self.quadrupole[3]+Rot[1,2]*self.quadrupole[5]+Rot[1,3]*self.quadrupole[1]+Rot[1,4]*self.quadrupole[2]+Rot[1,5]*self.quadrupole[4]
        quad_ZZ=Rot[2,0]*self.quadrupole[0]+Rot[2,1]*self.quadrupole[3]+Rot[2,2]*self.quadrupole[5]+Rot[2,3]*self.quadrupole[1]+Rot[2,4]*self.quadrupole[2]+Rot[2,5]*self.quadrupole[4]
        quad_XY=Rot[3,0]*self.quadrupole[0]+Rot[3,1]*self.quadrupole[3]+Rot[3,2]*self.quadrupole[5]+Rot[3,3]*self.quadrupole[1]+Rot[3,4]*self.quadrupole[2]+Rot[3,5]*self.quadrupole[4]
        quad_XZ=Rot[4,0]*self.quadrupole[0]+Rot[4,1]*self.quadrupole[3]+Rot[4,2]*self.quadrupole[5]+Rot[4,3]*self.quadrupole[1]+Rot[4,4]*self.quadrupole[2]+Rot[4,5]*self.quadrupole[4]
        quad_YZ=Rot[5,0]*self.quadrupole[0]+Rot[5,1]*self.quadrupole[3]+Rot[5,2]*self.quadrupole[5]+Rot[5,3]*self.quadrupole[1]+Rot[5,4]*self.quadrupole[2]+Rot[5,5]*self.quadrupole[4]
        self.quadrupole[0]=np.copy(quad_XX)
        self.quadrupole[1]=np.copy(quad_XY)
        self.quadrupole[2]=np.copy(quad_XZ)
        self.quadrupole[3]=np.copy(quad_YY)
        self.quadrupole[4]=np.copy(quad_YZ)
        self.quadrupole[5]=np.copy(quad_ZZ)
            
    def rotate(self,rotxy,rotxz,rotyz):
        """"
        Rotate the orbitals around the coordinate origin by specified angles.
        Orbitals will still point in the same direction (px orbital in x direction)
        but overlap between them changes and coordinates too. Dipole and quadrupole
        matrixies changes too. 
        First it rotate the structure around z axis (in xy plane), then around
        y axis and in the end around x axis in positive direction 
        (if right thumb pointing in direction of axis fingers are pointing in 
        positive rotation direction)
        
        Parameters
        --------
        rotxy,rotxz,rotxz : float
            Rotation angles in radians around z, y and x axis (in xy, xz and yz plane)
        
        """
        
        self.coor.rotate(rotxy,rotxz,rotyz)
        TransfMat=self._get_ao_rot_mat(rotxy,rotxz,rotyz)
        if self.overlap is not None:
            self.overlap=np.dot(TransfMat,np.dot(self.overlap,TransfMat.T))
        if self.dipole is not None:
            self._rotate_dipole(rotxy,rotxz,rotyz)
        if self.quadrupole is not None:
            self._rotate_quadrupole(rotxy,rotxz,rotyz)
    
    def rotate_1(self,rotxy,rotxz,rotyz):
        """" Inverse rotation of atomic orbitals to **rotate** function.
        First rotate the structure around x axis (in yz plane), then around
        y axis and in the end around z axis in negtive direction 
        (if left thumb pointing in direction of axis fingers are pointing in 
        negative rotation direction)
        
        Parameters
        --------
        rotxy,rotxz,rotxz : float
            Rotation angles in radians around z, y and x axis (in xy, xz and yz plane)
        """
        
        self.coor.rotate_1(rotxy,rotxz,rotyz)
        TransfMat=self._get_ao_rot_mat_1(rotxy,rotxz,rotyz)
        if self.overlap is not None:
            self.overlap=np.dot(TransfMat,np.dot(self.overlap,TransfMat.T))
        if self.dipole is not None:
            self._rotate_dipole_1(rotxy,rotxz,rotyz)
        if self.quadrupole is not None:
            self._rotate_quadrupole_1(rotxy,rotxz,rotyz)
    
# TODO: Add possibility to move quadrupole matrix
    def move(self,dx,dy,dz):
        """ Moves the atomic orbitals along the specified vector
        
        Parameters
        --------
        dx,dy,dz : float
            Displacements along x, y, and z coordinate
        """
        
        # So Far dipole defined only in internal units
        dx_int = self.coor.manager.convert_position_2_internal_u(dx)
        dy_int = self.coor.manager.convert_position_2_internal_u(dy)
        dz_int = self.coor.manager.convert_position_2_internal_u(dz)
        
        # Quadrupoles are defined coordinate idependent
#        if self.quadrupole is not None: # Must be first because we need unshifted dipoles
#            if self.dipole is None:
#                self.get_dipole_matrix()
#            if self.overlap is None:
#                self.get_overlap()
#            self.quadrupole[0,:,:] = self.quadrupole[0,:,:] + 2*dx_int*self.dipole['Dip_X'] + (dx_int**2)*self.overlap
#            self.quadrupole[1,:,:] = self.quadrupole[1,:,:] + dy_int*self.dipole['Dip_X'] + dx_int*self.dipole['Dip_Y'] + dx_int*dy_int*self.overlap
#            self.quadrupole[2,:,:] = self.quadrupole[2,:,:] + dz_int*self.dipole['Dip_X'] + dx_int*self.dipole['Dip_Z'] + dx_int*dz_int*self.overlap
#            self.quadrupole[3,:,:] = self.quadrupole[3,:,:] + 2*dy_int*self.dipole['Dip_Y'] + (dy_int**2)*self.overlap
#            self.quadrupole[4,:,:] = self.quadrupole[4,:,:] + dz_int*self.dipole['Dip_Y'] + dy_int*self.dipole['Dip_Z'] + dy_int*dz_int*self.overlap
#            self.quadrupole[5,:,:] = self.quadrupole[5,:,:] + 2*dz_int*self.dipole['Dip_Z'] + (dz_int**2)*self.overlap
##            * quadrupole[0,:,:] = quadrupole xx matrix <AO_i|x^2|AO_j>
##            * quadrupole[1,:,:] = quadrupole xy matrix <AO_i|xy|AO_j>
##            * quadrupole[2,:,:] = quadrupole xz matrix <AO_i|xz|AO_j>
##            * quadrupole[3,:,:] = quadrupole yy matrix <AO_i|yy|AO_j>
##            * quadrupole[4,:,:] = quadrupole yz matrix <AO_i|yz|AO_j>
##            * quadrupole[5,:,:] = quadrupole zz matrix <AO_i|zz|AO_j>
#            #print('AO: Translation of AO quadrupoles is not yet supported')
#            #self.quadrupole=None
        if self.dipole is not None:
            if self.overlap is None:
                self.get_overlap()
            self.dipole['Dip_X']=self.dipole['Dip_X']+dx_int*self.overlap
            self.dipole['Dip_Y']=self.dipole['Dip_Y']+dy_int*self.overlap
            self.dipole['Dip_Z']=self.dipole['Dip_Z']+dz_int*self.overlap
        self.coor.move(dx,dy,dz)
        
    def copy(self):
        ''' Create deep copy of the all information in the class. 
        
        Returns
        ----------
        ao_new : AO class
            AO class with exactly the same information as previous one 
            
        '''
        
        ao_new=deepcopy(self)        
        return ao_new
        
            
# TODO: change units to angstrom
            
def l_orient(l):
    ''' Possible orientation of atomic orbitals in same order as in gaussian output '''
    if isinstance(l,str):
        if l=='s':
            orient=[[0,0,0]]
        elif l=='p':
            orient=[[1,0,0],[0,1,0],[0,0,1]]
        elif l=='d':
            orient=[[2,0,0],[0,2,0],[0,0,2],[1,1,0],[1,0,1],[0,1,1]]
        elif l=='f':
            orient=[[3,0,0],[0,3,0],[0,0,3],[1,2,0],[2,1,0],[2,0,1],
                    [1,0,2],[0,1,2],[0,2,1],[1,1,1]]
        elif l=='5d':
            orient=[[-2,0,0],[-1,0,-1],[0,-1,-1],[0,-2,0],[-1,-1,0]]
        elif l=='7f':
            orient=[[0,0,-3],[-1,0,-2],[0,-1,-2],[-2,0,-1],[-1,-1,-1],
                    [-3,0,0],[-2,-1,0]]
        else:
            raise IOError('Unsupported type of orbital in l_orient')
    else:
        if l==0:
            orient=[[0,0,0]]
        elif l==1:
            orient=[[1,0,0],[0,1,0],[0,0,1]]
        elif l==2:
            orient=[[2,0,0],[0,2,0],[0,0,2],[1,1,0],[1,0,1],[0,1,1]]
        elif l==3:
            orient=[[3,0,0],[0,3,0],[0,0,3],[1,2,0],[2,1,0],[2,0,1],
                    [1,0,2],[0,1,2],[0,2,1],[1,1,1]]
        elif l==-2:
            orient=[[-2,0,0],[-1,0,-1],[0,-1,-1],[0,-2,0],[-1,-1,0]]
        elif l==-3:
            orient=[[0,0,-3],[-1,0,-2],[0,-1,-2],[-2,0,-1],[-1,-1,-1],
                    [-3,0,0],[-2,-1,0]]
        else:
            raise IOError('Unsupported type of orbital in l_orient')
    return orient