# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:12:49 2016

@author: uzivatel
"""
import numpy as np
from sys import platform
from copy import deepcopy
import timeit

from ..positioningTools import RotateAndMove, RotateAndMove_1
from .general import Energy,Dipole
from .density import DensityGrid

# TODO: Corect MO indexes in coeff to start from 0 and not from 1
# TODO: Check definition for quadrupoles (I think that they should be shifted to zero for every atom same as dipoles x-x_mu ...). This way we would have coordinate dependent quadrupoles
class Excitation:
    ''' 
    Class containing all informations about excitations
    
    coeff : list of integer and real (dimension Nsingle_trans x 3)
        At the first position is initial molecular orbital (MO) - **starting from 
        1**, at the 2nd there is final MO - **starting from 1** - and at the 3rd
        it is coefficient of contribution of this single electron transition in 
        whole transition. (for some methods sum of the coefficients sre normalized
        to 1/2, for others to 1)
    energy : Energy class
        Electronic tranistion energy (energy managed units)
    oscil : float
        Oscillator strength of the electronic transition
    symm : string
        Symmetry of the electronic transition 
    dipole : Dipole class
        transition dipole in units dipole.units
    method : string
        Quantum chemistry method for calculation of electronic excited state.
        Supported types are: 'CIS', 'TD-DFT', 'ZIndo', 'Huckel' and 'Generated'
    root : integer
        Specified for which excited state electronic density was optimized
        (same meaning as Gaussian 09 keyword **root**)
    multiplicity : string
        Specifies spin multiplicity of electronic excited state ('Singlets' or 
        'Triplets')
    tr_char : numpy array of float (dimension Natom)
        Transition atomic charges from integration of transition density on
        individual atoms
    tr_dip : numpy array of float (dimension Natom x 3)
        Transition atomic dipoles defined as: 
        ``tr_dip[i]=sum_{mu in atom i}{sum_{nu}{sum_{i->j}{c_{i->j}*c_{i,mu}*c_{j,nu}*
        int{AO_mu(x,y,z) (x-x_mu,y-y_mu,z-z_mu) AO_nu(x,y,z) }}}}``. This way we
        obtain atomic dipoles for multipole expansion in atomic distances which
        are also coordinate system independent
    tr_quadr2 : numpy array of float (dimension Natom)
        Transition atomic quadrupoles defined as:
        ``tr_dip[ii]=sum_{mu in atom ii}{sum_{nu}{sum_{i->j}{c_{i->j}*c_{i,mu}*c_{j,nu}*
        int{AO_mu(x,y,z) x^2+y^2+z^2 AO_nu(x,y,z) }}}}``
        Which is a different definition then for dipoles.
    tr_quadrR2 : numpy array of float (6 x Natom)
        Transition atomic quadrupoles defined as:
        ``tr_dip[:,ii]=sum_{mu in atom ii}{sum_{nu}{sum_{i->j}{c_{i->j}*c_{i,mu}*c_{j,nu}*
        int{AO_mu(x,y,z) [xx,xy,xz,yy,yz,zz] AO_nu(x,y,z) }}}}``
        Which is a different definition then for dipoles.
    esp_exct : numpy array of float (dimension Natom)
        Excited state atomic charges obtained from fitting of excited state
        electrostatic potential.
    esp_trans : numpy array of float (dimension Natom)
        Transition atomic charges obtained from fitting of transition 
        electrostatic potential.
    desnmat_trans : numpy array of float (dimension Nao_orient x Nao_orient)
        Transition electron density matrix in atomic orbitals. Defined as:
        ``desnmat_trans[mu,nu]=sum_{i->j}{c_{i->j}*c_{i,mu}*c_{j,nu}``
        where ``c_{i->j}`` are expansion coefficients of electronic transition 
        into transition between molecular orbitals and ``c_{i,mu}`` are expansion
        coefficients of molecular orbital i into atomic orbital basis (atomic
        orbital mu).
    densmat_exct : numpy array of float (dimension Nao_orient x Nao_orient)
        Excited state full electron density matrix in atomic orbitals. Defined
        same way as ground state density matrix as summ over all ocupied states
        but for tranition i->j we chage ocupation of orbital i to 1 and orbital 
        j also to 1. This we do for all expansion of the transition and weight 
        every contribution by corresponding expansion coefficient
    dens_exct : Density class
        Full electron excited state density (evaluated on the grid)
    dens_trans : Density class
        Transition electron density (evaluated on the grid)
        
    Functions
    ---------
    rotate :
        Rotate the transition quantities (transition dipole, atomic dipoles, ...)
        by specified angles in radians in positive direction.
    rotate_1 :
        Inverse totation to rotate
    move :
        Moves the transition quantities (transition dipole, atomic dipoles, ...)
        along specified vector
    copy :
        Create 1 to 1 copy of the transition properties with all classes and types.
    get_tr_densmat :
        Calculate transition electron density matrix
    get_transition_atomic_charges :
        Calculate transition atomic charges for multipole expansion in atomic 
        distances
    get_transition_dipole :
        Calculate transition atomic dipoles for multipole expansion in atomic 
        distances
    get_transition_atomic_quadrupole :
        Calculate transition atomic quadrupoles for multipole expansion in atomic 
        distances
    get_transition_density :
        Calculate transition density on given grid
    excitation_type :
        Determines a type of the excitation (if it is pi-pi, sigma-sigma, ... transition) 
    '''
    
    def __init__(self,energy,dipole,dip_units='AU',coeff=None,oscil=None,symm=None,method=None,root=None,multiplicity=None):
        if coeff is not None:
            self.coeff=coeff
        else:
            self.coeff=None
        if energy is not None:
            self.energy=Energy(energy)
        else:
            self.energy=None
        if oscil is not None:
            self.oscil=oscil
        else:
            self.oscil=None
        if symm is not None:
            self.symm=symm
        else:
            self.symm=None
        if dipole is not None:
            self.dipole=Dipole(np.array(dipole,dtype='f8'),dip_units)
        else:
            self.dipole=None
        if method is not None:
            self.method=method
        else:
            self.method=None
        if root is not None:
            self.root=root
        else:
            self.root=None
        if multiplicity is not None:
            self.multiplicity=multiplicity
        else:
            self.multiplicity=None
        self.tr_char=None
        self.tr_dip=None
        self.tr_quadr2=None
        self.tr_quadrR2=None
        self.esp_exct=None
        self.esp_trans=None
        self.desnmat_trans=None
        self.densmat_exct=None
        self.dens_exct=None
        self.dens_trans=None
        
    def rotate(self,rotxy,rotxz,rotyz,AO):
        """"
        Rotate excited state quantities around the coordinate origin by specified angles.
        Transitions and coefficients are still the same but atomic orbitals 
        rotates and therefore also density matrixies and atomic properties
        has to be rotated too. 
        First it rotate the structure around z axis (in xy plane), then around
        y axis and in the end around x axis in positive direction 
        (if right thumb pointing in direction of axis fingers are pointing in 
        positive rotation direction)
        
        Parameters
        --------
        rotxy,rotxz,rotxz : float
            Rotation angles in radians around z, y and x axis (in xy, xz and yz plane)
        AO : AO class
            Information about atomic orbitals. Ordering of atomic orbitals has 
            to be known to rotate the excited state quantities (because AO basis
            is rotated)
        
        """
        TransfMat=AO._get_ao_rot_mat(rotxy,rotxz,rotyz)
        if self.desnmat_trans is not None:
            self.desnmat_trans=np.dot(TransfMat,np.dot(self.desnmat_trans,TransfMat.T))
        if self.densmat_exct is not None:
            self.densmat_exct=np.dot(TransfMat,np.dot(self.densmat_exct,TransfMat.T))
        if self.dens_exct is not None:
            self.dens_exct.rotate(rotxy,rotxz,rotyz)
        if self.dens_trans is not None:
            self.dens_trans.rotate(rotxy,rotxz,rotyz)
        if self.tr_dip is not None:
            self.tr_dip=RotateAndMove(self.tr_dip,0.0,0.0,0.0,rotxy,rotxz,rotyz)
        if (self.tr_quadr2 is not None) or (self.tr_quadrR2 is not None):
            print('Rotation of quadrupoles is not implemented yet. Dealocation of all atomic quadrupoles')
            self.tr_quadr2=None
            self.tr_quadrR2=None
        if self.dipole is not None:
            self.dipole.rotate(rotxy,rotxz,rotyz)
    
    def rotate_1(self,rotxy,rotxz,rotyz,AO):
        """" Inverse rotation of excited state quantities to **rotate** function.
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
            to be known to rotate the excited state quantities (because AO basis
            is rotated)
            
        """
        TransfMat=AO._get_ao_rot_mat_1(rotxy,rotxz,rotyz)
        if self.desnmat_trans is not None:
            self.desnmat_trans=np.dot(TransfMat,np.dot(self.desnmat_trans,TransfMat.T))
        if self.densmat_exct is not None:
            self.densmat_exct=np.dot(TransfMat,np.dot(self.densmat_exct,TransfMat.T))
        if self.dens_exct is not None:
            self.dens_exct.rotate_1(rotxy,rotxz,rotyz)
        if self.dens_trans is not None:
            self.dens_trans.rotate_1(rotxy,rotxz,rotyz)
        if self.tr_dip is not None:
            self.tr_dip=RotateAndMove_1(self.tr_dip,0.0,0.0,0.0,rotxy,rotxz,rotyz)
        if (self.tr_quadr2 is not None) or (self.tr_quadrR2 is not None):
            print('Rotation of quadrupoles is not implemented yet. Dealocation of all atomic quadrupoles')
            self.tr_quadr2=None
            self.tr_quadrR2=None
        if self.dipole is not None:
            self.dipole.rotate_1(rotxy,rotxz,rotyz)
        
    def move(self,dx,dy,dz):
        """ Moves theexcited state quantities along the specified vector 
        (transition densities)
        
        Parameters
        --------
        dx,dy,dz : float
            Displacements along x, y, and z coordinate
        """
        
        if self.dens_exct is not None:
            self.dens_exct.move(dx,dy,dz)
        if self.dens_trans is not None:
            self.dens_trans.move(dx,dy,dz)
            
    def get_tr_densmat(self,MO):
        """Calculate transition electron density matrix.
        
        Parameters
        ----------
        MO : MO class
            Molecular orbital information - expansion coefficients into atomic
            orbital basis
        
        Definition
        ----------
        Transition density matrix is defined as:
        ``desnmat_trans[mu,nu]=sum_{i->j}{c_{i->j}*c_{i,mu}*c_{j,nu}``
        where ``c_{i->j}`` are expansion coefficients of electronic transition 
        into transition between molecular orbitals and ``c_{i,mu}`` are expansion
        coefficients of molecular orbital i into atomic orbital basis (atomic
        orbital mu).
        
        Notes
        ---------
        Transition density matrix is stored at: \n
        **self.desnmat_trans** \n
        as numpy array of float (dimension Nao_orient x Nao_orient)
        
        """ 
        
        if self.method in ['CIS', 'TD-DFT', 'ZIndo', 'Huckel','Generated']:
            if self.method=='TD-DFT':
                factor=2
            else:
                factor=1
        else:
            raise Warning('Unsupported type of excited state calculation')
        
        M_mat=np.zeros((len(MO.coeff[0]),len(MO.coeff[0])),dtype='f8')
        for nn in range(len(self.coeff)):
            coef=self.coeff[nn][2]
            mo_i=self.coeff[nn][0]-1
            mo_j=self.coeff[nn][1]-1
            Cj_mat,Ci_mat = np.meshgrid(MO.coeff[mo_j],MO.coeff[mo_i])
            M_mat+=np.multiply(factor*coef,np.multiply(Ci_mat,Cj_mat))
        
        self.desnmat_trans=np.copy(M_mat)
    
    def get_transition_atomic_charges(self,AO,MO=None,verbose=False):
        """Calculate transition atomic charges for multipole expansion in 
        inter-atomic distances.
        
        Parameters
        ----------
        AO : AO class
            Information about atomic orbitals. For calculation of transition 
            charges overlap between atomic orbitals is needed.
        MO : MO class (optional init = None)
            Molecular orbital information. Needed for calculation of transition
            desnity matrix (If allready calculated we dont need it)
        verbose : logical (optional init = False)
            Controls if sum af atomic transition charges is printed
        
        Returns
        ---------
        Tr_charge : numpy array of float (dimension Natoms)
            Transiton atomic charges for every atom.
        
        Definition
        ----------
        Transition atomic charges are defined as: \n
        ``tr_char[i]=sum_{mu in atom i}{sum_{nu}{sum_{i->j}{
        c_{i->j}*c_{i,mu}*c_{j,nu}*<AO_mu|AO_nu>}}`` \n
        where ``c_{i->j}`` are expansion coefficients of electronic transition 
        into transition between molecular orbitals and ``c_{i,mu}`` are expansion
        coefficients of molecular orbital i into atomic orbital basis (atomic
        orbital mu).
        
        Notes
        ---------
        Transition atomic charges are stored at: \n
        **self.tr_char** \n
        as numpy array of float (dimension Natoms)
        
        """ 
        
        if AO.overlap is None:
            AO.get_overlap()
            
        if self.desnmat_trans is None:
            self.get_tr_densmat(MO)
        
        Nat=max([AO.atom[ii].indx for ii in range(AO.nao)])+1
        TrCh_Mat=np.dot(self.desnmat_trans,AO.overlap)
        # same as: TrCh_Mat=np.dot(self.exct_spec[exct_i]['coeff_mat'].T,self.ao.overlap)
        if verbose:
            print('Sum of atomic charges:',np.trace(TrCh_Mat))
        # sumation of transition charge on individual atoms:
        Tr_charge=np.zeros(Nat,dtype='f8')
        counter=0
        for ii in range(AO.nao):
            for jj in range(len(AO.orient[ii])):
                Tr_charge[AO.atom[ii].indx] += TrCh_Mat[counter,counter]
                counter += 1
        
        self.tr_char=np.copy(Tr_charge)
        if verbose:
            print('Sum of transition charges is:',np.sum(self.tr_char))
        
        return Tr_charge
    
    def get_transition_dipole(self,AO,at_coor,MO=None):
        """Calculate transition atomic dipoles for multipole expansion in 
        inter-atomic distances.

        Parameters
        ----------
        AO : AO class
            Information about atomic orbitals. For calculation of transition 
            charges overlap between atomic orbitals is needed.
        at_coor : Coordinate class
            Atomic coordinates (position managed units)
        MO : MO class (optional init = None)
            Molecular orbital information. Needed for calculation of transition
            desnity matrix (If allready calculated we dont need it)
        
        Returns
        ---------
        dipole : numpy array of float (dimension 3)
            Transition dipole for whole molecule (complex)
        at_dipole : numpy array of float (dimension Natoms x 3)
            Transiton atomic dipoles for every atom.
        
        Definition
        ----------
        Transition atomic dipoles are defined as: \n
        ``tr_char[i]=sum_{mu in atom i}{sum_{nu}{sum_{i->j}{
        c_{i->j}*c_{i,mu}*c_{j,nu}*<AO_mu|(x-x_mu,y-y_mu,z-z_mu)|AO_nu>}}`` \n
        where ``c_{i->j}`` are expansion coefficients of electronic transition 
        into transition between molecular orbitals and ``c_{i,mu}`` are expansion
        coefficients of molecular orbital i into atomic orbital basis (atomic
        orbital mu).
        
        Notes
        ---------
        Transition atomic dipoles are also stored at: \n
        **self.tr_dip** \n
        as numpy array of float (dimension Natoms x 3)
        
        """ 
        
# TODO: change units of atomic orbital quantities from angstrom to Bohr
        #return2Angs=False
        #if AO.coor.units=='Angstrom':
        #    AO.Angst2Bohr()
        #    return2Angs=True
        
        if AO.dipole is None:
            AO.get_dipole_matrix()
        if self.tr_char is None:
            self.get_transition_atomic_charges(AO,MO)  
        if self.desnmat_trans is None:
            self.get_tr_densmat(MO) 
        
        Nat=max([AO.atom[ii].indx for ii in range(AO.nao)])+1
        M_mat=np.copy(self.desnmat_trans)
        dipole=np.zeros(3,dtype='f8')
            
        DD_X=np.zeros((AO.nao_orient,AO.nao_orient),dtype='f8')
        DD_Y=np.zeros((AO.nao_orient,AO.nao_orient),dtype='f8')
        DD_Z=np.zeros((AO.nao_orient,AO.nao_orient),dtype='f8')
        DD_X=np.dot(AO.dipole['Dip_X'],M_mat.T)
        DD_Y=np.dot(AO.dipole['Dip_Y'],M_mat.T)
        DD_Z=np.dot(AO.dipole['Dip_Z'],M_mat.T)
#            DD_X=np.dot(M_mat,AO.dipole['Dip_X'])
#            DD_Y=np.dot(M_mat,AO.dipole['Dip_Y'])
#            DD_Z=np.dot(M_mat,AO.dipole['Dip_Z'])

        
#        # at_dipole will be independent of coordinates:
#        R=np.zeros((AO.nao_orient,3))
#        counter=0
#        for ii in range(AO.nao):
#            for jj in range(len(AO.orient[ii])):
#                R[counter]=AO.coor._value[ii]   # Positions in Bohr radius
#                counter+=1
#        R=np.array(R)
#            DD1_X=DD_X-np.dot(M_mat.T,np.multiply(np.meshgrid(R[:,0],R[:,0])[1],self.mol_spec['AO_overlap']))
#            DD1_Y=DD_Y-np.dot(M_mat.T,np.multiply(np.meshgrid(R[:,1],R[:,1])[1],self.mol_spec['AO_overlap']))
#            DD1_Z=DD_Z-np.dot(M_mat.T,np.multiply(np.meshgrid(R[:,2],R[:,2])[1],self.mol_spec['AO_overlap']))            
#            DD1_X=DD_X-np.dot(M_mat,np.multiply(np.meshgrid(R[:,0],R[:,0])[1],self.mol_spec['AO_overlap']))
#            DD1_Y=DD_Y-np.dot(M_mat,np.multiply(np.meshgrid(R[:,1],R[:,1])[1],self.mol_spec['AO_overlap']))
#            DD1_Z=DD_Z-np.dot(M_mat,np.multiply(np.meshgrid(R[:,2],R[:,2])[1],self.mol_spec['AO_overlap']))            
#            DD1_X=DD_X-np.dot(np.meshgrid(R[:,0],R[:,0])[0],np.multiply(M_mat,self.mol_spec['AO_overlap']))
#            DD1_Y=DD_Y-np.dot(np.meshgrid(R[:,1],R[:,1])[0],np.multiply(M_mat,self.mol_spec['AO_overlap']))
#            DD1_Z=DD_Z-np.dot(np.meshgrid(R[:,2],R[:,2])[0],np.multiply(M_mat,self.mol_spec['AO_overlap']))            

        at_dipole=np.zeros((Nat,3),dtype='f8')
        counter=0
        for ii in range(AO.nao):
            for jj in range(len(AO.orient[ii])):
                at_dipole[AO.atom[ii].indx,0] += DD_X[counter,counter]
                at_dipole[AO.atom[ii].indx,1] += DD_Y[counter,counter]
                at_dipole[AO.atom[ii].indx,2] += DD_Z[counter,counter]
                counter += 1
        for ii in range(Nat):
            at_dipole[ii] -= at_coor._value[ii]*self.tr_char[ii]
        
#        self.struc.tr_dip=np.copy(at_dipole)
        self.tr_dip=np.copy(at_dipole)          
        
        dipole[0]=-np.trace(DD_X)
        dipole[1]=-np.trace(DD_Y)
        dipole[2]=-np.trace(DD_Z)
        
        return dipole,at_dipole
    
    
    def get_transition_atomic_quadrupole(self,AO,MO=None):
        """Calculate transition atomic quadrupoles for multipole expansion in 
        inter-atomic distances.
        
        Parameters
        ----------
        AO : AO class
            Information about atomic orbitals. For calculation of transition 
            charges overlap between atomic orbitals is needed.
        MO : MO class (optional init = None)
            Molecular orbital information. Needed for calculation of transition
            desnity matrix (If allready calculated we dont need it)
        
        Returns
        ---------
        Quad_r2 : numpy array of float (dimension Natoms)
            Atomic quadrupole element r^2 ( = xx+yy+zz) for every atom
        Quad_rR2 : numpy array of float (dimension 6 x Natoms)
            Atomic quadrupoles for every atom. Quad_rR2[:,i] = [quadr. xx, 
            quadr. xy, quadr. xz, quadr. yy, quadr. yz, quadr. zz] for atom i
        
        Definition
        ----------
        Transition atomic quadrupoles are defined as: \n
        ``tr_quadrR2[:,i]=sum_{mu in atom i}{sum_{nu}{sum_{i->j}{
        c_{i->j}*c_{i,mu}*c_{j,nu}*<AO_mu|(x^2, xy, xz, y^2, yz, z^2)|AO_nu>}}`` \n
        and \n
        ``tr_quadr2[i]=sum_{mu in atom i}{sum_{nu}{sum_{i->j}{
        c_{i->j}*c_{i,mu}*c_{j,nu}*<AO_mu|x^2 + y^2 + z^2|AO_nu>}}`` \n
        where ``c_{i->j}`` are expansion coefficients of electronic transition 
        into transition between molecular orbitals and ``c_{i,mu}`` are expansion
        coefficients of molecular orbital i into atomic orbital basis (atomic
        orbital mu).
        
        Notes
        ---------
        Transition atomic quadrupoles are also stored at: \n
        **self.tr_quadrR2** \n
        as numpy array of float (dimension 6 x Natoms) and at \n
        **self.tr_quadr2** \n
        as numpy array of float (dimension Natoms)
        
        """
        
        if AO.quadrupole is None:
            AO.get_quadrupole()
        if self.desnmat_trans is None:
            self.get_tr_densmat(MO)
        
        Nat=max([AO.atom[ii].indx for ii in range(AO.nao)])+1
        # Everything is ready for calculation of atomic transition quadrupoles
        Quad_Mat=np.zeros((len(AO.quadrupole[0]),len(AO.quadrupole[0])),dtype='f8')
        Quad_Mat=AO.quadrupole[0]+AO.quadrupole[3]+AO.quadrupole[5]
        Quad_Mat=np.dot(Quad_Mat,(self.desnmat_trans).T)
        
        Quad_r2=np.zeros(Nat,dtype='f8')
        counter=0
        for ii in range(AO.nao):
            for jj in range(len(AO.orient[ii])):
                Quad_r2[AO.atom[ii].indx] += Quad_Mat[counter,counter]
                counter += 1
        
#        self.struc.tr_quadr2=np.copy(Quad_r2)
        self.tr_quadr2=np.copy(Quad_r2)
        print('Sum of transitin quadrupoles r^2 is:',np.sum(self.tr_quadr2))
        
        Quad_rR2=np.zeros((6,Nat),dtype='f8')
        for kk in range(6):
            Quad_Mat=np.zeros((len(AO.quadrupole[kk]),len(AO.quadrupole[kk])),dtype='f8')
            Quad_Mat=np.copy(AO.quadrupole[kk])
            Quad_Mat=np.dot(Quad_Mat,(self.desnmat_trans).T)
            counter=0
            for ii in range(AO.nao):
                for jj in range(len(AO.orient[ii])):
                    Quad_rR2[kk,AO.atom[ii].indx] += Quad_Mat[counter,counter]
                    counter += 1
        
#        self.struc.tr_quadrR2=np.copy(Quad_rR2)
        self.tr_quadrR2=np.copy(Quad_rR2)
        
        return Quad_r2,Quad_rR2
        
    def get_transition_density(self,AO,grid,coor,ncharge,MO=None,nt=0,cutoff=0.01,List_MO=None):
# TODO: Mybe instead of coor and ncharge imput use Structure class imput
        """Evaluate transition density on given grid
        
        Parameters
        ----------
        AO : AO class
            Information about atomic orbitals. For calculation of transition 
            charges overlap between atomic orbitals is needed and for evaluation
            of AO basis on given grid
        grid : Grid class
            Information about grid on which transition density is evaluated.
        coor : Coordinate class or numpy.array of float (dimension Natoms x 3)
            Atomic coordinates for every atom
        ncharge : numpy array of float (dimension Natoms)
            Nuclear charges for every atom
        MO : MO class (optional init = None)
            Molecular orbital information. Needed for calculation of transition
            desnity matrix (If allready calculated we dont need it)
        nt : integer (optional init = 0)
            Specifies how many cores should be used for the calculation.
            Secial cases: 
            ``nt=0`` all available cores are used for the calculation. 
            ``nt=1`` serial calculation is performed.
            ``nt=N`` Ncores are used for the calculation.
        cutoff : float (optionl init = 0.01)
            Used only if slater orbital bassis is not evaluated on the given
            grid. Only transitions with coefficients which are higher then
            cutoff (in absolute value) will be used for evaluation of transition
            density on given grid.
        List_MO : list of integers and float (optional)
            If specified transitions between molecular orbitals and their
            coefficients are read from this file instead of default self.coeff.
            Example of List_MO=[[56,57,0.8],[55,58,-0.6]] means that transition 
            will be composed from 0.8*MO[56]->MO[57] + 0.6*MO[55]->MO[58] 
            (0.8*HOMO->LUMO + 0.6*HOMO-1->LUMO+1). Usefull for example if we 
            would like to look at only pi-pi transitions
        
        Returns
        ---------
        dens_grid : Density class
            Transition density evaluated on the grid in cube file format
        
        Notes
        ---------
        Transition desnity is also stored at: \n
        **self.dens_trans** \n
        as Density class.
        
        """
        
        from functools import partial
        from multiprocessing import Pool, cpu_count
        
        def partial_density_sumation(AO_grid,nt,indx):
            Nao=len(AO_grid)
            dens_grid=np.zeros(np.shape(AO_grid[0])) 
            DN=np.int(np.ceil(Nao/nt))    # in this case all processors have to do same work and therefore in order to overcome dificulties with creating too much copies od density it is easier to divide the job statically    
            if indx==0:
                for ii in range(DN):
                    dens_grid+=np.multiply(AO_grid[ii],AO_grid[ii])
            elif indx!=nt-1:
                for ii in range(DN*indx,DN*(indx+1)):
                    dens_grid+=np.multiply(AO_grid[ii],AO_grid[ii])
            elif indx==nt-1:
                for ii in range(DN*indx,Nao):
                    dens_grid+=np.multiply(AO_grid[ii],AO_grid[ii])
            return dens_grid
        
    
        if (platform=='cygwin' or platform=="linux" or platform == "linux2") and nt!=1 and nt>=0:
                typ='paralell'
        elif platform=='win32' or nt==1:
                typ='seriall'
        else:
                typ='seriall'    
        
        
        dens_grid=np.zeros(np.shape(grid.X))     
        counter=0
        excit_type=self.method
        mo_i_old=None
        mo_j_old=None
        if excit_type in ['CIS','TD-DFT','ZIndo','Huckel','Generated']:
            if excit_type=='TD-DFT':
                factor=2
            else:
                factor=1
        else:
            raise IOError('Unsupported excitation type in transition density calculation')
            
        if (AO.grid is None) or (List_MO is not None):  # pokud nejsou slaterovy orbitaly allokovany na mrizce tak...
            if MO is None:
                raise IOError('If all slater orbitals are not allocated on the grid molecular orbitals have to be imputed')
            if List_MO is None:
                coeffs=self.coeff
            else:
                coeffs=List_MO
            for nn in range(len(coeffs)):
                coef= coeffs[nn][2]
                if abs(coef)>=cutoff:
                    # initialization
                    mo_i=coeffs[nn][0]-1
                    mo_j=coeffs[nn][1]-1 
                    
                    # calculation of first MO
                    if mo_i!=mo_i_old:
                        mo_grid1=MO.get_mo_grid(AO,grid,mo_i)

                    # calculation of second MO
                    if mo_j!=mo_j_old:
                        mo_grid2=MO.get_mo_grid(AO,grid,mo_j)
                    # both MO calculated
                   
                    # Calculation of transition density
                    dens_grid+=np.multiply(factor*coef,np.multiply(mo_grid2,mo_grid1))
                    counter+=1
                    mo_i_old=mo_i
                    mo_j_old=mo_j
                    print(counter," transitions calculated")
        else:
            if self.desnmat_trans is None:
                if MO is None:
                    raise IOError('For calculation of transition matrix in transition density calculation you need to imput MO information')
                else:
                    self.get_tr_densmat(MO)
            M_mat=self.desnmat_trans
            
            # calculation of transition density                
            start_time = timeit.default_timer()
            
            if typ=='paralell': 
                AO_grid_tmp=[]
                for ii in range(AO.nao_orient):
                    AO_grid_tmp.append(np.multiply(M_mat[ii,0],AO.grid[0]))
                    for jj in range(1,AO.nao_orient):
                        AO_grid_tmp[ii]+=np.multiply(M_mat[ii,jj],AO.grid[jj])
                if nt>0:
                    partial_density_sumation_partial = partial(partial_density_sumation,AO_grid_tmp,nt) 
                    pool = Pool(processes=nt)
                    index_list=range(nt)
                else:
                    partial_density_sumation_partial = partial(partial_density_sumation,AO_grid_tmp,cpu_count()) 
                    pool = Pool(processes=cpu_count())
                    index_list=range(cpu_count())
                dens_grid_tmp= pool.map(partial_density_sumation_partial,index_list)
                pool.close() # ATTENTION HERE
                pool.join()
            
                
                dens_grid=np.sum(dens_grid_tmp,0)        
                elapsed = timeit.default_timer() - start_time
                print('Elapsed time for parallel density sumation:',elapsed)
            
            if typ=='seriall':
                start_time = timeit.default_timer()
                for ii in range(AO.nao_orient):
                    for jj in range(AO.nao_orient):
                        dens_grid+=np.multiply(M_mat[ii,jj],np.multiply(AO.grid[ii],AO.grid[jj]))
                elapsed = timeit.default_timer() - start_time
                print('Elapsed time for serial density sumation:',elapsed)
                
        step=np.zeros((3,3))
        step[0,0]=grid.delta[0]
        step[1,1]=grid.delta[1]
        step[2,2]=grid.delta[2]
        self.dens_trans=DensityGrid(np.array(grid.origin),np.shape(grid.X),step,dens_grid,typ='transition',mo_indx=1,Coor=coor,At_charge=ncharge)
        return dens_grid
        
    def copy(self):
        ''' Create deep copy of the all informations in the class. 
        
        Returns
        ----------
        exct_new : Excitation class
            Excitation class with exactly the same information as previous one 
            
        '''
        exct_new=deepcopy(self)
        return exct_new
    
    def excitation_type(self,MO_type,krit):
        """ Determines the transition type ('Pi -> Pi', 'Pi -> Sigma', 
        'Sigma -> Pi' or 'Sigma -> Sigma'). Uses the information about molecular
        orbital types
        
        Parameters
        ----------
        MO_type : list of string (dimension Nmo)
            List with orbital types ('Pi' or 'Sigma') for each molecular orbital
        krit : float (optional - init=90)
            Percentage of 'Pi -> Pi' transition contribution which will be 
            considered as 'Pi -> Pi' transition etc.. Accepted values are (0,100) 
       
        Returns
        ----------
        Transition_type : string
            Type of transition ('Pi -> Pi', 'Pi -> Sigma', 'Sigma -> Pi' or 
            'Sigma -> Sigma')
        
        """
        
        if krit>=100 or krit<0:
            raise IOError('Kriteria for determine excitation type has to be in percentage of Pi orbilats in excitation (0,100)')
        if self.method in ['CIS','TD-DFT','ZIndo','Huckel','Generated']:
            if self.method=='TD-DFT':
                Transition = np.array([self.coeff[i][0:2] for i in range(len(self.coeff))],dtype='i4')
                Prob = np.array([self.coeff[i][2] for i in range(len(self.coeff))],dtype='f8')
                Prob=Prob**2
                Prob=Prob*2
            else:
                Transition = np.array([self.coeff[i][0:2] for i in range(len(self.coeff))],dtype='i4')
                Prob = np.array([self.coeff[i][2] for i in range(len(self.coeff))],dtype='f8')
                Prob=Prob**2
        else:
            raise IOError('Unsupported excitation type in transition density calculation')

        MO_int_Type=np.copy(MO_type)
        MO_int_Type[MO_int_Type=='Pi']=1
        MO_int_Type[MO_int_Type=='Sigma']=-1
        MO_int_Type=np.int32(MO_int_Type)
        Single_Trans_Type=np.array(MO_int_Type)[Transition]
        Single_Trans_Type=np.sum(Single_Trans_Type,axis=1)  # -2 is for sigma-sigma transition, 0 is for pi-sigma or sigma-pi transition and 2 is for Pi-Pi transition
        
        Mask_PiPi=np.array((Single_Trans_Type==2),dtype='i4')
        Mask_SigmaPi=np.array((Single_Trans_Type==0),dtype='i4')
        Mask_SigmaSigma=np.array((Single_Trans_Type==-2),dtype='i4')

        Prob_PiPi=np.dot(Mask_PiPi,Prob)
        Prob_SigmaPi=np.dot(Mask_SigmaPi,Prob)
        Prob_SigmaSigma=np.dot(Mask_SigmaSigma,Prob)
        Prob_all=np.sum(Prob)
        
        if Prob_PiPi/Prob_all*100>krit:
            return 'Pi -> Pi'
        elif Prob_SigmaPi/Prob_all*100>krit:
            return 'Sigma -> Pi or Pi -> Sigma'
        elif Prob_SigmaSigma/Prob_all*100>krit:
            return 'Sigma -> Sigma'
        else:
            return 'Combined'

"""
        * **'coeff_mat'** = numpy.array of real dimension (N_orient_orbitals x N_orient_orbitals)
          transition state electron density defined as `coeff_mat[mu,nu]=Sum_ij{coeff_ij*C_mu,i*C_nu,j}`.
          Electron transition density is then defined as 
          `rho(r)=Sum_mu,nu{coeff_mat[mu,nu]*AO-STO_mu(r)*AO-STO_nu(r)}`
        * **'ExctDens_mat'** = numpy.array of real dimension (N_orient_orbitals x N_orient_orbitals)
          excited state electron density defined as `ExctDens_mat[mu,nu]=Sum_exct{Sum_i{coeff_ij*occn_i*C_mu,i*C_nu,i}}`
        * **'at_quad_r2'** = numpy.array of real (dimension Natoms) with trasition quadrupole 
          matrix summed to every atom. 
          Sum_nu{Summ_i->j{coeff_i->j*Sum_(i->j){C_nu,i*C_nu,j<AO_nu|r^2|AO_nu>}}}
          where nu correspond to single atom (sum over all excitations between
          two molecular orbitals) and coeff is percentage contribution of single
          excitation in whole molecule excitation
        * **'at_quad_rR2'** = numpy.array of real (dimension Natoms) with transition quadrupole 
          matrix summed to every atom. 
          Sum_nu{Summ_i->j{coeff_i->j*Sum_(i->j){C_nu,i*C_nu,j<AO_nu|(rR)^2|AO_nu>}}}
          where nu correspond to single atom (sum over all excitations between
          two molecular orbitals) and coeff is percentage contribution of single
          excitation in whole molecule excitation
        * **'at_trcharge'** = numpy.array of real (dimension Natoms) with atomic transition
          charges calcualted from analitical integration of transition density
          on individual atoms.
          Sum_nu{Summ_i->j{coeff_i->j*Sum_(i->j){C_nu,i*C_nu,j}}}
          where nu correspond to single atom (sum over all excitations between
          two molecular orbitals) and coeff is percentage contribution of single
          excitation in whole molecule excitation
        * **'at_trdip'** = numpy.array of real (dimension Natoms x 3) with atomic transition
          dipoles calcualted from analitical integration of transition density
          with electronic coordinate on individual atoms. 
          Sum_nu{Summ_i->j{coeff_i->j*Sum_(i->j){C_nu,i*C_nu,j<AO_nu|r|AO_nu>}}}
          where nu correspond to single atom (sum over all excitations between
          two molecular orbitals) and coeff is percentage contribution of single
          excitation in whole molecule excitation
"""