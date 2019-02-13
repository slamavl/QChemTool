# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:33:56 2017

@author: Vladislav Sláma
"""
import numpy as np
from copy import deepcopy
from scipy.spatial.distance import pdist,squareform

from ..QuantumChem.interaction import charge_charge
from ..General.Potential import potential_charge, potential_dipole
from ..QuantumChem.calc import GuessBonds

debug=False

#==============================================================================
#  Definition of class for polarizable environment
#==============================================================================
class PolarAtom:
    ''' Class managing dielectric properties of individual atoms 
    
    Parameters
    ----------
    pol : real
        Polarizability in direction of the chemical bond
    amp : real
        Diference polarization between main directions. Should be smaller then
        pol
    per : integer
        Periodicity of the system (for Fluorographene it should be 3 or 6)
    phase : real
        Angle of the bond from the x-axis
    polz : real
        Polarizability in direction perpendicular to the fluorographene plane
    
    '''
    
    def __init__(self,polxy,amp,per,polz,phase=0.0):
        if abs(polxy)>abs(amp):
            self.polxy = abs(polxy)
            self.amp = amp
        else:
            self.polxy = abs(amp)
            self.amp = polxy
        
        self.per = per
        self.phase = phase
        self.polz = abs(polz)
    
    def _polarizability4angle(self,angle):
        phi = angle - self.phase
        n = self.per
        polar = self.polxy+self.amp*(np.cos(n*phi)-1)/2
        return np.array([polar,polar,self.polz],dtype='f8')
    
    def get_polarizability4elf(self,E):
        Phi=np.arctan2(E[1],E[0])
        
        # calculate polarizability for the angle
        polar = self._polarizability4angle(Phi)
        
        return polar
    
    def get_induced_dipole(self,E):
        
        polar = self.get_polarizability4elf(E)
        dipole = polar*E
        return dipole
    
    
class Dielectric:    
    ''' Class managing dielectric properties of the material 
    
    Parameters
    ----------
    coor : numpy.array of real (dimension Nx3) where N is number of atoms
        origin of density grid
    
    pol_type : list of strings
        POlarization atomic types. So far supported types are: ``CF`` for the 
        fluorographene carbon, ``FC`` for the fluorographne fluorine and ``C``
        for the defect carbon.
        
    charge : numpy.array or list of real (dimension N)
        charges on individual atoms (initial charges)

    dipole : numpy.array of real (dimension Nx3)
        dipole on individual atoms (initial dipole)
        
    polar_param : dictionary
        Polarization parameters for every polarization atom type.
    '''
    
    def __init__(self,coor,pol_type,charge,dipole,polar_param):
        self.coor=np.copy(coor)
        self.polar={}
        self.charge=np.copy(charge)
        self.dipole=np.copy(dipole)
        self.at_type=pol_type
        self.coarse_grain = polar_param['coarse_grain']
        self.Nat=len(coor)
        self.polar['AlphaE'] = []
        self.polar['Alpha_E'] = []
        self.polar['BetaEE'] = []
        self.polar['Alpha_st'] = []
        self.polar['Alpha_dyn'] = []
        try:
            self.VinterFG = polar_param["VinterFG"]
        except:
            self.VinterFG = 0.0
        self.assign_polar(polar_param["polarizability"])
        self._polar_allowed = ['AlphaE','Alpha_E','BetaEE','Alpha_st','Alpha_dyn']
        
    
    def assign_polar(self,params):
        
        ''' For now assignment is working only for fluorographene carbons with 
        type 'CF' and defect carbons with type 'CD' 

        Parameters
        ----------
        polar_param : dictionary
            Polarization parameters for every polarization atom type.
        
        Returns
        -------
        polar : numpy.array or list of real (dimension N)
            Polarizabilities for every atom. 'CF'=1.03595 and 'CD'=1.4
        '''
        
        polar_allowed = ['AlphaE','Alpha_E','BetaEE','Alpha_st']
        
        for poltype in polar_allowed:
            self.polar[poltype] = []
            
        for ii in range(self.Nat):
            at_type = self.at_type[ii]
            for poltype in polar_allowed:
                try: # For specified atom types assign polarizability, for others zero polarizability
                    at_prms = params[poltype][at_type]
                    polxy = at_prms[0]
                    polz = at_prms[1]
                    amp = at_prms[2]
                    per = at_prms[3]
                    try:
                        phase = at_prms[4]
                    except:
                        phase = 0.0
                    polarizable_atom = PolarAtom(polxy,amp,per,polz,phase=phase) 
                except:
                    polarizable_atom = PolarAtom(0.0,0.0,0,0.0) 
                self.polar[poltype].append(polarizable_atom)
        
        for ii in range(self.Nat):
            at_type = self.at_type[ii]
            poltype = 'Alpha_dyn'
            
            try: # For specified atom types assign polarizability, for others zero polarizability
                at_prmsE = params['AlphaE'][at_type]
                at_prms_E = params['Alpha_E'][at_type]
                polxy = (at_prmsE[0] + at_prms_E[0])/2
                polz = (at_prmsE[1] + at_prms_E[1])/2
                amp = (at_prmsE[2] + at_prms_E[2])/2
                per = at_prmsE[3]
                try:
                    phase = at_prmsE[4]
                except:
                    phase = 0.0
                polarizable_atom = PolarAtom(polxy,amp,per,polz,phase=phase) 
            except:
                polarizable_atom = PolarAtom(0.0,0.0,0,0.0) 
            self.polar[poltype].append(polarizable_atom)
    
    def _get_geom_phase(self):
        ''' Calculate angle between the chemical bond and the x axis
        
        '''
        from ..QuantumChem.calc import GuessBonds
        
        Nat = self.Nat
        phase = np.zeros(Nat,dtype="f8")
        bonds = GuessBonds(self.coor,bond_length=4.0)
        connected = []
        for ii in range(Nat):
            connected.append([])
        
        for ii in range(len(bonds)):
            atom1 = bonds[ii][0]
            atom2 = bonds[ii][1]
            connected[atom1].append(atom2)
            connected[atom2].append(atom1)
        
        pairs = np.zeros((2,Nat),dtype='i8')
        for ii in range(Nat):
            pairs[:,ii] = [ii,connected[ii][0]] 
        
        vecs = self.coor[pairs[1]] - self.coor[pairs[0]]
        phase = np.arctan2(vecs[:,1],vecs[:,0])
                
        # For all atom calculation phase is so far set to zero 
        # FIXME: Calculate phase for all atom simulations
        return phase
    
    def set_geom_phase(self,phase):
        ''' Assign phase (rotation of the origin of the symmetric polarizability)
        to all atoms
        
        Parameters
        ----------
        phase : numpy array of real (dimension Natoms)
            If ``phase`` is ``None`` the phase shift is automaticaly calculated.
            If phase is set it will be added to the original phase.
        '''
        if phase is None:
            phase = self._get_geom_phase()
        
        for ii in range(self.Nat):
            for poltype in self._polar_allowed:
                self.polar[poltype][ii].phase += phase[ii]
        
    
    def get_induced_dipoles(self,ElField,poltype):
        ''' Calculate induced dipole moments for given external electric field
        and atomic polarization type. Dipoles are calculated inefectively atom
        by atom.
        
        Parameters
        ----------
        ElField: numpy array of real (dimension Natom x 3)
            Vector of electric field at position of every single atom of polarizable 
            system. In ATOMIC UNITS
        poltype: string
            Atomic polarization type 
        '''
        
        induced_dipole = np.zeros((self.Nat,3),dtype='f8')
        
        for ii in range(self.Nat):
            polatom = self.polar[poltype][ii]
            induced_dipole[ii,:] = polatom.get_induced_dipole(ElField[ii])
        
        return induced_dipole
    
    def rescale_polarizabilities(self,poltype,scaling):
        for ii in range(self.Nat):
            polatom = self.polar[poltype][ii]
            polatom.polxy = polatom.polxy * scaling
            polatom.amp = polatom.amp * scaling
            polatom.polz = polatom.polz * scaling

    def get_distance_matrixes(self):
        '''
        Calculate inter-atom distance and vector matrix
        
        Returns
        ---------
        R: numpy array of real (dimension Natom x Natom x 3)
            ``R[i,j,:]`` corresponds to the vector from atom j to atom i 
            (in ATOMIC UNITS = bohr)
        RR: numpy array of real (dimension Natom x Natom)
            Interatomic distance matrix (in ATOMIC UNITS = bohr)
        '''
        # calculation of tensors with interatomic distances
        R=np.zeros((self.Nat,self.Nat,3),dtype='f8') # mutual distance vectors
        for ii in range(self.Nat):
            for jj in range(ii+1,self.Nat):
                R[ii,jj,:]=self.coor[ii]-self.coor[jj]
                R[jj,ii,:]=-R[ii,jj,:]
        RR=np.sqrt(np.power(R[:,:,0],2)+np.power(R[:,:,1],2)+np.power(R[:,:,2],2))  # mutual distances
        
        return R,RR
    
    def get_T_tensor(self,R=None,RR=None,RR3=None,RR5=None):
        ''' Calculate tensor for dipole dipole interaction between all atoms in
        the system.
        
        '''
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
        #for ii in range(self.Nat):
        #    P[ii,:]=np.dot(self.polar[typ][ii],ELFV[ii,:])
        P = self.get_induced_dipoles(ELFV,typ)
        
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
        
        #for ii in range(self.Nat):
        #    P[ii,:]=np.dot(self.polar[typ][ii],ELFV[ii,:])
        P = self.get_induced_dipoles(ELFV,typ)
        
        # -P should be 2nd order induced dipoles 
        self.dipole+=(-P)
        if debug:
            print('Dipole sum:',np.sum(self.dipole,axis=0))
    
# TODO: Add possibility for NN = -err to calculate dipoles until convergence is reached
    def _calc_dipoles_All(self,typ,Estatic=np.zeros(3,dtype='f8'),NN=60,eps=1,addition=False,debug=False,nearest_neighbor=True):
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
        addition : Logical 
            If true new induced dipoles will be added to already existing ones.
            otherwise the previous dipoles will be overwritten by the new ones.
            
        '''
        
        if debug:
            import timeit
            time0 = timeit.default_timer()
        R = np.tile(self.coor,(self.Nat,1,1))
        R = (np.swapaxes(R,0,1) - R)
        RR=squareform(pdist(self.coor))
        
#        if 0:
#            RR=np.sqrt(np.power(R[:,:,0],2)+np.power(R[:,:,1],2)+np.power(R[:,:,2],2))
#            RR2=squareform(pdist(self.coor))
#            print((RR2==RR).all())          # False
#            print(np.allclose(RR2,RR))      # True
#            if not (RR2==RR).all():
#                print(RR[0,1])
#                print(pdist(self.coor)[0])
#                print(RR[0,2])
#                print(pdist(self.coor)[1])
            
        
        if debug:
            time01 = timeit.default_timer()
            
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
        if not nearest_neighbor:
            bonds = GuessBonds(self.coor)
            for nn in bonds:
                MASK2[nn[0],nn[1]]=True
                MASK2[nn[1],nn[0]]=True
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
            if addition:
                ELF = np.zeros((self.Nat,self.Nat,3))
            else:
                ELF=(Q/RR3)*np.rollaxis(R,2)
                ELF=np.rollaxis(ELF,0,3)
            # ELF[i,j,:]  is electric field at position i generated by atom j - on diagonal there are zeros 
            ELF[MASK]=0.0
            
            # dipole electric field
            P=np.tile(self.dipole[:,:],(self.Nat,1,1))    # P[ii,:,:]=self.dipole[:,:]  for ii going through all atoms
            PR=np.sum(np.multiply(P,R),axis=2)
# TODO: This takes One second - make it faster
            for jj in range(3):             
                ELF[:,:,jj]+=(3*PR/RR5)*R[:,:,jj]
                ELF[:,:,jj]-=P[:,:,jj]/RR3
            ELF[MASK2]=0.0
            elf=np.sum(ELF,axis=1)/eps
# TODO: Think if this could be done in some efficient way
            if addition:
                self.dipole += self.get_induced_dipoles(elf + np.tile(Estatic,(self.Nat,1)),typ)
            else:
                self.dipole = self.get_induced_dipoles(elf + np.tile(Estatic,(self.Nat,1)),typ)
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

# =============================================================================
# TODO: Move to test part
# =============================================================================
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
# =============================================================================
#  END OF THE TEST PART        
# =============================================================================
        
                
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
        dipoles_pol: numpy array of float (dimension Natoms x 3)
            Induced atomic dipole moments for all atoms in the environment by 
            the second defect
        
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
        if '+' in typ and order==2:
            typ_arr = typ.split('+')
            self._calc_dipoles_All(typ_arr[0],NN=1,eps=1,debug=False)
            self._calc_dipoles_All(typ_arr[1],NN=1,eps=1,debug=False,addition=True)
        else: 
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
        if '+' in typ and order==2:
            typ_arr = typ.split('+')
            self._calc_dipoles_All(typ_arr[0],NN=1,eps=1,debug=False)
            self._calc_dipoles_All(typ_arr[1],NN=1,eps=1,debug=False,addition=True)
        else: 
            self._calc_dipoles_All(typ,NN=order,eps=1,debug=False)
        #self._calc_dipoles_All(typ,NN=order,eps=1,debug=False)
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
        
        if typ=='AlphaE' or typ=='BetaEE' or typ=='Alpha_st' or typ=='AlphaE+Alpha_st':
            return PolMAT,dipolesA,dipolesB,dipoles_polA,dipoles_polB
        elif typ=='Alpha_E' or typ=='Alpha_E+Alpha_st':
            PolMAT[[0,1],[0,1]] = PolMAT[[1,0],[1,0]]   # Swap AlphaMAT[0,0] with AlphaMAT[1,1]
            return PolMAT,dipolesA,dipolesB,dipoles_polA,dipoles_polB
    
    
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
    