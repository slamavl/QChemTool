# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:12:49 2016

@author: uzivatel
"""

import numpy as np
import scipy
from sys import platform
from copy import deepcopy

from .atomic_orbital import AO
from .structure import Structure
from .molecular_orbital import MO
from .density import DensityGrid
from .excitation import Excitation
from .general import Coordinate,Energy,Dipole

from ...General.functions import are_similar
from ...General.UnitsManager import position_units,energy_units
from ..read_mine import read_gaussian_fchk, read_gaussian_log, read_gaussian_esp, read_qchem_grid
from ..output import OutputMathematica
from ..positioningTools import fill_basis_transf_matrix, CenterMolecule

class Molecule:
    ''' 
    Class containing all informations about molecule
    
    name :
        Name of the molecule
    energy : Energy class
        Ground state energy (energy units managed)
    struc : Structure class
        Information about structure and atoms of the molecule
    ao : AO class
        Information about atomic orbital basis and overlaps etc.
    mo : MO class
        Information about molecular orbitals
    exct : list of Excitation class
        Informations about electronic transitions.
    repre : dictionary
        Different structure representations. For example repre['AllAtom'] is structure
        self.struc. We can also have multipole representation of the molecule or
        3D classical representation of the moelcule
    mol_spec : dictionary
        Contains additional information about molecule. This will be soon deleted
    vib_spec : dictionary
        Vibrational information about the molecule. Keys:
        
        * **'Hessian'** = numpy.array of dimension (3*Natoms x 3*Natoms) with Hessian 
          matrix in ATOMIC UNITS (Hartree/Bohr^2) which tells us how energy 
          is changed with change of single atom cartesian coordinate
        * **'NMinCart'** = numpy.array of dimesion (3*Natoms x Nnormal_modes) with 
          transformation matrix from normal mode displacements to cartesian
          displacements. (in colmns there are cartesian displacements corresponding
          to single normal mode)
        * **'CartInNM'** = numpy.array of real (dimesion Nnormal_modes x 3*Natoms) with 
          transformation matrix from cartesian displacements (without 
          roation and translation) to normal mode displacements. 
          (in colmns there are norma mode displacements corresponding
          to single cartesian displacement)
        * **'Frequency'** = numpy.array of real (dimension Nnormal_modes) frequency of
          normal modes (by default in INVERSE CETIMETER)
          but units of individual properties are listed in key `'Units'`
          transformation constant into ATOMIC UNITS is: 
          freq_in_AU = freq_in_cm/const.InternalToInvcm
        * **'RedMass'** = numpy.array of real (dimension Nnormal_modes) with reduced masses
          of normal modes (by default in ATOMIC MASS UNITS)
          but units of individual properties are listed in key `'Units'`
        * **'ForceConst'** = numpy.array of real (dimension Nnormal_modes) with force
          constats (by default in cm-1/Angstrom^2) transformation to ATOMIC UNITS
          is: ForceConst_AU = ForceConst_(cm-1/Angstrom^2) /const.HaToInvcm*(const.BohrToAngstrom^2)
        * **'Nmodes'** = Number of normal modes
        * **'Units'** = dictionary with strings specifying units for individula properties
     
    el_inter : dictionary
        Contains the information about single and double electron integrals.
        **Not yet implemented transformation from QCinfo class to this one**
    
    
        
    Functions
    ---------
    rotate :
        Rotate molecule with all its properties by specified angles in radians 
        in positive direction.
    rotate_1 :
        Inverse totation to rotate.
    move :
        Moves molecule with all its properties along specified vector.
    center :
        Center the molecule and allign in plane
    copy :
        Create 1 to 1 copy of the tmolecule with all its properties.
    print_atom_info :
        Prints information about specified atom - position, basis functions...
    print_ao_info :
        Prints information about specified atomic orbital
    load_Gaussian_fchk :
        Load molecule information from Gaussian 09 formated checkpoint file 
        (.fchk file)
    load_Gaussian_density_fchk :
        Reads density matrix from Gaussian 09 formated checkpoint file. Needed
        for read of transition density matrix.
    load_Gaussian_log :
        Reads additional information from Gaussian 09 .log file. For example 
        excited state properties, vibrational informations, SAC calculation...
    load_xyz :
        Loads structure of the molecule from xyz file
    load_pdb :
        Loads structure of the molecule from pdb file
    load_mol2 :
        Loads structure and some aditional information about the molecule from
        mol2 file
    read_TrEsp_charges :
        Reads the molecule from TrEsp charges fiting procedure. If molecule is 
        allready allocated only charges are read
    load_transition_density :
        Load and add transition desity into excited state and use the
        density to initialize molecule if not yet initialized.
    output_to_xyz :
        Create xyz file for the Molecule.
    output_mol2 :
        Output molecular information into mol2 file which can be used for 
        AMBER MD simulations.
    output_to_mathematica :
        Output molecular information into Mathematica notebook for visualization
    output_transition_density :
        Output density to cube file
    get_com :
        Outputs center of mass of the molecule
    guess_bonds :
        Add bonds betwenpairs of atoms.
    allocate_transition_dipole :
        Calculate transition dipole for selected excitation and revrite transition
        dipole in ``self.exct[exct_i].dipole``
    get_transition_atomic_charges :
        Calculate transition atomic charges for multipole expansion in atomic
        distances
    get_transition_dipole :
        Calculate transition atomic dipoles for multipole expansion in atomic
        distances. Transition atomic dipoles defined as: 
        ``tr_dip[i]=sum_{mu in atom i}{sum_{nu}{sum_{i->j}{c_{i->j}*c_{i,mu}*c_{j,nu}*
        <AO_mu| [x-x_mu,y-y_mu,z-z_mu] |AO_nu> }}}``.
    get_transition_atomic_quadrupole :
        Calculate transition atomic quadrupoles for multipole expansion in atomic 
        distances
    get_ESP_grid :
        Calculates electrostatic potential on grid - for ESP charges fitting.
    get_transition_density :
        Calculate transition density on given grid
    get_excitation_type :
        Determines the transition type ('Pi -> Pi', 'Pi -> Sigma', 
        'Sigma -> Pi' or 'Sigma -> Sigma') for all transitions in molecule.
    get_mo_cube :
        Create cube density for specified molecular orbital
    create_multipole_representation :
        Create multipole representation of the molecule. Needed for calculation
        of interaction energy with multipole expansion in atomic distances.
    create_3D_oscillator_representation :
        Create 3D classical oscillator representation of the structure. This 
        representation can be then used for calculation of interaction energies
        by classical oscillator method
    
    
    
    get_tr_densmat :
        Calculate transition electron density matrix
    
    '''
    
    def __init__(self,name):
        self.name = name
        self.vib_spec = {}
        self.energy = None    # ground state energy       
        self.el_inter = {}  # kinetic energy, potential energy, core energy, fermi energy, double electron integrals
        self.mol_spec = {}  # Overlap matrix,bonds,transition charges,Atomic dipoles (co nejobecnejsi definice+funkce na vypocet atomovych dipolu z telo definice),quadrupoly...        

        self.struc=Structure()
        self.ao=AO()        # atomic orbitals
        self.mo=MO()        # So far only one set of molecular orbitals will be avilable but this could be extended to list [MO,localized MO,CASSCF MO,...] 
        self.repre={'AllAtom': self.struc}       # Molecule representation - structure file with full atom representation, with Pi-conjugated chain representation, TrEsp representation....
        self.exct=[]

    def load_Gaussian_fchk(self,filename, exct_i = 0,**kwargs):
        '''Load molecule information from Gaussian 09 formated checkpoint file 
        (.fchk file)
        
        Parameters
        ----------
        filename : string
            Name of fchk file including the path if needed
        exct_i : integer (optional init = 0)
            Specified for which excited state excited density matrix is written
            if fchk file
        **kwargs : dictionary (optional)
            Specifies which density should be read from checkpoint file. If not present
            ground state density is read and stored in variable self.mo.densmat_grnd and if
            also excited state density is present it will be written in exct_dens
            in self.exct[0].densmat_exct. If non optimized excited state density is needed \n
            kwargs has to be specified as follows:
                 **{'CI_rho_density': True}
        '''
        
        self.struc,self.ao,self.mo,TotalEnergy,exct_dens,hess_fchk=read_gaussian_fchk(filename,**kwargs)
        self.energy=TotalEnergy
        
        # Vibrational information (because right hessian matrix is defined in fchk file)
        if len(hess_fchk)!=0:
            self.vib_spec['Hessian']=hess_fchk # Jak se zmeni energie pri posunuti podel kartezskych souradnic
            self.vib_spec['NMinCart']=None
            self.vib_spec['CartInNM']=None
            self.vib_spec['Frequency']=None
            self.vib_spec['RedMass']=None
            self.vib_spec['ForceConst']=None
            self.vib_spec['Nmodes']=None
            #self.vib_spec['Units']='frequency[cm-1],Reduced masses[AMU(atomic mass units),Force constats [cm-1/(Angstrom^2)],InterToCart dimensionles expansion coefficients (but in this context trasform iternal coordinate in Angstrom to cartesian displacement in angstrom)]'
            self.vib_spec['Units']={'Hessian': 'Hartree/(Bohr^2)'}

        # Excited state density matrix
        if len(exct_dens)!=0:
            if len(self.exct)==0:
                for ii in range(exct_i+1):
                    self.exct.append(Excitation(None,None))
                self.exct[exct_i].densmat_exct=np.copy(exct_dens)
            else:
                self.mol_spec['ElDensGauss_exct']=np.copy(exct_dens)
    
    def load_Gaussian_density_fchk(self,filename,typ='Ground',exct_i=0):
        ''' Reads electron density matrix from gaussian formated checkpoint file.
        For obtaining correct density matrixes for excited and transition
        states Gaussian calculation has to be done with **DENSITY** keyword.
        
        Parameters
        ----------
        filename : string
            Name of fchk file including the path if needed
        typ : string (optional - init='Ground')
        
            * ``typ='Ground'`` ground state density matrix will be written into
              self.mo.densmat_grnd.
            * ``typ='Excited'`` excited state density
              matrix will be written into self.exct[exct_i].densmat_exct.
            * ``typ='Transition'`` transition density matrix will be written
              into self.exct[exct_i].densmat_trans.
            * ``typ='ExcitedNoOpt'`` nonoptimized excited state electron 
              density matrix will be written into self.exct[exct_i].densmat_exct
        
        Notes
        ---------
        For reading the transition electron density matrix, excited state calculation
        has to be performed with **DENSITY** keyword and resulting chk and fchk
        files have to be converted by chk2den and writedensity.py (from Benedetta
        Mennucci group) into fchk files where ground state density matrix is 
        replaced by transition density matrix.
    
        '''
# TODO: excited state density matrix read in excited class
        if typ=='Ground' or typ=='Excited' or typ=='Transition':
            structure, ao, mo, TotalEnergy, exct_dens, hess_fchk = read_gaussian_fchk(filename)
        elif typ=='ExcitedNoOpt':
            structure, ao, mo, TotalEnergy, exct_dens, hess_fchk = read_gaussian_fchk(filename, **{'CI_rho_density': True})
        
        # Electronic density:
        if typ=='Ground':
            if len(mo.densmat_grnd)!=0:
                self.mo.densmat_grnd=np.copy(mo.densmat_grnd)
        elif typ=='ExcitedNoOpt':
            if len(mo.densmat_grnd)!=0:
                self.exct[exct_i].densmat_exct=np.copy(mo.densmat_grnd)
        elif typ=='Excited':
            if len(exct_dens)!=0:
                self.exct[exct_i].densmat_exct=np.copy(exct_dens)
        elif typ=='Transition':
            if len(mo.densmat_grnd)!=0:
                self.exct[exct_i].densmat_trans=np.copy(mo.densmat_grnd)
            else:
                print("There is no transition density in the fchk file")
            
    def print_atom_info(self,ii):
        """
        Prints information about specified atom
        
        Parameters
        ----------
        ii : integer
            Index of the atom for which information is printed (starting from 0)
        
        """
        
        if self.struc.nat==0:
            print('Atoms have not been defined yet')
        elif self.struc.nat<=ii:
            print('Atom index is higher than number of atoms in the system')
            print('Number of defined atoms is',str(self.struc.nat),sep="")
        else:
            if len(self.struc.coor.value)==0:
                raise IOError('Atom have to be defined at least by position')
            if len(self.struc.at_type)!=0:
                print('Atom: ',ii+1,self.struc.at_type[ii],sep="")
                print('         Position: ',self.struc.coor.value[ii],' ',self.struc.coor.unit_repr(),sep="")
            else:
                print('Atom type unknown')
                print('         Position: ',self.struc.coor.value[ii],' ',self.struc.coor.unit_repr(),sep="")
            if self.struc.mass is not None:
                print('         Mass: ',self.struc.mass[ii],' AMU',sep="")
            else:
                print('         Mass: undefined')
            if self.struc.ncharge is not None:
                print('         Nuclear charge: ',self.struc.ncharge[ii],' e',sep="")
            else:
                print('         Nuclear charge: undefined')
            if len(self.struc.at_type)!=0:
                AOrb=[]
                counter = {}
                for jj in range(self.ao.nao):
                    if int(self.ao.atom[jj].indx)==ii:
                        typ=''.join([i for i in self.ao.type[jj] if not i.isdigit()])
                        if typ in counter:
                            counter[typ]+=1
                        else:
                            counter[typ]=1+l_quant(typ)
                        AOrb.append([counter[typ],self.ao.type[jj]])
                    elif int(self.ao.atom[jj].indx)==ii+1:
                        break
                print('         Atomic orbitals located on this atom: ',end="")
                # asi by to chtelo seradit dle prvniho sloupce
                for jj in range(len(AOrb)):
                    if AOrb[jj][1]=='5d' or AOrb[jj][1]=='7f':
                        print(AOrb[jj][0],"(",AOrb[jj][1],")"," ",sep="",end="")
                    else:
                        print(AOrb[jj][0],AOrb[jj][1]," ",sep="",end="")
                print(" ")
            print(" ")

    def get_com(self):
        """
        Calculate center of mass of the molecule
        
        Returns
        -------
        Rcom : Coordinate class
            Position of molecular center of mass
        """
        
        Rcom=self.struc.get_com()
        return Rcom

    def get_atom_info(self,ii):
        return [ii+1,self.struc.at_type[ii],self.struc.coor.value[ii],self.struc.mass[ii],self.struc.ncharge[ii]]
    
    def get_ao_info(self,ii):
        return [self.ao.type[ii][1],self.ao.type[ii][0],self.ao.coor.value[ii],self.ao.coeff[ii],self.ao.exp[ii],self.ao.atom[ii].type,self.ao.atom[ii].indx]
            
    def print_ao_info(self,ii):
        """
        Prints information about specified atomic orbital
        
        Parameters
        ----------
        ii : integer
            Index of the atomic orbital for which information is printed 
            (starting from 0)
        
        """
        
        if not self.ao.init:
            print('Atomic orbitals have not been defined yet')
        elif self.ao.nao<=ii:
            print('Atomic orbital index is higher than number of atoms in the system')
            print('Number of defined atomic orbitals is ',str(self.ao.nao),sep="")
        else:
            atom_indx=self.ao.atom[ii].indx
            typ_orig=''.join([i for i in self.ao.type[ii] if not i.isdigit()])
            counter = l_quant(typ_orig)
            for jj in range(ii+1):
                if int(self.ao.atom[jj].indx)==atom_indx:
                    typ=''.join([i for i in self.ao.type[jj] if not i.isdigit()])
                    if typ==typ_orig:
                        counter+=1
            print('Atomic orbital: ',counter,self.ao.type[ii],sep="")
            print('         Centered at atom: ',self.ao.atom[ii].indx+1,self.ao.atom[ii].type,sep="")
            print('         Position: ',self.ao.coor.value[ii],' ',self.ao.coor.unit_repr(),sep="")
            print('         Possible orientations: ',l_orient_str(self.ao.type[ii]),sep="")
            print('         Possible orientation indexes: ',self.ao.orient[ii],sep="")
            print('         Expansion coefficients STO in GTO basis: ',self.ao.coeff[ii],sep="")
            print('         Exponents for expansion of STO in GTO basis: ',self.ao.exp[ii],sep="")
            print('         Slater orbital definition:')
        print(' ')


    def load_Gaussian_log(self,filename,add_excit=True, force_rewrite=False):
        '''
        Reads additional information from Gaussian 09 .log file. For example 
        excited state properties, vibrational informations, SAC calculation...
        
        Parameters
        ----------
        filename : string
            Name of fchk file including the path if needed
        add_excit : logical (optional init = True)
            Set what should be done when some informations about excited states
            are allready loaded. If ``add_excit=True`` then excitation only
            excitation specified by root will be rewritten and the others will 
            be added to the end of the file. If ``add_excit=False`` only 
            excitation specified by root will be rewritten but no other 
            excitations will be added 
        force_rewrite : logical (optional init = False)
            If true only information about excited state and transition density
            matrix is kept for all states but all other information is rewritten
            
        '''
        
        # TODO: Coefficients should be loaded from zero and not from one
        job_info, Excit_list, NM_info, Single_el_prop, Double_el_int, SAC, SAC_CI = read_gaussian_log(filename)
        # Excited state properties: - defined diferently than other properties besause for example CASSCF needs different properties than TD-DFT
        if len(self.exct)==0:
            init=False
        else:
            init=True
        
        if not init:
            self.exct=Excit_list
        else:
            if force_rewrite:
                for ii in range(len(self.exct)):
                    try:
                        densmat_exct=self.exct[ii].densmat_exct
                        Excit_list[ii].densmat_exct=densmat_exct
                    except:
                        print("Excitation", ii,"doesn't have a excited state desnity matrix or it has not been defined yet")
                    try:
                        densmat_trans=self.exct[ii].densmat_trans
                        Excit_list[ii].densmat_trans=densmat_trans
                    except:
                        print("Excitation", ii,"doesn't have a transition desnity matrix or it has not been defined yet")
                self.exct=Excit_list
            else:
                for ii in range(len(Excit_list)):
                    if Excit_list[ii].root==ii+1:
                        try:
                            densmat_exct=self.exct[ii].densmat_exct
                        except:
                            densmat_exct = None
                        try :
                            densmat_trans=self.exct[ii].densmat_trans
                        except:
                            densmat_trans=None
    
                        excit=Excit_list[ii]
                        self.exct[ii]=excit
                        
                        if densmat_exct is not None:
                            self.exct[ii].densmat_exct=densmat_exct
                        if densmat_trans is not None:
                            self.exct[ii].densmat_trans=densmat_trans
                    elif add_excit:
                        excit=Excit_list[ii]
                        self.exct.append(excit)


        # Vibrational information:
        if NM_info is not None:
            nm_coef=NM_info[0]
            nm_info=NM_info[1]
        if len(self.vib_spec)!=0 and ('Hessian' in  self.vib_spec.keys()):
            if len(self.vib_spec['Hessian'])!=len(nm_coef[:,0]):
                raise IOError('You are probably loading different molecule because loaded Hessian does not mach information in gaussian log file')
        if NM_info is not None:
            #print(nm_info)
            self.vib_spec['NMinCart']=nm_coef # in columns cartesian displacements wich belong to particular normal mode
            self.vib_spec['CartInNM']=None
            freq=[nm_info[ii]['frequency_cm-1'] for ii in range(len(nm_info)) ]
            redmass=[nm_info[ii]['red_mass_AMU'] for ii in range(len(nm_info)) ]
            force=[nm_info[ii]['force_cons_mDyne/A'] for ii in range(len(nm_info)) ]
            irint=[nm_info[ii]['IR_intens_KM/Mole'] for ii in range(len(nm_info)) ]
            self.vib_spec['Frequency']=np.array(freq,dtype='f8')
            self.vib_spec['Units']['Frequency']='cm-1' 
            self.vib_spec['RedMass']= np.array(redmass,dtype='f8')
            self.vib_spec['Units']['RedMass']='AMU (atomic mass units)'
            self.vib_spec['ForceConst']=np.array(force,dtype='f8')
            self.vib_spec['Units']['ForceConst']='mDyne/A'
            self.vib_spec['IR_intens']=np.array(irint,dtype='f8')
            self.vib_spec['Units']['IR_intens']='KM/Mole'
            self.vib_spec['Nmodes']=len(nm_coef[0,:])
        # Transform To better units
        
        # Electronic interaction properties:
        #self.el_inter = {}
        
        # Molecule specification
        
    def guess_bonds(self):
        """
        Add bonds betwen atoms. Bonded atoms specified by cut-off distance.
        In this function default cutt-off distance is used - for more options
        use ``self.struc.guess_bonds`` alternative.
        
        Notes
        ------
        Information about bonds is stored at \n
        **self.struc.bonds** \n
        as list of integers (dimension Nbonds x 2) where self.bonds[i,0]<self.bonds[i,1]
        and self.bonds[i,0], and self.bonds[i,1] are indexes of atoms between 
        which bond is formed.
        """
        self.struc.guess_bonds()
    
    def load_xyz(self,filename):
        """
        Loads structure of the molecule from xyz file
        
        Parameters
        -------
        filename : string
            Name of xyz file including the path if needed
        
        Notes
        -------
        Structure stored at \n
        **self.struc** \n
        as Structure class 
        """
        self.struc.load_xyz(filename)
    
    def load_pdb(self,filename):
        """
        Loads structure of the molecule from pdb file
        
        Parameters
        -------
        filename : string
            Name of pdb file including the path if needed
        
        Notes
        -------
        Structure stored at \n
        **self.struc** \n
        as Structure class 
        """
        self.struc.load_pdb(filename)
    
    def load_mol2(self,filename,state='Ground'):
        """
        Loads structure and some aditional information about the molecule from
        mol2 file
        
        Parameters
        -------
        filename : string
            Name of mol2 file including the path if needed
        state : string (optional init = 'Ground')
            Which charges are present in mol2 file. If ``state='Ground'`` it is
            assumed that charges in mol2 file correspond to ground state 
            (default), therefore they are loaded in ``self.struc.esp_grnd``. 
            If ``state='Excited'`` it is assumed that charges in mol2 file 
            correspond to excited state, therefore they are loaded in
            ``self.struc.esp_exct``. If ``state='Transition'`` it is assumed that 
            charges in mol2 file are transition charges, therefore they are 
            loaded in ``self.struc.esp_trans``.
        
        Notes
        -------
        Structure stored at \n
        **self.struc** \n
        as Structure class and aditional moleculer information at \n
        **self.mol_spec**
        """
        self.name,self.mol_spec['Charge_method'],self.mol_spec['Aditional_info']=self.struc.load_mol2(filename,state=state)
    
    def output_mol2(self,state='Ground',name=None):
        """ Create mol2 file for the molecule. This file can be then used for
        AMBER MD simulations
        
        Parameters
        -----------
        filename : string
            Name of the output file including the path if needed (including the 
            .mol2 suffix)
        state : string (optional init = 'Ground')
            Which charges are used for mo2 file generation. If ``state='Ground'``
            -> ``self.struc.esp_grnd``, if ``state='Excited'`` -> 
            ``self.struc.esp_exct`` and
            if ``state='Transition'`` -> ``self.struc.esp_trans``
        name : 3 character string (optional init = None)
            3 upper case character name of the system. If none is defined then
            default 'XXX' is used.

        """
        
        if name is None:
            name=self.name
        if ('Charge_method' in self.mol_spec.keys()) and ('Aditional_info' in self.mol_spec.keys()):
            self.struc.output_mol2(state=state,ch_method=self.mol_spec['Charge_method'],Info=self.mol_spec['Aditional_info'],Name=name)
        elif 'Charge_method' in self.mol_spec.keys():
            self.struc.output_mol2(state=state,ch_method=self.mol_spec['Charge_method'],Name=name)
        elif 'Aditional_info' in self.mol_spec.keys():
            self.struc.output_mol2(state=state,Info=self.mol_spec['Aditional_info'],Name=name)
        else:
            self.struc.output_mol2(state=state,Name=name)

    def output_to_mathematica(self,filename,scDip=1.0): #exct_i=0
        """ Output molecular information into Mathematica for visualization of
        moleule and molecular quantities.
        
        Parameters
        -----------
        filename : string
            Name of the output file including the path if needed (including the 
            .mol2 suffix)
        scDip : float
            Scaling factor for outputing dipoles (if they are too small or too
            big for output)

        """
        
        # add uotput of mulickan charges
        if self.struc.bonds is None:
            self.struc.guess_bonds()
        if (self.struc.tr_dip is not None) and (self.struc.tr_char is not None):
            OutputMathematica(filename,self.struc.coor.value,self.struc.bonds,self.struc.at_type,scaleDipole=scDip,AtDipole=self.struc.tr_dip,TrPointCharge=self.struc.tr_char)
        elif (self.struc.tr_dip is not None):
            OutputMathematica(filename,self.struc.coor.value,self.struc.bonds,self.struc.at_type,scaleDipole=scDip,AtDipole=self.struc.tr_dip)
        elif (self.struc.tr_char is not None):
            OutputMathematica(filename,self.struc.coor.value,self.struc.bonds,self.struc.at_type,scaleDipole=scDip,TrPointCharge=self.struc.tr_char)
        else:
            OutputMathematica(filename,self.struc.coor.value,self.struc.bonds,self.struc.at_type)

#        if exct_i<len(self.exct_spec):
#            if 'at_trdip' in self.exct_spec[exct_i].keys() and 'at_trcharge' in self.exct_spec[exct_i].keys():
#                out.OutputMathematica(filename,self.at_spec['Coor'],self.mol_spec['Bonds'],self.at_spec['AtType'],scaleDipole=scDip,AtDipole=self.exct_spec[exct_i]['at_trdip'],TrPointCharge=self.exct_spec[exct_i]['at_trcharge'])
#            elif 'at_trdip' in self.exct_spec[exct_i].keys():
#                out.OutputMathematica(filename,self.at_spec['Coor'],self.mol_spec['Bonds'],self.at_spec['AtType'],scaleDipole=scDip,AtDipole=self.exct_spec[exct_i]['at_trdip'])
#            elif 'at_trcharge' in self.exct_spec[exct_i].keys():
#                out.OutputMathematica(filename,self.at_spec['Coor'],self.mol_spec['Bonds'],self.at_spec['AtType'],TrPointCharge=self.exct_spec[exct_i]['at_trcharge'])
#        else:
#            out.OutputMathematica(filename,self.at_spec['Coor'],self.mol_spec['Bonds'],self.at_spec['AtType'])
        
    def output_to_xyz(self,filename='Molecule.xyz'):
        """ Create xyz file for the Molecule.
        
        Parameters
        -----------
        filename : string (optional init = 'Molecule.xyz')
            Name of the output file including the path if needed (including the 
            .xyz suffix)
        """
        
        self.struc.output_to_xyz(filename)


# TODO: Leave this only in structure class
    def read_TrEsp_charges(self,filename,state='Ground',verbose=True):
        ''' Reads the molecule information from output file used for fiting of
        Esp charges. If the structure is not initialized all structural 
        informations are read and structure is set up. If the system is allready
        initialized only ESP charges are read. **THERE IS NO CHECK OF GEOMETRY 
        FOR THIS CASE. BE SURE THAT ORDERING OF ATOMS IS THE
        SAME IN THE INPUT FILE AND IN THE STRUCTURE**

        Parameters
        ---------
        filename : string
            Name of the input file including the path if needed. (Output filename
            from chargefit is: ``fitted_charges.out``)
        state : string (optional init = 'Ground')
            Which charges are present in input file. If ``state='Ground'`` the
            imput file was generated from ground state ESP fitting, therefore
            fitted charegs are loaded into: ``self.struc.esp_grnd``. If 
            ``state='Excited'`` the imput file was generated from excited state
            ESP fitting, therefore fitted charegs are loaded into: 
            ``self.struc.esp_exct``. If ``state='Transition'`` the imput file was 
            generated from transition ESP fitting, therefore fitted charegs are
            loaded into: ``self.struc.esp_trans``.
        verbose : logical (optional init = False)
            Controls if information about exact and ESP fitted dipole from atomic
            charges is printed.
        
        Notes
        ---------
        Reason for not checking the structure is that sometimes it is needed to
        read the charges into distorted system geometry.
      
        '''
        self.struc.read_TrEsp_charges(filename,state=state,verbose=verbose)
        
        
    def get_transition_atomic_charges(self,exct_i,mo_indx=0,verbose=False):
        """
        Calculate transition atomic charges for specified excitation
        for multipole expansion in atomic distances (for interaction energy
        calculation).
        
        Parameters
        ---------
        exct_i : integer
            Index specifying which transition should be used for calculation
            of transition atomic charges (self.exct[exct_i]) starting from zero.
        
        Notes
        ---------
        Transition atomic charges are stored at \n
        **self.struc.tr_char** \n
        and
        **self.exct[exct_i].tr_char**
        
        """
        
        if type(self.mo)!=list:
            self.struc.tr_char=self.exct[exct_i].get_transition_atomic_charges(self.ao,MO=self.mo,verbose=verbose)
        else:
            self.struc.tr_char=self.exct[exct_i].get_transition_atomic_charges(self.ao,MO=self.mo[mo_indx],verbose=verbose)
    
    def get_transition_dipole(self,exct_i,mo_indx=0):
        """Calculate transition atomic dipoles for specified excitation for
        multipole expansion in inter-atomic distances (for interaction energy
        calculation).

        Parameters
        ---------
        exct_i : integer
            Index specifying which transition should be used for calculation
            of transition atomic dipoles (self.exct[exct_i]) starting from zero.
        
        Returns
        ---------
        dipole : numpy array of float (dimension 3)
            Transition dipole for the molecular excitation
        
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
        **self.struc.tr_dip** \n
        and \n
        **self.exct[exct_i].tr_dip** \n
        as numpy array of float (dimension Natoms x 3)

        """ 
        
        if type(self.mo)!=list:
            dipole,self.struc.tr_dip=self.exct[exct_i].get_transition_dipole(self.ao,self.struc.coor,MO=self.mo)
        else:
            dipole,self.struc.tr_dip=self.exct[exct_i].get_transition_dipole(self.ao,self.struc.coor,MO=self.mo[mo_indx])        
        return dipole


    def allocate_transition_dipole(self,exct_i,mo_indx=0):
        """
        Calculate transition dipole for selected excitation and revrite transition
        dipole in **self.exct[exct_i].dipole**
        
         Parameters
        ---------
        exct_i : integer
            Index specifying which transition should be used for calculation
            of transition dipole.
        
        """
        self.exct[exct_i].dipole=self.get_transition_dipole(self,mo_indx=0)


    def get_transition_atomic_quadrupole(self,exct_i,mo_indx=0):
        """Calculate transition atomic quadrupoles for multipole expansion in 
        inter-atomic distances.
        
        Parameters
        ---------
        exct_i : integer
            Index specifying which transition should be used for calculation
            of transition atomic quadrupoles (self.exct[exct_i]) starting from
            zero.

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
        **self.exct[exct_i].tr_quadrR2** \n
        and \n
        **self.struc.tr_quadrR2** \n
        as numpy array of float (dimension 6 x Natoms) and at \n
        **self.exct[exct_i].tr_quadr2** \n
        and \n
        **self.struc.tr_quadr2** \n
        as numpy array of float (dimension Natoms)
        
        """
        
        if type(self.mo)!=list:
            self.struc.tr_quadr2,self.struc.tr_quadrR2=self.exct[exct_i].get_transition_atomic_quadrupole(self.ao,MO=self.mo)
        else:
            self.struc.tr_quadr2,self.struc.tr_quadrR2=self.exct[exct_i].get_transition_atomic_quadrupole(self.ao,MO=self.mo[mo_indx])        

           
    def move(self,dx,dy,dz):
        """ Moves the molecule with all its properties along the specified vector
        
        Parameters
        --------
        dx,dy,dz : float
            Displacements along x, y, and z coordinate
        """
        
        self.struc.move(dx,dy,dz)
        self.ao.move(dx,dy,dz)
        for ii in range(len(self.exct)):
            self.exct[ii].move(dx,dy,dz)
        for key in self.repre.keys():
            if key!='AllAtom':
                self.repre[key].move(dx,dy,dz)
            
    
    def _get_cartesian_rot_mat(self,rotxy,rotxz,rotyz):
        """
        Rotation matrix for cartesian coordinates of the molecule
        
        Parameters
        --------
        rotxy,rotxz,rotxz : float
            Rotation angles in radians around z, y and x axis (in xy, xz and yz plane)
        
        Returns
        --------
        TransfMat : numpy.array of float (dimension Natom x Natom)
            Transformation matrix for atomic coordinates. 
            Rotated_coor_in_line = np.dot(TransfMat,coor_reshaped_in_line)
        
        """
        
        
        RotA,RotB,RotC=fill_basis_transf_matrix('p',rotxy,rotxz,rotyz)
        Rot=np.dot(RotC,np.dot(RotB,RotA))
        TransfMat=np.dot(RotC,np.dot(RotB,RotA))
        for ii in range(1,self.struc.nat):
            TransfMat=scipy.linalg.block_diag(TransfMat,Rot)
        return np.array(TransfMat)
    
    
    def rotate(self,rotxy,rotxz,rotyz):
        """"
        Rotate the molecule and its properties around the coordinate origin by
        specified angles. First it rotate the molecule around z axis (in xy 
        plane), then around y axis and in the end around x axis in positive 
        direction (if right thumb pointing in direction of axis fingers are 
        pointing in positive rotation direction)
        
        Parameters
        --------
        rotxy,rotxz,rotxz : float
            Rotation angles in radians around z, y and x axis (in xy, xz and yz plane)
        
        """
        self.mo.rotate(rotxy,rotxz,rotyz,self.ao)
        self.struc.rotate(rotxy,rotxz,rotyz)
        self.ao.rotate(rotxy,rotxz,rotyz)        
        if len(self.exct)!=0:
            for ii in range(len(self.exct)):
                self.exct[ii].rotate(rotxy,rotxz,rotyz,self.ao)
                
                    
# TODO: Test this part
        if 'Hessian' in self.vib_spec.keys():
            RotMat=self._get_cartesian_rot_mat(rotxy,rotxz,rotyz)
            self.vib_spec['Hessian']=np.dot(RotMat,np.dot(self.vib_spec['Hessian'],RotMat.T))
        if 'NMinCart' in self.vib_spec.keys():
            if self.vib_spec['NMinCart'] is not None:
                RotMat=self._get_cartesian_rot_mat(rotxy,rotxz,rotyz)
                self.vib_spec['NMinCart']=np.dot(RotMat,self.vib_spec['NMinCart'])
        if 'CartInNM' in self.vib_spec.keys():
            if self.vib_spec['CartInNM'] is not None:
                RotMat=self._get_cartesian_rot_mat(rotxy,rotxz,rotyz)
                self.vib_spec['CartInNM']=np.dot(self.vib_spec['CartInNM'],RotMat.T)                  
    
    def rotate_1(self,rotxy,rotxz,rotyz):
        """" Inverse rotation of structure to **rotate** fuction.
        First rotate the molecule around x axis (in yz plane), then around
        y axis and in the end around z axis in negtive direction 
        (if left thumb pointing in direction of axis fingers are pointing in 
        negative rotation direction)
        
        Parameters
        --------
        rotxy,rotxz,rotxz : float
            Rotation angles in radians around z, y and x axis (in xy, xz and yz plane)
        """
        
        self.mo.rotate_1(rotxy,rotxz,rotyz,self.ao)
        self.struc.rotate_1(rotxy,rotxz,rotyz)
        self.ao.rotate_1(rotxy,rotxz,rotyz)        
        if len(self.exct)!=0:
            for ii in range(len(self.exct)):
                self.exct[ii].rotate_1(rotxy,rotxz,rotyz,self.ao)
                
                    
# TODO: Introduce inverse rotatio also for this part
        if 'Hessian' in self.vib_spec.keys():
            RotMat=self._get_cartesian_rot_mat(rotxy,rotxz,rotyz)
            self.vib_spec['Hessian']=np.dot(RotMat,np.dot(self.vib_spec['Hessian'],RotMat.T))
        if 'NMinCart' in self.vib_spec.keys():
            RotMat=self._get_cartesian_rot_mat(rotxy,rotxz,rotyz)
            self.vib_spec['NMinCart']=np.dot(RotMat,self.vib_spec['NMinCart'])
        if 'CartInNM' in self.vib_spec.keys():
            if self.vib_spec['CartInNM']!=None:
                RotMat=self._get_cartesian_rot_mat(rotxy,rotxz,rotyz)
                self.vib_spec['CartInNM']=np.dot(self.vib_spec['CartInNM'],RotMat.T)
    

    def center_molecule(self,indx_center,indx_x,indx_y,debug=False):
        """
        Centers the molecule and its properties according to defined center and
        two main axis. Axis and center are defined by atomic indexes and after 
        centering center would be in coordinate system origin, vector from 
        origin to atom indx_x will be aligned along x-axis and vector from 
        origin to indx_y will be in xy plane (z component of the vector will be
        zero)
        
        Parameters
        ----------
        indx_center : integer or list of integers
            Position of the center of the molecule. Origin of coordinate system
            will be shifted to this point. When `indx_center`=i it refers to 
            atomic coordnitate of ith atom (counted from zero) => center=coor[i,:].
            When `indx_center`=[i,j,k,..] than center is center of all listed
            atoms (average coordinate) => center=(coor[i,:]+coor[j,:]+coor[k,:]...)/N
        indx_x : int or list of int of length 2 or 4
            When `indx_x`=i than vector X is defined as coor[i,:]-center.
            When `indx_x`=[i,j] than vector X is defined as coor[j,:]-coor[i,:].
            When `indx_x`=[i,j,k,l] than vector X is defined as 
            (coor[j,:]-coor[i,:])+(coor[l,:]-coor[k,:]).
        indx_y : int or list of int of length 2 or 4
            When `indx_y`=i than vector Y is defined as coor[i,:]-center.
            When `indx_y`=[i,j] than vector Y is defined as coor[j,:]-coor[i,:].
            When `indx_y`=[i,j,k,l] than vector Y is defined as 
            (coor[j,:]-coor[i,:])+(coor[l,:]-coor[k,:]).
        
        Returns
        ---------
        Phi,Psi,Chi : float
            Rotation angles around z, y and x axis which are used for rotation 
            of the molecule 
        center : numpy array (dimension 3)
            Position of center (before moving to origin of coordinate system)
            
        Notes
        --------
        To do the same transformation on different system first move the system 
        by same vector as original structure: \n 
        ``move(-center[0],-center[1],-center[2])`` \n
        and then rotate it: \n
        ``rotate(Phi,Psi,Chi)`` \n
        or alternatively with position_tools: \n
        ``coor_moved=position_tools.RotateAndMove(coor,-center[0],-center[1],-center[2],0.0,0.0,0.0)``
        ``coor_rotated=position_tools.RotateAndMove(coor_moved,0.0,0.0,0.0,Phi,Psi,Chi)``
            
        """
        New_coor,Phi,Psi,Chi,center=CenterMolecule(self.struc.coor.value,indx_center,indx_x,indx_y,print_angles=True,debug=debug)
        self.move(-center[0],-center[1],-center[2])
        self.rotate(Phi,Psi,Chi)
        
        return Phi,Psi,Chi,center

        
    def get_ESP_grid(self,state_indx=0,load_grid=None,gridfile="ESPGrid",grid=None,verbose=True):
        '''Calculates electrostatic potential on grid.
        
          **For this kind of calculation it is important to run Gaussian wit 5D 7F option**

        Parameters
        ----------
        state_indx : integer
            Index of state for which is ESP calculated:
                
            * 0 = ground state (default)
            * 1 = 1. excited state
            * 2 = 2. excited state
            * and so on...
            
            For excited state - ground state, transition and excited state will be calculated.
          
        load_grid : string
            
            * **'qchem':** grid file format for QChem ESP calculation will be loaded
            * **'GausESP':** grid file format for Gaussian ESP calculation will be loaded
            
        Optional parameters
        ----------
        gridfile : string (init = "ESPGrid")
            File name from which we load the grid points
        grid : grid class
            Definition of orthogonal grid for calculation of ESP cube files
          
        Returns
        ----------
          potential_grnd : numpy array (dimension Npoints x 4)
              potential_grnd[ii,0:3] are cartesian coordinates of point where
              ground state ESP was calculated in ATOMIC UNITS (Bohr) and 
              potential_grnd[ii,0:3] are values of the ground state electrostatic 
              potential in ATOMIC UNITS
          potential_tran : numpy array (dimension Npoints x 4)
              potential_tran[ii,0:3] are cartesian coordinates of point where
              transition ESP was calculated in ATOMIC UNITS (Bohr) and 
              potential_tran[ii,0:3] are values of the transition electrostatic 
              potential in ATOMIC UNITS
          potential_exct : numpy array (dimension Npoints x 4)
              potential_exct[ii,0:3] are cartesian coordinates of point where
              excited state ESP was calculated in ATOMIC UNITS (Bohr) and 
              potential_exct[ii,0:3] are values of the excited state electrostatic 
              potential in ATOMIC UNITS
        
        Notes
        ----------
        Excited state and transition ESP outputed only if excited state calculation
        is requested
        '''
        
        from ..Qch2PYscf import potential_basis_PYscf_grid
        
        debug=True
        if state_indx==0: # ESP of ground state will be calculated.
            do_ground_state=True
        else:
            do_ground_state=False
        
        if verbose:
            print(' ')
            print(' ')
            print('    BEGINING OF ESP CALCULATION:')
            
        ''' Calculation of ground state density matrix '''
        if self.mo.densmat_grnd is None:
            if type(self.mo)!=list:
                self.mo.get_grnd_densmat()
            else:
                raise IOError('Calculation of ESP for molecule with more MO sets is not implemented')
        
        if verbose:
            print('        First four lines of lower triangle of ground state density matrix:')
            for ii in range(4):
                print('            ',self.mo.densmat_grnd[ii,:ii+1])
        
#self.exct[exct_i].densmat_exct
#self.exct[exct_i].densmat_trans
        if not do_ground_state:
            exct_i=state_indx-1
            # Excited state coef matrix is given by M_mat=sum_{a->j} sum_{n=1,Nocc/2 where a->j} 2* C^{2}_{a,j} *C_{n,mu} *C_{n,nu}
            if self.exct[exct_i].densmat_exct is None:
                raise IOError('Calculation of excited state electron density is not implemented due to errors in its calculation')
            if self.exct[exct_i].densmat_trans is None:
                raise IOError('Calculation of transition state electron density is not implemented due to errors in its calculation')
            
            
            if verbose:
                print('        First four lines of lower triangle of excited state density matrix:')
                for ii in range(4):
                    print('            ',self.exct[exct_i].densmat_exct[ii,:ii+1])
                print(" ")
                print('        First four lines of lower triangle of transition density matrix:')
                for ii in range(4):
                    print('            ',self.exct[exct_i].densmat_trans[ii,:ii+1])
                print(" ")
                   
                    
        M_mat_grd = np.copy(self.mo.densmat_grnd)
        if not do_ground_state:
            M_mat_tran = np.copy(self.exct[exct_i].densmat_trans)
            M_mat_exct = np.copy(self.exct[exct_i].densmat_exct)
        
        if (load_grid is None) and (grid is None):
            grid=self.struc.get_grid()
        
        if (load_grid is None) and (grid is not None):
            print(np.shape(grid.X))
            Npoints=np.product(np.shape(grid.X))
            print('Pocet bodu v gridu: ',Npoints)
            positions=np.zeros((Npoints,3),dtype='f8')
            positions[:,0]=np.reshape(grid.X,Npoints)
            positions[:,1]=np.reshape(grid.Y,Npoints)
            positions[:,2]=np.reshape(grid.Z,Npoints)
        elif load_grid=='qchem':
            if verbose:
                print('        Reading qchem grid from file:',gridfile)
# TODO: use Unit manager to be sure that I'm using Atomic Units
            with position_units('Bohr'):
                positions=read_qchem_grid(gridfile)
        elif load_grid=='GausESP':
            if verbose:
                print('        Reading Gaussien esp grid from file:',gridfile)
# TODO: use Unit manager to be sure that I'm using Atomic Units
            with position_units('Bohr'):
                positions,ESP_Gauss,Coor_gauss=read_gaussian_esp(gridfile)
        
        if verbose:
            print('           First three grid points:')
            print('            ',positions[0])
            print('            ',positions[1])
            print('            ',positions[2])
            
            print("        Grid for ESP calcualtion loaded/generated") 
            print(" ")
        
        if do_ground_state:
            Nmat=1
            M_mat=M_mat_grd
        else:
            Nmat=3
            M_mat=[M_mat_grd,M_mat_tran,M_mat_exct]

        if verbose:
            print("        Starting ESP calculation with PYscf...")

        Npoints=len(positions)
        potential_el,potential_nuc=potential_basis_PYscf_grid(self,positions,M_mat,Nmat)
        
        if verbose:
            print("        ESP potential calculated") 
            print("        Load grid: ",load_grid)
        
        if debug:
            print("             Debuging part - writing separated electronic and nuclear potential")
            if do_ground_state:
                self.mol_spec['Grnd_el_pot']=np.copy(potential_el)
                self.mol_spec['Nuc_pot']=np.copy(potential_nuc)
                self.mol_spec['Points_pot']=np.copy(positions)
            else:
                self.mol_spec['Grnd_el_pot']=np.copy(potential_el[:,0])
                self.mol_spec['Tran_el_pot']=np.copy(potential_el[:,1])
                self.mol_spec['Exct_el_pot']=np.copy(potential_el[:,2])
                self.mol_spec['Nuc_pot']=np.copy(potential_nuc)
                self.mol_spec['Points_pot']=np.copy(positions)
            print("             End of Debuging part")
            
        if load_grid=='GausESP' or load_grid=='qchem':
            print("        Output for loaded grids") 
            if do_ground_state:
                print("           ...only ground state will be written")
                potential_grnd=np.zeros((Npoints,4))
                potential_grnd[:,0]=positions[:,0]
                potential_grnd[:,1]=positions[:,1]
                potential_grnd[:,2]=positions[:,2]
                potential_grnd[:,3]=potential_el+potential_nuc
                print('Ground state ESP calculation finished')
                return potential_grnd
                
            else:
                print("           ...ground, transition and excited state will be written")
                potential_exct=np.zeros((Npoints,4))
                potential_exct[:,0]=positions[:,0]
                potential_exct[:,1]=positions[:,1]
                potential_exct[:,2]=positions[:,2]
                for ii in range(Npoints):
                    potential_exct[ii,3]=potential_el[ii,2]+potential_nuc[ii]                
                
                potential_grnd=np.zeros((Npoints,4))
                potential_grnd[:,0]=positions[:,0]
                potential_grnd[:,1]=positions[:,1]
                potential_grnd[:,2]=positions[:,2]
                for ii in range(Npoints):
                    potential_grnd[ii,3]=potential_el[ii,0]+potential_nuc[ii]
                
                potential_tran=np.zeros((Npoints,4))
                potential_tran[:,0]=positions[:,0]
                potential_tran[:,1]=positions[:,1]
                potential_tran[:,2]=positions[:,2]
                potential_tran[:,3]=np.copy(potential_el[:,1])
                print('        ESP calculation finished')
                return potential_grnd,potential_tran,potential_exct
                
        else:
            if do_ground_state:
                potential_grnd=np.reshape(potential_el+potential_nuc,np.shape(grid.X))
            else:
                potential_grnd=np.reshape(potential_el[:,0]+potential_nuc,np.shape(grid.X))
                potential_tran=np.reshape(potential_el[:,1],np.shape(grid.X))
                potential_exct=np.reshape(potential_el[:,2]+potential_nuc,np.shape(grid.X))
            
            
            step_mat=grid.delta
            # save result as density class
            if do_ground_state:
                self.mol_spec['Grnd_ESP']=DensityGrid(grid.origin,np.shape(grid.X),step_mat,potential_grnd,typ='transition',mo_indx=state_indx,Coor=self.struc.coor.value,At_charge=self.struc.ncharge)
            else:
                self.mol_spec['Grnd_ESP']=DensityGrid(grid.origin,np.shape(grid.X),step_mat,potential_grnd,typ='transition',mo_indx=state_indx,Coor=self.struc.coor.value,At_charge=self.struc.ncharge)
                self.mol_spec['exct_ESP']=DensityGrid(grid.origin,np.shape(grid.X),step_mat,potential_exct,typ='transition',mo_indx=state_indx,Coor=self.struc.coor.value,At_charge=self.struc.ncharge)
                self.mol_spec['trans_ESP']=DensityGrid(grid.origin,np.shape(grid.X),step_mat,potential_tran,typ='transition',mo_indx=state_indx,Coor=self.struc.coor.value,At_charge=self.struc.ncharge)            
            print('ESP calculation finished')
            

    def get_transition_density(self,exct_i,nt=0,cutoff=0.01,List_MO=None,AOalloc=False,extend=5.0,step=0.4):
        """Evaluate transition density on given grid
        
        Parameters
        ----------
        exct_i : integer
            Index specifying which transition should be used for calculation
            of transition density (self.exct[exct_i]) starting from
            zero.
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
        AOalloc : logical (optional init = False)
            Evaluate atomic orbital basis on the grid ad store it at 
            **self.ao.grid**
        extend : float (optional init = 5.0)
            Extension of the grid in every dimension from farthermost atoms
        step : float (optional init = 0.4)
            Spacing between individual grid points
            
        
        Returns
        ---------
        trdens_cub : Density class
            Transition density evaluated on the grid in cube file format
        
        Notes
        ---------
        Transition desnity is also stored at: \n
        **self.exct[exct_i].dens_trans** \n
        as Density class. Used grid for evaluation of transition density is read
        from **self.struc.grid**, if not present default setting will be used for 
        grid alocation
        
        """
        
        if self.struc.grid is None:
            self.struc.get_grid(extend=extend,step=step)
        if AOalloc and (self.ao.grid is None):
            self.ao.get_all_slater_grid(self.struc.grid,nt=nt)
        trdens_cub=self.exct[exct_i].get_transition_density(self.ao,self.struc.grid,self.struc.coor.value,self.struc.ncharge,MO=self.mo,nt=nt,cutoff=cutoff,List_MO=List_MO)
        return trdens_cub
    
        
    
    def get_mo_cube(self,mo_i,AOalloc=False,extend=5.0,step=0.4,nt=0):
        """ Create cube density for specified molecular orbital
        
        Parameters
        ----------
        mo_i : integer
            Index of molecular orbital which will be evaluated on the grid 
            (starting from 0)
        AOalloc : logical (optional init = False)
            Evaluate atomic orbital basis on the grid ad store it at 
            **self.ao.grid**
        extend : float (optional init = 5.0)
            Extension of the grid in every dimension from farthermost atoms
        step : float (optional init = 0.4)
            Spacing between individual grid points
        nt : integer (optional init = 0)
            Specifies how many cores should be used for the calculation.
            Secial cases: 
            ``nt=0`` all available cores are used for the calculation. 
            ``nt=1`` serial calculation is performed.
            ``nt=N`` Ncores are used for the calculation.
        
        Returns
        --------
        mo_cube : Density class
            Cube density of specified molecular orbital.
            
        Notes
        ------
        Used grid for evaluation of transition density is read
        from **self.struc.grid**, if not present default setting will be used for 
        grid alocation
        """
        
        if self.struc.grid is None:
            self.struc.get_grid(extend=extend,step=step)
        if AOalloc and (self.ao.grid is None):
            self.ao.get_all_slater_grid(self.struc.grid,nt=nt)
            
        grid=self.struc.grid
        mo_grid=self.mo.get_mo_grid(self.ao,grid,mo_i)
        step=np.zeros((3,3))
        step[0,0]=grid.delta[0]
        step[1,1]=grid.delta[1]
        step[2,2]=grid.delta[2]
        mo_cube=DensityGrid(np.array(grid.origin),np.shape(grid.X),step,mo_grid,typ='mo',mo_indx=mo_i+1,Coor=self.struc.coor._value,At_charge=self.struc.ncharge)
        return mo_cube
    
    
    def get_excitation_type(self,krit=85,nvec=[0.0,0.0,1.0]):
        """ Determines the transition type ('Pi -> Pi', 'Pi -> Sigma', 
        'Sigma -> Pi' or 'Sigma -> Sigma') for all transitions in molecule.
        
        Parameters
        ----------
        krit : float (optional - init=90)
            Percentage of pi state contribution which will be considered as 
            pi-molecular orbital. accepted values are (0,100) 
        nvec : numpy array or list dimension 3 (optional - init=[0.0,0.0,1.0])
            Normal vector to pi-conjugated plane.

        Returns
        ----------
        Excit_Type : list of string (dimension Nexcit)
            Type of transition ('Pi -> Pi', 'Pi -> Sigma', 'Sigma -> Pi' or 
            'Sigma -> Sigma')
        
        """
        
        Excit_Type=[]
        MO_type=self.mo.get_MO_type(self.ao,krit,nvec)
        for ii in range(len(self.exct)):
            Excit_Type.append(self.exct[ii].excitation_type(MO_type,krit))
        return Excit_Type
        
    def output_transition_density(self,exct_i,filename='trans_dens.cub'):
        ''' Output density to cube file 
        
        Parameters
        ----------
        exct_i : integer
            Index specifying which transition from which transition density 
            should be outputed (self.exct[exct_i]) starting from
            zero. 
        filename : string (optional init = 'trans_dens.cub')
            Output file name including the path to output folder
        
        '''
        
        self.exct[exct_i].dens_trans.output(filename)
    
    def _add_transition_density(self,exct_i,density):
        """
        Add transition desity into excited state properties or use the density
        to initialize molecule if not yet initialized.
        
        Parameters
        ----------
        exct_i : integer
            Index specifying to which transition (self.exct[exct_i]) excitation
            correspond. 
        density : Density class
            Transition density as Density class
            
        Notes
        ----------
        Transition density is stored at \n
        **self.exct[exct_i].dens_trans** \n
        as Density class.
        
        """ 
        # check if density coresponds to same molecule - same atomic positions:
        if self.struc.init:
            # Defined molecule has already defined atoms
            if len(density.coor.value)!=self.struc.nat:
                raise Warning('Molecule definition from transition density has different number of atoms than defined molecule')
                return
            if are_similar(density.coor.value,self.struc.coor.value,do_print=True,treshold=0.001,min_number=3e-4):
                if len(self.exct)==0:
                    for ii in range(exct_i+1):
                        self.exct.append(Excitation(None,None,dip_units='AU'))
                self.exct[exct_i].dens_trans=density
            else:
                raise Warning('Atomic positions from density are not similar to already defined molecule. If you are certain what you are doing write: mol_name..exct_spec[exct_i]["trans_dens"]=density')
                return
        else:
            # Atomic positions have not been defined yet and therefore they will be defined from density
            
# TODO: check atomic type definition
            #at_type=['X']*len(density.at_charge)
            self.struc.add_atom(density.coor,None,ncharge=density.at_charge)

            if len(self.exct)==0:
                for ii in range(exct_i+1):
                    self.exct.append(Excitation(None,None,dip_units='AU'))
                self.exct[exct_i].dens_trans=density
            else:
                if exct_i>=len(self.exct_spec):
                    for ii in range(exct_i+1-len(self.exct_spec)):
                        self.exct.append(Excitation(None,None,dip_units='AU'))
                    self.exct[exct_i].dens_trans=density
                else:
                    self.exct[exct_i].dens_trans=density

    
    def load_transition_density(self,exct_i,filename):
        """
        Load and add transition desity into excited state properties or use the
        density to initialize molecule if not yet initialized.
        
        Parameters
        ----------
        exct_i : integer
            Index specifying to which excitation (self.exct[exct_i]) transition density
            correspond. 
        filename : string
            Cube file name with transition density including the path to output folder 
            
        Notes
        ----------
        Transition density is stored at \n
        **self.exct[exct_i].dens_trans** \n
        as Density class.
        
        """
        density=DensityGrid(None,None,None,None)
        density.import_cub(filename)
        self._add_transition_density(exct_i,density)
    
    def create_multipole_representation(self,exct_i,rep_name='AllAtom',MultOrder=0):
        """
        Create multipole representation of the molecule. Needed for calculation
        of interaction energy with multipole expansion in atomic distances.
        
        Parameters
        ----------
        exct_i : integer
            Index specifying which excitation (self.exct[exct_i]) should be used
            for creating multipole representation
        rep_name : string (optional init = 'AllAtom')
            Name of the representation into which atomic multipoles will be added
            if the representation is not yet defined it will be created.
        MultOrder : integer
            Multipole order: 0 -> transition charges, 1 -> transition dipoles, 
            2 -> transition quadrupoles
        
        Notes
        -------
        Atomic qaudrupoles will be stored at \n
        **self.repre[rep_name].tr_char** \n
        **self.repre[rep_name].tr_dip** \n
        **self.repre[rep_name].tr_quadr2** \n
        **self.repre[rep_name].tr_quadrR2** \n
        and also in \n
        **self.exct[exct_i].tr_char** \n
        **self.exct[exct_i].tr_dip** \n
        **self.exct[exct_i].tr_quadr2** \n
        **self.exct[exct_i].tr_quadrR2** \n
        
        """
        
        if MultOrder>-1:
            if self.exct[exct_i].tr_char is None:
                self.get_transition_atomic_charges(exct_i)
            at_charges=self.exct[exct_i].tr_char.copy()
        if MultOrder>0:
            if self.exct[exct_i].tr_dip is None:
                self.get_transition_dipole(exct_i)
            at_dipoles=self.exct[exct_i].tr_dip.copy()
        if MultOrder>1:
            if self.exct[exct_i].tr_quadr2 is None:
                self.get_transition_atomic_quadrupole(exct_i)
            at_quad_r2=self.exct[exct_i].tr_quadr2.copy()
            at_quad_rR2=self.exct[exct_i].tr_quadrR2.copy()
        
        if not (rep_name in self.repre.keys()):
            structure_new=self.struc.copy()
            self.repre[rep_name]=structure_new
        self.repre[rep_name].tr_char=at_charges.copy()
        self.repre[rep_name].tr_dip=at_dipoles.copy()
        self.repre[rep_name].tr_quadr2=at_quad_r2.copy()
        self.repre[rep_name].tr_quadrR2=at_quad_rR2.copy()
    
    def create_3D_oscillator_representation(self,NMN1,TrDip1,At_list=None,scale_by_overlap=False,nearest_neighbour=True,centered="Bonds",verbose=False,rep_name='Oscillator',**kwargs):
        '''
        Create 3D classical oscillator representation of the structure. This 
        representation can be then used for calculation of interaction energies
        by classical oscillator method
        
        Parameters
        ----------
        NMN : integer or list of integers
            Index of oscillator normal mode which will be used for representation
            of the structure. If two or more are written linear combination of
            both will be used for representation of structure (in this case
            also coefficients of linear combination have to be specified in **kwargs).
        TrDip : real
            Total transition dipole size of excitation which is represented by
            classical oscillator in ATOMIC UNITS (Bohr*charge_in_e)
        At_list : list of integers (optional)
            List of indexes atoms used for 3D classical oscillator representation.
            If not present all atoms in the structure will be used
        scale_by_overlap : logical (optionalinit = False)
            Controls if the interaction between two atoms should be scaled by
            overlap of corresponding atomic p-orbitals. DO NOT USE THIS OPTION
        nearest_neighbour : logical (optional init = True)
            If ``True`` only nearest neighbour interaction is used. Interaction
            between more distant atoms is set to zero. If ``false`` interaction
            between all atoms is calculated by dipole dipole interaction
        centered : string (optional init = "Bonds")
            Controls if point dipoles should be centered in the center of the 
            bonds (``centered="Bonds"``) or on individual atoms 
            (``centered="Atoms"``). SO FAR ONLY CENETRING ON THE BONDS IS 
            IMPLEMENTED
        rep_name : string (optional init = 'Oscillator')
            Name of the molecule representation into which 3D classical oscilator 
            representation will be added. If the representation is not yet 
            defined it will be created.
        **kwargs : dictionary
            specification of aditional parameers when using linear combination of
            more transitions. With name ``'TotTrDip'`` there have to be defined 
            transition dipoles for individual transitions. With name ``'Coef'``
            is definition of expansion coefficients for tranitions.
            
        Notes
        -------
        3D classical oscillator representation will be stored at \n
        **self.repre[rep_name]** \n
        as Structure class where in 'tr_dip' there are stored classical dipoles
        in ATOMIC UNITS and in 'coor' there are coordinates of the dipoles 
        '''
        
        if len(kwargs)==0:
            oscillator_struc=self.struc.get_3D_oscillator_representation(NMN1,TrDip1,At_list=At_list,scale_by_overlap=scale_by_overlap,nearest_neighbour=nearest_neighbour,centered=centered,verbose=verbose)
        else:
            oscillator_struc=self.struc.get_3D_oscillator_representation(NMN1,TrDip1,At_list=At_list,scale_by_overlap=scale_by_overlap,nearest_neighbour=nearest_neighbour,centered=centered,verbose=verbose,**kwargs)
        self.repre['Oscillator']=oscillator_struc
    
    def copy(self):
        ''' Create deep copy of the all informations in the molecule. 
        
        Returns
        ----------
        mol_new : MO class
            Molecule class with exactly the same information as previous one 
            
        '''
        mol_new=deepcopy(self)
        mol_new.name=''.join([self.name,' copy'])
        return mol_new
        
# TODO:    def get_HOMO_indx(self):
#        
#        

# TODO: create Huckel representation

            

    def add_transition(self,coeff,energy):
        """
        Add transitionto the molecule with calculated transition dipole and 
        specified transition energy. Transition is defined by list of transition
        coefficients.
        
        Parameters
        ---------
        coeff : list of integer and real (dimension Nsingle_trans x 3)
            At the first position is initial molecular orbital (MO) - **starting from 
            1**, at the 2nd there is final MO - **starting from 1** - and at the 3rd
            it is coefficient of contribution of this single electron transition in 
            whole transition. (for some methods sum of the coefficients sre normalized
            to 1/2, for others to 1)
        energy : float or Energy class
            Transition energy for new transition
            
        Notes
        -------
        Excitation is added to list of excitations to \n
        **self.exct**
        
        """
        
        root=len(self.exct)
        excit=Excitation(energy,None,coeff=coeff,method='Generated',root=root)
        dipole,at_dipole=excit.get_transition_dipole(self.ao,self.struc.coor,MO=self.mo)
        excit.dipole=Dipole(np.array(dipole,dtype='f8'),'AU')
        self.exct.append(excit)
                    
         
'''
    mol_spec : dictionary
        Contains additional information about molecule such as bonds, electron 
        densities...
        keys:
        
        * **'ElDensGauss'** = numpy.array of real (dimension N_AO_orient x N_AO_orient)
          Ground state density matrix
        * **'ElDensGauss_exct'** = numpy.array of real (dimension N_AO_orient x N_AO_orient)
          Excited state density matrix
        * **'Grnd_ESP'** = DensityGrid class with ground state electrostatic potential 
        * **'GrDens_mat'** = numpy.array of real dimension (N_orient_orbitals x N_orient_orbitals)
          ground state electron density matrix defined as `GrDens_mat[mu,nu]=Sum_MO_i{occn_i*C_mu,i*C_nu,i}}`
          where C_mu,i is expansion coefficient of MO i into AO mu
        * **'Grnd_el_pot'** = numpy.array of real (dimension Npoints) with electrostatic
          potential in ATOMIC UNITS (Hartree/e) induced by atomic electrons for 
          molecule in ground state
        * **'Tran_el_pot'** =  numpy.array of real (dimension Npoints) with electrostatic
          transition potential in ATOMIC UNITS (Hartree/e) induced by electronic 
          transition density
        * **'Exct_el_pot'** = numpy.array of real (dimension Npoints) with electrostatic
          potential in ATOMIC UNITS (Hartree/e) induced by atomic electrons for 
          molecule in excited state
        * **'Nuc_pot'** = numpy.array of real (dimension Npoints) with electrostatic
          potential in ATOMIC UNITS (Hartree/e) induced by atomic cores
        * **'Points_pot'** = numpy.array of real (dimension Npoints x 3) with
          coordinates in ATOMIC UNITS of points where electrostatic potential is calculated         
        * **'ElDensGauss'** = numpy.array of real dimension (N_orient_orbitals x N_orient_orbitals)
          ground state electron density defined as `gr_dens[mu,nu]=Sum_i{occn_i*C_mu,i*C_nu,i}`.
          Total electron density is then defined as 
          `rho(r)=Sum_mu,nu{gr_dens[mu,nu]*AO-STO_mu(r)*AO-STO_nu(r)}`. Only diference
          to GrDens_mat is that to this variable can be imported also transition 
          density (if fchk file was generated by chk2den from excited state calculation)
        * **'tot-dens'** = DensityGrid class with ground state total electron density
        * **'Charge_method'** = Method used for calculation of atomic charges
        * **'Aditional_info'** = Some aditional info about molecule (for mol2 input/output)
   
    exct_spec : list of dictionaries
        For every excitationthere is dictionary with more information. First 
        excitation is with index 0.
        keys: 
        
        * **'coefficients'** : list of integer and real (dimension Nsingle_trans x 3)
          At the fisrt position is initial molecular orbital (MO), at the 2nd
          there is final MO and at the 3rd it is coefficient of contribution
          of this single electron transition in whole transition. (for some 
          methods sum of the coefficients is normalized to 1/2, for others to 1)
        * **'energy_eV'** = transition energy in eV
        * **'energy_nm'** = transition energy in nm
        * **'oscillator'** = oscillator strength of given transition in ATOMIC UNITS
        * **'symmetry'** = symmetry of excited state
        * **'trans_dip'** = numpy.array of real (dimension 3) with transition dipole 
          in ATOMIC UNITS (e*Bohr)
        * **'method'** = string with name of the method used for the excited state 
          calculation ('CIS','TDDFT','ZINDO')
        * **'multiplicity'** = multiplicity of excited state (singlet/triplet)
        * **'root'** = state which was specified for excited state calculation (most precise)
        * **'TrESP_charges'** = numpy.array of real (dimension Natoms) with transition
          atomic charges calculated from fitting ESP
        * **'ESP_charges'** = numpy.array of real (dimension Natoms) with excited state
          atomic charges calculated from fitting ESP
        * **'TrESP_dipole'** = numpy.array of real (dimension 3). Transition dipole
          calculated from atomic transition charges defined in `'TrESP_charges'`
        * **'ESP_dipole'** = numpy.array of real (dimension 3). Excited state dipole
          calculated from atomic excited state charges defined in `'ESP_charges'`
        * **'exct_ESP'** = DensityGrid class with excited state electrostatic potential 
        * **'trans_ESP'** = DensityGrid class with transition state electrostatic potential
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
        * **'trans_dens'** = DensityGrid class with transition electron density
    vib_spec : dictionary
        Contains information about vibrational properties of the molecule
        keys:
        
        * **'Hessian'** = numpy.array of dimension (3*Natoms x 3*Natoms) with Hessian 
          matrix in ATOMIC UNITS (Hartree/Bohr^2) which tells us how energy 
          is changed with change of single atom cartesian coordinate
        * **'NMinCart'** = numpy.array of dimesion (3*Natoms x Nnormal_modes) with 
          transformation matrix from normal mode displacements to cartesian
          displacements. (in colmns there are cartesian displacements corresponding
          to single normal mode)
        * **'CartInNM'** = numpy.array of real (dimesion Nnormal_modes x 3*Natoms) with 
          transformation matrix from cartesian displacements (without 
          roation and translation) to normal mode displacements. 
          (in colmns there are norma mode displacements corresponding
          to single cartesian displacement)
        * **'Frequency'** = numpy.array of real (dimension Nnormal_modes) frequency of
          normal modes (by default in INVERSE CETIMETER)
          but units of individual properties are listed in key `'Units'`
          transformation constant into ATOMIC UNITS is: 
          freq_in_AU = freq_in_cm/const.InternalToInvcm
        * **'RedMass'** = numpy.array of real (dimension Nnormal_modes) with reduced masses
          of normal modes (by default in ATOMIC MASS UNITS)
          but units of individual properties are listed in key `'Units'`
        * **'ForceConst'** = numpy.array of real (dimension Nnormal_modes) with force
          constats (by default in cm-1/Angstrom^2) transformation to ATOMIC UNITS
          is: ForceConst_AU = ForceConst_(cm-1/Angstrom^2) /const.HaToInvcm*(const.BohrToAngstrom^2)
        * **'Nmodes'** = Number of normal modes
        * **'Units'** = dictionary with strings specifying units for individula properties
    
    el_inter : dictionary
        Contains the information about single and double electron integrals.
        **Not yet implemented transformation from QCinfo class to this one**
    mol_spec : dictionary
        Contains additional information about molecule such as bonds, electron 
        densities...
        keys:
        
        * **'ElDensGauss'** = numpy.array of real (dimension N_AO_orient x N_AO_orient)
          Ground state density matrix
        * **'ElDensGauss_exct'** = numpy.array of real (dimension N_AO_orient x N_AO_orient)
          Excited state density matrix
        

        * **'Grnd_ESP'** = DensityGrid class with ground state electrostatic potential 
        * **'GrDens_mat'** = numpy.array of real dimension (N_orient_orbitals x N_orient_orbitals)
          ground state electron density matrix defined as `GrDens_mat[mu,nu]=Sum_MO_i{occn_i*C_mu,i*C_nu,i}}`
          where C_mu,i is expansion coefficient of MO i into AO mu
        * **'Grnd_el_pot'** = numpy.array of real (dimension Npoints) with electrostatic
          potential in ATOMIC UNITS (Hartree/e) induced by atomic electrons for 
          molecule in ground state
        * **'Tran_el_pot'** =  numpy.array of real (dimension Npoints) with electrostatic
          transition potential in ATOMIC UNITS (Hartree/e) induced by electronic 
          transition density
        * **'Exct_el_pot'** = numpy.array of real (dimension Npoints) with electrostatic
          potential in ATOMIC UNITS (Hartree/e) induced by atomic electrons for 
          molecule in excited state
        * **'Nuc_pot'** = numpy.array of real (dimension Npoints) with electrostatic
          potential in ATOMIC UNITS (Hartree/e) induced by atomic cores
        * **'Points_pot'** = numpy.array of real (dimension Npoints x 3) with
          coordinates in ATOMIC UNITS of points where electrostatic potential is calculated         
        * **'ElDensGauss'** = numpy.array of real dimension (N_orient_orbitals x N_orient_orbitals)
          ground state electron density defined as `gr_dens[mu,nu]=Sum_i{occn_i*C_mu,i*C_nu,i}`.
          Total electron density is then defined as 
          `rho(r)=Sum_mu,nu{gr_dens[mu,nu]*AO-STO_mu(r)*AO-STO_nu(r)}`. Only diference
          to GrDens_mat is that to this variable can be imported also transition 
          density (if fchk file was generated by chk2den from excited state calculation)
        * **'tot-dens'** = DensityGrid class with ground state total electron density
        * **'AO_multiwfn_overlap'** = the same as **'AO_overlap'** but this 
          quantity is loaded from Multiwfn analysis of Gaussian results 
        * **'Grnd_ESP_charges'** = numpy.array of real (dimension Natoms) with ground state
          atomic charges calculated from fitting ESP - if charges are not here 
          there could be in **at_spec['ESP_charges']**
        * **'Charge_method'** = Method used for calculation of atomic charges
        * **'Aditional_info'** = Some aditional info about molecule (for mol2 input/output)
''' 
        
def l_orient_str(l):
    ''' Possible orientation of atomic orbitals in same order as in gaussian output '''
    if isinstance(l,str):
        if l=='s':
            orient='1'
        elif l=='p':
            orient=['X','Y','Z']
        elif l=='d':
            orient=['XX','YY','ZZ','XY','XZ','YZ']
        elif l=='f':
            orient=['XXX','YYY','ZZZ','XYY','XXY','XXZ',
                    'XZZ','YZZ','YYZ','XYZ']
        elif l=='5d':
            orient=['2ZZ-XX-YY','XY','YZ','XX-YY','XY']
        elif l=='7f':
            orient=['-ZXX-ZYY','-XXX-XYY','-YXX-YYY','XXZ-YYZ','XYZ',
                    'XXX-XYY','XXY-YYY']
        else:
            raise IOError('Unsupported type of orbital in l_orient')
    else:
        if l==0:
            orient='1'
        elif l==1:
            orient=['X','Y','Z']
        elif l==2:
            orient=['XX','YY','ZZ','XY','XZ','YZ']
        elif l==3:
            orient=['XXX','YYY','ZZZ','XYY','XXY','XXZ',
                    'XZZ','YZZ','YYZ','XYZ']
        elif l==-2:
            orient=['2ZZ-XX-YY','XY','YZ','XX-YY','XY']
        elif l==-3:
            orient=['-ZXX-ZYY','-XXX-XYY','-YXX-YYY','XXZ-YYZ','XYZ',
                    'XXX-XYY','XXY-YYY']
        else:
            raise IOError('Unsupported type of orbital in l_orient')
    return orient

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
                
def add_quant(Orbital,typ):
    if typ=='s':
        Orbital[0]+=1
    elif typ=='p':
        Orbital[1]+=1
    elif typ=='d' or typ=='5d':
        Orbital[2]+=1
    elif typ=='f' or typ=='7f':
        Orbital[3]+=1
    elif typ=='g':
        Orbital[4]+=1
    else:
        raise IOError('Unsupported orbital type:',typ)

def l_quant(typ):
    if typ=='s':
        return 0
    elif typ=='p':
        return 1
    elif typ=='d' or typ=='5d':
        return 2
    elif typ=='f' or typ=='7f':
        return 3
    elif typ=='g':
        return 4
    else:
        raise IOError('Unsupported orbital type:',typ)
    
        
        # funkce ktera z ao vygeneruje mozne orientace        
        
supp_orbitals=['s','p','d','5d','f']        


orbit = 'spdfghijk'
lquant = dict([(j, i) for i,j in enumerate(orbit)])
'''
Transformation Between Cartesian and (Real) Pure Spherical Harmonic Gaussians

adapted from H.B. Schlegel and M.J. Frisch 
International Journal of Quantum Chemistry, Vol. 54, 83-87 (1995).
'''
sqrt = np.sqrt
cart2sph = [ #: Transformation Between Cartesian and (Real) Pure Spherical Harmonic Gaussians
  [
  [[(0,0,0)], [1.], 1.]
  ],                                    # s orbitals
  [
  [[(0,1,0)], [1.], 1.],
  [[(0,0,1)], [1.], 1.],
  [[(1,0,0)], [1.], 1.],
  ],                                    # p orbitals
  [
  [[(1,1,0)], [1.], 1.],
  [[(0,1,1)], [1.], 1.],
  [[(0,0,2),(2,0,0),(0,2,0)], [1., -1/2., -1/2.], 1.],
  [[(1,0,1)], [1.], 1.],
  [[(2,0,0),(0,2,0)], [1.,-1.], sqrt(3)/2.],
  ],                                    # d orbitals
  [
  [[(0,3,0),(2,1,0)], [-sqrt(5), 3.], 1/(2.*sqrt(2))],
  [[(1,1,1)], [1.], 1.],
  [[(0,1,2),(0,3,0),(2,1,0)], [sqrt(3/5.), -sqrt(3)/4., -sqrt(3)/(4.*sqrt(5))], sqrt(2)] ,
  [[(0,0,3),(2,0,1),(0,2,1)], [1.,-3/(2*sqrt(5)),-3/(2*sqrt(5))], 1.],
  [[(1,0,2),(3,0,0),(1,2,0)], [sqrt(3/5.), -sqrt(3)/4., -sqrt(3)/(4.*sqrt(5))], sqrt(2)],
  [[(2,0,1),(0,2,1)], [1.,-1.], sqrt(3)/2.],
  [[(3,0,0),(1,2,0)], [sqrt(5), -3.], 1/(2.*sqrt(2))],
  ],                                    # f orbitals
  [
  [[(3,1,0), (1,3,0)], [1.,-1.], sqrt(2) * sqrt(5/8.)],
  [[(0,3,1), (2,1,1)], [-sqrt(5)/4.,3/4.], sqrt(2)],
  [[(1,1,2), (3,1,0), (1,3,0)], [3/sqrt(14), -sqrt(5)/(2*sqrt(14)), -sqrt(5)/(2*sqrt(14))], sqrt(2)],
  [[(0,3,1), (0,3,1), (2,1,1)], [sqrt(5/7.), -3*sqrt(5)/(4.*sqrt(7)), -3/(4.*sqrt(7))], sqrt(2)],
  [[(0,0,4), (4,0,0), (0,4,0), (2,0,2), (0,2,2), (2,2,0)], [1., 3/8., 3/8., -3*sqrt(3)/sqrt(35), -3*sqrt(3)/sqrt(35), -1/4.], sqrt(2)],
  [[(1,0,3), (3,0,1), (1,2,1)], [sqrt(5/7.), -3*sqrt(5)/(4.*sqrt(7)), -3/(4.*sqrt(7))], sqrt(2)],
  [[(2,0,2), (0,2,2), (4,0,0), (0,4,0)], [3*sqrt(3)/(2.*sqrt(14)), -3*sqrt(3)/(2.*sqrt(14)), -sqrt(5)/(4.*sqrt(2)), sqrt(5)/(4.*sqrt(2))], sqrt(2)],
  [[(3,0,1), (1,2,1)], [sqrt(5)/4., -3/4.], sqrt(2)],
  [[(4,0,0), (0,4,0), (2,2,0)], [sqrt(35)/(8.*sqrt(2)), sqrt(35)/(8.*sqrt(2)), -3*sqrt(3)/(4.*sqrt(2))], sqrt(2)],
  ],                                    # g orbitals
]

# Molden AO order 
exp = []
exp.append([(0,0,0)])                   # s orbitals

exp.append([(1,0,0), (0,1,0), (0,0,1)]) # p orbitals

exp.append([(2,0,0), (0,2,0), (0,0,2),
            (1,1,0), (1,0,1), (0,1,1)]) # d orbitals

exp.append([(3,0,0), (0,3,0), (0,0,3),
            (1,2,0), (2,1,0), (2,0,1),
            (1,0,2), (0,1,2), (0,2,1),
            (1,1,1)])                   # f orbitals
    
exp.append([(4,0,0), (0,4,0), (0,0,4),
            (3,1,0), (3,0,1), (1,3,0),
            (0,3,1), (1,0,3), (0,1,3),
            (2,2,0), (2,0,2), (0,2,2),
            (2,1,1), (1,2,1), (1,1,2)]) # g orbitals
            
VdW_radius={'ca': 1.9080,'ha': 1.4590,'f': 1.75}

def get_cart2sph(l,m):
  '''Returns the linear combination required for the transformation Between 
  the Cartesian and (Real) Pure Spherical Harmonic Gaussian basis.
  
  Adapted from H.B. Schlegel and M.J. Frisch,
  International Journal of Quantum Chemistry, Vol. 54, 83-87 (1995).
  
  **Parameters:**
  
  l : int
    Angular momentum quantum number.
  m : int
    Magnetic quantum number.
  
  **Returns:**
  
  cart2sph[l][l+m] : list
    Contains the conversion instructions with three elements
      
      1. Exponents of Cartesian basis functions (cf. `core.exp`): list of tuples
      2. The corresponding expansion coefficients: list of floats 
      3. Global factor  
  
  ..hint: 
  
    The conversion is currently only supported up to g atomic orbitals.
  '''
  return cart2sph[l][l+m]

def get_lxlylz(ao_spec,get_assign=False):
  '''Extracts the exponents lx, ly, lz for the Cartesian Gaussians.
  
  **Parameters:**
  
  ao_spec : 
    See :ref:`Central Variables` in the manual for details.
  get_assign : bool, optional
    Specifies, if the index of the atomic orbital shall be returned as well.
  
  **Returns:**
  
  lxlylz : numpy.ndarray, dtype=numpy.int64, shape = (NAO,3)
    Contains the expontents lx, ly, lz for the Cartesian Gaussians.
  assign : list of int, optional
    Contains the index of the atomic orbital in ao_spec.
  '''
  
  lxlylz = []
  assign = []
  for sel_ao in range(len(ao_spec)):
    if 'exp_list' in ao_spec[sel_ao].keys():
      l = ao_spec[sel_ao]['exp_list']
    else:
      l = exp[lquant[ao_spec[sel_ao]['type']]]
    lxlylz.extend(l)
    assign.extend([sel_ao]*len(l))
  if get_assign:
    return np.array(lxlylz,dtype=np.int64), assign
  
  return np.array(lxlylz,dtype=np.int64) 


#==============================================================================
# TESTS
#==============================================================================
       
'''----------------------- TEST PART --------------------------------'''
if __name__=="__main__":
    print(' ')
    print('TESTS')
    print('-------')

    ''' fchk file read'''
    mol=Molecule('TEST')
    #molecule.QchMolecule.vib_spec[]
    if (platform=='cygwin' or platform=="linux" or platform == "linux2"):
        MolDir='/mnt/sda2/Dropbox/PhD/Programy/Python/Test/'
    elif platform=='win32':
        MolDir='C:/Dropbox/PhD/Programy/Python/Test/'
    mol.load_Gaussian_fchk("".join([MolDir,'ethen_freq.fchk']))
    # Atomic orbital info OK
    # Atomic structure OK
    # Molecular orbitals OK
    #mol.get_com() OK


    
    ''' Normal modes - read'''
    from sys import platform
    from math import isclose
    
    mol=Molecule('TEST')
    #molecule.QchMolecule.vib_spec[]
    if (platform=='cygwin' or platform=="linux" or platform == "linux2"):
        MolDir='/mnt/sda2/Dropbox/PhD/Programy/Python/Test/'
    elif platform=='win32':
        MolDir='C:/Dropbox/PhD/Programy/Python/Test/'
    mol.load_Gaussian_fchk("".join([MolDir,'ethen_freq.fchk']))
    mol.load_Gaussian_log("".join([MolDir,'ethen_freq.log']))
    Freq=[832.2234,961.4514,976.8042,1069.4514,1241.7703,1388.7119,1483.4161,
          1715.6638,3144.5898,3160.3419,3220.2977,3246.2149]    
    RedMass=[1.0427,1.5202,1.1607,1.0078,1.5261,1.214,1.112,3.1882,1.0477,
             1.0748,1.1146,1.1177]
    Forces=[0.4255,0.8279,0.6525,0.6791,1.3865,1.3794,1.4417,5.5292,6.104,
            6.3247,6.8102,6.9394]
    Hessian=[[6.38411997e-01, 2.75141057e-05,-5.37681873e-13,-2.66270576e-01,
              1.21243801e-01, 1.07762585e-12,-2.66282120e-01,-1.21273001e-01,
              -8.60166016e-13,-1.13450589e-01,-1.02575193e-06,-1.03742764e-12,
              3.79460942e-03, 2.99655336e-02, 1.33045136e-12, 3.79667933e-03,
              -2.99628217e-02,-5.05861479e-14],
              [2.75141057e-05,8.78790767e-01, 9.97851160e-13, 1.25022780e-01,
               -1.31318626e-01,-3.07467951e-13,-1.25051703e-01,-1.31342208e-01,
               -1.22702053e-12,-6.29420112e-07,-5.90037931e-01, 2.64148199e-12,
               -2.06887340e-03,-1.30465096e-02,-1.63484391e-12, 2.07091150e-03,
               -1.30454932e-02,-3.17165323e-13]]
    test=True
    for ii in range(len(Freq)):
        if (not isclose(mol.vib_spec['Frequency'][ii],Freq[ii],abs_tol=1e-4)) or \
           (not isclose(mol.vib_spec['RedMass'][ii],RedMass[ii],abs_tol=1e-4)) or \
           (not isclose(mol.vib_spec['ForceConst'][ii],Forces[ii],abs_tol=1e-4)):
            print(ii,mol.vib_spec['Frequency'][ii],Freq[ii],mol.vib_spec['RedMass'][ii],RedMass[ii],mol.vib_spec['ForceConst'][ii],Forces[ii])
            test=False 
    for ii in range(2):
        for jj in range(len(Hessian[ii])):
            if (not isclose(mol.vib_spec['Hessian'][ii,jj],Hessian[ii][jj],abs_tol=1e-7)):
                test=False 
    if mol.vib_spec['Nmodes']!=len(Freq):
        test=False
    if test:
        print('Normal mode read     ...    OK')
    else:
        print('Normal mode read     ...    Error')
        
        
    if 0:
        ''' Rotation test '''
        is_Translation=True
        is_Rotation=True
        
        test=False
        # Overit normu MO
        # Overit jestli se nejak nemeni i AO overlap => prinejmensim rotaci se overlap nemeni
        # Overit na papire jestli spravne nasobim tou matici rozvojovych koef. Jestli ma  byt M.T*SS nebo M*SS
    
        if (platform=='cygwin' or platform=="linux" or platform == "linux2"):
            MolDir='/mnt/sda2/Dropbox/PhD/Programy/Python/Test/'
        elif platform=='win32':
            MolDir='C:/Dropbox/PhD/Programy/Python/Test/'
    
        exct_index=0
        mo_index=0
        mol1=Molecule('N7-Polyene')
        mol1.load_Gaussian_fchk("".join([MolDir,'N7_exct_CAM-B3LYP_LANL2DZ_geom_CAM-B3LYP_LANL2DZ.fchk']))
        mol1.load_Gaussian_log("".join([MolDir,'N7_exct_CAM-B3LYP_LANL2DZ_geom_CAM-B3LYP_LANL2DZ.log']))
        mol1.get_transition_atomic_charges(0)
        dipole=mol1.get_transition_dipole(0)
        d_dip=np.sqrt(np.dot(dipole-mol1.exct[exct_index].dipole.value,dipole-mol1.exct[exct_index].dipole.value))
        if d_dip<0.02:
            print('Calculation of transition dipole   ...    OK      - diference in dipoles: {:0.4f}AU'.format(d_dip))
        else:
            print('Calculation of transition dipole   ...    Error   - diference in dipoles: {:0.4f}AU'.format(d_dip))
    
        SS_befRot=np.copy(mol1.ao.overlap)
        if is_Translation:
            mol1.move(12.0,-10.0,6.5)
            mol1.move(0.0,5.0,-2.0)
        if is_Rotation:
            rotxy=scipy.pi/5
            rotxz=-scipy.pi/6
            rotyz=scipy.pi/3
            mol1.rotate(rotxy,rotxz,rotyz)
            TransfMat=mol1.ao._get_ao_rot_mat(rotxy,rotxz,rotyz)
        SS_afterRot=np.copy(mol1.ao.overlap)
    #    mol1.ao.get_overlap()
    #    SS_trasf=np.copy(mol1.ao.overlap)
        SS_trasf=np.dot(TransfMat,np.dot(SS_befRot,TransfMat.T))
        if are_similar(SS_trasf,SS_afterRot) and are_similar(SS_afterRot,SS_trasf):
            test=True
        
        if test:
            print('Rotation of AO       ...    OK')
        else:
            print('Rotation of AO       ...    Error')


    ''' Multipole interaction test '''
    if (platform=='cygwin' or platform=="linux" or platform == "linux2"):
        MolDir='/mnt/sda2/Dropbox/PhD/Programy/Python/Test/'
    elif platform=='win32':
        MolDir='C:/Dropbox/PhD/Programy/Python/Test/'
    import Program_Manager.QuantumChem.interaction as inter
    from Program_Manager.QuantumChem.Classes.general import Energy,PositionAxis
    import matplotlib.pyplot as plt
    
    exct_index=0
    # Load molecule from gaussian output
    mol1=Molecule('N7-Polyene')
    mol1.load_Gaussian_fchk("".join([MolDir,'N7_exct_CAM-B3LYP_LANL2DZ_geom_CAM-B3LYP_LANL2DZ.fchk']))
    mol1.load_Gaussian_log("".join([MolDir,'N7_exct_CAM-B3LYP_LANL2DZ_geom_CAM-B3LYP_LANL2DZ.log']))
    # create Huckel representation
    indx_Huckel=np.where(np.array(mol1.struc.at_type)=='C')[0]
    Huckel_mol1=mol1.struc.get_Huckel_molecule([0.0, 0.0, 1.0],At_list=indx_Huckel)
    # Create classical oscillator representation
    mol1.create_3D_oscillator_representation(0,mol1.exct[exct_index].dipole.value,At_list=indx_Huckel)
    
    # assign multipoles
    mol1.create_multipole_representation(exct_index,rep_name='Multipole2',MultOrder=2)
    Huckel_mol1.create_multipole_representation(0,rep_name='Multipole2',MultOrder=2)
    dipole=mol1.get_transition_dipole(exct_index)
    
    print('Transition dipole:',dipole)
    print('Gaussian dipole:',mol1.exct[exct_index].dipole.value)
    print('Huckel dipole:',Huckel_mol1.exct[exct_index].dipole.value)
    
    # duplicate molecules and move them 5Angstroms in z direction
    mol2=mol1.copy()
    Huckel_mol2=Huckel_mol1.copy()
    with position_units("Angstrom"):
        mol2.move(0.0,0.0,5.0)
        Huckel_mol2.move(0.0,0.0,5.0)
        TrVec=Coordinate(mol2.struc.coor.value[0]+mol2.struc.coor.value[1]-mol2.struc.coor.value[6]-mol2.struc.coor.value[7])
        TrVec.normalize()
        print(TrVec.value)
    print(TrVec.value)
    
    # Calculation of interaction energy
#    TrVec.value=TrVec.value*const.AmgstromToBohr   # length of translation vector will be 1 Angstrom in Bohrs
    eng=inter.multipole_at_distances(mol1.repre['Multipole2'],mol2.repre['Multipole2'],MultOrder=2)
    E_mult2=Energy(None)
    E_mult1=Energy(None)
    E_mult0=Energy(None)
    E_Huckel2=Energy(None)
    E_Oscillator=Energy(None)
    E_dip=Energy(None)
    X_axis=PositionAxis(0.0,np.linalg.norm(TrVec.value),41)
    with position_units('Angstrom'):
        for ii in range(41):
            E_mult0.add_energy(inter.multipole_at_distances(mol1.repre['Multipole2'],mol2.repre['Multipole2'],MultOrder=0))
            E_mult1.add_energy(inter.multipole_at_distances(mol1.repre['Multipole2'],mol2.repre['Multipole2'],MultOrder=1))
            E_mult2.add_energy(inter.multipole_at_distances(mol1.repre['Multipole2'],mol2.repre['Multipole2'],MultOrder=2))
            E_Huckel2.add_energy(inter.multipole_at_distances(Huckel_mol1.repre['Multipole2'],Huckel_mol2.repre['Multipole2'],MultOrder=2))
            E_Oscillator.add_energy(inter.multipole_at_distances(mol1.repre['Oscillator'],mol2.repre['Oscillator'],MultOrder=2))
            E_dip.add_energy(inter.dipole_dipole_molecule(mol1,0,mol2,0))
            mol2.move(TrVec.value[0],TrVec.value[1],TrVec.value[2])
            Huckel_mol2.move(TrVec.value[0],TrVec.value[1],TrVec.value[2])


    with energy_units("1/cm"):
        with position_units("Angstrom"):
            plt.plot(X_axis.value,E_mult0.value)
            plt.plot(X_axis.value,E_mult1.value)
            plt.plot(X_axis.value,E_mult2.value)
            plt.plot(X_axis.value,E_dip.value, color="black", linestyle="--")
            plt.ylim([-700,1800])
            plt.xlim([min(X_axis.value),max(X_axis.value)])
            plt.legend(["Gaussian multipole order=0","Gaussian multipole order=1","Gaussian multipole order=2","Dipole-dipole"])
            plt.title("Interaction energy Gaussian multipole \n comparison of different orders")
            plt.show()
            
            plt.plot(X_axis.value,E_mult2.value,color='blue')
            plt.plot(X_axis.value,E_Huckel2.value, color="red")
            plt.plot(X_axis.value,E_Oscillator.value, color='green')
            plt.plot(X_axis.value,E_dip.value, color="black", linestyle="--")
            plt.ylim([-700,1800])
            plt.xlim([min(X_axis.value),max(X_axis.value)])
            plt.legend(["Gaussian multipole","Huckel multipole","Classical oscillator","Dipole-dipole"])
            plt.title("Interaction energy \n comparison of different approximations")
            plt.show()
    

    if 0:
        ''' Transition density calculation '''
        grid_alloc=False
        transdens=False
        
        if (platform=='cygwin' or platform=="linux" or platform == "linux2"):
            MolDir='/mnt/sda2/Dropbox/PhD/Programy/Python/Test/TrDens/'
        elif platform=='win32':
            MolDir='C:/Dropbox/PhD/Programy/Python/Test/TrDens/'
        
        Gaussian_cube=DensityGrid(None,None,None,None)
        Gaussian_cube.import_cub("".join([MolDir,'perylene_exct_wB97XD_LANL2DZ_geom_BLYP_LANL2DZ_sph_coarse_0-1.cub']))
        
        mol=Molecule('Perylene')
        mol.load_Gaussian_fchk("".join([MolDir,'perylene_exct_wB97XD_LANL2DZ_geom_BLYP_LANL2DZ_sph.fchk']))
        mol.load_Gaussian_log("".join([MolDir,'perylene_exct_wB97XD_LANL2DZ_geom_BLYP_LANL2DZ_sph.log']))
        mol.struc.get_grid_cube(Gaussian_cube)
        
        if (np.array([37,75,90],dtype='i8')==mol.struc.grid.get_dim()).all and (mol.struc.grid.delta[:]==0.333333).all:
            if (mol.struc.grid.origin==np.array([-6.000000,-12.488563,-14.922313])).all:
                grid_alloc=True
        if grid_alloc:
            print('Grid alloc cube      ...    OK')
        else:
            print('Grid alloc cube      ...    Error')
        
        mol.get_transition_density(0,AOalloc=True)
        if are_similar(Gaussian_cube.data,mol.exct[0].dens_trans.data):
            transdens=True
        if grid_alloc:
            print('Transition density   ...    OK')
            mol.exct[0].dens_trans.output("".join([MolDir,'TrDens_calc.cub']))
        else:
            print('Transition density   ...    Error')
            mol.exct[0].dens_trans.output("".join([MolDir,'TrDens_calc.cub']))