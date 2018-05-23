# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:12:49 2016

@author: uzivatel
"""

from copy import deepcopy
from scipy.spatial import cKDTree
import scipy
import numpy as np
import networkx as nx
from os import path


from ..read_mine import read_xyz, read_VMD_pdb, read_mol2, read_gaussian_gjf, read_AMBER_prepc
from ..read_mine import read_TrEsp_charges as read_TrEsp
from .general import Coordinate,Grid 
from ...General.UnitsManager import position_units,PositionUnitsManaged,energy_units
from ...General.types import UnitsManaged
from ..positioningTools import RotateAndMove, RotateAndMove_1, CenterMolecule
from ..output import OutputToXYZ, OutputTOmol2, OutputToPDB

nist_file = path.join(path.dirname(path.realpath(__file__)),
                      'supporting_data/Atomic_Weights_NIST.html')
nist_mass = None
nist_indx = None
 
    
class Structure(PositionUnitsManaged):
    ''' Class containing all information about atomic properties (coordinates, 
    atom types, atomic mass...)
        
    Properties
    -----------
    name : string
        Name of the structure.
    nat : integer
        Number of atoms in the molecule
    coor : Coordinate class
        information about position of every atom. Units are coor.units. Default 
        units are ATOMIC UNITS. Coordinates in coor.values (dimension Natoms x 3).
    at_type : list of characters (dimension Natoms)
        Array of atomic types for all atoms of the molecule (for example ['C','N','C','C',...])
    ncharge : numpy array of integer (dimension Natoms)
        Nuclear charges for every atom in the molecule
    mass : numpy.array of real (dimension Natoms)
        Vector with real atomic masses for every atom in the molecule
    bonds : list of integers (dimension Nbonds x 2)
        List with atoms between which there is a bond. self.bonds[i,0]<self.bonds[i,1]
        and self.bonds[i,0], and self.bonds[i,1] are indexes of atoms between 
        which bond is formed.
    bond_type :
        Type of the bond between atoms
    ff_type : numpy.array of string (dimension Natoms)
        Vector with Force Field atom types (for example for GAFF AtType=['ca','c3','ca','ha',...])
    vdw_rad : numpy array of real (dimension Natoms)
        Van der Waals radius for every atom (used for creating cavity in 
        polarizable dielectric model). Radii taken from GAFF forcefield  
    esp_grnd : numpy array of real (dimension Natoms)
        Ground state charge from ESP calculation for every atom
    esp_exct : numpy array of real (dimension Natoms)
        Excited state charge from ESP calculation for every atom
    esp_trans : numpy array of real (dimension Natoms)
        Transition charge from ESP calculation for every atom
    tr_char : numpy array of real (dimension Natoms)
        Atomic transition charges calculated by multipole expansion
    tr_dip : numpy array of real (dimension Natoms x 3)
        Atomic transition dipoles calculated by multipole expansion
    tr_quadr2 : numpy array of real (dimension Natoms)
        Atomic transition quadrupoles calculated by multipole expansion for r^2
        operator
    tr_quadrR2 : numpy array of real (6 x dimension Natoms)
        Atomic transition quadrupoles calculated by multipole expansion for xx,
        xy, xz, yy, yz, zz components of the quadrupole.
    grid : Grid class
        Contains the information about the grid for calculation of 3D 
        properties on a grid
    
    Functions
    -----------
    add_atom : 
        Add coordinate, atomic types, nuclear charges and mass into the 
        structure. 
    add_coor : 
        The same as add_atom but it can handle also arrays (more atoms at once)
    get_com :
        Outputs center of mass in Coordinate class
    move :
        Moves the whole structure along specified vector
    rotate :
        Rotate the structure by specified angles in radians in positive direction
    rotate_1 :
        Inverse totation to rotate
    center :
        Center the molecule and allign in plane
    copy :
        Create 1 to 1 copy of the structure with all classes and types. Or if 
        specified by indexes only copy information about subset of the structure.
    delete_by_indx :
        Delete atoms specified by indexes from the structure.
    guess_bonds :
        Add bonds between atoms which are close together
    get_bonded_atoms :
        
    count_fragments :
        Count how many separate structures are in the structure and outputs
        indexes of individual separate units.
    load_xyz :
        Reads the molecule from xyz file into structure
    load_pdb :
        Reads the molecule from pdb file into structure
    load_mol2 :
        Reads the molecule from mol2 file into structure
    load_prepc :
        Reads the molecule from AMBER prepc file into structure
    load_gjf :
        Reads the molecule from Gaussian input file into structure
    read_TrEsp_charges :
        Reads the molecule from TrEsp charges fiting procedure. If molecule is 
        allready allocated only charges are read
    output_mol2 :
        Output structure into mol2 file needed for AMBER MD simulation
    output_to_xyz :
        Output structure into xyz file
    output_to_pdb :
        Output structure into pdb file
    get_FF_types :
        Assign GAFF forcefield types to the atoms. So far only working for 
        fluorographene systes.
    get_FF_rad:
        Assign vdW GAFF radii to most comon FF types
    get_grid :
        Create eqiuidistant grid around the molecule for calculation of 3D 
        properties on a grid
    get_grid_cube :
        Initialize the grid from cube file 
    add_H :
        Add specified number of hydroens to specified atom
    fragmentize :
        Create list of structures fragments corresponding to separate units in
        the structure
    get_Huckel_molecule :
        Create Huckel representation of the molecule defined in structure 
    get_3D_oscillator_representation :
        Create 3D classical oscillator representation of the structure or part
        of the structure defined by atomic indexes
    '''
    
    vdw_rad = UnitsManaged("vdw_rad")
    
    def __init__(self):
        self.name='Molecule'
        self.coor=None
        self.at_type=None
        self.nat=0
        self.bonds=None
        self.bond_type=None
        self.init=False
        self.ncharge=None
        self.mass=None
        self.vdw_rad=None
        self.ff_type=None
        self.esp_grnd=None
        self.esp_exct=None
        self.esp_trans=None
        self.tr_char=None
        self.tr_dip=None
        self.tr_quadr2=None
        self.tr_quadrR2=None
        self.grid=None
# TODO: at_type should be list and not npumpy array
# Proton number information is in self.ncharge
        
    def add_coor(self,coor,at_type,ncharge=None,mass=None):
        if len(at_type)!=len(coor):
            raise Warning('For every atom coordinate there has to be atom type')
        
        if not self.init:
            self.coor=Coordinate(coor)
            if (at_type is None) and (ncharge is not None):
                self.at_type=[]
                for ii in range(len(ncharge)):
                    self.at_type.append(get_atom_symbol(ncharge[ii]))
                at_type=self.at_type.copy()
            else:
                self.at_type=at_type.copy()
            self.init=True
            self.nat=len(at_type)
            if ncharge is not None:
                np_ncharge=np.zeros(len(ncharge),dtype='i4')
                for ii in range(len(ncharge)):
                    np_ncharge[ii]=int(float(ncharge[ii]))
                self.ncharge=np_ncharge
            elif at_type is not None:
                self.ncharge=np.zeros(self.nat,dtype='i4')
                for ii in range(self.nat):
                    self.ncharge[ii]=get_atom_indx(at_type[ii])
            if mass is not None:
                self.mass=np.array(mass,dtype='f8')
            elif at_type is not None:
                self.mass=np.zeros(self.nat,dtype='f8')
                for ii in range(self.nat):
                    self.mass[ii]=get_mass(at_type[ii])
        else:
            for ii in range(len(at_type)):
                self.coor.add_coor(coor[ii])
                self.at_type.append(at_type[ii])
                self.nat+=1
            if ncharge is not None:
                self.ncharge=np.append(self.ncharge,ncharge)
            else:
                ncharge=[]
                for ii in range(len(at_type)):
                    ncharge.append(get_atom_indx(at_type[ii]))
                self.ncharge=np.append(self.ncharge,ncharge)
            if mass is not None:
                self.mass=np.append(self.mass,mass)
            else:
                mass=[]
                for ii in range(len(at_type)):
                    mass.append(get_mass(at_type[ii]))
                self.mass=np.append(self.mass,mass)
            self.bonds=None
                
    def add_atom(self,coor,at_type,ncharge=None,mass=None):
        if self.ncharge is not None:
#            if len(self.ncharge)==self.nat and ncharge is None:
#                raise Warning('For every atom there are defined nuclear charges. In odred to be consistent you have to specify them for new atoms')
#                return
            if len(self.ncharge)!=self.nat: 
                raise Warning('There are not specified nuclear charges for all atoms. Set these charges before you add new atoms')
                return
#        if self.mass is not None:
#            if len(self.mass)==self.nat and mass is None:
#                raise Warning('For every atom there is defined nuclear mass. In odred to be consistent you have to specify it for new atoms')
#                return
        if mass is not None:
            if len(self.mass)!=self.nat: 
                raise Warning('There is not specified nuclear mass for all atoms. Set the mass for all atoms before you add new ones')
                return
        
        if not self.init:
            self.coor=Coordinate(coor)
            self.at_type=at_type.copy()
            self.nat=len(at_type)
        if coor.__class__.__name__=='Coordinate':
            self.coor.add_coor(coor.value)
        else:
            self.coor.add_coor(coor)
        self.at_type.append(at_type)
        self.nat+=1
        self.init=True
        if mass is not None:
            self.mass=np.append(self.mass,mass)
        else:
            mass=get_mass(at_type)
            self.mass=np.append(self.mass,mass)
        if ncharge is not None:
            self.ncharge=np.append(self.ncharge,ncharge)
        else:
            ncharge=get_atom_indx(at_type)
            self.ncharge=np.append(self.ncharge,ncharge)
        self.bonds=None

    def get_com(self):
        """ Outputs center of mass in as coordinate class
        
        Returns
        --------
        Rcom : Coordinate class
            Center of mass of the whole structure
        """
        mass_w_coor=np.multiply(np.vstack((self.mass,self.mass,self.mass)).T,self.coor.value)
        Rcom=Coordinate(np.sum(mass_w_coor,axis=0)/np.sum(self.mass))
        return Rcom
    
    def move(self,dx,dy,dz):
        """ Moves the structure along the specified vector
        
        Parameters
        --------
        dx,dy,dz : float
            Displacements along x, y, and z coordinate
        """
        self.coor.move(dx,dy,dz)
    
    def rotate(self,rotxy,rotxz,rotyz):
        """"
        Rotate the structure around the coordinate origin by specified angles.
        first it rotate the structure around z axis (in xy plane), then around
        y axis and in the end around x axis in positive direction 
        (if right thumb pointing in direction of axis fingers are pointing in 
        positive rotation direction)
        
        Parameters
        --------
        rotxy,rotxz,rotxz : float
            Rotation angles in radians around z, y and x axis (in xy, xz and yz plane)
        
        """
        
        self.coor.rotate(rotxy,rotxz,rotyz)
        if self.tr_dip is not None:
            self.tr_dip=RotateAndMove(self.tr_dip,0.0,0.0,0.0,rotxy,rotxz,rotyz)
        if (self.tr_quadr2 is not None) or (self.tr_quadrR2 is not None):
            print('Rotation of quadrupoles is not implemented yet. Dealocation of all atomic quadrupoles')
            self.tr_quadr2=None
            self.tr_quadrR2=None
# TODO: Rotate grid
    
    def rotate_1(self,rotxy,rotxz,rotyz):
        """" Inverse rotation of structure to **rotate** fuction.
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
        if self.tr_dip is not None:
            self.tr_dip=RotateAndMove_1(self.tr_dip,0.0,0.0,0.0,rotxy,rotxz,rotyz)
        if (self.tr_quadr2 is not None) or (self.tr_quadrR2 is not None):
            print('Rotation of quadrupoles is not implemented yet. Dealocation of all atomic quadrupoles')
            self.tr_quadr2=None
            self.tr_quadrR2=None
# TODO: Rotate grid
            
    def center(self,indx_center,indx_x,indx_y,debug=False,**kwargs):
        """
        Centers the structure according to defined center and two main axis.
        Axis and center are defined by atomic indexes and after centering center
        would be in coordinate system origin, vector from origin to atom indx_x
        will be aligned along x-axis and vector from origin to indx_y will be
        in xy plane (z component of the vector will be zero)
        
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
        
        New_coor,Phi,Psi,Chi,center=CenterMolecule(self.coor.value,indx_center,indx_x,indx_y,print_angles=True,debug=debug,**kwargs)
        self.move(-center[0],-center[1],-center[2])
        self.rotate(Phi,Psi,Chi)
        
        return Phi,Psi,Chi,center
        
    def guess_bonds(self,bond_length=None):
        ''' Function guesses pairs of atoms between which bond might occure.
        
        Parameters
        ----------
        bond_length : float (optional)
            Larges distance between to atoms between which bond is formed

        Notes
        ---------
        Bonds will be outputed to: \n
        **self.bonds** \n 
        as list of integers (dimension Nbonds x 2) where self.bonds[i,0]<self.bonds[i,1]
        and self.bonds[i,0], and self.bonds[i,1] are indexes of atoms between 
        which bond is formed.
        '''    
        
        if bond_length is None:
            bond_length=4.0
        else:
            bond_length = self.coor.manager.convert_position_2_internal_u(bond_length)

        if len(self.at_type)!=0:
            is_AtType=True
        else:
            is_AtType=False

        # Prepare
        if is_AtType:
            indxH = np.argwhere(self.at_type=='H').T[0]
            nH = len(indxH)
        test = cKDTree(self.coor._value)
        Bonds = test.query_pairs(bond_length,output_type='ndarray')
        # Rapair bonds between two hydrogens
        if is_AtType:
            if nH>1:
                list_to_delete = []
                for ii in range(len(Bonds)):
                    if self.at_type[Bonds[ii,0]] == 'H':
                        if self.at_type[Bonds[ii,1]] == 'H':
                            list_to_delete.append(ii)
        
                for ii in reversed(list_to_delete):
                    Bonds = np.delete(Bonds,ii,axis=0)
        
        self.bonds = Bonds
    
    def get_bonded_atoms(self):
        """ Finds connected atoms by chemical bond
        
        """
        if self.bonds is None:
            self.guess_bonds()
        
        connected = []
        for ii in range(self.nat):
            connected.append([])
        
        for ii in range(len(self.bonds)):
            atom1 = self.bonds[ii][0]
            atom2 = self.bonds[ii][1]
            
            connected[atom1].append(atom2)
            connected[atom2].append(atom1)
        return connected
        
    def count_fragments(self,verbose=False):
        ''' Divide the structure into individual units between which there is
        no bond.
        
        Returns
        --------
        Molecules : list of integers
            In the list there are atomic indexes corresponding to separate units.
            ``len(Molecules)`` is number of separate units. ``Molecules[i]`` is
            list of atomic indexes in i-th unit.
        '''
        
        if self.bonds is None:
            self.guess_bonds()
        
        Nat=self.nat
        g = nx.Graph()  # g is graph
        Nbonds=len(self.bonds)
        g.name = "Molecular complex or supermolecule" # name of the graph
        for ii in range(Nat):
            g.add_node(ii)
        for ii in range(Nbonds):
            g.add_edge(self.bonds[ii][0],self.bonds[ii][1])
        
        Molecules=[]
        # Calculate separate subgraphs
        #count fragmnets
        Nfrag=0
        for cc in nx.connected_component_subgraphs(g):
            Nfrag+=1
        if Nfrag==1:
            if verbose:
                print('There is only single molecule in',self.name)
                print(' ')
            Molecules=[np.arange(self.nat)]
        else:
            for cc in nx.connected_component_subgraphs(g):
                if verbose:
                    print('Fragment',cc, "has", len(cc.nodes()),"atoms.")
                    print('      atom indexes (starting from zero):',cc.nodes())
                Molecules.append(cc.nodes())
            if verbose:
                print(' ')
        return Molecules
    
    
    def load_xyz(self,filename):
        """ Loads all structure information from xyz file. If the structure is 
        allready initialized it adds the molecule to the structure instead of 
        rewriting the existing information.
        
        Parameters
        -----------
        filename : string
            Name of the input file including the path if needed
        """
        
        Geom,At_type=read_xyz(filename,verbose=False)
        with position_units('Angstrom'):
            if not self.init:
                self.coor=Coordinate(Geom)
                self.nat=len(At_type)
                self.at_type=[]
                for jj in range(self.nat):
                    self.at_type.append(''.join([i for i in At_type[jj] if not i.isdigit()]))
                    self.at_type[jj]=self.at_type[jj].capitalize()
                #self.at_type=At_type.copy()
                self.ncharge=np.zeros(self.nat,dtype='i4')
                for ii in range(self.nat):
                    self.ncharge[ii]=get_atom_indx(self.at_type[ii])
                self.mass=np.zeros(self.nat,dtype='f8')
                for ii in range(self.nat):
                    self.mass[ii]=get_mass(self.at_type[ii])
                self.init=True
            else:
                at_type = []
                for jj in range(len(At_type)):
                    at_type.append(''.join([i for i in At_type[jj] if not i.isdigit()]))
                    at_type[jj]=at_type[jj].capitalize()
                self.add_coor(np.array(Geom),at_type)
                
    
    def load_pdb(self,filename):
        """ Loads all structure information from pdb file. If the structure is 
        allready initialized it adds the molecule to the structure instead of 
        rewriting the existing information.
        
        Parameters
        -----------
        filename : string
            Name of the input file including the path if needed
        """
        
        MD=read_VMD_pdb(filename)
        at_type=[]
        for jj in range(MD.NAtom):
            at_type.append(''.join([i for i in MD.at_name[jj] if not i.isdigit()]))
            at_type[jj]=at_type[jj].capitalize()         
        coor=MD.geom[:,:,0]
        with position_units('Angstrom'):  
            if not self.init:
                self.coor=Coordinate(coor)
                self.at_type=at_type.copy()
                self.nat=len(self.at_type)
                self.init=True
                self.ncharge=np.zeros(self.nat,dtype='i4')
                for ii in range(self.nat):
                    self.ncharge[ii]=get_atom_indx(at_type[ii])
                self.mass=np.zeros(self.nat,dtype='f8')
                for ii in range(self.nat):
                    self.mass[ii]=get_mass(at_type[ii])
            else:
                self.add_coor(coor,at_type)

# TODO: Add possibility to add the information to allready initialized structure
    def load_mol2(self,filename,state='Ground'):
        """ Loads all structure information from mol2 file. In present stage
        it can be called only from not initialized structure.
        
        Parameters
        -----------
        filename : string
            Name of the input file including the path if needed
        state : string (optional init = 'Ground')
            Which charges are present in mol2 file. If ``state='Ground'`` it is
            assumed that charges in mol2 file correspond to ground state 
            (default), therefore they are loaded in ``self.esp_grnd``. 
            If ``state='Excited'`` it is assumed that charges in mol2 file 
            correspond to excited state, therefore they are loaded in
            ``self.esp_exct``. If ``state='Transition'`` it is assumed that 
            charges in mol2 file are transition charges, therefore they are 
            loaded in ``self.esp_trans``.
        
        Returns
        ---------
        Name : string
            3 character name of the molecule (system) in mol2 file
        Charge_method : string
            Name of the method which was used for generation of mol2 charges
            for example (esp - for esp charges, bcc, gas, ...)
        Aditional_info : string
            Some aditional information about the molecule (usualy not important
            text)
        """
        
        Coor,Bond,Charge,AtName,AtType,Molecule,MolNameAt,AditionalInfo=read_mol2(filename)
        Name=AditionalInfo[0]
        Nat=AditionalInfo[1]
        with position_units('Angstrom'): 
            if not self.init:
                self.nat=int(Nat)      
                self.coor=Coordinate(Coor)
                self.bonds=Bond[:,1:3]
                self.bond_type=Bond[:,3]
                self.ff_type=np.copy(AtType)
                if state=='Ground':
                    self.esp_grnd=np.array(Charge,dtype='f8')
                elif state=='Excited':
                    self.esp_exct=np.array(Charge,dtype='f8')
                elif state=='Transition':
                    self.esp_trans=np.array(Charge,dtype='f8')
                # or self.mol_spec['Grnd_ESP_charges']=np.copy(Charge)
                
                at_type=[]
                for jj in range(Nat):
                    at_type.append(''.join([i for i in AtName[jj] if not i.isdigit()]))
                    at_type[jj]=at_type[jj].capitalize()
                self.at_type=at_type.copy()
                
                self.ncharge=np.zeros(self.nat,dtype='i4')
                for ii in range(self.nat):
                    self.ncharge[ii]=get_atom_indx(self.at_type[ii])
                self.mass=np.zeros(self.nat,dtype='f8')
                for ii in range(self.nat):
                    self.mass[ii]=get_mass(self.at_type[ii])
                self.init=True
            else:
                raise Warning('Adding the information from mol2 file into existing structure is not yet supported')
# TODO: Add option to read information from mol2 file into existing structure
            
        Charge_method=AditionalInfo[4]
        Aditional_info=AditionalInfo[5]

        return Name,Charge_method,Aditional_info
    
    
    def load_prepc(self,filename,state='Ground'):
        """ Loads all structure information from AMBER preps file. In present stage
        it can be called only from not initialized structure.
        
        Parameters
        -----------
        filename : string
            Name of the input file including the path if needed
        state : string (optional init = 'Ground')
            Which charges are present in prepc file. If ``state='Ground'`` it is
            assumed that charges in prepc file correspond to ground state 
            (default), therefore they are loaded in ``self.esp_grnd``. 
            If ``state='Excited'`` it is assumed that charges in prepc file 
            correspond to excited state, therefore they are loaded in
            ``self.esp_exct``. If ``state='Transition'`` it is assumed that 
            charges in prepc file are transition charges, therefore they are 
            loaded in ``self.esp_trans``.
        
        Returns
        ---------
        indxlist of integers specifying position of every atom in original file
        from which prepc file was generated. For every atom type INDX is from 1
        to number of atoms of that type.

        """
        
        Coor,Charge,AtType,FFType,MolName,indx_orig = read_AMBER_prepc(filename)
        with position_units('Angstrom'): 
            if not self.init:
                self.nat=len(AtType)      
                self.coor=Coordinate(Coor)                
                self.ff_type=np.copy(FFType)
                if state=='Ground':
                    self.esp_grnd=np.array(Charge,dtype='f8')
                elif state=='Excited':
                    self.esp_exct=np.array(Charge,dtype='f8')
                elif state=='Transition':
                    self.esp_trans=np.array(Charge,dtype='f8')
                at_type=[]
                for jj in range(self.nat):
                    at_type.append(''.join([i for i in AtType[jj] if not i.isdigit()]))
                    at_type[jj]=at_type[jj].capitalize()
                self.at_type=at_type.copy()
                
                self.ncharge=np.zeros(self.nat,dtype='i4')
                for ii in range(self.nat):
                    self.ncharge[ii]=get_atom_indx(self.at_type[ii])
                self.mass=np.zeros(self.nat,dtype='f8')
                for ii in range(self.nat):
                    self.mass[ii]=get_mass(self.at_type[ii])
                self.init=True
                self.name = MolName
            else:
                raise Warning('Adding the information from mol2 file into existing structure is not yet supported')
            
        return indx_orig
    
    def load_gjf(self,filename):
        """ Loads all structure information from Gaussian gjf input file.
        If the structure is allready initialized it adds the molecule to the
        structure instead of rewriting the existing information.
        
        Parameters
        -----------
        filename : string
            Name of the input file including the path if needed
        """
        
        Geom,at_type=read_gaussian_gjf(filename,verbose=False)
#        at_type=[]
#        for ii in charge:
#            at_type.append(get_atom_symbol(ii))
        with position_units('Angstrom'):
            if not self.init:
                self.coor=Coordinate(Geom)
                self.at_type=at_type.copy()
                self.nat=len(self.at_type)
                self.ncharge=np.zeros(self.nat,dtype='i4')
                for ii in range(self.nat):
                    self.ncharge[ii]=get_atom_indx(at_type[ii])
                self.mass=np.zeros(self.nat,dtype='f8')
                for ii in range(self.nat):
                    self.mass[ii]=get_mass(at_type[ii])
                self.init=True
            else:
                self.add_coor(np.array(Geom),at_type)


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
            fitted charegs are loaded into: ``self.esp_grnd``. If 
            ``state='Excited'`` the imput file was generated from excited state
            ESP fitting, therefore fitted charegs are loaded into: 
            ``self.esp_exct``. If ``state='Transition'`` the imput file was 
            generated from transition ESP fitting, therefore fitted charegs are
            loaded into: ``self.esp_trans``.
        verbose : logical (optional init = False)
            Controls if information about exact and ESP fitted dipole from atomic
            charges is printed.
        
        Returns
        ---------
        DipoleTrESP : numpy array of float (dimension 3)
            Dipole calculated from atomic ESP charges in ATOMIC UNITS (Bohr*e).
            (for ``state='Transition'`` it will corespnd to transition dipole)
        
        Notes
        ---------
        Reason for not checking the structure is that sometimes it is neaded to
        read the charges into distorted system geometry.
      
        '''

        coor,charge,at_type,DipoleTrESP,DipoleExact=read_TrEsp(filename,dipole=True)

        with position_units('Angstrom'):
            if not self.init:
                # read geometry and charges
                self.coor=Coordinate(coor)
                self.at_type=at_type
                self.nat=len(at_type)
                if state=='Ground':
                    self.esp_grnd=charge.copy()
                elif state=='Excited':
                    self.esp_exct=charge.copy()
                elif state=='Transition':
                    self.esp_trans=charge.copy()
                
                self.ncharge=np.zeros(self.nat,dtype='i4')
                for ii in range(self.nat):
                    self.ncharge[ii]=get_atom_indx(self.at_type[ii])
                self.mass=np.zeros(self.nat,dtype='f8')
                for ii in range(self.nat):
                    self.mass[ii]=get_mass(self.at_type[ii])
                self.init=True
            
            else:
                # read only charges and suppose correct ordering of the atoms
                # fast test if structures are the same
                if self.nat!=len(at_type):
                    raise IOError('TrEsp molecule has diffent number of atoms than defined structure')
                if not self.at_type==at_type:
                    raise IOError('Atomic types from structure and from imput file do not match') 
                if state=='Ground':
                    self.esp_grnd=charge.copy()
                elif state=='Excited':
                    self.esp_exct=charge.copy()
                elif state=='Transition':
                    self.esp_trans=charge.copy()
            
        if verbose:
            print('Dipole from ESP charges:',DipoleTrESP,'with size:',np.sqrt(np.dot(DipoleTrESP,DipoleTrESP)),'AU')
            print('Dipole exact:',DipoleExact,'with size:',np.sqrt(np.dot(DipoleExact,DipoleExact)),'AU')
        
        return DipoleTrESP


    def output_mol2(self,filename,state='Ground',ch_method=None,Info=None,Name='XXX'):
        """ Create mol2 file for the structure. This file can be then used for
        AMBER MD simulations
        
        Parameters
        -----------
        filename : string
            Name of the output file including the path if needed (including the 
            .mol2 suffix)
        state : string (optional init = 'Ground')
            Which charges are used for mo2 file generation. If ``state='Ground'``
            -> ``self.esp_grnd``, if ``state='Excited'`` -> ``self.esp_exct`` and
            if ``state='Transition'`` -> ``self.esp_trans``
        ch_method : string (optional)
            Name of the method which was used for generation charges
            for example (``'resp'``, ``'bcc'``, ``'gas'``, ...). If nothing 
            specified default charge method name ``'resp'`` will be used.
        Info : string (optional)
            Some aditional information wich will be added to the end of mol2
            file. Doesn't influent the calculation.
        Name : 3 character string (optional init = 'XXX')
            3 upper case character name of the system

        """
        
        if state=='Ground':
            charge=self.esp_grnd
        elif state=='Excited':
            charge=self.esp_exct
        elif state=='Transition':
            charge=self.esp_trans
        
        if self.bonds is None:
            self.guess_bonds()

        if self.ff_type is None:        
            self.get_FF_types()
            
        with position_units('Angstrom'):
            if (ch_method is not None) and (Info is not None):
                OutputTOmol2(self.coor.value,self.bonds,charge,self.at_type,self.ff_type,Name,filename,ChargeMethod=ch_method,AditionalInfo=Info)
            elif ch_method is not None:
                OutputTOmol2(self.coor.value,self.bonds,charge,self.at_type,self.ff_type,Name,filename,ChargeMethod=ch_method)
            elif Info is not None:
                OutputTOmol2(self.coor.value,self.bonds,charge,self.at_type,self.ff_type,Name,filename,AditionalInfo=Info)
            else:
                OutputTOmol2(self.coor.value,self.bonds,charge,self.at_type,self.ff_type,Name,filename)
        
    def output_to_xyz(self,filename='Molecule.xyz'):
        """ Create xyz file for the structure.
        
        Parameters
        -----------
        filename : string (optional init = 'Molecule.xyz')
            Name of the output file including the path if needed (including the 
            .xyz suffix)
        """
        
        with position_units('Angstrom'):
            OutputToXYZ(self.coor.value,self.at_type,filename)
    
    def output_to_pdb(self,filename='Molecule.pdb'):
        """ Create pdb file for the structure with unique atom names.
        
        Parameters
        -----------
        filename : string (optional init = 'Molecule.xyz')
            Name of the output file including the path if needed (including the 
            .xyz suffix)
        """
        
        with position_units('Angstrom'):
            OutputToPDB(self.coor.value,self.at_type,filename=filename)
        
    
    def get_FF_types(self):
        ''' 
        Assign forcefield types and according vdW radii to all atoms
        So far this is only working for cyclic hydrocarbons - carbons (ca - GAFF type)
        and hydrogens (ha - GAFF type) 
        
        ca VdW radius = 1.9080
        ha VdW radius = 1.4590
        
        Notes
        -------
        Forcefield types are outputed to:\n
        **self.ff_type** \n
        as list of charactes and vdW radii in ANGSTROMS are assign to these 
        types as well into:\n
        **self.vdw_rad** \n
        as numpy array of float 
        '''

        self.ff_type = []
        Vdw_rad = np.zeros(self.nat)
        
        for ii in range(self.nat):
            if self.at_type[ii]=='H':
                self.ff_type.append('ha')
                Vdw_rad[ii]=VdW_radius['ha']
            elif self.at_type[ii]=='C':
                self.ff_type.append('ca')
                Vdw_rad[ii]=VdW_radius['ca']
            elif self.at_type[ii]=='F':
                self.ff_type.append('f')
                Vdw_rad[ii]=VdW_radius['f']
            else: 
                print('Unsupported atom type')
        with position_units('Angstrom'):
            self.vdw_rad=Vdw_rad
                
    def get_FF_rad(self):
        ''' 
        Assign GAFF forcefield vdW radii to most standard atoms (forcefield types)
        
        Notes
        -------
        VdW radii are outputed in ANGSTROMS to:\n
        **self.vdw_rad** \n
        as numpy array of float
        
        '''
        
        if self.ff_type is None:
            self.get_FF_types()
        else:
            Vdw_rad = np.zeros(self.nat)
            for ii in range(self.nat):
                Vdw_rad[ii]=VdW_radius[self.ff_type[ii]]
        
        with position_units('Angstrom'):
            self.vdw_rad=Vdw_rad
        

    def get_grid(self,extend=5.0,step=0.4):
        ''' Generate the equidistant grid around the molecule
        
        Parameters
        ----------
        extend : float (optional init = 5.0)
            Extension of the grid in every dimension from farthermost atoms
        step : float (optional init = 0.4)
            Spacing between individual grid points
        
        Returns
        ----------
        grid : Grid class
            Contains the information about the grid.
            
        Notes
        ----------
        Grid information is also stored as grid class in:\n
        **self.grid** \n
        Grid is position units managed, therefore ``extend`` and ``step`` can
        be set in any supported position units.
        
        '''
        
        if self.grid is None:
            self.grid=Grid()
            self.grid.set_to_geom(self.coor.value,self.coor.units,extend=extend,step=step)
        
        return self.grid
    
    def get_grid_cube(self,density):
        ''' Generate grid from denisty class (from cube file). The resulting
        grid will be exactly the same as grid used for generation of cube density
        
        Parameters
        ----------
        density : Density class
            Contains the informations about the grid which will be replicated
        
        Returns
        ----------
        grid : Grid class
            Contains the information about the grid.
            
        Notes
        ----------
        Grid information is also stored as grid class in:\n
        **self.grid** 
        
        '''
        self.grid=Grid()
        self.grid.init_from_cub(density)
        
        return self.grid

    def copy(self,indx=None):
        ''' Create deep copy of the structure. If no indexes are defined it 
        creates 1-1 copy of whole structure. If indexis are present it copy
        all the information relevat only to the selected atoms (resulting 
        structure would contain only atoms defined by indexes)
        
        Parameters
        ----------
        indx : list of integers (optional)
            If specified only atoms corresponding to the specified indexes will
            be copied into new structure.
        
        Returns
        ----------
        new_struc : Structure class
            
        '''
        
        new_struc=deepcopy(self)
        if indx is None:
            return new_struc
        else:
            new_struc=Structure()
            new_struc.name="".join([self.name,' copy'])
            if self.coor is not None:
                coor=np.zeros((len(indx),3),dtype='f8')
                coor_all=self.coor.value
                for ii in range(len(indx)):
                    coor[ii,:]=coor_all[indx[ii]]
                new_struc.coor=Coordinate(coor)
            if self.at_type is not None:
                at_type=[]
                for ii in indx:
                    at_type.append(self.at_type[ii])
                new_struc.at_type=at_type
            new_struc.nat=len(indx)
            new_struc.init=True     
            if self.ncharge is not None:
                new_struc.ncharge=np.zeros(len(indx),dtype='i4')
                for ii in range(len(indx)):
                    new_struc.ncharge[ii]=self.ncharge[indx[ii]]
            if self.mass is not None:
                new_struc.mass=np.zeros(len(indx),dtype='f8')
                for ii in range(len(indx)):
                    new_struc.mass[ii]=self.mass[indx[ii]]
            if self.vdw_rad is not None:
                vdw_rad_new=np.zeros(len(indx),dtype='f8')
                vdw_rad_all=self.vdw_rad
                for ii in range(len(indx)):
                    vdw_rad_new[ii]=vdw_rad_all[indx[ii]]
                new_struc.vdw_rad=vdw_rad_new
            if self.ff_type is not None:
                new_struc.ff_type=[]
                for ii in indx:
                    new_struc.ff_type.append(self.ff_type[ii])
                    
            return new_struc
    
        return new_struc
    
    def delete_by_indx(self,indx):
        ''' Create a new structure by deleting atoms from the original one.
        
        Parameters
        ----------
        indx : list of integers
            List of atomic indexes specifiing which atioms will be deleted (wont
            be present in new structure)
        
        Returns
        ----------
        new_struc : Structure class
            Structure without deleted atoms.
        
        Notes
        ----------
        By deleting the atoms we also loose some information. For example if 
        charges for all atoms were present then these charges do not correspond
        to the structure without specified atoms
            
        '''
        if type(indx)!=list:
            raise IOError('Only list of indexes is supported in delete_by_indx')
        new_struc=deepcopy(self)
        indx_sorted=sorted(indx)
        mask=np.ones(new_struc.nat,dtype='bool')
        mask[indx_sorted] = False
        if new_struc.ncharge is not None:
            new_struc.ncharge = new_struc.ncharge[mask]
        if new_struc.mass is not None:
            new_struc.mass = new_struc.mass[mask]
        for ii in reversed(indx_sorted):
            if new_struc.at_type is not None:
                del(new_struc.at_type[ii])
            if new_struc.coor is not None:
                new_struc.coor.del_coor(ii)
#            if new_struc.ncharge is not None:
#                print(ii,new_struc.ncharge.shape,new_struc.ncharge[ii],new_struc.ncharge.__class__)
#                np.delete(new_struc.ncharge,ii)
#                print(ii,new_struc.ncharge.shape,new_struc.ncharge[ii],new_struc.ncharge.__class__)
#            if self.mass is not None:
#                np.delete(new_struc.mass,ii)
            if self.vdw_rad is not None:
                del(new_struc._vdw_rad[ii])
            if self.ff_type is not None:
                del(new_struc.ff_type[ii])
        
        new_struc.name=''.join([self.name,' cut'])
        new_struc.nat-=len(indx)
        new_struc.bonds=None
        new_struc.bond_type=None
        new_struc.esp_grnd=None
        new_struc.esp_exct=None
        new_struc.esp_trans=None
        new_struc.tr_char=None
        new_struc.tr_dip=None
        new_struc.tr_quadr2=None
        new_struc.tr_quadrR2=None
        new_struc.grid=None
        
        return new_struc   
    
    def fragmentize(self,force=False):
        """ Cut the structure into individual units between which there is no 
        bond.
        
        Parameters
        ----------
        force : logical (optional init = False)
            If true new bonds will be calculated even if some are allready 
            present.
        
        Returns
        ----------
        Fragments : list of Structure type
            List with individual units in structure class
        
        """
        
        Fragments=[]
        if self.bonds is None or force:
            self.guess_bonds()
        Molecules=self.count_fragments()
        Nfrag=len(Molecules)
        indx_all=list(np.arange(self.nat))
        for ii in range(Nfrag):
            indx=sorted(list(set(indx_all) - set(Molecules[ii])))
            frag=deepcopy(self)
            frag=frag.delete_by_indx(indx)
            Fragments.append(frag)
        return Fragments

    
    def add_H(self,indx1,indx2,indx3,nH=3,Hdist=2.0031,At_type='H'):
        """ Add hydrogens to specified atom of the structure
        
        Parameters
        ----------
        indx1,indx2,indx3 : integer
            indx1 specifies the atom index to which hydrogens will be added.
            indx1-indx2 specifies the rotational axis for added hydrogens and 
            indx1-indx2-index3 specifies the initial plane for adding hydrogens
        nH : integer (optional init = 3)
            Number of added hydrogen in atom with index ``indx1``
        Hdist : float (optional init = 3)
            Distance H-indx1 atom in Bohr
        At_type : string (optional init = 'H')
            Atom type which is added if we would like to add different atom then
            hydrogen. **IF OTHER ATOM IS USED - CORRECT CHARGE AND MASS IN 
            STRUCTURE**
       
        Notes
        ----------
        New atom will be added to the existing structure
        
        """
        
        with position_units('Bohr'):
            if nH==1:
                vecH=Coordinate(self.coor._value[indx1]-self.coor._value[indx2])
                vecH.normalize()
                vecH.value=vecH.value*Hdist
                vecH.value+=self.coor._value[indx1]
                if self.ncharge is not None and self.mass is not None:
                    self.add_atom(vecH,At_type,ncharge=1,mass=1.00782504)
                elif self.ncharge is not None:
                    self.add_atom(vecH,At_type,ncharge=1,mass=None)
                elif self.mass is not None:
                    self.add_atom(vecH,At_type,ncharge=None,mass=1.00782504)
                else:
                    self.add_atom(vecH,At_type,ncharge=None,mass=None)
            if nH==2:
                PhiH=120.0  #108.0
                Vec=np.zeros((2,3),dtype='f8')
                coor_cent,Phi,Psi,Chi,center=CenterMolecule(self.coor._value,indx2,indx1,indx3,print_angles=True)
                Vec[0,:]=[np.cos(np.deg2rad(180.0-PhiH)),np.sin(np.deg2rad(180.0-PhiH)),0.0]
                Vec[1,:]=[np.cos(np.deg2rad(180.0-PhiH)),-np.sin(np.deg2rad(180.0-PhiH)),0.0]
                Vec=Vec*Hdist
                VecH=RotateAndMove_1(Vec,0.0,0.0,0.0,Phi,Psi,Chi)
                for ii in range(nH):
                    VecH[ii,:]=VecH[ii,:]+self.coor._value[indx1]
                if self.ncharge is not None and self.mass is not None:
                    self.add_coor(VecH,[At_type]*nH,ncharge=[1]*nH,mass=[1.00782504]*nH)
                elif self.ncharge is not None:
                    self.add_coor(VecH,[At_type]*nH,ncharge=[1]*nH,mass=None)
                elif self.mass is not None:
                    self.add_coor(VecH,[At_type]*nH,ncharge=None,mass=[1.00782504]*nH)
                else:
                    self.add_coor(VecH,[At_type]*nH,ncharge=None,mass=None)
            if nH==3:
                PhiH=108.0
                Vec=np.zeros((3,3),dtype='f8')
                coor_cent,Phi,Psi,Chi,center=CenterMolecule(self.coor._value,indx2,indx1,indx3,print_angles=True)
                Vec[0,:]=[np.cos(np.deg2rad(180.0-PhiH)),0.0,np.sin(np.deg2rad(180.0-PhiH))]
                Vec[1,:]=RotateAndMove(Vec[0,:],0.0,0.0,0.0,0.0,0.0,np.deg2rad(120.0))
                Vec[2,:]=RotateAndMove(Vec[0,:],0.0,0.0,0.0,0.0,0.0,-np.deg2rad(120.0))
                Vec=Vec*Hdist
                VecH=RotateAndMove_1(Vec,0.0,0.0,0.0,Phi,Psi,Chi)
                for ii in range(nH):
                    VecH[ii,:]=VecH[ii,:]+self.coor._value[indx1]
                if self.ncharge is not None and self.mass is not None:
                    self.add_coor(VecH,[At_type]*nH,ncharge=[1]*nH,mass=[1.00782504]*nH)
                elif self.ncharge is not None:
                    self.add_coor(VecH,[At_type]*nH,ncharge=[1]*nH,mass=None)
                elif self.mass is not None:
                    self.add_coor(VecH,[At_type]*nH,ncharge=None,mass=[1.00782504]*nH)
                else:
                    self.add_coor(VecH,[At_type]*nH,ncharge=None,mass=None)
    

    def get_Huckel_molecule(self,nvec,At_list=None,Type='Gaussian_STO3G',overlap_scaled=True,add_excit=True,**kwargs):
        """ Create Huckel representation of the structure
        
        Parameters
        ----------
        nvec : numpy array or list (dimension 3)
            Normal vector to the plane of pi-conjugated system
        At_list : list of integers (optional)
            List of indexes atoms forming pi-conjugated system. If not present
            all atom in the structure will be used
        Type : string (optional init = 'Gaussian_STO3G')
            Specifies which expansion of p-orbitals into cartesian gaussian
            atomic orbitals is used. So far only supported type is STO3G basis.
        overlap_scaled : logical (optional init = True)
            If ``overlap_scaled=True`` interaction between atoms is scaled by 
            overlap of corresponding atomic orbitals. Otherwise the same 
            interaction energy will be used for all atoms of same type
        add_excit : logical (optional init = True)
            If ``add_excit=True`` HOMO-LUMO transition will be added as first 
            excited state.
        **kwargs : dict (optional)
            Specifies if other thendefault atomic energies or interaction 
            energies should be used. For changing default atomic energies
            ``kwargs={'Energy': dict}`` where dict is dictionary with atomic
            types and corresponding energies (e.g. ``dict={'C': -0.4333,
            'N': -0.5677, ...}). For changing the interaction energies 
            ``kwargs{'Interaction': dict}`` where dict is dictionary with pair 
            of atomic types and corresponding interaction energy (e.g. 
            ``dict={'CC': 1.864456, 'CN': 1.9, ...}). Both can be used 
            simultaneously. Units for the energies should be Hartree (ATOMIC
            UNITS)
        
        Returns
        ---------
        Huckel_mol : Molecule class
            Huckel molecule representation with atomic and molecular orbitals,
            structure, excitations ...
            
        Notes
        ---------
        Default atomic energies are:\n
        ``Energy_p={'C': -0.4333, 'N': -0.5677, 'O': -0.6319}`` \n
        Default interaction energy is same for all atoms and it is -1.864456.
        All of these values are in Hartrees (ATOMIC UNITS)
        
        """
        
        from .molecule import Molecule
        from .excitation import Excitation
        from .general import Dipole
        
        if Type=='Gaussian_STO3G':
            Exps_p={'C': [2.9412494, 0.6834831, 0.2222899],
                    'N': [3.7804559, 0.8784966, 0.2857144],
                    'O': [5.0331513, 1.1695961, 0.3803890]}          # in AU
            Coeff_p={'C': [0.15591627, 0.60768372, 0.39195739],
                     'N': [0.15591627, 0.60768372, 0.39195739],
                     'O': [0.15591627, 0.60768372, 0.39195739]}        
        else:
            raise IOError("Unsupported type of orbitals for Huckel representation")
            
# TODO: Check if energies and interactions should be scaled differently (calculation with output of hamiltonian elements)
        if 'Energy' in kwargs.keys():
# TODO: In energy units Hartree
            Energy_p=kwargs['Energy']
        else:
            Energy_p={'C': -0.4333, 
                     'N': -0.5677,
                     'O': -0.6319}      # In hartree
        
        if 'Interaction' in kwargs.keys():
            Interaction_p=kwargs['Interaction']
        else:
            Interaction_p={}
            if overlap_scaled:
                for ii in Energy_p:
                    for jj in Energy_p:
                        typ="".join([ii,jj])
                        Interaction_p[typ]=-1.864456    # in Hartree
            else:
                raise IOError('Calculation without scaling the interaction by overlap is not yet implemented')
        
                        

            
        Huckel_mol=Molecule("".join([self.name," Huckel rep."]))
        
        # structure definition
        Huckel_mol.struc=self.copy(indx=At_list)
        Huckel_mol.repre['AllAtom']=Huckel_mol.struc

        # center molecule before you add orbitals because for every atom I 

        # atomic orbital definition
        for ii in range(Huckel_mol.struc.nat):
            with position_units("Bohr"):
                atom_type=Huckel_mol.struc.at_type[ii]
                coeffs=Coeff_p[atom_type]
                exps=Exps_p[atom_type]
                coor=Huckel_mol.struc.coor.value
                orbit_type='p'
                atom_indx=ii
                Huckel_mol.ao.add_orbital(coeffs,exps,coor,orbit_type,atom_indx,atom_type)
        
        # contraction of orbital overlap from 3 orbitals for px, py, pz into one p orbital in direction of nvec
        Huckel_mol.ao.get_overlap()
        nvec=np.array(nvec)/np.linalg.norm(nvec)
        ContM=np.zeros((Huckel_mol.struc.nat,Huckel_mol.ao.nao_orient),dtype='f8')
        for ii in range(Huckel_mol.struc.nat):
            for jj in range(3):
                ContM[ii,3*ii+jj]=nvec[jj]
        SS=np.dot(ContM,np.dot(Huckel_mol.ao.overlap,ContM.T))
        
        # Calculation of molecular orbitals
        HH=np.zeros((Huckel_mol.struc.nat,Huckel_mol.struc.nat),dtype='f8')
        for ii in range(Huckel_mol.struc.nat):
            atom_type=Huckel_mol.struc.at_type[ii]
            HH[ii,ii]=Energy_p[atom_type]
            for jj in range(ii+1,Huckel_mol.struc.nat):
                inter_type="".join([atom_type,Huckel_mol.struc.at_type[jj]])
                HH[ii,jj]=Interaction_p[inter_type]*SS[ii,jj]
                HH[jj,ii]=HH[ii,jj]
        eng,vec=scipy.linalg.eigh(HH,SS)    # energies in vector and eigenvectors in columns
        
        # expand single p orbital expansioon coefficients into px, py, pz
        coeff=np.dot(ContM.T,vec)
        
        # Molecular orbital definition
        symm=['None']*len(eng)
        occ=np.zeros(len(eng),dtype='f8')
        if Huckel_mol.struc.nat%2==0:
            occ[:Huckel_mol.struc.nat//2]=2.0
        else:
            occ[:Huckel_mol.struc.nat//2]=2.0
            occ[:Huckel_mol.struc.nat//2+1]=1.0
        with energy_units('Ha'):
            Huckel_mol.mo.add_all(coeff.T,eng,occ,symm) # for this function molecular expansion coefficients have to be in rows        
        
        if add_excit:
            # Add lowest transition state (HOMO->LUMO)
            with energy_units('Ha'):
                if Huckel_mol.struc.nat%2==0:
                    HOMO=Huckel_mol.struc.nat//2  # for definition of el. transitions indexing is from 1
                    LUMO=HOMO+1                   # for definition of el. transitions indexing is from 1
                excit=Excitation(eng[LUMO-1]-eng[HOMO-1],None,coeff=[[HOMO,LUMO,1.0]],method='Huckel',root=1)
                dipole,at_dipole=excit.get_transition_dipole(Huckel_mol.ao,Huckel_mol.struc.coor,MO=Huckel_mol.mo)
                excit.dipole=Dipole(np.array(dipole,dtype='f8'),'AU')
                Huckel_mol.exct.append(excit)
        
# TODO: calculate transition dipole
        return Huckel_mol
    
    def get_3D_oscillator_representation(self,NMN,TrDip,At_list=None,scale_by_overlap=False,nearest_neighbour=True,centered="Bonds",verbose=False,**kwargs):
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
        **kwargs : dictionary
            specification of aditional parameers when using linear combination of
            more transitions. With name ``'TotTrDip'`` there have to be defined 
            transition dipoles for individual transitions. With name ``'Coef'``
            is definition of expansion coefficients for tranitions.
            
        Returns
        -------
        struc_oscilator :  Structure class
            In 'tr_dip' there are stored classical dipoles in ATOMIC UNITS.
            in 'coor' there are coordinates of dipoles 
        '''
        
        from ..calc import molecule_osc_3D
                      
        if At_list is None:
            struc1=self.copy()
        else:
            struc1=self.copy(indx=At_list)
        
        if struc1.bonds is None:
            struc1.guess_bonds()
        
        Ndip1=len(struc1.bonds)
        bond1=struc1.bonds
        
        # Elementary dipole size is dependent on atom types (dipole in center of N-C bond should be 
        # different than dipole in center of C-C and also in centrer N-N). So far all dipoles are the 
        # same for all atom types. This should be changed and be dependent on AtType
        if scale_by_overlap:
            from ..qch_functions import dipole_STO
            
            factor1=np.zeros(Ndip1,dtype='f8')
            coef=np.array([0.06899907,0.31642396,0.74430829])   # only for C-C bonds and for pz orbitals
            exp=np.array([7.86827235,1.88128854,0.54424926])    # only for C-C bonds and for pz orbitals
            r1=np.array([0.0,0.0,0.0])
            rr1=struc1.coor._value
            for ii in range(Ndip1):
                R=np.sqrt(np.dot(rr1[bond1[ii,1],:]-rr1[bond1[ii,0],:],rr1[bond1[ii,1],:]-rr1[bond1[ii,0],:]))
                r2=np.array([R,0.0,0.0])            
                Dip=dipole_STO(r1,r2,coef,coef,exp,exp,[0,0,1],[0,0,1]) # only dependent on radial distance and not actual positions - it could be made more accurate this is only simplest approximation
                factor1[ii]=np.sqrt(np.dot(Dip,Dip)) 
        else:
            factor1=np.ones(Ndip1,dtype='f8')
        
        if isinstance(TrDip,list):
            if len(TrDip)==len(NMN):
                MultiNM1=True
            else:
                raise IOError('Normal mode number list have to have the same dimension as list with dipoles')
        else:
            MultiNM1=False
        
        do1=np.zeros((Ndip1,3),dtype='f8')
        
        if verbose:
            print('first molecule')
        if MultiNM1:
            DipoleSize=kwargs['TotTrDip']
            Coef=kwargs['Coef']
            for ii in range(len(TrDip)):
                ro_tmp, do_tmp = molecule_osc_3D(struc1.coor._value,struc1.bonds,
                                               factor1,NMN[ii],TrDip[ii],
                                               centered,nearest_neighbour,verbose=verbose)
                ro1=np.copy(ro_tmp)
                do1+=Coef[ii]*do_tmp
            # normalize total dipole
            Dip=np.sum(do1,0)
            norm=np.sqrt(np.dot(Dip,Dip))
            do1=do1*DipoleSize/norm
        else:
            ro1, do1 = molecule_osc_3D(struc1.coor._value,struc1.bonds,factor1,NMN,
                                    TrDip,centered,nearest_neighbour,verbose=verbose)
        
        dipole1=np.sum(do1,0)
        if verbose:
            print('TotalDip1:',dipole1)
        
        struc_oscilator=Structure()
        with position_units('Bohr'):
            struc_oscilator.coor=Coordinate(ro1)
        struc_oscilator.tr_dip=do1
        struc_oscilator.nat=Ndip1
        struc_oscilator.name="".join([self.name,' 3D oscillator representation'])
        for ii in range(struc_oscilator.nat):
            struc_oscilator.at_type="".join([struc1.at_type[bond1[ii][0]],struc1.at_type[bond1[ii][1]]])                        
        struc_oscilator.tr_char=np.zeros(Ndip1,dtype='f8')
        struc_oscilator.tr_quadr2=np.zeros(Ndip1,dtype='f8')
        struc_oscilator.tr_quadrR2=np.zeros((6,Ndip1),dtype='f8')
        
        return struc_oscilator

VdW_radius={'ca': 1.9080,'f': 1.75,'h1': 1.3870,'h2': 1.2870,'h3': 1.4090,
            'h4': 1.4090,'h5': 1.3590,'ha': 1.4590,'hc':1.4870,'hn': 0.6000,'ho': 0.0000,
            'hx': 1.1000,'o': 1.6612,'oh': 1.7210,'c': 1.9080,'c1': 1.9080,'c2': 1.9080,
            'c3': 1.9080,'cc': 1.9080,'cd': 1.9080,'ce': 1.9080,'cf': 1.9080,'cg': 1.9080,
            'ch': 1.9080,'cp': 1.9080,'cq': 1.9080,'cu': 1.9080,'cv': 1.9080,'cx': 1.9080,
            'cy': 1.9080,'cz': 1.9080,'n': 1.8240,'n1': 1.8240,'n2': 1.8240,'n3': 1.8240,'n4': 1.8240,
            'na': 1.8240,'nb': 1.8240,'nc': 1.8240,'nd': 1.8240,'ne': 1.8240,'nf': 1.8240,'nh': 1.8240,
            'no': 1.8240,'cl': 1.948}

def read_nist():
  '''Reads and converts the atomic masses from the "Linearized ASCII Output"
  into global variable nist_mass, see http://physics.nist.gov.
  
  nist_mass : array dimension (Nx3)
      nist_mass[i]=[atom_i_name,atom_i_mass]
  '''
  global nist_mass, nist_indx    
  #''' Pole Nx2 kde jsou na prvnim miste nazvy atomu a na druhem miste
  #jsou atomve hmotnosti. Atomy jsou serazeny podle protonovych cisel. Pole je cislovano od 0!!!
  #dulezita zmena oproti fortranu ''' 
  
  f = open(nist_file,'r')
  flines = f.readlines()
  f.close()
  #print(flines)
  
  nist_mass = []
  index = None
  new = True
  
  def rm_brackets(text,rm=['(',')','[',']']):
    for i in rm:
      text = text.replace(i,'')
    return text
  
  for line in flines:
    thisline = line.split()
    if 'Atomic Number =' in line:
      i = int(thisline[-1]) - 1
      new = (i != index)
      if new:
        nist_mass.append(['',0])
      index = i
    elif 'Atomic Symbol =' in line and new:
      nist_mass[index][0] = thisline[-1]
    elif 'Standard Atomic Weight =' in line and new:
      nist_mass[index][1] = float(rm_brackets(thisline[-1]))
  
  nist_indx={}
  for ii in range(len(nist_mass)):
      nist_indx[nist_mass[ii][0]] = ii+1
    
def get_mass(atom):
  '''Returns the standard atomic mass of a given atom.
    
  Parameters
  ----------
  atom : int or str
    Contains the name or atomic number of the atom.
  
  Returns
  -------
  mass : float
    Contains the atomic mass in atomic units.
  '''
  
  if nist_mass is None:
    read_nist()
  try:                          # If atom defined by proton number then return its mass
    atom = int(atom) - 1
    return nist_mass[atom][1]
  except ValueError:            # if atom defined by string return its mass
    return nist_mass[get_atom_indx(atom)][1]
    #return dict(nist_mass)[atom.title()]

def get_atom_symbol(atom):
  '''Returns the atomic symbol of a given atom.
    
  Parameters
  ----------
  atom : int or str
    Contains the atomic number of the atom.
  
  Returns
  -------
  symbol : str
    Contains the atomic symbol.
  '''
  
  if nist_mass is None:
    read_nist()  
  try:
    atom = int(atom) - 1
    return nist_mass[atom][0]
  except ValueError:    
    return atom.upper()

def get_atom_indx(atom):
    '''Returns proton number of a given atom.
    
    Parameters
    ----------
    atom : str
      Contains the atomic number of the atom.
    
    Returns
    -------
    index : integer
      Proton number of given atom
    '''
    
    if nist_indx is None:
        read_nist()  
    try:
        atom = int(atom) - 1
        return nist_mass[atom][0]
    except ValueError:
#        indx=np.where(np.array(nist_mass)[:,0]==atom)[0]
#        if len(indx)==0:
#            raise IOError('Unknown atom type')
#        else:
#            index = indx[0]+1
#            return index
        return nist_indx[atom]
  

#==============================================================================
# TESTS
#==============================================================================
       
'''----------------------- TEST PART --------------------------------'''
if __name__=="__main__":
    print(' ')
    print('TESTS')
    print('-------')
    
    import Program_Manager.QuantumChem.qch_functions as qch
    import matplotlib.pyplot as plt
    from sys import platform
    
    R=np.arange(0,4,0.1)
    Psi_func_gauss=np.zeros((len(R),3),dtype='f8')
    Rad_func_fit=np.zeros(len(R),dtype='f8')
    Psi_func_slater=np.zeros((len(R),3),dtype='f8')
    
    # Gaussian orbital
    Exps_Cp=[2.9412494, 0.6834831,  0.2222899]
    Coeff_Cp=[0.15591627,  0.60768372,  0.39195739]
    for ii in range(len(R)):
        r=np.array([R[ii],0.0,0.0]) # POsition vector in x direction
        Psi_func_gauss[ii,0]=qch.Psi_STO_form_GTO2(r,Coeff_Cp,Exps_Cp,[0,0,1])
        Psi_func_gauss[ii,1]=Psi_func_gauss[ii,0]
        r=np.array([0.0,0.0,R[ii]])
        Psi_func_gauss[ii,2]=qch.Psi_STO_form_GTO2(r,Coeff_Cp,Exps_Cp,[0,0,1])
        
    # Slater 2pz carbon orbital
    n=2             # for 2p orbital
    Z=3.5  #3.136         # effective charge for carbon 2p orbital 3.136 found in table but 3.5 was apparently used for gaussian fitting
    for ii in range(len(R)):
        if ii!=0:
            r=np.array([R[ii],0.0,0.0]) # Position vector in x direction
            rho=2*Z*R[ii]/n
            Psi_func_slater[ii,0]=1/(4*np.sqrt(2*np.pi))*rho*Z**(3/2)*np.exp(-rho/2)*r[2]/R[ii]
            Psi_func_slater[ii,1]=Psi_func_slater[ii,0]
            r=np.array([0.0,0.0,R[ii]]) # Position vector in z direction
            Psi_func_slater[ii,2]=1/(4*np.sqrt(2*np.pi))*rho*Z**(3/2)*np.exp(-rho/2)*r[2]/R[ii]
            
    # Fited parameters
    Exps_Cp=np.array([0.185742, 1.12,  13.5413])
    Coeff_Cp=[0.276651,  0.433794,  0.242884] 
    
    plt.plot(R,Psi_func_gauss[:,2],color='blue')
    plt.plot(R,Psi_func_slater[:,2],color='red')
    plt.show()

    if (platform=='cygwin' or platform=="linux" or platform == "linux2"):
        MolDir='/mnt/sda2/Dropbox/PhD/Programy/Python/Test/'
    elif platform=='win32':
        MolDir='C:/Dropbox/PhD/Programy/Python/Test/'
    

    from molecule import Molecule
    exct_index=0
    mol=Molecule('N7-Polyene')
    mol.load_Gaussian_fchk("".join([MolDir,'N7_exct_CAM-B3LYP_LANL2DZ_geom_CAM-B3LYP_LANL2DZ.fchk']))
    indx_Huckel=np.where(np.array(mol.struc.at_type)=='C')[0]
    Huckel_mol=mol.struc.get_Huckel_molecule([0.0, 0.0, 1.0],At_list=indx_Huckel)
    Huckel_mol.create_multipole_representation(0,rep_name='Multipole2',MultOrder=2)
    print(Huckel_mol.exct[0].dipole.value,Huckel_mol.exct[0].dipole.units)
    