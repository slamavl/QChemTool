# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:33:56 2017

@author: Vladislav Sláma
"""
import numpy as np

from ..QuantumChem.Classes.structure import Structure
from ..QuantumChem.interaction import charge_charge
from ..QuantumChem.calc import identify_molecule
from ..QuantumChem.read_mine import read_TrEsp_charges, read_mol2, read_gaussian_esp
from ..General.Potential import potential_charge, ElField_charge


#==============================================================================
#  Definition of class for polarizable environment
#==============================================================================
    
class Electrostatics:    
    ''' Class which manages electrostatic properties of the material 
    
    Parameters
    ----------
    coor : numpy.array of real (dimension Nx3) where N is number of atoms
        origin of density grid        
    charge : numpy.array or list of real (dimension N)
        charges on individual atoms        
    at_type : numpy.array of characters (dimension N)
        Array of atomic types for assignment of atomic charges
        (for example at_type=['CF','CF2','CD','FC','F2C',...])
        'CF' - inner FG carbon, 'CF2' - border FG carbon, 'CD' - defect atom,
        'FC'- fluorine connected to inner carbon, 'F2C'- fluorine connected to border
        carbon.
    Nat : real
        Number of atoms in the system
    '''
    
    def __init__(self,coor,charge,at_type):
        self.coor=np.copy(coor)
        self.charge=np.copy(charge)
        self.at_type=np.copy(at_type)
        self.Nat=len(coor)

# FIXME: Rename this function because it does not calculate Energy shift but only 
# eletrostatic interaction energy
    def get_EnergyShift(self,index=None,charge=None):
        ''' Function calculates electrostatic interaction energy between
        environment and defect.
        
        Parameters (optional)
        ------------
        index: list of integers
            List of atom indexes for which we would like to change the default
            charge
        charge: numpy array of real (dimension = len(index))
            New charges for specified atoms in index. If not specified zero 
            charges will be used.
        
        Returns
        -------
        Eshift : real
            Change in interaction energy for pigment in ground state and in 
            excited state in ATOMIC UNITS (Hartree)
        '''
        
        # Zero charges on atoms defined by index
        if index is not None:
            charge_tmp = np.zeros(len(index),dtype='f8')
            if charge is None:
                charge = np.zeros(len(index),dtype='f8')

            for ii in range(len(index)):
                charge_tmp[ii] = self.charge[index[ii]]
                self.charge[index[ii]] = charge[ii]
        
        # ground state interaction
        Def_charge=[]
        Def_coor=[]
        Env_charge=[]
        Env_coor=[]
        # Separate defect and environment
        for ii in range(self.Nat):
            if self.at_type[ii]=='CD':
                if index is not None:
                    if ii in index:
                        Env_charge.append(self.charge[ii])
                        Env_coor.append(self.coor[ii])
                    else:
                        Def_charge.append(self.charge[ii])
                        Def_coor.append(self.coor[ii])
                else:
                    Def_charge.append(self.charge[ii])
                    Def_coor.append(self.coor[ii])
            else:
                 Env_charge.append(self.charge[ii])
                 Env_coor.append(self.coor[ii])
        
        Def_charge=np.array(Def_charge,dtype='f8')
        Def_coor=np.array(Def_coor,dtype='f8')
        Env_charge=np.array(Env_charge,dtype='f8')
        Env_coor=np.array(Env_coor,dtype='f8')
        
        # calculate interaction energy:
        Eshift,dist=charge_charge(Def_coor,Def_charge,Env_coor,Env_charge,'Hartree')
        
        # return charges back to original value
        if index is not None:
            for ii in range(len(index)):
                self.charge[index[ii]] = charge_tmp[ii]
        
        return Eshift
    
    def get_EnergyShift_and_Derivative(self,index=None,charge=None):
        ''' Function calculates change in electrostatic interaction energy between
        environment and defect in ground state and defect in excited state and 
        derivative of this energy with respect to atomic coordinates.
        <A|V|A>-<G|V|G> and d(<A|V|A>-<G|V|G>)/dRi
        
        Parameters
        -------
        index : list of integers
            Indexes of atoms of one of the defects for which charges should be
            replaced by charges defined in ``charge``
        charge : list or numpy array of real
            New charges for atoms defined in ``index`` for calculation of
            interaction energy (usually ground state or zero charges)
        
        Returns
        -------
        Eshift : real
            Change in interaction energy for pigment in ground state and in 
            excited state in ATOMIC UNITS (Hartree)
        dEshift_R : numpy array of real (dimension 3*Natoms)
            Derivative of Coulomb interaction energy with respect to atomic 
            coordinates
        '''
        
        # Zero charges on atoms defined by index
        if index is not None:
            charge_tmp = np.zeros(len(index),dtype='f8')
            if charge is None:
                charge = np.zeros(len(index),dtype='f8')

            for ii in range(len(index)):
                charge_tmp[ii] = self.charge[index[ii]]
                self.charge[index[ii]] = charge[ii]
            
            
        
        # ground state interaction
        Def_charge=[]
        Def_coor=[]
        Env_charge=[]
        Env_coor=[]
        Mask = np.zeros(self.Nat,dtype="bool")
        # Separate defect and environment
        for ii in range(self.Nat):
            if self.at_type[ii]=='CD':
                if index is not None:
                    if ii in index:
                        Env_charge.append(self.charge[ii])
                        Env_coor.append(self.coor[ii])
                    else:
                        Def_charge.append(self.charge[ii])
                        Def_coor.append(self.coor[ii])
                        Mask[ii]=True
                else:
                    Def_charge.append(self.charge[ii])
                    Def_coor.append(self.coor[ii])
                    Mask[ii]=True
            else:
                 Env_charge.append(self.charge[ii])
                 Env_coor.append(self.coor[ii])
        
        Def_charge=np.array(Def_charge,dtype='f8')
        Def_coor=np.array(Def_coor,dtype='f8')
        Env_charge=np.array(Env_charge,dtype='f8')
        Env_coor=np.array(Env_coor,dtype='f8')
        
        # Calculate distance matrix
        R_env = np.tile(Env_coor,(len(Def_coor),1,1))
        R_def = np.tile(Def_coor,(len(Env_coor),1,1))
        R_def2env = (R_env - np.swapaxes(R_def,0,1))            # R[ii,jj,:]=coor_env[jj]-coor_def[ii]
        R_env2def = (R_def - np.swapaxes(R_env,0,1))
        
        # calculate potential and electric field of defect charges
        Potential = potential_charge(Def_charge,R_def2env)
        El_field_def2env = ElField_charge(Def_charge,R_def2env)
        El_field_env2def = ElField_charge(Env_charge,R_env2def)
        
        # Calculate electrostatic shifts and it's derivative
        Eshift = np.dot(Env_charge,Potential)
        Def_charge_3D = np.tile(Def_charge,(3,1))
        Def_charge_3D = np.swapaxes(Def_charge_3D,0,1)
        Env_charge_3D = np.tile(Env_charge,(3,1))
        Env_charge_3D = np.swapaxes(Env_charge_3D,0,1)
        dEshift_env = -El_field_def2env*Env_charge_3D
        dEshift_def = -El_field_env2def*Def_charge_3D
        dEshift_R = np.zeros((self.Nat,3),dtype='f8')
        count_def = 0
        count_env = 0
        for ii in range(self.Nat):
            if Mask[ii]:
                dEshift_R[ii] = dEshift_def[count_def]
                count_def += 1 
            else:
                dEshift_R[ii] = dEshift_env[count_env]
                count_env += 1 
        dEshift_R = np.reshape(dEshift_R,3*self.Nat)
        
        # return charges back to original value
        if index is not None:
            for ii in range(len(index)):
                self.charge[index[ii]] = charge_tmp[ii]
        
        return Eshift, dEshift_R
        

def PrepareMolecule_1Def(filenames,indx,FG_charges_in,ChargeType='qchem',verbose=False,**kwargs):
    ''' Read all informations needed for Electrostatics class and transform system
    with single defect into this class. Useful for calculation of transition site
    energy shifts for molecules surrounded by electrostatic environment. 
    
    Parameters
    ----------
    filenames : list of dictionary (dimension Nsystems)
        In the dictionaries there are specified all needed files which contains 
        nessesary information for transformig the system into Electrostatics class.
        keys:
        `'1def_structure'`: xyz file with system geometry and atom types
        `'charge_structure'`: xyz file with defect like molecule geometry for which transition charges were calculated
        `charge_grnd`: file with ground state charges for the defect
        `'charge_exct'`: file with excited state charges for the defect
    indx : list of integers (dimension Nsystems x 6)
        For every system there are specified indexes neded for asignment of defect 
        atoms. First three indexes correspond to center and two main axes of 
        reference structure (structure which was used for charges calculation)
        and the remaining three indexes are corresponding atoms of the defect
    FG_charges_in : list of real (dimension 2)
        [charge on inner fluorographene atom, charge on borded fluorographe carbon]
    ChargeType : string (optional - init='qchem')
        Specifies which method was used for calcultion of ground and excited state
        charges for defect atoms. Allowed types are: 'qchem','qchem_all','AMBER'
        and 'gaussian'. **'qchem'** - charges calculated by fiting Q-Chem ESP on carbon
        atoms. **'qchem_all'** - charges calculated by fiting Q-Chem ESP on all
        atoms, only carbon charges are used and same charge is added to all carbon
        atoms in order to have neutral molecule. **'AMBER'** and **'gaussian'**
        not yet fully implemented.
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    **kwargs : dictionary (optional)
        If structure of the system and indexes of the defects are known they
        could be included as ``{'Structure': struc,'index1': index1}`` to 
        avoid re-assignment and save time 
    
    Returns
    -------
    Elstat_mol : Electrostatics class 
        Fluorographene with defect in Electrostatics class which contains all 
        information needed for calculation of <E|V|E>-<G|V|G> for fluorographene 
        system.
    at_type : numpy.array of characters (dimension N)
        Array of atomic types of molecule atoms
        (for example at_type=['C','N','C','C',...])

    '''    
    indx_center_test=indx[0]
    indx_x_test=indx[1]
    indx_y_test=indx[2]
    
    indx_center1=indx[3]
    indx_x1=indx[4]
    indx_y1=indx[5]
    
    FG_charges={'CF': FG_charges_in[0],'FC': -FG_charges_in[0],'CF2': FG_charges_in[1],
                'F2C': -FG_charges_in[1]/2.0,'CD': 0.0,'C': 0.0}

    # Specify files:
    xyzfile2=filenames['charge_structure']
    filenameESP_grnd=filenames['charge_grnd']
    filenameESP_exct=filenames['charge_exct']
    xyzfile=filenames['1def_structure']
    
    # Read filesif verbose:
    if verbose:
        print('        Loading molecules and charges...')
    struc_test=Structure()
    struc_test.load_xyz(xyzfile2)
    if 'Structure' in kwargs.keys():
        struc=kwargs['Structure'].copy()    # no need to copy structure is not changed
    else:
        struc=Structure()
        struc.load_xyz(xyzfile)  #FG with single defect
    
    if ChargeType=='qchem' or ChargeType=='qchem_all':
        coor_grnd,charge_grnd,at_type=read_TrEsp_charges(filenameESP_grnd,verbose=False)
        coor_exct,charge_exct,at_type=read_TrEsp_charges(filenameESP_exct,verbose=False)
    if ChargeType=='AMBER':
        coor_grnd,Bond,charge_grnd,AtName,AtType,MOL,Molname,info=read_mol2(filenameESP_grnd)
        coor_exct,Bond,charge_exct,AtName,AtType,MOL,Molname,info=read_mol2(filenameESP_exct)
    if ChargeType=='gaussian':
        Points,ESP,coor_grnd,charge_grnd=read_gaussian_esp(filenameESP_grnd,output_charge=True)
        Points,ESP,coor_exct,charge_exct=read_gaussian_esp(filenameESP_exct,output_charge=True)
#    struc.center(indx_center1,indx_x1,indx_y1)
    
    if ChargeType=='qchem_all':
        # ground state charges
        chtmp=0.0
        charge=[]
        coor=[]
        for ii in range(len(at_type)):
            if at_type[ii]=='H':
                chtmp+=charge_grnd[ii]
            else:
                charge.append(charge_grnd[ii])
                coor.append(coor_grnd[ii])
        charge=np.array(charge,dtype='f8')
        coor=np.array(coor,dtype='f8')
        charge+=chtmp/len(charge)
        coor_grnd=np.copy(coor)
        charge_grnd=np.copy(charge)

        # excited state charges
        chtmp=0.0
        charge=[]
        coor=[]
        for ii in range(len(at_type)):
            if at_type[ii]=='H':
                chtmp+=charge_exct[ii]
            else:
                charge.append(charge_exct[ii])
                coor.append(coor_exct[ii])
        charge=np.array(charge,dtype='f8')
        coor=np.array(coor,dtype='f8')
        charge+=chtmp/len(charge)
        coor_exct=np.copy(coor)
        charge_exct=np.copy(charge)
    
    if "index1" in kwargs.keys():
        index1=kwargs["index1"]
    else:
        index1=identify_molecule(struc,struc_test,indx_center1,indx_x1,indx_y1,indx_center_test,indx_x_test,indx_y_test,onlyC=True)
        if len(index1)!=len(np.unique(index1)):
            raise IOError('There are repeating elements in index file')
        
    # Identify inside carbons connected to fluorines and outside ones
    if struc.bonds is None:
        struc.guess_bonds()
    NBonds=np.zeros(len(struc.bonds),dtype='i')
    for ii in range(len(struc.bonds)):
        if struc.at_type[struc.bonds[ii,0]]=='C' and struc.at_type[struc.bonds[ii,1]]=='F':
            NBonds[struc.bonds[ii,0]]+=1
    
    # Assing type for every atom - in this step all fluorines will have the same type
    Elstat_Type=[]
    for ii in range(struc.nat):
        if struc.at_type[ii]=='C':
            if NBonds[ii]==2:
                Elstat_Type.append('CF2')
            elif NBonds[ii]==1:
                Elstat_Type.append('CF')
            else:
                Elstat_Type.append('CD')
        elif struc.at_type[ii]=='F':
            Elstat_Type.append('FC')
    
    # Assign different atom types for fluorines at the border:
    for ii in range(len(struc.bonds)):
        if Elstat_Type[struc.bonds[ii,0]]=='CF2' and Elstat_Type[struc.bonds[ii,1]]=='FC':
            Elstat_Type[struc.bonds[ii,1]]='F2C'
    
    # Check if defect carbons were correctly determined:
    for ii in range(struc.nat):
        if Elstat_Type[ii]=='CD' and ( not (ii in index1)):
            raise IOError('Wrongly determined defect atoms')
            
    # Asign charges for fluorographene:
    Elstat_Charge=np.zeros(struc.nat,dtype='f8')
    for ii in range(struc.nat):
        Elstat_Charge[ii]=FG_charges[Elstat_Type[ii]]
    
    # Asign charges for defect:
    Elstat_Charge_grnd=np.copy(Elstat_Charge)
    Elstat_Charge_exct=np.copy(Elstat_Charge)
    for ii in range(len(index1)):
        Elstat_Charge_grnd[index1[ii]]=charge_grnd[ii]
        Elstat_Charge_exct[index1[ii]]=charge_exct[ii]
        Elstat_Charge[index1[ii]]=charge_exct[ii]-charge_grnd[ii]
    
    Elstat_mol=Electrostatics(struc.coor._value,Elstat_Charge,Elstat_Type) 
    
    return Elstat_mol,index1,charge_grnd, charge_exct

def PrepareMolecule_2Def(filenames,indx,FG_charges_in,ChargeType='qchem',verbose=False,**kwargs):
    ''' Read all informations needed for Electrostatics class and transform system
    with single defect into this class. Useful for calculation of transition site
    energy shifts for molecules surrounded by electrostatic environment. 
    
    Parameters
    ----------
    filenames : list of dictionary (dimension Nsystems)
        In the dictionaries there are specified all needed files which contains 
        nessesary information for transformig the system into Electrostatics class.
        keys:
        `'2def_structure'`: xyz file with system geometry and atom types
        `'charge_structure'`: xyz file with defect like molecule geometry for which transition charges were calculated
        `charge_grnd`: file with ground state charges for the defect
        `'charge_exct'`: file with excited state charges for the defect
    indx : list of integers (dimension 9)
        There are specified indexes neded for asignment of defect 
        atoms. First three indexes correspond to center and two main axes of 
        reference structure (structure which was used for charges calculation)
        and the remaining six indexes are corresponding atoms of the defects 
        on fluorographene system (three correspond to first defect and the last
        three to the second one).
    FG_charges_in : list of real (dimension 2)
        [charge on inner fluorographene atom, charge on borded fluorographe carbon]
    ChargeType : string (optional - init='qchem')
        Specifies which method was used for calcultion of ground and excited state
        charges for defect atoms. Allowed types are: 'qchem','qchem_all','AMBER'
        and 'gaussian'. **'qchem'** - charges calculated by fiting Q-Chem ESP on carbon
        atoms. **'qchem_all'** - charges calculated by fiting Q-Chem ESP on all
        atoms, only carbon charges are used and same charge is added to all carbon
        atoms in order to have neutral molecule. **'AMBER'** and **'gaussian'**
        not yet fully implemented.
    verbose : logical (optional - init=False)
        If `True` aditional information about whole proces will be printed
    **kwargs : dictionary (optional)
        If structure of the system and indexes of the defects are known they
        could be included as ``{'Structure': struc,'index1': index1,'index2': 
        index2}`` to avoid re-assignment and save time
    
    Returns
    -------
    Elstat_mol : Electrostatics class 
        Fluorographene with defect in Electrostatics class which contains all 
        information needed for calculation of <E|V|E>-<G|V|G> for fluorographene 
        system.
    at_type : numpy.array of characters (dimension N)
        Array of atomic types of molecule atoms
        (for example at_type=['C','N','C','C',...])

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
    
    FG_charges={'CF': FG_charges_in[0],'FC': -FG_charges_in[0],'CF2': FG_charges_in[1],
                'F2C': -FG_charges_in[1]/2.0,'CD': 0.0,'C': 0.0}
    
    # Specify files:
    xyzfile_chrg1=filenames['charge1_structure']
    filenameESP_grnd1=filenames['charge1_grnd']
    filenameESP_exct1=filenames['charge1_exct']
    xyzfile_chrg2=filenames['charge2_structure']
    filenameESP_grnd2=filenames['charge2_grnd']
    filenameESP_exct2=filenames['charge2_exct']
    xyzfile=filenames['2def_structure']
    
    # Read files:
    if verbose:
        print('        Loading molecules and charges...')
    struc1_test=Structure()  # from charge fitting
    struc2_test=Structure()  # from charge fitting
    struc1_test.load_xyz(xyzfile_chrg1)
    struc2_test.load_xyz(xyzfile_chrg2)
    if 'Structure' in kwargs.keys():
        struc=kwargs['Structure'].copy()    # no need to copy structure is not changed
    else:
        struc=Structure()
        struc.load_xyz(xyzfile)  #FG with two defects
    if ChargeType=='qchem' or ChargeType=='qchem_all':
        coor_grnd1,charge_grnd1,at_type1=read_TrEsp_charges(filenameESP_grnd1,verbose=False)
        coor_exct1,charge_exct1,at_type1=read_TrEsp_charges(filenameESP_exct1,verbose=False)
        coor_grnd2,charge_grnd2,at_type2=read_TrEsp_charges(filenameESP_grnd2,verbose=False)
        coor_exct2,charge_exct2,at_type2=read_TrEsp_charges(filenameESP_exct2,verbose=False)
    if ChargeType=='AMBER':
        coor_grnd1,Bond,charge_grnd1,AtName,AtType,MOL,Molname,info=read_mol2(filenameESP_grnd1)
        coor_exct1,Bond,charge_exct1,AtName,AtType,MOL,Molname,info=read_mol2(filenameESP_exct1)
        coor_grnd2,Bond,charge_grnd2,AtName,AtType,MOL,Molname,info=read_mol2(filenameESP_grnd2)
        coor_exct2,Bond,charge_exct2,AtName,AtType,MOL,Molname,info=read_mol2(filenameESP_exct2)
    if ChargeType=='gaussian':
        Points,ESP,coor_grnd1,charge_grnd1=read_gaussian_esp(filenameESP_grnd1,output_charge=True)
        Points,ESP,coor_exct1,charge_exct1=read_gaussian_esp(filenameESP_exct1,output_charge=True)
        Points,ESP,coor_grnd2,charge_grnd2=read_gaussian_esp(filenameESP_grnd2,output_charge=True)
        Points,ESP,coor_exct2,charge_exct2=read_gaussian_esp(filenameESP_exct2,output_charge=True)
    if ChargeType=='qchem_all':
        # ground state charges
        chtmp=0.0
        charge=[]
        coor=[]
        for ii in range(len(at_type1)):
            if at_type1[ii]=='H':
                chtmp+=charge_grnd1[ii]
            else:
                charge.append(charge_grnd1[ii])
                coor.append(coor_grnd1[ii])
        charge=np.array(charge,dtype='f8')
        coor=np.array(coor,dtype='f8')
        charge+=chtmp/len(charge)
        coor_grnd1=np.copy(coor)
        charge_grnd1=np.copy(charge)
        
        chtmp=0.0
        charge=[]
        coor=[]
        for ii in range(len(at_type2)):
            if at_type2[ii]=='H':
                chtmp+=charge_grnd2[ii]
            else:
                charge.append(charge_grnd2[ii])
                coor.append(coor_grnd2[ii])
        charge=np.array(charge,dtype='f8')
        coor=np.array(coor,dtype='f8')
        charge+=chtmp/len(charge)
        coor_grnd2=np.copy(coor)
        charge_grnd2=np.copy(charge)

        # excited state charges
        chtmp=0.0
        charge=[]
        coor=[]
        for ii in range(len(at_type1)):
            if at_type1[ii]=='H':
                chtmp+=charge_exct1[ii]
            else:
                charge.append(charge_exct1[ii])
                coor.append(coor_exct1[ii])
        charge=np.array(charge,dtype='f8')
        coor=np.array(coor,dtype='f8')
        charge+=chtmp/len(charge)
        coor_exct1=np.copy(coor)
        charge_exct1=np.copy(charge)
        
        chtmp=0.0
        charge=[]
        coor=[]
        for ii in range(len(at_type2)):
            if at_type2[ii]=='H':
                chtmp+=charge_exct2[ii]
            else:
                charge.append(charge_exct2[ii])
                coor.append(coor_exct2[ii])
        charge=np.array(charge,dtype='f8')
        coor=np.array(coor,dtype='f8')
        charge+=chtmp/len(charge)
        coor_exct2=np.copy(coor)
        charge_exct2=np.copy(charge)
    
    if "index1" in kwargs.keys() and "index2" in kwargs.keys():
        index1=kwargs["index1"]
        index2=kwargs["index2"]
    else:    
        index1=identify_molecule(struc,struc1_test,indx_center1,indx_x1,indx_y1,indx_center_test,indx_x_test,indx_y_test,onlyC=True)
        index2=identify_molecule(struc,struc2_test,indx_center2,indx_x2,indx_y2,indx_center_test,indx_x_test,indx_y_test,onlyC=True)
        if len(index1)!=len(np.unique(index1)) or len(index2)!=len(np.unique(index2)):
            raise IOError('There are repeating elements in index file')


    # Identify inside carbons connected to fluorines and outside ones
    if struc.bonds is None:
        struc.guess_bonds()
    NBonds=np.zeros(len(struc.bonds),dtype='i')
    for ii in range(len(struc.bonds)):
        if struc.at_type[struc.bonds[ii,0]]=='C' and struc.at_type[struc.bonds[ii,1]]=='F':
            NBonds[struc.bonds[ii,0]]+=1
    
    # Assing type for every atom - in this step all fluorines will have the same type
    Elstat_Type=[]
    for ii in range(struc.nat):
        if struc.at_type[ii]=='C':
            if NBonds[ii]==2:
                Elstat_Type.append('CF2')
            elif NBonds[ii]==1:
                Elstat_Type.append('CF')
            elif ii in index1:
                Elstat_Type.append('CD')
            elif ii in index2:
                Elstat_Type.append('CD')
            else:
                raise IOError("Wrong number of bonds for carbon atom")
        elif struc.at_type[ii]=='F':
            Elstat_Type.append('FC')
    
    # Assign different atom types for fluorines at the border:
    for ii in range(len(struc.bonds)):
        if Elstat_Type[struc.bonds[ii,0]]=='CF2' and Elstat_Type[struc.bonds[ii,1]]=='FC':
            Elstat_Type[struc.bonds[ii,1]]='F2C'
    
    # Check if defect carbons were correctly determined:
    for ii in range(struc.nat):
        if Elstat_Type[ii]=='CD':
            if not (ii in index1 or ii in index2):
                raise IOError('Wrongly determined defect atoms')
#        if Elstat_Type[ii]=='CD' and ( not (ii in index1)):
#            raise IOError('Wrongly determined defect atoms')
#        if Elstat_Type[ii]=='C' and ( not (ii in index2)):
#            raise IOError('Wrongly determined defect atoms')
            
    # Asign charges for fluorographene:
    Elstat_Charge=np.zeros(struc.nat,dtype='f8')
    for ii in range(struc.nat):
        Elstat_Charge[ii]=FG_charges[Elstat_Type[ii]]
    
    # Asign charges for defect:
    Elstat_Charge_grnd=np.copy(Elstat_Charge)
    Elstat_Charge_exct=np.copy(Elstat_Charge)
    for ii in range(len(index1)):
        Elstat_Charge_grnd[index1[ii]]=charge_grnd1[ii]
        Elstat_Charge_exct[index1[ii]]=charge_exct1[ii]
        Elstat_Charge[index1[ii]]=charge_exct1[ii]-charge_grnd1[ii]
    for ii in range(len(index2)):
        Elstat_Charge_grnd[index2[ii]]=charge_grnd2[ii]
        Elstat_Charge_exct[index2[ii]]=charge_exct2[ii]
        Elstat_Charge[index2[ii]]=charge_exct2[ii]-charge_grnd2[ii]
    
    Elstat_mol=Electrostatics(struc.coor._value,Elstat_Charge,Elstat_Type)
        
    return Elstat_mol,index1,index2,charge_grnd1,charge_grnd2,charge_exct1,charge_exct2

#def _CalculateEshift(filenames,ShortName,index_all,Eshift_QCH,Eshift_all,FG_charges,AlphaE,Alpha_E,BetaE,nvec_all,order=82,ChargeType='qchem',verbose=False):
#    ''' Calculates transition energy shift for molecule embeded in polarizable 
#    atom environment
#    
#    **It is better to use function in polarization module**    
#    '''
#    for ii in range(len(filenames)):
#        if verbose:
#            print('Calculation of interaction energy for:',ShortName[ii])
#        
#        Eshift2=0.0
#        factor=1.0
#        # read and prepare molecule
#        if ('1per' in ShortName[ii]) or ('1ant' in ShortName[ii]):
#            mol_Elstat,at_type=PrepareMolecule_1Def(filenames[ii],index_all[ii],FG_charges,ChargeType=ChargeType,verbose=False)
#            mol_polar,index1,charge=PolMod.prepare_molecule_1Def(filenames[ii],index_all[ii],AlphaE,Alpha_E,BetaE,verbose=False)
#        else:
#            mol_Elstat,at_type=PrepareMolecule_2Def(filenames[ii],index_all[ii],FG_charges,ChargeType=ChargeType,verbose=False)
#            #mol_polar,index1,index2,charge=PolMod.prepare_molecule_2Def(filenames[ii],index_all[ii],AlphaE/2.0,Alpha_E/2.0,BetaE/2.0,nvec=nvec_all[ii],verbose=False)
#            mol_polar,index1,index2,charge=PolMod.prepare_molecule_2Def(filenames[ii],index_all[ii],AlphaE,Alpha_E,BetaE,nvec=nvec_all[ii],verbose=False)
#            
#        # calculate energy shift - Electrostatics     
#        Eshift=mol_Elstat.get_EnergyShift()
#        
#        # calculate energy shift - Polarization
#        Eshift2,factor=mol_polar.calculate_EnergyShift(index1,charge,order=order,rescale_charges=True)
#        
#        print('Factor:',factor)
#        print('        Calc shift:',Eshift*factor*const.HaToInvcm,Eshift2*const.HaToInvcm,'QCH shift:',Eshift_QCH[ii])
#        Eshift_all[ii]=(Eshift*factor+Eshift2)*const.HaToInvcm
#        
#        # output charges to mathematica
#        Bonds=inter.GuessBonds(mol_Elstat.coor,bond_length=4.0)
#        mat_filename="".join(['Pictures/ElStat_',ShortName[ii],'_diff.nb'])
#        out.OutputMathematica(mat_filename,mol_Elstat.coor,Bonds,at_type,**{'TrPointCharge': mol_Elstat.charge})
#
#
