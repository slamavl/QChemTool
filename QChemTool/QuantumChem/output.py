# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:29:14 2016

@author: User
"""
import numpy as np
from copy import deepcopy
from QChemTool.General.UnitsManager import position_units

def OutputToPDB(Coord,AtType,filename='OutputPython.pdb'):
    ''' Write coordinates to pbd file
    
    Parameters
    ----------
    Coord : numpy.array of real (dimension Natoms x 3)
        Atomic coordinates in ANGSTROMS for every atom
    AtType : numpy.array or list of characters (dimension Natoms)
        List of atomic types (for example `AtType=['C','N','C','C',...]`)
    filename : string (optional - init='OutputPython.pdb')
        Specifies the filename for the output (including the path if needed)
    
    Notes
    -------
    ** Function should be changed to use ATOMIC UNITS instead of ANGSTROMS **
    '''
    
    print('Warning: Function OutputToPDB need ANGSTROMS as input coordinate units')
    
    if np.shape(Coord)[0]!=len(AtType):
         raise IOError('Wrong dimension of Coord or Atom types, try input Coord.T')
    NAtom=len(AtType)
    types_unq = np.unique(AtType)
    count = {}
    for ii in types_unq:
        count[ii] = 0
    with open(filename, "wt") as f:
        # Vypis hlavicky
        f.write("CRYST1    0.000    0.000    0.000  90.00  90.00  90.00 P 1           1 \n") # so far only molecules without PBC box are inmplemented
        counter=0        
        for ii in range(NAtom):
            count[AtType[ii]] += 1
            f.write("ATOM")
            f.write("{:7d}".format(ii+1))
            if counter<100:
                f.write("{:>3}".format(AtType[ii]))
                f.write("{:<3d}".format( count[AtType[ii]] ))
            else:
                f.write("{:>2}".format(AtType[ii]))
                f.write("{:<4d}".format( count[AtType[ii]] ))
#            f.write("{:<4d}".format(counter))
            f.write("NAM X   1")
            f.write("{:12.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format(Coord[ii,0],Coord[ii,1],Coord[ii,2],0.0,0.0))
        f.write("END")
    f.close()

def OutputToPDBbuff(Coord,AtType,count,f):
    ''' Add coordinates to pbd file which is allready opened
    
    Parameters
    ----------
    Coord : numpy.array of real (dimension Natoms x 3)
        Atomic coordinates in ANGSTROMS for every atom
    AtType : numpy.array or list of characters (dimension Natoms)
        List of atomic types (for example `AtType=['C','N','C','C',...]`)
    count : integer
        Counter how many molecules or MD steps were already written into pdb file
        (starting from 0)
    f : pointer
        pointer to opened file (example: with open(filename, "wt") as f)
    
    Notes
    -------
    ** Function should be changed to use ATOMIC UNITS instead of ANGSTROMS **
    '''
    
    print('Warning: Function OutputToPDBbuff need ANGSTROMS as input coordinate units')
    
    if np.shape(Coord)[0]!=len(AtType):
         raise IOError('Wrong dimension of Coord or Atom types, try input Coord.T')
    NAtom=len(AtType)
    if count==0:
            # Vypis hlavicky
            f.write("CRYST1    0.000    0.000    0.000  90.00  90.00  90.00 P 1           1 \n") # so far only molecules without PBC box are inmplemented
            counter=0        
            for ii in range(NAtom):
                counter=counter+1
                f.write("ATOM")
                f.write("{:7d}".format(ii+1))
                if counter<100:
                    f.write("{:>3}".format(AtType[ii]))
                    f.write("{:<3d}".format(counter))
                else:
                    f.write("{:>2}".format(AtType[ii]))
                    f.write("{:<4d}".format(counter))
    #            f.write("{:<4d}".format(counter))
                f.write("NAM X   1")
                f.write("{:12.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format(Coord[ii,0],Coord[ii,1],Coord[ii,2],0.0,0.0))
            f.write("END\n")
    else:
        counter=0
        for ii in range(NAtom):
            counter=counter+1
            f.write("ATOM")
            f.write("{:7d}".format(ii+1))
            if counter<100:
                f.write("{:>3}".format(AtType[ii]))
                f.write("{:<3d}".format(counter))
            else:
                f.write("{:>2}".format(AtType[ii]))
                f.write("{:<4d}".format(counter))
    #        f.write("{:<4d}".format(counter))
            f.write("NAM X   1")
            f.write("{:12.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format(Coord[ii,0],Coord[ii,1],Coord[ii,2],0.0,0.0))
        f.write("END\n")


def OutputToXYZ(Coord,AtType,filename='OutputPython.xyz'):
    ''' Write coordinates to xyz file
    
    Parameters
    ----------
    Coord : numpy.array of real (dimension Natoms x 3)
        Atomic coordinates in ANGSTROMS for every atom
    AtType : numpy.array or list of characters (dimension Natoms)
        List of atomic types (for example `AtType=['C','N','C','C',...]`)
    filename : string (optional - init='OutputPython.xyz')
        Specifies the filename for the output (including the path if needed)
    
    Notes
    -------
    ** Function should be changed to use ATOMIC UNITS instead of ANGSTROMS **
    '''

    print('Warning: Function OutputToXYZ need ANGSTROMS as input coordinate units')

    if np.shape(Coord)[0]!=len(AtType):
        raise IOError('Wrong dimension of Coord or Atom types, try input Coord.T')
    NAtom=len(AtType)
    with open(filename, "wt") as f:
        # Vypis hlavicky
        f.write("{:<7d}\n\n".format(NAtom))
        # Vypis coordinat
        for ii in range(NAtom):
            f.write("{:>2}".format(AtType[ii]))
            f.write("{:15.8f}{:15.8f}{:15.8f}\n".format(Coord[ii,0],Coord[ii,1],Coord[ii,2]))
    f.close()
    
def OutputTOmol2(Coor,Bond,Charge,AtName,AtType,MolName,filename,ChargeMethod='resp',AditionalInfo=[],**kwargs):
    ''' Write moleculat information to mol2 file
    
    Parameters
    ----------
    Coor : numpy.array of real (dimension Natoms x 3)
        Atomic coordinates in ANGSTROMS for every atom
    Bond : numpy.array of integer (dimension Nx2)
        In the array there are written pairs of atom indexes of atoms which are
        connected by chemical bond.
    Charge : numpy.array of real (dimension N)
        Array of ground state charges (default but it could be also excited state
        or transition charges for special cases) for every atom
    AtName : numpy.array or list of characters (dimension Natoms)
        List of atomic types (for example `AtType=['C','N','C','C',...]`)
    AtType : numpy.array of characters (dimension N)
        Forcefield atom type for every atom (for example for GAFF 
        AtType=['ca','c3','ca','ha',...])
    MolName : string
        Name of molecule for which mol2 file is generated. 3 uppercase letters.
    filename : string
        Specifies the filename for the output (including the path if needed)
    ChargeMethod : string (optional - init='resp')
        Name of the method used for obtaining charges. (For example 'resp', 'bcc', ...)
    AditionalInfo : list of strings (optional - init=[])
        Aditional info which should be written to the end of mol2 file
    **kwargs : dictionary
        Aditional info about molecule. For example with 'BondType' key
        we can specify multiplicity of the bond between two atoms as list of 
        integers - if not specified all bond are assumed to be single.
        With 'MolIndxAt' we can specify index of molecule to which every atom
        corresponds - if not specified all atoms are assumed to be from single 
        molecule. With 'MolNameAt' key we can specify molecule name for every 
        atom.
    
    Notes
    -------
    ** Function should be changed to use ATOMIC UNITS instead of ANGSTROMS **
    '''    
    
    print('Warning: Function OutputTOmol2 need ANGSTROMS as input coordinate units')

    is_MolNameAt=False
    is_MolIndxAt=False
    is_BondType=False
    for key in list(kwargs.keys()):
        if key=='MolNameAt':
            MolNameAt=kwargs['MolNameAt']
            is_MolNameAt=True
        elif key=='MolIndxAt':
            MolIndxAt=kwargs['MolIndxAt']
            is_MolIndxAt=True
        elif key=='BondType':
            BondType=kwargs['BondType']
            is_BondType=True
            
    Nat=len(Coor)
    Nbond=len(Bond)
    Ninfo=len(AditionalInfo)
    
    if not is_MolNameAt:
        MolNameAt=np.array([MolName]*Nat)
    if not is_MolIndxAt:
        MolIndxAt=np.ones(Nat,dtype='i8')
    if not is_BondType:
        BondType=np.ones(Nbond,dtype='i8')
        
    
    AtNameUnq=np.unique(AtName)
    NatName=len(AtNameUnq)
    
    counterName=np.zeros(NatName,dtype='i8')
    counter=0
    
    with open(filename, "wt") as f:
        f.write("@<TRIPOS>MOLECULE \n")
        f.write(MolName)
        f.write("\n")
        f.write("{:>5d}{:>6d}{:>6d}{:>6d}{:>6d} \n".format(Nat,Nbond,Ninfo,0,0))
        f.write("SMALL \n")
        f.write(ChargeMethod)
        f.write("\n \n \n")
        f.write("@<TRIPOS>ATOM \n")
        
        for ii in range(Nat):
            counter+=1
            f.write("{:>7d} ".format(counter))
            for jj in range(NatName):
                if AtName[ii]==AtNameUnq[jj]:
                    counterName[jj]+=1
                    f.write("{:1}{:<7d}".format(AtName[ii],counterName[jj]))
                    break
            f.write("{:10.4f}{:10.4f}{:10.4f}".format(Coor[ii,0],Coor[ii,1],Coor[ii,2]))
            f.write(" {:<3}".format(AtType[ii]))
            f.write("{:>8d}".format(MolIndxAt[ii]))
            f.write(" {:3}".format(MolNameAt[ii]))
            f.write("{:14.6f}\n".format(Charge[ii]))
        
        f.write("@<TRIPOS>BOND \n")
        counter=0
        for ii in range(Nbond):
            counter+=1
            f.write("{:>7d}".format(counter))
            f.write("{:>6d}{:>6d}{:>2d}\n".format(Bond[ii,0],Bond[ii,1],BondType[ii]))
        if Ninfo!=0:
            f.write("@<TRIPOS>SUBSTRUCTURE \n")
        for ii in range(Ninfo):
            f.write(AditionalInfo[ii])
            f.write("\n")
        
        f.write("\n")
    f.close()  
        
            
        

def OutputToGauss(Coord,AtType,calctype,method,basis,charge=0,
                  mutiplicity=1,NCPU=12,filename='OutputPython.gjf',
                  namechk='OutputPython.chk',old_chk=None,other_options=' ',
                  constrain=None,MEM_GB=None,Tight=True,verbose=True):
    ''' Create basic imput file for gaussian calculation (so far only for TD-DFT
    excited state calculation)
    
    Parameters
    ----------
    Coord : numpy.array of real (dimension Natoms x 3)
        Atomic coordinates in ANGSTROMS for every atom
    AtType : numpy.array or list of characters (dimension Natoms)
        List of atomic types (for example `AtType=['C','N','C','C',...]`)
    calctype : string
        ``calctype='excit'`` is for TD-DFT excited state properties calculation.
        ``calctype='opt_restricted'`` is for ground state geometry optimization
        with constrains.
    method : string
        HF, CIS or functional which is used for DFT calculation
    basis : string
        Basis set which is used for the calculation (e.g. 6-31G**)
    charge : integer (optional - init=0)
        Total charge of the molecule in times of electron charge.
    multiplicity : integer (optional - init=0)
        Spin multiplicity of calculated state (usualy ground state is singlet)
    NCPU : integer (optional - init=12)
        Number of CPUs used for parallel gaussian calculation.
    filename : string (optional - init='OutputPython.gjf')
        Name of gaussian input file.
    namechk : string (optional - init='OutputPython.chk')
        Name of gaussian checkpoint file
    oldchk : string (optional)
        Name of gaussian checkpoint file from previous calculation (if used)
    other_options : string ( optional - init=' ')
        Other options which should be used for gaussian calculation like specifying
        maximum numer of SCF cycles etc.
    constrain : list or numpy array of real (optional)
        Indexes of atoms which should be kept frozen during geometry optimization
        (starting from 0)
    MEM_GB : integer (optional)
        Size of required memory for Gaussian 09 calcuation in GB. If not present
        Gaussian default value will be used
    Tight : logical (optional init = True)
        Specify if tight convergence criteria should be used for SCF procedure 
    
    Notes
    -------
    ** Function should be changed to use ATOMIC UNITS instead of ANGSTROMS **
    Default option for excited state calculation is:
    ` DENSITY=Current scf=(Tight,XQC) Symmetry=(Loose,Follow) GFPrint GFInput IOp(6/7=3) IOp(9/40=3)`
    '''    
    
    if verbose:
        print('Warning: Function OutputToGauss need ANGSTROMS as input coordinate units')
    
    if Tight:
        tight_str="Tight,"
    else:
        tight_str=""
    
    if Coord is not None:
        if np.shape(Coord)[0]!=len(AtType):
            raise IOError('Wrong dimension of Coord or Atom types, try input Coord.T')
        NAtom=len(AtType)
    with open(filename, "wt") as f:
        # Vypis hlavicky         
        f.write("$RunGauss \n")
        f.write("%NProcShared=")
        f.write("{:<7d}\n".format(NCPU))
        if MEM_GB is not None:
            f.write("%mem={:}GB\n".format(MEM_GB))
        if old_chk is not None:
            f.write("".join(["%oldchk=",old_chk,'\n']))
        f.write("%chk=")
        f.write(namechk)
        f.write("\n")
        f.write("#p ")
        f.write(method)
        f.write("/")
        f.write(basis)
        if calctype=='excit':
            f.write("".join([" TD=(Singlets,NStates=10,Root=1) DENSITY=Current scf=(",tight_str,"XQC) Symmetry=(Loose,Follow) GFPrint GFInput IOp(6/7=3) IOp(9/40=3) "]))        
        elif calctype=='opt_restricted':
            f.write("".join([" Symmetry=(Loose,Follow) scf=(",tight_str,"XQC) opt=(",tight_str,"ModRedundant) GFPrint GFInput"]))
        f.write(other_options)
        f.write("\n\n")
        f.write("Gaussian imput generated by Python program \n\n")
        f.write("{:2d}{:>2d}\n".format(charge,mutiplicity))
        # Vypis coordinat
        if Coord is not None:
            for ii in range(NAtom):
                f.write("{:>2}".format(AtType[ii]))
                f.write("{:13.6f}{:13.6f}{:13.6f}\n".format(Coord[ii,0],Coord[ii,1],Coord[ii,2]))
        if constrain is not None:
            f.write("\n")
            for ii in constrain:
                f.write("X {:0d} F \n".format(ii+1))
        f.write("\n\n")
            
    f.close()    
    
def OutputToFile(data,filename):
    with open(filename, "wt") as f:
        for ii in range(len(data)):
            f.write("{:13.7f}\n".format(data[ii]))
    f.close()

def OutputToESP(ESP,filename='plot.esp'):
    ''' Writes coordinates of points and corresponding electrostatic potential 
    values into esp file with format from Q-chem calculation.
    
    Parameters
    ----------
    ESP : numpy.array of real (dimension Npoints x 4)
        Array where first three positions correspond to coordinates in ATOMIC 
        UNITS of point where electrostatic potential (ESP) is calculated and last
        position correspond to potential at that point in ATOMIC UNITS 
        (Hartre/charge_in_e) (for example `ESP[i]=[x_i,y_i,z_i,ESP_i]`)
    filename : string (optional - init='plot.esp')
        Name of esp file. Default name from Q-chem calculation is 'plot.esp'

    Notes
    -------
    
    '''
    
    with open(filename, "wt") as f:
        # Vypis hlavicky
        f.write(" Grid point positions and esp values from SCF density (all in a.u.) \n")
        f.write("\n")
        f.write("       X            Y            Z           GS                                  \n")
        f.write("\n")
        for ii in range(len(ESP)):
            f.write("{:.6e} {:.6e} {:.6e} {:.6e} \n".format(ESP[ii,0],ESP[ii,1],ESP[ii,2],ESP[ii,3]))
			
            #f.write("{:12.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format(Coord[ii,0],Coord[ii,1],Coord[ii,2],0.0,0.0))
        #f.write("END")
    f.close()               
        
    
def OutputEnergyScanNAB(NAtom,q,filesCOOR,filePRMTOP,filename,mm_options="cut=999.0, ntpr=50"):
    ''' Create basic imput file for calculation of energy scan with AMBER MD
    package (NAB software)
    
    Parameters
    ----------
    NAtom : integer
        Number of atoms in the molecule
    q : list or numpy.array of real (dimension Nsteps)
        List with values of coordinate along which energy is calculated. Units 
        for this coordinate are ANGSTROMS
    filesCOOR : list of string (dimension Nsteps)
        List of file names of pdb files with molecule geometry aong scanned 
        coordinate
    filename : string
        Name of NAB input file for calculation of energy scan
    mm_options : string (optional - init="cut=999.0, ntpr=50")
        Parameters wich specifies how the total energy for single configuration 
        is calculated. Other amber options could be specified however this is enough 
        and should not be changed.
    
    Notes
    -------
    
    '''
    
    NSteps=len(filesCOOR)
    with open(filename, "wt") as f:
        # Vypis hlavicky
        f.write("// run programe on computer with AMBER instaled with folowing commands: \n")
        f.write("// nab ")
        f.write(filename)
        f.write("\n")
        f.write('// ./a.out > EnergyScan.dat \n ')
        f.write('\n')
        f.write("molecule m; \n")
        f.write("float x[")
        f.write(str(3*NAtom))
#        f.write("{:d}".format(3*NAtom))
        f.write("], m_xyz[")
        f.write("{:d}".format(3*NAtom))
        f.write("], f_xyz[")
        f.write("{:d}".format(3*NAtom))
        f.write("], fret; \n")
        f.write("\n")
        
        for ii in range(NSteps):
            f.write('m = getpdb( "')
            f.write(filesCOOR[ii])
            f.write('" ); \n')
            
            f.write('readparm( m, "')
            f.write(filePRMTOP)
            f.write('" ); \n')
            
            f.write('mm_options( "')
            f.write(mm_options)
            f.write('" ); \n')
            
            f.write('mme_init( m, NULL, "::z", x, NULL); \n')
            f.write('setxyz_from_mol(m, NULL, m_xyz); \n')
            f.write('fret = mme( m_xyz, f_xyz, 0 ); \n')
            f.write('printf( "Energy of q= ')
            f.write("{:13.6f}".format(q[ii]))
            f.write(r' Angstrom is %20.12f kcal/mol \n", fret ); ')

            
            f.write("\n\n")
        f.write("\n")
    f.close()

def OutputMathematica(filename,Coor,Bonds,AtType,scaleDipole=1.0,**kwargs):
    ''' Create basic imput file for visualization of molecular properties with
    wolfram mathematica software
    
    Parameters
    ----------
    filename : string
        Name of mathematica input file. Shoud be in form of "name.nb"
    Coor : numpy.array of real (dimension Natoms x 3)
        Atomic coordinates for every atom. For mathematica it doesn't matter 
        which units we choose but we should be consistent for all properties
        of single molecule. Best is to plot everything in ATOMIC UNITS
    Bonds : numpy.array of integer (dimension Nx2)
        In the array there are written pairs of atom indexes of atoms which are
        connected by chemical bond.
    AtType : numpy.array or list of characters (dimension Natoms)
        List of atomic types (for example `AtType=['C','N','C','C',...]`)
    scaleDipole : real (optional - init=1.0)
        Factor which is used to scale atomic dipoles in order to be more visible
        after ploting because atomic dipoles are generaly small.
    **kwargs : dictionary
        Aditional options and properties which could be ploted. These properties
        will be listed by their keys in dictionary:
        'Charge' : numpy.array of real (dimension Natoms)
            List of atomic ground state charges
        'TrPointCharge' : numpy.array of real (dimension Natoms)
            List of atomic charges (ground state, transition charges,...) charges
            will be plotted in colour (not centered around zero bur around 
            (max-min)/2 )
        'AtDipole' : numpy.array of real (dimension Natoms x 3)
            Array with dipoles on every atom (transition dipoles, ground state,...)
        'rSphere_dip' : real
            Definition of atom sphere radius for dipole visualization (default 0.6)
        'rCylinder_dip' : real
            Definition of bond cylinder radius for dipole visualizatin (default 0.16)
        'DipoleCoor' : numpy.array of real (dimension Ndip x 3)
            Coordinates for every dipole in AtDipole if other dipoles than atom
            centered dipoles are used. Units should be the same as for atomic
            coordinates (ATOMIC UNITS)
        'Points' : numpy.array of real (dimension Npoints x 3)
            Coordinates of aditional points which should be plotted around molecule.
            Good for visualization of points where ESP was calculated. Units
            should be the same as atomic coordinates (ATOMIC UNITS)
        'Potential' : 
            If this key is present potential from atomic point charges defined 
            in 'TrPointCharge' is calculated and ploted in three planes defined
            by coordinate axes.
            
    Notes
    -------
    
    '''
    
    Nat=len(Coor)
    
    is_Charge=False
    is_TrPointCharge=False
    is_AtDipole=False
    is_Dipole=False
    is_Points=False
    is_Potential=False
    
    rSphere=0.6
    rCylinder=0.16
    rSphere_dip=0.6
    rCylinder_dip=0.16    
    
    for key in list(kwargs.keys()):
        if key=='Charge':
            GrCh=kwargs['Charge']
            is_Charge=True
        elif key=='TrPointCharge':
            TrCh=kwargs['TrPointCharge']
            normTrCh=max(TrCh)-min(TrCh)
            #if max(TrCh)>np.abs(min(TrCh)):
            #    normTrCh=max(TrCh)
            #else:
            #    normTrCh=np.abs(min(TrCh))
            is_TrPointCharge=True
        elif key=='Points':
            is_Points=True
            Points=kwargs['Points']
        elif key=='Potential':
            is_Potential=True
        elif key=='AtDipole':
            AtDip=kwargs['AtDipole']
            is_AtDipole=True
        elif key=='rSphere_dip':
            rSphere_dip=kwargs['rSphere_dip']
        elif key=='rCylinder_dip':
            rCylinder_dip=kwargs['rCylinder_dip']
        elif key=='Dipole' and 'DipoleCoor' in kwargs.keys():
            is_Dipole=True
            AtDip=np.copy(kwargs['Dipole'])
            DipoleCoor=np.copy(kwargs['DipoleCoor'])
    
    with open(filename, "wt") as f:
        f.write('arrowAxes[arrowLength_] :=Map[{Apply[RGBColor, #], Arrow[Tube[{{0, 0, 0}, #}]]} &, arrowLength IdentityMatrix[3]] \n')
        f.write('Graphics3D[{arrowAxes[10],')
        f.write("".join(['{',AtomColorMath(AtType[0]),',Sphere[','{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[0,0],Coor[0,1],Coor[0,2]),'},','{:0.3f}'.format(rSphere),']}']))
        for ii in range(1,Nat):
            f.write("".join([',{',AtomColorMath(AtType[ii]),',Sphere[','{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2]),'},','{:0.3f}'.format(rSphere),']}']))
            #f.write(',{',AtomColorMath(AtType[ii]),',Sphere[{{:0.3f},{:0.3f},{:0.3f}},{:0.3f}]}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2],rSphere))
        for ii in range(len(Bonds)):
            f.write("".join([',{Cylinder[{{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[Bonds[ii,0],0],Coor[Bonds[ii,0],1],Coor[Bonds[ii,0],2]),'},{']))
            f.write("".join(['{:0.3f},{:0.3f},{:0.3f}'.format(Coor[Bonds[ii,1],0],Coor[Bonds[ii,1],1],Coor[Bonds[ii,1],2]),'}},','{:0.3f}'.format(rCylinder),']}']))
            #f.write(',{Cylinder[{{{:0.3f},{:0.3f},{:0.3f}},{{:0.3f},{:0.3f},{:0.3f}}},{:0.3f}]}'.format(Coor[Bonds[ii,0],0],Coor[Bonds[ii,0],1],Coor[Bonds[ii,0],2],Coor[Bonds[ii,1],0],Coor[Bonds[ii,1],1],Coor[Bonds[ii,1],2],rCylinder))
        if is_Points:
            f.write(",Point[{")
            f.write("".join(['{','{:0.3f},{:0.3f},{:0.3f}'.format(Points[0,0],Points[0,1],Points[0,2]),'}'])) 
            for ii in range(1,len(Points)):
                f.write("".join([',{','{:0.3f},{:0.3f},{:0.3f}'.format(Points[ii,0],Points[ii,1],Points[ii,2]),'}'])) 
            f.write('}]')
        f.write('},Boxed->False,ImageSize->800,AxesOrigin -> {0, 0, 0}]')
        f.write("\n")
        
        if is_Charge:
            f.write('Graphics3D[{arrowAxes[10],')
            f.write("".join(['{ColorData["TemperatureMap",','{:0.3f}'.format(GrCh[0]/normTrCh),'],Sphere[','{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[0,0],Coor[0,1],Coor[0,2]),'},','{:0.3f}'.format(rSphere),']}']))
            for ii in range(1,Nat):
                f.write("".join([',{ColorData["TemperatureMap",','{:0.3f}'.format(GrCh[ii]/normTrCh),'],Sphere[','{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2]),'},','{:0.3f}'.format(rSphere),']}']))
                #f.write(',{',AtomColorMath(AtType[ii]),',Sphere[{{:0.3f},{:0.3f},{:0.3f}},{:0.3f}]}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2],rSphere))
            for ii in range(len(Bonds)):
                f.write("".join([',{Cylinder[{{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[Bonds[ii,0],0],Coor[Bonds[ii,0],1],Coor[Bonds[ii,0],2]),'},{']))
                f.write("".join(['{:0.3f},{:0.3f},{:0.3f}'.format(Coor[Bonds[ii,1],0],Coor[Bonds[ii,1],1],Coor[Bonds[ii,1],2]),'}},','{:0.3f}'.format(rCylinder),']}']))
                #f.write(',{Cylinder[{{{:0.3f},{:0.3f},{:0.3f}},{{:0.3f},{:0.3f},{:0.3f}}},{:0.3f}]}'.format(Coor[Bonds[ii,0],0],Coor[Bonds[ii,0],1],Coor[Bonds[ii,0],2],Coor[Bonds[ii,1],0],Coor[Bonds[ii,1],1],Coor[Bonds[ii,1],2],rCylinder))
            for ii in range(Nat):
                f.write("".join([',Text[','{:0.3f}'.format(GrCh[ii]),',{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2]),'}]']))                
            f.write('},Boxed->False,ImageSize->800,AxesOrigin -> {0, 0, 0}]')
            f.write("\n")
        
        if is_TrPointCharge:
            rSphere=0.85
            rCylinder=0.25
            f.write('Graphics3D[{arrowAxes[10],')
            f.write("".join(['{ColorData["TemperatureMap",','{:0.3f}'.format((TrCh[0]-min(TrCh))/normTrCh),'],Sphere[','{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[0,0],Coor[0,1],Coor[0,2]),'},','{:0.3f}'.format(rSphere),']}']))
            for ii in range(1,Nat):
                f.write("".join([',{ColorData["TemperatureMap",','{:0.3f}'.format((TrCh[ii]-min(TrCh))/normTrCh),'],Sphere[','{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2]),'},','{:0.3f}'.format(rSphere),']}']))
                #f.write(',{',AtomColorMath(AtType[ii]),',Sphere[{{:0.3f},{:0.3f},{:0.3f}},{:0.3f}]}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2],rSphere))
            for ii in range(len(Bonds)):
                f.write("".join([',{Cylinder[{{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[Bonds[ii,0],0],Coor[Bonds[ii,0],1],Coor[Bonds[ii,0],2]),'},{']))
                f.write("".join(['{:0.3f},{:0.3f},{:0.3f}'.format(Coor[Bonds[ii,1],0],Coor[Bonds[ii,1],1],Coor[Bonds[ii,1],2]),'}},','{:0.3f}'.format(rCylinder),']}']))
                #f.write(',{Cylinder[{{{:0.3f},{:0.3f},{:0.3f}},{{:0.3f},{:0.3f},{:0.3f}}},{:0.3f}]}'.format(Coor[Bonds[ii,0],0],Coor[Bonds[ii,0],1],Coor[Bonds[ii,0],2],Coor[Bonds[ii,1],0],Coor[Bonds[ii,1],1],Coor[Bonds[ii,1],2],rCylinder))
            for ii in range(Nat):
                f.write("".join([',Text[','{:0.3f}'.format(TrCh[ii]),',{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2]),'}]']))                
            f.write('},Boxed->False,ImageSize->800,AxesOrigin -> {0, 0, 0}]')
            f.write("\n")
        
        if is_AtDipole:
            #Dipole output
            f.write('Graphics3D[{arrowAxes[10],')
            f.write("".join(['{',AtomColorMath(AtType[0]),',Sphere[','{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[0,0],Coor[0,1],Coor[0,2]),'},','{:0.3f}'.format(rSphere_dip),']}']))
            for ii in range(1,Nat):
                f.write("".join([',{',AtomColorMath(AtType[ii]),',Sphere[','{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2]),'},','{:0.3f}'.format(rSphere_dip),']}']))
                #f.write(',{',AtomColorMath(AtType[ii]),',Sphere[{{:0.3f},{:0.3f},{:0.3f}},{:0.3f}]}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2],rSphere))
            for ii in range(len(Bonds)):
                f.write("".join([',{Cylinder[{{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[Bonds[ii,0],0],Coor[Bonds[ii,0],1],Coor[Bonds[ii,0],2]),'},{']))
                f.write("".join(['{:0.3f},{:0.3f},{:0.3f}'.format(Coor[Bonds[ii,1],0],Coor[Bonds[ii,1],1],Coor[Bonds[ii,1],2]),'}},','{:0.3f}'.format(rCylinder_dip),']}']))
                #f.write(',{Cylinder[{{{:0.3f},{:0.3f},{:0.3f}},{{:0.3f},{:0.3f},{:0.3f}}},{:0.3f}]}'.format(Coor[Bonds[ii,0],0],Coor[Bonds[ii,0],1],Coor[Bonds[ii,0],2],Coor[Bonds[ii,1],0],Coor[Bonds[ii,1],1],Coor[Bonds[ii,1],2],rCylinder))
            # pridat vypis dipolu
            for ii in range(Nat):
                f.write("".join([',{Blue,Arrowheads[0.02],Arrow[Tube[{{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2]),'},{']))
                f.write("".join(['{:0.3f},{:0.3f},{:0.3f}'.format(Coor[ii,0]+AtDip[ii,0]*scaleDipole,Coor[ii,1]+AtDip[ii,1]*scaleDipole,Coor[ii,2]+AtDip[ii,2]*scaleDipole),'}},0.06]]}']))
            #f.write('},Boxed->False,Lighting->"Neutral"]')
            f.write('},Boxed->False,ImageSize->800,AxesOrigin -> {0, 0, 0}]')
            f.write("\n")
        
        if is_Dipole:
            #Dipole output
            f.write('Graphics3D[{arrowAxes[10],')
            f.write("".join(['{',AtomColorMath(AtType[0]),',Sphere[','{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[0,0],Coor[0,1],Coor[0,2]),'},','{:0.3f}'.format(rSphere_dip),']}']))
            for ii in range(1,Nat):
                f.write("".join([',{',AtomColorMath(AtType[ii]),',Sphere[','{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2]),'},','{:0.3f}'.format(rSphere_dip),']}']))
                #f.write(',{',AtomColorMath(AtType[ii]),',Sphere[{{:0.3f},{:0.3f},{:0.3f}},{:0.3f}]}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2],rSphere))
            for ii in range(len(Bonds)):
                f.write("".join([',{Cylinder[{{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[Bonds[ii,0],0],Coor[Bonds[ii,0],1],Coor[Bonds[ii,0],2]),'},{']))
                f.write("".join(['{:0.3f},{:0.3f},{:0.3f}'.format(Coor[Bonds[ii,1],0],Coor[Bonds[ii,1],1],Coor[Bonds[ii,1],2]),'}},','{:0.3f}'.format(rCylinder_dip),']}']))
                #f.write(',{Cylinder[{{{:0.3f},{:0.3f},{:0.3f}},{{:0.3f},{:0.3f},{:0.3f}}},{:0.3f}]}'.format(Coor[Bonds[ii,0],0],Coor[Bonds[ii,0],1],Coor[Bonds[ii,0],2],Coor[Bonds[ii,1],0],Coor[Bonds[ii,1],1],Coor[Bonds[ii,1],2],rCylinder))
            # pridat vypis dipolu
            for ii in range(len(DipoleCoor)):
                f.write("".join([',{Blue,Arrowheads[0.02],Arrow[Tube[{{','{:0.3f},{:0.3f},{:0.3f}'.format(DipoleCoor[ii,0],DipoleCoor[ii,1],DipoleCoor[ii,2]),'},{']))
                f.write("".join(['{:0.3f},{:0.3f},{:0.3f}'.format(DipoleCoor[ii,0]+AtDip[ii,0]*scaleDipole,DipoleCoor[ii,1]+AtDip[ii,1]*scaleDipole,DipoleCoor[ii,2]+AtDip[ii,2]*scaleDipole),'}},0.06]]}']))
            #f.write('},Boxed->False,Lighting->"Neutral"]')
            f.write('},Boxed->False,ImageSize->800,AxesOrigin -> {0, 0, 0}]')
            f.write("\n")
        
        if is_AtDipole and is_TrPointCharge:
            f.write('Graphics3D[{arrowAxes[10],')
            f.write("".join(['{ColorData["TemperatureMap",','{:0.3f}'.format((TrCh[0]-min(TrCh))/normTrCh),'],Sphere[','{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[0,0],Coor[0,1],Coor[0,2]),'},','{:0.3f}'.format(rSphere_dip),']}']))
            for ii in range(1,Nat):
                f.write("".join([',{ColorData["TemperatureMap",','{:0.3f}'.format((TrCh[ii]-min(TrCh))/normTrCh),'],Sphere[','{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2]),'},','{:0.3f}'.format(rSphere_dip),']}']))
                #f.write(',{',AtomColorMath(AtType[ii]),',Sphere[{{:0.3f},{:0.3f},{:0.3f}},{:0.3f}]}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2],rSphere))
            for ii in range(len(Bonds)):
                f.write("".join([',{Cylinder[{{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[Bonds[ii,0],0],Coor[Bonds[ii,0],1],Coor[Bonds[ii,0],2]),'},{']))
                f.write("".join(['{:0.3f},{:0.3f},{:0.3f}'.format(Coor[Bonds[ii,1],0],Coor[Bonds[ii,1],1],Coor[Bonds[ii,1],2]),'}},','{:0.3f}'.format(rCylinder_dip),']}']))
                #f.write(',{Cylinder[{{{:0.3f},{:0.3f},{:0.3f}},{{:0.3f},{:0.3f},{:0.3f}}},{:0.3f}]}'.format(Coor[Bonds[ii,0],0],Coor[Bonds[ii,0],1],Coor[Bonds[ii,0],2],Coor[Bonds[ii,1],0],Coor[Bonds[ii,1],1],Coor[Bonds[ii,1],2],rCylinder))
            for ii in range(Nat):
                f.write("".join([',Text[','{:0.3f}'.format(TrCh[ii]),',{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2]),'}]'])) 
            for ii in range(Nat):
                f.write("".join([',{Blue,Arrowheads[0.02],Arrow[Tube[{{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2]),'},{']))
                f.write("".join(['{:0.3f},{:0.3f},{:0.3f}'.format(Coor[ii,0]+AtDip[ii,0]*scaleDipole,Coor[ii,1]+AtDip[ii,1]*scaleDipole,Coor[ii,2]+AtDip[ii,2]*scaleDipole),'}},0.06]]}']))
            f.write('},Boxed->False,ImageSize->800,AxesOrigin -> {0, 0, 0}]')
            f.write("\n")
        
        if is_Dipole and is_TrPointCharge:
            f.write('Graphics3D[{arrowAxes[10],')
            f.write("".join(['{ColorData["TemperatureMap",','{:0.3f}'.format((TrCh[0]-min(TrCh))/normTrCh),'],Sphere[','{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[0,0],Coor[0,1],Coor[0,2]),'},','{:0.3f}'.format(rSphere_dip),']}']))
            for ii in range(1,Nat):
                f.write("".join([',{ColorData["TemperatureMap",','{:0.3f}'.format((TrCh[ii]-min(TrCh))/normTrCh),'],Sphere[','{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2]),'},','{:0.3f}'.format(rSphere_dip),']}']))
                #f.write(',{',AtomColorMath(AtType[ii]),',Sphere[{{:0.3f},{:0.3f},{:0.3f}},{:0.3f}]}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2],rSphere))
            for ii in range(len(Bonds)):
                f.write("".join([',{Cylinder[{{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[Bonds[ii,0],0],Coor[Bonds[ii,0],1],Coor[Bonds[ii,0],2]),'},{']))
                f.write("".join(['{:0.3f},{:0.3f},{:0.3f}'.format(Coor[Bonds[ii,1],0],Coor[Bonds[ii,1],1],Coor[Bonds[ii,1],2]),'}},','{:0.3f}'.format(rCylinder_dip),']}']))
                #f.write(',{Cylinder[{{{:0.3f},{:0.3f},{:0.3f}},{{:0.3f},{:0.3f},{:0.3f}}},{:0.3f}]}'.format(Coor[Bonds[ii,0],0],Coor[Bonds[ii,0],1],Coor[Bonds[ii,0],2],Coor[Bonds[ii,1],0],Coor[Bonds[ii,1],1],Coor[Bonds[ii,1],2],rCylinder))
            for ii in range(Nat):
                f.write("".join([',Text[','{:0.3f}'.format(TrCh[ii]),',{','{:0.3f},{:0.3f},{:0.3f}'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2]),'}]'])) 
            for ii in range(len(DipoleCoor)):
                f.write("".join([',{Blue,Arrowheads[0.02],Arrow[Tube[{{','{:0.3f},{:0.3f},{:0.3f}'.format(DipoleCoor[ii,0],DipoleCoor[ii,1],DipoleCoor[ii,2]),'},{']))
                f.write("".join(['{:0.3f},{:0.3f},{:0.3f}'.format(DipoleCoor[ii,0]+AtDip[ii,0]*scaleDipole,DipoleCoor[ii,1]+AtDip[ii,1]*scaleDipole,DipoleCoor[ii,2]+AtDip[ii,2]*scaleDipole),'}},0.06]]}']))
            f.write('},Boxed->False,ImageSize->800,AxesOrigin -> {0, 0, 0}]')
            f.write("\n")
        
        if is_Potential:
            f.write('dist=Function[{x,y,z,x2,y2,z2},Sqrt[(x-x2)^2+(y-y2)^2+(z-z2)^2]]; \n')
            f.write('Potential=Function[{x,y,z},')
            f.write("".join(['{:0.7f}'.format(TrCh[0]),'/dist[x,y,z,{:0.6f},{:0.6f},{:0.6f}]'.format(Coor[0,0],Coor[0,1],Coor[0,2])]))
            for ii in range(Nat):
                f.write("".join(['+{:0.7f}'.format(TrCh[ii]),'/dist[x,y,z,{:0.6f},{:0.6f},{:0.6f}]'.format(Coor[ii,0],Coor[ii,1],Coor[ii,2])]))
            f.write('];\n')
            f.write('ContourPlot[Potential[0, y, z], {y, -30, 30}, {z, -30, 30},Contours -> 50, ColorFunction -> "TemperatureMap",ClippingStyle -> Automatic, PlotRange -> {-0.2, 0.2},MaxRecursion -> 3,ImageSize->800] \n')
            f.write('SliceContourPlot3D[Potential[x, y, z],"CenterPlanes",{x,-30,30},{y,-10,10},{z,-30,30},Contours->50,ColorFunction->"TemperatureMap",ClippingStyle->Automatic,PlotRange->{-0.5,0.5},MaxRecursion->3,ImageSize->800,AspectRatio->Automatic,BoxRatios->Automatic]\n')            
            f.write('(* For better resolution try MaxRecursion -> 4, change x,y,z according to molecule *) \n')
    f.close()    
        
def AtomColorMath(Atom):
    ''' Function which assign colour to atom type 
    
    Parameters
    ----------
    Atom : string
        Atom name in capital letters (e.g: 'C', 'N', ...)
        
    Returns
    ----------
    Color : string
        Wolfram mathematica color definition
    '''
    
    if Atom=='C':
        return 'Gray'
    elif Atom=='N':
        return 'Blue'
    elif Atom=='O':
        return 'Red'
    elif Atom=='H':
        return 'White' 
    elif Atom=='F':
        return 'Green' 
        
        
def outputComplex2Mead(Complex,grid,outputfolder,typ='Transition'):
    ''' Function which reads list of molecules and output all nessesary files
    for interaction energy calculation with MEAD. MEAD calculates interaction 
    energy between individual chromophores in continuous polarizable dielectric
    medium.
    
    Parameters
    ----------
    Complex : list of QchMolecule type
        List of molecules of QchMolecule type between which interaction energy 
        should be calculated
    grid : list (dimension Ngrid x 2)
        Definition of grid structure for numerical solution of Poisson equation.
        First grid should be large and rough and the last grid should cover all
        pigments and shoul be finest. At lest three grid are recommended for 
        calculation. grid[i]=[number of grid points in every dimension (integer)
        , distance between grid points]. Alowed numbers od grid points are: 
        97,161,257... and dinstances between grid points: powers of 0.25.
        Distance between grid points should be defined in ANGSTROMS.
    outputfolder : string
        Output foldef for files needed for MEAD calculation.
    
    Notes
    ----------
    Running MEAD calculation:
    multiflex -blab3 -nopotats -solrad 1.0 -epsin 1.0 -epssol 2.0 complex > complex.out
    where number after `solrad` is solvatation radius for creating cavity (or more
    precisely solvent accessible arrea) for individual molecules in dielectric,
    number after `epsin` is dielectric permitivity inside cavity (it is suposed to
    be 1.0 - vacuum) and number after `epssol` is solvent dielectric permitivity.
    '''

# TODO: complex would be list of structures and not molecules
    from .interaction import charge_charge, dipole_dipole
    from ..General.units import conversion_facs_position  
    
    outname="".join([outputfolder,'Vacuum_coupling.dat'])
    with open(outname, "wt") as f:
        for ii in range(len(Complex)):
            for jj in range(ii+1,len(Complex)):
                struc1=Complex[ii]
                struc2=Complex[jj]
                
                coor1=struc1.coor.value
                coor2=struc2.coor.value
                
                if typ=='Transition':
                    Einter,dist=charge_charge(coor1,struc1.esp_trans,coor2,struc2.esp_trans,'cm-1')
                elif typ=='Ground':
                    Einter,dist=charge_charge(coor1,struc1.esp_grnd,coor2,struc2.esp_grnd,'cm-1')
                elif typ=='Excited':
                    Einter,dist=charge_charge(coor1,struc1.esp_exct,coor2,struc2.esp_exct,'cm-1')
                
                f.write("TrEsp interaction energy in cm-1 and distance: \n")
                f.write("{:>12d}{:>12d}".format(ii+1,jj+1))
                f.write("{:>20.10f}{:>20.10f}\n".format(Einter,dist))
                if struc1.mass is not None:
                    center1=struc1.get_com().value     # coordinate type
                    center2=struc2.get_com().value     # coordinate type
                else:
                    center1=np.sum(coor1,axis=0)/struc1.nat
                    center2=np.sum(coor2,axis=0)/struc2.nat
                if typ=='Transition':
                    dipole1_ESP=np.dot(struc1.esp_trans,coor1)
                    dipole2_ESP=np.dot(struc2.esp_trans,coor2)
                elif typ=='Ground':
                    dipole1_ESP=np.dot(struc1.esp_grnd,coor1)
                    dipole2_ESP=np.dot(struc2.esp_grnd,coor2)
                elif typ=='Excited':
                    dipole1_ESP=np.dot(struc1.esp_exct,coor1)
                    dipole2_ESP=np.dot(struc2.esp_exct,coor2)
                    
                Einter=dipole_dipole(center1,dipole1_ESP,center2,dipole2_ESP,'cm-1')
                dist=np.sqrt(np.dot(center1-center2,center1-center2))*conversion_facs_position["Angstrom"]
                f.write("Dipole-dipole interaction energy in cm-1 and distance: \n")
                f.write("{:>12d}{:>12d}".format(ii+1,jj+1))
                f.write("{:>20.10f}{:>20.10f}".format(Einter,dist))
    f.close()

    MolName=[]
    for ii in range(len(Complex)):
        MolName.append(Complex[ii].name)
    
    MolNameUnq=np.unique(MolName)
    print(MolNameUnq)
    
    ''' Create molname.st files '''
    total_charge=0.0
    for ii in range(len(MolNameUnq)):
        outname="".join([outputfolder,MolNameUnq[ii],'.st'])
        for jj in range(len(Complex)):
            if Complex[jj].name==MolNameUnq[ii]:
                with open(outname, "wt") as f:
                    # Total charge output
                    f.write("{:7.4f} \n".format(total_charge))
                    # Molname, atomname_unq, charge, 0.000000
                    struc=Complex[jj]
                    for kk in range(struc.nat):
                        f.write(' ')
                        f.write(MolNameUnq[ii])
                        f.write('  ')
                        AtName_unq="".join([struc.at_type[kk],str(kk+1)])
                        f.write("{:6}".format(AtName_unq))
                        if typ=='Transition':
                            f.write("{:>9.6f}{:>11.6f} \n".format(struc.esp_trans[kk],0.0))
                        elif typ=='Ground':
                            f.write("{:>9.6f}{:>11.6f} \n".format(struc.esp_grnd[kk],0.0))
                        elif typ=='Excited':
                            f.write("{:>9.6f}{:>11.6f} \n".format(struc.esp_exct[kk],0.0))
                f.close()
                break
            
    ''' create complex.sites file '''
    outname="".join([outputfolder,'complex.sites'])
    with open(outname, "wt") as f:
        count=0
        for ii in range(len(Complex)):
            if Complex[ii].name!='DUM':
                count+=1
                f.write("{:>4d}".format(count))
                f.write("{:>5} \n".format(Complex[ii].name))
    f.close()
    
    ''' create complex.pqr file '''
    outname="".join([outputfolder,'complex.pqr'])
    with open(outname, "wt") as f:
        with position_units("Angstrom"):
            counter=0
            for ii in range(len(Complex)):
                struc=deepcopy(Complex[ii])
                for jj in range(struc.nat):
                    counter+=1
                    f.write("{:4}".format('ATOM'))
                    f.write("{:>7d}".format(counter))
                    f.write('  ')
                    AtName_unq="".join([struc.at_type[jj],str(jj+1)])
                    f.write("{:6}".format(AtName_unq))
                    f.write("{:5}".format(Complex[ii].name))
                    f.write("{:6d}".format(ii+1))
                    f.write("{:9.3f}{:9.3f}{:9.3f}".format(struc.coor.value[jj,0],struc.coor.value[jj,1],struc.coor.value[jj,2]))
                    f.write("{:>7.3f}{:>8.3f}\n".format(0.000,struc.vdw_rad[jj]))
    f.close()  
    
    
    ''' create file with grig parameters: '''
    outname="".join([outputfolder,'complex.ogm'])
    with open(outname, "wt") as f:
        for ii in range(len(grid)):
            f.write('ON_GEOM_CENT ')
            f.write(str(grid[ii][0]))
            f.write(' ')
            f.write(str(grid[ii][1]))
            f.write('\n')
    f.close
    
    outname="".join([outputfolder,'complex.mgm'])
    with open(outname, "wt") as f:
        for ii in range(-2,0):
            f.write('ON_GEOM_CENT ')
            f.write(str(grid[ii][0]))
            f.write(' ')
            f.write(str(grid[ii][1]))
            f.write('\n')
    f.close

#def outputComplex2TAPBS(Complex,grid,layer_point,layer_nvec,outputfolder,eps_cav=1.0,eps_solv=1.002,eps_layer=2.0,solv_rad=1.4,NZCH=[0,-1],replicate=False,Nrep=2,RepStep=1.0,MolRep='All',finite_layer=False,vecbox=None,verbose=True,typ='Transition',**kwargs):
def outputComplex2TAPBS(Complex,grid,layer_point,layer_nvec,outputfolder,eps_cav=1.0,eps_solv=1.002,eps_layer=2.0,solv_rad=1.4,NZCH=[0,-1],verbose=True,typ='Transition'):
    ''' Function which reads list of molecules and output all nessesary files
    for interaction energy calculation with TAPBS. TAPBS calculates interaction 
    energy between individual chromophores in continuous polarizable dielectric
    medium with membrane. It is alsso possible to define 2D slab dielectric medium.
    
    Parameters
    ----------
    Complex : list of QchMolecule type
        List of molecules of QchMolecule type between which interaction energy 
        should be calculated
    grid : list (dimension Ngrid x 2)
        Definition of grid structure for numerical solution of Poisson equation.
        First grid should be large and rough and the last grid should cover all
        pigments and shoul be finest. At lest three grid are recommended for 
        calculation. grid[i]=[number of grid points in every dimension (integer)
        , distance between grid points]. Alowed numbers od grid points are: 
        97,161,257... and dinstances between grid points: powers of 0.25.
        Distance between grid points should be defined in ANGSTROMS.
    layer_point : numpy array of real (dimension 3)
        Coordinates in ANGSTROMS of point on one side of layer (membrane). Layer
        is defined by point on one side and normal vector pointing inside the 
        membrane with size of membrane thicknes.
    layer_nvec : numpy array of real (dimension 3)
        Normal vector in ANGSTROMS pointing inside the membrane with size of
        membrane thicknes.
    outputfolder : string
        Output foldef for files needed for TAPBS calculation.
    eps_cav : real (optional - init=1.0)
        Relative dielectric permitivity inside a cavity in which molecules are placed.
        By default this cavity should be vaccuum (relative permitivity 1.0)
    eps_solv : real (optional - init=1.002)
        Realtive dielectric permitivity for solvent. For protein and water
        environment it should be around 2.0 for dynamic polarizability. This number
        has to be different than eps_cav. Therefore for using dielectric slab
        surrounded by vacuum this permitivity should be chosen close to 1.0, but
        little bit higher than eps_cav (1.002 should be good value)
    eps_layer real (optional - init=2.0)
        Relative permitivity of membrane layer or dielectric slab. This value 
        has to be different than eps_cav and eps_solv.
    solv_rad : real (optional - init=1.4)
        Solvatation radius for creating cavity (or more precisely solvent 
        accessible arrea) for individual molecules in dielectric.
    NZCH : list of integers (dimension 2-..) (optional - init=[0,-1])
        List of atoms where should be nonzero charges for reference state. For 
        calculation of interaction energy between two chromophores there shoudl
        be zero charges for reference state and transition charges for the state
        in interest. However, TAPBS software does not support all zero charges 
        and therefore at least two atoms has to be chosen to have smallest nonzero charges
        (0.001 and -0.001). Atoms are defined as integers wich refers to atomic
        position in molecule. Atom with positive integer will have positive charge
        and atom with negative integer will have negative charge. These to atoms 
        should be atoms with highest transition chage (to have the smallest deviation
        in interaction energy)
    replicate : logical (optional - init=False)
        If `True` molecule will be replicated in order to create larger cavity 
        in dimension defined by layer normal vector. Atoms of replicated molecule
        will be DUMMY atoms and these molecules are not included in interaction
        energy calculation.
    Nrep : integer (optional - init=2)
        Number of replicated molecules under and above the molecule.        
    RepStep : real (optional - init=1.0)
        Distance in ATOMIC UNITS (Bohr) between individual replicated molecules.
    MolRep : list of integers or string (optional - init='All')
        If `MolRep='All'` for all molecules defined in `Complex` there will be created
        replicas. If `MolRep` is list of integers only molecules with positions
        in `Complex` defined by `MolRep` will be replicated.
    finite_layer : logical (optional - init=False)
        If `False` infinite layer (membrane or 2D dielectric slab) will be used.
        If `True` dummy atoms will be placed arould molecule in order to create 
        semi finite layer (membrane or 2D dielectric slab)
    vecbox : numpy.array of real (dimension 2x3) (optional - init=None)
        If finite layer the two vectors defined by `vecbox` define finite layer 
        borders. Only orientation and not size of the vectors will be used.
        by default box 140x140xthicknes_of_the_layer Bohr^3 will be created.
    verbose : logical (optional - init=True)
        If `True` more information about molecule and TAPBS calculation will
        be printed
    **kwargs : dictionary
        Arguments for creating finite layer.
    
    
        
    Notes
    ----------
    ** RepStep and kwargs using ATOMIC UNITS and grid and layer definition using ANGSTROMS.
    Function should be rewritten to use only ATOMIC UNITS (or at least all the 
    same units) **
    
    '''

    from .interaction import charge_charge, dipole_dipole
    from ..General.units import conversion_facs_position    
    
    outname="".join([outputfolder,'Vacuum_coupling.dat'])
    with open(outname, "wt") as f:
        for ii in range(len(Complex)):
            for jj in range(ii+1,len(Complex)):
                struc1=Complex[ii]
                struc2=Complex[jj]
                
                coor1=struc1.coor.value
                coor2=struc2.coor.value
                
                if typ=='Transition':
                    Einter,dist=charge_charge(coor1,struc1.esp_trans,coor2,struc2.esp_trans,'cm-1')
                elif typ=='Ground':
                    Einter,dist=charge_charge(coor1,struc1.esp_grnd,coor2,struc2.esp_grnd,'cm-1')
                elif typ=='Excited':
                    Einter,dist=charge_charge(coor1,struc1.esp_exct,coor2,struc2.esp_exct,'cm-1')
                
                f.write("TrEsp interaction energy in cm-1 and distance: \n")
                f.write("{:>12d}{:>12d}".format(ii+1,jj+1))
                f.write("{:>20.10f}{:>20.10f}\n".format(Einter,dist))
                if struc1.mass is not None:
                    center1=struc1.get_com().value     # coordinate type
                    center2=struc2.get_com().value     # coordinate type
                else:
                    center1=np.sum(coor1,axis=0)/struc1.nat
                    center2=np.sum(coor2,axis=0)/struc2.nat
                if typ=='Transition':
                    dipole1_ESP=np.dot(struc1.esp_trans,coor1)
                    dipole2_ESP=np.dot(struc2.esp_trans,coor2)
                elif typ=='Ground':
                    dipole1_ESP=np.dot(struc1.esp_grnd,coor1)
                    dipole2_ESP=np.dot(struc2.esp_grnd,coor2)
                elif typ=='Excited':
                    dipole1_ESP=np.dot(struc1.esp_exct,coor1)
                    dipole2_ESP=np.dot(struc2.esp_exct,coor2)
                    
                Einter=dipole_dipole(center1,dipole1_ESP,center2,dipole2_ESP,'cm-1')
                dist=np.sqrt(np.dot(center1-center2,center1-center2))*conversion_facs_position["Angstrom"]
                f.write("Dipole-dipole interaction energy in cm-1 and distance: \n")
                f.write("{:>12d}{:>12d}".format(ii+1,jj+1))
                f.write("{:>20.10f}{:>20.10f}".format(Einter,dist))
    outname="".join([outputfolder,'TrEsp_coupl.dat'])
    
    MolName=[]
    for ii in range(len(Complex)):
        MolName.append(Complex[ii].name)
    
    MolNameUnq=np.unique(MolName)
    if verbose:
        print(MolNameUnq)
    
    ''' Create molname.st files '''
    pK1=0.0
    pK2=0.0
    for ii in range(len(MolNameUnq)):
        outname="".join([outputfolder,MolNameUnq[ii],'.st'])
        for jj in range(len(Complex)):
            if Complex[jj].name==MolNameUnq[ii]:
                struc=Complex[jj]
                with open(outname, "wt") as f:
                    # Total charge output
                    f.write("{:4.2f} pK R \n".format(pK1))
                    # Molname, atomname_unq, charge, 0.000000
                    for kk in range(struc.nat):
                        f.write('ATOM')
                        f.write("{:>7d}".format(kk+1))
                        AtName_unq="".join([struc.at_type[kk],str((kk+1)%99)])
                        f.write("  {:<4}".format(AtName_unq))
                        f.write(MolNameUnq[ii])
                        f.write(' A   1    9999.9999999.9999999.999')
                        if kk in NZCH:
                            f.write("{:>6.3f}".format(0.001))
                        elif -kk in NZCH:
                            f.write("{:>6.3f}".format(-0.001))
                        else:
                            f.write("{:>6.3f}".format(0.0))
                        f.write('99.999      COMP ')
                        f.write(struc.at_type[kk])
                        f.write('\n')
                        
                    f.write("{:4.2f} pK D \n".format(pK2))
                    # Molname, atomname_unq, charge, 0.000000
                    for kk in range(struc.nat):
                        f.write('ATOM')
                        f.write("{:>7d}".format(kk+1))
                        AtName_unq="".join([struc.at_type[kk],str((kk+1)%99)])
                        f.write("  {:<4}".format(AtName_unq))
                        f.write(MolNameUnq[ii])
                        f.write(' A   1    9999.9999999.9999999.999')
                        f.write("{:>6.3f}".format(struc.esp_trans[kk]))
                        f.write('99.999      COMP ')
                        f.write(struc.at_type[kk])
                        f.write('\n')
                f.close()
                break
            
    ''' create complex.sites file '''
    outname="".join([outputfolder,'complex.sites'])
    with open(outname, "wt") as f:
        for ii in range(len(Complex)):
            f.write('COMP')
            f.write("{:>4d}".format(ii+1))
            f.write("{:>4}".format(Complex[ii].name))
            f.write("{:>7} \n".format("".join([Complex[ii].name,'.st'])))
    f.close()
    
    ''' create complex.pqr file '''
    outname="".join([outputfolder,'complex.pqr'])
    with open(outname, "wt") as f:
        counter=0
        for ii in range(len(Complex)):
            struc=Complex[ii]
            for jj in range(struc.nat):
                counter+=1
                f.write("{:4}".format('ATOM'))
                f.write("{:>7d}".format(counter))
                AtName_unq="".join([struc.at_type[jj],str((jj+1)%99)])
                f.write("{:>5}".format(AtName_unq))
                f.write("{:>4}".format(struc.name))
                f.write(' A')
                f.write("{:>4d}".format(ii+1))
                f.write("{:12.3f}{:8.3f}{:8.3f}".format(struc.coor.value[jj,0]*conversion_facs_position["Angstrom"],struc.coor.value[jj,1]*conversion_facs_position["Angstrom"],struc.coor.value[jj,2]*conversion_facs_position["Angstrom"]))            
                f.write("{:>6.3f}{:>6.3f}".format(0.000,struc.vdw_rad[jj]))
                f.write('      COMP \n')
#        if replicate:
#            counter_mol=len(Complex)
#            if MolRep=='All':
#                for ii in range(len(Complex)):
#                    # create dummy atoms
#                    Nvec=np.array(layer_nvec)/np.sqrt(np.dot(layer_nvec,layer_nvec))
#                    if verbose:
#                        print(Nvec)
#                    for kk in range(Nrep):
#                        vec=Nvec*RepStep*(kk+1)
#                        for jj in range(Complex[ii].at_spec['NAtoms']):
#                            counter+=1
#                            f.write("{:4}".format('ATOM'))
#                            f.write("{:>7d}".format(counter))
#                            AtName_unq="".join([Complex[ii].at_spec['AtType'][jj],str(jj+1)])
#                            f.write("{:>5}".format(AtName_unq))
#                            f.write("{:>4}".format('DUM'))
#                            f.write(' A')
#                            f.write("{:>4d}".format(counter_mol+1))
#                            f.write("{:12.3f}{:8.3f}{:8.3f}".format((Complex[ii].at_spec['Coor'][jj,0]+vec[0])*conversion_facs_position["Angstrom"],(Complex[ii].at_spec['Coor'][jj,1]+vec[1])*conversion_facs_position["Angstrom"],(Complex[ii].at_spec['Coor'][jj,2]+vec[2])*conversion_facs_position["Angstrom"]))            
#                            f.write("{:>6.3f}{:>6.3f}".format(0.000,Complex[ii].at_spec['VdW_rad'][jj]))
#                            f.write('      COMP \n')
#                        counter_mol+=1
#                    for kk in range(Nrep):
#                        vec=-Nvec*RepStep*(kk+1)
#                        for jj in range(Complex[ii].at_spec['NAtoms']):
#                            counter+=1
#                            f.write("{:4}".format('ATOM'))
#                            f.write("{:>7d}".format(counter))
#                            AtName_unq="".join([Complex[ii].at_spec['AtType'][jj],str(jj+1)])
#                            f.write("{:>5}".format(AtName_unq))
#                            f.write("{:>4}".format('DUM'))
#                            f.write(' A')
#                            f.write("{:>4d}".format(counter_mol+1))
#                            f.write("{:12.3f}{:8.3f}{:8.3f}".format((Complex[ii].at_spec['Coor'][jj,0]+vec[0])*conversion_facs_position["Angstrom"],(Complex[ii].at_spec['Coor'][jj,1]+vec[1])*conversion_facs_position["Angstrom"],(Complex[ii].at_spec['Coor'][jj,2]+vec[2])*conversion_facs_position["Angstrom"]))            
#                            f.write("{:>6.3f}{:>6.3f}".format(0.000,Complex[ii].at_spec['VdW_rad'][jj]))
#                            f.write('      COMP \n')
#                        counter_mol+=1
#            else:
#                for ii in range(len(MolRep)):
#                    # create dummy atoms
#                    Nvec=np.array(layer_nvec)/np.sqrt(np.dot(layer_nvec,layer_nvec))
#                    if verbose:
#                        print(Nvec)
#                    for kk in range(Nrep):
#                        vec=Nvec*RepStep*(kk+1)
#                        for jj in range(Complex[MolRep[ii]].at_spec['NAtoms']):
#                            counter+=1
#                            f.write("{:4}".format('ATOM'))
#                            f.write("{:>7d}".format(counter))
#                            AtName_unq="".join([Complex[MolRep[ii]].at_spec['AtType'][jj],str(jj+1)])
#                            f.write("{:>5}".format(AtName_unq))
#                            f.write("{:>4}".format('DUM'))
#                            f.write(' A')
#                            f.write("{:>4d}".format(counter_mol+1))
#                            f.write("{:12.3f}{:8.3f}{:8.3f}".format((Complex[MolRep[ii]].at_spec['Coor'][jj,0]+vec[0])*conversion_facs_position["Angstrom"],(Complex[MolRep[ii]].at_spec['Coor'][jj,1]+vec[1])*conversion_facs_position["Angstrom"],(Complex[MolRep[ii]].at_spec['Coor'][jj,2]+vec[2])*conversion_facs_position["Angstrom"]))            
#                            f.write("{:>6.3f}{:>6.3f}".format(0.000,Complex[MolRep[ii]].at_spec['VdW_rad'][jj]))
#                            f.write('      COMP \n')
#                        counter_mol+=1
#                    for kk in range(Nrep):
#                        vec=-Nvec*RepStep*(kk+1)
#                        for jj in range(Complex[MolRep[ii]].at_spec['NAtoms']):
#                            counter+=1
#                            f.write("{:4}".format('ATOM'))
#                            f.write("{:>7d}".format(counter))
#                            AtName_unq="".join([Complex[MolRep[ii]].at_spec['AtType'][jj],str(jj+1)])
#                            f.write("{:>5}".format(AtName_unq))
#                            f.write("{:>4}".format('DUM'))
#                            f.write(' A')
#                            f.write("{:>4d}".format(counter_mol+1))
#                            f.write("{:12.3f}{:8.3f}{:8.3f}".format((Complex[MolRep[ii]].at_spec['Coor'][jj,0]+vec[0])*conversion_facs_position["Angstrom"],(Complex[MolRep[ii]].at_spec['Coor'][jj,1]+vec[1])*conversion_facs_position["Angstrom"],(Complex[MolRep[ii]].at_spec['Coor'][jj,2]+vec[2])*conversion_facs_position["Angstrom"]))            
#                            f.write("{:>6.3f}{:>6.3f}".format(0.000,Complex[MolRep[ii]].at_spec['VdW_rad'][jj]))
#                            f.write('      COMP \n')
#                        counter_mol+=1
#        if finite_layer:
#            '''Parameters for finite layer'''
#            Atomtype='H'
#            xmax=140.0
#            dx=1.4/conversion_facs_position["Angstrom"] # 0.75
#            ymax=140.0
#            dy=1.4/conversion_facs_position["Angstrom"] # 0.75 
#            orient='vec1'
#            
#            is_shiftpype=False
#            is_rpipe=False
#            is_dz=False 
#            is_VdW_rad=False
#            debug=False
#            
#            for key in list(kwargs.keys()):
#                if key=='shiftpipe':
#                    shiftpipe=kwargs['shiftpipe']
#                    is_shiftpype=True
#                elif key=='rpipe':
#                    rpipe=kwargs['rpipe']
#                    is_rpipe=True
#                elif key=='dz':
#                    dz=kwargs['dz']/conversion_facs_position["Angstrom"]
#                    is_dz=True
#                elif key=='VdW_rad':
#                    VdW_rad=kwargs['VdW_rad']
#                    is_VdW_rad=True
#                elif key=='debug':
#                    debug=kwargs['debug']
#                elif key=='xmax':
#                    xmax=kwargs['xmax']
#                elif key=='ymax':
#                    ymax=kwargs['ymax']
#                elif key=='orient':
#                    orient=kwargs['orient']
#            
#            if not is_shiftpype:
#                shiftpipe=1.0
#            if not is_rpipe:
#                rpipe=1.8
#            if not is_dz:
#                dz=0.8/conversion_facs_position["Angstrom"]
#            if not is_VdW_rad:
#                VdW_rad=1.8
#            
#            if verbose:
#                print('xmax:',xmax)
#                print('ymax:',ymax)            
#            
#            ''' Create grid of cavity points '''
#            vec1s=np.sqrt(np.dot(vecbox[0],vecbox[0]))
#            vec2s=np.sqrt(np.dot(vecbox[1],vecbox[1]))
#            vec3s=np.sqrt(np.dot(layer_nvec,layer_nvec))
#            vec1n=np.array(vecbox[0]/vec1s)
#            vec2n=np.array(vecbox[1]/vec2s)
#            vec3n=np.array(layer_nvec/vec3s)            
#
#            if max(np.abs(vec3n))==abs(vec3n[2]):
#                orientation='xy'
#                if verbose:
#                    print('Layer is oriented in xy plane')
#            elif max(np.abs(vec3n))==abs(vec3n[0]):
#                orientation='yz'
#                if verbose:
#                    print('Layer is oriented in yz plane')
#            elif max(np.abs(vec3n))==abs(vec3n[1]):
#                orientation='xz'
#                if verbose:
#                    print('Layer is oriented in xz plane')
#
#            Nx=int(np.ceil(xmax/dx))
#            Ny=int(np.ceil(ymax/dy))
#            Nz=int(np.ceil(vec3s/2.0/dz))
#            
#            #print('vec[0]: ',vecbox[0],vec1n)
#            #print('vector:',vec1n*(vec1s+(1.75+2.0)/conversion_facs_position["Angstrom"]))
#            #print(np.array([vec1n*(vec1s+(1.75+2.0)/conversion_facs_position["Angstrom"]),vec2n*(vec2s+(1.75+2.0)/conversion_facs_position["Angstrom"]),vec3n*(vec3s+(1.75+2.0)/conversion_facs_position["Angstrom"])]).T)
#            
#            #if orientation=='xy':
#            vmat=np.linalg.inv(np.array([vec1n*(vec1s+(1.75+VdW_rad)/conversion_facs_position["Angstrom"]),vec2n*(vec2s+(1.75+VdW_rad)/conversion_facs_position["Angstrom"]),vec3n*(vec3s+(1.75+VdW_rad)/conversion_facs_position["Angstrom"])]).T)
#            #if orientation=='yz':
#            #    vmat=np.linalg.inv(np.array([vec3n*(vec3s+(1.75+2.0)/conversion_facs_position["Angstrom"]),vec2n*(vec2s+(1.75+2.0)/conversion_facs_position["Angstrom"]),vec1n*(vec1s+(1.75+2.0)/conversion_facs_position["Angstrom"])]).T)
#            #if orientation=='xz':
#            #    vmat=np.linalg.inv(np.array([vec1n*(vec1s+(1.75+2.0)*/conversion_facs_position["Angstrom"]),vec3n*(vec3s+(1.75+2.0)/conversion_facs_position["Angstrom"]),vec2n*(vec2s+(1.75+2.0)/conversion_facs_position["Angstrom"])]).T)
#            center=layer_point+layer_nvec/2
#            coor_cavity=[] 
#            
#            counter=0
#            def outside_box(rr,vmat):
#                a=np.dot(vmat,rr.T).T
#                if abs(a[0])>1.0 or abs(a[1])>1.0 or abs(a[2])>1.0:
#                    return True
#                else:
#                    return False
#                    
#            def distance_point_line(point,center,vector):         
#                dist=np.linalg.norm(np.cross(point-center,point-center-vector)) 
#                dist=dist/np.linalg.norm(vector)
#                return dist
#                
#            if orient=='vec1':
#                vec_pipe=np.copy(vec1n)
#            elif orient=='vec2':
#                vec_pipe=np.copy(vec2n)
#                
#            for ii in range(-Nx,Nx+1):
#                for jj in range(-Ny,Ny+1):
#                    for kk in range(-Nz,Nz+1):
#                        # Only for xy rovinu jelikoz z je tu vzdy ve smyslu smeru normaloveho vektoru vrstvy
#                        if orientation=='xy':
#                            rr=np.array([dx*ii,dy*jj,dz*kk])
#                        elif orientation=='yz':
#                            rr=np.array([dz*kk,dy*jj,dx*ii])
#                        elif orientation=='xz':
#                            rr=np.array([dx*ii,dz*kk,dy*jj])
#                        
#                        if outside_box(rr,vmat) and distance_point_line(rr+center,layer_point-shiftpipe/conversion_facs_position["Angstrom"]*vec3n,vec_pipe)>rpipe/conversion_facs_position["Angstrom"]:
#                        #if outside_box(rr,vmat) and distance_point_line(rr+center,layer_point-0.5/conversion_facs_position["Angstrom"]*vec3n,vec1n)>1.8/conversion_facs_position["Angstrom"]:
#                            coor_cavity.append(rr+center)
#                            
#                        if not outside_box(rr,vmat):
#                            counter+=1
#            
#            if verbose:
#                print('Center of excluded space:',center)
#                print('TOTAL EXCLUDED POINTS:',counter)
#            coor_cavity=np.array(coor_cavity)
#            
#            if debug:
#                print("DEBUGING IS ON")
#                # output cavity
#                Nat=len(coor_cavity)
#                AtType=[Atomtype]*Nat
#                OutputToXYZ(coor_cavity*conversion_facs_position["Angstrom"],AtType,filename="".join([outputfolder,'Cavity.xyz']))
#                # output molecule:
#                for ii in range(len(Complex)):
#                    name="".join([outputfolder,'molecule',str(ii),'.xyz'])
#                    OutputToXYZ(Complex[ii].at_spec['Coor']*conversion_facs_position["Angstrom"],Complex[ii].at_spec['AtType'],name)
#                # output layer characteristic:
#                with open("".join([outputfolder,'LayerChar.txt']), "wt") as lch:
#                    lch.write("Layer point: {:12.6f} {:12.6f} {:12.6f}".format(layer_point[0],layer_point[1],layer_point[2]))
#                    lch.write("Layer vector: {:12.6f} {:12.6f} {:12.6f}".format(layer_nvec[0],layer_nvec[1],layer_nvec[2]))
#                lch.close()
#                    
#                
#            
#            ''' Write the outside cavity into the srt file '''
#            for ii in range(len(coor_cavity)//900):
#                # create dummy atoms
#                for jj in range(900): 
#                    counter+=1
#                    f.write("{:4}".format('ATOM'))
#                    f.write("{:>7d}".format(counter))
#                    AtName_unq="".join([Atomtype,str(jj+1)])
#                    f.write("{:>5}".format(AtName_unq))
#                    f.write("{:>4}".format('DUM'))
#                    f.write(' A')
#                    f.write("{:>4d}".format(counter_mol+1))
#                    f.write("{:12.3f}{:8.3f}{:8.3f}".format(coor_cavity[ii*900+jj,0]*conversion_facs_position["Angstrom"],coor_cavity[ii*900+jj,1]*conversion_facs_position["Angstrom"],coor_cavity[ii*900+jj,2]*conversion_facs_position["Angstrom"]))            
#                    #if distance_point_line(coor_cavity[ii*900+jj],center+layer_nvec/2,vec1n)<np.sqrt((dx/2)**2+(dy/2)**2+(dz/2)**2) and (coor_cavity[ii*900+jj,2]==minz+center[2] or coor_cavity[ii*900+jj,2]==maxz+center[2]):
#                    #if coor_cavity[ii*900+jj,2]==minz+center[2] and coor_cavity[ii*900+jj,0]==0.0+center[0]:
#                    #     print('Distance:',distance_point_line(coor_cavity[ii*900+jj],center+layer_nvec/2,vec1n))                         
#                    #     f.write("{:>6.3f}{:>6.3f}".format(0.000,VdW_rad_pipe))
#                    #else:
#                    f.write("{:>6.3f}{:>6.3f}".format(0.000,VdW_rad))
#                    f.write('      COMP \n')
#                counter_mol+=1
#            
#            nn=(len(coor_cavity)//900)*900
#            for jj in range(len(coor_cavity)%900):
#                # create dummy atoms
#                counter+=1
#                f.write("{:4}".format('ATOM'))
#                f.write("{:>7d}".format(counter))
#                AtName_unq="".join([Atomtype,str(jj+1)])
#                f.write("{:>5}".format(AtName_unq))
#                f.write("{:>4}".format('DUM'))
#                f.write(' A')
#                f.write("{:>4d}".format(counter_mol+1))
#                f.write("{:12.3f}{:8.3f}{:8.3f}".format(coor_cavity[nn+jj,0]*conversion_facs_position["Angstrom"],coor_cavity[nn+jj,1]*conversion_facs_position["Angstrom"],coor_cavity[nn+jj,2]*conversion_facs_position["Angstrom"]))            
#                #if distance_point_line(coor_cavity[nn+jj],center+layer_nvec/2,vec1n)<np.sqrt((dx/2)**2+(dy/2)**2+(dz/2)**2):
#                #if distance_point_line(coor_cavity[nn+jj],center+layer_nvec/2,vec1n)<np.sqrt((dx/2)**2+(dy/2)**2+(dz/2)**2) and (coor_cavity[nn+jj,2]==minz+center[2] or coor_cavity[nn+jj,2]==maxz+center[2]):
#                #if coor_cavity[nn+jj,2]==minz+center[2] and coor_cavity[nn+jj,0]==0.0+center[0]:
#                #    print('Distance:',distance_point_line(coor_cavity[nn+jj],center+layer_nvec/2,vec1n))
#                #    f.write("{:>6.3f}{:>6.3f}".format(0.000,VdW_rad_pipe))
#                #else:
#                f.write("{:>6.3f}{:>6.3f}".format(0.000,VdW_rad))
#                f.write('      COMP \n')
#            counter_mol+=1
#            
#            
#            # Find edges of the layer     
#            
#            
    f.close()  
    
    ''' create TAPBS imput file: '''
    outname="".join([outputfolder,'complex_tapbs.in'])
    with open(outname, "wt") as f:
        f.write('pqr complex.pqr \n')
        f.write('sites complex.sites \n')
        f.write('output complex_tapbs \n')
        f.write('#writemap \n')
        f.write(' \n')
        f.write('bcfl sdh \n')
        f.write('pdie {:<9.4f} \n'.format(eps_cav))
        f.write('sdie {:<9.4f} \n'.format(eps_solv))
        f.write('temperature 300 \n')
        f.write('ion 0.0 0.0 0.0 \n')
        f.write('ion 0.0 0.0 0.0 \n')
        f.write('srfm mol \n')
        f.write('srad {:<8.4f} \n'.format(solv_rad))
        f.write('chgm spl2 \n')
        
        # layer definition
        f.write(' \n')
        f.write('membrane {:>11.4f} {:>11.4f} {:>11.4f} {:>11.4f} {:>11.4f} {:>11.4f} {:<9.4f}'.format(layer_point[0],layer_point[1],layer_point[2],layer_nvec[0],layer_nvec[1],layer_nvec[2],eps_layer))
        f.write('\n \n')
        
        # grid definition
        for ii in range(len(grid)):
            f.write('dimension_of_protein_grid ')
            f.write("".join([str(grid[ii][0]),' ',str(grid[ii][0]),' ',str(grid[ii][0])]))
            f.write('\n')
            f.write('spacing_of_protein_grid ')
            f.write("".join([str(grid[ii][1]),' ',str(grid[ii][1]),' ',str(grid[ii][1])]))
            f.write('\n')
            f.write('center_of_protein_grid oncent \n')
            f.write(' \n')
        
        for ii in range(-2,0):
            f.write('dimension_of_model_grid ')
            f.write("".join([str(grid[ii][0]),' ',str(grid[ii][0]),' ',str(grid[ii][0])]))
            f.write('\n')
            f.write('spacing_of_model_grid ')
            f.write("".join([str(grid[ii][1]),' ',str(grid[ii][1]),' ',str(grid[ii][1])]))
            f.write('\n')
            f.write('center_of_model_grid oncent \n')
            f.write(' \n')
        
        # aditional parameters
        f.write('errtol 1E-6 \n')
        f.write('itmax 4 \n')
        f.write('presmooth 2 \n')
        f.write('postsmooth 2 \n')
        f.write('iinfo 0 \n')
        f.write(' \n')
            
    f.close()
    

    
