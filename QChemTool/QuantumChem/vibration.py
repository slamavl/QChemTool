# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:54:57 2016

@author: User
"""

''' Name of the module was changed from vibration2 to vibration '''

import numpy
import matplotlib.pyplot as plt
from itertools import chain, repeat
from copy import deepcopy


from ..General.units import conversion_facs_position,conversion_facs_energy
from ..General.UnitsManager import position_units
from ..QuantumChem.Classes.general import Coordinate
# TODO: Get rid of constants and use only units module
from ..General.constants import InternalToInvcm, Kb
from .output import OutputToPDB, OutputToPDBbuff
from .positioningTools import AlignMolecules, SolveAngle, RotateAndMove_1


# Try to write everything without class Trida (without qc)

debug=False
use_log=False           
            

def NormalModes_mol(mol,verbal=False):
    ''' Function for calculation of normal modes from information stored in 
    QchMolecule type
    
    Parameters
    ----------
    mol : molecule class
        Molecule class molecule with loaded information about hessian inside
    verbal : logical (optional - init=False)
        If true more information during calculation is printed
    
    Returns
    -------
    Freqcm1,RedMass,ForcesCm1Agstrom2,InternalToCartesian,CartesianToInternal,Units
    
    Freqcm1 : numpy.array (dimension=number of normal modes=Nmodes)
        Frequency of normal modes ( = frequency/speed_of_light = nu where nu is 
        wavenumber) in inverse centimeters (actualy wavenumber
        but in gaussian it is denoted as frequency) 
    RedMass : numpy.array (dimension=Nmodes)
        Reduced masses in Atomic Mass Units (AMU) in vector
    ForcesCm1Agstrom2 : numpy.array (dimension=Nmodes)
        Force constants of harmonic modes in inverse centimeters devided by angstrom
        squared (cm^-1/A^2) in vector.
    InternalToCartesian : numpy.array (dimension 3*NatomsxNmodes)
        Matrix where in columns are numal mode displacements in cartesian displacemen
        numpy.dot(InternalToCartesian,numpy.array([1,0,0,0...]))= normal mode 1
        in cartesian coordinates.
        Transform normal mode displacements into cartesian displacements
    CartesianToInternal : numpy.array (dimension Nmodesx3*Natoms)
        Transformation matrix from cartesian(with separated rotational motion)
        to normal mode coordinates.
        Separation of translational motion can be done through AlignMolecules
        in PositioningTools module and then througn SolveAngle and rotation
        
    '''
    
    #qc.Nat -> mol.at_spec['NAtoms'] -> mol.struc.nat
    #qc.at_mass -> mol.at_spec['AtMass'] -> mol.struc.mass
    #qc.at_coord -> mol.at_spec['Coor'] -> mol.struc.coor._value
    #qc.hess_log -> mol.vib_spec['Hessian']
    #qc.hess_fchk -> mol.vib_spec['Hessian']
    
    NAtom=mol.struc.nat
    
    # Fill mass vectors    
    MassVec=numpy.zeros(NAtom*3)
    for ii in range(NAtom*3):
        MassVec[ii]=numpy.sqrt(mol.struc.mass[ii//3])  # MassVec=sqrt(m)
    MassVec1=1/MassVec    
    
    # Calculation of translational vectors for separation of translational motion    
    TransVec=numpy.zeros((3,NAtom*3))        
    for ii in range(NAtom*3):
        TransVec[ii%3,ii]=MassVec[ii]
    if debug:
        print(' Translational vectors:')
        print(TransVec)
    
    # Calculation of center of mass
    MassMat=numpy.zeros((3,NAtom*3))
    for ii in range(NAtom*3):
        MassMat[ii%3,ii]=mol.struc.mass[ii//3]  # squared numbers of TransVec
    Rcom=numpy.zeros(3)    
    AtCoor=mol.struc.coor._value.reshape(NAtom*3)   # atomic coordinates writen in 1D list AtCoor=[x1,y1,z1,x2,y2,z2,x3..]    
    Rcom=numpy.dot(AtCoor,MassMat.T)/numpy.sum(mol.struc.mass)
    if debug:
        print('Center of Mass:')    
        print(Rcom)
    
    '''Calculation of inertia tensor and its eigenvectors for separation of rotational motion '''
    # 1) for this calculation position vectors should be shisted rcom = r-Rcom
    AtCoorCOM=numpy.zeros((NAtom,3))    
    for ii in range(NAtom):
        AtCoorCOM[ii,:]=mol.struc.coor._value[ii,:]-Rcom

    #vis.PlotMolecule(AtCoorCOM,qc.at_info[:,0])    
    
    if debug:
        suma=numpy.zeros(3)    
        for ii in range(NAtom):
            suma[0]+=AtCoorCOM[ii,0]*mol.struc.mass[ii]
            suma[1]+=AtCoorCOM[ii,1]*mol.struc.mass[ii]
            suma[2]+=AtCoorCOM[ii,2]*mol.struc.mass[ii]
        print('suma:')
        print(suma)
        
        print('at_coor in Angstrom')
        for ii in range(NAtom):
            print(["%0.7f" % (float(jj)*conversion_facs_position["Angstrom"]) for jj in mol.struc.coor._value[ii,:]])
        print('at_coor in Bohr')
        for ii in range(NAtom):
            print(["%0.7f" % jj for jj in mol.struc.coor._value[ii,:]])
    
    # 2) Fill inertia tensor
    Itens=numpy.zeros((3,3))
    Itens[0,0]=numpy.dot(mol.struc.mass,numpy.add(numpy.multiply(AtCoorCOM[:,1],AtCoorCOM[:,1]),numpy.multiply(AtCoorCOM[:,2],AtCoorCOM[:,2])))
    Itens[1,1]=numpy.dot(mol.struc.mass,numpy.add(numpy.multiply(AtCoorCOM[:,0],AtCoorCOM[:,0]),numpy.multiply(AtCoorCOM[:,2],AtCoorCOM[:,2])))
    Itens[2,2]=numpy.dot(mol.struc.mass,numpy.add(numpy.multiply(AtCoorCOM[:,0],AtCoorCOM[:,0]),numpy.multiply(AtCoorCOM[:,1],AtCoorCOM[:,1])))    
    for ii in range(3):
        for jj in range(ii):
            Itens[ii,jj]=-numpy.dot(mol.struc.mass,numpy.multiply(AtCoorCOM[:,jj],AtCoorCOM[:,ii]))
            Itens[jj,ii]=Itens[ii,jj]  
            
    # 3) find eigenvectors and eigenvalues of intertia tensor
    tmp,XX=numpy.linalg.eigh(Itens)  # XX = eigenvectors of the Inetria tensor (eigenvectors are in columns)

    if debug:    
        print('Inertia tensor:')    
        print(Itens)
        #XX[:,1]=-XX[:,1]
        print('Inertia tensor eigenvectors:')      
        print(XX)
    if verbal:
        print(' ')
        print('--------------------------------------------------------')
        print('Inertia tensor eigenvalues:') # Can be found in gaussian output file as: Principal axes and moments of inertia in atomic units
        print(["%0.5f" % ii for ii in tmp])
        print('--------------------------------------------------------')
    
    # 4) calculate P matrix and rotational vectors     
    PP=numpy.dot(AtCoorCOM,XX) # NAtom x 3 matrix   # dle navodu v gausianu ma byt P nasobek souradnice a RADKY XX tudiz proto je tam XX.T - da to hodnoty uplne mimo tak necham jen XX
    PPT=numpy.zeros((3,NAtom*3))
    PPT[0,:]= list(chain.from_iterable(repeat(e, 3) for e in PP[:,0]))
    PPT[1,:]= list(chain.from_iterable(repeat(e, 3) for e in PP[:,1]))
    PPT[2,:]= list(chain.from_iterable(repeat(e, 3) for e in PP[:,2]))
    XXT=numpy.zeros((3,NAtom*3))
    XXT[0,:]=list(chain.from_iterable(repeat(XX[:,0], NAtom)))
    XXT[1,:]=list(chain.from_iterable(repeat(XX[:,1], NAtom)))
    XXT[2,:]=list(chain.from_iterable(repeat(XX[:,2], NAtom)))
    if debug:
        print('PP matrix:')      
        print(PP)    
        print('XXT matrix:')      
        #print(XXT)
        print(["%0.4f" % ii for ii in XXT[0,:]])
        print(["%0.4f" % ii for ii in XXT[1,:]])
        print(["%0.4f" % ii for ii in XXT[2,:]])
        print('PPT matrix:')      
        print(["%0.4f" % ii for ii in PPT[0,:]])
        print(["%0.4f" % ii for ii in PPT[1,:]])
        print(["%0.4f" % ii for ii in PPT[2,:]])     
    RotVec=numpy.zeros((3,NAtom*3))
    RotVec[0,:]=numpy.multiply(numpy.multiply(PPT[1,:],XXT[2,:])-numpy.multiply(PPT[2,:],XXT[1,:]),MassVec)
    RotVec[1,:]=numpy.multiply(numpy.multiply(PPT[2,:],XXT[0,:])-numpy.multiply(PPT[0,:],XXT[2,:]),MassVec)
    RotVec[2,:]=numpy.multiply(numpy.multiply(PPT[0,:],XXT[1,:])-numpy.multiply(PPT[1,:],XXT[0,:]),MassVec)
    if debug:
        print('Rotational vectors:')
        print(RotVec)
        print('Orthogonality:')
        print(numpy.dot(RotVec,RotVec.T))
        print(numpy.dot(RotVec,TransVec.T))
    RotVecTMP=numpy.zeros((3,NAtom*3))
    RotVecTMP[0,:]=numpy.multiply(RotVec[0,:],MassVec1)
    RotVecTMP[1,:]=numpy.multiply(RotVec[1,:],MassVec1)
    RotVecTMP[2,:]=numpy.multiply(RotVec[2,:],MassVec1)   
    
    '''' Normalization of translational and rotational vectors '''
    # 1) check if norm of vector is close to zero ( for translation vectors this is not neaded and only one rotational vector could be close to zero)    
    indx=None
    norm=numpy.zeros(3)    
    for ii in range(3):
        norm[ii]=numpy.dot(RotVec[ii,:],RotVec[ii,:])
        if norm[ii]<1E-10:
            indx=ii
    # 2) remove vector wit norm close to zero
    if indx!=None:
        RotVec_Tmp=numpy.zeros(2,3*NAtom)
        counter=0
        for ii in range(3):
            if ii!=indx:
                RotVec_Tmp[counter,:]=RotVec[ii,:]
                counter+=1            
    # 3) normalization Of translational and rotational vectors
    for ii in range(len(RotVec[:,0])):
        RotVec[ii,:]=RotVec[ii,:]/numpy.sqrt(norm[ii])
    normTrans=numpy.dot(TransVec[0,:],TransVec[0,:])
    TransVec=numpy.multiply(TransVec,1/numpy.sqrt(normTrans))

    ''' Eigenvalues before separation of translation and rotation motion - does not have to be calculated - only for comparison with gaussian output'''
    MM1,MM2 = numpy.meshgrid(MassVec1,MassVec1)
    MDiag=numpy.zeros((3*NAtom,3*NAtom))
    for ii in range(3*NAtom):    
        MDiag[ii,ii]=1/numpy.sqrt(mol.struc.mass[ii//3])
    MMM=numpy.zeros((3*NAtom,3*NAtom))
    for ii in range(3*NAtom):    
        MMM[ii,ii]=mol.struc.mass[ii//3]
#    MM=numpy.multiply(MM1,MM2)    # Mass Matrix with matrix elements M[i,j]=1/sqrt(mi*mj)
    
    MHess_fchk=numpy.dot(MDiag,numpy.dot(mol.vib_spec['Hessian'],MDiag)) # Mass weighted cartesian force constants (Mass weighted hessian matrix)
    val_fchk,vec_fchk=numpy.linalg.eigh(MHess_fchk)
    if verbal:
        print('Eigenvalues before sepaartion of translational and rotational motion - force constants from Fchk file:')
    sign=numpy.sign(val_fchk)
    if verbal:
        print(["%0.4f" % ii for ii in numpy.multiply(numpy.sqrt(numpy.abs(val_fchk))*InternalToInvcm,sign)])    
    
    
    ''' Transformation into rotation and translation frame '''
    NTrasRot=3+len(RotVec[:,0])      # number of translational and rotational degrees of freedom
    Basis=numpy.zeros((NAtom*3,3*NAtom+NTrasRot))
    for ii in range(3):
        Basis[:,ii]=TransVec[ii,:]
    for ii in range(len(RotVec[:,0])):
        Basis[:,ii+3]=RotVec[ii,:]
    for ii in range(NAtom*3):
        #Basis[ii,ii+NTrasRot]=1
        Basis[ii,ii+NTrasRot]=mol.struc.mass[ii//3]
    
    DD=numpy.zeros((3*NAtom,3*NAtom-NTrasRot))
    q=numpy.linalg.qr(Basis)[0]          # orthonormalization of basis
    DD=q[:,NTrasRot:3*NAtom]
    MHess_Inter=numpy.zeros((3*NAtom-NTrasRot,3*NAtom-NTrasRot))
    MHess_Inter=numpy.dot(DD.T,numpy.dot(MHess_fchk,DD))
    val,vec=numpy.linalg.eigh(MHess_Inter)
    InternalToCartesianNN=numpy.dot(MDiag,numpy.dot(DD,vec))  # not normalized internal coordinates in cartesian coordinates (first column is dq1 in cartesian coordiantes(dx1,dy1,dz1,dx2..), second column is  dq2 in cartesian coordiantes(dx1,dy1,dz1,dx2..),...)

    if debug:
        print('Basis:')
        print(Basis)    
        print('q matrix:')
        #print(q)
        for ii in range(3*NAtom):        
            print(["%0.3f" % jj for jj in q[ii,:]])
        print('DD matrix (orthonormalized vectors without rotation and translation):')
        #print(DD)
        for ii in range(3*NAtom):        
            print(["%0.3f" % jj for jj in DD[ii,:]])
        print('Normalization of D matrix')
        #print(numpy.dot(DD.T,DD))
        NDD=numpy.dot(DD.T,DD)
        for ii in range(3*NAtom-NTrasRot):        
            print(["%0.2f" % jj for jj in NDD[ii,:]])    
    
    ''' calculation of reduced mass and normalization of eigenvectors '''
    RedMass=numpy.zeros(3*NAtom-NTrasRot)
    for ii in range(3*NAtom-NTrasRot):
        RedMass[ii]=1/numpy.dot(InternalToCartesianNN[:,ii],InternalToCartesianNN[:,ii])
    MatNorm=numpy.zeros((3*NAtom-NTrasRot,3*NAtom-NTrasRot))
    for ii in range(3*NAtom-NTrasRot):
        MatNorm[ii,ii]=numpy.sqrt(RedMass[ii])
    MatNorm1=numpy.zeros((3*NAtom-NTrasRot,3*NAtom-NTrasRot))
    for ii in range(3*NAtom-NTrasRot):
        MatNorm1[ii,ii]=1/numpy.sqrt(RedMass[ii])
    InternalToCartesian=numpy.dot(InternalToCartesianNN,MatNorm)
    CartesianToInternal=numpy.dot(MatNorm1,numpy.dot(vec.T,numpy.dot(DD.T,numpy.sqrt(MMM))))
    if verbal:
        print(' ')
        print(' Final output: ')
        print('---------------------------------------------------------')
        print('Reduced masses:')
        print(RedMass)
        print('Eigenvalues ccalculated from fchk force constants:')
    sign=numpy.sign(val)
    #print(numpy.sqrt(numpy.abs(val))*InternalToInvcm)
    
    if verbal:
        print(["%0.4f" % ii for ii in numpy.multiply(numpy.sqrt(numpy.abs(val))*InternalToInvcm,sign)])     
        #print('eigenvectors:')
        #print(InternalToCartesian)
        print('---------------------------------------------------------')

    if debug:
        print('Reduced Masses:')
        print(RedMass)
        print('Diagonal matrix with eigenvalues:')
        print(numpy.sqrt(numpy.abs(numpy.dot(vec.T,numpy.dot(MHess_Inter,vec))))*InternalToInvcm)
        print('Abs eigenvalues')
        print(numpy.sqrt(numpy.abs(val))*InternalToInvcm)
        print('eigenvectors:')
        numpy.set_printoptions(precision=3)
        print(numpy.dot(InternalToCartesianNN,MatNorm))
        print('tets of eigenvectors')
        print(numpy.sqrt(numpy.abs(numpy.dot(InternalToCartesianNN.T,numpy.dot(mol.vib_spec['Hessian'],InternalToCartesianNN))))*InternalToInvcm)
        print('orthogonality of eigenvectors:')
        print(numpy.dot(InternalToCartesianNN.T,numpy.dot(MMM,InternalToCartesianNN)))    
    
#        MHess_Inter=numpy.zeros((3*NAtom-NTrasRot,3*NAtom-NTrasRot))
#        MHess_Inter=numpy.dot(DD.T,numpy.dot(MHess_log,DD))
#        val2,vec2=numpy.linalg.eigh(MHess_Inter)
#        print('Eigenvalues calculated from hessian matrix from log file')
#        print(numpy.sqrt(numpy.abs(val2))*InternalToInvcm)
    
    if verbal:
        #print('orthogonality of eigenvectors:')
        #print(numpy.dot(InternalToCartesian.T,InternalToCartesian)) 
        print('orthogonality of eigenvectors:')
        #print(numpy.dot(InternalToCartesian.T,numpy.dot(MMM,InternalToCartesian))) 
        print(numpy.dot(CartesianToInternal,InternalToCartesian))

    sign=numpy.sign(val)
    Freqcm1=numpy.multiply(numpy.sqrt(numpy.abs(val))*InternalToInvcm,sign)
    ForcesHaBohr2=numpy.multiply(val,RedMass)
    #print(Freqcm1)
    #print( numpy.sqrt(ForcesHaBohr2/RedMass/1822.8886154)/(2*numpy.pi)*conversion_facs_energy["1/cm"] )
    
    #ForcesCm1Agstrom2=ForcesHaBohr2*const.HaToInvcm/(const.BohrToAngstrom*const.BohrToAngstrom)
    ForcesCm1Agstrom2=ForcesHaBohr2*conversion_facs_energy["1/cm"]/(conversion_facs_position["Angstrom"]**2)
    # conversion_facs_position
    Units = 'frequency[cm-1],Reduced masses[AMU(atomic mass units),Force constats [cm-1/(Angstrom^2)],InterToCart dimensionles expansion coefficients (but in this context trasform iternal coordinate in Angstrom to cartesian displacement in angstrom)]'
    #InternalToCartesian            # transformation matrix from internal coordiantes to cartesian coordinates iverse transformation is simply InterToCart.T
        
    # units are dependent on units of force constant (in this program all transformed in Angstroms and energy to cm-1 and masses are in atomic mass units AMU)
    
    if verbal:
        print(' ')
        print('forces in cm-1/Angstrom^2:')
        print(ForcesCm1Agstrom2)
        print(' ')    
            
    # frequency in gaussian is wavenumber (frequency/speed_of_light)
        
    return Freqcm1,RedMass,ForcesCm1Agstrom2,InternalToCartesian,CartesianToInternal,Units

def Proces_AMBER_Nmodes(AM_geom,AM_Freq,AM_NormalModes,At_Mass):
    ''' Function which proceses AMBER normal mode output and write all
    characteristic and in same units as vibrational analysis from quantum 
    chemistry (function NormalModes_mol) 

    Parameters
    ----------
    AM_geom : numpy.array of real (dimension Natoms x 3)
        Atomic coordinates of molecule in optimal geometry. So far used for
        counting number of atoms in molecule and therefore units are not important.
    AM_Freq : numpy.array of real (dimension 3*Natoms)
        Frequencies of individual normal modes in INVERSE CENTIMETERS 
        (default AMBER output units)
    AM_NormalModes : numpy.array of real (dimension 3*Natoms x 3*Natoms)
        In colulmns there are non-normalized normal modes in cartesian 
        displacements. In order to obtain gaussian transformation matrix
        from internal displacements to cartessian ones it has to be normalized.
        Normalization factor is equal to square root of reduced mass of normal
        mode which can be calculated as: 
        RedMass[i]=1/numpy.dot(AM_NormalModes[:,i],AM_NormalModes[:,i])
        where RedMass[ii] is reduced mass of i-th normal mode. 
        ** This should be checked beause for calculation of reduced mass is 
        used different algirithm ** 
    At_Mass : numpy.array of real (dimension Natoms)
        Vector of atomic mass for all atoms in AMU (atomic mass units) - default
        for both AMBER and GAUSSIAN
        
    Returns
    -------
    Freqcm1,RedMass,ForcesCm1Agstrom2,InternalToCartesian,CartesianToInternal,Units
    
    Freqcm1 : numpy.array (dimension=number of normal modes=Nmodes)
        Frequency of normal modes in inverse centimeters (actualy wavenumber
        but in gaussian it is denoted as frequency) 
    RedMass : numpy.array (dimension=Nmodes)
        Reduced masses in Atomic Mass Units (AMU) in vector
    ForcesCm1Agstrom2 : numpy.array (dimension=Nmodes)
        Force constants of harmonic modes in inverse centimeters devided by angstrom
        squared (cm^-1/A^2) in vector.
    InternalToCartesian : numpy.array (dimension 3*NatomsxNmodes)
        Matrix where in columns are numal mode displacements in cartesian displacemen
        numpy.dot(InternalToCartesian,numpy.array([1,0,0,0...]))= normal mode 1
        in cartesian coordinates.
        Transform normal mode displacements into cartesian displacements
    CartesianToInternal : numpy.array (dimension Nmodesx3*Natoms)
        Transformation matrix from cartesian(with separated rotational motion)
        to normal mode coordinates.
        Separation of translational motion can be done through AlignMolecules
        in PositioningTools module and then througn SolveAngle and rotation     
    
    '''
    NAtom=numpy.shape(AM_geom)[0]    
    MassVec=numpy.zeros(NAtom*3)
    for ii in range(NAtom*3):
        MassVec[ii]=At_Mass[ii//3]  # MassVec=sqrt(m)
    MassVec1=1/MassVec

    # pocet translacnich a rotacnich stupnu volnosti
    NTrasRot=0    
    for ii in range(6):
        if AM_Freq[ii]<0.8:
            NTrasRot+=1
    NModes=3*NAtom-NTrasRot
    
    # vypocet redukovane hmotnosti jednotlivych modu (AMBER vypisuje nenormalizovane koeficienty)
    RedMass=numpy.zeros(NModes)
    Norm=numpy.zeros(3*NAtom)
    for ii in range(3*NAtom):
        Norm[ii]=1/numpy.dot(AM_NormalModes[:,ii],AM_NormalModes[:,ii]) # vypocet redukovane hustoty = norma normalnich modu
        AM_NormalModes[:,ii]=numpy.sqrt(Norm[ii])*AM_NormalModes[:,ii]   # Normalizace normalnich modu
    RedMass=Norm[NTrasRot:3*NAtom]
    
    # vypocet inverzni matice k InternalToCartesian
    NormalModes1=numpy.linalg.inv(AM_NormalModes)    
    
    # Vypis pouze normalnich modu bez translacnich a rotacnich vektoru
    InternalToCartesian=numpy.zeros((3*NAtom,3*NAtom-NTrasRot))
    CartesianToInternal=numpy.zeros((3*NAtom-NTrasRot,3*NAtom))
    InternalToCartesian=AM_NormalModes[:,NTrasRot:3*NAtom]
    CartesianToInternal=NormalModes1[NTrasRot:3*NAtom,:]
    
#    print('CartesianToInternal Norm:')
#    Norm=numpy.zeros(3*NAtom-NTrasRot)
#    for ii in range(3*NAtom-NTrasRot):
#        Norm[ii]=numpy.dot(CartesianToInternal[ii,:],CartesianToInternal[ii,:])
#    print(Norm)
#    
#    print('InternalToCartesian Norm:')
#    for ii in range(3*NAtom-NTrasRot):
#        Norm[ii]=numpy.dot(InternalToCartesian[:,ii],InternalToCartesian[:,ii])
#    print(Norm)
#    
#    print('Red Mass = (norm InternalToCartesian)^-1 :')
#    for ii in range(3*NAtom-NTrasRot):
#        Norm[ii]=1/numpy.dot(InternalToCartesian[:,ii],InternalToCartesian[:,ii])
#    print(Norm)
    

    RedMass=numpy.zeros(3*NAtom-NTrasRot)
    temp=numpy.square(CartesianToInternal)
    RedMass=1/numpy.dot(temp,MassVec1)

    Freqcm1=numpy.zeros(3*NAtom-NTrasRot)

    Freqcm1=AM_Freq[NTrasRot:3*NAtom]
    ForcesHaBohr2=numpy.multiply(numpy.square(Freqcm1),RedMass)/(InternalToInvcm**2)
    #ForcesCm1Agstrom2=ForcesHaBohr2*const.HaToInvcm/(const.BohrToAngstrom**2)
    ForcesCm1Agstrom2=ForcesHaBohr2*conversion_facs_energy["1/cm"]/(conversion_facs_position["Angstrom"]**2)
    
    Units = 'frequency[cm-1],Reduced masses[AMU(atomic mass units),Force constats [cm-1/(Angstrom^2)],InterToCart dimensionles expansion coefficients (but in this context trasform iternal coordinate in Angstrom to cartesian displacement in angstrom)]'
    
    return Freqcm1,RedMass,ForcesCm1Agstrom2,InternalToCartesian,CartesianToInternal,Units
        
    
def MDtoNormalModes(md,struc,CartToInter,indx_center,indx_x,indx_y,NumMin=True,maxiter=5,RefConf=None,Coupling=None,OutputPDB=False,CouplingOffset=15.0):
    ''' Transforms cartesian displacements from MD symulation to normal mode 
    displacements
    
    Parameters
    ----------
    md : MDinfo class
        All information from AMBER molecular dynamics symulation.
    struc : Structure class
        Contains structural information about molecule molecule. For this kind of calculation it is needed only 
        for atomic mass, geometry and atomic types. It could be done more general
        wthout this class.
    indx_center : integer
        Index of atom which will be used as a center for molecule alignment 
        of molecule from MD symulations and reference molecule with optimal 
        geometry. I should be atom close to center of molecule.
    indx_x : integer
        Index of atom which wil together with center atom form one axis for the
        first step of molecular alignment
    indx_y : integer
        Index of atom which wil together with center atom form second amin axis
        for the first step of molecular alignment
    NumMin : logical (optional - init=True)
        If `NumMin=True` numerical procedure will be used for geting rid of 
        translational and rotational displacements in addition to simple alignment
        of main molecular axes defined by `indx_center`, `indx_x` and `indx_y`
    maxiter : integer (optional - init=5)
        Number of step of numerical procedure for separation of translational 
        and rotational motion. Recommended is at least 20 steps.
    RefConf : Coordinate type
        If no reference geometry is loaded quantum chemistry optimal geometry
        will be used for calculation of cartesian displacements (this is not
        recommended because we would compare two different things difference
        between MD and QCH and and displacement due to vibrational motion). If
        reference geometry is loaded (strongly recommended) QCH optimal geometry
        will be used only for align axis of MD minimized molecule (in MD optimal
        geometry) to be the same as for QCH calculation. CHANGES IMPUTED CONFIGURATION
    Coupling : numpy.array of real (dimension md.NStep) (optional - init=None)
        If couplings between two monomers are defined for every timestep of MD 
        simulation, the configurations with high coupling will be analyzed and
        corresponding normal modes to these configurations will be outputed in
        SignificantNM. This could help us to analyze which normal modes are most
        important for interaction of two molecules.
    CouplingOffset : real (optional - init=15.0)
        Number which specifies which couplings are considered to be high (>CouplingOffset)
    
    Returns
    -------
    NormalCoor : numpy.array of real (dimension N_NM x md.NStep)
        In columns there are normal mode displacements in ANGSTROMS for every
        timestep of MD simulation.
    SignificantNM : numpy.array of real (dimension N_NM x N_significant_config) optional
        In columns there are normal mode displacements in ANGSTROMS for 
        configurations with high coupling
    
    NOTES
    ----------
    If Coupling is used to distinguish high coupling normal modes units of 
    Coupling and CouplingOffset have to be the same but it doesn't matter which
    one.
    ** Change units from ANGSTROMS to ATOMIC UNITS ** 
    
    '''
    NNormal=numpy.shape(CartToInter)[0]
    if numpy.shape(CartToInter)[1]!=3*md.NAtom:
        print(numpy.shape(CartToInter)[1],3*md.NAtom)
        raise IOError('Normal modes have to be calculated for same molecule as MD simulation')
    else:
        NAtom=md.NAtom
    NormalCoor=numpy.zeros((NNormal,md.NStep))
    Rcom=struc.get_com()
    IputCoor=struc.coor.value
    for ii in range(NAtom):
        IputCoor[ii,:] -= Rcom.value
    IputCoor=Coordinate(IputCoor)

    if RefConf is not None:
        RefConf._value=AlignMolecules(IputCoor._value,RefConf._value,indx_center,indx_x,indx_y) # Just i case molecule would rotate during MD minimization
        phi,psi,chi=SolveAngle(IputCoor._value,RefConf._value,struc.mass,'minimize',50)
        RefConf._value=RotateAndMove_1(RefConf._value,0.0,0.0,0.0,phi,psi,chi)
        with position_units("Angstrom"):
            OutputToPDB(IputCoor.value,struc.at_type,filename='ImputCoor.pdb')
            OutputToPDB(RefConf.value,struc.at_type,filename='RefCoor.pdb')
        IputCoor=deepcopy(RefConf)
    
    if Coupling is not None:
        PrintSignificantNM=True
    else:
        PrintSignificantNM=False
    SignificantNM=[]     

    if OutputPDB:
        Significant_coor=[]

    proc2=0.0
    for ii in range(md.NStep):
        proc1=numpy.round(ii*10/md.NStep)
        if proc1!=proc2:
            print('Finished',proc1*10,'%')
        proc2=proc1
        # calculate center of mass of molecule from MD
        Rcom=numpy.zeros(3)    
        AtCoor=md.geom[:,:,ii].reshape(3*NAtom)   # atomic coordinates writen in 1D list AtCoor=[x1,y1,z1,x2,y2,z2,x3..]    
        
        # Calculation of center of mass
        MassMat=numpy.zeros((3,NAtom*3))
        for ii in range(NAtom*3):
            MassMat[ii%3,ii]=struc.mass[ii//3]  # squared numbers of TransVec
        Rcom=numpy.dot(AtCoor,MassMat.T)/numpy.sum(struc.mass)
        
        RRCOM=numpy.array(list(chain.from_iterable(repeat(Rcom, NAtom))))
        with position_units("Angstrom"):
            AtCoorCOM=Coordinate( (AtCoor-RRCOM).reshape((NAtom,3)) )
        AtCoorCOM._value=AlignMolecules(IputCoor._value,AtCoorCOM._value,indx_center,indx_x,indx_y)
        if NumMin:
            phi,psi,chi=SolveAngle(IputCoor._value,AtCoorCOM._value,struc.mass,'minimize',maxiter)
            AtCoorCOM._value=RotateAndMove_1(AtCoorCOM._value,0.0,0.0,0.0,phi,psi,chi)
        with position_units("Angstrom"):
            CartDisplacement=(IputCoor.value).reshape(NAtom*3)-(AtCoorCOM.value).reshape(NAtom*3)
        NormalCoor[:,ii]=numpy.dot(CartToInter,CartDisplacement)
        if PrintSignificantNM:
            if Coupling[ii]>=CouplingOffset:
                SignificantNM.append(NormalCoor[:,ii])
                if OutputPDB:
                    with position_units("Angstrom"):
                        Significant_coor.append(AtCoorCOM.value)

#    print(CartDisplacement)
    SignificantNM=numpy.array(SignificantNM).T
    
    print('Warning: Function MDtoNormalModes outputs normal mode displacements in ANGSTROMS as default units')
    print('Warning: Output of SignificantNM was changed to optional insted of default')
    
    if OutputPDB:
        with open('GeomSigCoup.pdb', "wt") as f:
            for ii in range(len(Significant_coor)):
                OutputToPDBbuff(Significant_coor[ii],struc.at_type,ii,f)
    
    if PrintSignificantNM:
        return NormalCoor,SignificantNM
    else:
        return NormalCoor


# TODO: Put these functions into different module
#==============================================================================
# FUNCTIONS FOR VISUALIZATION - SHOULD BE IN SEPARATE MODULE 
#==============================================================================
def plotHistogram(data,db,MinBin,MaxBin):
    Nbins=int(numpy.ceil((MaxBin-MinBin)/db))
    BinList=numpy.zeros(Nbins+1)
    for ii in range(Nbins+1):
        BinList[ii]=MinBin+ii*db
    plt.hist(data, bins=BinList) #(array([0, 2, 1]), array([0, 1, 2, 3]), <a list of 3 Patch objects>)
    plt.show()
    
def plotHistogramAndProb(data,MDnum,db,MinBin,MaxBin,Temp,Force):
    Nbins=int(numpy.ceil((MaxBin-MinBin)/db))
    BinList=numpy.zeros(Nbins)
    Prob=numpy.zeros(Nbins)
    
    for ii in range(Nbins):
        BinList[ii]=MinBin+ii*db
    
    Prob=numpy.sqrt(Force/(2.0*numpy.pi*Kb*Temp))*numpy.exp(-Force*(numpy.square(BinList))/(2.0*Kb*Temp))    
    plt.hist(data, bins=BinList) #(array([0, 2, 1]), array([0, 1, 2, 3]), <a list of 3 Patch objects>)
    plt.plot(BinList,Prob*MDnum*db,'r', lw=3)    
    plt.show()
    
    
#==============================================================================
# TESTS
#==============================================================================
       
'''----------------------- TEST PART --------------------------------'''
if __name__=="__main__":
    print(' ')
    print('TESTS')
    print('-------')
    
    
    ''' Normal modes - read'''
    from Program_Manager.QuantumChem.Classes.molecule import Molecule
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
        
    ''' Normal modes - molecule'''
    Freqcm1,RedMass,ForcesCm1Agstrom2,InternalToCartesian,CartesianToInternal,Units=NormalModes_mol(mol)
    test=True
    for ii in range(len(Freq)):
        if (not isclose(mol.vib_spec['Frequency'][ii],Freq[ii],abs_tol=1e-4)) or \
           (not isclose(mol.vib_spec['RedMass'][ii],RedMass[ii],abs_tol=1e-4)):
               test=False
    
    IdentMat=numpy.identity(mol.vib_spec['Nmodes'])
    IdentMat2=numpy.dot(CartesianToInternal,InternalToCartesian)
    for i in range(mol.vib_spec['Nmodes']):
        for j in range(mol.vib_spec['Nmodes']):
            if (not isclose(IdentMat[i,j],IdentMat2[i,j],abs_tol=1e-6)):
                test=False
            
    if test:
        print('NormalModes_mol      ...    OK')
    else:
        print('NormalModes_mol      ...    Error')
        
    #print(' ')
    #print(RedMass[0]*(Freqcm1[0]**2)*const.Freqcm_1ToForcecmA_2,RedMass[1]*(Freqcm1[1]**2)*const.Freqcm_1ToForcecmA_2)
    #print(ForcesCm1Agstrom2[0],ForcesCm1Agstrom2[1])