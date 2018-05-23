# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:54:57 2016

@author: User
"""

import numpy
import scipy.optimize
from math import isclose
from ..General.units import conversion_facs_position        
        
def RotateAndMove(coord,dx,dy,dz,rotxy,rotxz,rotyz):
    '''Rotate matrix or vector in 3D in positive rotational angle 
    (if right thumb pointing in direction of axes fingers are pointing in 
    positive rotation direction). This function do not change original input
    matrix. First is rotation aroud z axes then around y axes and then around
    y axes. After rotation there is translation.
    
    Parameters
    ----------
    coord : numpy.array of real
        Matrix which we want to rotate. Usualy matrix with atomic positions. 
        This matrix remains unchanged. Dimension of `coord` must be 3xN or Nx3
        where N=1,2,3...
    dx,dy,dz : real
        `dx` resp. `dy` resp. `dz` is translation in direction of axes x 
        resp. y resp. z
    rotxy,rotxz,rotyz : real
        `rotxy` resp. `rotxz` resp. `rotyz` is rotation around z resp. y 
        resp. x axes in positive direction
        
    Returns
    -------
    numpy.array of real
        Same shape as input matrix. If input matrix is vector then output is also vector
    '''
    
    is_vec=False
    if numpy.ndim(coord)==1:
        is_vec=True
        
    TransVec=numpy.array([dx,dy,dz])
    RotA=numpy.zeros((3,3))
    RotB=numpy.zeros((3,3))
    RotC=numpy.zeros((3,3))
    
    # Rotace okolo osy z
    RotA[0,0]=numpy.cos(rotxy)
    RotA[1,0]=numpy.sin(rotxy)
    RotA[1,1]=RotA[0,0]
    RotA[0,1]=-RotA[1,0]
    RotA[2,2]=1.0
    
    # Rotace okolo osy y
    RotB[0,0]=numpy.cos(rotxz)    
    RotB[0,2]=numpy.sin(rotxz)
    RotB[1,1]=1.0
    RotB[2,2]=RotB[0,0]
    RotB[2,0]=-RotB[0,2]
    
    # Rotace okolo osy x
    RotC[1,1]=numpy.cos(rotyz)    
    RotC[2,1]=numpy.sin(rotyz)
    RotC[0,0]=1.0
    RotC[2,2]=RotC[1,1]
    RotC[1,2]=-RotC[2,1]
    
    TransRotCoord=numpy.zeros(numpy.shape(coord),dtype='f8')
    if is_vec:
        TransRotCoord=numpy.dot(RotC,numpy.dot(RotB,numpy.dot(RotA,coord)))
        TransRotCoord=TransRotCoord+TransVec
        return TransRotCoord
    else:
        if numpy.shape(coord)[1]==3:
            TransRotCoord=(numpy.dot(RotC,numpy.dot(RotB,numpy.dot(RotA,coord.T)))).T
            for ii in range(numpy.shape(coord)[0]):
                TransRotCoord[ii,:]=TransRotCoord[ii,:]+TransVec
        elif numpy.shape(coord)[0]==3:
            TransRotCoord=numpy.dot(RotC,numpy.dot(RotB,numpy.dot(RotA,coord)))
            for ii in range(numpy.shape(coord)[1]):
                TransRotCoord[:,ii]=TransRotCoord[:,ii]+TransVec
        if numpy.shape(coord)[1]==3 and numpy.shape(coord)[0]==1:
            return TransRotCoord[0]
        else:
            return TransRotCoord
    
def RotateAndMove_1(coord,dx,dy,dz,rotxy,rotxz,rotyz):
    '''Inverse function to RotateAndMove.
    Rotate matrix or vector in 3D in negative rotational angle 
    (if left thumb pointing in direction of axes fingers are pointing in 
    positive rotation direction). This function do not change original input
    matrix. First is rotation aroud x axes then around y axes and then around
    z axes. After rotation there is translation.
    
    Parameters
    ----------
    coord : numpy.array of real
        Matrix which we want to rotate. Usualy matrix with atomic positions. 
        This matrix remains unchanged. Dimension of `coord` must be 3xN or Nx3
        where N=1,2,3...
    dx,dy,dz : real
        `dx` resp. `dy` resp. `dz` is translation in direction of axes x 
        resp. y resp. z
    rotxy,rotxz,rotyz : real
        `rotxy` resp. `rotxz` resp. `rotyz` is rotation around z resp. y 
        resp. x axes in positive direction
        
    Returns
    -------
    numpy.array of real
        Same shape as input matrix. If input matrix is vector then output is also vector
    '''
    
    TransVec=numpy.array([dx,dy,dz])
    RotA=numpy.zeros((3,3))
    RotB=numpy.zeros((3,3))
    RotC=numpy.zeros((3,3))
    
    # Rotace okolo osy z
    RotA[0,0]=numpy.cos(rotxy)
    RotA[0,1]=numpy.sin(rotxy)
    RotA[1,1]=RotA[0,0]
    RotA[1,0]=-RotA[0,1]
    RotA[2,2]=1.0
    
    # Rotace okolo osy y
    RotB[0,0]=numpy.cos(rotxz)    
    RotB[2,0]=numpy.sin(rotxz)
    RotB[1,1]=1.0
    RotB[2,2]=RotB[0,0]
    RotB[0,2]=-RotB[2,0]
    
    # Rotace okolo osy x
    RotC[1,1]=numpy.cos(rotyz)    
    RotC[1,2]=numpy.sin(rotyz)
    RotC[0,0]=1.0
    RotC[2,2]=RotC[1,1]
    RotC[2,1]=-RotC[1,2]
    
    TransRotCoord=numpy.zeros(numpy.shape(coord))
    if numpy.shape(coord)[1]==3:
        TransRotCoord=(numpy.dot(RotA,numpy.dot(RotB,numpy.dot(RotC,coord.T)))).T
        for ii in range(numpy.shape(coord)[0]):
            TransRotCoord[ii,:]=TransRotCoord[ii,:]+TransVec    
    elif numpy.shape(coord)[0]==3:
        TransRotCoord=numpy.dot(RotA,numpy.dot(RotB,numpy.dot(RotC,coord)))
        for ii in range(numpy.shape(coord)[1]):
            TransRotCoord[:,ii]=TransRotCoord[:,ii]+TransVec
    if numpy.shape(coord)[1]==3 and numpy.shape(coord)[0]==1:
        return TransRotCoord[0]
    else:
        return TransRotCoord

def fill_basis_transf_matrix(typ,rotxy,rotxz,rotyz):
    '''Rotation matrix for atomic orbital expansion coefficients.
    (Positive rotational angle = if right thumb pointing in direction of axes
    fingers are pointing in positive rotation direction).
    
    Parameters
    ----------
    typ : str
        Atomic orbital type. Allowed `typ` are: 's','p','d','5d','f'  
    rotxy,rotxz,rotyz : real
        `rotxy` resp. `rotxz` resp. `rotyz` is rotation around z resp. y 
        resp. x axes in positive direction
        
    Returns
    -------
    RotA,RotB,RotC : numpy.array of real
        `RotA` is transformational matrix for expansion coeficients of given 
        atomic orbital type around z axes.
        `RotB` is transformational matrix for expansion coeficients of given 
        atomic orbital type around y axes.
        `RotC` is transformational matrix for expansion coeficients of given 
        atomic orbital type around x axes.
    '''
    
    if typ=='s':
        RotA=1
        RotB=1
        RotC=1
        
    elif typ=='p':
        RotA=numpy.zeros((3,3),dtype='f8')
        RotB=numpy.zeros((3,3),dtype='f8')
        RotC=numpy.zeros((3,3),dtype='f8')

        # Rotace okolo osy z
        RotA[0,0]=numpy.cos(rotxy)
        RotA[1,0]=numpy.sin(rotxy)
        RotA[1,1]=RotA[0,0]
        RotA[0,1]=-RotA[1,0]
        RotA[2,2]=1.0
        
        # Rotace okolo osy y
        RotB[0,0]=numpy.cos(rotxz)    
        RotB[0,2]=numpy.sin(rotxz)
        RotB[1,1]=1.0
        RotB[2,2]=RotB[0,0]
        RotB[2,0]=-RotB[0,2]
        
        # Rotace okolo osy x
        RotC[1,1]=numpy.cos(rotyz)    
        RotC[2,1]=numpy.sin(rotyz)
        RotC[0,0]=1.0
        RotC[2,2]=RotC[1,1]
        RotC[1,2]=-RotC[2,1]       

    elif typ=='d':
        RotA=numpy.zeros((6,6),dtype='f8')
        RotB=numpy.zeros((6,6),dtype='f8')
        RotC=numpy.zeros((6,6),dtype='f8')
        
        # rotace okolo z
        RotA[0,0]=numpy.cos(rotxy)**2
        RotA[0,1]=numpy.sin(rotxy)**2
        RotA[1,0]=RotA[0,1]
        RotA[1,1]=RotA[0,0]
        RotA[2,2]=1.0
        RotA[0,3]=-numpy.cos(rotxy)*numpy.sin(rotxy)
        RotA[1,3]=-RotA[0,3]
        RotA[3,3]=RotA[0,0]-RotA[0,1]
        RotA[3,1]=2*RotA[0,3]
        RotA[3,0]=-2*RotA[0,3]
        RotA[4,4]=numpy.cos(rotxy)
        RotA[4,5]=-numpy.sin(rotxy)
        RotA[5,4]=-RotA[4,5]
        RotA[5,5]=RotA[4,4]
        
        # rotace okolo y
        RotB[0,0]=numpy.cos(rotxz)**2
        RotB[0,2]=numpy.sin(rotxz)**2
        RotB[0,4]=numpy.cos(rotxz)*numpy.sin(rotxz)
        RotB[1,1]=1.0
        RotB[2,0]=RotB[0,2]
        RotB[2,2]=RotB[0,0]
        RotB[2,4]=-RotB[0,4]
        RotB[3,3]=numpy.cos(rotxz)
        RotB[3,5]=numpy.sin(rotxz)
        RotB[4,0]=-2*RotB[0,4]
        RotB[4,2]=2*RotB[0,4]
        RotB[4,4]=RotB[0,0]-RotB[0,2]
        RotB[5,3]=-RotB[3,5]
        RotB[5,5]=RotB[3,3]
        
        # rotace okolo x
        RotC[0,0]=1
        RotC[1,1]=numpy.cos(rotyz)**2
        RotC[1,2]=numpy.sin(rotyz)**2
        RotC[1,5]=-numpy.cos(rotyz)*numpy.sin(rotyz)
        RotC[2,1]=RotC[1,2]
        RotC[2,2]=RotC[1,1]
        RotC[2,5]=-RotC[1,5]
        RotC[3,3]=numpy.cos(rotyz)
        RotC[3,4]=-numpy.sin(rotyz)
        RotC[4,3]=-RotC[3,4]
        RotC[4,4]=RotC[3,3]
        RotC[5,1]=-2*RotC[1,5]
        RotC[5,2]=2*RotC[1,5]
        RotC[5,5]=RotC[1,1]-RotC[1,2]
        
    elif typ=='5d':
        RotA=numpy.zeros((5,5),dtype='f8')
        RotB=numpy.zeros((5,5),dtype='f8')
        RotC=numpy.zeros((5,5),dtype='f8')
        
        # rotace okolo z
        RotA[0,0]=1.0
        RotA[1,1]=numpy.cos(rotxy)
        RotA[1,2]=-numpy.sin(rotxy)
        RotA[2,1]=-RotA[1,2]
        RotA[2,2]=RotA[1,1]
        RotA[3,3]=numpy.cos(rotxy)**2-numpy.sin(rotxy)**2
        RotA[3,4]=-numpy.cos(rotxy)*numpy.sin(rotxy)
        RotA[4,3]=-4*RotA[3,4]
        RotA[4,4]=RotA[3,3]
        
        # rotace okolo y
        RotB[0,0]=numpy.cos(rotxz)**2-(numpy.sin(rotxz)**2)/2.0
        RotB[0,1]=-numpy.cos(rotxz)*numpy.sin(rotxz)/2.0
        RotB[0,3]=(numpy.sin(rotxz)**2)/2.0
        RotB[1,0]=6*numpy.cos(rotxz)*numpy.sin(rotxz)
        RotB[1,1]=numpy.cos(rotxz)**2-numpy.sin(rotxz)**2
        RotB[1,3]=-2*numpy.cos(rotxz)*numpy.sin(rotxz)
        RotB[2,2]=numpy.cos(rotxz)
        RotB[2,4]=-numpy.sin(rotxz)
        RotB[3,0]=(3*numpy.sin(rotxz)**2)/2.0
        RotB[3,1]=-RotB[0,1]
        RotB[3,3]=(numpy.cos(rotxz)**2+1)/2.0
        RotB[4,2]=-RotB[2,4]
        RotB[4,4]=RotB[2,2]
        
        # rotace okolo x
        RotC[0,0]=numpy.cos(rotyz)**2-(numpy.sin(rotyz)**2)/2.0
        RotC[0,2]=numpy.cos(rotyz)*numpy.sin(rotyz)/2.0
        RotC[0,3]=-(numpy.sin(rotyz)**2)/2.0
        RotC[1,1]=numpy.cos(rotyz)
        RotC[1,4]=numpy.sin(rotyz)
        RotC[2,0]=-6*numpy.cos(rotyz)*numpy.sin(rotyz)
        RotC[2,2]=numpy.cos(rotyz)**2-numpy.sin(rotyz)**2
        RotC[2,3]=-2*numpy.cos(rotyz)*numpy.sin(rotyz)
        RotC[3,0]=-(3*numpy.sin(rotyz)**2)/2.0
        RotC[3,2]=RotC[0,2]
        RotC[3,3]=(numpy.cos(rotyz)**2+1)/2.0
        RotC[4,1]=-RotC[1,4]
        RotC[4,4]=RotC[1,1]
        
    elif typ=='f':
        RotA=numpy.zeros((10,10),dtype='f8')
        RotB=numpy.zeros((10,10),dtype='f8')
        RotC=numpy.zeros((10,10),dtype='f8')
        
        # rotace okolo z
        RotA[0,0]=numpy.cos(rotxy)**3
        RotA[0,1]=-numpy.sin(rotxy)**3
        RotA[0,3]=numpy.cos(rotxy)*numpy.sin(rotxy)**2
        RotA[0,4]=-numpy.sin(rotxy)*numpy.cos(rotxy)**2
        RotA[1,0]=-RotA[0,1]
        RotA[1,1]=RotA[0,0]
        RotA[1,3]=-RotA[0,4]
        RotA[1,4]=RotA[0,3]
        RotA[2,2]=1.0
        RotA[3,0]=3*RotA[0,3]
        RotA[3,1]=3*RotA[0,4]
        RotA[3,3]=RotA[0,0]-2*RotA[0,3]
        RotA[3,4]=RotA[0,1]-2*RotA[0,4]
        RotA[4,0]=-RotA[3,1]
        RotA[4,1]=RotA[3,0]
        RotA[4,3]=-RotA[3,4]
        RotA[4,4]=RotA[3,3]
        RotA[5,5]=numpy.cos(rotxy)**2
        RotA[5,8]=numpy.sin(rotxy)**2
        RotA[5,9]=-numpy.cos(rotxy)*numpy.sin(rotxy)
        RotA[6,6]=numpy.cos(rotxy)
        RotA[6,7]=-numpy.sin(rotxy)
        RotA[7,6]=-RotA[6,7]
        RotA[7,7]=RotA[6,6]
        RotA[8,5]=RotA[5,8]
        RotA[8,8]=RotA[5,5]
        RotA[8,9]=-RotA[5,9]
        RotA[9,5]=-2*RotA[5,9]
        RotA[9,8]=2*RotA[5,9]
        RotA[9,9]=RotA[5,5]-RotA[5,8]

        # rotace okolo y
        RotB[0,0]=numpy.cos(rotxz)**3
        RotB[0,2]=numpy.sin(rotxz)**3
        RotB[0,5]=numpy.sin(rotxz)*numpy.cos(rotxz)**2
        RotB[0,6]=numpy.cos(rotxz)*numpy.sin(rotxz)**2
        RotB[1,1]=1.0
        RotB[2,0]=-RotB[0,2]
        RotB[2,2]=RotB[0,0]
        RotB[2,5]=RotB[0,6]
        RotB[2,6]=-RotB[0,5]
        RotB[3,3]=numpy.cos(rotxz)
        RotB[3,8]=numpy.sin(rotxz)
        RotB[4,4]=numpy.cos(rotxz)**2
        RotB[4,7]=numpy.sin(rotxz)**2
        RotB[4,9]=numpy.sin(rotxz)*numpy.cos(rotxz)
        RotB[5,0]=-3*RotB[0,5]
        RotB[5,2]=3*RotB[0,6]
        RotB[5,5]=RotB[0,0]-2*RotB[0,6]
        RotB[5,6]=2*RotB[0,5]-RotB[0,2]
        RotB[6,0]=RotB[5,2]
        RotB[6,2]=-RotB[5,0]
        RotB[6,5]=-RotB[5,6]
        RotB[6,6]=RotB[5,5]
        RotB[7,4]=RotB[4,7]
        RotB[7,7]=RotB[4,4]
        RotB[7,9]=-RotB[4,9]
        RotB[8,3]=-RotB[3,8]
        RotB[8,8]=RotB[3,3]
        RotB[9,4]=-2*RotB[4,9]
        RotB[9,7]=2*RotB[4,9]
        RotB[9,9]=RotB[4,4]-RotB[4,7]
        
        # rotace okolo x
        RotC[0,0]=1.0
        RotC[1,1]=numpy.cos(rotyz)**3
        RotC[1,2]=-numpy.sin(rotyz)**3
        RotC[1,7]=numpy.cos(rotyz)*numpy.sin(rotyz)**2
        RotC[1,8]=-numpy.sin(rotyz)*numpy.cos(rotyz)**2
        RotC[2,1]=-RotC[1,2]
        RotC[2,2]=RotC[1,1]
        RotC[2,7]=-RotC[1,8]
        RotC[2,8]=RotC[1,7]
        RotC[3,3]=numpy.cos(rotyz)**2
        RotC[3,6]=numpy.sin(rotyz)**2
        RotC[3,9]=-numpy.sin(rotyz)*numpy.cos(rotyz)
        RotC[4,4]=numpy.cos(rotyz)
        RotC[4,5]=-numpy.sin(rotyz)
        RotC[5,4]=-RotC[4,5]
        RotC[5,5]=RotC[4,4]
        RotC[6,3]=RotC[3,6]
        RotC[6,6]=RotC[3,3]
        RotC[6,9]=-RotC[3,9]
        RotC[7,1]=3*RotC[1,7]
        RotC[7,2]=3*RotC[1,8] #
        RotC[7,7]=RotC[1,1]-2*RotC[1,7]
        RotC[7,8]=RotC[1,2]-2*RotC[1,8]
        RotC[8,1]=-RotC[7,2]
        RotC[8,2]=RotC[7,1]
        RotC[8,7]=-RotC[7,8]
        RotC[8,8]=RotC[7,7]
        RotC[9,3]=-2*RotC[3,9]
        RotC[9,6]=2*RotC[3,9]
        RotC[9,9]=RotC[3,3]-RotC[3,6]
        
    elif typ=='7f':
        raise IOError('7f functions are not properly implemented - add 10F and 6D keyword to gaussian route section in order to use cartesian atomic orbitals which are well implemented')
        
    return RotA,RotB,RotC
        
def CenterMolecule(Coor,indx_center,indx_x,indx_y,print_angles=False,debug=False, **kwargs):
    '''Center molecular coordinates in following way. Center will be in origin
    of coordinate system (will have [0.0,0.0,0.0] coordinates) and vector X
    will be pointing into direction of x axes and vector Y will be in xy plane.
    Vector X and Y defined by atomic indexes.
    
    Parameters
    ----------
    coord : numpy.array of real (dimension Nx3)
        Matrix which we want to rotate. Usualy matrix with atomic positions. 
        This matrix remains unchanged. Dimension of `coord` must be Nx3
        where N=1,2,3...
    indx_center : int or list of int
        When `indx_center`=i it refers to atomic coordnitate of ith atom 
        (counted from zero) => center=coor[i,:]. When `indx_center`=[i,j,k,..]
        than center is center of all listed atoms (average coordinate) => 
        center=(coor[i,:]+coor[j,:]+coor[k,:]...)/N
    indx_x : int or list of int of length 2 or 4
        When `indx_x`=i than vector X is defined as Coor[i,:]-center.
        When `indx_x`=[i,j] than vector X is defined as Coor[j,:]-Coor[i,:].
        When `indx_x`=[i,j,k,l] than vector X is defined as 
        (Coor[j,:]-Coor[i,:])+(Coor[l,:]-Coor[k,:]).
    indx_y : int or list of int of length 2 or 4
        When `indx_y`=i than vector Y is defined as Coor[i,:]-center.
        When `indx_y`=[i,j] than vector Y is defined as Coor[j,:]-Coor[i,:].
        When `indx_y`=[i,j,k,l] than vector Y is defined as 
        (Coor[j,:]-Coor[i,:])+(Coor[l,:]-Coor[k,:]).
    print_angles : logical (optional parameter - init value=False)
        If `print_angles`=True also rotation angles and translation vestor
        will be written
    **kwargs : dictionary (optional)
        Provides a way how to set angles and center without specifying atomic 
        indexes. kwargs={"center": vector with coordinates of center,"vec": 
        matrix 2x3 with vector X and Y specified}
        
    Returns
    -------
    Coor_centered(,Phi,Psi,Chi,center)
    Coor_centered : numpy.array of real (dimension Nx3)
        Coordinates of centered molecule. Original coordinates remains unchanged.
    Phi,Psi,Chi : real (optional output)
        Transformation angles which are used to obtain centered molecule. Rotation
        is done throug Coor_rotated=RotateAndMove(Coor,0.0,0.0,0.0,Phi,Psi,Chi)
    center: real (optional output)
        Coordinates of the center of the molecule along which molecule is rotated.
        Molecule has to be firs moved to has its center in origin and then rotated.
    '''
    
    if "center" in kwargs and "vec" in kwargs:
        # Define center and VecX and VecY
        center = kwargs["center"]
        VecX = kwargs["vec"][0]
        VecY = kwargs["vec"][1]
        NAtom=numpy.shape(Coor)[0]
    else:
        if isinstance(indx_center,list):
            listcenter=True
        else:
            listcenter=False
        
        if isinstance(indx_x,list):
            if len(indx_x)==2 or len(indx_x)==4:
                listx=True
            else:
                raise IOError('x axis cannot be defined by more than two indexes')
        else:
            listx=False
            
        if isinstance(indx_y,list):
            if len(indx_y)==2 or len(indx_y)==4:
                listy=True
            else:
                raise IOError('y axis cannot be defined by more than two indexes')
        else:
            listy=False
        
        # Vypocet polohy a uhlu hlavnich smeru
        NAtom=numpy.shape(Coor)[0]
        if listcenter:
            center=numpy.zeros(3)
            for ii in range(len(indx_center)):
                center+=Coor[indx_center[ii],:]
            center=center/len(indx_center)
        else:
            center=Coor[indx_center,:]
            
        if listx and (len(indx_x)==2):
            VecX=Coor[indx_x[1],:]-Coor[indx_x[0],:]
        elif listx and (len(indx_x)==4):
            VecX=Coor[indx_x[1],:]-Coor[indx_x[0],:]+Coor[indx_x[3],:]-Coor[indx_x[2],:]
        else:
            VecX=Coor[indx_x,:]-center
        
        if listy and (len(indx_y)==2):
            VecY=Coor[indx_y[1],:]-Coor[indx_y[0],:]
        elif listy and (len(indx_y)==4):
            VecY=Coor[indx_y[1],:]-Coor[indx_y[0],:]+Coor[indx_y[3],:]-Coor[indx_y[2],:]
        else:
            VecY=Coor[indx_y,:]-center
    
    # Normalization of main axes
    norm=numpy.sqrt(numpy.dot(VecX,VecX))
    VecX=VecX/norm
    norm=numpy.sqrt(numpy.dot(VecY,VecY))
    VecY=VecY/norm
    
    #VecZ je vektorovy soucin
    VecZ=numpy.cross(VecX,VecY)
    norm=numpy.sqrt(numpy.dot(VecZ,VecZ))
    VecZ=VecZ/norm
    
    if debug:
        print('Original vectors:')
        print('     VecX: {:8.4f} {:8.4f} {:8.4f}'.format(VecX[0],VecX[1],VecX[2]))
        print('     VecY: {:8.4f} {:8.4f} {:8.4f}'.format(VecY[0],VecY[1],VecY[2]))
        print('     VecZ: {:8.4f} {:8.4f} {:8.4f}'.format(VecZ[0],VecZ[1],VecZ[2]))

    # VecX musi byt normovany
    if (isclose(VecX[1],0,abs_tol=1e-7) and isclose(VecX[0],0,abs_tol=1e-7)) or isclose(abs(VecX[2]),1.0,abs_tol=1e-4):
        Phi=0.0
    elif isclose(VecX[0],0,abs_tol=1e-4):
        if VecX[1]<0:
            Phi=-numpy.pi/2
        else:
            Phi=numpy.pi/2
    else:
        Phi=numpy.arctan(VecX[1]/VecX[0])

    if (VecX[0]<0) and (not isclose(VecX[0],0,abs_tol=1e-4)):
        Phi=Phi+numpy.pi
    Psi=numpy.arcsin(VecX[2])
    
    if debug:
        print('Rotation around z axis Psi:',Psi/numpy.pi*180)
#
#    Phi=numpy.arctan(VecX[1]/VecX[0])
#    if (VecX[0]<0):
#        Phi=Phi+numpy.pi
#    Psi=numpy.arcsin(VecX[2])
    
    axis=numpy.zeros((3+NAtom,3))
    axis[0,:]=VecX
    axis[1,:]=VecY
    axis[2,:]=VecZ
    for ii in range(NAtom):
        axis[ii+3,:]=Coor[ii,:]-center
#    print('Vektory pred transformaci')
#    print(VecX)
#    print(VecY)
#    print(VecZ)
    axis=RotateAndMove(axis,0.0,0.0,0.0,-Phi,Psi,0.0)
    VecX=axis[0,:]
    VecY=axis[1,:]
    VecZ=axis[2,:]
    
    if debug:
        print('Vectors after rotation around z axis (Psi):')
        print('     VecX: {:8.4f} {:8.4f} {:8.4f}'.format(VecX[0],VecX[1],VecX[2]))
        print('     VecY: {:8.4f} {:8.4f} {:8.4f}'.format(VecY[0],VecY[1],VecY[2]))
        print('     VecZ: {:8.4f} {:8.4f} {:8.4f}'.format(VecZ[0],VecZ[1],VecZ[2]))
    
    if isclose(VecY[1],0,abs_tol=1e-7) and VecY[2]>0.0:
        Chi=numpy.pi/2
    elif isclose(VecY[1],0,abs_tol=1e-7) and VecY[2]<0.0:
        Chi=-numpy.pi/2
    elif VecY[1]>0:
        Chi=numpy.arctan(VecY[2]/VecY[1])
    else:
        Chi=numpy.arctan(VecY[2]/VecY[1])-numpy.pi
    
#    if VecY[1]>0:
#        Chi=numpy.arctan(VecY[2]/VecY[1])
#    else:
#        Chi=numpy.arctan(VecY[2]/VecY[1])-numpy.pi
        
        
        
        
    axis=RotateAndMove(axis,0.0,0.0,0.0,0.0,0.0,-Chi)
    VecX=axis[0,:]
    VecY=axis[1,:]
    VecZ=axis[2,:]
#    print('Podle x y z:')
#    print(VecX)
#    print(VecY)
#    print(VecZ)
    
    #Molecule should be centered and main axes oriented
    Coor_centered=numpy.zeros(Coor.shape)    
    for ii in range(NAtom):
        Coor_centered[ii,:]=axis[ii+3,:]
    
    if print_angles:
        return Coor_centered,-Phi,Psi,-Chi,center
    else:
        return Coor_centered

def AlignMolecules(CoorRef,Coor,indx_center,indx_x,indx_y,indx_center2=None,indx_x2=None,indx_y2=None,print_angles=False):
    ''' Allign two molecules (same or different) resp. their coordinates.
    Atoms for align defined by atomic indexes. For align of the same molecules
    (also same atom ordering) it is enough to definne indexes only for reference
    molecule because for the other one it will be the same)
    
    Parameters
    ----------
    CoorRef : numpy.array of real (dimension Nx3 where N=1,2,3...)
        Matrix of atomic coordinates for reference molecule, this molecule is
        not rotated.
    Coor : numpy.array of real (dimension Nx3 where N=1,2,3...)
        Matrix of atomic coordinates of molecule wich will be aligned with
        reference molecule. This variable is remain unchanged
    indx_center : int or list of int
        When `indx_center`=i it refers to atomic coordnitate of ith atom 
        (counted from zero) => center=coor[i,:]. When `indx_center`=[i,j,k,..]
        than center is center of all listed atoms (average coordinate) => 
        center=(coor[i,:]+coor[j,:]+coor[k,:]...)/N
    indx_x : int or list of int of length 2 or 4
        When `indx_x`=i than vector X is defined as Coor[i,:]-center.
        When `indx_x`=[i,j] than vector X is defined as Coor[j,:]-Coor[i,:].
        When `indx_x`=[i,j,k,l] than vector X is defined as 
        (Coor[j,:]-Coor[i,:])+(Coor[l,:]-Coor[k,:]).
    indx_y : int or list of int of length 2 or 4
        When `indx_y`=i than vector Y is defined as Coor[i,:]-center.
        When `indx_y`=[i,j] than vector Y is defined as Coor[j,:]-Coor[i,:].
        When `indx_y`=[i,j,k,l] than vector Y is defined as 
        (Coor[j,:]-Coor[i,:])+(Coor[l,:]-Coor[k,:]).    
    indx_center2 : int or list of int
        Exacty the same as `indx_center` for refenence molecule but for molecule which will be
        aligned
    indx_x2 : int or list of int
        Exacty the same as `indx_x` for refenence molecule but for molecule which will be
        aligned
    indx_y2 : int or list of int
        Exacty the same as `indx_y` for refenence molecule but for molecule which will be
        aligned
    print_angles : logical
        If true than transformation angles will be printed this way: coor,Phi,Psi,Chi,center.
        If these angles and center will be used for the same transformation on 
        other molecule, the molecule have to be first centered and than rotated
        (not only rotated):
        Coor_cent=CenterMolecule(Coor,indx_center,indx_x,indx_y)
        Coor_aligned=RotateAndMove_1(Coor_cent,center[0],center[1],center[2],Phi,Psi,Chi)
        
    Returns
    -------
    Coor_new : numpy.array of real (dimension Nx3)
        Coordinates of aligned molecule with reference one. Original coordinates remains unchanged.
    Phi,Psi,Chi: real
        Amgles neaded for the same transformation on different molecule
    Center : numpy.array of real (dimension 3)        
        center alligned molecule - needed for the same transformation on different molecule
    
    Notes
    -------
     Center of molecule will have same position as center of reference molecule.
     Vector X of molecule which will be alligned will be pointing
     same direction as vector X of reference molecule. Vector Y of molecule which
     will be alligned will with vector X define same plane as coresponding vectors
     from reference molecule.
     Vector X and Y for both molecules defined by atomic indexes.

    '''
    
    debug=True
    
    if not isinstance(indx_center2,list):
        if indx_center2==None:
            indx_center2=indx_center
            indx_x2=indx_x
            indx_y2=indx_y
    if isinstance(indx_center,list):
        listcenter=True
        if len(indx_center)!=len(indx_center2):
            raise IOError('Indexes for first and second molecule must have same length')
    else:
        listcenter=False
        if isinstance(indx_center2,list):
            raise IOError('Indexes for first and second molecule must have same length')
    if isinstance(indx_x,list):
        if len(indx_x)==2 or len(indx_x)==4:
            listx=True
            if len(indx_x)!=len(indx_x2):
                raise IOError('Indexes for first and second molecule must have same length')
        else:
            raise IOError('x axis cannot be defined by more than two indexes')
    else:
        listx=False
        if isinstance(indx_x2,list):
            raise IOError('Indexes for first and second molecule must have same length')
        
    if isinstance(indx_y,list):
        if len(indx_y)==2 or len(indx_y)==4:
            listy=True
            if len(indx_y)!=len(indx_y2):
                raise IOError('Indexes for first and second molecule must have same length')
        else:
            raise IOError('y axis cannot be defined by more than two indexes')
    else:
        listy=False
        if isinstance(indx_y2,list):
            raise IOError('Indexes for first and second molecule must have same length')
    
    # Vypocet polohy a uhlu hlavnich smeru
    NAtom=numpy.shape(CoorRef)[0]
    if listcenter:
        center=numpy.zeros(3)
        for ii in range(len(indx_center)):
            center+=CoorRef[indx_center[ii],:]
        center=center/len(indx_center)
    else:
        center=CoorRef[indx_center,:]
        
    if listx and (len(indx_x)==2):
        VecX=CoorRef[indx_x[1],:]-CoorRef[indx_x[0],:]
    elif listx and (len(indx_x)==4):
        VecX=CoorRef[indx_x[1],:]-CoorRef[indx_x[0],:]+CoorRef[indx_x[3],:]-CoorRef[indx_x[2],:]
    else:
        VecX=CoorRef[indx_x,:]-center
    
    if listy and (len(indx_y)==2):
        VecY=CoorRef[indx_y[1],:]-CoorRef[indx_y[0],:]
    elif listy and (len(indx_y)==4):
        VecY=CoorRef[indx_y[1],:]-CoorRef[indx_y[0],:]+CoorRef[indx_y[3],:]-CoorRef[indx_y[2],:]
    else:
        VecY=CoorRef[indx_y,:]-center
    
    norm=numpy.sqrt(numpy.dot(VecX,VecX))
    VecX=VecX/norm
    norm=numpy.sqrt(numpy.dot(VecY,VecY))
    VecY=VecY/norm
    
    #VecZ je vektorovy soucin
    VecZ=numpy.cross(VecX,VecY)
    norm=numpy.sqrt(numpy.dot(VecZ,VecZ))
    VecZ=VecZ/norm
    

    # VecX musi byt normovany
    if (isclose(VecX[1],0,abs_tol=1e-7) and isclose(VecX[0],0,abs_tol=1e-7)) or isclose(abs(VecX[2]),1.0,abs_tol=1e-4):
        Phi=0.0
    elif isclose(VecX[0],0,abs_tol=1e-4):
        if VecX[1]<0:
            Phi=-numpy.pi/2
        else:
            Phi=numpy.pi/2
    else:
        Phi=numpy.arctan(VecX[1]/VecX[0])

    if (VecX[0]<0) and (not isclose(VecX[0],0,abs_tol=1e-4)):
        Phi=Phi+numpy.pi
    Psi=numpy.arcsin(VecX[2])
    
    axis=numpy.zeros((3+NAtom,3))
    axis[0,:]=VecX
    axis[1,:]=VecY
    axis[2,:]=VecZ
    for ii in range(NAtom):
        axis[ii+3,:]=CoorRef[ii,:]-center
        
#    print('Vektory pred transformaci')
#    print(VecX)
#    print(VecY)
#    print(VecZ)
    
    axis=RotateAndMove(axis,0.0,0.0,0.0,-Phi,Psi,0.0)

    VecX=axis[0,:]
    VecY=axis[1,:]
    VecZ=axis[2,:]   
    
    if debug:
        if isclose(VecY[1],0,abs_tol=1e-7) and VecY[2]>0.0:
            Chi=numpy.pi/2
        elif isclose(VecY[1],0,abs_tol=1e-7) and VecY[2]<0.0:
            Chi=-numpy.pi/2
        elif VecY[1]>0:
            Chi=numpy.arctan(VecY[2]/VecY[1])
        else:
            Chi=numpy.arctan(VecY[2]/VecY[1])-numpy.pi
    else:
        if VecY[1]>0:
            Chi=numpy.arctan(VecY[2]/VecY[1])
        else:
            Chi=numpy.arctan(VecY[2]/VecY[1])-numpy.pi
    
    #print(Phi/numpy.pi*180,Psi/numpy.pi*180,Chi/numpy.pi*180)
    axis=RotateAndMove(axis,0.0,0.0,0.0,0.0,0.0,-Chi)
    
    VecX=axis[0,:]
    VecY=axis[1,:]
    VecZ=axis[2,:]
    
#    print('Podle x y z:')
#    print(VecX)
#    print(VecY)
#    print(VecZ)
    
    # Nyny otocit molekulu do nuly a pak preorientovat tak jako referencni 
    if listcenter:
        center_tmp=numpy.zeros(3)
        for ii in range(len(indx_center2)):
            center_tmp+=Coor[indx_center2[ii],:]
        center_tmp=center_tmp/len(indx_center2)
    else:
        center_tmp=Coor[indx_center2,:]
    
    if listx and (len(indx_x2)==2):
        VecX_tmp=Coor[indx_x2[1],:]-Coor[indx_x2[0],:]
    elif listx and (len(indx_x2)==4):
        VecX_tmp=Coor[indx_x2[1],:]-Coor[indx_x2[0],:]+Coor[indx_x2[3],:]-Coor[indx_x2[2],:]
    else:
        VecX_tmp=Coor[indx_x2,:]-center_tmp
    
    if listy and (len(indx_y2)==2):
        VecY_tmp=Coor[indx_y2[1],:]-Coor[indx_y2[0],:]
    elif listy and (len(indx_y2)==4):
        VecY_tmp=Coor[indx_y2[1],:]-Coor[indx_y2[0],:]+Coor[indx_y2[3],:]-Coor[indx_y2[2],:]
    else:    
        VecY_tmp=Coor[indx_y2,:]-center_tmp
        
    norm=numpy.sqrt(numpy.dot(VecX_tmp,VecX_tmp))
    VecX_tmp=VecX_tmp/norm
    norm=numpy.sqrt(numpy.dot(VecY_tmp,VecY_tmp))
    VecY_tmp=VecY_tmp/norm
    VecZ_tmp=numpy.cross(VecX_tmp,VecY_tmp)
    norm=numpy.sqrt(numpy.dot(VecZ_tmp,VecZ_tmp))
    VecZ_tmp=VecZ_tmp/norm

    NAtom=numpy.shape(Coor)[0]
    axis=numpy.zeros((3+NAtom,3))
    axis[0,:]=VecX_tmp
    axis[1,:]=VecY_tmp
    axis[2,:]=VecZ_tmp
    for ii in range(NAtom):
        axis[ii+3,:]=Coor[ii,:]-center_tmp
    
#    print('Vektory pred transformaci')
#    print(VecX_tmp)
#    print(VecY_tmp)
#    print(VecZ_tmp)

    if (isclose(VecX_tmp[1],0,abs_tol=1e-7) and isclose(VecX_tmp[0],0,abs_tol=1e-7)) or isclose(abs(VecX_tmp[2]),1.0,abs_tol=1e-4):
        Phi_tmp=0.0
    elif isclose(VecX_tmp[0],0,abs_tol=1e-4):
        if VecX_tmp[1]<0:
            Phi_tmp=-numpy.pi/2
        else:
            Phi_tmp=numpy.pi/2
    else:
        Phi_tmp=numpy.arctan(VecX_tmp[1]/VecX_tmp[0])    
    
    #Phi_tmp=numpy.arctan(VecX_tmp[1]/VecX_tmp[0])
    if (VecX_tmp[0]<0) and (not isclose(VecX_tmp[0],0,abs_tol=1e-4)):
        Phi_tmp=Phi_tmp+numpy.pi
    Psi_tmp=numpy.arcsin(VecX_tmp[2])

    axis=RotateAndMove(axis,0.0,0.0,0.0,-Phi_tmp,Psi_tmp,0.0)
    VecX_tmp=axis[0,:]
    VecY_tmp=axis[1,:]
    VecZ_tmp=axis[2,:]
    
    if debug:
        if isclose(VecY_tmp[1],0,abs_tol=1e-7) and VecY_tmp[2]>0.0:
            Chi_tmp=numpy.pi/2
        elif isclose(VecY_tmp[1],0,abs_tol=1e-7) and VecY_tmp[2]<0.0:
            Chi_tmp=-numpy.pi/2
        elif VecY_tmp[1]>0:
            Chi_tmp=numpy.arctan(VecY_tmp[2]/VecY_tmp[1])
        else:
            Chi_tmp=numpy.arctan(VecY_tmp[2]/VecY_tmp[1])-numpy.pi
    else:
        if VecY_tmp[1]>0:
            Chi_tmp=numpy.arctan(VecY_tmp[2]/VecY_tmp[1])
        else:
            Chi_tmp=numpy.arctan(VecY_tmp[2]/VecY_tmp[1])-numpy.pi

   # print(Phi_tmp/numpy.pi*180,Psi_tmp/numpy.pi*180,Chi_tmp/numpy.pi*180)        
        
    axis=RotateAndMove(axis,0.0,0.0,0.0,0.0,0.0,-Chi_tmp)
    
#    for ii in range(NAtom):
#        print(ii,axis[ii+3,:])    
    
    VecX_tmp=axis[0,:]
    VecY_tmp=axis[1,:]
    VecZ_tmp=axis[2,:]
#    print('Podle x y z:')
#    print(VecX_tmp)
#    print(VecY_tmp)
#    print(VecZ_tmp)
#    print(-Phi_tmp*180/numpy.pi,Psi_tmp*180/numpy.pi,-Chi_tmp*180/numpy.pi)
    axis=RotateAndMove(axis,0.0,0.0,0.0,0.0,0.0,Chi)
    axis=RotateAndMove(axis,0.0,0.0,0.0,0.0,-Psi,0.0)
    axis=RotateAndMove(axis,0.0,0.0,0.0,Phi,0.0,0.0)
#    print('Vektory po transformaci')
#    print(axis[0,:])
#    print(axis[1,:])
#    print(axis[2,:])
    
    
    Coor_new=numpy.zeros(numpy.shape(Coor),dtype='f8')
    for ii in range(NAtom):
        Coor_new[ii,:]=axis[ii+3,:]+center
        #Coor[ii,:]=axis[ii+3,:]+center
    #vis.PlotTwoMolecules(CoorRef,qc.at_info[:,0],Coor,qc.at_info[:,0])
    if print_angles:
        #print(' Molecule has to be first rotated an translated to have center in 0 and vector x must point in x direction and vector y must be in xy plane - With CenterMolecule(Coor,indx_center,indx_x,indx_y)')
        return Coor_new,-Phi,Psi,-Chi,center
    else:
        return Coor_new
        #return Coor

def SolveAngle(coor_ref,coor,mass,method='minimize',maxiter=50,bounds=((-0.5,0.5),(-0.5,0.5),(-0.5,0.5))):
    ''' Function for solving of rotation angles between reference and actual
    configuration. If molecule is rotated than only diference from reference is
    vibrational motion and not totational. Molecules must be shifted to have 
    center of mass at the origin.
    !!! Molecules have to have center of mass in origin !!!
    
    Parameters
    ----------
    coor_ref : numpy.array of real (dimension Nx3 where N=1,2,3...)
        Matrix of atomic coordinates for reference molecule - optimized molecule
    coor : numpy.array of real (dimension Nx3 where)
        Matrix of atomic coordinates for which we would like to obtain rotational
        angles. coor=numpy.array([[x1,y1,z1],[x2,y2,z2], ... ])  
    mass : numpy.array of real (dimension N)
        Vector of atomic weights mass=numpy.array([m1,m2,m3,.. mNAtom])
    method : str (fsolve,minimize)
        fsolve: calculate angles by solving nonlinear equation.
        minimize: calculate angles by minimization of square of the function.
    maxiter : int
        maximal interation number if convergence is not reached (only for method=minimize)
    bounds : tuple
        bounds=((Phi_min,Phi_max),(Psi_min,Psi_max),(Chi_min,Chi_max))
        boundary where we are founding rotational angles with nonlinear equations 
        
    Returns
    ----------
    Phi,Psi,Chi : real
        angles how we have to rotate molecule to get rid of rotational motion
        
    Notes
    ----------
        In order to get rid of rotational motion we have to rotate molecule with:
        RotateAndMove_1(Coor,0.0,0.0,0.0,Phi,Psi,Chi)
    '''            
    
    T=numpy.zeros((3,3))
    for ii in range(3):
        for jj in range(ii+1):
            for kk in range(len(coor_ref[:,0])):
                T[ii,jj]+=mass[kk]*coor_ref[kk,ii]*coor[kk,jj]
            #T[ii,jj]=numpy.sum(coor_ref[:,jj]*coor[:,ii])
            if ii!=jj:            
                T[jj,ii]=T[ii,jj]
#    print(T)                
    def equations(p):
        Phi,Psi,Chi = p
        f1 = -numpy.sin(Psi)*T[1,0] - numpy.cos(Psi)*numpy.sin(Chi)*T[1,1] + numpy.cos(Psi)*numpy.cos(Chi)*T[1,2] - numpy.cos(Psi)*numpy.sin(Phi)*T[2,0] - (numpy.cos(Phi)*numpy.cos(Chi) - numpy.sin(Phi)*numpy.sin(Psi)*numpy.sin(Chi))*T[2,1] - (numpy.cos(Chi)*numpy.sin(Phi)*numpy.sin(Psi) - numpy.cos(Phi)*numpy.sin(Chi))*T[2,2] 
        f2 = numpy.cos(Phi)*numpy.cos(Psi)*T[2,0] + (-numpy.cos(Chi)*numpy.sin(Phi) - numpy.cos(Phi)*numpy.sin(Psi)*numpy.sin(Chi))*T[2,1] + (numpy.cos(Phi)*numpy.cos(Chi)*numpy.sin(Psi) + numpy.sin(Phi)*numpy.sin(Chi))*T[2,2] + numpy.sin(Psi)*T[0,0] + numpy.cos(Psi)*numpy.sin(Chi)*T[0,1] - numpy.cos(Psi)*numpy.cos(Chi)*T[0,2]   
        f3 = numpy.cos(Psi)*numpy.sin(Phi)*T[0,0] + (numpy.cos(Phi)*numpy.cos(Chi) - numpy.sin(Phi)*numpy.sin(Psi)*numpy.sin(Chi))*T[0,1] + (numpy.cos(Chi)*numpy.sin(Phi)*numpy.sin(Psi) - numpy.cos(Phi)*numpy.sin(Chi))*T[0,2] - numpy.cos(Phi)*numpy.cos(Psi)*T[1,0] - (-numpy.cos(Chi)*numpy.sin(Phi) - numpy.cos(Phi)*numpy.sin(Psi)*numpy.sin(Chi))*T[1,1] - (numpy.cos(Phi)*numpy.cos(Chi)*numpy.sin(Psi) + numpy.sin(Phi)*numpy.sin(Chi))*T[1,2]    
        return (f1,f2,f3)
    def equations2(p):
        Phi,Psi,Chi = p
        f1 = -numpy.sin(Psi)*T[1,0] - numpy.cos(Psi)*numpy.sin(Chi)*T[1,1] + numpy.cos(Psi)*numpy.cos(Chi)*T[1,2] - numpy.cos(Psi)*numpy.sin(Phi)*T[2,0] - (numpy.cos(Phi)*numpy.cos(Chi) - numpy.sin(Phi)*numpy.sin(Psi)*numpy.sin(Chi))*T[2,1] - (numpy.cos(Chi)*numpy.sin(Phi)*numpy.sin(Psi) - numpy.cos(Phi)*numpy.sin(Chi))*T[2,2] 
        f2 = numpy.cos(Phi)*numpy.cos(Psi)*T[2,0] + (-numpy.cos(Chi)*numpy.sin(Phi) - numpy.cos(Phi)*numpy.sin(Psi)*numpy.sin(Chi))*T[2,1] + (numpy.cos(Phi)*numpy.cos(Chi)*numpy.sin(Psi) + numpy.sin(Phi)*numpy.sin(Chi))*T[2,2] + numpy.sin(Psi)*T[0,0] + numpy.cos(Psi)*numpy.sin(Chi)*T[0,1] - numpy.cos(Psi)*numpy.cos(Chi)*T[0,2]   
        f3 = numpy.cos(Psi)*numpy.sin(Phi)*T[0,0] + (numpy.cos(Phi)*numpy.cos(Chi) - numpy.sin(Phi)*numpy.sin(Psi)*numpy.sin(Chi))*T[0,1] + (numpy.cos(Chi)*numpy.sin(Phi)*numpy.sin(Psi) - numpy.cos(Phi)*numpy.sin(Chi))*T[0,2] - numpy.cos(Phi)*numpy.cos(Psi)*T[1,0] - (-numpy.cos(Chi)*numpy.sin(Phi) - numpy.cos(Phi)*numpy.sin(Psi)*numpy.sin(Chi))*T[1,1] - (numpy.cos(Phi)*numpy.cos(Chi)*numpy.sin(Psi) + numpy.sin(Phi)*numpy.sin(Chi))*T[1,2]    
        return numpy.dot([f1,f2,f3],[f1,f2,f3])
        
    def equations3(p):
        Phi,Psi,Chi = p
        f=numpy.zeros(3)
        RotA=numpy.zeros((3,3))
        RotB=numpy.zeros((3,3))
        RotC=numpy.zeros((3,3))
        
        # Rotace okolo osy z
        RotA[0,0]=numpy.cos(Phi)
        RotA[1,0]=numpy.sin(Phi)
        RotA[1,1]=RotA[0,0]
        RotA[0,1]=-RotA[1,0]
        RotA[2,2]=1.0
        
        # Rotace okolo osy y
        RotB[0,0]=numpy.cos(Psi)    
        RotB[0,2]=numpy.sin(Psi)
        RotB[1,1]=1.0
        RotB[2,2]=RotB[0,0]
        RotB[2,0]=-RotB[0,2]
        
        # Rotace okolo osy x
        RotC[1,1]=numpy.cos(Chi)    
        RotC[2,1]=numpy.sin(Chi)
        RotC[0,0]=1.0
        RotC[2,2]=RotC[1,1]
        RotC[1,2]=-RotC[2,1]
    
        CC=numpy.dot(RotA,numpy.dot(RotB,RotC))
        for ii in range(len(coor_ref[:,0])):
            f+=mass[ii]*numpy.cross(coor_ref[ii,:],numpy.dot(CC,coor[ii,:]))
        return numpy.dot(f,f)
    
#    print(equations3((0.0,0.0,0.0)))
    #print(equations((numpy.pi/2,0.0,0.0)))
    if method=='fsolve':    
        Phi,Psi,Chi =  scipy.optimize.fsolve(equations,(1.6,1.6,1.6))
    elif method=='minimize':
        result=scipy.optimize.minimize(equations3,(0.0,0.0,0.0),bounds=bounds,options={'maxiter':maxiter})
        Phi,Psi,Chi=result.x
#    print(result)    
    #Phi,Psi,Chi =  scipy.optimize.root(equations,(1.6,1.0,1.0))
    #print(scipy.optimize.root(equations,(1.6,0.0,0.0)))
#    print('output from nonlinear rotation solving')
#    print(equations3((Phi,Psi,Chi)))
#    print(Phi,Psi,Chi)
    return -Phi,-Psi,-Chi

def MergeMolecules(coord1,AtType1,coord2,AtType2):
    ''' Merge two molecules (atomic positions and types) into one.
    
    Parameters
    ----------
    coord1 : numpy.array of real (dimension Nx3 where N=1,2,3...)
        Matrix of atomic coordinates for first molecule
    AtType1 : numpy.array or list of str (dimension N)
        List of atomic types for all atoms in first molecule
    coord2 : numpy.array of real (dimension Mx3 where M=1,2,3...)
        Matrix of atomic coordinates for second molecule
    AtType2 : numpy.array or list of str (dimension M)
        List of atomic types for all atoms in second molecule
        
    Returns
    -------
    MergedCoord: numpy.array of real (dimension N+Mx3)
        Coordinates of merged molecule. Original coordinates for both molecules
        remains unchanged.
    MergedType: numpy.array or list of str (dimension N+M)
        List of atomic types for all atoms in merged molecule.
    '''
    
    NAtom1=len(AtType1)
    NAtom2=len(AtType2)
    MergedType=[]
    if numpy.shape(coord1)[1]==3 and numpy.shape(coord2)[1]==3:
        MergedCoord=numpy.zeros((NAtom1+NAtom2,3))
        for ii in range(NAtom1):
            MergedCoord[ii,:]=coord1[ii,:]
            MergedType.append(AtType1[ii])
        for ii in range(NAtom2):
            MergedCoord[ii+NAtom1,:]=coord2[ii,:]
            MergedType.append(AtType2[ii])
    elif numpy.shape(coord1)[0]==3 and numpy.shape(coord2)[0]==3:
        MergedCoord=numpy.zeros((3,NAtom1+NAtom2))
        for ii in range(NAtom1):
            MergedCoord[:,ii]=coord1[:,ii]
            MergedType.append(AtType1[ii])
        for ii in range(NAtom2):
            MergedCoord[:,ii+NAtom1]=coord2[:,ii]
            MergedType.append(AtType2[ii])
    else:
        raise IOError('Coordinates of both molecules must have same shape')
            
    return MergedCoord,MergedType  

def project_on_plane(coor,nvec,origin):
    ''' Prject all coordinates on plane defined by normal vector nvec and
    origin 
    
    Parameters
    ----------
    coor : numpy.array of real (dimension Nx3 where N=1,2,3...)
        Matrix of atomic coordinates
    nvec : numpy.array or list of real (dimension 3)
        normal vector of plane to which we would like to project
    origin : numpy.array or list of real (dimension 3)
        Point which define plane (point in plane)
        
    Returns
    -------
    ProjCoor: numpy.array of real (dimension Nx3)
        Projected coordiantes into plane
        
    Notes
    -------
        Original coordinates remains unchanged.
    '''
    
    ProjCoor=numpy.zeros((len(coor),3),dtype='f8')
    # projection of point into plane defined by normal vector layer_nvec. Plane defined by ax+by+cz+d=0
    # 1) normalize normal vector to the surface
    nvec=nvec/numpy.sqrt(numpy.dot(nvec,nvec))
    # 2) find distance from surface for every point. DIST=(numpy.dot(nvec,coor)+d)/numpy.dot(nvec,nvec) but for normalized nvec numpy.dot(nvec,nvec)=1
    #a=nvec[0]
    #b=nvec[1]
    #c=nvec[2]
    d=-numpy.dot(nvec,origin)
    for ii in range(len(coor)):
        dist=numpy.dot(nvec,coor[ii])+d
        # 3) move point to the plane
        ProjCoor[ii,:]=coor[ii,:]-dist*nvec
    
    return ProjCoor


def fit_plane(coor):
    ''' Find normal vector of plane and point in plane, which best fit the 
    specified coordinates. 
    
    Parameters
    ----------
    coor : numpy.array of real (dimension Nx3 where N=1,2,3...)
        Matrix of atomic coordinates
        
    Returns
    -------
     nvec : numpy.array or list of real (dimension 3)
        normal vector of plane to which we would like to project
    origin : numpy.array or list of real (dimension 3)
        Point which define plane (point in plane)
        
    Notes
    -------
        Original coordinates remains unchanged.
    '''
    
    # calculate variance in every dimension:
    variance = numpy.var(coor,axis=0)
    indx = numpy.argmin(variance) # in this dimension there will be highest normal vectorcontribution
    
    # setup matrix for parameters calculation
    B = -coor[:,indx]
    A = coor.copy()
    A[:,indx] = 1.0
    
    # solve the equations for the plane
    A_inv = numpy.dot( numpy.linalg.inv(numpy.dot(A.T,A)), A.T)
    vec = numpy.dot(A_inv,B)
    
    # obtain normal vector and one point in the plane
    d = vec[indx]
    vec[indx] = 1.0
    origin = numpy.zeros(3,dtype='f8')
    origin[indx] = -d
    norm = numpy.linalg.norm(vec)
    nvec = vec/norm
    
    return nvec, origin

def prepare_alkene(Dim,Position=numpy.array([0.0,0.0,0.0]),vec_x=numpy.array([1.0,0.0,0.0]),vec_y=numpy.array([0.0,1.0,0.0])):
    ''' Function to generate carbon position for model alkene
    
    Parameters
    ----------
    Dim : int
        Number of atoms in alkene chain
    Position : numpy.array of real (dimension 3)
        Coordinates of alkene center
    vec_x : numpy.array of real (dimension 3)
        Cabron chain will be oriented in `vec_x` direction
    vec_y : numpy.array of real (dimension 3)
        Whole carbon chain will be in `vec_x vec_y` plane
        
    Returns
    -------
    Coor : numpy.array of real (dimension Dimx3 )
        Matrix wit atomic coordinates of all alkene carbons 
    '''
    
    alpha = 120.0/180*scipy.pi
    cc_bond = 1.4/conversion_facs_position["Angstrom"] 
    Coor=numpy.zeros((Dim,3),dtype='f8')
    for ii in range(Dim):
        Coor[ii,2] = 0.0
        if ii%2 == 1:
            Coor[ii,1]= 0.0
        else:
            Coor[ii,1] = cc_bond*numpy.cos(alpha/2)
        Coor[ii,0] = cc_bond*numpy.sin(alpha/2)*ii
    
    # Center molecule
    center = numpy.sum(Coor,0)
    center=center/Dim
    Coor=RotateAndMove(Coor,-center[0],-center[1],-center[2],0.0,0.0,0.0)
    
    # Rotate molecule
    VecX=vec_x
    VecY=vec_y

    norm=numpy.sqrt(numpy.dot(VecX,VecX))
    VecX=VecX/norm
    norm=numpy.sqrt(numpy.dot(VecY,VecY))
    VecY=VecY/norm

    #VecZ je vektorovy soucin
    VecZ=numpy.cross(VecX,VecY)
    norm=numpy.sqrt(numpy.dot(VecZ,VecZ))
    VecZ=VecZ/norm

    # VecX musi byt normovany
    if isclose(VecX[1],0,abs_tol=1e-7) and isclose(VecX[0],0,abs_tol=1e-7):
        Phi=0.0
    else:
        Phi=numpy.arctan(VecX[1]/VecX[0])
    
    if (VecX[0]<0):
        Phi=Phi+numpy.pi
    Psi=numpy.arcsin(VecX[2])

    axis=numpy.zeros((3,3))
    axis[0,:]=VecX
    axis[1,:]=VecY
    axis[2,:]=VecZ

    axis=RotateAndMove(axis,0.0,0.0,0.0,-Phi,Psi,0.0)

    VecX=axis[0,:]
    VecY=axis[1,:]
    VecZ=axis[2,:]
    
    if isclose(VecY[1],0,abs_tol=1e-7) and VecY[2]>0.0:
        Chi=numpy.pi/2
    elif isclose(VecY[1],0,abs_tol=1e-7) and VecY[2]<0.0:
        Chi=-numpy.pi/2
    elif VecY[1]>0:
        Chi=numpy.arctan(VecY[2]/VecY[1])
    else:
        Chi=numpy.arctan(VecY[2]/VecY[1])-numpy.pi

#    print(Phi/numpy.pi*180,Psi/numpy.pi*180,Chi/numpy.pi*180)        
    
    Coor=RotateAndMove(Coor,0.0,0.0,0.0,0.0,0.0,Chi)
    Coor=RotateAndMove(Coor,0.0,0.0,0.0,0.0,-Psi,0.0)
    Coor=RotateAndMove(Coor,0.0,0.0,0.0,Phi,0.0,0.0)
    
    # Move molecule to its position
    Coor=RotateAndMove(Coor,Position[0],Position[1],Position[2],0.0,0.0,0.0)
    
    return Coor
    
def prepare_carotenoid(Dim,Position=numpy.array([0.0,0.0,0.0]),vec_x=numpy.array([1.0,0.0,0.0]),vec_y=numpy.array([0.0,1.0,0.0])):
    ''' Function to generate carbon position for model carotenoid
    
    Parameters
    ----------
    Dim : int
        Number of atoms in carotenoid
    Position : numpy.array of real (dimension 3)
        Coordinates of carotenoid center
    vec_x : numpy.array of real (dimension 3)
        Cabron chain will be oriented in `vec_x` direction
    vec_y : numpy.array of real (dimension 3)
        Whole carbon chain will be in `vec_x vec_y` plane
        
    Returns
    -------
    Coor : numpy.array of real (dimension Dimx3 )
        Matrix wit atomic coordinates of all carotenoid carbons 
    '''    
    
    alpha = 120.0/180*scipy.pi
    cc_bond = 1.4/conversion_facs_position["Angstrom"] 
    Coor=numpy.zeros((Dim,3),dtype='f8')
    
    for ii in range(1,Dim-1):
        Coor[ii,2] = 0.0
        if ii%2 == 1:
            Coor[ii,1]= 0.0
        else:
            Coor[ii,1] = cc_bond*numpy.cos(alpha/2)
        Coor[ii,0] = cc_bond*numpy.sin(alpha/2)*ii
    
    # Now we have to add First and Last atom which are not in the same line with others
    rot = numpy.zeros((3,3))
    rot[0,0] = numpy.cos(alpha)
    rot[0,1] = numpy.sin(alpha)
    rot[1,1] = numpy.cos(alpha)
    rot[1,0] = -numpy.sin(alpha)
    rot[2,2] = numpy.cos(alpha)
    
    Coor[0,:] = Coor[1,:] + numpy.dot(rot,Coor[2,:]-Coor[1,:])
    Coor[Dim-1,:] = Coor[Dim-2,:] + numpy.dot(rot,Coor[Dim-3,:]-Coor[Dim-2,:])
    
    # Center molecule
    center = numpy.sum(Coor,0)
    center=center/Dim
    Coor=RotateAndMove(Coor,-center[0],-center[1],-center[2],0.0,0.0,0.0)
    
    # Rotate molecule
    VecX=vec_x
    VecY=vec_y

    norm=numpy.sqrt(numpy.dot(VecX,VecX))
    VecX=VecX/norm
    norm=numpy.sqrt(numpy.dot(VecY,VecY))
    VecY=VecY/norm

    #VecZ je vektorovy soucin
    VecZ=numpy.cross(VecX,VecY)
    norm=numpy.sqrt(numpy.dot(VecZ,VecZ))
    VecZ=VecZ/norm

    # VecX musi byt normovany
    if isclose(VecX[1],0,abs_tol=1e-7) and isclose(VecX[0],0,abs_tol=1e-7):
        Phi=0.0
    else:
        Phi=numpy.arctan(VecX[1]/VecX[0])
    
    if (VecX[0]<0):
        Phi=Phi+numpy.pi
    Psi=numpy.arcsin(VecX[2])

    axis=numpy.zeros((3,3))
    axis[0,:]=VecX
    axis[1,:]=VecY
    axis[2,:]=VecZ

    axis=RotateAndMove(axis,0.0,0.0,0.0,-Phi,Psi,0.0)
    VecX=axis[0,:]
    VecY=axis[1,:]
    VecZ=axis[2,:]
    
    if isclose(VecY[1],0,abs_tol=1e-7) and VecY[2]>0.0:
        Chi=numpy.pi/2
    elif isclose(VecY[1],0,abs_tol=1e-7) and VecY[2]<0.0:
        Chi=-numpy.pi/2
    elif VecY[1]>0:
        Chi=numpy.arctan(VecY[2]/VecY[1])
    else:
        Chi=numpy.arctan(VecY[2]/VecY[1])-numpy.pi
        
    Coor=RotateAndMove(Coor,0.0,0.0,0.0,0.0,0.0,Chi)
    Coor=RotateAndMove(Coor,0.0,0.0,0.0,0.0,-Psi,0.0)
    Coor=RotateAndMove(Coor,0.0,0.0,0.0,Phi,0.0,0.0)
    
    # Move molecule to its position
    Coor=RotateAndMove(Coor,Position[0],Position[1],Position[2],0.0,0.0,0.0)
    
    return Coor

def molecule_to_dipoles(rr,bond,direction=None):       
    ''' Converts molecule to set of dipoles for calculation of interaction energies
    
    Parameters
    ----------
    rr : numpy.array of real (dimension Natx3)
        position of atoms of the molecule
    bond : numpy.array of int (dimension Nbondx2)
        Indexes of atoms between which is chemical bond. For this purpose it can
        be used function GuessBonds from interaction module
        
    Returns
    -------
    ro : numpy.array of real (dimension Nbondx3 )
        Position of dipoles
    do : numpy.array of real (dimension Nbondx3 )
        Unit dipole orientation (can be then rescaled)
    '''
    Ndip=len(bond)
    ro=numpy.zeros((Ndip,3),dtype='f8')
    do=numpy.zeros((Ndip,3),dtype='f8')
    
    # Place unity dipole moments in centers of all bonds
    for ii in range(Ndip):
        if direction!=None:
            do[ii,:]=direction[:]
        else:
            do[ii,:]=rr[bond[ii,1],:]-rr[bond[ii,0],:]
            norm=numpy.sqrt(numpy.dot(do[ii,:],do[ii,:]))
            do[ii,:]=do[ii,:]/norm
        ro[ii,:]=(rr[bond[ii,1],:]+rr[bond[ii,0],:])/2
    return ro,do

#==============================================================================
#       Need to be fixed according to new better definition of rotation matrix        
#==============================================================================
        
def AlignMolecules2(CoorRef,Coor,OtherVec,indx_center,indx_x,indx_y):  
    ''' Same as AlignMolecules but same operations as for molecules are done  on vectors OtherVec'''
    
    if isinstance(indx_center,list):
        listcenter=True
    else:
        listcenter=False
    
    if isinstance(indx_x,list):
        if len(indx_x)==2 or len(indx_x)==4:
            listx=True
        else:
            raise IOError('x axis cannot be defined by more than two indexes')
    else:
        listx=False
        
    if isinstance(indx_y,list):
        if len(indx_y)==2 or len(indx_y)==4:
            listy=True
        else:
            raise IOError('y axis cannot be defined by more than two indexes')
    else:
        listy=False
    
    # Vypocet polohy a uhlu hlavnich smeru
    NAtom=numpy.shape(CoorRef)[0]
    if listcenter:
        center=numpy.zeros(3)
        for ii in range(len(indx_center)):
            center+=CoorRef[indx_center[ii],:]
        center=center/len(indx_center)
    else:
        center=CoorRef[indx_center,:]
        
    if listx and (len(indx_x)==2):
        VecX=CoorRef[indx_x[1],:]-CoorRef[indx_x[0],:]
    elif listx and (len(indx_x)==4):
        VecX=CoorRef[indx_x[1],:]-CoorRef[indx_x[0],:]+CoorRef[indx_x[3],:]-CoorRef[indx_x[2],:]
    else:
        VecX=CoorRef[indx_x,:]-center
    
    if listy and (len(indx_y)==2):
        VecY=CoorRef[indx_y[1],:]-CoorRef[indx_y[0],:]
    elif listy and (len(indx_y)==4):
        VecY=CoorRef[indx_y[1],:]-CoorRef[indx_y[0],:]+CoorRef[indx_y[3],:]-CoorRef[indx_y[2],:]
    else:
        VecY=CoorRef[indx_y,:]-center
    
    norm=numpy.sqrt(numpy.dot(VecX,VecX))
    VecX=VecX/norm
    norm=numpy.sqrt(numpy.dot(VecY,VecY))
    VecY=VecY/norm
    
    #VecZ je vektorovy soucin
    VecZ=numpy.cross(VecX,VecY)
    norm=numpy.sqrt(numpy.dot(VecZ,VecZ))
    VecZ=VecZ/norm

    # VecX musi byt normovany
    Phi=numpy.arctan(VecX[1]/VecX[0])
    if (VecX[0]<0):
        Phi=Phi+numpy.pi
    Psi=numpy.arcsin(VecX[2])
    
    axis=numpy.zeros((3+NAtom,3))
    axis[0,:]=VecX
    axis[1,:]=VecY
    axis[2,:]=VecZ
    for ii in range(NAtom):
        axis[ii+3,:]=CoorRef[ii,:]-center
#    print('Vektory pred transformaci')
#    print(VecX)
#    print(VecY)
#    print(VecZ)
    axis=RotateAndMove(axis,0.0,0.0,0.0,Phi,Psi,0.0)
    VecX=axis[0,:]
    VecY=axis[1,:]
    VecZ=axis[2,:]
    if VecY[1]>0:
        Chi=numpy.arctan(VecY[2]/VecY[1])
    else:
        Chi=numpy.arctan(VecY[2]/VecY[1])-numpy.pi
    axis=RotateAndMove(axis,0.0,0.0,0.0,0.0,0.0,Chi)
    VecX=axis[0,:]
    VecY=axis[1,:]
    VecZ=axis[2,:]
#    print('Podle x y z:')
#    print(VecX)
#    print(VecY)
#    print(VecZ)
    
    # Nyny otocit molekulu do nuly a pak preorientovat tak jako referencni 
    if listcenter:
        center_tmp=numpy.zeros(3)
        for ii in range(len(indx_center)):
            center_tmp+=Coor[indx_center[ii],:]
        center_tmp=center_tmp/len(indx_center)
    else:
        center_tmp=Coor[indx_center,:]
    
    if listx and (len(indx_x)==2):
        VecX_tmp=Coor[indx_x[1],:]-Coor[indx_x[0],:]
    elif listx and (len(indx_x)==4):
        VecX_tmp=Coor[indx_x[1],:]-Coor[indx_x[0],:]+Coor[indx_x[3],:]-Coor[indx_x[2],:]
    else:
        VecX_tmp=Coor[indx_x,:]-center_tmp
    
    if listy and (len(indx_y)==2):
        VecY_tmp=Coor[indx_y[1],:]-Coor[indx_y[0],:]
    elif listy and (len(indx_y)==4):
        VecY_tmp=Coor[indx_y[1],:]-Coor[indx_y[0],:]+Coor[indx_y[3],:]-Coor[indx_y[2],:]
    else:    
        VecY_tmp=Coor[indx_y,:]-center_tmp
        
    norm=numpy.sqrt(numpy.dot(VecX_tmp,VecX_tmp))
    VecX_tmp=VecX_tmp/norm
    norm=numpy.sqrt(numpy.dot(VecY_tmp,VecY_tmp))
    VecY_tmp=VecY_tmp/norm
    VecZ_tmp=numpy.cross(VecX_tmp,VecY_tmp)
    norm=numpy.sqrt(numpy.dot(VecZ_tmp,VecZ_tmp))
    VecZ_tmp=VecZ_tmp/norm

    axis=numpy.zeros((3+NAtom,3))
    axis[0,:]=VecX_tmp
    axis[1,:]=VecY_tmp
    axis[2,:]=VecZ_tmp
    for ii in range(NAtom):
        axis[ii+3,:]=Coor[ii,:]-center_tmp
    
#    print('Vektory pred transformaci')
#    print(VecX_tmp)
#    print(VecY_tmp)
#    print(VecZ_tmp)
    Phi_tmp=numpy.arctan(VecX_tmp[1]/VecX_tmp[0])
    if (VecX_tmp[0]<0):
        Phi_tmp=Phi_tmp+numpy.pi
    Psi_tmp=numpy.arcsin(VecX_tmp[2])

#    print('OtherVec 1.:',OtherVec)
    axis=RotateAndMove(axis,0.0,0.0,0.0,Phi_tmp,Psi_tmp,0.0)
    if OtherVec!=None:
        OtherVec=RotateAndMove(OtherVec,0.0,0.0,0.0,Phi_tmp,Psi_tmp,0.0)
#        print('Rotuji dipole o uhley Phi:',Phi_tmp,' Psi:',Psi_tmp)
    
    VecX_tmp=axis[0,:]
    VecY_tmp=axis[1,:]
    VecZ_tmp=axis[2,:]
#    print('OtherVec 2.:',OtherVec)
    if VecY_tmp[1]>0:
        Chi_tmp=numpy.arctan(VecY_tmp[2]/VecY_tmp[1])
    else:
        Chi_tmp=numpy.arctan(VecY_tmp[2]/VecY_tmp[1])-numpy.pi
    axis=RotateAndMove(axis,0.0,0.0,0.0,0.0,0.0,Chi_tmp)
    if OtherVec!=None:
        OtherVec=RotateAndMove(OtherVec,0.0,0.0,0.0,0.0,0.0,Chi_tmp)
#        print('Rotuji dipole o uhley Chi:',Chi_tmp)
#    print('OtherVec 3.:',OtherVec)
    
    VecX_tmp=axis[0,:]
    VecY_tmp=axis[1,:]
    VecZ_tmp=axis[2,:]
#    print('Podle x y z:')
#    print(VecX_tmp)
#    print(VecY_tmp)
#    print(VecZ_tmp)
    axis=RotateAndMove(axis,0.0,0.0,0.0,0.0,0.0,-Chi)
    axis=RotateAndMove(axis,0.0,0.0,0.0,0.0,-Psi,0.0)
    axis=RotateAndMove(axis,0.0,0.0,0.0,-Phi,0.0,0.0)
    if OtherVec!=None:
        OtherVec=RotateAndMove(OtherVec,0.0,0.0,0.0,0.0,0.0,-Chi)
        OtherVec=RotateAndMove(OtherVec,0.0,0.0,0.0,0.0,-Psi,0.0)
        OtherVec=RotateAndMove(OtherVec,0.0,0.0,0.0,-Phi,0.0,0.0)
#        print('Rotuji dipole o uhly Chi:',-Chi,' Psi:',-Psi,' Phi:',-Phi)
#    print('OtherVec 4.:',OtherVec)
#    print('Vektory po transformaci')
#    print(axis[0,:])
#    print(axis[1,:])
#    print(axis[2,:])
    
    for ii in range(NAtom):
        Coor[ii,:]=axis[ii+3,:]+center
    #vis.PlotTwoMolecules(CoorRef,qc.at_info[:,0],Coor,qc.at_info[:,0])
    return Coor,OtherVec


'''----------------------- TEST PART --------------------------------'''
if __name__=="__main__":
    
    ''' Rotation'''
    test=True
    Alk_Coor=prepare_alkene(10)
    Alk_Coor_rot=RotateAndMove(Alk_Coor,0.0,0.0,0.0,numpy.pi/2,0.0,0.0)
    for ii in range(len(Alk_Coor)):
        if (not isclose(Alk_Coor_rot[ii,0],-Alk_Coor[ii,1])) or (not isclose(Alk_Coor_rot[ii,1],Alk_Coor[ii,0])) or Alk_Coor_rot[ii,2]!=Alk_Coor[ii,2]:
            #print(ii,Alk_Coor_rot[ii,0],-Alk_Coor[ii,1],Alk_Coor_rot[ii,1],Alk_Coor[ii,0])
            test=False
    Alk_Coor_rot=RotateAndMove(Alk_Coor,0.0,0.0,0.0,numpy.pi/2,numpy.pi/2,0.0)
    for ii in range(len(Alk_Coor)):
        if (not isclose(Alk_Coor_rot[ii,2],Alk_Coor[ii,1])) or (not isclose(Alk_Coor_rot[ii,1],Alk_Coor[ii,0])) or (not isclose(Alk_Coor_rot[ii,0],Alk_Coor[ii,2],abs_tol=1e-8)):
            #print(ii,Alk_Coor_rot[ii,2],Alk_Coor[ii,1],Alk_Coor_rot[ii,1],Alk_Coor[ii,0],Alk_Coor_rot[ii,0],Alk_Coor[ii,2])
            test=False
    if test:
        print('RotateAndMove        ...    OK')
    else:
        print('RotateAndMove        ...    Error')
        
        
        
    ''' Alkene preparation'''
    test=True
    Alk_Coor=prepare_alkene(10)
    Alk_Coor_rot=RotateAndMove(Alk_Coor,0.0,0.0,0.0,numpy.pi/2,numpy.pi/2,0.0)
    Alk_Coor=prepare_alkene(10,vec_x=numpy.array([0.0,1.0,0.0]),vec_y=numpy.array([0.0,0.0,1.0]))
    for ii in range(len(Alk_Coor)):
        if (not isclose(Alk_Coor_rot[ii,0],Alk_Coor[ii,0],abs_tol=1e-8)) or (not isclose(Alk_Coor_rot[ii,1],Alk_Coor[ii,1])) or (not isclose(Alk_Coor_rot[ii,2],Alk_Coor[ii,2])):
            print(ii,Alk_Coor_rot[ii,2],Alk_Coor[ii,1],Alk_Coor_rot[ii,1],Alk_Coor[ii,0],Alk_Coor_rot[ii,0],Alk_Coor[ii,2])
            test=False   
    test=True
    Alk_Coor=prepare_alkene(10)
    Alk_Coor_rot=RotateAndMove(Alk_Coor,0.0,0.0,0.0,0.0,-numpy.pi/2,0.0)
    Alk_Coor=prepare_alkene(10,vec_x=numpy.array([0.0,0.0,1.0]),vec_y=numpy.array([0.0,1.0,0.0]))
    for ii in range(len(Alk_Coor)):
        if (not isclose(Alk_Coor_rot[ii,0],Alk_Coor[ii,0],abs_tol=1e-8)) or (not isclose(Alk_Coor_rot[ii,1],Alk_Coor[ii,1])) or (not isclose(Alk_Coor_rot[ii,2],Alk_Coor[ii,2])):
            print(ii,Alk_Coor_rot[ii,0],Alk_Coor[ii,0],Alk_Coor_rot[ii,1],Alk_Coor[ii,1],Alk_Coor_rot[ii,2],Alk_Coor[ii,2])
            test=False 
    if test:
        print('prepare_alkene       ...    OK')
    else:
        print('prepare_alkene       ...    Error')
        
        
        
    ''' Carotenoid preparation'''
    test=True
    Alk_Coor=prepare_carotenoid(10)
    Alk_Coor_rot=RotateAndMove(Alk_Coor,0.0,0.0,0.0,numpy.pi/2,numpy.pi/2,0.0)
    Alk_Coor=prepare_carotenoid(10,vec_x=numpy.array([0.0,1.0,0.0]),vec_y=numpy.array([0.0,0.0,1.0]))
    for ii in range(len(Alk_Coor)):
        if (not isclose(Alk_Coor_rot[ii,0],Alk_Coor[ii,0],abs_tol=1e-8)) or (not isclose(Alk_Coor_rot[ii,1],Alk_Coor[ii,1])) or (not isclose(Alk_Coor_rot[ii,2],Alk_Coor[ii,2])):
            print(ii,Alk_Coor_rot[ii,2],Alk_Coor[ii,1],Alk_Coor_rot[ii,1],Alk_Coor[ii,0],Alk_Coor_rot[ii,0],Alk_Coor[ii,2])
            test=False   
    Alk_Coor=prepare_carotenoid(10)
    Alk_Coor_rot=RotateAndMove(Alk_Coor,0.0,0.0,0.0,0.0,-numpy.pi/2,0.0)
    Alk_Coor=prepare_carotenoid(10,vec_x=numpy.array([0.0,0.0,1.0]),vec_y=numpy.array([0.0,1.0,0.0]))
    for ii in range(len(Alk_Coor)):
        if (not isclose(Alk_Coor_rot[ii,0],Alk_Coor[ii,0],abs_tol=1e-8)) or (not isclose(Alk_Coor_rot[ii,1],Alk_Coor[ii,1])) or (not isclose(Alk_Coor_rot[ii,2],Alk_Coor[ii,2])):
            print(ii,Alk_Coor_rot[ii,0],Alk_Coor[ii,0],Alk_Coor_rot[ii,1],Alk_Coor[ii,1],Alk_Coor_rot[ii,2],Alk_Coor[ii,2])
            test=False 
    if test:
        print('prepare_carotenoid   ...    OK')
    else:
        print('prepare_carotenoid   ...    Error')
        
        
        
    ''' Align '''
    test=True
    Coor=prepare_carotenoid(10)
    Coor_rot=RotateAndMove(Coor,1.3,-2.6,-0.8,numpy.pi/4,-numpy.pi/2,numpy.pi/6)
#    for ii in range(len(Coor_rot)):
#        print(ii,Coor_rot[ii,:])    
    Coor_aligned,Phi,Psi,Chi,center=AlignMolecules(Coor_rot,Coor,4,8,5,print_angles=True)
    for ii in range(len(Coor_rot)):
        if (not isclose(Coor_aligned[ii,0],Coor_rot[ii,0],abs_tol=1e-8)) or (not isclose(Coor_aligned[ii,1],Coor_rot[ii,1])) or (not isclose(Coor_aligned[ii,2],Coor_rot[ii,2])):
            print(ii,Coor_aligned[ii,0],Coor_rot[ii,0],Coor_aligned[ii,1],Coor_rot[ii,1],Coor_aligned[ii,2],Coor_rot[ii,2])
            test=False

    if test:
        print('AlignMolecules       ...    OK       - for two identical molecules')
    else:
        print('AlignMolecules       ...    Error    - for two identical molecules')
    
    #print(Phi/numpy.pi*180,Psi/numpy.pi*180,Chi/numpy.pi*180)
    
    test=True
    Coor_rot2=CenterMolecule(Coor_aligned,4,8,5)
    
    #print(Coor_rot2)
    if not (isclose(Coor_rot2[4,0],0.0,abs_tol=1e-8) and isclose(Coor_rot2[4,1],0.0,abs_tol=1e-8) and isclose(Coor_rot2[4,2],0.0,abs_tol=1e-8)):
       test=False
    if not (isclose(Coor_rot2[8,1],0.0,abs_tol=1e-8) and isclose(Coor_rot2[8,2],0.0,abs_tol=1e-8) and isclose(Coor_rot2[5,2],0.0,abs_tol=1e-8)):
       test=False
       
    if test:
        print('CenterMolecule       ...    OK')
    else:
        print('CenterMolecule       ...    Error')
    
    test=True
#    for ii in range(len(Coor_rot2)):
#        print(ii,Coor_rot2[ii,:])
    Coor_rot2=RotateAndMove_1(Coor_rot2,center[0],center[1],center[2],Phi,Psi,Chi)
    for ii in range(len(Coor_rot2)):
        if (not isclose(Coor_rot2[ii,0],Coor_rot[ii,0],abs_tol=1e-8)) or (not isclose(Coor_rot2[ii,1],Coor_rot[ii,1])) or (not isclose(Coor_rot2[ii,2],Coor_rot[ii,2])):
            print(ii,Coor_rot2[ii,0],Coor_rot[ii,0],Coor_rot2[ii,1],Coor_rot[ii,1],Coor_rot2[ii,2],Coor_rot[ii,2])
            test=False
    if test:
        print('AlignMolecules       ...    OK       - for angle and center output')
    else:
        print('AlignMolecules       ...    Error    - for angle and center output')
        

    ''' Solve angle ''' 
    test=True       
    Coor=prepare_carotenoid(10)
    Phi=numpy.pi/6
    Psi=-numpy.pi/8
    Chi=numpy.pi/7
    Coor_rot=RotateAndMove(Coor,0.0,0.0,0.0,Phi,Psi,Chi)
    mass=numpy.array([1.0]*10)    
    Phi,Psi,Chi=SolveAngle(Coor_rot,Coor,mass,method='minimize',maxiter=150,bounds=((-0.6,0.6),(-0.6,0.6),(-0.6,0.6)))
    Coor_rot2=RotateAndMove_1(Coor,0.0,0.0,0.0,Phi,Psi,Chi)
    for ii in range(len(Coor_rot2)):
        if (not isclose(Coor_rot2[ii,0],Coor_rot[ii,0],abs_tol=1e-4)) or (not isclose(Coor_rot2[ii,1],Coor_rot[ii,1],abs_tol=1e-4)) or (not isclose(Coor_rot2[ii,2],Coor_rot[ii,2],abs_tol=1e-4)):
            print(ii,Coor_rot2[ii,0],Coor_rot[ii,0],Coor_rot2[ii,1],Coor_rot[ii,1],Coor_rot2[ii,2],Coor_rot[ii,2])
            test=False
    if test:
        print('SolveAngle           ...    OK')
    else:
        print('SolveAngle           ...    Error')
        
    ''' TEST for project_on_plane '''
        
