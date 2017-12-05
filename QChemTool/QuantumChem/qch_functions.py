import numpy
import scipy   # for electronic integrals

# TODO: some parts are dependent on class Trida - correct this dependence

# =============================================================================
# Overlap between atomic orbitals
# =============================================================================

def _overlap_s(indx,ri,rj,a,b,ii,jj):
    ''' Calculates part of one dimensional overlap between two cartesian 
    gaussian orbitals after transformations of coordinates.
    
    Parameters
    -------------
    index : integer {0,1,2}
        Specifies in which dimension integral would be performed: 0->x, 1->y, 2->z
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr) 
    a : float
        Exponent of first atomic gaussian orbital
    b : float
        Exponent of second atomic gaussian orbital
    ii : integer >= 0
        Power of coorndinate in specified direction (see Definition section) for
        the first cartesian gaussian orbital. For example for py orbital in 
        x direction ii=0, in y direction ii=1 and in z direction ii=0.
    jj : integer >= 0
        Power of coorndinate in specified direction (see Definition section) for
        the second cartesian gaussian orbital. For example for dxz orbital in 
        x direction ii=1, in y direction ii=0 and in z direction ii=1.
        
    Returns
    -------------
    res : float
        Value of overlap integral in one dimension after coordinate transformation
        (see Definition section)
    
    Definition
    -------------
    ``res=int_{-infty}^{infty}{ ( x - a/(a+b)(x1-x2) )^ii * ( x - b/(a+b)(x2-x1) )^jj * e^( -(a+b)*x^2 ) dx}`` \n
    Where in our definition x1=ri and x2=rj \n
    Whole overlap between two cartesian orbitals in sigle dimension would be: \n
    ``Overlap= sqrt(pi/(a+b)) * e^(-(a*b/(a+b))*|x1-x2|^2) * _overlap_s(indx,ri,rj,a,b,ii,jj)`` \n
    Where definition of Overlap is: \n
    ``Overlap = int_{-infty}^{infty}{ (x-x1)^ii * (x-x2)^jj * e^(-a*|x-x1|^2) * e^(-b*|x-x2|^2) dx}``
    
    Notes
    -------------
    This function is using recurent relations defined in **http://www.mathematica-journal.com/2012/02/evaluation-of-gaussian-molecular-integrals/**
    
    '''
    
    if len(ri)!=3 or len(rj)!=3 or ii<0 or jj<0 or indx<0 or indx>2:
        raise IOError('Wrong parameters for function overlap_s')
    
    if ii==0 and jj==0:
        res=1.0        
    elif ii==1 and jj==0:
        res = -(ri[indx]-((a*ri[indx]+b*rj[indx])/(a+b)))
    elif ii>1 and jj==0:
        res = -(ri[indx]-(a*ri[indx]+b*rj[indx])/(a+b))*_overlap_s(indx,ri,rj,a,b,ii-1,0) + ((ii-1)/(2*(a+b)))*_overlap_s(indx,ri,rj,a,b,ii-2,0)
    else:
        res = _overlap_s(indx,ri,rj,a,b,ii+1,jj-1) + (ri[indx]-rj[indx])*_overlap_s(indx,ri,rj,a,b,ii,jj-1)
        
    return res
    
def _overlap_GTO_MathJour_RegType(ri,rj,a,b,index1,index2):
    ''' 
    Calculate overlap integral between two cartesian gaussian atomic orbitals
    
    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr) 
    a : float
        Exponent of first atomic gaussian orbital
    b : float
        Exponent of second atomic gaussian orbital
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of first atomic orbital. For exaple dxz orbital
        is defined as ``index1=[1,0,1]``, dyy orbital as ``index1=[0,2,0]``,
        px orbital as ``index1=[1,0,0]``, s orbital as ``index1=[0,0,0]`` and 
        so on...
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of the second atomic orbital. 
        (For more info see index1 definition)
    
    Returns
    -----------
    res : float
        Overlap between two catesian atomic gaussian orbitals
    
    Definition
    -----------
    ``res = (pi/(a+b))^(3/2)`` \n
    ``res = res * int_{-infty}^{infty}{ (x-ri[0])^index1[0] * (x-rj[0])^index2[0] * e^(-a*|x-ri[0]|^2) * e^(-b*|x-rj[0]|^2) dx}`` \n
    ``res = res * int_{-infty}^{infty}{ (y-ri[1])^index1[1] * (y-rj[1])^index2[1] * e^(-a*|y-ri[1]|^2) * e^(-b*|y-rj[1]|^2) dy}`` \n
    ``res = res * int_{-infty}^{infty}{ (z-ri[2])^index1[2] * (z-rj[2])^index2[2] * e^(-a*|z-ri[2]|^2) * e^(-b*|z-rj[2]|^2) dz}``
    
    Notes
    -------------
    This function is using recurent relations defined in **http://www.mathematica-journal.com/2012/02/evaluation-of-gaussian-molecular-integrals/**
    
    '''
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)<0 or numpy.sum(index2)<0:
        raise IOError('Wrong imput parameters in function _overlap_GTO_MathJour_RegType',index1,index2)
    
    EAB=numpy.exp(-(a*b/(a+b))*numpy.dot(ri-rj,ri-rj))
    res=EAB*numpy.power(numpy.pi/(a+b),3/2)*_overlap_s(0,ri,rj,a,b,index1[0],index2[0])*_overlap_s(1,ri,rj,a,b,index1[1],index2[1])*_overlap_s(2,ri,rj,a,b,index1[2],index2[2])        
    return res

    
def _overlap_GTO_MathJour_5DRegType(ri,rj,a,b,index1,index2):
    ''' Calculate overlap between cartesian gaussian atomic orbital and
    spherical harmonics gaussian atomic orbital 5d. First index must correspond
    to 5d orbital.
    
    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr) 
    a : float
        Exponent of first atomic gaussian orbital
    b : float
        Exponent of second atomic gaussian orbital
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of the 5d atomic orbital (first orbital). Definition
        of indexes for 5d orbital is:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of second atomic orbital. For exaple dxz orbital
        is defined as ``index1=[1,0,1]``, dyy orbital as ``index1=[0,2,0]``,
        px orbital as ``index1=[1,0,0]``, s orbital as ``index1=[0,0,0]`` and 
        so on...
    
    Returns
    -----------
    res : float
        Overlap between 5d spherical harmonic gaussian atomic orbital and 
        any cartesian gaussian atomic orbital
        
    Notes
    -------------
    This function is using recurent relations defined in **http://www.mathematica-journal.com/2012/02/evaluation-of-gaussian-molecular-integrals/**
    
    '''    
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)!=(-2) or numpy.sum(index2)<0:
        raise IOError('Wrong imput parameters in function _overlap_GTO_MathJour_5DRegType')    
    
    if index1[0]==(-2):
        #3ZZ-RR
        res=2*_overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,0,2],index2)-_overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],index2)-_overlap_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],index2)
    elif index1[0]==(-1) and index1[2]==(-1):
        #XZ
        res=_overlap_GTO_MathJour_RegType(ri,rj,a,b,[1,0,1],index2)
    elif index1[1]==(-1) and index1[2]==(-1):
        #YZ
        res=_overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,1,1],index2)
    elif index1[1]==(-2):
        #XX-YY
        res=_overlap_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],index2)-_overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],index2)
    elif index1[0]==(-1) and index1[1]==(-1):
        #XY
        res=_overlap_GTO_MathJour_RegType(ri,rj,a,b,[1,1,0],index2)
    else:
        raise IOError('Error in _overlap_GTO_MathJour_5DRegType')
    return res
    
def _overlap_GTO_MathJour_5D5D(ri,rj,a,b,index1,index2):
    ''' Calculate overlap between two 5d spherical harmonics gaussian atomic 
    orbitals.
    
    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr) 
    a : float
        Exponent of first atomic gaussian orbital
    b : float
        Exponent of second atomic gaussian orbital
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first 5d atomic orbital. Definition
        of indexes for 5d orbital is:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of the second 5d atomic orbital. (See **index1** for 
        definition)
        
    
    Returns
    -----------
    res : float
        Overlap between two 5d spherical harmonic gaussian atomic orbitals
    
    Notes
    -------------
    This function is using recurent relations defined in **http://www.mathematica-journal.com/2012/02/evaluation-of-gaussian-molecular-integrals/**
    
    '''

    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)!=(-2) or numpy.sum(index2)!=(-2):
        raise IOError('Wrong imput parameters in function _overlap_GTO_MathJour_5D5D') 
    
    if numpy.dot(index1,index1)==2 and numpy.dot(index2,index2)==2:
        res=_overlap_GTO_MathJour_RegType(ri,rj,a,b,numpy.abs(index1),numpy.abs(index2))
    elif numpy.dot(index1,index1)==4 and numpy.dot(index2,index2)==2:
        if index1[0]==(-2):
            res=2*_overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,0,2],numpy.abs(index2))-_overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],numpy.abs(index2))-_overlap_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],numpy.abs(index2))
        elif index1[1]==(-2):
            res=_overlap_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],numpy.abs(index2))-_overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],numpy.abs(index2))
        else:
            raise IOError('Error in _overlap_5D5D')
    elif numpy.dot(index1,index1)==2 and numpy.dot(index2,index2)==4:
        if index2[0]==(-2):
            res=2*_overlap_GTO_MathJour_RegType(ri,rj,a,b,numpy.abs(index1),[0,0,2])-_overlap_GTO_MathJour_RegType(ri,rj,a,b,numpy.abs(index1),[0,2,0])-_overlap_GTO_MathJour_RegType(ri,rj,a,b,numpy.abs(index1),[2,0,0])
        elif index2[1]==(-2):
            res=_overlap_GTO_MathJour_RegType(ri,rj,a,b,numpy.abs(index1),[2,0,0])-_overlap_GTO_MathJour_RegType(ri,rj,a,b,numpy.abs(index1),[0,2,0])
        else:
            raise IOError('Error in _overlap_5D5D')
    elif numpy.dot(index1,index1)==4 and numpy.dot(index2,index2)==4:
        if index1[0]==(-2) and index2[0]==(-2):
            res=4*_overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,0,2],[0,0,2]) - 2*_overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,0,2],[0,2,0]) - 2*_overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,0,2],[2,0,0])
            res += -2*_overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],[0,0,2]) + _overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],[0,2,0]) + _overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],[2,0,0])
            res += -2*_overlap_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],[0,0,2]) + _overlap_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],[0,2,0]) + _overlap_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],[2,0,0])
        elif index1[1]==(-2) and index2[1]==(-2):
            res = _overlap_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],[2,0,0]) - _overlap_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],[0,2,0])
            res += -_overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],[2,0,0]) + _overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],[0,2,0])
        elif index1[0]==(-2) and index2[1]==(-2):
            res = 2*_overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,0,2],[2,0,0]) - 2*_overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,0,2],[0,2,0]) - _overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],[2,0,0])
            res += _overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],[0,2,0]) - _overlap_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],[2,0,0]) + _overlap_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],[0,2,0])
        elif index1[1]==(-2) and index2[0]==(-2):
            res = 2*_overlap_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],[0,0,2]) - _overlap_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],[0,2,0]) - _overlap_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],[2,0,0])
            res += -2*_overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],[0,0,2]) + _overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],[0,2,0]) + _overlap_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],[2,0,0])
        else:
            raise IOError('Error in _overlap_GTO_MathJour_5D5D')
    else:
        raise IOError('Error in _overlap_GTO_MathJour_5D5D')
    return res
    

def _overlap_GTO_MathJour_7FRegType(ri,rj,a,b,index1,index2):
    ''' Should calculate overlap between 7f spherical harmonics gaussian atomic
    orbital and cartesian gaussian atomic orbital, but by comparing the results 
    The '7f' decomposition into cartesian 'f' functions is most probably wrong.
    
    **DO NOT USE IT FOR ANY CALCULATION**
    '''    
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)!=(-3) or (numpy.sum(index2)<0 and numpy.sum(index2)!=(-2)):
        raise IOError('Wrong imput parameters in function _overlap_GTO_MathJour_7FRegType')    
    
    if index1[2]==(-3):
        #ZZZ-ZRR
        res = -(overlap_GTO_MathJour(ri,rj,a,b,[2,0,1],index2)+overlap_GTO_MathJour(ri,rj,a,b,[0,2,1],index2))
    elif index1[0]==(-1) and index1[2]==(-2):
        #XZZ-XRR
        res = -(overlap_GTO_MathJour(ri,rj,a,b,[3,0,0],index2)+overlap_GTO_MathJour(ri,rj,a,b,[1,2,0],index2))
    elif index1[1]==(-1) and index1[2]==(-2):
        #YZZ-YRR
        res = -(overlap_GTO_MathJour(ri,rj,a,b,[2,1,0],index2)+overlap_GTO_MathJour(ri,rj,a,b,[0,3,0],index2))
    elif index1[0]==(-2) and index1[2]==(-1):
        #XXZ-YYZ
        res = overlap_GTO_MathJour(ri,rj,a,b,[2,0,1],index2) - overlap_GTO_MathJour(ri,rj,a,b,[0,2,1],index2)
    elif index1[0]==(-1) and index1[1]==(-1) and index1[2]==(-1):
        #XYZ
        res = overlap_GTO_MathJour(ri,rj,a,b,[1,1,1],index2)
    elif index1[0]==(-3):
        #XXX-XYY
        res = overlap_GTO_MathJour(ri,rj,a,b,[3,0,0],index2) - overlap_GTO_MathJour(ri,rj,a,b,[1,2,0],index2)
    elif index1[0]==(-2) and index1[1]==(-1):
        #XXY-YYY
        res = overlap_GTO_MathJour(ri,rj,a,b,[2,1,0],index2) - overlap_GTO_MathJour(ri,rj,a,b,[0,3,0],index2)
    else:
        raise IOError('Error in _overlap_GTO_MathJour_7FRegType')
    return res
    
def _overlap_GTO_MathJour_7FRegType_newdef(ri,rj,a,b,index1,index2):
    ''' Should calculate overlap between 7f spherical harmonics gaussian atomic
    orbital and cartesian gaussian atomic orbital, but by comparing the results 
    The '7f' decomposition into cartesian 'f' functions is most probably wrong.
    
    **DO NOT USE IT FOR ANY CALCULATION**
    '''    
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)!=(-3) or (numpy.sum(index2)<0 and numpy.sum(index2)!=(-2)):
        raise IOError('Wrong imput parameters in function _overlap_GTO_MathJour_7FRegType')    
    
    if index1[2]==(-3):
        # 2*ZZZ-3*XXZ-3*YYZ
        res = 2*overlap_GTO_MathJour(ri,rj,a,b,[0,0,3],index2)-3*(overlap_GTO_MathJour(ri,rj,a,b,[2,0,1],index2)+overlap_GTO_MathJour(ri,rj,a,b,[0,2,1],index2))
    elif index1[0]==(-1) and index1[2]==(-2):
        # 4*XZZ-XXX-XYY
        res = 4*overlap_GTO_MathJour(ri,rj,a,b,[1,0,2],index2) - (overlap_GTO_MathJour(ri,rj,a,b,[3,0,0],index2)+overlap_GTO_MathJour(ri,rj,a,b,[1,2,0],index2))
    elif index1[1]==(-1) and index1[2]==(-2):
        # 4*YZZ-XXY-YYY
        res = 4*overlap_GTO_MathJour(ri,rj,a,b,[0,1,2],index2) - (overlap_GTO_MathJour(ri,rj,a,b,[2,1,0],index2)+overlap_GTO_MathJour(ri,rj,a,b,[0,3,0],index2))
    elif index1[0]==(-2) and index1[2]==(-1):
        #XXZ-YYZ
        res = overlap_GTO_MathJour(ri,rj,a,b,[2,0,1],index2) - overlap_GTO_MathJour(ri,rj,a,b,[0,2,1],index2)
    elif index1[0]==(-1) and index1[1]==(-1) and index1[2]==(-1):
        #XYZ
        res = overlap_GTO_MathJour(ri,rj,a,b,[1,1,1],index2)
    elif index1[0]==(-3):
        # XXX-3*XYY
        res = overlap_GTO_MathJour(ri,rj,a,b,[3,0,0],index2) - 3*overlap_GTO_MathJour(ri,rj,a,b,[1,2,0],index2)
    elif index1[0]==(-2) and index1[1]==(-1):
        # 3*XXY-YYY
        res = 3*overlap_GTO_MathJour(ri,rj,a,b,[2,1,0],index2) - overlap_GTO_MathJour(ri,rj,a,b,[0,3,0],index2)
    else:
        raise IOError('Error in _overlap_GTO_MathJour_7FRegType')
    return res

def _overlap_GTO_MathJour_7F7F(ri,rj,a,b,index1,index2):
    ''' Should calculate overlap between two 7f spherical harmonics gaussian
    atomic orbitals, but by comparing the results the '7f' decomposition into
    cartesian 'f' functions is most probably wrong.
    
    **DO NOT USE IT FOR ANY CALCULATION**
    '''  
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)!=(-3) or numpy.sum(index2)!=(-3):
        raise IOError('Wrong imput parameters in function _overlap_GTO_MathJour_7F7F')    
    
    if index2[2]==(-3):
        #ZZZ-ZRR
        res = -(_overlap_GTO_MathJour_7FRegType(ri,rj,a,b,index1,[2,0,1])+_overlap_GTO_MathJour_7FRegType(ri,rj,a,b,index1,[0,2,1]))
    elif index2[0]==(-1) and index2[2]==(-2):
        #XZZ-XRR
        res = -(_overlap_GTO_MathJour_7FRegType(ri,rj,a,b,index1,[3,0,0])+_overlap_GTO_MathJour_7FRegType(ri,rj,a,b,index1,[1,2,0]))
    elif index2[1]==(-1) and index2[2]==(-2):
        #YZZ-YRR
        res = -(_overlap_GTO_MathJour_7FRegType(ri,rj,a,b,index1,[2,1,0])+_overlap_GTO_MathJour_7FRegType(ri,rj,a,b,index1,[0,3,0]))
    elif index2[0]==(-2) and index2[2]==(-1):
        #XXZ-YYZ
        res = _overlap_GTO_MathJour_7FRegType(ri,rj,a,b,index1,[2,0,1]) - _overlap_GTO_MathJour_7FRegType(ri,rj,a,b,index1,[0,2,1])
    elif index2[0]==(-1) and index2[1]==(-1) and index2[2]==(-1):
        #XYZ
        res = _overlap_GTO_MathJour_7FRegType(ri,rj,a,b,index1,[1,1,1])
    elif index2[0]==(-3):
        #XXX-XYY
        res = _overlap_GTO_MathJour_7FRegType(ri,rj,a,b,index1,[3,0,0]) - _overlap_GTO_MathJour_7FRegType(ri,rj,a,b,index1,[1,2,0])
    elif index2[0]==(-2) and index2[1]==(-1):
        #XXY-YYY
        res = _overlap_GTO_MathJour_7FRegType(ri,rj,a,b,index1,[2,1,0]) - _overlap_GTO_MathJour_7FRegType(ri,rj,a,b,index1,[0,3,0])
    else:
        raise IOError('Error in _overlap_GTO_MathJour_7F7F')
    return res
    

def overlap_GTO_MathJour(ri,rj,a,b,index1,index2):
    ''' Calculate overlap two gaussian atomic orbitals. All cartesian atomic
    orbitals are supported and 5d spherical harmonic orbital is also supported.
    
    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr) 
    a : float
        Exponent of first atomic gaussian orbital
    b : float
        Exponent of second atomic gaussian orbital
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first atomic orbital. For exaple for cartesian 
        gaussian atomic orbitals: dxz orbital is defined as ``index1=[1,0,1]``,
        dyy orbital as ``index1=[0,2,0]``, px orbital as ``index1=[1,0,0]``,
        s orbital as ``index1=[0,0,0]`` and so on...
        For the 5d orbital different indexing and decomposition into cartesian
        orbitals is used:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of the second atomic orbital. (See **index1** for 
        the definition)
    
    Returns
    -----------
    res : float
        Overlap between two gaussian atomic orbitals
    
    Definition
    -----------
    ``res = (pi/(a+b))^(3/2)`` \n
    ``res = res * int_{-infty}^{infty}{ (x-ri[0])^index1[0] * (x-rj[0])^index2[0] * e^(-a*|x-ri[0]|^2) * e^(-b*|x-rj[0]|^2) dx}`` \n
    ``res = res * int_{-infty}^{infty}{ (y-ri[1])^index1[1] * (y-rj[1])^index2[1] * e^(-a*|y-ri[1]|^2) * e^(-b*|y-rj[1]|^2) dy}`` \n
    ``res = res * int_{-infty}^{infty}{ (z-ri[2])^index1[2] * (z-rj[2])^index2[2] * e^(-a*|z-ri[2]|^2) * e^(-b*|z-rj[2]|^2) dz}``

    Notes
    -------------
    This function is using recurent relations defined in **http://www.mathematica-journal.com/2012/02/evaluation-of-gaussian-molecular-integrals/**
    Transformation of spherical harmonic gaussian orbitals into cartesian ones 
    is taken from definitions is Gaussian 09 quantum chemistry software.
    
    '''
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3:
        raise IOError('Wrong imput parameters in function overlap_GTO_MathJour')
    if numpy.sum(index1)==-3 or numpy.sum(index2)==-3:
        raise IOError('7f functions are not properly implemented - add 10F and 6D keyword to gaussian route section in order to use cartesian atomic orbitals which are well implemented')

    if numpy.sum(index1)>=0 and numpy.sum(index2)>=0:
        res = _overlap_GTO_MathJour_RegType(ri,rj,a,b,index1,index2)
    elif numpy.sum(index1)==-2 and numpy.sum(index2)>=0: # overlap of 5d orbital withregular type orbital
        res = _overlap_GTO_MathJour_5DRegType(ri,rj,a,b,index1,index2)
    elif numpy.sum(index2)==-2 and numpy.sum(index1)>=0: # is second orbital is 5D move this orbital to first place
        res = _overlap_GTO_MathJour_5DRegType(rj,ri,b,a,index2,index1)
    elif numpy.sum(index1)==-2 and numpy.sum(index2)==-2:
        res = _overlap_GTO_MathJour_5D5D(ri,rj,a,b,index1,index2)
    elif numpy.sum(index1)==-3 and (numpy.sum(index2)>=0 or numpy.sum(index2)==(-2)): # overlap of 7f orbital withregular type orbital or with 5d
        res = _overlap_GTO_MathJour_7FRegType(ri,rj,a,b,index1,index2)
        #res = _overlap_GTO_MathJour_7FRegType_newdef(ri,rj,a,b,index1,index2)
    elif numpy.sum(index2)==-3 and (numpy.sum(index1)>=0 or numpy.sum(index1)==(-2)): # if second orbital is 7d move this orbital to first place
        res = _overlap_GTO_MathJour_7FRegType(rj,ri,b,a,index2,index1)
        #res = _overlap_GTO_MathJour_7FRegType_newdef(rj,ri,b,a,index2,index1)
    elif numpy.sum(index1)==-3 and numpy.sum(index2)==-3:
        res = _overlap_GTO_MathJour_7F7F(rj,ri,b,a,index2,index1)
    else:
        raise IOError('Unsupported type of gaussian orbitals for calculation of overlap')
    return res


def overlap_STO(ri,rj,coef1,coef2,exp1,exp2,index1,index2):
    ''' Calculate overlap between two slater orbitals
    
    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr)
    coef1 : numpy array or list of floats 
        Expansion coefficients of first slater atomic orbital into gaussian 
        atomic orbitals
    coef2 : numpy array or list of floats 
        Expansion coefficients of second slater atomic orbital into gaussian 
        atomic orbitals
    exp1 : numpy array or list of floats 
        Exponents of gaussian orbitals in expansion of first slater atomic orbital.
    exp2 : numpy array or list of floats 
        Exponents of gaussian orbitals in expansion of second slater atomic orbital.
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first atomic orbital. For exaple for cartesian 
        atomic orbitals: dxz orbital is defined as ``index1=[1,0,1]``,
        dyy orbital as ``index1=[0,2,0]``, px orbital as ``index1=[1,0,0]``,
        s orbital as ``index1=[0,0,0]`` and so on...
        For the 5d orbital different indexing and decomposition into cartesian
        orbitals is used:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of the second atomic orbital. (See **index1** for 
        the definition)
    
    Returns
    -----------
    overlap : float
        Overlap between two slater atomic orbitals
    
    Definition
    -----------
    ``overlap = sum_{n1,n2}{ coef1[n1]*coef2[n2]*norm_GTO(ri,exp1[n1],index1)``
    ``* norm_GTO(rj,exp2[n2],index2) * int_{-infty}^{infty}{ ``
    ``(x-xi)^index1[0] *(y-yi)^index1[1] * (z-zi[2])^index1[2]``
    ``* (x-xj)^index2[0] *(y-yj)^index2[1] * (z-zj[2])^index2[2]``
    ``* e^(-exp1[n1]*|r-ri|^2) * e^(-exp2[n2]*|r-rj|^2) dri drj } }`` 
    
    Notes
    -------------
    This function is using recurent relations defined in **http://www.mathematica-journal.com/2012/02/evaluation-of-gaussian-molecular-integrals/**
    Transformation of spherical harmonic gaussian orbitals into cartesian ones 
    is taken from definitions is Gaussian 09 quantum chemistry software.
    
    '''

    if len(coef1)!=len(exp1) and len(coef2)!=len(exp2):
        raise IOError('Dimension error in overlap_STO')
    overlap=0.0
    for ii in range(len(coef1)):
        for jj in range(len(coef2)):
            #overlap += coef1[ii]*coef2[jj]*norm_GTO(ri,exp1[ii],index1)*norm_GTO(rj,exp2[jj],index2)*overlap_GTO(ri,rj,exp1[ii],exp2[jj],index1,index2)
            overlap += coef1[ii]*coef2[jj]*norm_GTO(ri,exp1[ii],index1)*norm_GTO(rj,exp2[jj],index2)*overlap_GTO_MathJour(ri,rj,exp1[ii],exp2[jj],index1,index2)
    return overlap
    

def norm_GTO(r,a,index):
    ''' Calculates norm of gaussian atomic orbital
    
    Parameters
    -------------
    r : numpy array (dimension 3)
        Position of the atomic orbital center in ATOMIC UNITS (Bohr)
    a : float
        Exponent of the atomic gaussian orbital
    index : numpy array or list of integer (dimension 3)
        Spatial orientation of the gaussian atomic orbital. For exaple for cartesian 
        gaussian atomic orbitals: dxz orbital is defined as ``index1=[1,0,1]``,
        dyy orbital as ``index1=[0,2,0]``, px orbital as ``index1=[1,0,0]``,
        s orbital as ``index1=[0,0,0]`` and so on...
        For the 5d orbital different indexing and decomposition into cartesian
        orbitals is used:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    Returns
    -------------
    norm : float
        Norm of the gaussian atomic orbital.
        
    Notes
    -------------
    Norm of the gaussian orbital is defined as 1/sqrt(overlap of gaussian atomic orbital
    with itself).
    
    '''
    
    #norm=1/numpy.sqrt(overlap_GTO(r,r,a,a,index,index))
    norm=1/numpy.sqrt(overlap_GTO_MathJour(r,r,a,a,index,index))
    return norm
    

def norm_STO(r,coef,exp,index):
    ''' Calculates norm of slater atomic orbital
    
    Parameters
    -------------
    r : numpy array (dimension 3)
        Position of the slater atomic orbital center in ATOMIC UNITS (Bohr)
    coef1 : numpy array or list of floats 
        Expansion coefficients of the slater atomic orbital into gaussian 
        atomic orbitals
    exp1 : numpy array or list of floats 
        Exponents of gaussian orbitals in expansion of the slater atomic orbital.
    index : numpy array or list of integer (dimension 3)
        Spatial orientation of the slater atomic orbital. For exaple for cartesian 
        atomic orbitals: dxz orbital is defined as ``index1=[1,0,1]``,
        dyy orbital as ``index1=[0,2,0]``, px orbital as ``index1=[1,0,0]``,
        s orbital as ``index1=[0,0,0]`` and so on...
        For the 5d orbital different indexing and decomposition into cartesian
        orbitals is used:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    Returns
    -----------
    norm : float
        Norm of the slater atomic orbital.
        
    Notes
    -------------
    Norm of the slater orbital is defined as 1/sqrt(overlap of slater atomic orbital
    with itself).
    '''
    
    norm=1/numpy.sqrt(overlap_STO(r,r,coef,coef,exp,exp,index,index))
    return norm


# =============================================================================
# Dipoles between atomic orbitals
# =============================================================================

def _dipole_GTO_MathJour_RegType(ri,rj,a,b,index1,index2):
    ''' Calculate dipole moment between two cartesian gaussian atomic orbitals
    or between spherical harmonics 5d atomic orbital and any cartesian atomic 
    orbital (5d stomic orbital must be the first one).

    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr) 
    a : float
        Exponent of first atomic gaussian orbital
    b : float
        Exponent of second atomic gaussian orbital
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first atomic orbital. For exaple for cartesian 
        gaussian atomic orbitals: dxz orbital is defined as ``index1=[1,0,1]``,
        dyy orbital as ``index1=[0,2,0]``, px orbital as ``index1=[1,0,0]``,
        s orbital as ``index1=[0,0,0]`` and so on...
        For the 5d orbital different indexing and decomposition into cartesian
        orbitals is used:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of the second atomic orbital. (See **index1** for 
        the definition)
        
    Returns
    -----------
    dipole : numpy array of real (dimension 3)
        Dipole between two atomic orbitals

    Definition
    -----------
    ``dipole[:] = (pi/(a+b))^(3/2) `` \n
    ``dipole[0] = dipole[0] * int_{-infty}^{infty}{ x * (x-ri[0])^index1[0] * (x-rj[0])^index2[0] * e^(-a*|x-ri[0]|^2) * e^(-b*|x-rj[0]|^2) dx}`` \n
    ``dipole[0] = dipole[0] * int_{-infty}^{infty}{ (y-ri[1])^index1[1] * (y-rj[1])^index2[1] * e^(-a*|y-ri[1]|^2) * e^(-b*|y-rj[1]|^2) dy}`` \n
    ``dipole[0] = dipole[0] * int_{-infty}^{infty}{ (z-ri[2])^index1[2] * (z-rj[2])^index2[2] * e^(-a*|z-ri[2]|^2) * e^(-b*|z-rj[2]|^2) dz}``
    
    ``dipole[1] = dipole[1] * int_{-infty}^{infty}{ (x-ri[0])^index1[0] * (x-rj[0])^index2[0] * e^(-a*|x-ri[0]|^2) * e^(-b*|x-rj[0]|^2) dx}`` \n
    ``dipole[1] = dipole[1] * int_{-infty}^{infty}{ y * (y-ri[1])^index1[1] * (y-rj[1])^index2[1] * e^(-a*|y-ri[1]|^2) * e^(-b*|y-rj[1]|^2) dy}`` \n
    ``dipole[1] = dipole[1] * int_{-infty}^{infty}{ (z-ri[2])^index1[2] * (z-rj[2])^index2[2] * e^(-a*|z-ri[2]|^2) * e^(-b*|z-rj[2]|^2) dz}``
    
    ``dipole[2] = dipole[2] * int_{-infty}^{infty}{ (x-ri[0])^index1[0] * (x-rj[0])^index2[0] * e^(-a*|x-ri[0]|^2) * e^(-b*|x-rj[0]|^2) dx}`` \n
    ``dipole[2] = dipole[2] * int_{-infty}^{infty}{ (y-ri[1])^index1[1] * (y-rj[1])^index2[1] * e^(-a*|y-ri[1]|^2) * e^(-b*|y-rj[1]|^2) dy}`` \n
    ``dipole[2] = dipole[2] * int_{-infty}^{infty}{ z * (z-ri[2])^index1[2] * (z-rj[2])^index2[2] * e^(-a*|z-ri[2]|^2) * e^(-b*|z-rj[2]|^2) dz}``
    
    '''
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index2)<0:
        raise IOError('Wrong imput parameters in function _dipole_GTO_MathJour_RegType')

    dipole=numpy.zeros(3,dtype='f8')
    for ii in range(3):
        index_dip=numpy.zeros(3)
        index_dip[ii]=1
        dipole[ii]=_overlap_GTO_MathJour_RegType(ri,rj,a,b,index1,numpy.add(index2,index_dip)) + rj[ii]*_overlap_GTO_MathJour_RegType(ri,rj,a,b,index1,index2)
    return dipole

def _dipole_GTO_MathJour_5D5D(ri,rj,a,b,index1,index2):
    """ Calculate dipole moment between two spherical harmonics 5d atomic 
    orbitals.

    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr) 
    a : float
        Exponent of first atomic gaussian orbital
    b : float
        Exponent of second atomic gaussian orbital
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first 5d atomic orbital. Definition
        of indexes for 5d orbital is:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of the second 5d atomic orbital. (See **index1** for 
        the definition)
        
    Returns
    -----------
    dipole : numpy array of real (dimension 3)
        Dipole between two 5d spherical harmonic atomic orbitals
    
    """
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)!=(-2) or numpy.sum(index2)!=(-2):
        raise IOError('Wrong imput parameters in function _dipole_GTO_MathJour_5D5D') 
    
    dipole=numpy.zeros(3,dtype='f8')
    if numpy.dot(index1,index1)==2 and numpy.dot(index2,index2)==2:
        for ii in range(3):
            index_dip=numpy.zeros(3)
            index_dip[ii]=1
            dipole[ii]=overlap_GTO_MathJour(ri,rj,a,b,numpy.add(numpy.abs(index1),index_dip),numpy.abs(index2)) + ri[ii]*overlap_GTO_MathJour(ri,rj,a,b,numpy.abs(index1),numpy.abs(index2))
    elif numpy.dot(index1,index1)==4 and numpy.dot(index2,index2)==2:
        if index1[0]==(-2):
            for ii in range(3):
                index_dip=numpy.zeros(3)
                index_dip[ii]=1
                indx2=numpy.add(numpy.abs(index2),index_dip)
                dipole[ii]=2*overlap_GTO_MathJour(ri,rj,a,b,[0,0,2],indx2)-overlap_GTO_MathJour(ri,rj,a,b,[0,2,0],indx2)-overlap_GTO_MathJour(ri,rj,a,b,[2,0,0],indx2)
                dipole[ii] += rj[ii]*(2*overlap_GTO_MathJour(ri,rj,a,b,[0,0,2],numpy.abs(index2))-overlap_GTO_MathJour(ri,rj,a,b,[0,2,0],numpy.abs(index2))-overlap_GTO_MathJour(ri,rj,a,b,[2,0,0],numpy.abs(index2)))
        elif index1[1]==(-2):
            for ii in range(3):
                index_dip=numpy.zeros(3)
                index_dip[ii]=1
                indx2=numpy.add(numpy.abs(index2),index_dip)
                dipole[ii]=overlap_GTO_MathJour(ri,rj,a,b,[2,0,0],indx2)-overlap_GTO_MathJour(ri,rj,a,b,[0,2,0],indx2)
                dipole[ii] += rj[ii]*(overlap_GTO_MathJour(ri,rj,a,b,[2,0,0],numpy.abs(index2))-overlap_GTO_MathJour(ri,rj,a,b,[0,2,0],numpy.abs(index2)))
        else:
            raise IOError('Error in _dipole_GTO_MathJour_5D5D')
    elif numpy.dot(index1,index1)==2 and numpy.dot(index2,index2)==4:
        if index2[0]==(-2):
            for ii in range(3):
                index_dip=numpy.zeros(3)
                index_dip[ii]=1
                indx1=numpy.add(numpy.abs(index1),index_dip)
                dipole[ii]=2*overlap_GTO_MathJour(ri,rj,a,b,indx1,[0,0,2])-overlap_GTO_MathJour(ri,rj,a,b,indx1,[0,2,0])-overlap_GTO_MathJour(ri,rj,a,b,indx1,[2,0,0])
                dipole[ii] += ri[ii]*(2*overlap_GTO_MathJour(ri,rj,a,b,numpy.abs(index1),[0,0,2])-overlap_GTO_MathJour(ri,rj,a,b,numpy.abs(index1),[0,2,0])-overlap_GTO_MathJour(ri,rj,a,b,numpy.abs(index1),[2,0,0]))
        elif index2[1]==(-2):
            for ii in range(3):
                index_dip=numpy.zeros(3)
                index_dip[ii]=1
                indx1=numpy.add(numpy.abs(index1),index_dip)
                dipole[ii]=overlap_GTO_MathJour(ri,rj,a,b,indx1,[2,0,0])-overlap_GTO_MathJour(ri,rj,a,b,indx1,[0,2,0])
                dipole[ii] += ri[ii]*(overlap_GTO_MathJour(ri,rj,a,b,numpy.abs(index1),[2,0,0])-overlap_GTO_MathJour(ri,rj,a,b,numpy.abs(index1),[0,2,0]))
        else:
            raise IOError('Error in _dipole_GTO_MathJour_5D5D')
    elif numpy.dot(index1,index1)==4 and numpy.dot(index2,index2)==4:
        if index1[0]==(-2) and index2[0]==(-2):
            for ii in range(3):
                index_dip=numpy.zeros(3)
                index_dip[ii]=1
                dipole[ii]=4*overlap_GTO_MathJour(ri,rj,a,b,numpy.add([0,0,2],index_dip),[0,0,2]) - 2*overlap_GTO_MathJour(ri,rj,a,b,numpy.add([0,0,2],index_dip),[0,2,0]) - 2*overlap_GTO_MathJour(ri,rj,a,b,numpy.add([0,0,2],index_dip),[2,0,0])
                dipole[ii] += -2*overlap_GTO_MathJour(ri,rj,a,b,numpy.add([0,2,0],index_dip),[0,0,2]) + overlap_GTO_MathJour(ri,rj,a,b,numpy.add([0,2,0],index_dip),[0,2,0]) + overlap_GTO_MathJour(ri,rj,a,b,numpy.add([0,2,0],index_dip),[2,0,0])
                dipole[ii] += -2*overlap_GTO_MathJour(ri,rj,a,b,numpy.add([2,0,0],index_dip),[0,0,2]) + overlap_GTO_MathJour(ri,rj,a,b,numpy.add([2,0,0],index_dip),[0,2,0]) + overlap_GTO_MathJour(ri,rj,a,b,numpy.add([2,0,0],index_dip),[2,0,0])
                dipole[ii] += ri[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,index2)
        elif index1[1]==(-2) and index2[1]==(-2):
            for ii in range(3):
                index_dip=numpy.zeros(3)
                index_dip[ii]=1
                dipole[ii] = overlap_GTO_MathJour(ri,rj,a,b,numpy.add([2,0,0],index_dip),[2,0,0]) - overlap_GTO_MathJour(ri,rj,a,b,numpy.add([2,0,0],index_dip),[0,2,0])
                dipole[ii] += -overlap_GTO_MathJour(ri,rj,a,b,numpy.add([0,2,0],index_dip),[2,0,0]) + overlap_GTO_MathJour(ri,rj,a,b,numpy.add([0,2,0],index_dip),[0,2,0])
                dipole[ii] += ri[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,index2)
        elif index1[0]==(-2) and index2[1]==(-2):
            for ii in range(3):
                index_dip=numpy.zeros(3)
                index_dip[ii]=1
                dipole[ii] = 2*overlap_GTO_MathJour(ri,rj,a,b,numpy.add([0,0,2],index_dip),[2,0,0]) - 2*overlap_GTO_MathJour(ri,rj,a,b,numpy.add([0,0,2],index_dip),[0,2,0]) - overlap_GTO_MathJour(ri,rj,a,b,numpy.add([0,2,0],index_dip),[2,0,0])
                dipole[ii] += overlap_GTO_MathJour(ri,rj,a,b,numpy.add([0,2,0],index_dip),[0,2,0]) - overlap_GTO_MathJour(ri,rj,a,b,numpy.add([2,0,0],index_dip),[2,0,0]) + overlap_GTO_MathJour(ri,rj,a,b,numpy.add([2,0,0],index_dip),[0,2,0])
                dipole[ii] += ri[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,index2)
        elif index1[1]==(-2) and index2[0]==(-2):
            for ii in range(3):
                index_dip=numpy.zeros(3)
                index_dip[ii]=1
                dipole[ii] = 2*overlap_GTO_MathJour(ri,rj,a,b,numpy.add([2,0,0],index_dip),[0,0,2]) - overlap_GTO_MathJour(ri,rj,a,b,numpy.add([2,0,0],index_dip),[0,2,0]) - overlap_GTO_MathJour(ri,rj,a,b,numpy.add([2,0,0],index_dip),[2,0,0])
                dipole[ii] += -2*overlap_GTO_MathJour(ri,rj,a,b,numpy.add([0,2,0],index_dip),[0,0,2]) + overlap_GTO_MathJour(ri,rj,a,b,numpy.add([0,2,0],index_dip),[0,2,0]) + overlap_GTO_MathJour(ri,rj,a,b,numpy.add([0,2,0],index_dip),[2,0,0])
        else:
            raise IOError('Error in _dipole_GTO_MathJour_5D5D')
    else:
        raise IOError('Error in _dipole_GTO_MathJour_5D5D')
        
    
    
#----------------------- Probably wrong definition of 7f state ---------------------------
#------------------------------ DON'T USE THIS BLOCK -------------------------------------
        
def _dipole_GTO_MathJour_7D5D(ri,rj,a,b,index1,index2):
    ''' Should calculate dipole moment between 7f spherical harmonic gaussian orbital
    and 5d spherical harmonics orbital orbital (7D must be the first = with index1).
    However definition of decomposition of 7f atomic orbital into cartesian 
    gaussian atomic orbital is wrong.
    
    **DO NOT USE IT FOR ANY CALCULATION**
    '''
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index2)!=(-2) or numpy.sum(index1)!=(-3):
        raise IOError('Wrong imput parameters in function _dipole_GTO_MathJour_RegType')

    dipole=numpy.zeros(3,dtype='f8')
    if index2[0]==(-2):
        #3ZZ-RR
        for ii in range(3):
            index_dip=numpy.zeros(3)
            index_dip[ii]=1
            dipole[ii] = 2*(overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([0,0,2],index_dip))+rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[0,0,2]))
            dipole[ii] -= (overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([0,2,0],index_dip))+rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[0,2,0]))
            dipole[ii] -= (overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([2,0,0],index_dip))+rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[2,0,0]))
    elif index2[0]==(-1) and index2[2]==(-1):
        #XZ
        for ii in range(3):
            index_dip=numpy.zeros(3)
            index_dip[ii]=1
            dipole[ii] = overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([1,0,1],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[1,0,1])
    elif index2[1]==(-1) and index2[2]==(-1):
        #YZ
        for ii in range(3):
            index_dip=numpy.zeros(3)
            index_dip[ii]=1
            dipole[ii] = overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([0,1,1],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[0,1,1])
    elif index2[1]==(-2):
        #XX-YY
        for ii in range(3):
            index_dip=numpy.zeros(3)
            index_dip[ii]=1
            dipole[ii] = overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([2,0,0],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[2,0,0])
            dipole[ii] -= (overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([0,2,0],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[0,2,0]))
    elif index2[0]==(-1) and index2[1]==(-1):
        #XY
        for ii in range(3):
            index_dip=numpy.zeros(3)
            index_dip[ii]=1
            dipole[ii] = overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([1,1,0],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[1,1,0])
    else:
        raise IOError('Error in _dipole_GTO_MathJour_7D5D')
    
    return dipole
    
def _dipole_GTO_MathJour_SpecType7D(ri,rj,a,b,index1,index2):
    ''' Should calculate dipole moment between two 7f spherical harmonic gaussian
    orbitals. However definition of decomposition of 7f atomic orbital into cartesian 
    gaussian atomic orbital is wrong.
    
    **DO NOT USE IT FOR ANY CALCULATION**
    '''
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index2)!=(-3) or (numpy.sum(index1)!=(-3) or numpy.sum(index1)!=(-2)):
        raise IOError('Wrong imput parameters in function _dipole_GTO_MathJour_RegType')

    dipole=numpy.zeros(3,dtype='f8')
    if index2[2]==(-3):
        #ZZZ-ZRR
        for ii in range(3):
            index_dip=numpy.zeros(3)
            index_dip[ii]=1
            dipole[ii] = -(overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([2,0,1],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[2,0,1]))
            dipole[ii] -= (overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([0,2,1],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[0,2,1]))
    elif index2[0]==(-1) and index2[2]==(-2):
        #XZZ-XRR
        for ii in range(3):
            index_dip=numpy.zeros(3)
            index_dip[ii]=1
            dipole[ii] = -(overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([3,0,0],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[3,0,0]))
            dipole[ii] -= (overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([1,2,0],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[1,2,0]))
    elif index2[1]==(-1) and index2[2]==(-2):
        #YZZ-YRR
        for ii in range(3):
            index_dip=numpy.zeros(3)
            index_dip[ii]=1
            dipole[ii] = -(overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([2,1,0],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[2,1,0]))
            dipole[ii] -= (overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([0,3,0],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[0,3,0]))
    elif index2[0]==(-2) and index2[2]==(-1):
        #XXZ-YYZ
        for ii in range(3):
            index_dip=numpy.zeros(3)
            index_dip[ii]=1
            dipole[ii] = overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([2,0,1],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[2,0,1])
            dipole[ii] -= (overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([0,2,1],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[0,2,1]))
    elif index2[0]==(-1) and index2[1]==(-1) and index2[2]==(-1):
        #XYZ
        for ii in range(3):
            index_dip=numpy.zeros(3)
            index_dip[ii]=1
            dipole[ii] = overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([1,1,1],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[1,1,1])
    elif index2[0]==(-3):
        #XXX-XYY
        for ii in range(3):
            index_dip=numpy.zeros(3)
            index_dip[ii]=1
            dipole[ii] = overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([3,0,0],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[3,0,0])
            dipole[ii] -= (overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([1,2,0],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[1,2,0]))
    elif index2[0]==(-2) and index2[1]==(-1):
        #XXY-YYY
        for ii in range(3):
            index_dip=numpy.zeros(3)
            index_dip[ii]=1
            dipole[ii] = overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([2,1,0],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[2,1,0])
            dipole[ii] -= (overlap_GTO_MathJour(ri,rj,a,b,index1,numpy.add([0,3,0],index_dip)) + rj[ii]*overlap_GTO_MathJour(ri,rj,a,b,index1,[0,3,0])) 
    else:
        raise IOError('Error in _dipole_GTO_MathJour_SpecType7D')
        
    return dipole
#---------------------- END OF THE WRONG DEFINITION OF 7f STATE -------------------------------------    


def dipole_GTO_MathJour(ri,rj,a,b,index1,index2):
    ''' Calculate dipole moment between two gaussian atomic orbitals.
    Supported orbital types are s,p,d,f... and one special orbital 5d
    
    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr) 
    a : float
        Exponent of first atomic gaussian orbital
    b : float
        Exponent of second atomic gaussian orbital
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first atomic orbital. For exaple for cartesian 
        gaussian atomic orbitals: dxz orbital is defined as ``index1=[1,0,1]``,
        dyy orbital as ``index1=[0,2,0]``, px orbital as ``index1=[1,0,0]``,
        s orbital as ``index1=[0,0,0]`` and so on...
        For the 5d orbital different indexing and decomposition into cartesian
        orbitals is used:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of the second atomic orbital. (See **index1** for 
        the definition)
    
    Returns
    -----------
    dipole : numpy array of real (dimension 3)
        Dipole between two atomic orbitals
    
    Definition
    -----------
    ``dipole[:] = (pi/(a+b))^(3/2) `` \n
    ``dipole[0] = dipole[0] * int_{-infty}^{infty}{ x * (x-ri[0])^index1[0] * (x-rj[0])^index2[0] * e^(-a*|x-ri[0]|^2) * e^(-b*|x-rj[0]|^2) dx}`` \n
    ``dipole[0] = dipole[0] * int_{-infty}^{infty}{ (y-ri[1])^index1[1] * (y-rj[1])^index2[1] * e^(-a*|y-ri[1]|^2) * e^(-b*|y-rj[1]|^2) dy}`` \n
    ``dipole[0] = dipole[0] * int_{-infty}^{infty}{ (z-ri[2])^index1[2] * (z-rj[2])^index2[2] * e^(-a*|z-ri[2]|^2) * e^(-b*|z-rj[2]|^2) dz}``
    
    ``dipole[1] = dipole[1] * int_{-infty}^{infty}{ (x-ri[0])^index1[0] * (x-rj[0])^index2[0] * e^(-a*|x-ri[0]|^2) * e^(-b*|x-rj[0]|^2) dx}`` \n
    ``dipole[1] = dipole[1] * int_{-infty}^{infty}{ y * (y-ri[1])^index1[1] * (y-rj[1])^index2[1] * e^(-a*|y-ri[1]|^2) * e^(-b*|y-rj[1]|^2) dy}`` \n
    ``dipole[1] = dipole[1] * int_{-infty}^{infty}{ (z-ri[2])^index1[2] * (z-rj[2])^index2[2] * e^(-a*|z-ri[2]|^2) * e^(-b*|z-rj[2]|^2) dz}``
    
    ``dipole[2] = dipole[2] * int_{-infty}^{infty}{ (x-ri[0])^index1[0] * (x-rj[0])^index2[0] * e^(-a*|x-ri[0]|^2) * e^(-b*|x-rj[0]|^2) dx}`` \n
    ``dipole[2] = dipole[2] * int_{-infty}^{infty}{ (y-ri[1])^index1[1] * (y-rj[1])^index2[1] * e^(-a*|y-ri[1]|^2) * e^(-b*|y-rj[1]|^2) dy}`` \n
    ``dipole[2] = dipole[2] * int_{-infty}^{infty}{ z * (z-ri[2])^index1[2] * (z-rj[2])^index2[2] * e^(-a*|z-ri[2]|^2) * e^(-b*|z-rj[2]|^2) dz}``

    Notes
    -------------
    This function is using recurent relations defined in **http://www.mathematica-journal.com/2012/02/evaluation-of-gaussian-molecular-integrals/**
    Transformation of spherical harmonic gaussian orbitals into cartesian ones 
    is taken from definitions is Gaussian 09 quantum chemistry software.
    
    '''
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3:
        raise IOError('Wrong imput parameters in function dipole_GTO_MathJour')
    
    dipole=numpy.zeros(3,dtype='f8')
    if numpy.sum(index2)>=0:
        dipole = _dipole_GTO_MathJour_RegType(ri,rj,a,b,index1,index2)
    elif numpy.sum(index1)>=0: # is second orbital is 5D move this orbital to first place
        dipole = _dipole_GTO_MathJour_RegType(rj,ri,b,a,index2,index1)
    elif numpy.sum(index1)==-2 and numpy.sum(index2)==-2:
        dipole = _dipole_GTO_MathJour_5D5D(ri,rj,a,b,index1,index2)
    elif (numpy.sum(index1)==-2 or numpy.sum(index1)==-3) and numpy.sum(index2)==-3:
        dipole = _dipole_GTO_MathJour_SpecType7D(ri,rj,a,b,index1,index2)
    elif (numpy.sum(index2)==-2 or numpy.sum(index2)==-3) and numpy.sum(index1)==-3:
        dipole = _dipole_GTO_MathJour_SpecType7D(rj,ri,b,a,index2,index1)
    else:
        raise IOError('Unsupported type of gaussian orbitals for calculation of overlap')
    
    return dipole 


def dipole_STO(ri,rj,coef1,coef2,exp1,exp2,index1,index2):
    ''' Calculate dipole moment between two slater atomic orbitals.
    Supported orbital types are s,p,d,f... and one special orbital 5d
    
    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr)
    coef1 : numpy array or list of floats 
        Expansion coefficients of first slater atomic orbital into gaussian 
        atomic orbitals
    coef2 : numpy array or list of floats 
        Expansion coefficients of second slater atomic orbital into gaussian 
        atomic orbitals
    exp1 : numpy array or list of floats 
        Exponents of gaussian orbitals in expansion of first slater atomic orbital.
    exp2 : numpy array or list of floats 
        Exponents of gaussian orbitals in expansion of second slater atomic orbital.
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first atomic orbital. For exaple for cartesian 
        atomic orbitals: dxz orbital is defined as ``index1=[1,0,1]``,
        dyy orbital as ``index1=[0,2,0]``, px orbital as ``index1=[1,0,0]``,
        s orbital as ``index1=[0,0,0]`` and so on...
        For the 5d orbital different indexing and decomposition into cartesian
        orbitals is used:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of the second atomic orbital. (See **index1** for 
        the definition)
    
    Returns
    -----------
    dipole : numpy array of real (dimension 3)
        Dipole moment between two atomic orbitals
    
    Definition
    -----------
    ``overlap = sum_{n1,n2}{ coef1[n1]*coef2[n2]*norm_GTO(ri,exp1[n1],index1)``
    ``* norm_GTO(rj,exp2[n2],index2) * dipole_GTO_MathJour(ri,rj,a,b,index1,index2) `` \n
    More can be found in :ref:func:`dipole_GTO_MathJour`

    Notes
    -------------
    This function is using recurent relations defined in **http://www.mathematica-journal.com/2012/02/evaluation-of-gaussian-molecular-integrals/**
    Transformation of spherical harmonic gaussian orbitals into cartesian ones 
    is taken from definitions is Gaussian 09 quantum chemistry software.
    '''
    
    if len(coef1)!=len(exp1) and len(coef2)!=len(exp2):
        raise IOError('Dimension error in overlap_STO')
    dipole=numpy.zeros(3,dtype='f8')
    for ii in range(len(coef1)):
        for jj in range(len(coef2)):
            #dipole += coef1[ii]*coef2[jj]*norm_GTO(ri,exp1[ii],index1)*norm_GTO(rj,exp2[jj],index2)*numpy.array(dipole_GTO(ri,rj,exp1[ii],exp2[jj],index1,index2))
            dipole += coef1[ii]*coef2[jj]*norm_GTO(ri,exp1[ii],index1)*norm_GTO(rj,exp2[jj],index2)*numpy.array(dipole_GTO_MathJour(ri,rj,exp1[ii],exp2[jj],index1,index2))
    return dipole



# =============================================================================
# Quadrupole calculation
# =============================================================================
def _quadrupole_r2_GTO_MathJour_RegType(ri,rj,a,b,index1,index2):
    ''' calculate quadrupole (XX+YY+ZZ) between two cartesian gaussian atomic orbitals or
    between 5d spherical harmonics gaussian orbital and any cartesian gaussian
    atomic orbital (5d orbital must be the first one = with index1)
    
    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr) 
    a : float
        Exponent of first atomic gaussian orbital
    b : float
        Exponent of second atomic gaussian orbital
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first atomic orbital. For exaple for cartesian 
        gaussian atomic orbitals: dxz orbital is defined as ``index1=[1,0,1]``,
        dyy orbital as ``index1=[0,2,0]``, px orbital as ``index1=[1,0,0]``,
        s orbital as ``index1=[0,0,0]`` and so on...
        For the 5d orbital different indexing and decomposition into cartesian
        orbitals is used:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of the second atomic orbital. For exaple for cartesian 
        gaussian atomic orbitals: dxz orbital is defined as ``index1=[1,0,1]``,
        dyy orbital as ``index1=[0,2,0]``, px orbital as ``index1=[1,0,0]``,
        s orbital as ``index1=[0,0,0]`` and so on...
    
    
    Returns
    -----------
    quadrupole : float
        outputs quadrupole which corresponds to r^2 (XX+YY+ZZ)
    
    Notes
    -----------
    The result should be the same as from 
    _quadrupole_GTO_MathJour_RegType(ri,rj,a,b,index1,index2) where we sum 0th, 
    3rd and 5th element of the result
    
        
    '''
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index2)<0:
        raise IOError('Wrong imput parameters in function _dipole_GTO_MathJour_RegType')

    quadrupole=0.0
    for ii in range(3):
        index_dip=numpy.zeros(3)
        index_dip[ii]=2
        quadrupole+=_overlap_GTO_MathJour_RegType(ri,rj,a,b,index1,numpy.add(index2,index_dip))
        index_dip[ii]=1
        quadrupole+= 2*rj[ii]*_overlap_GTO_MathJour_RegType(ri,rj,a,b,index1,numpy.add(index2,index_dip))
        quadrupole+= (rj[ii]**2)*_overlap_GTO_MathJour_RegType(ri,rj,a,b,index1,index2)
    return quadrupole


# TODO: Add posibility to read overlaps and dipoles to save time
def _quadrupole_GTO_MathJour_RegType(ri,rj,a,b,index1,index2,**kwargs):
    ''' calculate quadrupole between two cartesian gaussian atomic orbitals or
    between 5d spherical harmonics gaussian orbital and any cartesian gaussian
    atomic orbital (5d orbital bust be the second one = with index2)
    
    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr) 
    a : float
        Exponent of first atomic gaussian orbital
    b : float
        Exponent of second atomic gaussian orbital
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first atomic orbital. For exaple for cartesian 
        gaussian atomic orbitals: dxz orbital is defined as ``index1=[1,0,1]``,
        dyy orbital as ``index1=[0,2,0]``, px orbital as ``index1=[1,0,0]``,
        s orbital as ``index1=[0,0,0]`` and so on...
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first atomic orbital. For exaple for cartesian 
        gaussian atomic orbitals: dxz orbital is defined as ``index1=[1,0,1]``,
        dyy orbital as ``index1=[0,2,0]``, px orbital as ``index1=[1,0,0]``,
        s orbital as ``index1=[0,0,0]`` and so on...
        For the 5d orbital different indexing and decomposition into cartesian
        orbitals is used:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    Returns
    -----------
    quadrupole : numpy array of real (dimension 6)
        outputs quadrupole between two gaussian atomic orbitals. Ordering of
        quadrupoles is [XX,XY,XZ,YY,YZ,ZZ]
        
    '''
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)<0:
        raise IOError('Wrong imput parameters in function _dipole_GTO_MathJour_RegType')

    quadrupole=numpy.zeros(6,dtype='f8')
    #X^2, Y^2, Z^2
    counter=0
    if numpy.dot(ri,ri)!=0.0:
        for ii in range(3):
            for jj in range(ii,3):
                index_quad=numpy.zeros(3)
                index_dip1=numpy.zeros(3)
                index_dip2=numpy.zeros(3)
                index_dip1[ii]+=1
                index_quad[ii]+=1
                index_dip2[jj]+=1
                index_quad[jj]+=1
                quadrupole[counter] = _overlap_GTO_MathJour_RegType(ri,rj,a,b,numpy.add(index1,index_quad),index2)
                quadrupole[counter] += ri[ii]*_overlap_GTO_MathJour_RegType(ri,rj,a,b,numpy.add(index1,index_dip2),index2)
                quadrupole[counter] += ri[jj]*_overlap_GTO_MathJour_RegType(ri,rj,a,b,numpy.add(index1,index_dip1),index2)
                quadrupole[counter] += ri[ii]*ri[jj]*_overlap_GTO_MathJour_RegType(ri,rj,a,b,index1,index2)
                counter+=1
    else:
        for ii in range(3):
            for jj in range(ii,3):
                index_quad=numpy.zeros(3)
                index_dip1=numpy.zeros(3)
                index_dip2=numpy.zeros(3)
                index_dip1[ii]+=1
                index_quad[ii]+=1
                index_dip2[jj]+=1
                index_quad[jj]+=1
                quadrupole[counter] = _overlap_GTO_MathJour_RegType(ri,rj,a,b,numpy.add(index1,index_quad),index2)
                counter+=1
            
        
    return quadrupole
    
def _quadrupole_GTO_MathJour_5D5D(ri,rj,a,b,index1,index2):
    ''' calculate quadrupole between two 5d spherical harmonics gaussian orbitals.
    
    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr) 
    a : float
        Exponent of first atomic gaussian orbital
    b : float
        Exponent of second atomic gaussian orbital
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first 5d atomic orbital. Definition
        of indexes for 5d orbital is:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of the second 5d atomic orbital. (See **index1** for 
        the definition)
    
    Returns
    -----------
    quadrupole : numpy array of real (dimension 6)
        outputs quadrupole between two 5d spherical harmonic gaussian atomic 
        orbitals. Ordering of quadrupoles is [XX,XY,XZ,YY,YZ,ZZ]
        
    '''
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index2)!=-2 or numpy.sum(index1)!=-2:
        raise IOError('Wrong imput parameters in function _dipole_GTO_MathJour_RegType')
    
    quadrupole=numpy.zeros(6,dtype='f8')
    if index1[0]==(-2):
        #3ZZ-RR
        quadrupole=2*_quadrupole_GTO_MathJour_RegType(ri,rj,a,b,[0,0,2],index2)-_quadrupole_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],index2)-_quadrupole_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],index2)
    elif index1[0]==(-1) and index1[2]==(-1):
        #XZ
        quadrupole=_quadrupole_GTO_MathJour_RegType(ri,rj,a,b,[1,0,1],index2)
    elif index1[1]==(-1) and index1[2]==(-1):
        #YZ
        quadrupole=_quadrupole_GTO_MathJour_RegType(ri,rj,a,b,[0,1,1],index2)
    elif index1[1]==(-2):
        #XX-YY
        quadrupole=_quadrupole_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],index2)-_quadrupole_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],index2)
    elif index1[0]==(-1) and index1[1]==(-1):
        #XY
        quadrupole=_quadrupole_GTO_MathJour_RegType(ri,rj,a,b,[1,1,0],index2)
    else:
        raise IOError('Error in _overlap_GTO_MathJour_5DRegType')
    return quadrupole
    
    
def _quadrupole_r2_GTO_MathJour_5Dall(ri,rj,a,b,index1,index2):
    ''' calculate quadrupole r^2 (XX+YY+ZZ) between 5d spherical harmonic
    gaussian orbital and any gaussian atomic orbital (including 5d).
    (5d orbital must be the first one = with index1)
    
    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr) 
    a : float
        Exponent of first atomic gaussian orbital
    b : float
        Exponent of second atomic gaussian orbital
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first 5d atomic orbital. Definition
        of indexes for 5d orbital is:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first atomic orbital. For exaple for cartesian 
        gaussian atomic orbitals: dxz orbital is defined as ``index1=[1,0,1]``,
        dyy orbital as ``index1=[0,2,0]``, px orbital as ``index1=[1,0,0]``,
        s orbital as ``index1=[0,0,0]`` and so on...
        For the 5d orbital different indexing and decomposition into cartesian
        orbitals is used:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    
    Returns
    -----------
    quadrupole : float
        outputs quadrupole which corresponds to r^2 (XX+YY+ZZ)
    
    Notes
    -----------
    The result should be the same as from 
    _quadrupole_GTO_MathJour_5D5D(ri,rj,a,b,index1,index2) where we sum 0th, 
    3rd and 5th element of the result
    
        
    '''
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)!=-2:
        raise IOError('Wrong imput parameters in function _dipole_GTO_MathJour_RegType')
    
    quadrupole=0.0
    if index1[0]==(-2):
        #3ZZ-RR
        quadrupole=2*_quadrupole_r2_GTO_MathJour_RegType(ri,rj,a,b,[0,0,2],index2)-_quadrupole_r2_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],index2)-_quadrupole_r2_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],index2)
    elif index1[0]==(-1) and index1[2]==(-1):
        #XZ
        quadrupole=_quadrupole_r2_GTO_MathJour_RegType(ri,rj,a,b,[1,0,1],index2)
    elif index1[1]==(-1) and index1[2]==(-1):
        #YZ
        quadrupole=_quadrupole_r2_GTO_MathJour_RegType(ri,rj,a,b,[0,1,1],index2)
    elif index1[1]==(-2):
        #XX-YY
        quadrupole=_quadrupole_r2_GTO_MathJour_RegType(ri,rj,a,b,[2,0,0],index2)-_quadrupole_r2_GTO_MathJour_RegType(ri,rj,a,b,[0,2,0],index2)
    elif index1[0]==(-1) and index1[1]==(-1):
        #XY
        quadrupole=_quadrupole_r2_GTO_MathJour_RegType(ri,rj,a,b,[1,1,0],index2)
    else:
        raise IOError('Error in _overlap_GTO_MathJour_5DRegType')
    return quadrupole


def quadrupole_GTO_MathJour(ri,rj,a,b,index1,index2):
    ''' calculate quadrupole between two cartesian gaussian atomic orbitals or
    between 5d spherical harmonics gaussian orbital (5d orbital bust be the
    second one = with index2)
    
    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr) 
    a : float
        Exponent of first atomic gaussian orbital
    b : float
        Exponent of second atomic gaussian orbital
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first atomic orbital. For exaple for cartesian 
        gaussian atomic orbitals: dxz orbital is defined as ``index1=[1,0,1]``,
        dyy orbital as ``index1=[0,2,0]``, px orbital as ``index1=[1,0,0]``,
        s orbital as ``index1=[0,0,0]`` and so on...
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of the second atomic orbital. For exaple for cartesian 
        gaussian atomic orbitals: dxz orbital is defined as ``index2=[1,0,1]``,
        dyy orbital as ``index2=[0,2,0]``, px orbital as ``index2=[1,0,0]``,
        s orbital as ``index2=[0,0,0]`` and so on...
        For the 5d orbital different indexing and decomposition into cartesian
        orbitals is used:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    Returns
    -----------
    quadrupole : numpy array of real (dimension 6)
        outputs quadrupole between two gaussian atomic orbitals. Ordering of
        quadrupoles is [XX,XY,XZ,YY,YZ,ZZ]
        
    '''
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3:
        raise IOError('Wrong imput parameters in function dipole_GTO_MathJour')
        
    # In this case I can not reorder orbitals because I would like to use this for calculation of situation where position of first orbital will be zero and this will simplify the calculation
    if numpy.sum(index1)>=0: 
        quadrupole = _quadrupole_GTO_MathJour_RegType(ri,rj,a,b,index1,index2)
    elif numpy.sum(index1)==-2 and (numpy.sum(index2)==-2 or numpy.sum(index2)>=0):
        quadrupole = _quadrupole_GTO_MathJour_5D5D(ri,rj,a,b,index1,index2)
    else:
        raise IOError('Unsupported type of gaussian orbitals for calculation of overlap')
    
    return quadrupole

    
def quadrupole_r2_GTO_MathJour(ri,rj,a,b,index1,index2):
    ''' calculate quadrupole (XX+YY+ZZ) between two cartesian gaussian atomic orbitals or
    5d spherical harmonics gaussian orbital (including 5d). If only one 5d
    orbital is present it must be at the second place = the one with index2)

    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr) 
    a : float
        Exponent of first atomic gaussian orbital
    b : float
        Exponent of second atomic gaussian orbital
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first atomic orbital. For exaple for cartesian 
        gaussian atomic orbitals: dxz orbital is defined as ``index1=[1,0,1]``,
        dyy orbital as ``index1=[0,2,0]``, px orbital as ``index1=[1,0,0]``,
        s orbital as ``index1=[0,0,0]`` and so on...
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of the second atomic orbital. For exaple for cartesian 
        gaussian atomic orbitals: dxz orbital is defined as ``index2=[1,0,1]``,
        dyy orbital as ``index2=[0,2,0]``, px orbital as ``index2=[1,0,0]``,
        s orbital as ``index2=[0,0,0]`` and so on...
        For the 5d orbital different indexing and decomposition into cartesian
        orbitals is used:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    Returns
    -----------
    quadrupole : float
        outputs quadrupole which corresponds to r^2 (XX+YY+ZZ)
    
    Notes
    -----------
    The result should be the same as from 
    quadrupole_GTO_MathJour(ri,rj,a,b,index1,index2) where we sum 0th, 
    3rd and 5th element of the result
        
    '''
    
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3:
        raise IOError('Wrong imput parameters in function dipole_GTO_MathJour')
    
    if numpy.sum(index2)>=0:
        quadrupole = _quadrupole_r2_GTO_MathJour_RegType(ri,rj,a,b,index1,index2)
    elif numpy.sum(index1)>=0: # is second orbital is 5D move this orbital to first place
        quadrupole = _quadrupole_r2_GTO_MathJour_RegType(rj,ri,b,a,index2,index1)
    elif numpy.sum(index1)==-2 and numpy.sum(index2)==-2:
        quadrupole = _quadrupole_r2_GTO_MathJour_5Dall(ri,rj,a,b,index1,index2)
    else:
        raise IOError('Unsupported type of gaussian orbitals for calculation of overlap')
    
    return quadrupole

def quadrupole_r2_STO(ri,rj,coef1,coef2,exp1,exp2,index1,index2):
    ''' calculate quadrupole r^2 (XX+YY+ZZ) between two slater atomic orbitals

    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr)
    coef1 : numpy array or list of floats 
        Expansion coefficients of first slater atomic orbital into gaussian 
        atomic orbitals
    coef2 : numpy array or list of floats 
        Expansion coefficients of second slater atomic orbital into gaussian 
        atomic orbitals
    exp1 : numpy array or list of floats 
        Exponents of gaussian orbitals in expansion of first slater atomic orbital.
    exp2 : numpy array or list of floats 
        Exponents of gaussian orbitals in expansion of second slater atomic orbital.
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first atomic orbital. For exaple for cartesian 
        atomic orbitals: dxz orbital is defined as ``index1=[1,0,1]``,
        dyy orbital as ``index1=[0,2,0]``, px orbital as ``index1=[1,0,0]``,
        s orbital as ``index1=[0,0,0]`` and so on...
        For the 5d orbital different indexing and decomposition into cartesian
        orbitals is used:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of the second atomic orbital. (See **index1** for 
        the definition)
    
    Returns
    -----------
    quadrupole : float
        Quadrupole which corresponds to r^2 (XX+YY+ZZ)
    
    Notes
    -----------
    The result should be the same as from 
    quadrupole_STO(ri,rj,coef1,coef2,exp1,exp2,index1,index2) where we sum 0th, 
    3rd and 5th element of the result
        
    '''
    
    if len(coef1)!=len(exp1) and len(coef2)!=len(exp2):
        raise IOError('Dimension error in overlap_STO')
    quadrupole=0.0
    for ii in range(len(coef1)):
        for jj in range(len(coef2)):
            quadrupole += coef1[ii]*coef2[jj]*norm_GTO(ri,exp1[ii],index1)*norm_GTO(rj,exp2[jj],index2)*quadrupole_r2_GTO_MathJour(ri,rj,exp1[ii],exp2[jj],index1,index2)
    return quadrupole


def quadrupole_STO(ri,rj,coef1,coef2,exp1,exp2,index1,index2):
    ''' calculate quadrupole moments between two slater atomic orbitals. 
    Supported orbital types are: s, p, d, f, ... and 5d

    Parameters
    -------------
    ri : numpy array (dimension 3)
        Position of the first atomic orbital center in ATOMIC UNITS (Bohr)
    rj : numpy array (dimension 3)
        Position of the second atomic orbital center in ATOMIC UNITS (Bohr)
    coef1 : numpy array or list of floats 
        Expansion coefficients of first slater atomic orbital into gaussian 
        atomic orbitals
    coef2 : numpy array or list of floats 
        Expansion coefficients of second slater atomic orbital into gaussian 
        atomic orbitals
    exp1 : numpy array or list of floats 
        Exponents of gaussian orbitals in expansion of first slater atomic orbital.
    exp2 : numpy array or list of floats 
        Exponents of gaussian orbitals in expansion of second slater atomic orbital.
    index1 : numpy array or list of integer (dimension 3)
        Spatial orientation of the first atomic orbital. For exaple for cartesian 
        atomic orbitals: dxz orbital is defined as ``index1=[1,0,1]``,
        dyy orbital as ``index1=[0,2,0]``, px orbital as ``index1=[1,0,0]``,
        s orbital as ``index1=[0,0,0]`` and so on...
        For the 5d orbital different indexing and decomposition into cartesian
        orbitals is used:
            
          * [-2,0,0] -> d0 where d0 = 2dzz-dxx-dyy
          * [-1,0,-1] -> d+1 where d+1 = dxz
          * [0,-1,-1] -> d-1 where d-1 = dyz
          * [0,-2,0] -> d+2 where d+2 = dxx-dyy
          * [-1,-1,0] -> d-2 where d-2 = dxy
    
    index2 : numpy array or list of integer (dimension 3)
        Spatial orientation of the second atomic orbital. (See **index1** for 
        the definition)
    
    Returns
    -----------
    quadrupole : numpy array of real (dimension 6)
        Quadrupole moments between two gaussian atomic orbitals. Ordering of
        quadrupoles is [XX,XY,XZ,YY,YZ,ZZ]
        
    '''
    
    if len(coef1)!=len(exp1) and len(coef2)!=len(exp2):
        raise IOError('Dimension error in overlap_STO')
    quadrupole=numpy.zeros(6,dtype='f8')
    for ii in range(len(coef1)):
        for jj in range(len(coef2)):
            #dipole += coef1[ii]*coef2[jj]*norm_GTO(ri,exp1[ii],index1)*norm_GTO(rj,exp2[jj],index2)*numpy.array(dipole_GTO(ri,rj,exp1[ii],exp2[jj],index1,index2))
            quadrupole += coef1[ii]*coef2[jj]*norm_GTO(ri,exp1[ii],index1)*norm_GTO(rj,exp2[jj],index2)*numpy.array(quadrupole_GTO_MathJour(ri,rj,exp1[ii],exp2[jj],index1,index2))
    return quadrupole

# =============================================================================
# Old type of the calculation (slower and not general)
# =============================================================================

def R_GTO(a,rr):
    ''' Radial part of s gaussian orbital.
    
    Parameters
    ----------
    rr : float
        Distance from the center of s-orbital (rr = |r|^2)
    a : float
        Exponent of gaussian function
    
    Return
    --------
    gto : float
        Value of radial part of gaussian s-orbital at distance rr
    
    Definition
    --------
    gto = (2*a/pi)^3/4 * e^(-a*rr)
    '''
    
    norm=numpy.power(2*a/numpy.pi,3/4)    
    gto=norm*numpy.exp(-a*rr)
    return gto
    
def _overlap_SS(ri,rj,a,b):
    ''' overlap between two gaussian s orbitals'''
    if len(ri)!=3 or len(rj)!=3:
        raise IOError('Position vectors in _overlap_SS must be 3D')
    res=numpy.exp(-a*b/(a+b)*numpy.dot(ri-rj,ri-rj))
    res=res*numpy.power(numpy.pi/(a+b),3/2)
    return res
    
def _overlap_PiS(ri,rj,a,b,index1,index2):
    ''' Overlap between p orbital and s orbital. First orbital have to be p (orbital with exponent a and position ri) !!!!!'''
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.dot(index1,index1)!=1 or numpy.sum(index2)!=0:
        raise IOError('Position vectors in _overlap_PiS must be 3D')
    res=_overlap_SS(ri,rj,a,b)
    for ii in range(3):
        if index1[ii]!=0:
            res=res*b/(a+b)*(rj[ii]-ri[ii])
    return res
    
def _overlap_PiPj(ri,rj,a,b,index1,index2):
    ''' Overlap between two p orbitals '''
    if len(ri)!=3 or len(rj)!=3 or len(index2)!=3 or len(index1)!=3 or numpy.dot(index1,index1)!=1 or numpy.dot(index2,index2)!=1:
        raise IOError('Position vectors in _overlap_PiPj must be 3D')
    for ii in range(3):
        if index1[ii]==1:
            indx1=ii
        if index2[ii]==1:
            indx2=ii
    if indx1!=indx2:
        res=_overlap_SS(ri,rj,a,b)
        res=res*(-2*a*b*(rj[indx1]-ri[indx1])*(rj[indx2]-ri[indx2]))/(2*((a+b)**2))
        #res = _overlap_SS(vecA, vecB, a, b)
	  #res = res*(-2*a*b*(vecB(i) - vecA(i))*(vecB(j) - vecA(j)))/(2*((a + b)**2))
    else:
        res=_overlap_SS(ri,rj,a,b)
        res=res*(a+b-2*a*b*(rj[indx1]-ri[indx1])*(rj[indx2]-ri[indx2]))/(2*((a+b)**2))
        #res = _overlap_SS(vecA, vecB, a, b)
	  #res = res*(a + b - 2*a*b*(vecB(i) - vecA(i))*(vecB(j) - vecA(j)))/(2*((a + b)**2))
    return res
    
def _overlap_DijS(ri,rj,a,b,index1,index2):
    ''' Overlap between d orbital and s orbital. First orbital have to be d (orbital with exponent a and position ri) !!!!!'''
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)!=2 or numpy.sum(index2)!=0:
        raise IOError('Position vectors in _overlap_DijS must be 3D')
    #print(ri,rj,a,b,index1,index2)
    if numpy.dot(index1,index1)==2:
        # mam 2 odlisne indexy xy,yz,xz
        res=_overlap_SS(ri,rj,a,b)
        for ii in range(3):
            if index1[ii]==1:
                res=res*b/(a+b)*(rj[ii]-ri[ii])
    elif numpy.dot(index1,index1)==4:
        # mam 1 index xx,yy,zz
        res=_overlap_SS(ri,rj,a,b)
        for ii in range(3):
            if index1[ii]==2:
                res=res*(a+b*(1+2*b*((ri[ii]-rj[ii])**2)))/(2*((a+b)**2))
                #(a + b (1 + 2 b (xi - xj)^2))/(2 (a + b)^2)
    return res
    
def _overlap_DijPk(ri,rj,a,b,index1,index2):
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)!=2 or numpy.sum(index2)!=1:
        raise IOError('Position vectors in _overlap_DijPk must be 3D')
    if numpy.dot(index1,index2)==0:
        res=_overlap_DijS(ri,rj,a,b,index1,[0,0,0])
        for ii in range(3):
            if index2[ii]==1:
                res=res*a/(a+b)*(ri[ii]-rj[ii])
    elif numpy.dot(index1,index2)==1:
        res=_overlap_PiPj(ri,rj,a,b,index2,index2)
        for ii in range(3):
            if index1[ii]==1 and index2[ii]!=1:
                res=res*b/(a+b)*(rj[ii]-ri[ii])
    elif numpy.dot(index1,index2)==2:
        res=_overlap_SS(ri,rj,a,b)
        for ii in range(3):
            if index1[ii]==2:
                res=res*((a**2) - 2*(b**2) + a*b*(-1+2*b*(ri[ii]-rj[ii])**2) )*(ri[ii]-rj[ii])/(2*((a+b)**3))
                # ((a^2 - 2 b^2 + a b (-1 + 2 b (xi - xj)^2)) (xi - xj))/(2 (a + b)^3)
    else:
        raise IOError('Error in _overlap_DijPk')
    
    return res
    
def _overlap_DijDkl(ri,rj,a,b,index1,index2):
    ''' Overlap between two d orbitals '''
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)!=2 or numpy.sum(index2)!=2:
        raise IOError('Position vectors in _overlap_DijDkl must be 3D')
    if numpy.dot(index1,index2)==0:
        res=_overlap_DijS(ri,rj,a,b,index1,[0,0,0])
        if numpy.dot(index2,index2)==2:
            for ii in range(3):
                if index2[ii]==1:
                    res=res*a/(a+b)*(ri[ii]-rj[ii])
        elif numpy.dot(index2,index2)==4:
            for ii in range(3):
                if index2[ii]==2:
                    res=res*(b+a*(1+2*a*((rj[ii]-ri[ii])**2)))/(2*((b+a)**2))
        else:
            raise IOError('Error in _overlap_DijPk')
    elif numpy.dot(index1,index2)==1:
        ind1=index1
        ind2=index2
        for ii in range(3):
            if index1[ii]==1 and index2[ii]==1:
                ind1[ii]=0
                ind2[ii]=0
                res=_overlap_PiPj(ri,rj,a,b,ind1,ind2)
                res=res*(a+b-2*a*b*((ri[ii]-rj[ii])**2))/(2*((a+b)**2))
    elif numpy.dot(index1,index2)==2:
        if numpy.dot(index1,index1)==2 and numpy.dot(index2,index2)==2:
            res=_overlap_SS(ri,rj,a,b)
            for ii in range(3):
                if index1[ii]==1 and index2[ii]==1:
                    res=res*(a+b-2*a*b*((ri[ii]-rj[ii])**2))/(2*((a+b)**2))
        elif numpy.dot(index1,index1)==4 and numpy.dot(index2,index2)==2:
            ind1=index1
            ind2=index2
            for ii in range(3):
                if index2[ii]==1 and index1[ii]==2:
                    ind1[ii]=0
                    ind2[ii]=0
                    res=_overlap_PiS(rj,ri,b,a,ind2,ind1) # Zde musi byt prohozene indexy jelikoz prvni je s a druhy je p
                    res=res*(a**2 - 2*b**2 + a*b*(-1 + 2*b*(ri[ii]-rj[ii])**2))*(ri[ii]-rj[ii])/(2*(a+b)**3)
        elif numpy.dot(index1,index1)==2 and numpy.dot(index2,index2)==4:
            ind1=index1
            ind2=index2
            for ii in range(3):
                if index2[ii]==2 and index1[ii]==1:
                    ind1[ii]=0
                    ind2[ii]=0
                    res=_overlap_PiS(ri,rj,a,b,ind1,ind2) 
                    res=res*(b**2 - 2*a**2 + a*b*(-1 + 2*a*(rj[ii]-ri[ii])**2))*(rj[ii]-ri[ii])/(2*(b+a)**3)
    elif numpy.dot(index1,index2)==4:
        for ii in range(3):
            if index1[ii]==2 and index2[ii]==2:
                res=_overlap_SS(ri,rj,a,b)
                res=res*(-6*a*b*(-1 + b*(ri[ii]-rj[ii])**2) + (b**2)*(3 + 2*b*(ri[ii]-rj[ii])**2) + (a**2)*(3 - 6*b*(ri[ii]-rj[ii])**2 + 4*(b**2)*(ri[ii]-rj[ii])**4) + 2*(a**3)*(ri[ii]-rj[ii])**2)/(4*(a+b)**4)
    else:
        raise IOError('Error in _overlap_DijDkl')
    return res
    
def _overlap_5DSPD(ri,rj,a,b,index1,index2):
    ''' Overlap between 5d orbital and s or p or d orbital '''
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)!=(-2) or numpy.sum(index2)<0 or numpy.sum(index2)>2 :
        raise IOError('Position vectors in _overlap_5DSPD must be 3D')
    if index1[0]==(-2):
        #3ZZ-RR
        res=2*_overlap_GTO(ri,rj,a,b,[0,0,2],index2)-_overlap_GTO(ri,rj,a,b,[0,2,0],index2)-_overlap_GTO(ri,rj,a,b,[2,0,0],index2)
    elif index1[0]==(-1) and index1[2]==(-1):
        #XZ
        res=_overlap_GTO(ri,rj,a,b,[1,0,1],index2)
    elif index1[1]==(-1) and index1[2]==(-1):
        #YZ
        res=_overlap_GTO(ri,rj,a,b,[0,1,1],index2)
    elif index1[1]==(-2):
        #XX-YY
        res=_overlap_GTO(ri,rj,a,b,[2,0,0],index2)-_overlap_GTO(ri,rj,a,b,[0,2,0],index2)
    elif index1[0]==(-1) and index1[1]==(-1):
        #XY
        res=_overlap_GTO(ri,rj,a,b,[1,1,0],index2)
    else:
        raise IOError('Error in _overlap_5DSPD')
    return res

def _overlap_5D5D(ri,rj,a,b,index1,index2): 
    ''' Overlap between two 5d orbitals '''
    
    if numpy.dot(index1,index1)==2 and numpy.dot(index2,index2)==2:
        res=_overlap_GTO(ri,rj,a,b,numpy.abs(index1),numpy.abs(index2))
    elif numpy.dot(index1,index1)==4 and numpy.dot(index2,index2)==2:
        if index1[0]==(-2):
            res=2*_overlap_GTO(ri,rj,a,b,[0,0,2],numpy.abs(index2))-_overlap_GTO(ri,rj,a,b,[0,2,0],numpy.abs(index2))-_overlap_GTO(ri,rj,a,b,[2,0,0],numpy.abs(index2))
        elif index1[1]==(-2):
            res=_overlap_GTO(ri,rj,a,b,[2,0,0],numpy.abs(index2))-_overlap_GTO(ri,rj,a,b,[0,2,0],numpy.abs(index2))
        else:
            raise IOError('Error in _overlap_5D5D')
    elif numpy.dot(index1,index1)==2 and numpy.dot(index2,index2)==4:
        if index2[0]==(-2):
            res=2*_overlap_GTO(ri,rj,a,b,numpy.abs(index1),[0,0,2])-_overlap_GTO(ri,rj,a,b,numpy.abs(index1),[0,2,0])-_overlap_GTO(ri,rj,a,b,numpy.abs(index1),[2,0,0])
        elif index2[1]==(-2):
            res=_overlap_GTO(ri,rj,a,b,numpy.abs(index1),[2,0,0])-_overlap_GTO(ri,rj,a,b,numpy.abs(index1),[0,2,0])
        else:
            raise IOError('Error in _overlap_5D5D')
    elif numpy.dot(index1,index1)==4 and numpy.dot(index2,index2)==4:
        if index1[0]==(-2) and index2[0]==(-2):
            res=4*_overlap_GTO(ri,rj,a,b,[0,0,2],[0,0,2]) - 2*_overlap_GTO(ri,rj,a,b,[0,0,2],[0,2,0]) - 2*_overlap_GTO(ri,rj,a,b,[0,0,2],[2,0,0])
            res += -2*_overlap_GTO(ri,rj,a,b,[0,2,0],[0,0,2]) + _overlap_GTO(ri,rj,a,b,[0,2,0],[0,2,0]) + _overlap_GTO(ri,rj,a,b,[0,2,0],[2,0,0])
            res += -2*_overlap_GTO(ri,rj,a,b,[2,0,0],[0,0,2]) + _overlap_GTO(ri,rj,a,b,[2,0,0],[0,2,0]) + _overlap_GTO(ri,rj,a,b,[2,0,0],[2,0,0])
        elif index1[1]==(-2) and index2[1]==(-2):
            res = _overlap_GTO(ri,rj,a,b,[2,0,0],[2,0,0]) - _overlap_GTO(ri,rj,a,b,[2,0,0],[0,2,0])
            res += -_overlap_GTO(ri,rj,a,b,[0,2,0],[2,0,0]) + _overlap_GTO(ri,rj,a,b,[0,2,0],[0,2,0])
        elif index1[0]==(-2) and index2[1]==(-2):
            res = 2*_overlap_GTO(ri,rj,a,b,[0,0,2],[2,0,0]) - 2*_overlap_GTO(ri,rj,a,b,[0,0,2],[0,2,0]) - _overlap_GTO(ri,rj,a,b,[0,2,0],[2,0,0])
            res += _overlap_GTO(ri,rj,a,b,[0,2,0],[0,2,0]) - _overlap_GTO(ri,rj,a,b,[2,0,0],[2,0,0]) + _overlap_GTO(ri,rj,a,b,[2,0,0],[0,2,0])
        elif index1[1]==(-2) and index2[0]==(-2):
            res = 2*_overlap_GTO(ri,rj,a,b,[2,0,0],[0,0,2]) - _overlap_GTO(ri,rj,a,b,[2,0,0],[0,2,0]) - _overlap_GTO(ri,rj,a,b,[2,0,0],[2,0,0])
            res += -2*_overlap_GTO(ri,rj,a,b,[0,2,0],[0,0,2]) + _overlap_GTO(ri,rj,a,b,[0,2,0],[0,2,0]) + _overlap_GTO(ri,rj,a,b,[0,2,0],[2,0,0])
        else:
            raise IOError('Error in _overlap_5D5D')
    else:
        raise IOError('Error in _overlap_5D5D')
    return res
    
def _overlap_FijkS(ri,rj,a,b,index1,index2):
    ''' Overlap between f orbital and s orbital. First orbital have to be f (orbital with exponent a and position ri) !!!!!'''
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)!=3 or numpy.sum(index2)!=0:
        raise IOError('Position vectors in _overlap_FijkS must be 3D')
    #print(ri,rj,a,b,index1,index2)
    if numpy.dot(index1,index1)==3:
        # mam 3 odlisne indexy xyz
        res=_overlap_SS(ri,rj,a,b)
        for ii in range(3):
            if index1[ii]==1:
                res=res*b/(a+b)*(rj[ii]-ri[ii])
    elif numpy.dot(index1,index1)==5:
        # mam dva indexy xxy xxz, xyy, yyz, xzz, yzz
        indx1=index1
        for ii in range(3):
            if index1[ii]==1:
                indx1[ii]=0
                res=_overlap_DijS(ri,rj,a,b,indx1,index2)
                res=res*b/(a+b)*(rj[ii]-ri[ii])
    elif numpy.dot(index1,index1)==9:
        # mam 1 index xxx,yyy,zzz
        res=_overlap_SS(ri,rj,a,b)
        for ii in range(3):
            if index1[ii]==3:
                res = res*b*(3*a + b*(3 + 2*b*((ri[ii]-rj[ii])**2)))*(rj[ii]-ri[ii])/(2*((a+b)**3))
                #-((b (3 a + b (3 + 2 b (xi - xj)^2)) (xi - xj))/(2 (a + b)^3))
    return res
    
def _overlap_FijkPl(ri,rj,a,b,index1,index2):
    ''' Overlap between f orbital and p orbital. First orbital have to be f (orbital with exponent a and position ri) !!!!!'''
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)!=3 or numpy.sum(index2)!=1:
        raise IOError('Position vectors in _overlap_DijPk must be 3D')
    if numpy.dot(index1,index2)==0:
        res=_overlap_FijkS(ri,rj,a,b,index1,[0,0,0])
        for ii in range(3):
            if index2[ii]==1:
                res=res*a/(a+b)*(ri[ii]-rj[ii])
    elif numpy.dot(index1,index2)==1:
        res=_overlap_PiPj(ri,rj,a,b,index2,index2)
        for ii in range(3):
            if index1[ii]==1 and index2[ii]!=1:
                res=res*b/(a+b)*(rj[ii]-ri[ii])
            elif index1[ii]==2 and index2[ii]!=1:
                res=res*(a+b*(1+2*b*((ri[ii]-rj[ii])**2)))/(2*((a+b)**2))                
    elif numpy.dot(index1,index2)==2:
        indx1=index1
        for ii in range(3):
            if index1==1 and index2!=1:
                indx1[ii]=0
                res=_overlap_DijPk(ri,rj,a,b,indx1,index2)
                res=res*b/(a+b)*(rj[ii]-ri[ii])
    elif numpy.dot(index1,index2)==3:
        res=_overlap_SS(ri,rj,a,b)
        for ii in range(3):
            if index1[ii]==3:
                res=res*((a**2)*(3 - 6*b*(ri[ii]-rj[ii])**2) + 3*(b**2)*(1 + 2*b*(ri[ii]-rj[ii])**2) + 2*a*b*(3 - 2*(b**2)*(ri[ii]-rj[ii])**4) )/(4*(a+b)**4)
                # (a^2 (3 - 6 b (xi - xj)^2) + 3 b^2 (1 + 2 b (xi - xj)^2) +  2 a b (3 - 2 b^2 (xi - xj)^4))/(4 (a + b)^4)
    else:
        raise IOError('Error in _overlap_FijkPl')
    
    return res

def _overlap_FijkDlm(ri,rj,a,b,index1,index2):
    ''' Overlap between f orbital and d orbital. First orbital have to be f (orbital with exponent a and position ri) !!!!!'''        
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)!=3 or numpy.sum(index2)!=2:
        raise IOError('Position vectors in _overlap_FijkDlm must be 3D')
    if numpy.dot(index1,index2)==0:
        if numpy.dot(index1,index1)==5:
            indx1=index1
            for ii in range(3):
                if index1[ii]==1 and index2[ii]==0:
                    indx1[ii]=0
                    res=_overlap_DijDkl(ri,rj,a,b,indx1,index2)
                    res = res*b/(a+b)*(rj[ii]-ri[ii])
        elif numpy.dot(index1,index1)==9:
            res=_overlap_DijS(rj,ri,b,a,index2,[0,0,0])
            for ii in range(3):
                if index1[ii]==3 and index2[ii]==0:
                    res=res*b*(3*a + b*(3 + 2*b*((ri[ii]-rj[ii])**2)))*(rj[ii]-ri[ii])/(2*((a+b)**3))
        else:
            raise IOError('Error in _overlap_FijkDlm')
    elif numpy.dot(index1,index2)==1:
        indx2=index2
        for ii in range(3):
            if index1[ii]==0 and index2[ii]==1:
                indx2[ii]=0
                res=_overlap_FijkPl(ri,rj,a,b,index1,indx2)
                res=res*a/(a+b)*(ri[ii]-rj[ii])
    elif numpy.dot(index1,index2)==2:
        if numpy.dot(index1,index1)==3:
            if numpy.dot(index2,index2)==2:
                res=_overlap_DijDkl(ri,rj,a,b,index2,index2)
                for ii in range(3):
                    if index1[ii]==1 and index2[ii]==0:
                        res=res*b/(a+b)*(rj[ii]-ri[ii])
            elif numpy.dot(index2,index2)==4:
                indx1=index1
                for ii in range(3):
                    if index1[ii]==1 and index2[ii]==2:
                        indx1[ii]=0
                        res=_overlap_DijS(ri,rj,a,b,indx1,[0,0,0])
                        res=res*(b**2 - a*b + 2*(a**2)*(-1 + b*(ri[ii]-rj[ii])**2))*(rj[ii]-ri[ii])/(2*(a+b)**2)
                        # -(((-a b + b^2 + 2 a^2 (-1 + b (xi - xj)^2)) (xi - xj))/(2 (a + b)^3))
            else:
                raise IOError('Error in _overlap_FijkDlm')
        elif numpy.dot(index1,index1)==5:
            if numpy.dot(index2,index2)==2:
                indx1=index1
                for ii in range(3):
                    if index1[ii]==1 and index2[ii]==0:
                        indx1[ii]=0
                        res=_overlap_DijDkl(ri,rj,a,b,indx1,index2)
                        res=res*b/(a+b)*(rj[ii]-ri[ii])
            elif numpy.dot(index2,index2)==4:
                indx1=index1
                for ii in range(3):
                    if index1[ii]==1 and index2[ii]==2:
                        indx1[ii]=0
                        res=_overlap_DijS(ri,rj,a,b,indx1,[0,0,0])
                        res=res*(b**2 - a*b + 2*(a**2)*(-1 + b*(ri[ii]-rj[ii])**2))*(rj[ii]-ri[ii])/(2*(a+b)**2)
                        # -(((-a b + b^2 + 2 a^2 (-1 + b (xi - xj)^2)) (xi - xj))/(2 (a + b)^3))
            else:
                raise IOError('Error in _overlap_FijkDlm')
        else:
            raise IOError('Error in _overlap_FijkDlm')
    elif numpy.dot(index1,index2)==3:
        if numpy.dot(index2,index2)==4:
            raise IOError('Error in _overlap_FijkDlm')
        if numpy.dot(index1,index1)==5:
            indx1=index1
            indx2=index2
            for ii in range(3):
                if index1[ii]==1 and index2[ii]==1:
                    indx1[ii]=0
                    indx2[ii]=0
                    res=_overlap_DijPk(ri,rj,a,b,indx1,indx2)
                    res=res*(b + a*(1 - 2*b*(ri[ii]-rj[ii])**2) )/(2*(a+b)**2)
                    # (b + a (1 - 2 b (xi - xj)^2))/(2 (a + b)^2)
        elif numpy.dot(index1,index1)==9:
            indx2=index2
            for ii in range(3):
                if index1[ii]==0 and index2[ii]==1:
                    indx2[ii]=0
                    res=_overlap_FijkPl(ri,rj,a,b,index1,indx2)
                    res=res*a/(a+b)*(ri[ii]-rj[ii])
        else: 
            raise IOError('Error in _overlap_FijkDlm')
    elif numpy.dot(index1,index2)==4:
        if numpy.dot(index1,index1)!=5 and numpy.dot(index2,index2)!=4:
            raise IOError('Error in _overlap_FijkDlm')
        indx1=index1
        for ii in range(3):
            if index1[ii]==1 and index2[ii]==0:
                indx1[ii]==0
                res=_overlap_DijDkl(ri,rj,a,b,indx1,index2)
                res=res*b/(a+b)*(rj[ii]-ri[ii])
    elif numpy.dot(index1,index2)==5:
        print(index1,index2)
        raise IOError('Error in _overlap_FijkDlm')
    elif numpy.dot(index1,index2)==6:
        res=_overlap_SS(ri,rj,a,b)
        for ii in range(3):
            if index1[ii]==3 and index2[ii]==2:
                res=res*(2*a*(b**2)*(6 - 5*b*(ri[ii]-rj[ii])**2) + 6*(a**3)*(-1 + b*(ri[ii]-rj[ii])**2) + (b**3)*(9 + 2*b*(ri[ii]-rj[ii])**2) + (a**2)*b*(-3 - 6*b*(ri[ii]-rj[ii])**2 + 4*(b**2)*(ri[ii]-rj[ii])**4 ) )*(rj[ii]-ri[ii])/(4*(a+b)**5)
                # -(((2 a b^2 (6 - 5 b (xi - xj)^2) + 6 a^3 (-1 + b (xi - xj)^2) + b^3 (9 + 2 b (xi - xj)^2) + 
                # a^2 b (-3 - 6 b (xi - xj)^2 + 4 b^2 (xi - xj)^4)) (xi - xj))/( 4 (a + b)^5))
    else:
        raise IOError('Error in _overlap_DijDkl')
    return res

def _overlap_5DFijk(ri,rj,a,b,index1,index2):
    ''' Overlap between f orbital and 5d orbital. 5D orbital has to be first '''
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)!=(-2) or numpy.sum(index2)!=3:
        raise IOError('Position vectors in overlap_Fijk5D must be 3D')    
    
    if index1[0]==(-2):
        #3ZZ-RR
        res=2*_overlap_GTO(ri,rj,a,b,[0,0,2],index2)-_overlap_GTO(ri,rj,a,b,[0,2,0],index2)-_overlap_GTO(ri,rj,a,b,[2,0,0],index2)
    elif index1[0]==(-1) and index1[2]==(-1):
        #XZ
        res=_overlap_GTO(ri,rj,a,b,[1,0,1],index2)
    elif index1[1]==(-1) and index1[2]==(-1):
        #YZ
        res=_overlap_GTO(ri,rj,a,b,[0,1,1],index2)
    elif index1[1]==(-2):
        #XX-YY
        res=_overlap_GTO(ri,rj,a,b,[2,0,0],index2)-_overlap_GTO(ri,rj,a,b,[0,2,0],index2)
    elif index1[0]==(-1) and index1[1]==(-1):
        #XY
        res=_overlap_GTO(ri,rj,a,b,[1,1,0],index2)
    else:
        raise IOError('Error in _overlap_5DSPD')
    return res
    
    
#----------------------------------------------------------------------------------------------#    


# Vypocet dipolu:



#----------------------------------------------------------------------------------------------#
        
def _overlap_GTO(ri,rj,a,b,index1,index2):
    ''' calculate overlap between two gaussian orbitals '''
    res=0
    if  (numpy.abs(numpy.sum(index2))>numpy.abs(numpy.sum(index1)) and numpy.sum(index1)>=0) or numpy.sum(index2)==-2:
        # prohod orbitaly - prvni orbital musi mit vetsi kvantove cislo nez 2. pro obdrzeni spravnych vysledku
        [ri,rj]=[rj,ri]
        [a,b]=[b,a]
        [index1,index2]=[index2,index1]
    if numpy.sum(index1)==0:
        res=_overlap_SS(ri,rj,a,b)
#        if res<0.0001:
#            print('Almost zero overlap in _overlap_SS')
    elif numpy.sum(index1)==1 and numpy.sum(index2)==0:
        res=_overlap_PiS(ri,rj,a,b,index1,index2)
#        if res<0.000001:
#            print('Almost zero overlap in _overlap_PiS')
    elif numpy.sum(index1)==1 and numpy.sum(index2)==1:
        res=_overlap_PiPj(ri,rj,a,b,index1,index2)
#        if res<0.000001:
#            print('Almost zero overlap in _overlap_PiPj')
    elif numpy.sum(index1)==2 and numpy.sum(index2)==0:
        res=_overlap_DijS(ri,rj,a,b,index1,index2)
#        if res<0.000001:
#            print('Almost zero overlap in _overlap_DijS')
    elif numpy.sum(index1)==2 and numpy.sum(index2)==1:
        res=_overlap_DijPk(ri,rj,a,b,index1,index2)
#        if res<0.000001:
#            print('Almost zero overlap in _overlap_DijPk')
    elif numpy.sum(index1)==2 and numpy.sum(index2)==2:
        res=_overlap_DijDkl(ri,rj,a,b,index1,index2)
#        if res<0.000001:
#            print('Almost zero overlap in _overlap_DijDkl')
    elif numpy.sum(index1)==3 and numpy.sum(index2)==0:
        res=_overlap_FijkS(ri,rj,a,b,index1,index2)
    elif numpy.sum(index1)==3 and numpy.sum(index2)==1:
        res=_overlap_FijkPl(ri,rj,a,b,index1,index2)
    elif numpy.sum(index1)==3 and numpy.sum(index2)==2:
        res=_overlap_FijkDlm(ri,rj,a,b,index1,index2)
        #res=_overlap_DijDkl(ri,rj,a,b,index1,index2)
    elif numpy.sum(index1)==(-2) and numpy.sum(index2)>=0 and numpy.sum(index2)<3:
        res=_overlap_5DSPD(ri,rj,a,b,index1,index2)
#        if res<0.000001:
#            print('Almost zero overlap in _overlap_5DSPD')
    elif numpy.sum(index1)==(-2) and numpy.sum(index2)==3:
        res=_overlap_5DFijk(ri,rj,a,b,index1,index2)
    elif numpy.sum(index1)==(-2) and numpy.sum(index2)==(-2):
        res=_overlap_5D5D(ri,rj,a,b,index1,index2)
#        if res<0.000001:
#            print('Almost zero overlap in _overlap_5D5D')
    else:
        raise IOError('Only supported types of orbitals in _overlap_GTO are s,p,d,5d and f, but overlap between two f orbitals is not yet implemented')
    return res
    #_overlap_GTO
    
def _dipole_SPDSPD(ri,rj,a,b,index1,index2):
    ''' Dipole moment between two gaussian orbitals (s or p or d orbitals) !!!!!'''        
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)>2 or numpy.sum(index1)<0 or numpy.sum(index2)>2 or numpy.sum(index2)<0:
        raise IOError('Position vectors in _dipole_SPDSPD must be 3D') 
    dipole=numpy.zeros(3)
    for ii in range(3):
        index_dip=numpy.zeros(3)
        index_dip[ii]=1
        dipole[ii]=_overlap_GTO(ri,rj,a,b,index1,numpy.add(index2,index_dip)) + rj[ii]*_overlap_GTO(ri,rj,a,b,index1,index2)
    return dipole

def _dipole_5DSPD(ri,rj,a,b,index1,index2):
    ''' Dipole moment between 5d and s,p or d gaussian orbitals '''
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.abs(numpy.sum(index1))!=2 or numpy.abs(numpy.sum(index2))!=2:
        raise IOError('Position vectors in dipole_5DDij must be 3D')
    dipole=numpy.zeros(3)
    if numpy.sum(index1)==-2 and numpy.sum(index2)<3 and numpy.sum(index2)>=0:
        for ii in range(3):
            index_dip=numpy.zeros(3)
            index_dip[ii]=1
            dipole[ii]=_overlap_GTO(ri,rj,a,b,index1,numpy.add(index2,index_dip)) + rj[ii]*_overlap_GTO(ri,rj,a,b,index1,index2)
    elif numpy.sum(index2)==-2 and numpy.sum(index1)<3 and numpy.sum(index1)>=0:
        for ii in range(3):
            index_dip=numpy.zeros(3)
            index_dip[ii]=1
            dipole[ii]=_overlap_GTO(ri,rj,a,b,numpy.add(index1,index_dip),index2) + ri[ii]*_overlap_GTO(ri,rj,a,b,index1,index2)
    else:
        raise IOError('Error in _dipole_5DSPD')
    
    return dipole
    
def _dipole_5D5D(ri,rj,a,b,index1,index2):
    ''' Dipole moment between two 5d gaussian orbitals '''
    if len(ri)!=3 or len(rj)!=3 or len(index1)!=3 or len(index2)!=3 or numpy.sum(index1)!=(-2) or numpy.sum(index2)!=(-2):
        raise IOError('Position vectors in _dipole_5D5D must be 3D')
    dipole=numpy.zeros(3)
    if numpy.dot(index1,index1)==2 and numpy.dot(index2,index2)==2:
        for ii in range(3):
            index_dip=numpy.zeros(3)
            index_dip[ii]=1
            dipole[ii]=_overlap_GTO(ri,rj,a,b,numpy.add(numpy.abs(index1),index_dip),numpy.abs(index2)) + ri[ii]*_overlap_GTO(ri,rj,a,b,numpy.abs(index1),numpy.abs(index2))
    elif numpy.dot(index1,index1)==4 and numpy.dot(index2,index2)==2:
        if index1[0]==(-2):
            for ii in range(3):
                index_dip=numpy.zeros(3)
                index_dip[ii]=1
                indx2=numpy.add(numpy.abs(index2),index_dip)
                dipole[ii]=2*_overlap_GTO(ri,rj,a,b,[0,0,2],indx2)-_overlap_GTO(ri,rj,a,b,[0,2,0],indx2)-_overlap_GTO(ri,rj,a,b,[2,0,0],indx2)
                dipole[ii] += rj[ii]*(2*_overlap_GTO(ri,rj,a,b,[0,0,2],numpy.abs(index2))-_overlap_GTO(ri,rj,a,b,[0,2,0],numpy.abs(index2))-_overlap_GTO(ri,rj,a,b,[2,0,0],numpy.abs(index2)))
        elif index1[1]==(-2):
            for ii in range(3):
                index_dip=numpy.zeros(3)
                index_dip[ii]=1
                indx2=numpy.add(numpy.abs(index2),index_dip)
                dipole[ii]=_overlap_GTO(ri,rj,a,b,[2,0,0],indx2)-_overlap_GTO(ri,rj,a,b,[0,2,0],indx2)
                dipole[ii] += rj[ii]*(_overlap_GTO(ri,rj,a,b,[2,0,0],numpy.abs(index2))-_overlap_GTO(ri,rj,a,b,[0,2,0],numpy.abs(index2)))
        else:
            raise IOError('Error in _dipole_5D5D')
    elif numpy.dot(index1,index1)==2 and numpy.dot(index2,index2)==4:
        if index2[0]==(-2):
            for ii in range(3):
                index_dip=numpy.zeros(3)
                index_dip[ii]=1
                indx1=numpy.add(numpy.abs(index1),index_dip)
                dipole[ii]=2*_overlap_GTO(ri,rj,a,b,indx1,[0,0,2])-_overlap_GTO(ri,rj,a,b,indx1,[0,2,0])-_overlap_GTO(ri,rj,a,b,indx1,[2,0,0])
                dipole[ii] += ri[ii]*(2*_overlap_GTO(ri,rj,a,b,numpy.abs(index1),[0,0,2])-_overlap_GTO(ri,rj,a,b,numpy.abs(index1),[0,2,0])-_overlap_GTO(ri,rj,a,b,numpy.abs(index1),[2,0,0]))
        elif index2[1]==(-2):
            for ii in range(3):
                index_dip=numpy.zeros(3)
                index_dip[ii]=1
                indx1=numpy.add(numpy.abs(index1),index_dip)
                dipole[ii]=_overlap_GTO(ri,rj,a,b,indx1,[2,0,0])-_overlap_GTO(ri,rj,a,b,indx1,[0,2,0])
                dipole[ii] += ri[ii]*(_overlap_GTO(ri,rj,a,b,numpy.abs(index1),[2,0,0])-_overlap_GTO(ri,rj,a,b,numpy.abs(index1),[0,2,0]))
        else:
            raise IOError('Error in _dipole_5D5D')
    elif numpy.dot(index1,index1)==4 and numpy.dot(index2,index2)==4:
        if index1[0]==(-2) and index2[0]==(-2):
            for ii in range(3):
                index_dip=numpy.zeros(3)
                index_dip[ii]=1
                dipole[ii]=4*_overlap_GTO(ri,rj,a,b,numpy.add([0,0,2],index_dip),[0,0,2]) - 2*_overlap_GTO(ri,rj,a,b,numpy.add([0,0,2],index_dip),[0,2,0]) - 2*_overlap_GTO(ri,rj,a,b,numpy.add([0,0,2],index_dip),[2,0,0])
                dipole[ii] += -2*_overlap_GTO(ri,rj,a,b,numpy.add([0,2,0],index_dip),[0,0,2]) + _overlap_GTO(ri,rj,a,b,numpy.add([0,2,0],index_dip),[0,2,0]) + _overlap_GTO(ri,rj,a,b,numpy.add([0,2,0],index_dip),[2,0,0])
                dipole[ii] += -2*_overlap_GTO(ri,rj,a,b,numpy.add([2,0,0],index_dip),[0,0,2]) + _overlap_GTO(ri,rj,a,b,numpy.add([2,0,0],index_dip),[0,2,0]) + _overlap_GTO(ri,rj,a,b,numpy.add([2,0,0],index_dip),[2,0,0])
                dipole[ii] += ri[ii]*_overlap_GTO(ri,rj,a,b,index1,index2)
        elif index1[1]==(-2) and index2[1]==(-2):
            for ii in range(3):
                index_dip=numpy.zeros(3)
                index_dip[ii]=1
                dipole[ii] = _overlap_GTO(ri,rj,a,b,numpy.add([2,0,0],index_dip),[2,0,0]) - _overlap_GTO(ri,rj,a,b,numpy.add([2,0,0],index_dip),[0,2,0])
                dipole[ii] += -_overlap_GTO(ri,rj,a,b,numpy.add([0,2,0],index_dip),[2,0,0]) + _overlap_GTO(ri,rj,a,b,numpy.add([0,2,0],index_dip),[0,2,0])
                dipole[ii] += ri[ii]*_overlap_GTO(ri,rj,a,b,index1,index2)
        elif index1[0]==(-2) and index2[1]==(-2):
            for ii in range(3):
                index_dip=numpy.zeros(3)
                index_dip[ii]=1
                dipole[ii] = 2*_overlap_GTO(ri,rj,a,b,numpy.add([0,0,2],index_dip),[2,0,0]) - 2*_overlap_GTO(ri,rj,a,b,numpy.add([0,0,2],index_dip),[0,2,0]) - _overlap_GTO(ri,rj,a,b,numpy.add([0,2,0],index_dip),[2,0,0])
                dipole[ii] += _overlap_GTO(ri,rj,a,b,numpy.add([0,2,0],index_dip),[0,2,0]) - _overlap_GTO(ri,rj,a,b,numpy.add([2,0,0],index_dip),[2,0,0]) + _overlap_GTO(ri,rj,a,b,numpy.add([2,0,0],index_dip),[0,2,0])
                dipole[ii] += ri[ii]*_overlap_GTO(ri,rj,a,b,index1,index2)
        elif index1[1]==(-2) and index2[0]==(-2):
            for ii in range(3):
                index_dip=numpy.zeros(3)
                index_dip[ii]=1
                dipole[ii] = 2*_overlap_GTO(ri,rj,a,b,numpy.add([2,0,0],index_dip),[0,0,2]) - _overlap_GTO(ri,rj,a,b,numpy.add([2,0,0],index_dip),[0,2,0]) - _overlap_GTO(ri,rj,a,b,numpy.add([2,0,0],index_dip),[2,0,0])
                dipole[ii] += -2*_overlap_GTO(ri,rj,a,b,numpy.add([0,2,0],index_dip),[0,0,2]) + _overlap_GTO(ri,rj,a,b,numpy.add([0,2,0],index_dip),[0,2,0]) + _overlap_GTO(ri,rj,a,b,numpy.add([0,2,0],index_dip),[2,0,0])
        else:
            raise IOError('Error in _dipole_5D5D')
    else:
        raise IOError('Error in _dipole_5D5D')
        
    return dipole
    
def _dipole_GTO(ri,rj,a,b,index1,index2):
    ''' Dipole moment between two gaussian orbitals '''
    dipole=numpy.zeros(3)
    if numpy.sum(index1)<3 and numpy.sum(index1)>=0 and numpy.sum(index2)<3 and numpy.sum(index2)>=0:
        dipole=_dipole_SPDSPD(ri,rj,a,b,index1,index2)
    elif (numpy.sum(index1)==(-2) and numpy.sum(index2)>=0 and numpy.sum(index2)<3) or (numpy.sum(index2)==(-2) and numpy.sum(index1)>=0 and numpy.sum(index1)<3):
        dipole=_dipole_5DSPD(ri,rj,a,b,index1,index2)
    elif numpy.sum(index1)==(-2) and numpy.sum(index2)==(-2):
        dipole=_dipole_5D5D(ri,rj,a,b,index1,index2)
    else:
        raise IOError('Only supported types of orbitals in _overlap_GTO are s,p,d and 5d. Dipole moment with f orbitals is not yet implemented')
    
    return dipole

                
def R_STO(a,r):
    ''' radial part of slater orbital. Same for all l quantum numbers
    r is radius from orbitals position (in atomic units => in Bohr) 
    and a is exponent 
    if we want to use angstroms we should rescale a->a/a0 where a0 is bohr radius'''

    norm=numpy.sqrt(a**3/numpy.pi)     
    sto=norm*numpy.exp(-a*abs(r))
    return sto

def R_STO_form_GTO(r,coef_list,exp_list):
    ''' construction slater type radial distribution function from gaussian 
    orbitals.
    r is radius from orbital's position
    coef_list is list or sigle value of expansion coefficients
    exp_list is list or sigle value of exponents of gaussian orbitals '''

    sto=0.0    
    if hasattr(r, "__len__"):
        if len(r)==3:
            rr=r[0]**2+r[1]**2+r[2]**2
        else:
            raise IOError(' Radius has to be scalar or vector (x,y,z) ')
    else:
        rr=r**2
    
    if hasattr(coef_list, "__len__") or hasattr(coef_list, "__len__"):
        if len(coef_list) != len(exp_list):
            raise IOError('Expansion coefficients of slater orbital has to have same length as gaussian exponents')
        for ii in range(len(coef_list)):
           sto += coef_list[ii]*R_GTO(exp_list[ii],rr) 
    else:
        sto=coef_list*R_GTO(exp_list,rr)
        
    return sto

# Test Function
def Psi_STO_form_GTO2(r,coef_list,exp_list,index):
    ''' construction slater type radial distribution function from gaussian 
    orbitals.
    r is radius from orbital's position
    coef_list is list or sigle value of expansion coefficients
    exp_list is list or sigle value of exponents of gaussian orbitals '''

    sto=0.0    
    if hasattr(r, "__len__"):
        if len(r)==3:
            rr=r[0]**2+r[1]**2+r[2]**2
        else:
            raise IOError(' Radius has to be scalar or vector (x,y,z) ')
    else:
        rr=r**2
    
    r0=numpy.zeros(3)
    
    if hasattr(coef_list, "__len__") or hasattr(coef_list, "__len__"):
        if len(coef_list) != len(exp_list):
            raise IOError('Expansion coefficients of slater orbital has to have same length as gaussian exponents')
        for ii in range(len(coef_list)):
           sto += coef_list[ii]*numpy.exp(-exp_list[ii]*rr)*norm_GTO(r0,exp_list[ii],index) 
    else:
        sto=coef_list*numpy.exp(-exp_list[ii]*rr)*norm_GTO(r0,exp_list,index)
    
    m=index[0]
    n=index[1]
    o=index[2]            
    if m!=0:
        sto=sto*r[0]**m
    if n!=0:
        sto=sto*r[1]**n
    if o!=0:
        sto=sto*r[2]**o
        
    return sto


def gSTO(r,norb,m,n,o,qc):    
    ''' cartesian slater orbital
    ** r=coordinate vector
    ** m,n,o cartesian slater orbital has type x^m * y^n * z^o * R_STO_form_GTO
    ** norb=orbital number in qc
    '''
    #print(r,norb,m,n,o)
    if (m+n+o)!=lquant[qc.ao_type[norb,0]]:
        raise IOError('Error In gSTO. m+n+o is not equal to l - orbital quantum number')
    if len(r)!=3:
        raise IOError('coordinate vector has to be 3 dimensional')    
    res=(r[0]**m)*(r[1]**n)*(r[2]**o)*R_STO_form_GTO(r,qc.ao_expcoef[norb][:,1],qc.ao_expcoef[norb][:,0])
    return res
    

# TODO: Put electronic integrals into separate module
def integral_to_index(integral,typ='Python'):
    ''' transform 4 integral index into list index ''' 
    if len(integral)!=4:
        raise IOError('For calculation of 2electron integral 4 atomic orbitals has to be defined')
    
    def reorder_integral(indx):
        [i,j,k,l]=[indx[0],indx[1],indx[2],indx[3]]
        if i<j:
            t=i
            i=j
            j=t
        if k<l:
            t=k
            k=l
            l=t
        if i<k:
            indx_out=[k,l,i,j]
        elif i==k and j<l:
            indx_out=[k,l,i,j]
        else:
            indx_out=[i,j,k,l]
        return indx_out
                
        
    if typ=='Python':
        [i,j,k,l]=[integral[0]+1,integral[1]+1,integral[2]+1,integral[3]+1]
    elif typ=='Fortran':
        [i,j,k,l]=[integral[0],integral[1],integral[2],integral[3]]
    [i,j,k,l]=reorder_integral([i,j,k,l])
    if i<k or i<j or k<l:
        raise IOError('In integral (ij|kl) i>=j and i>=k and k>=l ')
    index=0
    for ii in range(1,i):
        index+=ii*(((ii-1)*(ii+2))//2+1)
    index+=(j-1)*((i*(i+1))//2)
    index+=(k*(k-1))//2
    index+=l-1
    return index
        
def Energy_ground(qc): # zatim pouze na HF a myslim i na DFT
    energy=0.0
    if len(qc.del_int)<1 or len(qc.sel_core)<1:
        raise IOError('electron integrals has to be imported from gaussian log file before energy calculation')
    Nel=sum(qc.N_el)
    Nocc=Nel//2
    if sum(qc.N_el)%2!=0 or qc.N_el[0]!=qc.N_el[0]:
        raise IOError('Only closed shell energy is implemented')
    Norb=len(qc.mo_coef[0])
    PP=numpy.zeros((Norb,Norb))
    HH=numpy.zeros((Norb,Norb))
    
    # allocatedensity matrix P
    Cmu_mat=numpy.array(qc.mo_coef[0:Nocc]).T 
    Cnu_mat=numpy.array(qc.mo_coef[0:Nocc])
    #PP=numpy.multiply(2,numpy.dot(Cnu_mat,Cmu_mat))
    PP=numpy.multiply(2,numpy.dot(Cmu_mat,Cnu_mat))
    
    # allocate core hamiltonian
    HH=qc.sel_core
    
    # calculation of single electron integrals (energy)
    SelEnergy=numpy.trace(numpy.dot(PP.T,HH))
    
    PotE=-numpy.trace(numpy.dot(PP.T,qc.sel_pot))
    KinE=numpy.trace(numpy.dot(PP.T,qc.sel_kin))
    
    print('Potential Energy:',PotE)
    print('Kinetic Energy:',KinE)
    
    # calculation of two electron integrals (energy)
    DelEnergy=0.0
    MM=numpy.zeros((Norb,Norb))
    EE1=numpy.zeros((Norb,Norb))
    EE2=numpy.zeros((Norb,Norb))
    for mu in range(Norb):
        for nu in range(Norb):
            INT_mat=numpy.zeros((Norb,Norb))
            INT1_mat=numpy.zeros((Norb,Norb))
            INT2_mat=numpy.zeros((Norb,Norb))
            for ii in range(Norb):
                for jj in range(Norb):
                    INT_mat[ii,jj]=qc.del_int[integral_to_index([mu,nu,jj,ii])][4]-qc.del_int[integral_to_index([mu,ii,jj,nu])][4]/2.0    #(mu,nu|ii,jj)-1/2*(mu,jj|ii,nu)
                    INT1_mat[ii,jj]=qc.del_int[integral_to_index([mu,nu,ii,jj])][4]    #(mu,nu|ii,jj)
                    INT2_mat[ii,jj]=qc.del_int[integral_to_index([mu,jj,ii,nu])][4]/2    #(mu,jj|ii,nu)
            #MM[mu,nu]=numpy.trace(numpy.dot(PP.T,INT1_mat-INT2_mat))
            MM[mu,nu]=numpy.trace(numpy.dot(PP.T,INT_mat))
            EE1[mu,nu]=numpy.trace(numpy.dot(PP.T,INT1_mat))
            EE2[mu,nu]=numpy.trace(numpy.dot(PP.T,INT2_mat))
            
    if True:
        energy2=0.0
        for mu in range(Norb):
            for nu in range(Norb):
                energy2+=PP[mu,nu]*qc.sel_core[mu,nu]
                for lbda in range(Norb):
                    for sigma in range(Norb):
                        #energy2+=1/2*PP[mu,nu]*PP[lbda,sigma]*(qc.del_int[integral_to_index([mu,nu,lbda,sigma])][4]-1/2*qc.del_int[integral_to_index([mu,sigma,lbda,nu])][4])    # -1/2*qc.del_int[integral_to_index([mu,sigma,lbda,nu])][4]
                        energy2+=1/2*PP[mu,nu]*PP[lbda,sigma]*(qc.del_int[integral_to_index([mu,nu,lbda,sigma])][4]-1/2*qc.del_int[integral_to_index([mu,sigma,lbda,nu])][4]) 
        print(energy2)
    print('trace(P.T*S)',numpy.trace(numpy.dot(PP.T,qc.sel_over)),'should be equal to',Nel)    

    if True:
        mu=0
        nu=1
        INT12_mat=numpy.zeros((Norb,Norb))
        INT21_mat=numpy.zeros((Norb,Norb))
        for ii in range(Norb):
            for jj in range(Norb):
                INT12_mat[ii,jj]=qc.del_int[integral_to_index([mu,jj,ii,nu])][4]    #(mu,jj|ii,nu)
                print('12:','mu:',mu,'nu:',nu,'labda:',ii,'sigma:',jj,qc.del_int[integral_to_index([mu,jj,ii,nu])])
        mu=1
        nu=0
        for ii in range(Norb):
            for jj in range(Norb):
                INT21_mat[ii,jj]=qc.del_int[integral_to_index([mu,jj,ii,nu])][4]    #(mu,jj|ii,nu)
                print('21:',qc.del_int[integral_to_index([mu,jj,ii,nu])])
        print(' ')
        print('I12:')
        print(INT12_mat)
        print(' ')
        print('I21:')
        print(INT21_mat)
        print(' ')
        print('I12-I21.T:')
        numpy.set_printoptions(precision=3)
        print(INT12_mat-INT21_mat.T)
        
        mu=1
        nu=2
        INT12_mat=numpy.zeros((Norb,Norb))
        INT21_mat=numpy.zeros((Norb,Norb))
        for ii in range(Norb):
            for jj in range(Norb):
                INT12_mat[ii,jj]=qc.del_int[integral_to_index([mu,jj,ii,nu])][4]    #(mu,jj|ii,nu)
                print('12:','mu:',mu,'nu:',nu,'labda:',ii,'sigma:',jj,qc.del_int[integral_to_index([mu,jj,ii,nu])])
        mu=2
        nu=1
        for ii in range(Norb):
            for jj in range(Norb):
                INT21_mat[ii,jj]=qc.del_int[integral_to_index([mu,jj,ii,nu])][4]    #(mu,jj|ii,nu)
                print('21:',qc.del_int[integral_to_index([mu,jj,ii,nu])])
        print(' ')
        print('I12:')
        print(INT12_mat)
        print(' ')
        print('I21:')
        print(INT21_mat)
        print(' ')
        print('I12-I21.T:')
        numpy.set_printoptions(precision=3)
        print(INT12_mat-INT21_mat.T)
        
    # Mullicken population analysis
    charge=numpy.zeros(qc.Nat)
    counter=0
    MC=numpy.dot(PP.T,qc.sel_over)
    for ii in range(len(qc.ao_type)):
        for jj in range(l_deg(qc.ao_type[ii][0])):
            charge[int(qc.ao_type[ii][3])-1]+=MC[counter,counter]
            counter+=1
    print('Mulliken charges:')
    for ii in range(len(charge)):
        charge[ii]=float(qc.at_info[ii][2])-charge[ii]
        print(qc.at_info[ii][0],charge[ii])
    
    print(counter)
    #print('Mulliken charges',charge)                      
        
    #FF=HH+MM
    FF=qc.sel_kin-qc.sel_pot+MM
    CC=numpy.array(qc.mo_coef[0:Norb]).T
    SS=qc.sel_over
    XX=numpy.linalg.inv(scipy.linalg.sqrtm(qc.sel_over))
    FF2=numpy.dot(XX.T,numpy.dot(FF,XX))
    CC2=numpy.dot(numpy.linalg.inv(XX),CC)
    Eps=numpy.dot(numpy.dot(CC.T,numpy.dot(numpy.linalg.inv(SS),FF)),CC)
    Eps2=numpy.dot(CC2.T,numpy.dot(FF2,CC2))
    #Eps=numpy.dot(numpy.dot(CC.T,FF),CC)
    print('mela by byt diag. matice s vlastnimi energiemi')
    numpy.set_printoptions(precision=7)
    print(Eps) 
    print('mela by byt diag. matice s vlastnimi energiemi')
    print(Eps2)
    print(' ')
    print('Fock matrix:')
    print(FF)
    print(' ')
    print('EE1 matrix:')
    print(EE1) 
    print(' ')
    print('EE2 matrix:')
    print(EE2)      
    print(' ')
    print('EE matrix:')
    print(MM) 
    print(' ')
    print('PP-PP.T matrix:')
    print(PP-PP.T)
    print('PP-PP.T sum abs:')
    print(numpy.sum(numpy.abs(PP-PP.T)))
    print('Final density mamtrix:')
    print(PP)         
    DelEnergy=1/2*numpy.trace(numpy.dot(PP.T,MM))
    print(Norb,Nocc)
    energy=SelEnergy+DelEnergy
    numpy.set_printoptions(precision=13)
    print('1el energy:',SelEnergy)
    print('2el energy:',DelEnergy)
    print('Potential Energy:',PotE)
    print('Kinetic Energy:',KinE)
    print('trace(P.T*S)',numpy.trace(numpy.dot(PP.T,qc.sel_over)),'should be equal to',Nel) 
    print('Ground state energy:',energy+qc.E_Nuc)
    print('Ground state energy2:',energy2+qc.E_Nuc)
    return [energy,PP]        
    

    
strg='abcdefghijklmnopqrstuvwxyz'
orbit = 'spd' + strg[5:].replace('s','').replace('p','')
lquant = dict([(j, i) for i,j in enumerate(orbit)])
#del i,j

def l_deg(l=0,ao=None,cartesian_basis=True):
  '''Calculates the degeneracy of a given atomic orbitals.
  
  **Options:**
  
  Works with the molpro output nomenclature for Cartesian Harmonics:
    s->'s', p->['x','y','z'], d-> ['xx','yy', etc.], etc.
    e.g., l_deg(ao='xxy')
  
  Works with quantum number l for the Cartesian Harmonic:
    e.g., l_deg(l=1)
  
  Works with name of the Cartesian Harmonic:
    e.g., l_deg(l='p')
  ''' 
  if l=='5d':
      l=-2
  else:
      if ao != None:
        if ao == 's':
          return 1
        else:
          l = len(ao)
      elif isinstance(l,str):
          l = lquant[l]
  return int((l+1)*(l+2)/2) if (cartesian_basis and l>0) else (2*abs(l)+1)