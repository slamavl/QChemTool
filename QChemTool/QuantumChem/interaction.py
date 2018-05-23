import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from sys import platform
from scipy.spatial import distance as scipy_dist 
import timeit

from ..General.units import conversion_facs_energy, conversion_facs_position
from ..General.UnitsManager import position_units,energy_units
from .qch_functions import dipole_STO
from .positioningTools import prepare_alkene
from .Classes.general import Energy,Coordinate,Dipole


if (platform=='cygwin' or platform=="linux" or platform == "linux2"):
        from .Fortran.wrapper import w_fortranmodule as fortran
#if (platform=='cygwin' or platform=="linux" or platform == "linux2"):
#        from Program_Test.QuantumChem.Fortran.wrapper import w_fortranmodule as fortran

# =============================================================================
# General interaction functions
# =============================================================================

def charge_charge(coor1,charge1,coor2,charge2,*args):
    '''Calculate interaction energy between two groups of point charges.
    
    Parameters
    ----------
    center1 : numpy.array of real (dimension Nx3)
        Array of centers of first group of point charges in ATOMIC UNITS (Bohr)
    charge : numpy.array of real (dimension N)
        Array of charges of first group of point charges in times of electron charge
    center2 : numpy.array of real (dimension 3)
        Array of centers of second group of point charges in ATOMIC UNITS (Bohr)
    charge2 : numpy.array of real (dimension N)
        Array of charges of first group of point charges in times of electron charge
    *args : string
        If `args`='Hartree' then interaction energy and distance will be returned
        in atomic units (Hartree and Bohr).
        If `args`='cm-1' then interaction energy will be retured in wavenumbers
        (inverse centimeters) and distance in Angstroms
        
        
    Returns
    -------
    Ech_dip :  real
        Interaction energy between two groups of point charges
    Distance : real
        Mean value of distance between all pairs from diferent groups (molecules)
    
    '''
    
    RR=scipy_dist.cdist(coor1,coor2)
    RR1=1/RR
    
    # calculation of interaction energy
    # point chargepoint charge
    Q2,Q1= np.meshgrid(charge1,charge2)
# TODO: this could definitely be done faster if I need only trace - But this is not slowing down the calculation that much            
    Energy_Hartree=np.trace(np.dot(np.multiply(Q1,Q2),RR1))
    Energy_cm1=Energy_Hartree*conversion_facs_energy["1/cm"]
    
    distance=np.sum(RR)/(len(coor1)*len(coor2))
    
    if args:
        for a in args:
            if a=='Hartree':
                return Energy_Hartree,distance
            elif a=='cm-1':
                return Energy_cm1,distance*conversion_facs_position["Angstrom"] 
            else:
                raise Warning('Unknown argument in charge_charge function')
    return Energy_Hartree,distance


def charge_dipole(center1,charge,center2,dipole,*args):
    '''Calculate interaction energy between point charge and point dipole.
    
    Parameters
    ----------
    center1 : numpy.array of real (dimension 3)
        Center of point charge in ATOMIC UNITS (Bohr)
    charge : real
        charge in times of electron charge
    center2 : numpy.array of real (dimension 3)
        Center of dipole in ATOMIC UNITS (Bohr)
    dipole2 : numpy.array of real (dimension 3)
        Vector of dipole in ATOMIC UNITS (Bohr*charge_in_e)
    *args : string
        If `args`='Hartree' then interaction energy will be returned in atomic
        units (Hartree).
        If `args`='cm-1' then interaction energy will be retured in wavenumbers
        (inverse centimeters)
        
        
    Returns
    -------
    Ech_dip :  real
        Interaction energy of point charge and point dipole
    '''

    r12=np.zeros(3)    
    r12=center2-center1
    R=np.sqrt(np.dot(r12,r12))
    
    dr=np.dot(dipole,r12)   
    
    Ech_dip_Ha=-(charge*dr)/(R**3)   # interaction energy in hartree
    Ech_dip_cm1=Ech_dip_Ha*conversion_facs_energy["1/cm"]

    if args:
        for a in args:
            if a=='Hartree':
                return Ech_dip_Ha
            elif a=='cm-1':
                return Ech_dip_cm1
            else:
                raise Warning('Unknown argument in charge_dipole function')
                
    return Ech_dip_Ha
 

def dipole_dipole(center1,dipole1,center2,dipole2,*args):
    '''Calculate dipole dipole interaction energy between two point dipoles.
    
    Parameters
    ----------
    center1 : numpy.array of real (dimension 3)
        Center of first dipole in ATOMIC UNITS (Bohr)
    dipole1 : numpy.array of real (dimension 3)
        Vector of first dipole in ATOMIC UNITS (Bohr*charge_in_e)
    center2 : numpy.array of real (dimension 3)
        Center of second dipole in ATOMIC UNITS (Bohr)
    dipole2 : numpy.array of real (dimension 3)
        Vector of second dipole in ATOMIC UNITS (Bohr*charge_in_e)
    *args : string
        If `args`='Hartree' then interaction energy will be returned in atomic
        units (Hartree).
        If `args`='cm-1' then interaction energy will be retured in wavenumbers
        (inverse centimeters)
        
        
    Returns
    -------
    Edip_dip :  real
        Interaction energy of two point dipoles
    '''
    
    if 0:   # both provides the same results
        if np.ndim(center1)==1:
            r1=np.array([center1],dtype='f8')
            dip1=np.array([dipole1],dtype='f8')
        else: 
            r1=np.copy(center1)
            dip1=np.copy(dipole1)
        if np.ndim(center2)==1:
            r2=np.array([center2],dtype='f8')
            dip2=np.array([dipole2],dtype='f8')
        else: 
            r2=np.copy(center2)
            dip2=np.copy(dipole2)
        R1 = np.tile(r1,(len(r2),1,1))
        R1 = np.swapaxes(R1,0,1)
        R2 = np.tile(r2,(len(r1),1,1))
        R = R2 - R1
        RR=np.sqrt(np.sum(R*R,axis=2))
        P1=np.tile(dip1,(len(dip2),1,1))
        P1 = np.swapaxes(P1,0,1)
        P2=np.tile(dip2,(len(dip1),1,1))
        PP=np.sum(P1*P2,axis=2)
        PR1=np.sum(P1*R,axis=2)
        PR2=np.sum(P2*R,axis=2)
        
        Edip_dip_Hartree=np.sum(PP/RR**3-3*PR1*PR2/RR**5)
        with energy_units('Ha'):
            energy=Energy(Edip_dip_Hartree)
        return energy
    
    else:
        r12=np.zeros(3)    
        r12=center2-center1
        R=np.sqrt(np.dot(r12,r12))
        d12=np.dot(dipole1,dipole2)
        
        if R<1.0e-4 and d12<1.0e-4:
            return 0.0
        
        
        dr1=np.dot(dipole1,r12)/R
        dr2=np.dot(dipole2,r12)/R
        
        Edip_dip_Ha=(d12 - 3*dr1*dr2)/(R**3)   # interaction energy in hartree
        Edip_dip_cm1=Edip_dip_Ha*conversion_facs_energy["1/cm"]
    
        if args:
            for a in args:
                if a=='Hartree':
                    return Edip_dip_Ha
                elif a=='cm-1':
                    return Edip_dip_cm1
                    
        return Edip_dip_cm1   
   

def _trans_density_row(origin1,grid1,step1,rho1,origin2,grid2,step2,rho2,RR2_x,RR2_y,RR2_z,MinDistance,indx): # it might be faster than _trans_density_row
    ''' 
    Function which is used for parallelization of calculation of interaction energy
    between points from density with fisrt index defined by indx and all points 
    from density2. This function is similar to the `_trans_density_row` but this
    one uses by default matrix approach where interaction between one point from
    first density and all points from the second density is calculated at one step. 
    
    Parameters
    ----------
    origin1 : numpy.array of real (dimension 3)
        Coordinates of origin of cube grid of first density in ATOMIC UNITS (Bohr)
        for the first density
    grid1 : numpy.array of integer (dimension 3)
        Number of points of the grid in every dimension for the first density.
        grid=numpy.array([M,N,O])
    step1 : numpy.array of real (dimension 3x3)
        In rows there are translational vectors defining the coordinates of grid
        points for the first density. For example coordinates of the point 
        (m,n,o)=m*step[0,:]+n*step[1,:]+o*step[2,:]+origin
    rho1 : numpy array of real (dimension NxMxO)
        Matrix with density value for every grid point for the first density.
    origin2 : numpy.array of real (dimension 3)
        Coordinates of origin of cube grid of first density in ATOMIC UNITS (Bohr)
        for the second density
    grid2 : numpy.array of integer (dimension 3)
        Number of points of the grid in every dimension for the second density.
    step2 : numpy.array of real (dimension 3x3)
        In rows there are translational vectors defining the coordinates of grid
        points for the second density. For example coordinates of the point 
        (m,n,o)=m*step[0,:]+n*step[1,:]+o*step[2,:]+origin
    rho2 : numpy array of real (dimension NxMxO)
        Matrix with density value for every grid point for the second density.
        grid=numpy.array([M,N,O])
    RR2_x,RR2_y,RR2_z : numpy.array of real (dimension NxMxO):
        Matrixes with x, y and z coordinates of every grid point of second density
        in ATOMIC UNITS (Bohr)
    MinDistance : real
        minimal distance between points where transition density is defined which
        is used for calculation of interaction energy (in odred to avoid overlap 
        of points from different densities which would result in infinite 
        interaction energy)
    indx : integer
        Index of first axes of first density for which interaction energy is calculated 
        and sumed
    
    Returns
    -------
    res :  real
        Sum of interaction energies in ATOMIC UNITS (Hartree)
    '''

    res=0.0
    print(indx,'/',grid1[0])
    use_memory_save=False
    start_time = timeit.default_timer()
    if use_memory_save:
        for j in range(grid1[1]):
            for k in range(grid1[2]):
                rr1=origin1 + indx*step1[0,:]+j*step1[1,:]+k*step1[2,:]
                for m in range(grid2[0]):
                    for n in range(grid2[1]):
                        for o in range(grid2[2]):
                            rr2=origin2 + m*step2[0,:]+n*step2[1,:]+o*step2[2,:]
                            drr=rr2-rr1
                            norm=np.sqrt(np.dot(drr,drr))
                            if (norm>MinDistance):
                                res += rho1[indx,j,k]*rho2[m,n,o]/norm
    else:
        for j in range(grid1[1]):
            for k in range(grid1[2]):
                rr1=origin1 + indx*step1[0,:]+j*step1[1,:]+k*step1[2,:]
                dR_x=np.subtract(RR2_x,rr1[0])
                dR_y=np.subtract(RR2_y,rr1[1])
                dR_z=np.subtract(RR2_z,rr1[2])                        
                norm=np.sqrt(np.add(np.power(dR_x, 2),np.add(np.power(dR_y, 2),np.power(dR_z, 2))))
                del dR_x # to free memory because this matrix is quite large
                del dR_y # to free memory because this matrix is quite large
                del dR_z # to free memory because this matrix is quite large
                norm[norm < MinDistance] = 0.1
                res_mat=np.divide(rho2,norm)
                res += rho1[indx,j,k]*np.sum(res_mat)
                #if res<0.0:
                #    print(res,rho1[indx,j,k],np.sum(res_mat))
                del norm # to free memory because this matrix is quite large
                del res_mat # to free memory because this matrix is quite large
    elapsed = timeit.default_timer() - start_time
    print('Time for calculation of single x step:',elapsed)
    print('res=',res)
    return res

def trans_density(density1,density2,MinDistance=0.1,nt=1):
    ''' Function for calculation of interaction energy through transition density
    cube method (TDC).
    
    Parameters
    ----------
    density1 : DensityGrid type
        Information about transition density spatial distribution of first 
        molecule
    density2 : DensityGrid type
        Information about transition density spatial distribution of second 
        molecule
    MinDistance : real (optional - init=0.1)
        minimal distance in ATOMIC UNITS (Bohr) between points where transition
        density is defined which is used for calculation of interaction energy
        (in order to avoid overlap of points from different densities which would
        result in infinite interaction energy)
    nt : integer (optional - init=1)
        Parameter which sets which numerical approach will be used for calculation
        of TDC interaction energy.
        nt=-2 : fortran parallel calculation is used - fastest (recomended, so far only working in linux)
        nt=-1 : serial non-vector  type calculation is used - slowest (only for testing - simplest)
        nt=0 and linux OS is used : python parallel calculation (works so far only in linux)
        nt=1 : python serial vector type calculation (recomended for windows OS) 
        
    Returns
    -------
    res :  real
        Interaction energy calculated by transition density cube method in 
        ATOMIC UNITS (Hartree)
    
    '''
    res=0.0
   
    if nt==-2:
        typ='Fortran'
    elif nt==-3:
        typ='Fortran_old'
    elif nt<0:
        typ='serial_old'
    elif (platform=='cygwin' or platform=="linux" or platform == "linux2") and nt!=1 and nt>=0:
        typ='paralell'
    elif platform=='win32' or nt==1:
        typ='serial'
    else:
        typ='serial_old'
        
    vecX=np.copy(density1.step[0,:])
    vecY=np.copy(density1.step[1,:])
    vecZ=np.array([vecX[1]*vecY[2]-vecX[2]*vecY[1],vecX[2]*vecY[0]-vecX[0]*vecY[2],vecX[0]*vecY[1]-vecX[1]*vecY[0]])
    dV1=np.dot(vecZ,density1.step[2,:])
            
    vecX=np.copy(density2.step[0,:])
    vecY=np.copy(density2.step[1,:])
    vecZ=np.array([vecX[1]*vecY[2]-vecX[2]*vecY[1],vecX[2]*vecY[0]-vecX[0]*vecY[2],vecX[0]*vecY[1]-vecX[1]*vecY[0]])
    dV2=np.dot(vecZ,density2.step[2,:])
        
    if typ=='paralell' or typ=='serial':
        RR2_x=np.zeros((density1.grid[0],density1.grid[1],density1.grid[2]),dtype='f8')
        RR2_y=np.zeros((density1.grid[0],density1.grid[1],density1.grid[2]),dtype='f8')
        RR2_z=np.zeros((density1.grid[0],density1.grid[1],density1.grid[2]),dtype='f8')
        for m in range(density2.grid[0]):
            for n in range(density2.grid[1]):
                for o in range(density2.grid[2]):
                    rr2=density2.origin + m*density2.step[0,:]+n*density2.step[1,:]+o*density2.step[2,:]           
                    RR2_x[m,n,o]=rr2[0]
                    RR2_y[m,n,o]=rr2[1]
                    RR2_z[m,n,o]=rr2[2]        
        
    if typ=='serial_old':
        res=0.0
        if 1==0:
            for i in range(density1.grid[0]):
                for j in range(density1.grid[1]):
                    for k in range(density1.grid[2]):
                        rr1=density1.origin + i*density1.step[0,:]+j*density1.step[1,:]+k*density1.step[2,:]
                        for m in range(density2.grid[0]):
                            for n in range(density2.grid[1]):
                                for o in range(density2.grid[2]):
                                    rr2=density2.origin + m*density2.step[0,:]+n*density2.step[1,:]+o*density2.step[2,:]
                                    drr=rr2-rr1
                                    norm=np.sqrt(np.dot(drr,drr))
                                    if (norm>MinDistance):
                                        res += density1.data[i,j,k]*density2.data[m,n,o]/norm
        else:
            # This for RR1 is not needed it could be comented in order to save memory - I think it doesn lead to speedup the calculation                             
#            RR1=np.zeros((density1.grid[0],density1.grid[1],density1.grid[2]),dtype='f8')
#            for i in range(density1.grid[0]):
#                for j in range(density1.grid[1]):
#                    for k in range(density1.grid[2]):
#                        RR1[i,j,k]=density1.origin + i*density1.step[0,:]+j*density1.step[1,:]+k*density1.step[2,:]
            RR2_x=np.zeros((density1.grid[0],density1.grid[1],density1.grid[2]),dtype='f8')
            RR2_y=np.zeros((density1.grid[0],density1.grid[1],density1.grid[2]),dtype='f8')
            RR2_z=np.zeros((density1.grid[0],density1.grid[1],density1.grid[2]),dtype='f8')
            for m in range(density2.grid[0]):
                for n in range(density2.grid[1]):
                    for o in range(density2.grid[2]):
                        rr2=density2.origin + m*density2.step[0,:]+n*density2.step[1,:]+o*density2.step[2,:]           
                        RR2_x[m,n,o]=rr2[0]
                        RR2_y[m,n,o]=rr2[1]
                        RR2_z[m,n,o]=rr2[2]
            
            for i in range(density1.grid[0]):
                print(i,'/',density1.grid[0])
                start_time = timeit.default_timer()
                for j in range(density1.grid[1]):
                    for k in range(density1.grid[2]):
                        rr1=density1.origin + i*density1.step[0,:]+j*density1.step[1,:]+k*density1.step[2,:]
                        #dR=np.subtract(RR2,RR1[i,j,k])
                        dR_x=np.subtract(RR2_x,rr1[0])
                        dR_y=np.subtract(RR2_y,rr1[1])
                        dR_z=np.subtract(RR2_z,rr1[2])                        
                        norm=np.sqrt(np.add(np.power(dR_x, 2),np.add(np.power(dR_y, 2),np.power(dR_z, 2))))
                        del dR_x # to free memory because this matrix is quite large
                        del dR_y # to free memory because this matrix is quite large
                        del dR_z # to free memory because this matrix is quite large
                        res_mat=np.multiply(density1.data[i,j,k],np.divide(density2.data,norm))
                        res += np.sum(res_mat)
                        del norm # to free memory because this matrix is quite large
                        del res_mat # to free memory because this matrix is quite large
                elapsed = timeit.default_timer() - start_time
                print('Time for calculation of single x step:',elapsed)        
                        
    elif typ=='paralell':
        print('Calculating TDC interaction energy with python parallel rutine')
        trans_density_row_partial = partial(_trans_density_row, density1.origin,density1.grid,density1.step,density1.data,density2.origin,density2.grid,density2.step,density2.data,RR2_x,RR2_y,RR2_z,MinDistance)
        if nt>0:
            pool = Pool(processes=nt)
        else:
            pool = Pool(processes=cpu_count())
        index_list=range(density1.grid[0])
        x = pool.map_async(trans_density_row_partial,index_list)
        pool.close() # ATTENTION HERE
        pool.join()
        
        x=np.array(x.get())
        res=np.sum(x)
        print('Result=',res)
    
    elif typ=='serial':
        res=0.0
        trans_density_row_partial = partial(_trans_density_row, density1.origin,density1.grid,density1.step,density1.data,density2.origin,density2.grid,density2.step,density2.data,RR2_x,RR2_y,RR2_z,MinDistance)
        index_list=range(density1.grid[0])
        for ii in range(density1.grid[0]):
            start_time = timeit.default_timer()
            res+=trans_density_row_partial(ii)
            elapsed = timeit.default_timer() - start_time
            print('Time for calculation of single x step:',elapsed)
            
    elif typ=='Fortran': 
        print('Calculating TDC interaction energy with fortran rutine')
        s1=np.copy(density1.grid)
        s2=np.copy(density2.grid)
        tstart = timeit.default_timer()
        res = fortran.tdc(density1.grid,density1.step,density1.origin,density1.data,density2.grid,density2.step,density2.origin,density2.data,s1[0],s1[1],s1[2],s2[0],s2[1],s2[2])            
        dt = timeit.default_timer() - tstart
        print('Time elapsed:',dt) 
    elif typ=='Fortran_old':   # Neco tady je spatne
        s1=np.copy(density1.grid)
        s2=np.copy(density2.grid)
        tstart = timeit.default_timer()
        res = fortran.tdc_old(density1.grid,density1.step,density1.origin,density1.data,density2.grid,density2.step,density2.origin,density2.data,s1[0],s1[1],s1[2],s2[0],s2[1],s2[2])            
        dt = timeit.default_timer() - tstart
        print('Time elapsed:',dt)
    if typ!='Fortran' and typ!='Fortran_old' :  
        #print('Not Fortran type')
        vecX=np.copy(density1.step[0,:])
        vecY=np.copy(density1.step[1,:])
        vecZ=np.array([vecX[1]*vecY[2]-vecX[2]*vecY[1],vecX[2]*vecY[0]-vecX[0]*vecY[2],vecX[0]*vecY[1]-vecX[1]*vecY[0]])
        dV1=np.dot(vecZ,density1.step[2,:])
                
        vecX=np.copy(density2.step[0,:])
        vecY=np.copy(density2.step[1,:])
        vecZ=np.array([vecX[1]*vecY[2]-vecX[2]*vecY[1],vecX[2]*vecY[0]-vecX[0]*vecY[2],vecX[0]*vecY[1]-vecX[1]*vecY[0]])
        dV2=np.dot(vecZ,density2.step[2,:])
        res=res*dV1*dV2
    
    return res             
    

def Oscilator3D(rr1,bond1,AtType1,NMN1,TotDip1,rr2,bond2,AtType2,NMN2,TotDip2,scale_by_overlap=True,next_neighbour=False,*args,**kwargs):
    '''
    Calculates interaction energy between two molecules with 3D classical 
    oscillator method - only structure of pi-conjugated subsystem should be 
    used for this calculation.
    
    Parameters
    ----------
    rr1 : numpy.array of real (dimension Nx3)
        Atomic coordinates of first molecule in ATOMIC UNITS (Bohr)
    bond1 : numpy.array of integer (dimension Nx2)
        In the array there are written pairs of atom indexes of atoms from first
        molecule which are connected by chemical bond.
    AtType1 : numpy.array of characters (dimension N)
        Array of atomic types of first molecule
        (for example AtType1=['C','N','C','C',...])
    NMN1 : integer or list of integers
        Oscilator normal mode of first molecule which will be used for calculation
        of interaction energy. If two or more are written linear combination of
        both will be used for calculation of interaction energy (in this case
        also coefficients of linear combination have to be specified in **kwargs).
    TotDip1 : real
        Total transition dipole of first molecule in ATOMIC UNITS (Bohr*charge_in_e)
    rr2 : numpy.array of real (dimension Nx3)
        Atomic coordinates of second molecule in ATOMIC UNITS (Bohr)
    bond2 : numpy.array of integer (dimension Nx2)
        In the array there are written pairs of atom indexes of atoms from second
        molecule which are connected by chemical bond.
    AtType2 : numpy.array of characters (dimension N)
        Array of atomic types of second molecule
        (for example AtType2=['C','N','C','C',...])
    NMN2 : integer or list of integers
        Oscilator normal mode of second molecule which will be used for calculation
        of interaction energy. If two or more are written linear combination of
        both will be used for calculation of interaction energy (in this case
        also coefficients of linear combination have to be specified in **kwargs).
    TotDip2 : real
        Total transition dipole of second molecule
    *args : string
        If `args`='Hartree' then interaction energy will be returned in atomic
        units (Hartree).
        If `args`='cm-1' then interaction energy will be retured in wavenumbers
        (inverse centimeters)
    **kwargs : dictionary
        specification of aditional parameers when using linear combination of
        more transitions. With name 'TotTrDip1' resp. 'TotTrDip2' there have to
        be defined transition dipoles for individual transitions on first resp.
        second molecule. With name 'Coef1' resp. 'Coef2' is definition of 
        expansion coefficients for tranitions on first resp. second molecule.
        
    Returns
    -------
    res :  real
        Interaction energy between two molecules calculated by 3D classical 
        oscillator method.
    '''
       
    use_model_carotenoid=False
    verbose=False
    output_dipole=False
    output_mol_dip=False
    
    if use_model_carotenoid:
        center=np.sum(rr1,0)
        center=center/len(rr1)
        VecX=rr1[len(rr1)//2+2,:]-rr1[len(rr1)//2,:]
        VecY=rr1[len(rr1)//2+1,:]-rr1[len(rr1)//2,:]
        rr1=prepare_alkene(len(rr1),Position=center,vec_x=VecX,vec_y=VecY)
        
        center=np.sum(rr2,0)
        center=center/len(rr2)
        VecX=rr2[len(rr2)//2+2,:]-rr2[len(rr2)//2,:]
        VecY=rr2[len(rr2)//2+1,:]-rr2[len(rr2)//2,:]
        rr2=prepare_alkene(len(rr2),Position=center,vec_x=VecX,vec_y=VecY)

    
    def molecule_osc_3D(rr,bond,factor,NMN,TotDip,next_neighbour=next_neighbour,**kwsargs):     
        TotDip_norm=np.sqrt(np.dot(TotDip,TotDip))
        
        Ndip=len(bond)
        ro=np.zeros((Ndip,3),dtype='f8')
        do=np.zeros((Ndip,3),dtype='f8')
        
        # Place unity dipole moments in centers of all bonds
        for ii in range(Ndip):
            do[ii,:]=rr[bond[ii,1],:]-rr[bond[ii,0],:]
            norm=np.sqrt(np.dot(do[ii,:],do[ii,:]))
            do[ii,:]=do[ii,:]/norm*factor[ii]
            ro[ii,:]=(rr[bond[ii,1],:]+rr[bond[ii,0],:])/2
        
        # Calculate dipole-dipole interaction energy between all dipoles
        hh=np.zeros((Ndip,Ndip),dtype='f8')
        for ii in range(Ndip):
            for jj in range(ii+1,Ndip):
                hh[ii,jj]=dipole_dipole(ro[ii,:],do[ii,:],ro[jj,:],do[jj,:])
                hh[jj,ii]=hh[ii,jj]

        if next_neighbour:
            for ii in range(Ndip):
                for jj in range(ii+2,Ndip):
                    hh[ii,jj]=0.0
                    hh[jj,ii]=hh[ii,jj]       
        
        # Calculate normal modes (eigenvectors and eigenvalues of hh)
        val,vec=np.linalg.eigh(hh) # val= vector with eigenvalues and vec= matrix with eigenvectors in columns

        if verbose:
            for jj in range(6):
                Dip=np.dot(vec[:,jj],do)
                print('state',jj,'dipole:',Dip,'dx/dy:',Dip[0]/Dip[1])
        
        Dip=np.dot(vec[:,NMN],do)
        norm=np.sqrt(np.dot(Dip,Dip))
        
        for ii in range(Ndip):
            do[ii,:]=do[ii,:]*vec[ii,NMN]*TotDip_norm/norm
        
        return ro,do
            
            
    Ndip1=len(bond1)
    Ndip2=len(bond2)
            
    is_units=False
    if args:
        for a in args:
            if a=='Hartree':
                is_units=True
                units='Hartree'
            elif a=='cm-1':
                is_units=True
                units='cm-1'
            elif a=='output_dipole':
                output_dipole=True
            elif a=='output_mol_dipole':
                output_mol_dip=True
    
    # Elementary dipole size is dependent on atom types (dipole in center of N-C bond should be 
    # different than dipole in center of C-C and also in centrer N-N). So far all dipoles are the 
    # same for all atom types. This should be changed and be dependent on AtType
    if scale_by_overlap:
        factor1=np.zeros(Ndip1,dtype='f8')
        factor2=np.zeros(Ndip2,dtype='f8')
        coef=np.array([0.06899907,0.31642396,0.74430829])   # only for C-C bonds and for pz orbitals
        exp=np.array([7.86827235,1.88128854,0.54424926])    # only for C-C bonds and for pz orbitals
        r1=np.array([0.0,0.0,0.0])
        for ii in range(Ndip1):
            R=np.sqrt(np.dot(rr1[bond1[ii,1],:]-rr1[bond1[ii,0],:],rr1[bond1[ii,1],:]-rr1[bond1[ii,0],:]))
            r2=np.array([R,0.0,0.0])            
            Dip=dipole_STO(r1,r2,coef,coef,exp,exp,[0,0,1],[0,0,1]) # only dependent on radial distance and not actual positions - it could be made more accurate this is only simplest approximation
            factor1[ii]=np.sqrt(np.dot(Dip,Dip)) 
        for ii in range(Ndip2):
            R=np.sqrt(np.dot(rr2[bond2[ii,1],:]-rr2[bond2[ii,0],:],rr2[bond2[ii,1],:]-rr2[bond2[ii,0],:]))
            r2=np.array([R,0.0,0.0])            
            Dip=dipole_STO(r1,r2,coef,coef,exp,exp,[0,0,1],[0,0,1]) # only dependent on radial distance and not actual positions - it could be made more accurate this is only simplest approximation            
            factor2[ii]=np.sqrt(np.dot(Dip,Dip)) 
    else:
        factor1=np.ones(Ndip1,dtype='f8')
        factor2=np.ones(Ndip2,dtype='f8')
    
    if isinstance(TotDip1,list):
        if len(TotDip1)==len(NMN1):
            MultiNM1=True
        else:
            raise IOError('Normal mode number list have to have the same dimension as list with dipoles')
    else:
        MultiNM1=False
    
    if isinstance(TotDip2,list):
        if len(TotDip2)==len(NMN2):
            MultiNM2=True
        else:
            raise IOError('Normal mode number list have to have the same dimension as list with dipoles')
    else:
        MultiNM2=False
    
    do1=np.zeros((Ndip1,3),dtype='f8')
    do2=np.zeros((Ndip2,3),dtype='f8')
    
    if verbose:
        print('first molecule')
    if MultiNM1:
        DipoleSize=kwargs['TotTrDip1']
        Coef=kwargs['Coef1']
        for ii in range(len(TotDip1)):
            ro_tmp,do_tmp=molecule_osc_3D(rr1,bond1,factor1,NMN1[ii],TotDip1[ii])
            ro1=np.copy(ro_tmp)
            do1+=Coef[ii]*do_tmp
        # normalize total dipole
        Dip=np.sum(do1,0)
        norm=np.sqrt(np.dot(Dip,Dip))
        do1=do1*DipoleSize/norm
    else:
        ro1,do1=molecule_osc_3D(rr1,bond1,factor1,NMN1,TotDip1)
    
    dipole1=np.sum(do1,0)
    if verbose:
        print('TotalDip1:',dipole1)
    
    if verbose:
        print('Second molecule')
    if MultiNM2:
        DipoleSize=kwargs['TotTrDip2']
        Coef=kwargs['Coef2']
        for ii in range(len(TotDip1)):
            ro_tmp,do_tmp=molecule_osc_3D(rr2,bond2,factor2,NMN2[ii],TotDip2[ii])
            ro2=np.copy(ro_tmp)
            do2+=Coef[ii]*do_tmp
        # normalize total dipole
        Dip=np.sum(do2,0)
        norm=np.sqrt(np.dot(Dip,Dip))
        do2=do2*DipoleSize/norm
    else:
        ro2,do2=molecule_osc_3D(rr2,bond2,factor2,NMN2,TotDip2)
    
    dipole2=np.sum(do2,0)
    if verbose:
        print('TotalDip2:',dipole2)    
    
    res=0.0
    if is_units:
        for ii in range(Ndip1):
            for jj in range(Ndip2):
                res += dipole_dipole(ro1[ii,:],do1[ii,:],ro2[jj,:],do2[jj,:],units)
    else:
        for ii in range(Ndip1):
            for jj in range(Ndip2):
                res += dipole_dipole(ro1[ii,:],do1[ii,:],ro2[jj,:],do2[jj,:])
    
    if output_dipole:
        return res,dipole1,dipole2
    elif output_mol_dip:
        d1=np.zeros((len(do1),3),dtype='f8')
        d2=np.zeros((len(do2),3),dtype='f8')
        for ii in range(len(do1)):
            d1[ii,:]=rr1[bond1[ii,1],:]-rr1[bond1[ii,0],:]
            d2[ii,:]=rr2[bond2[ii,1],:]-rr2[bond2[ii,0],:]
        
        d1s=np.zeros(len(do1),dtype='f8')
        for ii in range(len(do1)):
            d1s[ii]=np.sqrt(np.dot(do1[ii],do1[ii]))*np.sign(np.dot(do1[ii],d1[ii]))
        d2s=np.zeros(len(do2),dtype='f8')
        for ii in range(len(do2)):
            d2s[ii]=np.sqrt(np.dot(do2[ii],do2[ii]))*np.sign(np.dot(do2[ii],d2[ii]))
        return res,d1s,d2s
    else:
        return res
    

# =============================================================================
# Interaction functions for structure class
# =============================================================================

def dipole_dipole_molecule(mol1,exct1,mol2,exct2):
    ''' calculate interaction energy between two molecules by dipole dipole
    interaction between transition dipoles of whole molecules.
    
    Parameters
    ----------
    mol1 :  QchMolecule type
        Information about first molecule. Transition dipole has to be defined in
        mol1.exct[exct1].dipole
    exct1 : integer
        Specifies for which excitation transition dipole is taken for the first 
        molecule (exct1=0 means transition dipole for first excitation)
    mol2 :  QchMolecule type
        Information about second molecule. Transition dipole has to be defined in
        mol2.exct[exct2].dipole
    exct2 : integer
        Specifies for which excitation transition dipole is taken for the second
        molecule (exct2=0 means transition dipole for first excitation)
        
    Returns
    -------
    Energy_hartree :  real
        Dipole-dipole interaction energy between two molecules in ATOMIC UNITS 
        (Hartree)
        
    '''    
    
    dip1=mol1.exct[exct1].dipole.in_AU()
    dip2=mol2.exct[exct2].dipole.in_AU()
    

    RR1 = mol1.get_com()._value
    RR2 = mol2.get_com()._value
    
    Energy_hartree=dipole_dipole(RR1,dip1,RR2,dip2,'Hartree') 
    
    with energy_units('Ha'):
        energy=Energy(Energy_hartree)
    
    return energy


# Pokud chci Huckel interaction energy musim nejdrive preorientovat molekulu do roviny xy pote vzit jen konjugovany retezec a na kazdy atom navesit pouze 1 pz orbital a vypocitat prekryvovou matici a MO koeficienty z Huckela s prekryvem a vypocitam excitacni energii asi jako homo->lumo    
def TrEsp_energy(struc1,struc2,typ='Transition'):
    if typ=='Transition':
        at_charges1=struc1.esp_trans.copy()
        at_charges2=struc2.esp_trans.copy()
    elif typ=='Excited':
        at_charges1=struc1.esp_exct.copy()
        at_charges2=struc2.esp_exct.copy()
    elif typ=='Ground':
        at_charges1=struc1.esp_grnd.copy()
        at_charges2=struc2.esp_grnd.copy()
    else:
        raise IOError('Unknown method for TrEsp calculation')
        
    # Interaction energy calculation    
    RR=np.zeros((struc1.nat,struc2.nat),dtype='f8')
    for ii in range(struc1.nat):
        for jj in range(struc2.nat):
            RR[ii,jj]=np.sqrt(np.dot(struc1.coor._value[ii]-struc2.coor._value[jj],
                              struc1.coor._value[ii]-struc2.coor._value[jj]))
    
    if (RR<=0.21).any():
        raise IOError('Atoms from two molecules are in close contact')
    
    RR1=1/RR
    
    # point charge - point charge interaction energy
    Q2,Q1= np.meshgrid(at_charges2,at_charges1)            
    Energy_hartree=np.trace(np.dot(np.multiply(Q1,Q2),RR1.T))
    
    with energy_units('Ha'):
        energy=Energy(Energy_hartree)
    
    return energy
    

def multipole_at_distances(struc1,struc2,MultOrder=0):
    ''' Calculate interaction energy in hartree between two molecules from 
    quantum chemistry by multipole expansion in atomic distances 
    
    Parameters
    ----------
    mol1 :  QchMolecule type
        Information about first molecule. Information about excited states have
        to be also included
    mol2 :  QchMolecule type
        Information about second molecule. Information about excited states have
        to be also included
    MultOrder : integer (optional - init=0):
        Order of multipole expansion which is used for calculation of interaction
        energy:
        MultOrder=0 : Only zeroth order (electrostatic interaction between point
        charges)
        MultOrder=1 : First order (also charge-dipole interaction is included)
        MultOrder=2 : Second order (also dipole-dipole and charge-quadrupole
        interactions are included)

    Returns
    -------
    Energy :  Energy class
        Interaction energy between two molecules Energy.value in Energy.units units
        (ATOMIC UNITS by default)
        
    Notes
    -------
    To use this function first create multipole representation of the molecule:
    molecule.create_multipole_representation(exct_i,rep_name,MultOrder) where
    exct_i specifies for which transition multipoles are calculated 
    (exct1=0 means for the first excitation). After you do it for both molecules
    interaction energy is calculated by: 
    multipole_at_distances(mol1.repre[rep_name1],mol2.repre[rep_name2],MultOrder)
    
    '''

    if MultOrder>2:
        raise IOError('Higher multipole order than 2 is not supported')        
    if MultOrder>2:
        raise IOError('Higher multipole interaction energy than charge-charge have not been tested. I will be after connection of python with fortran in order to have parallelization')
    
#    if struc1.coor.units=='Angstrom':
#        struc1.coor.Angst2Bohr()
#    elif struc1.coor.units!='Bohr':
#        raise IOError('Working with other units than Agstroms and Bohrs is not supported')
#    if struc2.coor.units=='Angstrom':
#        struc2.coor.Angst2Bohr()
#    elif struc1.coor.units!='Bohr':
#        raise IOError('Working with other units than Agstroms and Bohrs is not supported')
        
        
    # prepare and check all requirements
    if MultOrder>-1:
        # First molecule - charges
        if struc1.tr_char is None:
            raise IOError('Atomic transition charges are not defined in structure class of the first molecule')
        else:
            at_charges1=struc1.tr_char.copy()  
        # Second molecule - charges
        if struc2.tr_char is None:
            raise IOError('Atomic transition charges are not defined in structure class of the second molecule')
        else:
            at_charges2=struc2.tr_char.copy()
    if MultOrder>0:
        # First molecule - dipoles
        if struc1.tr_dip is None:
            raise IOError('Atomic transition dipoles are not defined in structure class of the first molecule')
        else:
            at_dipoles1=struc1.tr_dip.copy()
        # Second molecule - dipoles
        if struc2.tr_dip is None:
            raise IOError('Atomic transition dipoles are not defined in structure class of the second molecule')
        else:
            at_dipoles2=struc2.tr_dip.copy()
    if MultOrder>1:
        # First molecule - quadrupoles
        if struc1.tr_quadr2 is None:
            raise IOError('Atomic transition quadrupoles are not defined in structure class of the first molecule')
        else:
            at_quad_r21=struc1.tr_quadr2.copy()
            at_quad_rR21=struc1.tr_quadrR2.copy()
        
        # Second molecule - quadrupoles
        if struc2.tr_quadr2 is None:
            raise IOError('Atomic transition dipoles are not defined in structure class of the second molecule')
        else:
            at_quad_r22=struc2.tr_quadr2.copy()
            at_quad_rR22=struc2.tr_quadrR2.copy()
        

    
    
   # Interaction energy calculation    
    RR=np.zeros((struc1.nat,struc2.nat),dtype='f8')
    for ii in range(struc1.nat):
        for jj in range(struc2.nat):
            RR[ii,jj]=np.sqrt(np.dot(struc1.coor._value[ii]-struc2.coor._value[jj],
                              struc1.coor._value[ii]-struc2.coor._value[jj]))
    
    if (RR<=0.21).any():
        raise IOError('Atoms from two molecules are in close contact')
    
    RR1=1/RR
    
    # point charge - point charge interaction energy
    Q2,Q1= np.meshgrid(at_charges2,at_charges1)            
    Energy_hartree=np.trace(np.dot(np.multiply(Q1,Q2),RR1))
    
    if MultOrder>0:
        for jj in range(struc1.nat):
            for ii in range(struc2.nat):
                Rij=struc2.coor._value[jj]-struc1.coor._value[ii]
                Rnorm=np.sqrt(np.dot(Rij,Rij))
                # E=Qj*(Rij*di)-Qi*(Rij*dj)
                
                Energy_hartree -= (at_charges1[ii]*np.dot(Rij,at_dipoles2[jj]) \
                        - at_charges2[jj]*np.dot(Rij,at_dipoles1[ii])) / (Rnorm**3)
    if MultOrder==2:
        for ii in range(struc1.nat):
            for jj in range(struc2.nat):
                Rij=struc2.coor._value[jj]-struc1.coor._value[ii]
                Rnorm=np.sqrt(np.dot(Rij,Rij))
                # Eij=(-Qi*QPj-Qj*QPi+2*di*dj)/(2*R**3) + (-6*(di*Rij)(dj*Rij)+3*(dj*Rij*Qi+di*Rij*Qj)/(2*R**5)
                
                # cotribution from dipole-dipole interaction
                Energy_hartree += np.dot(at_dipoles1[ii],at_dipoles2[jj])/(Rnorm**3)
                Energy_hartree -= 3*(np.dot(at_dipoles1[ii],Rij)*np.dot(at_dipoles2[jj],Rij))/(Rnorm**5)

                # contribution from r2 quadrupole-charge interaction
                Energy_hartree -= (at_quad_r21[ii]*at_charges2[jj] \
                + at_quad_r22[jj]*at_charges1[ii])/(2*Rnorm**3)
                
                # + contribution from quadrupole *=(r*R)^2
                counter=0
                for kk in range(3):
                    for ll in range(kk,3):
                        if kk!=ll:
                            Energy_hartree += 3.0*(at_quad_rR21[counter,ii])*Rij[kk]*Rij[ll]*at_charges2[jj]/(Rnorm**5)
                            Energy_hartree += 3.0*(at_quad_rR22[counter,jj])*Rij[kk]*Rij[ll]*at_charges1[ii]/(Rnorm**5)
                        else:
                            Energy_hartree += 3.0/2.0*(at_quad_rR21[counter,ii])*Rij[kk]*Rij[ll]*at_charges2[jj]/(Rnorm**5)
                            Energy_hartree += 3.0/2.0*(at_quad_rR22[counter,jj])*Rij[kk]*Rij[ll]*at_charges1[ii]/(Rnorm**5)
                        counter+=1

    
    if MultOrder>2: # calculate dipole charge interaction energy
        print('Not implemented yet')
    
    with energy_units('Ha'):
        energy=Energy(Energy_hartree)
    
    return energy

    
'''----------------------- TEST PART --------------------------------'''
if __name__=="__main__":
    print(' ')
    print('TESTS')
    print('-------')
    
    if 1:
        ''' Multipole Polyene interaction test '''
        if (platform=='cygwin' or platform=="linux" or platform == "linux2"):
            MolDir='/mnt/sda2/Dropbox/PhD/Programy/Python/Test/'
        elif platform=='win32':
            MolDir='C:/Dropbox/PhD/Programy/Python/Test/'
        from .Classes.general import Energy,PositionAxis
        from .Classes.molecule import Molecule
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
        
        # Calculation of interaction energy
    #    TrVec.value=TrVec.value*const.AmgstromToBohr   # length of translation vector will be 1 Angstrom in Bohrs
        eng=multipole_at_distances(mol1.repre['Multipole2'],mol2.repre['Multipole2'],MultOrder=2)
        E_mult2=Energy(None)
        E_mult1=Energy(None)
        E_mult0=Energy(None)
        E_Huckel2=Energy(None)
        E_Oscillator=Energy(None)
        E_dip=Energy(None)
        X_axis=PositionAxis(0.0,np.linalg.norm(TrVec.value),41)
        with position_units('Angstrom'):
            for ii in range(41):
                E_mult0.add_energy(multipole_at_distances(mol1.repre['Multipole2'],mol2.repre['Multipole2'],MultOrder=0))
                E_mult1.add_energy(multipole_at_distances(mol1.repre['Multipole2'],mol2.repre['Multipole2'],MultOrder=1))
                E_mult2.add_energy(multipole_at_distances(mol1.repre['Multipole2'],mol2.repre['Multipole2'],MultOrder=2))
                E_Huckel2.add_energy(multipole_at_distances(Huckel_mol1.repre['Multipole2'],Huckel_mol2.repre['Multipole2'],MultOrder=2))
                E_Oscillator.add_energy(multipole_at_distances(mol1.repre['Oscillator'],mol2.repre['Oscillator'],MultOrder=2))
                E_dip.add_energy(dipole_dipole_molecule(mol1,0,mol2,0))
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
        ''' Multipole Chlorophyll interaction test '''
        if (platform=='cygwin' or platform=="linux" or platform == "linux2"):
            MolDir='/mnt/sda2/PhD/Ab-initio-META/BChl-c/'
        elif platform=='win32':
            MolDir='C:/PhD/Ab-initio-META/BChl-c/'
        from .Classes.general import Energy,PositionAxis
        from .Classes.molecule import Molecule
        import matplotlib.pyplot as plt
        
        
        exct_index=0
        # Load molecule from gaussian output
        mol1=Molecule('Methyl-Bacteriochlorophyll-c')
        mol1.load_Gaussian_fchk("".join([MolDir,'Me-BChl-c_str4_opt_aligned_exct.fchk']))
        mol1.load_Gaussian_log("".join([MolDir,'Me-BChl-c_str4_opt_aligned_exct.log']))
        
        # print dipoles
        print('Gaussian Qy dipole:',mol1.exct[0].dipole.value)
        print('Gaussian Qx dipole:',mol1.exct[1].dipole.value)
        
        Pi_indx=np.array([2,3,4,5,10,12,15,16,17,19,20,21,33,34,35,37,38,39,52,53,54,59,66,67])-1
        Huckel_mol1=mol1.struc.get_Huckel_molecule([0.0, 0.0, 1.0],At_list=Pi_indx)
        HOMO=Huckel_mol1.struc.nat//2
        Huckel_mol1.add_transition([[HOMO,HOMO+2,1.0]],None) # HOMO->LUMO+1 transition
        Huckel_mol1.add_transition([[HOMO-1,HOMO+1,1.0]],None) # HOMO-1->LUMO transition
        Huckel_mol1.add_transition([[HOMO-1,HOMO+2,1.0]],None) # HOMO-1->LUMO+1 transition
        Huckel_mol1.add_transition([[HOMO,HOMO+1,0.9236],[HOMO-1,HOMO+2,0.3834]],None) # HOMO-1->LUMO+1 transition
        print('Huckel HOMO->LUMO dipole:',Huckel_mol1.exct[0].dipole.value)
        print('Huckel HOMO->LUMO+1 dipole:',Huckel_mol1.exct[1].dipole.value)
        print('Huckel HOMO-1->LUMO dipole:',Huckel_mol1.exct[2].dipole.value)
        print('Huckel HOMO-1->LUMO+1 dipole:',Huckel_mol1.exct[3].dipole.value)
        print('Huckel Qy(85.3-14.7) dipole:',Huckel_mol1.exct[4].dipole.value)
        
# TODO: Output HOMO-1, HOMO, LUMO, LUMO+1 molecular orbitals cube files and compare with ones calculated by quantum chemistry
        HOMO=HOMO-1     # for MO indexing is from 0 (not from 1 as for excitation)
        print("\n Contribution of N atoms in each state for Huckel molecule")
        np.set_printoptions(precision=4)
        np.set_printoptions(suppress=True)
        for ii in range(HOMO-3,HOMO+5):
            mask=np.where(np.array(Huckel_mol1.struc.at_type)=='N')[0]*3   # for every atom there are three orbitals
            mask=mask+2            
            coeff_N=Huckel_mol1.mo.coeff[ii,mask]
            if ii-HOMO<0:
                print('HOMO',ii-HOMO,': ',coeff_N**2,'  Sum: ',np.sum(coeff_N**2),sep="")
            elif ii-HOMO>1:
                print('LUMO+',ii-HOMO-1,': ',coeff_N**2,'  Sum: ',np.sum(coeff_N**2),sep="")
            elif ii-HOMO==1:
                print('LUMO:   ',coeff_N**2,'  Sum: ',np.sum(coeff_N**2),sep="")
            elif ii-HOMO==0:
                print('HOMO:   ',coeff_N**2,'  Sum: ',np.sum(coeff_N**2),sep="")
                
        # Analysis of N contribution in Gaussian calculation
        HOMO=max(np.where(np.array(mol1.mo.occ)==2.0)[0])
        indx=[]
        at_indx=[]
        for ii in range(mol1.ao.nao_orient):
            ao_indx=mol1.ao.indx_orient[ii][0]
            if mol1.ao.atom[ao_indx].type=='N':
                indx.append(ii)
                at_indx.append(mol1.ao.atom[ao_indx].indx)
        indx=np.array(indx,dtype='i8')
        counter=0
        N_indx=np.zeros(len(at_indx),dtype='i8')
        for ii in range(1,len(at_indx)):
            if at_indx[ii]!=at_indx[ii-1]:
                counter+=1
            N_indx[ii]=counter
        
        # calculate overlap matrix:
        mol1.ao.get_overlap()
        
        print("\n Contribution of N atoms in each state for Gaussian calc.")
        for ii in range(HOMO-3,HOMO+5):
            C_N=np.zeros((4,mol1.mo.nmo),dtype='f8')            
            for jj in range(len(indx)):
                C_N[N_indx[jj],indx[jj]]=mol1.mo.coeff[ii,indx[jj]]
            coeff_N=np.diag(np.dot(C_N,np.dot(mol1.ao.overlap,C_N.T)))
            if ii-HOMO<0:
                print('HOMO',ii-HOMO,': ',coeff_N,'  Sum: ',np.sum(coeff_N),sep="")
            elif ii-HOMO>1:
                print('LUMO+',ii-HOMO-1,': ',coeff_N,'  Sum: ',np.sum(coeff_N),sep="")
            elif ii-HOMO==1:
                print('LUMO:   ',coeff_N,'  Sum: ',np.sum(coeff_N),sep="")
            elif ii-HOMO==0:
                print('HOMO:   ',coeff_N,'  Sum: ',np.sum(coeff_N),sep="")
            
        