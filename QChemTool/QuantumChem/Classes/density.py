# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:12:49 2016

@author: uzivatel
"""
import numpy as np
import scipy
from functools import partial
from copy import deepcopy

from .general import Coordinate,Grid
from ...General.UnitsManager import PositionUnitsManaged,position_units
from ...General.types import UnitsManaged
from ..positioningTools import RotateAndMove, RotateAndMove_1, CenterMolecule


class DensityGrid(PositionUnitsManaged):
    ''' Class representing electronic density on spatial grid (e.g. molecular 
    orbitals, transition density, ...)
    
    origin : numpy.array of real (dimension 3)
        origin of density grid (Position managed units)
        
    grid : numpy.array of integer (dimension 3)
        number of grid points at each dimension
        
    step : numpy.array of real (dimension 3x3)
        step[i,:] translational vector in first dimension (Position managed units)

    data : numpy.array of real (dimension Npoints_x x Npoints_y x Npoints_z)
        Density values on the grid. data[i,j,k] correspond to the point with 
        coordinates self.origin+i*self.step[0,:]+j*self.step[1,:]+kk*self.step[2,:]
    
    type : string
        If ``typ='mo'`` density values correspond to real wavefunction otherwise 
        it is an electron density
        
    indx : integer
        Index of molecular orbital to which wavefunction correspond
        
    coor : Coordinate class
        Atomic coordinates for every atom in the molecule or complex. 
        (Position managed units)
    
    at_charge : numpy array of real, integer or string (dimension Natoms)
        Proton number for every atom in the molecule or complex
        
    Functions
    ----------
    rotate :
        Rotate the density and all its properties by specified angles in 
        radians in positive direction.
    rotate_1 :
        Inverse totation to rotate
    move :
        Moves the density and all its properties  along specified vector
    center :
        Center the density and allign in defined plane
    copy :
        Create 1 to 1 deep copy of the density with all classes and types.
    import_cub :
        Read density from cube file
    output :
        Outputs density into cube file
    get_axes :
        Outputs x, y and z axis of the grid on which density is evaluated
        (only for nonrotated grid - oriented along coordinate axis)
    copy :
        Create 1 to 1 deep copy of the density with all classes and types.
    dipole :
        Numerical calculation of dipole from the density 
    dipole_partial :
        Numerical calculation of dipole for only specified spatial cut of the 
        density. (only for nonrotated grid - oriented along coordinate axis)
    cut :
        Spatial cut of the density which is outputed as a new density.
        (only for nonrotated grid - oriented along coordinate axis)
    calc_atomic_properties :
        Calculate atomic charges and dipoles from numerical integration of the
        density into individual atoms. Quantity from grid point will be assigned
        to nearest atom.
    '''
    
    origin=UnitsManaged("origin")
    step=UnitsManaged("step")
    
    def __init__(self,origin,grid,step,density,typ='mo',mo_indx=1,Coor=None,At_charge=None):
        if origin is None:
            self.origin=None
        else:
            self.origin = np.copy(origin)
        if grid is None:
            self.grid=None
        else:
            self.grid = np.copy(grid)
        if step is None:
            self.step=None
        else:
            self.step = np.copy(step)
        if density is None:
            self.data=None
        else:
            self.data = np.copy(density)
        self.type = typ
        self.indx = mo_indx
        if Coor is None:
            self.coor=None
        else:
            self.coor = Coordinate(Coor)
        self.at_charge = np.copy(At_charge)
        
        
    def output(self,filename='density.cub'):
        ''' Output density to cube file 
        
        Parameters
        ----------
        filename : string (optional - init='density.cub')
            Output file name including the path to output folder
        
        '''
        
        with position_units('Bohr'):
            Coor = np.copy(self.coor.value)
            Grid = np.copy(self.grid)
            Step = np.copy(self.step)
            At_charge = np.copy(self.at_charge)
            
        
        with open(filename, "wt") as f:
            # Vypis hlavicky
            f.write("____Zde muze byt napsano cokoliv____ \n MO coefficients \n")
    #        f.write(" %i %5.2f %5.2f %5.2f \n" % (-len(qc.at_coord),min_[0],min_[1],min_[2]))
            if self.type=='mo':
                f.write("{:5d}".format(-len(Coor)))
            else:
                f.write("{:5d}".format(len(Coor)))
                
            for ii in range(3):
                f.write("{:12.6f}".format(self.origin[ii]))
            f.write("{:5d}\n".format(1))
            f.write("{:5d}{:12.6f}{:12.6f}{:12.6f}\n".format(Grid[0], Step[0,0], Step[0,1], Step[0,2] ))
            f.write("{:5d}{:12.6f}{:12.6f}{:12.6f}\n".format(Grid[1], Step[1,0], Step[1,1], Step[1,2] ))
            f.write("{:5d}{:12.6f}{:12.6f}{:12.6f}\n".format(Grid[2], Step[2,0], Step[2,1], Step[2,2] ))
            for ii in range(len(Coor)):
                f.write("{:5d}{:12.6f}{:12.6f}{:12.6f}{:12.6f}\n".format(int(float(At_charge[ii])), float(At_charge[ii]), Coor[ii,0], Coor[ii,1], Coor[ii,2]))
            
            if self.type=='mo':
                f.write("{:5d}{:5d}\n".format(1, self.indx))
    
            # vypis molekuloveho orbitalu na gridu        
            for ii in range(self.grid[0]):
                for jj in range(self.grid[1]):
                    for kk in range(self.grid[2]):
                        f.write("{:13.5E}".format(self.data[ii,jj,kk]))
                        if (kk % 6) == 5:
                            f.write("\n")
                    #f.write("\n")
                    if self.grid[2]%6!=0:
                        f.write("\n")

                    
    def import_cub(self,filename):
        ''' Import data from density cube file 
        
        Parameters
        ----------
        filename : string
            Imput file name (.cub) including the path to file folder
        
        '''
        origin=np.zeros(3,dtype='f8')
        self.grid=np.zeros(3,dtype='i8')
        step=np.zeros((3,3),dtype='f8')
        
        fid    = open(filename,'r')   # Open the file
        flines = fid.readlines()      # Read the WHOLE file into RAM
        fid.close()                   # Close the file
            
        thisline = flines[2].split()
        Natom=np.abs(int(thisline[0]))
        if int(thisline[0]) < 0:
            self.type='mo'
        else:
            self.type='transition'
        
        self.at_charge=np.zeros(Natom,dtype='f')
        Coor=np.zeros((Natom,3),dtype='f8')
        for ii in range(3):
            origin[ii]=float(thisline[ii+1])
        
        for kk in range(3):
            thisline = flines[kk+3].split()
            self.grid[kk]=int(thisline[0])
            for ii in range(3):
                step[kk,ii]=float(thisline[ii+1])
        
        # atomic information:
        for kk in range(Natom):
            thisline = flines[kk+6].split()
            self.at_charge[kk]=float(thisline[1])
            if self.at_charge[kk] == 0.0:
                self.at_charge[kk]=float(thisline[0])
            for ii in range(3):
                Coor[kk,ii]=float(thisline[ii+2])
        
        if self.type=='mo':
            thisline = flines[Natom+6].split()    
            self.indx=int(thisline[1])
            il=7
        else:
            il=6
        
        with position_units('Bohr'):
            self.coor=Coordinate(Coor)
            self.origin=origin.copy()
            self.step=step.copy()
            
        
        # read density
        self.data=np.zeros((self.grid[0],self.grid[1],self.grid[2]),dtype='f8')
        counter=np.zeros(3,dtype='i8')        
        for kk in range(Natom+il,len(flines)):
            line = flines[kk]           # The current line as string
            thisline = line.split()     # The current line split into segments
            if len(thisline) == 0:      # Skip blank lines (even at the end of the file)
                continue
            for ii in range(6):
                self.data[counter[0],counter[1],counter[2]]=float(thisline[ii])
                counter[2]+=1
                if counter[2]==self.grid[2]:
                    counter[2]=0
                    counter[1]+=1
                    if counter[1]==self.grid[1]:
                        counter[1]=0
                        counter[0]+=1
                    break
    
    def get_axes(self):
        """ Outputs x, y and z axis of the grid. ** Working only for grid 
        oriented along coordinate axis (nonrotated grid)**
        
        Returns
        --------
        x,y,z : numpy array of float (dimension Grid_Nx, Grid_Ny, Grid_Nz)
            Coordinates of grid points in coordinate axes 
        
        """
        print("Working only for nonrotated grid oriented along coordinate axes")
        x=np.arange(self.grid[0])*self.step[0,0]+self.origin[0]
        y=np.arange(self.grid[1])*self.step[1,1]+self.origin[1]
        z=np.arange(self.grid[2])*self.step[2,2]+self.origin[2]
        
        return x,y,z
                
    def copy(self):
        ''' Copy DensityGrid class variable into the new one 
        
        Returns
        ----------
        density_new : DensityGrid class
            New DensityGrid class variable with exactly the same values as the 
            original one
        
        Notes
        ----------
        We have to use this function because simple density_new=density_old
        only create pointer to the old density and therefore all changes in
        density_new would be also done on density_old and this is what we don't
        want
        
        
        '''
        density_new = deepcopy(self)        
        return density_new
                
    def move(self,dx,dy,dz):
        ''' Moves density grid in space
        
        Parameters
        ----------
        dx,dy,dz : real
            Distance of density shift along x resp. y resp.
            z axis.
        
        '''
        
        vec=np.array([dx,dy,dz],dtype='f8')
        self.origin=self.origin+vec        
        self.coor.move(dx,dy,dz)
    
    def rotate(self,rotxy,rotxz,rotyz):
        ''' Rotate DENSITY in SPACE in positive rotational angle 
        (if right thumb pointing in direction of axes fingers are pointing in 
        positive rotation direction). First is rotation aroud z axes then around y axes and then around
        x axes.
        
        Parameters
        ----------
        rotxy,rotxz,rotyz : real
            `rotxy` resp. `rotxz` resp. `rotyz` is angle in RADIANS of rotation
            around z resp. y resp. x axis in positive direction
        
        '''
        # Rotation handled in atomic units
        #print('Pred rotaci')
        self._origin=RotateAndMove(np.array([self._origin]),0.0,0.0,0.0,rotxy,rotxz,rotyz)
        self.coor.rotate(rotxy,rotxz,rotyz)
        self._step=RotateAndMove(self._step,0.0,0.0,0.0,rotxy,rotxz,rotyz)
        #print('Po rotaci')
    
    def rotate_1(self,rotxy,rotxz,rotyz):
        ''' Rotate DENSITY in SPACE in negative rotational angle 
        (if right thumb pointing in direction of axes fingers are pointing in 
        positive rotation direction). First is rotation aroud x axes then around y axes and then around
        z axes. Inverse function to rotate(rotxy,rotxz,rotyz)
        
        Parameters
        ----------
        rotxy,rotxz,rotyz : real
            `rotxy` resp. `rotxz` resp. `rotyz` is angle in RADIANS of rotation
            around z resp. y resp. x axis in positive direction
        
        '''
        #print('Pred rotaci')
        self._origin=RotateAndMove_1(np.array([self._origin]),0.0,0.0,0.0,rotxy,rotxz,rotyz)
        self.coor.rotate_1(rotxy,rotxz,rotyz)
        self._step=RotateAndMove_1(self._step,0.0,0.0,0.0,rotxy,rotxz,rotyz)
        #print('Po rotaci')
        
    def center(self,indx_center,indx_x,indx_y):
        ''' Center density according to defined center and main axes 
        Center atom will be in origin of coordinate system 
        (will have [0.0,0.0,0.0] coordinates) and vector X will be pointing into
        direction of x axes and vector Y will be in xy plane. Vector X and Y 
        are defined by atomic indexes.
    
        Parameters
        ----------
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
        
        '''
        
        Coor_ext=[]
        for ii in range(len(self.coor._value)):
            Coor_ext.append(self.coor._value[ii])
        Coor_ext.append(self._origin)
        Coor_ext=np.array(Coor_ext)
        Coor_centered,Phi,Psi,Chi,center=CenterMolecule(Coor_ext,indx_center,indx_x,indx_y,print_angles=True)
        with position_units("Bohr"):
            self.coor=Coordinate(Coor_centered[0,:])
            for ii in range(1,len(Coor_centered)-1):
                self.coor.add_coor(Coor_centered[ii,:])
        self._origin=Coor_centered[len(self.coor._value),:]
        self._step=RotateAndMove(self._step,0.0,0.0,0.0,Phi,Psi,Chi)
    
    
    def dipole(self,output_center=False):
        ''' Calculate numericaly dipole from density. For ground state electron
        density it calculates ground state dipole and fror transition density
        it calculates transition dipole
        
        Returns
        ----------
        dipole : numpy.array of real (dimension 3)
            dipole in ATOMIC UNITS (e*bohr)
        
        Notes
        ----------
        It calculates Int{-r.rho(r)dxdydz} which is dipole
        
        '''

# TODO: repair matrix approach to be used also for rotated density
        if 0:   # This works only for nonrotated grid - change but keep the idea
            grid=Grid()
            grid.init_from_cub(self)
            dipole=np.zeros(3,dtype='f8')
            dipole[0]=np.sum(np.multiply(grid.X,self.data))
            dipole[1]=np.sum(np.multiply(grid.Y,self.data))
            dipole[2]=np.sum(np.multiply(grid.Z,self.data))
            dipole = -np.multiply(grid.ddV,dipole)
            dV=np.dot(self.step[0,:],np.cross(self.step[1,:],self.step[2,:]))
            dipole=np.multiply(-dV,dipole)
            return dipole
        
        else:
            # more efficient would be to create 3D grids with coordinates then multiply and then sum all
            dipole = np.zeros(3,dtype='f8')
            center = np.zeros(3,dtype='f8')
            for ii in range(self.grid[0]):
                for jj in range(self.grid[1]):
                    for kk in range(self.grid[2]):
                        rr=self._origin+ii*self._step[0,:]+jj*self._step[1,:]+kk*self._step[2,:]
                        dipole+=self.data[ii,jj,kk]*rr
                        center+=np.abs(self.data[ii,jj,kk])*rr
            dV=np.dot(self._step[0,:],np.cross(self._step[1,:],self._step[2,:]))
            dipole=dipole*dV
            center = center/np.sum(np.abs(self.data))
            print('Dipole calculated by function dipole was chaged from -dipole to dipole. Make sure that you are using right value')
            if output_center:
                return -dipole,center
            else:
                return -dipole
    
    def dipole_partial(self,x_min=None,x_max=None,y_min=None,y_max=None,z_min=None,z_max=None):
        ''' Calculate numericaly dipole from part of the density. For ground
        state electron density it calculates ground state partial dipole and 
        from transition density it calculates partial transition dipole.

        Parameters
        ---------- 
        x_min,x_max : real (optional - init=None)
            Specifies minimal and maximal x coordinate 
            between which density is used for calculation of dipole. If some of
            those values are not specified there is taken minimal resp. maximal
            x coordinate of the density.
        y_min,y_max : real (optional - init=None)
            Specifies minimal and maximal y coordinate 
            between which density is used for calculation of dipole. If some of
            those values are not specified there is taken minimal resp. maximal
            y coordinate of the density.
        z_min,z_max : real (optional - init=None)
            Specifies minimal and maximal z coordinate 
            between which density is used for calculation of dipole. If some of
            those values are not specified there is taken minimal resp. maximal
            z coordinate of the density.
        
        Returns
        ----------
        dipole : numpy.array of real (dimension 3)
            dipole in ATOMIC UNITS (e*bohr)
        
        
        Notes
        
        Resulting dipole is numericaly calculated integral 
        Int_{x_min,y_min,z_min}^{x_max,y_max,z_max} (-r.rho(r))dxdydz
        '''
        
        if x_min==None:
            x_min=-1.0e5
        else:
            x_min=PositionUnitsManaged.manager.convert_position_2_internal_u(x_min)
        if x_max==None:
            x_max=1.0e5
        else:
            x_max=PositionUnitsManaged.manager.convert_position_2_internal_u(x_max)
        if y_min==None:
            y_min=-1.0e5
        else:
            y_min=PositionUnitsManaged.manager.convert_position_2_internal_u(y_min)
        if y_max==None:
            y_max=1.0e5
        else:
            y_max=PositionUnitsManaged.manager.convert_position_2_internal_u(y_max)
        if z_min==None:
            z_min=-1.0e5
        else:
            z_min=PositionUnitsManaged.manager.convert_position_2_internal_u(z_min)
        if z_max==None:
            z_max=1.0e5
        else:
            z_max=PositionUnitsManaged.manager.convert_position_2_internal_u(z_max)
    
# TODO: Convert boundaries from current values to internal
        #print(x_min,x_max,y_min,y_max,z_min,z_max)
        
        dipole=np.zeros(3,dtype='f8')
        for ii in range(self.grid[0]):
            for jj in range(self.grid[1]):
                for kk in range(self.grid[2]):
                    rr=self._origin+ii*self._step[0,:]+jj*self._step[1,:]+kk*self._step[2,:]
                    if rr[0]>=x_min and rr[0]<=x_max and rr[1]>=y_min and rr[1]<=y_max and rr[2]>=z_min and rr[2]<=z_max:
                        dipole+=self.data[ii,jj,kk]*rr
        dV=np.dot(self._step[0,:],np.cross(self._step[1,:],self._step[2,:]))
        dipole=dipole*dV
        
        print('Dipole calculated by function dipole_partial was chaged from -dipole to dipole. Make sure that you are using right value')
        return -dipole
        
    def cut(self,x_min=None,x_max=None,y_min=None,y_max=None,z_min=None,z_max=None):
        ''' Takes a cut of density. **Works only for original (nonrotated) transition
        density with step[0,:] pointing along x axis, step[1,:] pointing along
        y axis and step[2,:] pointing along z axis.**
        
        Parameters
        ---------- 
        x_min,x_max : real (optional - init=None)
            Specifies minimal and maximal x coordinate in ATOMIC UNITS (Bohr) 
            between which density is outputed. If some of those values are not 
            specified there is taken minimal resp. maximal x coordinate of the 
            density
        y_min,y_max : real (optional - init=None)
            Specifies minimal and maximal y coordinate in ATOMIC UNITS (Bohr) 
            between which density is outputed. If some of those values are not 
            specified there is taken minimal resp. maximal y coordinate of the 
            density
        z_min,z_max : real (optional - init=None)
            Specifies minimal and maximal z coordinate in ATOMIC UNITS (Bohr) 
            between which density is outputed. If some of those values are not 
            specified there is taken minimal resp. maximal z coordinate of the 
            density
            
        Returns
        ---------- 
        cuted_density : DensityGrid class
            DensityGrid class with desity which is subsystem of original density
            and it is defined on grid points with coordinates: x_min <= x <= x_max,
            y_min <= y <= y_max and z_min <= z <= z_max.
        
        ''' 
        
        if x_min==None:
            if self._step[0,0]>0:
                x_min=self._origin[0]
            else:
                x_min=self._origin[0]+self._step[0,0]*(self.grid[0]-1)
        else:
            x_min=PositionUnitsManaged.manager.convert_position_2_internal_u(x_min)
        if x_max==None:
            if self._step[0,0]>0:
                x_max=self._origin[0]+self._step[0,0]*(self.grid[0]-1)
            else:
                x_max=self._origin[0]
        else:
            x_max=PositionUnitsManaged.manager.convert_position_2_internal_u(x_max)
        if y_min==None:
            if self._step[1,1]>0:
                y_min=self._origin[1]
            else:
                y_min=self._origin[1]+self._step[1,1]*(self.grid[1]-1)
        else:
            y_min=PositionUnitsManaged.manager.convert_position_2_internal_u(y_min)
        if y_max==None:
            if self._step[1,1]>0:
                y_max=self._origin[1]+self._step[1,1]*(self.grid[1]-1)
            else:
                y_max=self._origin[1]
        else:
            y_max=PositionUnitsManaged.manager.convert_position_2_internal_u(y_max)
        if z_min==None:
            if self._step[2,2]>0:
                z_min=self._origin[2]
            else:
                z_min=self._origin[2]+self._step[2,2]*(self.grid[2]-1)
        else:
            z_min=PositionUnitsManaged.manager.convert_position_2_internal_u(z_min)
        if z_max==None:
            if self._step[2,2]>0:
                z_max=self._origin[2]+self._step[2,2]*(self.grid[2]-1)
            else:
                z_max=self._origin[2]
        else:
            z_max=PositionUnitsManaged.manager.convert_position_2_internal_u(z_max)
        
        #print(x_min,x_max,y_min,y_max,z_min,z_max)
        
        x=[0,0]
        if self._step[0,0]>0:
            for ii in range(self.grid[0]):
                if self._origin[0]+self._step[0,0]*ii<x_min:
                    x[0]=ii+1
                elif self._origin[0]+self._step[0,0]*ii>x_max and x[1]==0:
                    x[1]=ii-1
            if x[1]==0:
                x[1]=self.grid[0]  
        else:
            for ii in range(self.grid[0]):
                if self._origin[0]+self._step[0,0]*ii>x_max:
                    x[0]=ii+1
                elif self._origin[0]+self._step[0,0]*ii<x_min and x[1]==0:
                    x[1]=ii-1
            if x[1]==0:
                x[1]=self.grid[0] 
                
        y=[0,0]
        if self._step[1,1]>0:
            for ii in range(self.grid[1]):
                if self._origin[1]+self._step[1,1]*ii<y_min:
                    y[0]=ii+1
                elif self._origin[1]+self._step[1,1]*ii>y_max and y[1]==0:
                    y[1]=ii-1
            if y[1]==0:
                y[1]=self.grid[1]
        else:
            for ii in range(self.grid[1]):
                if self._origin[1]+self._step[1,1]*ii>y_max:
                    y[0]=ii+1
                elif self._origin[1]+self._step[1,1]*ii<y_min and y[1]==0:
                    y[1]=ii-1
            if y[1]==0:
                y[1]=self.grid[0] 
                
                
        z=[0,0]
        if self._step[2,2]>0:
            for ii in range(self.grid[2]):
                if self._origin[2]+self._step[2,2]*ii<z_min:
                    z[0]=ii+1
                elif self._origin[2]+self._step[2,2]*ii>z_max and z[1]==0:
                    z[1]=ii-1
            if z[1]==0:
                z[1]=self.grid[2]
        else:
            print('z is negative')
            for ii in range(self.grid[2]):
                if self._origin[2]+self._step[2,2]*ii>z_max:
                    z[0]=ii+1
                elif self._origin[2]+self._step[2,2]*ii<z_min and z[1]==0:
                    z[1]=ii-1
            if z[1]==0:
                z[1]=self.grid[2] 
        
        #print(x,y,z)


        origin_new=self._origin[:]+self._step[0,:]*x[0]+self._step[1,:]*y[0]+self._step[2,:]*z[0]
        grid_new=np.array([x[1]-x[0],y[1]-y[0],z[1]-z[0]])
        data_new=self.data[x[0]:x[1],y[0]:y[1],z[0]:z[1]]
        step_new=np.copy(self._step)
        with position_units("Bohr"):
            cuted_density=DensityGrid(origin_new,grid_new,step_new,data_new,typ=np.copy(self.type),mo_indx=np.copy(self.indx),Coor=np.copy(self.coor.value),At_charge=np.copy(self.at_charge))
        return cuted_density
    
            
    def calc_atomic_properties(self):
        ''' Calculate atomic charges and atomic dipoles by numericaly integrating
        density. Fisrt it is determined to which atom the grid point is the closest
        and to this atom small delta charge and dipole is added. 
        
        Atomic charges are calculated as a sum of density from grid points for 
        which this atom is the closest one. The atomic dipoles are calculated 
        as vector from atom to grid point multiplied by density. 
        
        Returns
        ----------
        charges : numpy.array of real (dimension Natoms)
            Atomic charges for every atom of the system
        dipoles : numpy.array of real (dimension Natoms x 3)
            Atomic dipole in ATOMIC UNITS (e*bohr) for every atom
                
        '''
        Nat=len(self.coor._value)
        charges=np.zeros(Nat,dtype='f8')
        dipoles=np.zeros((Nat,3),dtype='f8')
        for ii in range(self.grid[0]):
            for jj in range(self.grid[1]):
                for kk in range(self.grid[2]):
                    rr=self._origin+ii*self._step[0,:]+jj*self._step[1,:]+kk*self._step[2,:]
                    dist_min=30.0
                    index=0
                    for ll in range(len(self.coor._value)):
                        dist=np.sqrt(np.dot(rr-self.coor._value[ll],rr-self.coor._value[ll]))
                        if dist<dist_min:
                            index=ll
                            dist_min=np.copy(dist)
                    charges[index]+=self.data[ii,jj,kk]
                    dipoles[index,:]+=(rr-self.coor._value[index])*self.data[ii,jj,kk]
        dV=np.dot(self._step[0,:],np.cross(self._step[1,:],self._step[2,:]))
        print('Atomic dipole calculated by function calc_atomic_properties was chaged from -dipole to dipole. Make sure that you are using right value')
        return charges*dV,-dipoles*dV
        
    def _elpot_at_position(self,position):
        ''' Calculate electrostatic potential for electronic density assumed that
        it is composed of cubic boxes with homogenous charge distribition 

        **THIS IS WERY CRUDE APPROXIMATION AND I HAVE SHOWN BY COMPARING CALCULATED
        POTENTIAL FROM ATOMIC ORBITALS WITH THIS ONE THAT IT DOESN'T PROVIDE (NOT
        EVEN CLOSE TO) REAL POTENTIAL**
        
        Parameters
        ----------  
        position : numpy.array of real (dimension 3)
            Coordinates in ATOMIC UNITS (Bohr) of point where we would like to 
            calculate electrostatic potential 
        Returns
        ----------
        result : real
            Potential at `position` in ATOMIC UNITS
        
        '''
        result=0.0        
        
        def aux_function(rr,stepx,stepy,stepz,t):
            res=scipy.special.erf(t*(stepx/2-rr[0]))+scipy.special.erf(t*(stepx/2+rr[0])) 
            res=res * (scipy.special.erf(t*(stepy/2-rr[1]))+scipy.special.erf(t*(stepy/2+rr[1])))
            res=res * (scipy.special.erf(t*(stepz/2-rr[2]))+scipy.special.erf(t*(stepz/2+rr[2])))
            res=res/t**3
            
            return res
        
        rr1=np.copy(position)
        for m in range(self.grid[0]):
            for n in range(self.grid[1]):
                for o in range(self.grid[2]):
                    rr2=self._origin + m*self._step[0,:]+n*self._step[1,:]+o*self._step[2,:]
                    dr=rr1-rr2
                    tmax=max([5/np.abs(np.abs(dr[0])-np.abs(self._step[0,0]/2)),5/np.abs(np.abs(dr[1])-np.abs(self._step[1,1]/2)),5/np.abs(np.abs(dr[2])-np.abs(self._step[2,2]/2))])
                    #if tmax<5e-1:
                    #    ESP_Grid[i,j,k]-=self.data[m,n,o]/np.sqrt(np.dot(dr,dr))*dV
                    #else:
                    tmax=max([200,tmax])
                    aux_function_partial = partial(aux_function,dr,self._step[0,0],self._step[1,1],self._step[2,2])
                    result-=self.data[m,n,o]*np.pi/4*scipy.integrate.quadrature(aux_function_partial,0,tmax,tol=1e-05,maxiter=100)[0]
        return result

    def _dens_to_ESP2(self):
        ''' This should create electrostatic potential grid file from electronic
        density assumed that it is composed of cubic boxes with homogenous 
        charge distribition.

        **THIS IS WERY CRUDE APPROXIMATION AND I HAVE SHOWN BY COMPARING CALCULATED
        POTENTIAL FROM ATOMIC ORBITALS WITH THIS ONE THAT IT DOESN'T PROVIDE (NOT
        EVEN CLOSE TO) REAL POTENTIAL**
        
        '''
        ESP=DensityGrid(self.origin,self.grid,self.step,None,Coor=self.coor.value,At_charge=self.at_charge)
        ''' Calculate volume element '''
        vecX=np.copy(self._step[0,:])
        vecY=np.copy(self._step[1,:])
        vecZ=np.cross(vecX,vecY)
        dV=np.dot(vecZ,self._step[2,:])
        ESP._origin=ESP._origin+self._step[0,:]/2.0+self._step[1,:]/2.0+self._step[2,:]/2.0
        
        ESP_Grid=np.zeros((self.grid[0],self.grid[1],self.grid[2]),dtype='f8')        
        
        def aux_function(rr,stepx,stepy,stepz,t):
            res=scipy.special.erf(t*(stepx/2-rr[0]))+scipy.special.erf(t*(stepx/2+rr[0])) 
            res=res * (scipy.special.erf(t*(stepy/2-rr[1]))+scipy.special.erf(t*(stepy/2+rr[1])))
            res=res * (scipy.special.erf(t*(stepz/2-rr[2]))+scipy.special.erf(t*(stepz/2+rr[2])))
            res=res/t**3
            
            return res
        
        for i in range(ESP.grid[0]):
                print(i,'/',ESP.grid[0])
                for j in range(ESP.grid[1]):
                    for k in range(ESP.grid[2]):
                        rr1=ESP._origin + i*ESP._step[0,:]+j*ESP._step[1,:]+k*ESP._step[2,:]
                        for m in range(self.grid[0]):
                            for n in range(self.grid[1]):
                                for o in range(self.grid[2]):
                                    rr2=self._origin + m*self._step[0,:]+n*self._step[1,:]+o*self._step[2,:]
                                    dr=rr1-rr2
                                    tmax=max([5/np.abs(np.abs(dr[0])-np.abs(self._step[0,0]/2)),5/np.abs(np.abs(dr[1])-np.abs(self._step[1,1]/2)),5/np.abs(np.abs(dr[2])-np.abs(self._step[2,2]/2))])
                                    if tmax<5e-1:
                                        ESP_Grid[i,j,k]-=self.data[m,n,o]/np.sqrt(np.dot(dr,dr))*dV
                                    else:
                                        tmax=max([200,tmax])
                                        aux_function_partial = partial(aux_function,dr,self._step[0,0],self._step[1,1],self._step[2,2])
                                        ESP_Grid[i,j,k]-=self.data[m,n,o]*np.sqrt(np.pi)/4*scipy.integrate.quadrature(aux_function_partial,0,tmax,tol=1e-05,maxiter=100)[0]
                        ESP_Grid[i,j,k]-=np.pi/tmax**2*self.data[i,j,k]
                                    
        #ESP_Grid=ESP_Grid
        #for m in range(ESP.grid[0]):
        #        for n in range(ESP.grid[1]):
        #            for o in range(ESP.grid[2]):
        #                for ii in range(len(self.coor)):
        #                    dr=ESP.origin + m*ESP.step[0,:]+n*ESP.step[1,:]+o*ESP.step[2,:]-self.coor[ii]
        #                    norm2=np.sqrt(np.dot(dr,dr))                        
        #                    ESP_Grid[m,n,o]+=self.at_charge[ii]/norm2
        ESP.data=np.copy(ESP_Grid)
        return ESP
        
        
