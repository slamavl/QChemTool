# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:12:49 2016

@author: uzivatel
"""
import numpy as np 
from  copy import deepcopy

from ...General.UnitsManager import PositionUnitsManaged,EnergyUnitsManaged,energy_units
from ...General.types import UnitsManagedArray,UnitsManaged
from ..positioningTools import RotateAndMove, RotateAndMove_1
from ...General.constants import AuToDebye
     
class Coordinate(PositionUnitsManaged):
    value = UnitsManaged("value")
    
    ''' Class for description of coordinates '''
    def __init__(self,coor):
        if coor.__class__.__name__=='Coordinate':
            self=deepcopy(coor)
        elif coor is None:
            self.value=None
        else:
            if np.ndim(coor)==1:
                self.value=np.array(coor,dtype='f8')
            else:
                self.value=np.array(coor,dtype='f8')
        
    def add_coor(self,coor):
        npcoor=np.array(coor,dtype='f8')
        if self.value is None:
            if coor.__class__.__name__=='Coordinate':
                self=deepcopy(coor)
            else:
                self.value=np.array(coor,dtype='f8')
        elif npcoor.ndim==1:
            self.value=np.vstack((self.value,npcoor))
        elif npcoor.ndim==2:
            if npcoor.shape[0]==3 and npcoor.shape[1]!=3:
                npcoor=npcoor.T
                print('add_coor: Coordinates were transformed into required shape Nx3')
            self.value=np.vstack((self.value,npcoor))
        else:
            raise Warning('Coordinates could be vectors or matrixes no higher dimensions are allowed')

    def del_coor(self,indx):
        if self._value.shape[0]==1 or np.ndim(self._value)==1:
            print('del_coor: Deleting last coordinate entry!')
        if self._value.shape[0]<=np.max(indx):
            raise Warning('Index of coordinate you are trying to delete is higher than number of coordinates!')
        else:
            self._value=np.delete(self._value,indx,axis=0)

# TODO: replace this by using vector
    def move(self,dx,dy,dz):
        TransVec=np.array([dx,dy,dz])
        if np.ndim(self.value)==1:
            self.value=self.value+TransVec
        else:
            TransVec=np.tile(TransVec,(self._value.shape[0],1))
            self.value=self.value+TransVec
    
    def rotate(self,rotxy,rotxz,rotyz):
        self._value=RotateAndMove(self._value,0.0,0.0,0.0,rotxy,rotxz,rotyz)
        
    def rotate_1(self,rotxy,rotxz,rotyz):
        self._value=RotateAndMove_1(self._value,0.0,0.0,0.0,rotxy,rotxz,rotyz)
    
    
    def copy(self):
        coor_copy=deepcopy(self)
        return coor_copy

    def normalize(self):
        """ Normalization of position vectors in current units 
        
        """
        
        if np.ndim(self._value)==1:
            self.value=self.value/np.linalg.norm(self.value)
        else:
            Norm=np.linalg.norm(self.value, axis=1)
            Norm=np.tile(Norm,(3,1))
            Norm=Norm.T
            self.value=np.divide(self.value,Norm)
    
    def _normalize(self):       # Normalization in interanl coordinates
        """ Normalization of position vectors in internal units (Bohr) 
        
        """
        
        if np.ndim(self._value)==1:
            self._value=self.value/np.linalg.norm(self._value)
        else:
            Norm=np.linalg.norm(self._value, axis=1)
            Norm=np.tile(Norm,(3,1))
            Norm=Norm.T
            self._value=np.divide(self._value,Norm)


'''----------------------- TEST PART --------------------------------'''
if __name__=="__main__":
    print(' ')
    print('TESTS')
    print('-------')
    print('Coordinate class test:')
    coor=Coordinate([1.2,2,3])
    coor.add_coor([2,5,8])
    coor.add_coor([[1,5,9],[7,5,3],[2,4.3,5]])
    if np.array_equal(coor.value,np.array([[1.2,2,3],[2,5,8],[1,5,9],[7,5,3],[2,4.3,5]],dtype='f8')):
        print('     add_coor:       OK')
    else:
        print('     add_coor:       Error')
    coor.del_coor(2)
    coor.del_coor([1,2])
    if np.array_equal(coor.value,np.array([[1.2,2,3],[2,4.3,5]],dtype='f8')):
        print('     del_coor:       OK')
    else:
        print('     del_coor:       Error')
        
        
        
class Energy(EnergyUnitsManaged):
    ''' Class containing all information about atomic orbitals '''
    
    value = UnitsManaged("value")
    
    def __init__(self,energy):
        if energy is None:
            self.value=None
        elif np.isscalar(energy):
            self.value=energy
        else:
            self.value=np.array(energy,dtype='f8')
    
        
    def add_energy(self,energy):              
        if self._value is None:
            if np.isscalar(energy):
                self.value=energy
            elif energy.__class__.__name__=='Energy':
                self.value=energy.value
            else:
                self.value=np.array(energy,dtype='f8')
        elif np.isscalar(self._value):
            if np.isscalar(energy):
                val=np.append(self.value,energy)
                self.value=np.array(val,dtype='f8')
            elif energy.__class__.__name__=='Energy':
                val=np.append(self.value,energy.value)
                self.value=np.array(val,dtype='f8')
            else:
                val=np.append(self.value,energy)
                self.value=np.array(val,dtype='f8')
        else:
            if np.isscalar(energy):
                self.value=np.append(self.value,energy)
            elif energy.__class__.__name__=='Energy':
                self.value=np.append(self.value,energy.value)
            else:
                self.value=np.append(self.value,energy)

    def del_energy(self,indx):
        if len(self._value)==1:
            print('del_coor: Deleting last coordinate entry!')
        if len(self._value)<=np.max(indx):
            raise Warning('Index of coordinate you are trying to delete is higher than number of coordinates!')
        else:
            self._value=np.delete(self._value,indx)
    
#    def Ha2cm_1(self):
#        if self.units=='Hartree':
#            self.value=self.value*const.HaToInvcm
#            self.units='cm-1'
#        elif self.units!='cm-1':
#            print('Unknown energy units. Unable to transform in inverse centimeters')
        
    def copy(self):
        energy_new=deepcopy(self)
        return energy_new
    
    
    
'''----------------------- TEST PART --------------------------------'''
if __name__=="__main__":
    print(' ')
    print('Energy class test:')
    energy=Energy(None)
    energy.add_energy(1)
    energy.add_energy(2)
    energy2=energy.copy()
    energy2.add_energy([1.2,3.5])
    if np.array_equal(energy.value,np.array([1,2],dtype='f8')) and np.array_equal(energy2.value,np.array([1,2,1.2,3.5],dtype='f8')):
        print('     add_energy:     OK')
    else:
        print('     add_energy:     Error')
    
    energy=Energy(None)
    energy.add_energy(1)
    with energy_units("1/cm"):
        energy.add_energy(10000)
        energy2=energy.copy()
    if np.array_equal(energy2.value,np.array([1.00000000e+00,0.045563352527674468])):
        print('     units manager:  OK')
    else:
        print('     units manager:  Error')
        
# TODO: Dipole managed class 
class Dipole:
    ''' Class containing all information about atomic orbitals '''
    def __init__(self,dipole,units):
        self.value=np.array(dipole,dtype='f8')
        self.units=units
    
    def rotate(self,rotxy,rotxz,rotyz):
        self.value=RotateAndMove(self.value,0.0,0.0,0.0,rotxy,rotxz,rotyz)
        
    def rotate_1(self,rotxy,rotxz,rotyz):
        self.value=RotateAndMove_1(self.value,0.0,0.0,0.0,rotxy,rotxz,rotyz)
        
    def AU2Debye(self):
        if self.units=='AU':
            self.value=self.value*AuToDebye
            self.units='Debye'
        elif self.units!='Debye':
            raise Warning('Units are not AU or Debye and therefore cannot be transformed')
        
    def Debye2AU(self):
        if self.units=='Debye':
            self.value=self.value/AuToDebye
            self.units='AU'
        elif self.units!='AU':
            raise Warning('Units are not AU or Debye and therefore cannot be transformed')
            
    def in_AU(self):
        if self.units=='Debye':
            return self.value/AuToDebye
        if self.units=='AU':
            return self.value
    
    def in_Debye(self):
        if self.units=='AU':
            return self.value*AuToDebye
        if self.units=='Debye':
            return self.value
            
class Grid(PositionUnitsManaged):
    ''' Class containing information about descrete 3D grid 
    
    Notes
    --------
    Only 'delta' and 'origin' property are units managed. 'X', 'Y' and 'Z' grids
    are always in atomic units.
    
    '''
    
    delta=UnitsManaged("delta")
    origin=UnitsManaged("origin")

# TODO: Position units managed  delta, origin, should be position managed X,Y,Z
# used for calc -> no need to used manaed units
    def __init__(self):
        self.intit=False
        self.X=None
        self.Y=None
        self.Z=None
        self.delta=None
        self.origin=None
    
    def set_to_geom(self,coor,units,extend=5.0,step=0.1):
        min_=np.zeros(3,dtype='f8')
        max_=np.zeros(3,dtype='f8')
        N_=np.zeros(3,dtype='i4')
        
        for i in range(3):
            min_[i] = min(coor[:,i]) - abs(extend)
            max_[i] = max(coor[:,i]) + abs(extend)
            N_[i] = np.int(np.ceil((max_[i] - min_[i])/step)) + 1
            # Correct maximum value, if necessary
            max_[i] = (N_[i] - 1)*step + min_[i]
        
        grid = [[],[],[]]
        delta_ = np.zeros(3,dtype='f8')
        
        # Loop over the three dimensions 
        for ii in range(3):
            if max_[ii] == min_[ii]:
                # If min-value is equal to max-value, write only min-value to grid  
                grid[ii]=np.array([min_[ii]],dtype='f8')
                delta_[ii] = 1.0
            else:
                # Calculate the grid using the input parameters 
                delta_[ii] = (max_[ii]-min_[ii]) / float(N_[ii] - 1)
                grid[ii] = min_[ii] + np.arange(N_[ii]) * delta_[ii]  
          
        # Write grid 
        x = np.array(grid[0])
        y = np.array(grid[1])  
        z = np.array(grid[2])
        
        # Change units to internal because every operation is defined only in AU
        x = PositionUnitsManaged.manager.convert_position_2_internal_u(x)
        y = PositionUnitsManaged.manager.convert_position_2_internal_u(y)
        z = PositionUnitsManaged.manager.convert_position_2_internal_u(z)
        
        
        Y_grid,X_grid,Z_grid = np.meshgrid( y, x, z) # Aby korespondovalo s vypisem Y se musi menit pro 2. dimenzi a kdyby to bylo X_grid,Y_grid,Z_grid tak by se myslim menilo ve 3.dim 
          
        self.X=X_grid
        self.Y=Y_grid
        self.Z=Z_grid
        self.delta=delta_
        self.origin=min_   
        self.intit=True
        
    def init_from_cub(self,density):
        self.origin=np.copy(density.origin)
        delta=np.zeros(3)
        for ii in range(3):
            delta[ii]=density.step[ii,ii]
        self.delta=delta.copy()
        
        N_=np.copy(density.grid)    
        grid = [[],[],[]]
        for ii in range(3):
            # Calculate the grid using the input parameters 
            grid[ii] = self.origin[ii] + np.arange(N_[ii]) * self.delta[ii]
        
        # Write grid 
        x = np.array(grid[0])
        y = np.array(grid[1])  
        z = np.array(grid[2])
        
        # Change units to internal because every operation is defined only in AU
        x = PositionUnitsManaged.manager.convert_position_2_internal_u(x)
        y = PositionUnitsManaged.manager.convert_position_2_internal_u(y)
        z = PositionUnitsManaged.manager.convert_position_2_internal_u(z)
        
        self.Y,self.X,self.Z = np.meshgrid( y, x, z)
        self.init=True

    def get_axes(self,units='Current'):
        """ Returns x, y and z axis for the grid """
        if units=='Current':
            x = PositionUnitsManaged.manager.convert_position_2_current_u(self.X[:,0,0])
            y = PositionUnitsManaged.manager.convert_position_2_current_u(self.Y[0,:,0])
            z = PositionUnitsManaged.manager.convert_position_2_current_u(self.Z[0,0,:])
        else:
            x=self.X[:,0,0]
            y=self.Y[0,:,0]
            z=self.Z[0,0,:]
        return x,y,z
    
    def get_dim(self):
        dim=np.zeros(3,dtype='i4')
        for ii in range(3):
            dim[ii]=self.X.shape[ii]
        return dim
    
    def copy(self):
        new_grid=deepcopy(self)
        return new_grid
    
# TODO: introduce rotation of the grid

class PositionAxis(PositionUnitsManaged):
    """ Position axis with units management
    
    Notes
    -------
    Rewrite this class to obtain value only if called otherwise store only origin
    and step -> easier for shifting the axis
    
    """
    
    value = UnitsManagedArray("value")
    start = UnitsManaged("start")
    step = UnitsManaged("step")
    
    def __init__(self,start,step,nsteps):
        self.start = start
        self.step = step
        self.nsteps = nsteps
        self.value = np.linspace(start,start+(nsteps-1)*step,nsteps)
        

    