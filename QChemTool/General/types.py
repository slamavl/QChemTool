# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:10:26 2017

@author: Vladislav Sláma
"""

import numpy
import numbers
from functools import partial
    
def UnitsManaged(name):
    """Scalar property with units managed
    
    Warning: The type of the property depends on the object; The object
    has to be EnergyUnitsManaged or similar.
    """    
    storage_name = '_'+name
    
    @property
    def prop(self): 
        val = getattr(self,storage_name)
        return self.convert_2_current_u(val) # This is a method defined in
                                             # the class which handles units
    @prop.setter
    def prop(self,value):
        setattr(self,storage_name,self.convert_2_internal_u(value))
        
    return prop         
        

        
def UnitsManagedArray(name,shape=None):
    """Array property with units managed
    
    Warning: The type of the property depends on the object; The object
    has to be EnergyUnitsManaged, PositionUnitsManaged or similar.
    """
    
    storage_name = '_'+name
    
    @property
    def prop(self): 
        val = getattr(self,storage_name)
        return self.convert_2_current_u(val) # This is a method defined in
                                             # the class which handles units


    @prop.setter
    def prop(self,value):

        try:
            vl = check_numpy_array(value)
            if not (shape == None):
                if not (shape == vl.shape):
                    raise TypeError(
                    '{} must be of shape {}'.format(name,shape))  
            setattr(self,storage_name,self.convert_2_internal_u(vl))
        except:
            raise TypeError(
            '{} must be either a list or numpy.array'.format(name))
    return prop 

def EnergyUnitsManagedArray(name,shape=None):
    """Array property with units managed
    
    Warning: The type of the property depends on the object; The object
    has to be EnergyUnitsManaged, PositionUnitsManaged or similar.
    """
    
    storage_name = '_'+name
    
    @property
    def prop(self): 
        val = getattr(self,storage_name)
        return self.convert_energy_2_current_u(val) # This is a method defined in
                                             # the class which handles units


    @prop.setter
    def prop(self,value):

        try:
            vl = check_numpy_array(value)
            if not (shape == None):
                if not (shape == vl.shape):
                    raise TypeError(
                    '{} must be of shape {}'.format(name,shape))  
            setattr(self,storage_name,self.convert_energy_2_internal_u(vl))
        except:
            raise TypeError(
            '{} must be either a list or numpy.array'.format(name))
    return prop 


#def UnitsManagedProperty(name):
#    """Array or scalar property with units managed
#    
#    Warning: The type of the property depends on the object; The object
#    has to be EnergyUnitsManaged, PositionUnitsManaged or similar.
#    """
#    
#    storage_name = '_'+name
#    
#    @property
#    def prop(self): 
#        val = getattr(self,storage_name)
#        return self.convert_2_current_u(val) # This is a method defined in
#                                             # the class which handles units
#
#
#    @prop.setter
#    def prop(self,value):
#        try:
#            setattr(self,storage_name,self.convert_2_internal_u(value))
#        except:
#            raise TypeError(
#            '{} must be either a list or numpy.array'.format(name))
#    return prop

def check_numpy_array(val):
    """ Checks if argument is a numpy array. 
    
    If the argument is a list, it converts it into numpy array. Otherwise,
    error occurs.
    
    """
    if isinstance(val,numpy.ndarray):
        return val
    elif isinstance(val,list):
        try:
            vl = numpy.array(val)
        except:
            raise TypeError('Numerical array is required')
        return numpy.array(vl)
    else:
        raise TypeError('List or numpy.ndarray required')

    

#UnitsManagedArray = units_managed_array_property()
#UnitsManaged = units_managed_property()
  
def typed_property(name,dtype):
    storage_name = '_'+name
            
    @property
    def prop(self):
        return getattr(self,storage_name)
    
    @prop.setter
    def prop(self,value):
        if isinstance(value,dtype):
            setattr(self,storage_name,value)
        else:
            raise TypeError('{} must be of type {}'.format(name,dtype))
        return prop
     
Float   = partial(typed_property,dtype=numbers.Real)
Integer = partial(typed_property,dtype=numbers.Integral)
Bool = partial(typed_property,dtype=bool)
