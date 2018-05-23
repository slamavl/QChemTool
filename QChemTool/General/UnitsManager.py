# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:49:36 2017

@author: Vladislav Sláma
"""

import numpy as np
from .units import conversion_facs_frequency
from .units import conversion_facs_energy
from .units import conversion_facs_position

class Singleton(type):
    """Base type of singletons, such as the main Quantarhei class Manager
    
    
    Recipe "Creating a singleton in Python" from Stack Overflow.
    
    
    Usage:
    ------
    class MyClass(BaseClass, metaclass=Singleton)
    
    """

    _instances = {}
    
    def __call__(cls,*args,**kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton,cls).__call__(*args, **kwargs)
        return cls._instances[cls]



class Manager(metaclass=Singleton):
    """ Main package Manager
    This class handles several important package wide tasks:
    1) Usage of units across objects storing data
    -) Calls to proper optimized implementations of numerically heavy
       sections of the calculations
    Manager is a singleton class, only one instance exists at all times
    and all managing objects have the instance of the Manager.
    
    Properies
    ---------
    allower_utypes : list
        contains a list of unit types which can be controlled by the Manager
        
    units : dictionary
        dictionary of available units for each units type
        
    units_repre : dictionary
        dictionary of abreviations used to represent various units
        
    units_repre_latex : dictionary
        dictionary of latex prepresentations of available units
    
    """
    

    # hard wired unit options
    allowed_utypes = ["energy",
                      "frequency",
                      "dipolemoment",
                      "temperature",
                      "time",
                      "position"]

    units = {"energy"       : ["int", "1/cm", "eV", "meV", "THz",
                               "J", "SI", "nm", "Ha", "AU"],
             "frequency"    : ["1/fs", "int", "1/cm", "THz","Hz","SI","nm"],
             "dipolemoment" : ["Debye"],
             "temperature"  : ["2pi/fs", "int", "Kelvin", "Celsius",
                               "1/cm", "eV", "meV", "Thz", "SI"],
             "time"         : ["fs", "int", "as", "ps", "ns", "Ms","ms",
                               "s", "SI"],
             "position"     : ["int","Bohr","Angstrom","m","SI"]}
             

    units_repre = {"Kelvin":"K",
                   "Celsius":"C",
                   "Debye":"D",
                   "AU":"AU",
                   "Hartree":"Ha",
                   "Bohr":"Bohr",
                   "Angstrom":"A",
                   "1/cm":"1/cm",
                   "THz":"THz",
                   "eV":"eV",
                   "2pi/fs":"2pi/fs",
                   "int":"2pi/fs",
                   "meV":"meV",
                   "nm":"nm"}
                   
    units_repre_latex = {"Kelvin":"K",
                   "Celsius":"C",
                   "Debye":"D",
                   "1/cm":"cm$^-1$",
                   "THz":"THz",
                   "eV":"eV",
                   "1/fs":"fs$^{-1}$",
                   "2pi/fs":"rad$\cdot$fs$^{-1}$",
                   "meV":"meV",
                   "nm":"nm",
                   "AU":"AU",
                   "Bohr":"Bohr",
                   "Angstrom":"\AA",
                   "Hartree":"Ha"}                  

    def __init__(self):

        #self.current_units = {}
        # Load or save 
        self.current_units = {"energy":"Ha", "frequency":"1/fs",
                               "dipolemoment":"AU",
                               "temperature":"Kelvin",
                               "position":"Bohr"}

        # internal units are hardwired
        self.internal_units = {"energy":"Ha", "frequency":"1/fs",
                               "dipolemoment":"AU",
                               "temperature":"Kelvin",
                               "position":"Bohr"}
        
        self.parallel_conf = None
        
        self.save_dict = {}
# TODO: delete print
        print('Manager initialized')
        
        
    def unit_repr(self,utype="energy",mode="current"):
        """Returns a string representing the currently used units
        
        
        """        
    
    
        if utype in self.allowed_utypes:
            if mode == "current":
                return self.units_repre[self.current_units[utype]]
            elif mode == "internal":
                return self.units_repre[self.internal_units[utype]]
            else:
                raise Exception("Unknown representation mode")
            
        else:
            raise Exception("Unknown unit type")
            
    def unit_repr_latex(self,utype="energy",mode="current"):
        """Returns a string representing the currently used units
        
        
        """        
    
    
        if utype in self.allowed_utypes:
            if mode == "current":
                return self.units_repre_latex[self.current_units[utype]]
            elif mode == "internal":
                return self.units_repre_latex[self.internal_units[utype]]
            else:
                raise Exception("Unknown representation mode")
            
        else:
            raise Exception("Unknown unit type")            
            
            
            
    def set_current_units(self, utype, units):
        """Sets current units
        
        
        """
        self._saved_units = {}
        self._saved_units[utype] = self.get_current_units(utype)
        
        if utype in self.allowed_utypes:
            if units in self.units[utype]:
                self.current_units[utype] = units
            else:
                print('Type:',utype,'Units:',units)
                raise Exception("Unknown units of %s" % utype)
        else:
            raise Exception("Unknown type of units")
        
    def unset_current_units(self, utype):
        """Restores previously saved units of a given type
        
        """
        try:
            cunits = self._saved_units[utype]
        except KeyError:
            raise Exception("Units to restore not found")
            
        if utype in self.allowed_utypes:
            if cunits in self.units[utype]:
                self.current_units[utype] = cunits
            else:
                raise Exception("Unknown units of %s" % utype)
        else:
            raise Exception("Unknown type of units")
        
        
        
    def get_current_units(self, utype):
        """
        
        """
        if utype in self.allowed_utypes:
            return self.current_units[utype]        
        else:
            raise Exception("Unknown type of units")
            
            
    def convert_energy_2_internal_u(self,val):
        """Convert energy from currently used units to internal units
        
        Parameters
        ==========
        val : number, array, list, tuple of numbers
            values to convert            
        
        """
        units = self.current_units["energy"]
        cfact = conversion_facs_energy[self.current_units["energy"]]
        
        if val is None:
            return None
        
        # special handling for nano meters
        if units == "nm":
            # zero is interpretted as zero energy
            try:
                ret = np.zeros(val.shape, dtype=val.dtype)
                ret[val!=0.0] = 1.0/val[val!=0]
                return ret/cfact
            except:            
                return (1.0/val)/cfact
            #if val == 0.0:
            #    return 0.0
            #return (1.0/val)/cfact
        else:
            return val/cfact
        
            
    def convert_energy_2_current_u(self,val):
        """Convert energy from internal units to currently used units
        
        Parameters
        ==========
        val : number, array, list, tuple of numbers
            values to convert            
        
        """
        units = self.current_units["energy"]
        cfact = conversion_facs_energy[units]
        
        # special handling for nanometers
        if units == "nm":
            # zero is interpretted as zero energy
            try:
                ret = np.zeros(val.shape, dtype=val.dtype)
                ret[val!=0.0] = 1.0/val[val!=0]
                return ret/cfact
            except:            
                return (1.0/val)/cfact
        else:
            return val*cfact 
        

    def convert_frequency_2_internal_u(self,val):
        """Convert frequency from currently used units to internal units
        
        Parameters
        ==========
        val : number, array, list, tuple of numbers
            values to convert            
        
        """
        return val*conversion_facs_frequency[self.current_units["frequency"]]

        
    def convert_frequency_2_current_u(self,val):   
        """Convert frequency from internal units to currently used units
        
        Parameters
        ==========
        val : number, array, list, tuple of numbers
            values to convert            
        
        """
        return val/conversion_facs_frequency[self.current_units["frequency"]] 
        
    
    def convert_position_2_internal_u(self,val):
        """Convert position from internal units to currently used units
        
        Parameters
        ==========
        val : number, array, list, tuple of numbers
            values to convert            
        
        """
        
        if val is None:
            return None
        
        units = self.current_units["position"]
        cfact = conversion_facs_position[units]
        
        return val/cfact 
        

    def convert_position_2_current_u(self,val):
        """Convert position from internal units to currently used units
        
        Parameters
        ==========
        val : number, array, list, tuple of numbers
            values to convert            
        
        """
        units = self.current_units["position"]
        cfact = conversion_facs_position[units]
        
        if val is None:
            return val
        
        return val*cfact 
        
    
#    def get_DistributedConfiguration(self):
#        """
#        
#        """
#        from .parallel import DistributedConfiguration
#        
#        if self.parallel_conf is None:
#            self.parallel_conf = DistributedConfiguration()
#        return self.parallel_conf
        
class Managed:
    """Base class for managed objects 
    
    
    
    """
    
    manager = Manager()


class UnitsManaged(Managed):
    """Base class for objects with management of units
    
    
    """    
    
    def convert_energy_2_internal_u(self,val):
        return self.manager.convert_energy_2_internal_u(val)
        
    def convert_energy_2_current_u(self,val):
        return self.manager.convert_energy_2_current_u(val)

    def convert_position_2_internal_u(self,val):
        return self.manager.convert_position_2_internal_u(val)
    
    def convert_position_2_current_u(self,val):
        return self.manager.convert_position_2_current_u(val)
    
    def convert_frequency_2_internal_u(self,val):
        return self.manager.convert_frequency_2_internal_u(val)
    
    def convert_frequency_2_current_u(self,val):
        return self.manager.convert_frequency_2_current_u(val)
        
    def unit_repr(self,utype="energy"):
        return self.manager.unit_repr(utype)
        
    def unit_repr_latex(self,utype="energy"):
        return self.manager.unit_repr_latex(utype)
    
    
class units_context_manager:
    """General context manager to manage physical units of values 
    
    
    """
    
    def __init__(self,utype="energy"):
        self.manager = Manager()
        if utype in self.manager.allowed_utypes:
            self.utype = utype
        else:
            raise Exception("Unknown units type")
    
    def __enter__(self):
        pass
    
    def __exit__(self):
        pass

        
class EnergyUnitsManaged(Managed):
    
    utype = "energy"
    units = "Ha"    
    
    def convert_2_internal_u(self,val):
        return self.manager.convert_energy_2_internal_u(val)
        
    def convert_2_current_u(self,val):
        return self.manager.convert_energy_2_current_u(val)

    def unit_repr(self):
        return self.manager.unit_repr("energy")

    def unit_repr_latex(self,utype="energy"):
        return self.manager.unit_repr_latex(utype)
        

class energy_units(units_context_manager):
    """Context manager for units of energy
    
    
    """
    
    def __init__(self,units):
        super().__init__(utype="energy")
        
        if units in self.manager.units["energy"]:
            self.units = units
        else:
            raise Exception("Unknown energy units")
            
    def __enter__(self):
        # save current energy units
        self.units_backup = self.manager.get_current_units("energy")
        self.manager.set_current_units(self.utype,self.units)
        
    def __exit__(self,ext_ty,exc_val,tb):
        self.manager.set_current_units("energy",self.units_backup)
 

class PositionUnitsManaged(Managed):
    
    utype = "position"
    units = "Bohr"    
    
    def convert_2_internal_u(self,val):
        return self.manager.convert_position_2_internal_u(val)
        
    def convert_2_current_u(self,val):
        return self.manager.convert_position_2_current_u(val)

    def unit_repr(self):
        return self.manager.unit_repr("position")

    def unit_repr_latex(self,utype="position"):
        return self.manager.unit_repr_latex(utype)  


class position_units(units_context_manager):
    """Context manager for units of position
    
    
    """
    
    def __init__(self,units):
        super().__init__(utype="position")
        
        if units in self.manager.units["position"]:
            self.units = units
        else:
            raise Exception("Unknown energy units")
            
    def __enter__(self):
        # save current energy units
        self.units_backup = self.manager.get_current_units("position")
        self.manager.set_current_units(self.utype,self.units)
        
    def __exit__(self,ext_ty,exc_val,tb):
        self.manager.set_current_units("position",self.units_backup)
        
class FrequencyUnitsManaged(Managed):
    
    utype = "frequency"
    units = "1/fs"    
    
    def convert_2_internal_u(self,val):
        return self.manager.convert_frequency_2_internal_u(val)
        
    def convert_2_current_u(self,val):
        return self.manager.convert_frequency_2_current_u(val)

    def unit_repr(self):
        return self.manager.unit_repr("frequency")

    def unit_repr_latex(self,utype="frequency"):
        return self.manager.unit_repr_latex(utype)  

class frequency_units(units_context_manager):
    """Context manager for units of frequency
    
    
    """
    
    def __init__(self,units):
        super().__init__(utype="frequency")
        
        if units in self.manager.units["frequency"]:
            self.units = units
        else:
            raise Exception("Unknown frequency units")
            
    def __enter__(self):
        # save current energy units
        self.units_backup = self.manager.get_current_units("frequency")
        self.manager.set_current_units(self.utype,self.units)
        
    def __exit__(self,ext_ty,exc_val,tb):
        self.manager.set_current_units("frequency",self.units_backup)       


def set_current_units(units=None):
    """Sets units globaly without the need for a context manager
    
    """
    manager = Manager()    
    if units is not None:
        # set units using a supplied dictionary
        for utype in units:
            if utype in manager.allowed_utypes:
                un = units[utype]
                # handle the identity of "frequency" and "energy"
#                if utype=="frequency":
#                    utype="energy"
#                    un = units["frequency"]
                    
                manager.set_current_units(utype,un)
            else:
                raise Exception("Unknown units type %s" % utype)

    else:
        # reset units to the default
        for utype in manager.internal_units:
            if utype in manager.allowed_utypes:
                manager.set_current_units(utype,manager.internal_units[utype])
            else:
                raise Exception("Unknown units type %s" % utype)
        
        
