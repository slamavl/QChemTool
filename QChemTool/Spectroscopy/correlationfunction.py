# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:13:50 2018

@author: Vladislav Sláma
"""

# Suggestions for improvement
#-------------------------------------------
# l132 : also values should be fourier transformed if frequency axis is used as
#        an input - FIXED 
# l191 : Creating user defined function within if values is None - FIXED 
# l206 : Calculate reorganization energy from correlation function
# l367 : the same as previous
# l405 : It can add only non-userdefined functions. It would be better to use
#        deepcopy to create copy of correlation function
# l521 : In measure_reorganization_energy it would be better to calculate the
#        reorganization energy by doing fourier transform of imaginary part 
#        of correlation function then divide it by frequency and integrate it.
# l660 : Also for user defined values first construct the correlation function
#        and then do the fourier transform (if axis is TimeAxis) it can be 
#        done the same way as in EvenFTCorrelationFunction and 
#        OddFTCorrelationFunction
# l676 : I would protect using user defiend get_OddFTCorrelationFunction and 
#        EvenFTCorrelationFunction with frequency axis or repair the first 
#        comment at l132 - FIXED by l132


# -*- coding: utf-8 -*-
"""
    Quantarhei package (http://www.github.com/quantarhei)
    correlationfunctions module
"""

import numpy
import scipy.interpolate as interp
import scipy.integrate as integrate

from ..General.dfunction import DFunction
from ..General.units import kB_int_freq as kB_int
from ..General.UnitsManager import UnitsManaged
from ..General.UnitsManager import frequency_units
from ..General.timeaxis import TimeAxis
from ..General.frequencyaxis import FrequencyAxis


class CorrelationFunction(DFunction, UnitsManaged):
    """Provides typical Bath correlation function types.
    Most important types of bath or energy gap correlation functions are
    provided. Where possible, the correlation function is calculated
    from the parameters from analytical formulae. Where such formulae are
    not available, correlation function is calculated by transformation
    of the spectral density.
    
    Parameters
    ----------
    axis : TimeAxis
        TimeAxis object specifying the time interval on which the
        correlation function is defined.
    params : dictionary
        A dictionary of the correlation function parameters
    values : optional
        Correlation function can be set by specifying values at all times
    
    Methods
    -------
    is_analytical()
        Returns `True` if the correlation function is calculated from an
        analytical formula, `False` otherwise.
    copy()
        Returns a copy of the CorrelationFunction object
    get_temperature()
        Returns the temperature of the correlation function
    get_reorganization_energy()
        Returns the reorganization energy parameters of
        the correlation function
    measure_reorganization_energy()
        Calculates reorganization energy from the shape of the correlation
        function
    get_FTCorrelationFunction()
        Returns the Fourier transform of the correlation function
    get_EvenFTCorrelationFunction()
        Returns the Fourier transform of the real part of the correlation
        function
    get_OddFTCorrelationFunction()
        Returns the Fourier transform of the imaginary part of the correlation
        function
    get_SpectralDensity()
        Returns numerically calculated spectral density
   
    Types of correlation function provided
    --------------------------------------
    OverdampedBrownian-HighTemperature :
        OverdampedBrownian oscillator in high temperature limit
    OverdampedBrownian :
        General overdampedBrownian oscillator
    
    Notes
    --------
    OverdampedBrownian:
        ctime = params["cortime"] \n
        lamb = params["reorg"] \n
        ``C''(t) = -(lamb/ctime)*exp( -t/ctime )`` \n
        ``C''(w) = (2.0*lamb/ctime)*w/( w**2 + (1.0/ctime)**2 )``
        
    UnderdampedBrownian:
        gm   = params["gamma"] \n
        lamb = params["reorg"] \n
        w0   = params["freq"] \n
        ``C''(t) = -lamb*(w0**2)/sqrt( w0**2 - gm**2/4 )
        *exp( -gm*abs(t)/2 )*sin( sqrt( w0**2 - gm**2/4 )*t )`` \n
        ``C''(w) = 2*gm*lamb*w*(w0**2)/( (gm**2)*(w**2) + (w**2 - w0**2)**2 )``
        \n
    
    
    where ``C(w) = [1+coth( w*hbar/(2*kB*T) )]*C''(w)`` and 
    ``C(t) = C'(t) + iC''(t)`` \n
    For fourier transform componet: \n
    ``C''(w) = int{t,-infty,infty}{exp(i*w*t )*i*C''(t)} = -2 * int{t,0,infty}{sin( w*t )*C''(t)}``
    
    Examples
    --------
    >>> from quantarhei import TimeAxis
    >>> params = dict(ftype="OverdampedBrownian", cortime=100, reorg=20, T=300)
    >>> time = TimeAxis(0.0,1000,1.0)
    >>> with frequency_units("1/cm"):
    ...     cf = CorrelationFunction(time,params)
    >>> with frequency_units("1/cm"):
    ...     print(cf.get_reorganization_energy())
    20.0
   
    Reorganization energy of a correlation function can be calculated from the
    shape of the spectral density by integrating over it. The accuracy
    of such estimation depends on numerics, hence the relative tolerance of
    only 1.0e-4 below
    >>> lamb_definition = cf.get_reorganization_energy()
    >>> lamb_measured = cf.measure_reorganization_energy()
    >>> print(numpy.allclose(lamb_definition, lamb_measured, rtol=1.0e-4))
    True
    
    """

    allowed_types = ("OverdampedBrownian-HighTemperature",
                     "OverdampedBrownian", 
                     "UnderdampedBrownian",
                     "Value-defined")

    analytical_types = ("OverdampedBrownian-HighTemperature",
                        "OverdampedBrownian")
    
    energy_params = ("reorg", "omega", "freq")

    def __init__(self, axis=None, params=None , values=None):
        super().__init__()
        
        if (axis is not None) and (params is not None):
            
            # FIXME: values might also need handling according to specified 
            # energy units
    
            # handle axis (it can be TimeAxis or FrequencyAxis)
            if not isinstance(axis, TimeAxis):
                taxis = axis.get_TimeAxis()
                self.axis = taxis
                
                # My aditional imput
                if isinstance(axis, FrequencyAxis) and values is not None: 
                    # I'm not sure if data should be transformed into internal units or not. data are in internal units in time domain - first transform to int units
                    #intvalues = self.convert_frequency_2_internal_u(values)
                    intvalues = values
                    Fom  = DFunction(axis,intvalues)
                    ft = Fom.get_inverse_Fourier_transform()
                    intvalues = ft.data
                
                # FIXME : also values should be fourier transformed if frequency axis is used as an input
            else:
                self.axis = axis
                intvalues = values
    
            # handle params
            self.params = []  # this will always be a list of components
            p2calc = []
            try:
                # if this passes, we assume params is a dictionary
                params.keys()
                self._is_composed = False
                p2calc.append(params)
                
            except:
                # othewise we assume it is a list of dictionaries 
                self._is_composed = True
                for p in params:
                    p2calc.append(p)
                
                
            self.lamb = 0.0
            self.temperature = -1.0
            self.cutoff_time = 0.0

            for params in p2calc:
                
                try:
                    ftype = params["ftype"]
                    
                    if ftype not in CorrelationFunction.allowed_types:
                        raise Exception("Unknown CorrelationFunction type")
        
                    # we mutate the parameters into internal units
                    prms = {}
                    for key in params.keys():
                        if key in self.energy_params:
                            # For spectroscopy we use frequency internal units instead of energy (which are used for quantum chemistry)
                            prms[key] = self.convert_frequency_2_internal_u(params[key])
                        else:
                            prms[key] = params[key]
                            
                except:
                    raise Exception("Dictionary of parameters does not contain "
                                    +" `ftype` key")
                    
                self.params.append(prms)                    
    
    
            
            #
            # loop over parameter sets
            #
            for prms in self.params:
    
                if ftype == "OverdampedBrownian-HighTemperature":
        
                    self._make_overdamped_brownian_ht(prms) #, values=values)
        
                elif ftype == "OverdampedBrownian":
        
                    self._make_overdamped_brownian(prms) #, values=values)
                    
                elif ftype == "UnderdampedBrownian":
                
                    if values is None:
                        self._make_underdamped_brownian(prms) #, values=values)
                    else:
                        self._make_value_defined(prms, values) # because UnderdampedBrownian is build from FT of spectral density
                
                elif ftype == "Value-defined":
 
                    # FIXME: If values are none we cannot create value defined function from values
                    self._make_value_defined(prms, values)
        
                else:
                    raise Exception("Unknown correlation function type or"+
                                    "type domain combination.")
                
#            else:
#                
#                # My version
#                # FIXME If params list then valueas has to be array - more functions in one 
#                self._make_value_defined(prms, intvalues)

                    #raise Exception("When values are not None"+
                    #            "`ftype` key of params must be Value-defined.")
                
                # Tomas version
#                self._add_me(self.axis, values) 
#                
#                # update reorganization energy
#                self.lamb = 0.0
#                self.temperature = self.params[0]["T"]
#                for prms in self.params:
#                    # TODO: reorganisation energy should be calculated from the
#                    # function and not read from parameters 
#                    self.lamb += prms["reorg"]
#                    if self.temperature != prms["T"]:
#                        raise Exception("Inconsistent temperature! "
#                                        +"Temperatures of all "
#                                        +"components have to be the same")
#                    
#                #FIXME: set cut-off time and temperature
#                #self._set_temperature_and_cutoff_time(self.params[0])
                    
                

    def _matsubara(self, kBT, ctime, nof):
        """Matsubara frequency part of the Brownian correlation function
        
        """
        msf = 0.0
        nut = 2.0*numpy.pi*kBT
        time = self.axis.data
        for i in range(0, nof):
            n = i+1
            msf += nut*n*numpy.exp(-nut*n*time)/((nut*n)**2-(1.0/ctime)**2)
        return msf

    def _set_temperature_and_cutoff_time(self, temperature, ctime):
        """Sets the temperature and cutoff time of for the component
        
        """
        
        # Temperatures of all components have to be the same
        # is this the first time that temperature is assigned?
        if self.temperature == -1.0: 
            self.temperature = temperature
        elif self.temperature != temperature:
            raise Exception("Inconsistent temperature! Temperatures of all "
                            +"components have to be the same")
            
        # longest cortime has to be preserved
        new_cutoff_time = 5.0*ctime             # Correlation time is usually defined as time when exponential decay = 1/e so at 5 times larger time it should be zero 
        if new_cutoff_time > self.cutoff_time: 
            self.cutoff_time = new_cutoff_time
            
            
    def _make_overdamped_brownian(self, params): #, values=None):
        """Creates the overdamped Brownian oscillator component
        of the correlation function
        
        """
        
        temperature = params["T"]
        ctime = params["cortime"]
        lamb = params["reorg"]

        if "matsubara" in params.keys():
            nmatsu = params["matsubara"]
        else:
            nmatsu = 10

        kBT = kB_int*temperature
        time = self.axis.data

        #if values is not None:
        #    cfce = values
            
        #else:
        if True:
            
            cfce = (lamb/(ctime*numpy.tan(1.0/(2.0*kBT*ctime))))\
                *numpy.exp(-time/ctime) \
                - 1.0j*(lamb/ctime)*numpy.exp(-time/ctime)

            cfce += (4.0*lamb*kBT/ctime) \
                *self._matsubara(kBT, ctime, nmatsu)
        
        # this is a call to the function inherited from DFunction class 
        self._add_me(self.axis, cfce)
        
        # update reorganization energy
        self.lamb += lamb
        # check temperature and update cutoff time
        self._set_temperature_and_cutoff_time(temperature, 5.0*ctime)      



    def _make_overdamped_brownian_ht(self, params): # , values=None):
        """Creates the high temperature overdamped Brownian oscillator 
        component of the correlation function
        
        """
        temperature = params["T"]
        ctime = params["cortime"]
        lamb = params["reorg"]
        
        kBT = kB_int*temperature
        time = self.axis.data

        #if values is not None:
        #    cfce = values
        #    
        #else:
        if True:
            cfce = 2.0*lamb*kBT*(numpy.exp(-time/ctime)
                                 - 1.0j*(lamb/ctime)*numpy.exp(-time/ctime))

        # this is a call to the function inherited from DFunction class 
        self._add_me(self.axis, cfce)

        # update reorganization energy
        self.lamb += lamb
        
        # check temperature and update cutoff time
        self._set_temperature_and_cutoff_time(temperature, 5.0*ctime)  
        

    def _make_underdamped_brownian(self, params): #, values=None):
        """Creates underdamped Brownian oscillator component of the correlation
        function
        
        
        """
        from .spectraldensity import SpectralDensity
        
        temperature = params["T"]
        ctime = params["gamma"]
        #omega = params["freq"]
        lamb = params["reorg"]
        
        #kBT = kB_int*temperature
        time = self.axis #.data

        #if values is not None:
        #    cfce = values
        #    
        #else:
        if True:
            with frequency_units("int"):
                # Make it via SpectralDensity
                fa = SpectralDensity(time, params)
            
                cf = fa.get_CorrelationFunction(temperature=temperature)
            
                cfce = cf.data
                   #2.0*lamb*kBT*(numpy.exp(-time/ctime)
                   #              - 1.0j*(lamb/ctime)*numpy.exp(-time/ctime))

        # this is a call to the function inherited from DFunction class 
        self._add_me(self.axis, cfce)

        # update reorganization energy
        self.lamb += lamb
        
        # check temperature and update cutoff time
        self._set_temperature_and_cutoff_time(temperature, 5.0/ctime)  
        
        
    def _make_value_defined(self, params, values):
        
        temperature = params["T"]
        
        if "cutoff-time" in params.keys():
            ctime = params["cutoff-time"]
        else:
            ctime = self.axis.max

        if values is not None:
            if len(values) == self.axis.length:
                cfce = values
            else:
                raise Exception("Incompatible values")
        else:
            raise Exception("Valued-defined correlation function without values")
 
        try:
            if self.params["reorg"] is None:
                lamb = __measure_reorganization_energy(self.axis,cfce)
            else:
                lamb = params["reorg"]
        except:
            lamb = __measure_reorganization_energy(self.axis,cfce)
    
        # this is a call to the function inherited from DFunction class 
        self._add_me(self.axis, cfce)

        # update reorganization energy
        self.lamb += lamb
        
        # check temperature and update cutoff time
        self._set_temperature_and_cutoff_time(temperature, ctime)  
           
    #
    # Aritmetic operations
    #
    
    def __add__(self, other):
        """Addition of two correlation functions
        
        """
        t1 = self.axis
        t2 = other.axis
        if t1 == t2:
        
            # FIXME: It seems to be defined only for predefined correlation functions and not for value defiend
            # Why not to use deepcopy for creating f and then add two functions
            f = CorrelationFunction(t1, params=self.params)
            f.add_to_data(other)
            
        else:
            raise Exception("In addition, functions have to share"
                            +" the same TimeAxis object")
            
        return f
    
    def __iadd__(self, other):
        """Inplace addition of two correlation functions
        
        """  
        self.add_to_data2(other)
        return self
    
            
    def add_to_data(self, other):
        """Addition of data from a specified CorrelationFunction to this object
        
        """
        t1 = self.axis
        t2 = other.axis
        if t1 == t2:
            
            self.data += other.data
            self.lamb += other.lamb  # reorganization energy is additive
            if other.cutoff_time > self.cutoff_time: 
                self.cutoff_time = other.cutoff_time  
                
            if self.temperature != other.temperature:
                raise Exception("Cannot add two correlation functions on different temperatures")
    
            for p in other.params:
                self.params.append(p)
                
            self._is_composed = True
            self._is_empty = False
            

        else:
            raise Exception("In addition, functions have to share"
                            +" the same TimeAxis object")
 
    def add_to_data2(self, other):
        """Addition of data from a specified CorrelationFunction to this object
        
        """
        if self == other:
            ocor = CorrelationFunction(other.axis,other.params)
        else:
            ocor = other
            
        t1 = self.axis
        t2 = ocor.axis
        if t1 == t2:
            
            self.data += ocor.data
            self.lamb += ocor.lamb  # reorganization energy is additive
            if ocor.cutoff_time > self.cutoff_time: 
                self.cutoff_time = ocor.cutoff_time  
                
            if self.temperature != ocor.temperature:
                raise Exception("Cannot add two correlation functions on different temperatures")
    

            for p in ocor.params:
                self.params.append(p)
            
            self._is_composed = True
            self._is_empty = False
            

        else:
            raise Exception("In addition, functions have to share"
                            +" the same TimeAxis object")
            
       
    def reorganization_energy_consistent(self, rtol=1.0e-3):
        """Checks if the reorganization energy is consistent with the data
        
        Calculates reorganization energy from the data and checks if it
        is within specified tolerance from the expected value
        """
        
        lamb1 = self.measure_reorganization_energy()
        lamb2 = self.convert_energy_2_current_u(self.lamb)
        if (abs(lamb1 - lamb2)/(lamb1+lamb2)) < rtol:
            return True
        else:
            return False

    def is_analytical(self):
        """Returns `True` if analytical
        Returns `True` if the CorrelationFunction object is constructed
        by analytical formula. Returns `False` if the object was constructed
        by numerical transformation from spectral density.
        """

        return bool(self.params["ftype"] in self.analytical_types)


    def get_temperature(self):
        """Returns the temperature of the correlation function
        """
        return self.temperature

    def get_reorganization_energy(self):
        """Returns the reorganization energy of the correlation function
        """
        return self.convert_energy_2_current_u(self.lamb)

    def measure_reorganization_energy(self):
        """Calculates the reorganization energy of the correlation function
        Calculates the reorganization energy of the correlation function by
        integrating its imaginary part.
        """
        
        # My definition
        if 1:
            with frequency_units("int"):
                # This alternative converges faster with length of time axis 
                # than the alternative through fourier transform
                lamb = -numpy.imag(integrate.simps(self.data,self.axis.data))
                
                #comega = self.get_Fourier_transform()
                #comega.data = comega.data/comega.axis.data
                #lamb = integrate.simps(comega.data,comega.axis.data)/(2*numpy.pi)
                
                #oftcf = self.get_OddFTCorrelationFunction()
                #data  = oftcf.data/oftcf.axis.data
                #primitive = c2h(oftcf.axis,data)
                #lamb = numpy.real(primitive[-1])/(2*numpy.pi)
                #lamb = integrate.simps(data,oftcf.axis.data)/(2*numpy.pi)
        
        # Tomas's definition - converges faster with length of timeaxis
        else:
            primitive = c2h(self.axis, self.data)
            lamb = -numpy.imag(primitive[-1])
        return self.convert_energy_2_current_u(lamb)


    def copy(self):
        """Creates a copy of the current correlation function
        """
        #with frequency_units(self.frequency_units):
        cfce = CorrelationFunction(self.axis, self.params)
        return cfce


    def get_SpectralDensity(self):
        """ Returns a corresponding SpectralDensity object
        Returns a SpectralDensity corresponding to this CorrelationFunction
        """

        from .spectraldensities import SpectralDensity

        # protect this from external units
        with frequency_units("int"):
            frequencies = self.axis.get_FrequencyAxis()
            vals = self.get_OddFTCorrelationFunction().data

            # FIXME: how to set the limit of SpectralDensity at w->0
            spectd = SpectralDensity(frequencies, self.params, values=vals)

        return spectd


    def get_FTCorrelationFunction(self):
        """Returns a Fourier transform of the correlation function
        Returns a Fourier transform of the correlation function in form
        of an instance of a special class ``FTCorrelationFunction``
        """
        with frequency_units("int"):
            ftcf = FTCorrelationFunction(self.axis, self.params)
        return ftcf

    def get_OddFTCorrelationFunction(self):
        """Returns a odd part of the Fourier transform of correlation function
        Returns the odd part of a Fourier transform of the correlation
        function in form of an instance of a special class
        ``OddFTCorrelationFunction``
        """

        with frequency_units("int"):
            # FIXME
            # self._is_composed
            if self.params[0]["ftype"] == "Value-defined":
                if self.data is None:
                    raise Exception()
                else:
                    oftcf = OddFTCorrelationFunction(self.axis, self.params, values=self.data)
            else:
                oftcf = OddFTCorrelationFunction(self.axis, self.params)
        return oftcf

    def get_EvenFTCorrelationFunction(self):
        """Returns a even part of the Fourier transform of correlation function
        Returns the even part of a Fourier transform of the correlation
        function in form of an instance of a special class
        ``EvenFTCorrelationFunction``
        """
        with frequency_units("int"):
            if self.params["ftype"] == "Value-defined":
                if self.data is None:
                    raise Exception()
                else:
                    eftcf = EvenFTCorrelationFunction(self.axis, self.params, values=self.data)
            else:
                eftcf = EvenFTCorrelationFunction(self.axis, self.params)
        return eftcf


class FTCorrelationFunction(DFunction, UnitsManaged):
    """Fourier transform of the correlation function
    Numerically calculated Fourier transform of the correlation function
    Parameters
    ----------
    axis: TimeAxis
        Time interval from which the frequency interval is calculated
    params: dictionary
        Dictionary of the correlation function parameters
    """
    energy_params = ("reorg", "omega", "freq")
    
    def __init__(self, axis, params, values=None):
        super().__init__()

        if not isinstance(axis, FrequencyAxis):
            with frequency_units["int"]:
                faxis = axis.get_FrequencyAxis()  
            self.axis = faxis
        else:
            self.axis = axis
        
        # handle params
        self.params = []  # this will always be a list of components
        p2calc = []
        try:
            # if this passes, we assume params is a dictionary
            params.keys()
            self._is_composed = False
            p2calc.append(params)
            
        except:
            # othewise we assume it is a list of dictionaries 
            self._is_composed = True
            for p in params:
                p2calc.append(p)
                
        for params in p2calc:
            ftype = params["ftype"]
            
            if ftype not in CorrelationFunction.allowed_types:
                raise Exception("Unknown CorrelationFunction type")

            # Parameters do not have to be transformed to internal units - this is done automaticaly when correlation function is created
            self.params.append(params)
            
            # We create CorrelationFunction and FTT it
            if params["ftype"] == "Value-defined":
                if values is None:
                    raise Exception()
                else:
                    cfce = CorrelationFunction(axis, params, values=values)
            elif values is not None:
                #FIXME BEcause of spectral density when this function is called for owerdampened Brownian 
                self.data = values
                return
                #cfce = CorrelationFunction(axis, params, values=values)
            else:
                cfce = CorrelationFunction(axis, params)

            # data have to be protected from change of units
            with frequency_units("int"):
                ftvals = cfce.get_Fourier_transform()
                ndata = ftvals.data

                self._add_me(self.axis,ndata)



#            # we mutate the parameters into internal units
#            prms = {}
#            for key in params.keys():
#                if key in self.energy_params:
#                    prms[key] = self.convert_energy_2_internal_u(params[key])
#                else:
#                    prms[key] = params[key]
#
#        self.params = prms
#
#        if values is None:
#            # data have to be protected from change of units
#            with frequency_units("int"):
#                cfce = CorrelationFunction(axis, self.params)
#                ftvals = cfce.get_Fourier_transform()   # porperty of DFunction
#                self.data = ftvals.data
#            #self.axis = ftvals.axis
#        else:
#            # FIXME: if axis is time axis then assume that values are in time domain - construct correlation function and then fourier transform
#            # FIXME: if axis is frequency axis assume that values are in frequency doman and self.data = values
#            # This is not protected from change of units!!!!
#            self.data = values
#            #self.axis = cfce.axis.get_FrequencyAxis()




class OddFTCorrelationFunction(DFunction, UnitsManaged):
    """Odd part of the Fourier transform of the correlation function
    Numerically calculated odd part Fourier transform of the correlation
    function. Calculated as  Fourier transform of the imaginary part of the
    correlation function.
    Parameters
    ----------
    axis: TimeAxis
        Time interval from which the frequency interval is calculated
    params: dictionary
        Dictionary of the correlation function parameter
    Examples
    --------
    >>> ta = TimeAxis(0.0,1000,1.0)
    >>> params = dict(ftype="OverdampedBrownian",reorg=20,cortime=100,T=300)
    >>> with frequency_units("1/cm"):
    ...    ocf = OddFTCorrelationFunction(ta,params)
    ...    print(numpy.allclose(ocf.at(-100), -ocf.at(100)))
    True
    """

    def __init__(self, axis, params, values=None):
        super().__init__()
        
        if not isinstance(axis, FrequencyAxis):
            with frequency_units["int"]:
                faxis = axis.get_FrequencyAxis()
            self.axis = faxis
        else:
            self.axis = axis     

        # handle params
        self.params = []  # this will always be a list of components
        p2calc = []
        try:
            # if this passes, we assume params is a dictionary
            params.keys()
            self._is_composed = False
            p2calc.append(params)
            
        except:
            # othewise we assume it is a list of dictionaries 
            self._is_composed = True
            for p in params:
                p2calc.append(p)
        
        for params in p2calc:
        
            ftype = params["ftype"]
            if ftype not in CorrelationFunction.allowed_types:
                raise Exception("Unknown Correlation Function Type")
    
            self.params.append(params)
                
            # We create CorrelationFunction and FTT it
            if params["ftype"] == "Value-defined":
                if values is None:
                    raise Exception()
                else:
                    cfce = CorrelationFunction(axis, params, values=values)
            else:
                cfce = CorrelationFunction(axis, params)
    
            cfce.data = 1j*numpy.imag(cfce.data)
    
            # data have to be protected from change of units
            with frequency_units("int"):
                ftvals = cfce.get_Fourier_transform()
                ndata = numpy.real(ftvals.data)

            self._add_me(self.axis,ndata)


class EvenFTCorrelationFunction(DFunction, UnitsManaged):
    """Even part of the Fourier transform of the correlation function
    Numerically calculated even part Fourier transform of the correlation
    function. Calculated as  Fourier transform of the real part of the
    correlation function.
    Parameters
    ----------
    axis: TimeAxis
        Time interval from which the frequency interval is calculated
    params: dictionary
        Dictionary of the correlation function parameter
    Examples
    --------
    >>> ta = TimeAxis(0.0,1000,1.0)
    >>> params = dict(ftype="OverdampedBrownian",reorg=20,cortime=100,T=300)
    >>> with frequency_units("1/cm"):
    ...    ecf = EvenFTCorrelationFunction(ta,params)
    ...    print(numpy.allclose(ecf.at(-100), ecf.at(100)))
    True
    """

    def __init__(self, axis, params, values=None):
        super().__init__()

        if not isinstance(axis, FrequencyAxis):
            with frequency_units["int"]:
                faxis = axis.get_FrequencyAxis()
            self.axis = faxis
        else:
            self.axis = axis 
            
        # handle params
        self.params = []  # this will always be a list of components
        p2calc = []
        try:
            # if this passes, we assume params is a dictionary
            params.keys()
            self._is_composed = False
            p2calc.append(params)
            
        except:
            # othewise we assume it is a list of dictionaries 
            self._is_composed = True
            for p in params:
                p2calc.append(p)
        
        for params in p2calc:
            
            ftype = params["ftype"]
            if ftype not in CorrelationFunction.allowed_types:
                raise Exception("Unknown Correlation Function Type: "+ftype)


            self.params.append(params)
            
            # We create CorrelationFunction and FTT it
            if params["ftype"] == "Value-defined":
                if values is None:
                    raise Exception()
                else:
                    # if axis is FreuqencyAxis values are assumed to be in frequency domain 
                    cfce = CorrelationFunction(axis, params, values=values)
            else:
                cfce = CorrelationFunction(axis, params)
                
            cfce.data = numpy.real(cfce.data)
    
            # data have to be protected from change of units
            with frequency_units("int"):
                ftvals = cfce.get_Fourier_transform()
                ndata = numpy.real(ftvals.data)

            self._add_me(self.axis,ndata)



def __measure_reorganization_energy(axis,values):
    if isinstance(axis, TimeAxis):
        lamb = -numpy.imag(integrate.simps(values,axis.data))
    else:
        lamb = 0.0
    return lamb

#FIXME: these functions can go to DFunction          
def c2g(timeaxis, coft):
    """ Converts correlation function to lineshape function
    Explicit numerical double integration of the correlation
    function to form a lineshape function.
    Parameters
    ----------
    timeaxis : cu.oqs.time.TimeAxis
        TimeAxis of the correlation function
    coft : complex numpy array
        Values of correlation function given at points specified
        in the TimeAxis object
    """

    time = timeaxis
    preal = numpy.real(coft)
    pimag = numpy.imag(coft)
    splr = interp.UnivariateSpline(time.data,
                                   preal, s=0).antiderivative()(time.data)
    splr = interp.UnivariateSpline(time.data,
                                   splr, s=0).antiderivative()(time.data)
    spli = interp.UnivariateSpline(time.data,
                                   pimag, s=0).antiderivative()(time.data)
    spli = interp.UnivariateSpline(time.data,
                                   spli, s=0).antiderivative()(time.data)
    goft = splr + 1j*spli
    return goft


def c2h(timeaxis, coft):
    """ Integrates correlation function in time with an open upper limit
    Explicit numerical integration of the correlation
    function to form a precursor to the lineshape function.
    Parameters
    ----------
    timeaxis : TimeAxis
        TimeAxis of the correlation function
    coft : complex numpy array
        Values of correlation function given at points specified
        in the TimeAxis object
    """

    time = timeaxis
    preal = numpy.real(coft)
    pimag = numpy.imag(coft)
    splr = interp.UnivariateSpline(time.data,
                                   preal, s=0).antiderivative()(time.data)
    spli = interp.UnivariateSpline(time.data,
                                   pimag, s=0).antiderivative()(time.data)
    hoft = splr + 1j*spli

    return hoft

def h2g(timeaxis, coft):
    """ Integrates and integrated correlation function
    Explicit numerical integration of the correlation
    function to form a precursor to the lineshape function.
    Parameters
    ----------
    timeaxis : TimeAxis
        TimeAxis of the correlation function
    coft : complex numpy array
        Values of correlation function given at points specified
        in the TimeAxis object
    """
    return c2h(timeaxis, coft)