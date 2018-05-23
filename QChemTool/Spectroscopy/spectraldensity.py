# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:05:11 2018

@author: Vladislav Sláma
"""

# -*- coding: utf-8 -*-
"""
    Quantarhei package (http://www.github.com/quantarhei)
    spectraldensities module
"""
import numpy

from ..General.dfunction import DFunction
from ..General.UnitsManager import UnitsManaged
from ..General.UnitsManager import frequency_units
from ..General.timeaxis import TimeAxis
from ..General.frequencyaxis import FrequencyAxis
from .correlationfunction import CorrelationFunction
from .correlationfunction import FTCorrelationFunction
from ..General.units import kB_int_freq as kB_int

#from .correlationfunctions import c2h

# Unable to add two value defined functions
# line 156 : calculate reorganization energy numericaly
# line 295 : calculate reorganization energy numericaly

class SpectralDensity(DFunction, UnitsManaged):
    """This class represents the so-called spectral density. ``C''(w)``
    
    Parameters
    ----------
    axis : TimeAxis, FrequencyAxis
        ValueAxis object specifying the frequency range directly or through
        Fourier transform frequencies corresponding to a TimeAxis
    params : dictionary
        Parameters of the spectral density
    values : numpy.array (optional)
        Values of spectral density for value defined function. When time TimeAxis
        is used as axis fourier transform of values are used for construction
        of spectral density
        
    Methods
    -------
    is_analytical()
        Returns `True` if the spectral density is calculated from an analytical
        formula, `False` otherwise.
    copy()
        Makes a copy of the SpectralDensity object
    get_temperature()
        Returns temperature assigned to the spectral density or raises an
        exception if it was not set
    get_reorganization_energy()
        Returns the reorganization energy parameters of
        the spectral density
    measure_reorganization_energy()
        Calculates reorganization energy from the shape of the spectral
        density
    get_CorrelationFunction()
        Returns numerically calculated correlation function, based on the
        spectral density
    get_FTCorrelationFunction()
        Returns a numerically calculated Fourier transform of the correlation
        function
        
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
    
    NMA:
        sigma = params["sheer"] \n
        gs = params["coupling"] \n
        omegas = params["freq"] \n
        
        ``C''(w) = ( 1/(sigma*sqrt(pi)) )*pi*(w**2)*Sum_{xi}{ gs[xi]* 
        [exp( -(w-omegas[xi])**2/a^2 ) - exp( -(w+omegas[xi])**2/a^2 ) }`` \n
    
    
    where ``C(w) = [1+coth( w*hbar/(2*kB*T) )]*C''(w)`` and 
    ``C(t) = C'(t) + iC''(t)`` \n
    For fourier transform componet: \n
    ``C''(w) = int{t,-infty,infty}{exp(i*w*t )*i*C''(t)} = -2 * int{t,0,infty}{sin( w*t )*C''(t)}``
        
    Examples
    --------
    `SpectralDensity` object can be ctreated with the same parameters as
    `CorrelationFunction`. The temperature can be set, but it is not
    a compulsory parameter.
    >>> from quantarhei import TimeAxis
    >>> params = dict(ftype="OverdampedBrownian", cortime=100, reorg=20, T=300)
    >>> time = TimeAxis(0.0,1000,1.0)
    >>> with frequency_units("1/cm"):\
           sd = SpectralDensity(time, params)
    >>> cf = sd.get_CorrelationFunction()
    >>> print(sd.get_temperature())
    300
    If we create the same without temperature
    >>> parwoT = dict(ftype="OverdampedBrownian", cortime=100, reorg=20)
    >>> with frequency_units("1/cm"):\
           sdwoT = SpectralDensity(time, parwoT)
    everything is alright. `CorrelationFunction`, however, cannot be created
    as above, because temperature must be known. Attempt to create `CorrelationFunction`
    as above would lead to an exception. We have to specify temperature as
    a parameter to the `get_CorrelationFunction` method.
    >>> cf = sdwoT.get_CorrelationFunction(temperature=300)
    Reorganization of the spectral density is an input parameter which
    can be obtained by calling the corresponding method
    >>> with frequency_units("1/cm"):
    ...     print(sdwoT.get_reorganization_energy())
    20.0
    At the same time, reorganization energy can be calculated from the
    shape of the spectral density by integrating over it. The accuracy
    of such estimation depends on numerics, hence the relative tolerance of
    only 1.0e-2 below
    >>> lamb_definition = sd.get_reorganization_energy()
    >>> lamb_measured = sd.measure_reorganization_energy()
    >>> print(numpy.allclose(lamb_definition, lamb_measured, rtol=1.0e-2))
    True
    """

    energy_params = ("reorg", "omega", "freq")
    analytical_types = ("OverdampedBrownian")

    def __init__(self, axis=None, params=None, values=None):
        super().__init__()

        if (axis is not None) and (params is not None):
            
            if isinstance(axis, TimeAxis):
                # protect the frequency axis creation from units management
                with frequency_units("int"):
                    faxis = axis.get_FrequencyAxis()
                    
                    if values is not None:
                        ft  = DFunction(axis,values)
                        fw = ft.get_Fourier_transform()
                        vals = fw.data
                    else:
                        vals = values

                self.axis = faxis
            else:
                self.axis = axis
                vals = values
                
    
            self.lim_omega = numpy.zeros(2)
            
#            if values is not None:
#                self.params = params
#                self.data = vals
#                self.lamb = 0.0
#                for p in self.params:
#                    self.lamb += p["reorg"]
#                    # FIXME: If reorganizationn energy is not specified - calculate numerically
#                    # maybe create correlation function and then numericaly integrate (more stable then direct integration of spectral density)
#                return
    
    
            self._splines_initialized = False
    
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
            #self.cutoff_time = 0.0
            
            #
            # loop over parameter sets
            #
            for params in p2calc:
    
                try:
                    ftype = params["ftype"]
                    if ftype not in CorrelationFunction.allowed_types:
                        raise Exception("Unknown Correlation Function Type")
    
                    # we mutate the parameters into internal units
                    prms = {}
                    for key in params.keys():
                        if key in self.energy_params:
                            prms[key] = \
                            self.convert_energy_2_internal_u(params[key])
                        else:
                            prms[key] = params[key]
                        
                except:
                    raise Exception
        
                # Now prms are in internal units
        
                if "T" in params.keys():
                    self.temperature = params["T"]
        
                if ftype == "OverdampedBrownian":
        
                    self._make_overdamped_brownian(prms, vals)
        
                elif ftype == "UnderdampedBrownian":
        
                    self._make_underdamped_brownian(prms, vals)
                    
                elif ftype == "Value-defined":
        
                    self._make_value_defined(prms, values=vals)
        
                else:
                    raise Exception("Unknown correlation function type or"+
                                    " type domain combination.")
    
                self.params.append(prms)

    def _make_overdamped_brownian(self, params, values=None):
        """ Sets the Overdamped Brownian oscillator spectral density
        """
        ctime = params["cortime"]
        lamb = params["reorg"]

        # protect calculation from units management
        with frequency_units("int"):
            omega = self.axis.data
            cfce = (2.0*lamb/ctime)*omega/(omega**2 + (1.0/ctime)**2)

        if values is not None:
            self._add_me(self.axis, values)
        else:
            self._add_me(self.axis, cfce)

        # this is in internal units
        self.lamb += lamb

        lim_omega = numpy.zeros(2)
        lim_omega[0] = 0.0
        lim_omega[1] = ctime
        for i in range(2):
            self.lim_omega[i] += lim_omega[i]

    def _make_underdamped_brownian(self, params, values=None):
         
        #temperature = params["T"]
        ctime = params["gamma"]
        # use the units in which params was defined
        omega0 = params["freq"]
        lamb = params["reorg"]
        
        # protect calculation from units management
        with frequency_units("int"):
            omega = self.axis.data
            #cfce = (lamb*ctime)*omega/((omega-omega0)**2 + (ctime)**2) \
            #      +(lamb*ctime)*omega/((omega+omega0)**2 + (ctime)**2)
            cfce = (2.0*lamb*ctime)*(omega0**2)*\
                  (omega/(((omega**2)-(omega0**2))**2 + (omega**2)*(ctime**2))) 
                  #+omega/((omega**2+omega0**2)**2 + (omega**2)*(ctime)**2))


        if values is not None:
            self._add_me(self.axis, values)
        else:
            self._add_me(self.axis, cfce)

        # this is in internal units
        self.lamb += lamb     
        
        lim_omega = numpy.zeros(2)
        lim_omega[0] = 0.0
        lim_omega[1] = 0.0
        for i in range(2):
            self.lim_omega[i] += lim_omega[i]
    
    def _make_discrete_modes(self, params, values=None):
        """ Spectral density for set of discrete modes
        """
        
        # FIXME transform sigma and omegas to frequency internal units 
        sigma = params["sheer"]
        gs = params["coupling"] # unit less
        omegas = params["freq"]
        
        # protect calculation from units management
        with frequency_units("int"):
            omega = self.axis.data
            cfce  = numpy.zeros(self.axis.length,dtype='f8')
            for xi in range(len(omegas)):
                cfce += gs[xi]*( 
                numpy.exp( -(omega-omegas[xi])*(omega-omegas[xi])/sigma**2 )
                -numpy.exp( -(omega+omegas[xi])*(omega-omegas[xi])/sigma**2 ) 
                )
            
            # FIXME for large omegas might be numerically unstable
            cfce = cfce*numpy.sqrt(numpy.pi)*omega*omega / sigma 
            
        self._add_me(self.axis, cfce)
        
        self.lamb += numpy.sum(omegas*gs*gs)

        lim_omega = numpy.zeros(2)
        lim_omega[0] = 0.0
        lim_omega[1] = 0.0
        for i in range(2):
            self.lim_omega[i] += lim_omega[i]
            
    def _make_value_defined(self, values=None):
        """ Value defined spectral density
        """
        if values is None:
            raise Exception()
            
        self._add_me(self.axis, values)
        try:
            if self.params["reorg"] is None:
                self.lamb += __measure_reorganization_energy(self.axis, values)
            else:
                self.lamb += self.params["reorg"]
        except:
            self.lamb += __measure_reorganization_energy(self.axis, values)

#        self.lamb += self.params["reorg"]

        lim_omega = numpy.zeros(2)
        lim_omega[0] = 0.0
        lim_omega[1] = 0.0
        for i in range(2):
            self.lim_omega[i] += lim_omega[i]

    #
    # Aritmetic operations
    #
    
    def __add__(self, other):
        """Addition of two spectral densities
        
        """
        t1 = self.axis
        t2 = other.axis
        if t1 == t2:
                      
            f = SpectralDensity(t1, params=self.params, values=self.data)
            f.add_to_data(other)
            
        else:
            raise Exception("In addition, functions have to share"
                            +" the same FrequencyAxis object")
            
        return f
    
    def __iadd__(self, other):
        """Inplace addition of two spectral densities
        
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
            for i in range(2):
                self.lim_omega[i] += other.lim_omega[i] 
                
            # cutoff time is take as the longer one of the two
            #self.cutoff_time = max(self.cutoff_time, other.cutoff_time)
            

            for p in other.params:
                self.params.append(p)            
            
            self._is_composed = True
            self._is_empty = False
           

        else:
            raise Exception("In the operation of addition, functions "
                           +"have to share the same FrequencyAxis object")


    def add_to_data2(self, other):
        """Addition of data from a specified SpectralDensity to this object
        
        """
        if self == other:
            ocor = SpectralDensity(other.axis, other.params, values=other.data)
        else:
            ocor = other
            
        t1 = self.axis
        t2 = ocor.axis
        if t1 == t2:
            
            self.data += ocor.data
            self.lamb += ocor.lamb  # reorganization energy is additive
            #if ocor.cutoff_time > self.cutoff_time: 
            #    self.cutoff_time = ocor.cutoff_time  
                
            #if self.temperature != ocor.temperature:
            #    raise Exception("Cannot add two correlation functions "
            #                   +"on different temperatures")
    

            for p in ocor.params:
                self.params.append(p)
            
            self._is_composed = True
            self._is_empty = False
            

        else:
            raise Exception("In the operation of addition, functions "
                           +"have to share the same FrequencyAxis object")
            
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
        if self.temperature > 0.0:
            return self.temperature
        else:
            raise Exception("SpectralDensity was not assigned temperature")

    def get_reorganization_energy(self):
        """Returns the reorganization energy of the cspectral density
        """
        return self.convert_energy_2_current_u(self.lamb)

    def measure_reorganization_energy(self):
        """Calculates the reorganization energy of the spectral density
        Calculates the reorganization energy of the spectral density by
        integrating over frequency.
        """
        import scipy.interpolate as interp

        integr = self.data/self.axis.data
        uvspl = interp.UnivariateSpline(self.axis.data, integr, s=0)
        integ = uvspl.integral(0.0, self.axis.max)/numpy.pi

        return integ


    def copy(self):
        """Creates a copy of the current correlation function
        """
        return SpectralDensity(self.axis, self.params)


    def get_CorrelationFunction(self, temperature=None):
        """Returns correlation function corresponding to the spectral density
        """

        params = []
        for pdict in self.params:
            newdict = pdict.copy()
            if temperature is not None:
                newdict["T"] = temperature
            T = newdict["T"]
            params.append(newdict)

        time = self.axis.get_TimeAxis()

        # everything has to be protected from change of units
        with frequency_units("int"):
            ftcf = self.get_FTCorrelationFunction(temperature=T)
            cftd = ftcf.get_inverse_Fourier_transform()
            cfce = CorrelationFunction(time, params, values=cftd.data)
        return cfce


    def get_FTCorrelationFunction(self, temperature=None):
        """Returns Fourier transformed correlation function
        Fourier transformed correlation function is calculated from the
        analytical formula connecting spectral density and FT correlation
        function.
        
        Parameters
        ----------
        temperature : optional
            Temperature which can be missing among the spectral density
            parameters
        """
        
        #
        # copy the parameters and change temperature if needed
        #
        k = 0
        newpars = []
        for prms in self.params:
            
            #params = self.params.copy()
            if temperature is not None:
                prms["T"] = temperature
    
            # FIXME: check that all temperatures are the same
            if k == 0:
                temp = prms["T"]
            elif temp != prms["T"]:
                raise Exception("Temperature of all components has to be the same")
            k += 1
            
            newpars.append(prms)
    
        ind_of_zero, diff = self.axis.locate(0.0)
        atol = 1.0e-7
        twokbt = 2.0*kB_int*temp

        #
        # Numerical evaluation is done on the whole data
        #
        #if True:
        with frequency_units("int"):
            # if zero is sufficiently away from any point that is evaluated
            if numpy.abs(diff) > atol:
                # do the evaluation directly
                vals = (1.0 +
                        (1.0/numpy.tanh(self.axis.data/twokbt)))*self.data
            # otherwise
            else:
                data = self.data
                vals = numpy.zeros(self.data.shape)
                # evaluate everything before zero
                omega = self.axis.data[0:ind_of_zero]
                spect = data[0:ind_of_zero]
                auxi = (1.0 + (1.0/numpy.tanh(omega/twokbt)))*spect
                vals[0:ind_of_zero] = auxi
                # then after zero
                omega = self.axis.data[ind_of_zero+1:self.axis.length]
                spect = data[ind_of_zero+1:self.axis.length]
                auxi = (1.0 + (1.0/numpy.tanh(omega/twokbt)))*spect
                vals[ind_of_zero+1:self.axis.length] = auxi
                
                # and used L'Hospital ruĺe to calculate the limit at zero 
                vals[ind_of_zero] = twokbt*(data[ind_of_zero+1]
                    -data[ind_of_zero-1])/(2.0*self.axis.step)

            ftc = FTCorrelationFunction(self.axis, newpars, values=vals)

        return ftc
    
def __measure_reorganization_energy(axis,values):
    """Calculates the reorganization energy of the spectral density
    Calculates the reorganization energy of the spectral density by
    integrating over frequency.
    """
    import scipy.interpolate as interp
    
    if isinstance(axis, FrequencyAxis):
        integr = values/axis.data
        uvspl = interp.UnivariateSpline(axis.data, integr, s=0)
        integ = uvspl.integral(0.0, axis.max)/numpy.pi
    else:
        integ = 0.0
    return integ