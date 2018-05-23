# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 18:02:51 2018

@author: Vladislav Sláma
"""
import numpy

from .valueaxis import ValueAxis
from .UnitsManager import frequency_units
from .UnitsManager import FrequencyUnitsManaged

from .types import UnitsManagedArray
from .types import UnitsManaged


class FrequencyAxis(ValueAxis, FrequencyUnitsManaged):
    """ Class representing frequency axis of calculations
    
    Parameters
    ----------
    start : float
        start of the FrequencyAxis
    length : int
        number of steps
    step : float
        time step
    atype : string {"complete","upper-half"}
        Axis type
    
    Attributes
    ----------
    data : float array
        Holds the values of frequency
    
    Examples
    --------
    The default type of the `FrequencyAxis` is `complete`. See the discussion
    of the types in the `TimeAxis` documentation.
    >>> wa = FrequencyAxis(0.0,100,0.05)
    >>> ta = wa.get_TimeAxis()
    >>> print(ta.length)
    100
    
    The type `upper-half` only refers to the corresponding TimeAxix. Everything
    about the FrequencyAxis remains the same as with `complete`.
    >>> wa = FrequencyAxis(0.0, 100, 0.05, atype = "upper-half")
    >>> ta = wa.get_TimeAxis()
    >>> print(ta.length)
    50
    #>>> print(ta.step,2.0*numpy.pi/(100*wa.step))
    >>> print(numpy.allclose(ta.step,2.0*numpy.pi/(100*wa.step)))
    True
    
    For `complete`, everything should work also for an odd number of points
    >>> wa = FrequencyAxis(0.0,99,0.05)
    >>> ta = wa.get_TimeAxis()
    >>> print(ta.length)
    99
    
    But `upper-half` throughs an exception, because by definition its number of
    points is `2*N`, where `N` is an integer.
    >>> wa = FrequencyAxis(0.0,99,0.05,atype="upper-half")
    >>> ta = wa.get_TimeAxis()
    Traceback (most recent call last):
    ...
    Exception: Cannot create upper-half TimeAxis from an odd number of points
    
    Relation between TimeAxis and FrequencyAxis
    -------------------------------------------
    Complete FrequencyAxis and even number of points
    >>> wa = FrequencyAxis(0.0,10,0.1,atype="complete")
    >>> frequencies = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    >>> print(numpy.allclose(wa.data,frequencies))
    True
    >>> ta = wa.get_TimeAxis()
    >>> times = 2.0*numpy.pi*numpy.fft.fftshift(numpy.fft.fftfreq(10,0.1))
    >>> print(numpy.allclose(ta.data,times))
    True
    >>> print(numpy.allclose(ta.step,times[1]-times[0]))
    True
    >>> wb = ta.get_FrequencyAxis()
    >>> print(numpy.allclose(wb.data,frequencies))
    True
    >>> tb = wb.get_TimeAxis()
    >>> print(numpy.allclose(tb.data,times))
    True
    Complete FrequencyAxis and odd number of points
    >>> wa = FrequencyAxis(0.0,11,0.1,atype="complete")
    >>> frequencies = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    >>> print(numpy.allclose(wa.data,frequencies))
    True
    >>> ta = wa.get_TimeAxis()
    >>> times = 2.0*numpy.pi*numpy.fft.fftshift(numpy.fft.fftfreq(11,0.1))
    >>> print(numpy.allclose(ta.data,times))
    True
    >>> print(numpy.allclose(ta.step,times[1]-times[0]))
    True
    >>> wb = ta.get_FrequencyAxis()
    >>> print(numpy.allclose(wb.data,frequencies))
    True
    >>> tb = wb.get_TimeAxis()
    >>> print(numpy.allclose(tb.data,times))
    True
    Upper-half FrequencyAxis and even number of points
    >>> wa = FrequencyAxis(0.0,10,0.1,atype="upper-half")
    >>> frequencies = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    >>> print(numpy.allclose(wa.data,frequencies))
    True
    >>> ta = wa.get_TimeAxis()
    >>> times = 2.0*numpy.pi*numpy.fft.fftshift(numpy.fft.fftfreq(10,0.1))
    >>> print(numpy.allclose(ta.data,times[5:10]))
    True
    >>> print(numpy.allclose(ta.step,times[1]-times[0]))
    True
    >>> wb = ta.get_FrequencyAxis()
    >>> print(numpy.allclose(wb.data,frequencies))
    True
    >>> tb = wb.get_TimeAxis()
    >>> print(numpy.allclose(tb.data,times[5:10]))
    True
    """

    data = UnitsManagedArray("data")
    start = UnitsManaged("start")
    step = UnitsManaged("step")

    def __init__(self, start=0.0, length=1, step=1.0,
                 atype='complete', time_start=0.0):

        #if step > 0:
        if True:
            self.step = step
        #else:
        #    raise Exception("Parameter step has to be > 0")
        self.start = start
        self.length = length

        super().__init__(start=start,
                         length=length, step=step)

        # This would be the alternative if calling super()__init__ would
        # break the units management
        #self.data = numpy.linspace(start,
        #                           start+(length-1)*step, length,
        #                           dtype=numpy.float)

        self.time_start = time_start

        self.allowed_atypes = ["upper-half", "complete"]
        if atype in self.allowed_atypes:
            self.atype = atype
        else:
            raise Exception("Unknown frequency axis type")

    def copy(self):
        axis = FrequencyAxis(self.start, self.length, self.step,
                             atype=self.atype, time_start=self.time_start)
        return axis
        
        
    def get_TimeAxis(self):
        """Returns the corresponding TimeAxis object
        """
        from .timeaxis import TimeAxis

        with frequency_units("int"):

            if self.atype == 'complete':

                # This is consistent with fourier transform defined in 
                # Fourier transform module
                times = numpy.fft.fftshift(
                    numpy.fft.fftfreq(self.length, self.step/(2.0*numpy.pi)))

                step = times[1]-times[0]
                start = self.time_start  + times[0]
                nosteps = self.length

                frequency_start = self.data[self.length//2]


            elif self.atype == 'upper-half':

                if (self.length % 2) != 0:
                    raise Exception("Cannot create upper-half TimeAxis"
                                    + " from an odd number of points for"
                                    + " time axis not begining at zero")


                times = numpy.fft.fftshift(
                    (2.0*numpy.pi)*numpy.fft.fftfreq(self.length, self.step))

# TODO: Check if different definition would not produce error in starting value ~ 4.441e-16 for trasformation to frequency and back

#                if (self.length % 2) != 0:
#                    start = times[int( (self.length-1)/2)]
#                    nosteps = int( (self.length+1)/2)
#                else:
                start = times[int(self.length/2)] + self.time_start
                nosteps = int(self.length/2)
                
                step = times[1]-times[0]

#                if (self.length % 2) != 0:
#                    frequency_start = self.data[(self.length-1)//2]
#                else:
                frequency_start = self.data[self.length//2]

            else:
                raise Exception("Unknown frequency axis type")

        return TimeAxis(start, nosteps, step, 
                        atype=self.atype, frequency_start=frequency_start)

# TODO: Try the transformation of time axis 
        
    def get_rTimeAxis(self):
        """Returns the corresponding TimeAxis object
        """
        from .timeaxis import TimeAxis

        with frequency_units("int"):
            if self.atype == 'upper-half':

                if (self.length % 2) != 0:
                    raise Exception("Cannot create upper-half TimeAxis"
                                    + " from an odd number of points for"
                                    + " time axis not begining at zero")


                times = numpy.fft.fftshift(
                    (2.0*numpy.pi)*numpy.fft.fftfreq(self.length, self.step))

                start = times[int(self.length/2)] + self.time_start
                nosteps = int(self.length/2+1)
                
                step = times[1]-times[0]

                frequency_start = self.data[self.length//2]

            else:
                raise Exception("Unknown frequency axis type")

        return TimeAxis(start, nosteps, step, 
                        atype=self.atype, frequency_start=frequency_start)