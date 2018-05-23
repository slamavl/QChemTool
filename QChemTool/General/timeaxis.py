# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 18:02:51 2018

@author: Vladislav Sláma
"""
import numpy

from .valueaxis import ValueAxis
from .UnitsManager import frequency_units

class TimeAxis(ValueAxis):
    """ Class representing time in time dependent calculations.
    The `TimeAxis` class stands in a close relation to `FrequencyAxis`.
    `FrequencyAxis` represents the frequencies one obtains in the Fourier 
    transform of a function of the `TimeAxis`. By default, 
    `TimeAxis` is of the type `upper-half` which
    means that by specifying the `start`, `length` and `step` we
    represent the upper half of the interval `<start-length*step,
    start+(length-1)*step>`. The Fourier transform of a time dependent
    object defined on the `TimeAxis` will then have twice as many points as
    the `TimeAxis` (for time axis with start=0.0 there will be one point less
    in frequency in order not to duplicate value for zero time when some time
    symmetry is present). This is usefull when the time dependent object has some
    special symmetries. One example is the so-called quantum bath correlation
    function which fulfills the relation (in LaTeX)
    C(-t) = C^{*}(t)
    
    Parameters
    ----------
    start : float
        start of the TimeAxis
    length : int
        number of steps
    step : float
        time step
    atype : string {"complete","upper-half"}
        Axis type
    
    Attributes
    ----------
    data : float array
        Holds the values of time, it is equivalent to the atribute
        TimeAxis.time
    
    """
    
    def __init__(self, start=0.0, length=1, step=1.0,
                 atype="upper-half", frequency_start=0.0):


        ValueAxis.__init__(self, start=start,
                           length=length, step=step)

        self.frequency_start = frequency_start

        self.allowed_atypes = ["upper-half", "complete"]
        if atype in self.allowed_atypes:
            self.atype = atype
        else:
            raise Exception("Unknown time axis type")


    def get_FrequencyAxis(self):
        """ Returns corresponding FrequencyAxis object
        """
        from .frequencyaxis import FrequencyAxis

        if self.atype == 'complete':

            # This correspond to the definition of angular frequency omega in
            # FourierTransform module
            frequencies = numpy.fft.fftshift(
                (2.0*numpy.pi)*numpy.fft.fftfreq(self.length, self.step))
            

            step = frequencies[1]-frequencies[0]
            start = frequencies[0] + self.frequency_start

            nosteps = len(frequencies)
            time_start = self.data[self.length//2]

        elif self.atype == 'upper-half':
            
#            if self.start == 0.0:
#                frequencies = numpy.fft.fftshift(
#                    (2.0*numpy.pi)*numpy.fft.fftfreq(2*self.length-1, self.step))
#            else:
            frequencies = numpy.fft.fftshift(
                (2.0*numpy.pi)*numpy.fft.fftfreq(2*self.length, self.step))
# TODO: Check if different definition would not produce error in starting value ~ 4.441e-16 for trasformation to frequency and back


            start = frequencies[0] + self.frequency_start
            step = frequencies[1] - frequencies[0]
            nosteps = len(frequencies)
            time_start = self.min

        else:
            raise Exception("Unknown time axis type")

        # this creation has to be protected from units management
        with frequency_units("int"):
            faxis = FrequencyAxis(start, nosteps, step,
                                  atype=self.atype, time_start=time_start)

        return faxis
    
        
    def get_rFrequencyAxis(self):
        """ Returns corresponding FrequencyAxis object
        """
        from .frequencyaxis import FrequencyAxis

        if self.atype == 'upper-half':
            
            frequencies = numpy.fft.fftshift(
                (2.0*numpy.pi)*numpy.fft.fftfreq(2*self.length-2, self.step))


            start = frequencies[0] + self.frequency_start
            step = frequencies[1] - frequencies[0]
            nosteps = len(frequencies)
            time_start = self.min

        else:
            raise Exception("Unknown time axis type")

        # this creation has to be protected from units management
        with frequency_units("int"):
            faxis = FrequencyAxis(start, nosteps, step,
                                  atype=self.atype, time_start=time_start)

        return faxis
