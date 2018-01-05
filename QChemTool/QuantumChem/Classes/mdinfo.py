# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:12:49 2016

@author: uzivatel
"""


class MDinfo:
    '''Class managing information from MD simulation.
    '''
    def __init__(self):
        self.NStep = []
        self.timestep = None
        self.NAtom = []
        self.at_name = []
        self.geom = []
    