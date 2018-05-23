# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 18:42:46 2018

@author: Vladislav Sláma
"""

import numpy as np

from ..QuantumChem.Classes.structure import Structure
from ..QuantumChem.Classes.general import Energy 
from ..General.UnitsManager import UnitsManaged
from ..QuantumChem.calc import identify_molecule
from ..General.UnitsManager import position_units, energy_units

global defects_database
defects_database = {}

class Defect(Structure, UnitsManaged):
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
    
    """

    def __init__(self, struc=None, energy=None, index=None, charges=None, params=None):
        super().__init__()
        
        if (struc is not None):
            if isinstance(struc, Structure):
                coor = struc.coor
                at_type = struc.at_type
                ncharge = struc.ncharge
                mass = struc.mass
            
            elif isinstance(struc, dict):
                coor = struc["coor"]
                at_type = struc["at_type"]
                ncharge = None
                mass = None
        else:
            raise IOError("Unknown structure type")
        
        
        self.add_coor(coor,at_type,ncharge=ncharge,mass=mass)
        self.level_of_int = 0
        
        
        if isinstance(energy,Energy):
            if np.isscalar(energy._value):
                energy._value = np.array(energy._value,dtype='f8')
            
            self.energy = energy
        else:
            energy = np.array([energy],dtype='f8')
            self.energy = Energy(energy)
            
        self.index = index
        
        try:
            self.elstat_index = params["elstat_indx"]
        except:
            self.elstat_index = index
        
        if isinstance(charges,dict):
            gr_charge = np.array(charges['ground'],dtype='f8')
            if np.ndim(charges['excited']) != np.ndim(charges['transition']):
                raise IOError("For every excited charge transition charges"  
                              + " has to be defined too")
                
            if np.ndim(charges['excited']) == 1:
                ex_charge = np.array([charges['excited']],dtype='f8')
                tr_charge = np.array([charges['transition']],dtype='f8')
            elif np.ndim(charges['excited']) == 2:
                ex_charge = np.array(charges['excited'],dtype='f8')
                tr_charge = np.array(charges['transition'],dtype='f8')
        else:
            gr_charge = None
            ex_charge = None
            tr_charge = None
        self.gr_charge = gr_charge
        self.ex_charge = ex_charge
        self.tr_charge = tr_charge
    
    def get_transition_energy(self):
        trE = Energy(self.energy.value[self.level_of_int])
        return trE

    def get_charge(self,state="transition"):
        if state == "transition":
            return self.tr_charge[self.level_of_int]
        elif state == "ground":
            return self.gr_charge
        elif state == "excited":
            return self.ex_charge[self.level_of_int]
        elif state == "excited-ground":
            return self.ex_charge[self.level_of_int] - self.gr_charge
        elif state == "ground-excited":
            return  self.gr_charge - self.ex_charge[self.level_of_int] 
    
    def get_dipole(self,state="transition"):
        charge =  self.get_charge(state=state)
        dipole =  np.dot(charge,self.coor._value)
        
        return dipole

    def _distance_matrix(self):
        # calculation of tensors with interatomic distances
        R=np.zeros((self.nat,self.nat,3),dtype='f8') # mutual distance vectors
        coor = self.coor.value
        for ii in range(self.nat):
            for jj in range(ii+1,self.nat):
                R[ii,jj,:]=coor[ii]-coor[jj]
                R[jj,ii,:]=-R[ii,jj,:]
        RR=np.sqrt(np.power(R[:,:,0],2)+np.power(R[:,:,1],2)+np.power(R[:,:,2],2))  # mutual distances
        
        return R,RR

    def identify_defect(self,other):
        """
        Identify atoms in between defects
        
        Parameters
        ---------
        other : Defect class
            Defect which will be identified with the original one
        
        Return
        --------
        index_res : list of integers
            indexes of atoms from other defect which correspond to the original one
        
        index_res[i]-th atom in the original defect = i-th atom of other defect 
        """
        vec1, dist1 = self._distance_matrix()
        vec2, dist2 = other._distance_matrix()
        
        variance1 = np.var(dist1,axis=0)
        mean1 = np.mean(dist1,axis=0)
        max1 = np.max(dist1,axis=0)
        
        variance2 = np.var(dist2,axis=0)
        mean2 = np.mean(dist2,axis=0)
        max2 = np.max(dist2,axis=0)
        
        # sort array
        index_sort1 = np.lexsort((max1, mean1, variance1))
        index_sort2 = np.lexsort((max2, mean2, variance2))
        
        connected1 = self.get_bonded_atoms()
        connected2 = other.get_bonded_atoms()

        condition = True
        ii = -1
        while condition:
            # pick corresponding atoms until they have only two bonds
            indx1 = index_sort1[ii]
            indx2 = index_sort2[ii]
            if len(connected1[indx1])==2 and len(connected2[indx2])==2:
                condition = False
            ii -= 1
            if abs(ii)>self.nat:
                raise Warning("No compatible atoms found")
        
        # identify the rest
        # Two possible combinations of vectors
        indx_center1 = indx1
        indx_x1 = connected1[indx1][0]
        indx_y1 = connected1[indx1][1]
        
        indx_center2 = indx2
        indx_x2 = connected2[indx2][0]
        indx_y2 = connected2[indx2][1]
        
        index1,RSMD1 = identify_molecule(self,other,indx_center1,indx_x1,indx_y1,indx_center2,indx_x2,indx_y2,onlyC=False,output_RSMD=True)
        index2,RSMD2 = identify_molecule(self,other,indx_center1,indx_x1,indx_y1,indx_center2,indx_y2,indx_x2,onlyC=False,output_RSMD=True)
        
        if RSMD1>RSMD2:
            index_res = index2
            RSMD = RSMD2
        else:
            index_res = index1
            RSMD = RSMD1
            
        return index_res,RSMD
    
    def load_charges_enegy_from_defect(self,other,debug=False):
        index,RSMD = other.identify_defect(self)
        
        if isinstance(other, Defect):
            self.gr_charge = other.gr_charge[index]
            self.ex_charge = []
            self.tr_charge = []
            for ii in range(len(other.tr_charge)):
                self.ex_charge.append(other.ex_charge[ii][index])
                self.tr_charge.append(other.tr_charge[ii][index])
            
            self.energy = other.energy
            
            if debug: # TEST
                self.output_to_pdb("def_self.pdb")
                other.coor.value = other.coor.value[index]
                other.output_to_pdb("def_other.pdb")
            
        else:
            pass

# =============================================================================
# Perylene
# =============================================================================
struc_A={}
charge={}
struc_A["coor"] = [[0.000000,1.266262,0.745575],
                [0.000000,    0.000000,    1.455600],
                [0.000000,   -1.266262,    0.745575],
                [0.000000,   -2.466042,    1.498146],
                [0.000000,   -2.460314,    2.923799],
                [0.000000,   -1.248407,    3.625728],
                [0.000000,    0.000000,    2.913210],
                [0.000000,    1.248407,    3.625728],
                [0.000000,    2.460314,    2.923799],
                [0.000000,    2.466042,    1.498146],
                [0.000000,   -1.266262,   -0.745575],
                [0.000000,   -2.466042,   -1.498146],
                [0.000000,   -2.460314,   -2.923799],
                [0.000000,   -1.248407,   -3.625728],
                [0.000000,    0.000000,   -2.913210],
                [0.000000,    1.248407,   -3.625728],
                [0.000000,    2.460314,   -2.923799],
                [0.000000,    2.466042,   -1.498146],
                [0.000000,    1.266262,   -0.745575],
                [0.000000,    0.000000,   -1.455600]]
struc_A["at_type"] = ["C"]*20
charge["ground"] = [-0.869066,  1.877573, -0.869066,  0.502141, -0.281315,  0.753299,
       -2.087692,  0.753299, -0.281315,  0.502141, -0.869066,  0.502141,
       -0.281315,  0.753299, -2.087692,  0.753299, -0.281315,  0.502141,
       -0.869066,  1.877573]
charge["excited"] = [[-0.826134,  1.795568, -0.826134,  0.462953, -0.264799,  0.739809,
        -2.019226,  0.739809, -0.264799,  0.462953, -0.826134,  0.462953,
        -0.264799,  0.739809, -2.019226,  0.739809, -0.264799,  0.462953,
        -0.826134,  1.795568]]
charge["transition"] = [[ 0.020079,  0.086397,  0.020079, -0.121994,  0.041025, -0.095796,
         0.001578, -0.095796,  0.041025, -0.121994, -0.020079,  0.121994,
        -0.041025,  0.095796, -0.001578,  0.095796, -0.041025,  0.121994,
        -0.020079, -0.086397]]

E01_cm={"exp": 22986.80041,"QC": 25405.43635}

            
perylene_defect = {"struc_angstrom": struc_A, "charges": charge,"E01_cm": E01_cm, "tr_dipole_exp": None}

# =============================================================================
# Anthanthrene
# =============================================================================
struc_A={}
charge={}
struc_A["coor"] = [[-1.309859,    2.250795,   -0.000000],
                    [-1.678206,    3.639609,    0.000000],
                    [-0.691853,    4.645211,    0.000000],
                    [0.691853 ,   4.307705 ,   0.000000 ],
                    [1.106359 ,   2.948747 ,   0.000000 ],
                    [0.102328 ,   1.901294 ,   0.000000 ],
                    [0.497997 ,   0.518926 ,   0.000000 ],
                    [-0.497997,   -0.518926,   -0.000000],
                    [-1.914985,   -0.160474,    0.000000],
                    [-2.287749,    1.199665,   -0.000000],
                    [-2.899395,   -1.236660,    0.000000],
                    [-2.514029,   -2.565962,   -0.000000],
                    [-1.106359,   -2.948747,    0.000000],
                    [-0.102328,   -1.901294,    0.000000],
                    [1.309859 ,  -2.250795 ,   0.000000 ],
                    [1.678206 ,  -3.639609 ,   0.000000 ],
                    [0.691853 ,  -4.645211 ,   0.000000 ],
                    [-0.691853,   -4.307705,    0.000000],
                    [2.287749 ,  -1.199665 ,   0.000000 ],
                    [1.914985 ,   0.160474 ,   0.000000 ],
                    [2.899395 ,   1.236660 ,  -0.000000 ],
                    [2.514029 ,   2.565962 ,   0.000000 ]]
struc_A["at_type"] = ["C"]*22

charge["ground"] = [-1.871484,  0.543482, -0.191817,  0.585278, -1.542893,  1.767992,
       -0.04442 , -0.04442 , -0.9735  ,  1.155994,  0.012021,  0.559347,
       -1.542893,  1.767992, -1.871484,  0.543482, -0.191817,  0.585278,
        1.155994, -0.9735  ,  0.012021,  0.559347]
charge["excited"] = [[-1.835677,  0.538806, -0.194141,  0.572344, -1.532669,  1.742148,
       -0.04018 , -0.04018 , -0.980761,  1.143841,  0.022745,  0.563544,
       -1.532669,  1.742148, -1.835677,  0.538806, -0.194141,  0.572344,
        1.143841, -0.980761,  0.022745,  0.563544]]
charge["transition"] = [[ 0.007236, -0.102847,  0.047689, -0.121857,  0.030959,  0.066166,
       -0.037418,  0.037418,  0.004832, -0.109276,  0.00569 ,  0.001517,
       -0.030959, -0.066166, -0.007236,  0.102847, -0.047689,  0.121857,
        0.109276, -0.004832, -0.00569 , -0.001517]]

E01_cm={"exp": 23067.45585,"QC": 24998.92356}

            
anthanthrene_defect = {"struc_angstrom": struc_A,"charges": charge,"E01_cm": E01_cm, "tr_dipole_exp": None}

# =============================================================================
# Bisanthrene
# =============================================================================
struc_A={}
charge={}
struc_A["coor"] = [[0.000000,    1.250479,    1.447878],
                    [0.000000,    2.521549,    0.742687],
                    [0.000000,    3.716350,    1.501862],
                    [0.000000,    3.702730,    2.930016],
                    [0.000000,    2.491014,    3.627633],
                    [0.000000,    1.240718,    2.909920],
                    [0.000000,    0.000000,    3.601113],
                    [0.000000,   -1.240718,    2.909920],
                    [0.000000,   -1.250479,    1.447878],
                    [0.000000,    0.000000,    0.734324],
                    [0.000000,    0.000000,   -0.734324],
                    [0.000000,   -1.250479,   -1.447878],
                    [0.000000,   -2.521549,   -0.742687],
                    [0.000000,   -3.716350,   -1.501862],
                    [0.000000,   -3.702730,   -2.930016],
                    [0.000000,   -2.491014,   -3.627633],
                    [0.000000,   -1.240718,   -2.909920],
                    [0.000000,    0.000000,   -3.601113],
                    [0.000000,    1.240718,   -2.909920],
                    [0.000000,    1.250479,   -1.447878],
                    [0.000000,    2.521549,   -0.742687],
                    [0.000000,    3.716350,   -1.501862],
                    [0.000000,    3.702730,   -2.930016], 
                    [0.000000,    2.491014,   -3.627633], 
                    [0.000000,   -2.521549,    0.742687], 
                    [0.000000,   -3.716350,    1.501862], 
                    [0.000000,   -3.702730,    2.930016], 
                    [0.000000,   -2.491014,    3.627633]]
struc_A["at_type"] = ["C"]*28

charge["ground"] = [ 1.496958, -0.769056,  0.406194, -0.069746,  0.381003, -1.770397,
        1.370574, -1.770397,  1.496958, -0.720487, -0.720487,  1.496958,
       -0.769056,  0.406194, -0.069746,  0.381003, -1.770397,  1.370574,
       -1.770397,  1.496958, -0.769056,  0.406194, -0.069746,  0.381003,
       -0.769056,  0.406194, -0.069746,  0.381003]
charge["excited"] = [[ 1.483251, -0.760417,  0.386602, -0.067345,  0.384849, -1.74968 ,
        1.36152 , -1.74968 ,  1.483251, -0.71604 , -0.71604 ,  1.483251,
       -0.760417,  0.386602, -0.067345,  0.384849, -1.74968 ,  1.36152 ,
       -1.74968 ,  1.483251, -0.760417,  0.386602, -0.067345,  0.384849,
       -0.760417,  0.386602, -0.067345,  0.384849]]
charge["transition"] = [[-0.036667, -0.007477,  0.089221, -0.014949,  0.043396,  0.005516,
        0.091638,  0.005516, -0.036667, -0.008777,  0.008777,  0.036667,
        0.007477, -0.089221,  0.014949, -0.043396, -0.005516, -0.091638,
       -0.005516,  0.036667,  0.007477, -0.089221,  0.014949, -0.043396,
       -0.007477,  0.089221, -0.014949,  0.043396]]

E01_cm={"exp": 15082.56729,"QC": 16154.04404}

            
bisanthrene_defect = {"struc_angstrom": struc_A,"charges": charge,"E01_cm": E01_cm, "tr_dipole_exp": None}

# =============================================================================
# Perylene2
# =============================================================================
struc_A={}
charge={}
struc_A["coor"] = [[-0.000000,   -0.000000 ,   1.443547],
                    [1.260508,    0.000000 ,   0.723447 ],
                    [1.260508,    0.000000 ,  -0.723447 ],
                    [0.000000,    0.000000 ,  -1.443547 ],
                    [-1.260508,    0.000000,   -0.723447],
                    [-1.260508,   -0.000000,    0.723447],
                    [2.489651,    0.000000 ,  -1.496444 ],
                    [0.000000,    0.000000 ,  -2.894728 ],
                    [1.247410,    0.000000 ,  -3.630817 ],
                    [2.487758,    0.000000 ,  -2.883513 ],
                    [1.226912,    0.000000 ,  -5.058214 ],
                    [0.000000,    0.000000 ,  -5.763235 ],
                    [-1.226912,    0.000000,   -5.058214],
                    [-1.247410,    0.000000,   -3.630817],
                    [-2.487758,    0.000000,   -2.883513],
                    [-2.489651,    0.000000,   -1.496444],
                    [2.489651 ,   0.000000 ,   1.496444 ],
                    [-0.000000,   -0.000000,    2.894728],
                    [-2.489651,   -0.000000,    1.496444],
                    [2.487758 ,  -0.000000 ,   2.883513 ],
                    [1.247410 ,  -0.000000 ,   3.630817 ],
                    [-2.487758,   -0.000000,    2.883513],
                    [-1.247410,   -0.000000,    3.630817], 
                    [1.226912 ,  -0.000000 ,   5.058214 ],
                    [-1.226912,   -0.000000,    5.058214],
                    [-0.000000,   -0.000000,    5.763235]]
struc_A["at_type"] = ["C"]*26

charge["ground"] = [0.062581, -0.22072 , -0.22072 ,  0.062581, -0.22072 , -0.22072 ,
        0.025906,  1.382386, -1.574847,  0.518613,  0.687885, -0.318641,
        0.687885, -1.574847,  0.518613,  0.025906,  0.025906,  1.382386,
        0.025906,  0.518613, -1.574847,  0.518613, -1.574847,  0.687885,
        0.687885, -0.318641]
charge["excited"] = [[0.088661, -0.234489, -0.234489,  0.088661, -0.234489, -0.234489,
        0.035014,  1.358979, -1.562976,  0.515427,  0.679739, -0.313072,
        0.679739, -1.562976,  0.515427,  0.035014,  0.035014,  1.358979,
        0.035014,  0.515427, -1.562976,  0.515427, -1.562976,  0.679739,
        0.679739, -0.313072]]
charge["transition"] = [[-0.048166, -0.042862,  0.042862,  0.048166,  0.042862, -0.042862,
       -0.013002, -0.064366, -0.030494,  0.068847,  0.101332, -0.028527,
        0.101332, -0.030494,  0.068847, -0.013002,  0.013002,  0.064366,
        0.013002, -0.068847,  0.030494, -0.068847,  0.030494, -0.101332,
       -0.101332,  0.028527]]

E01_cm={"exp": None,"QC": 24480.29712}

            
perylene2_defect = {"struc_angstrom": struc_A,"charges": charge,"E01_cm": E01_cm, "tr_dipole_exp": None}

def initialize_defect_database(energy_type="QC"):
    defects_database = {}
    # FIXME: do this automaticaly
    with position_units("Angstrom"):
        with energy_units("1/cm"):
            perylene = Defect(struc=perylene_defect["struc_angstrom"], 
                              energy= perylene_defect["E01_cm"][energy_type],
                              charges=perylene_defect["charges"])
            anthanthrene = Defect(struc=anthanthrene_defect["struc_angstrom"], 
                              energy= anthanthrene_defect["E01_cm"][energy_type],
                              charges=anthanthrene_defect["charges"])
            bisanthrene = Defect(struc=bisanthrene_defect["struc_angstrom"], 
                              energy= bisanthrene_defect["E01_cm"][energy_type],
                              charges=bisanthrene_defect["charges"])
            #print(perylene2_defect["struc_angstrom"]["coor"], perylene2_defect["struc_angstrom"]["at_type"])
            perylene2 = Defect(struc=perylene2_defect["struc_angstrom"], 
                              energy= perylene2_defect["E01_cm"][energy_type],
                              charges=perylene2_defect["charges"])
    defects_database["perylene"] = perylene
    defects_database["anthanthrene"] = anthanthrene
    defects_database["bisanthrene"] = bisanthrene
    defects_database["perylene2"] = perylene2
    
    return defects_database

    