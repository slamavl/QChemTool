# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:33:56 2017

@author: Vladislav Sláma
"""

from .polarization_periodic import Dielectric
from .electrostatic import Electrostatics
from .defect import Defect,initialize_defect_database
from ..General.UnitsManager import UnitsManaged, energy_units
from ..General.Potential import potential_dipole
from ..QuantumChem.Classes.general import Energy
from ..QuantumChem.interaction import charge_charge
from ..General.units import conversion_facs_energy,conversion_facs_mass
from ..QuantumChem.Fluorographene.fluorographene import constrainsFG
from ..QuantumChem.Classes.structure import Structure
import numpy as np

# TODO:
#   Add relative permitivity into calculation of derivatives
#   Add possibility to have different indexes for environment and polarizable def. 
#   Add possibility to specify defect for electrostatic energy calculation
#   Check if get_SingleDefectProperties can be also used on system with 2 defects

class PolarizableSystem(UnitsManaged):    
    ''' Class managing dielectric properties of the material 
    
    Parameters
    ----------
    coor : numpy.array of real (dimension Nx3) where N is number of atoms
        origin of density grid
        
    polar : numpy.array or list of real (dimension N)
        Polarizabilities for every atom
        
    charge : numpy.array or list of real (dimension N)
        charges on individual atoms (initial charges)

    dipole : numpy.array of real (dimension Nx3)
        dipole on individual atoms (initial dipole)
    '''
    
    allowed_charges = ['Hirshfeld',"ESPfit",'CM5','zero']
    allowed_coarse_grain = ["plane","C"]
    
    def __init__(self,diel = None, elstat = None, defect = None, params = None):
        
        if isinstance(diel,Dielectric):
            self.diel = diel
        elif isinstance(diel,dict):
            try:
                struc = diel["structure"]
                if "polarizability" in diel["polar"]:
                    polar_params = diel["polar"]
                else:
                    polar_params = diel["polar"]
                    charge_type = polar_params["charge_type"]
                    coarse_grain = polar_params["coarse_grain"]
                    approx = polar_params["approximation"]
                    use_VinterFG = polar_params["VinterFG"]
                    try:
                        symm = polar_params["symm"]
                    except:
                        symm = False
                    polar_params = _get_diel_params(charge_type, coarse_grain, approx, use_VinterFG,symm=symm)
            except:
                raise IOError("elstat must be Dielectric class or" + 
                              "dictionary with structure and polarizability " + 
                              "parameters")
                
            self.diel = self._build_polar(struc,polar_params)
        
        if isinstance(elstat,Electrostatics):
            self.elstat = elstat
        elif isinstance(elstat,dict):
            try: 
                struc = elstat["structure"]
                if isinstance(elstat["charge"],dict):
                    charge = elstat["charge"]
                elif isinstance(elstat["charge"],str):
                    charge = _get_elstat_params(elstat["charge"])
                else:
                    charge = None
            except:
                raise IOError("elstat must be Electrostatics class or" + 
                              "dictionary with structure and FG charges")
            
            self.elstat = self._build_elstat(struc,charge)
        
        
        self.defects = []
        if defect is not None:
            if isinstance(defect,Defect):
                self._add_defect(defect)
            else:
                for dfct in defect:
                    self._add_defect(defect)
        
        if params is not None:
            try:
                self.eps = params["permivity"]
            except:
                self.eps = 1.0
            try:
                self.order = params["order"]
            except:
                self.order = 2
            try:
                self.energy_type = params["energy_type"]
                self.defect_database = initialize_defect_database(params["energy_type"])
            except:
                self.defect_database = initialize_defect_database("QC")
                
        else:
            self.eps = 1.0
            self.order = 2
            self.defect_database = initialize_defect_database("QC")
        
    def _add_defect(self,defect):
        
        if isinstance(defect,Defect):
            self.defects.append(defect)
    
    def _build_elstat(self,struc,charge):
        # Identify inside carbons connected to fluorines and outside ones
        if struc.bonds is None:
            struc.guess_bonds()
        NBonds=np.zeros(len(struc.bonds),dtype='i')
        for ii in range(len(struc.bonds)):
            if struc.at_type[struc.bonds[ii,0]]=='C' and struc.at_type[struc.bonds[ii,1]]=='F':
                NBonds[struc.bonds[ii,0]]+=1
        
        # Assing type for every atom - in this step all fluorines will have the same type
        Elstat_Type=[]
        for ii in range(struc.nat):
            if struc.at_type[ii]=='C':
                if NBonds[ii]==2:
                    Elstat_Type.append('CF2')
                elif NBonds[ii]==1:
                    Elstat_Type.append('CF')
                else:
                    Elstat_Type.append('CD')
            elif struc.at_type[ii]=='F':
                Elstat_Type.append('FC')
        
        # Assign different atom types for fluorines at the border:
        for ii in range(len(struc.bonds)):
            if Elstat_Type[struc.bonds[ii,0]]=='CF2' and Elstat_Type[struc.bonds[ii,1]]=='FC':
                Elstat_Type[struc.bonds[ii,1]]='F2C'
        
        # Asign charges for fluorographene:
        Elstat_Charge=np.zeros(struc.nat,dtype='f8')
        for ii in range(struc.nat):
            Elstat_Charge[ii]=charge[Elstat_Type[ii]]
        
        # defects have zero charges
        
        return Electrostatics(struc.coor._value,Elstat_Charge,Elstat_Type)
    
    def _build_polar(self,struc,params):
        
        coarse_grain = params["coarse_grain"]
        
        PolCoor,Polcharge,PolType,PolPhase = _prepare_polar_structure(struc,coarse_grain,verbose=False)
        dipole_init = np.zeros((len(PolCoor),3),dtype='f8')
        
        mol_polar=Dielectric(PolCoor,PolType,Polcharge,dipole_init,params)
        
        mol_polar.set_geom_phase(PolPhase)
        
#        try:
#            kwargs={"PolValues":params["polarizability"]}
#            mol_polar.polar = mol_polar.assign_polar(**kwargs)
#        except:
#            raise IOError('Polarizabilities for all atom types has to be defined')
        
        return mol_polar
    
    def _get_defects(self):
        # Get defect carbon indexes
        system = self.elstat.get_structure()
        indx_FG = constrainsFG(system,border=False,defect=True)
        indx_FG = np.array(indx_FG,dtype="i8")
        
        # Get coordinates and atomic types of defect carbons
        coor = system.coor.value[indx_FG]
        at_type = []
        for ii in indx_FG:
            at_type.append( system.at_type[ii] )
        
        # Create structure only with defects - no FG environment
        FGdefects = Structure()
        FGdefects.add_coor(coor,at_type)
        
        # split structure to individual not connected defects
        indx_def = FGdefects.count_fragments()
        Ndef = len(indx_def)
        defects = []
        for ii in range(Ndef):
            at_type = []
            for jj in indx_def[ii]:
                at_type.append( FGdefects.at_type[jj] )
            struc = {"coor": FGdefects.coor.value[indx_def[ii]], "at_type": at_type}
            index = list(indx_FG[ indx_def[ii] ])
            defct = Defect(struc=struc, index=index)
            defects.append(defct)
        
        return defects
    
    def identify_defects(self):
        if not self.defects:
            self.defects = self._get_defects()
        
        count = 0
        for defect in self.defects:
            Ncarbon = defect.nat
            identifed = False
            for name in self.defect_database:
                tested_def = self.defect_database[name]
                # FIXME use better criterion when there are more defects with the same number of atoms
                if Ncarbon==tested_def.nat:
                    defect.name = name
                    identifed = True
                    defect.load_charges_enegy_from_defect(tested_def)
            
            if not identifed:
                print("Defect with index",count,"was not identified. It must be done manually" )
            count +=1
            
    def _induce_dipoles(self, poltype, ext_field = None, order = None):
        if ext_field is None:
            ext_field = np.zeros(3,dtype='f8')
        if order is None:
            self.diel._calc_dipoles_All(poltype,Estatic=ext_field,NN=self.order,
                                    eps=self.eps,debug=False)
        else:
            self.diel._calc_dipoles_All(poltype,Estatic=ext_field,NN=order,
                                    eps=self.eps,debug=False)
    
    def get_elstat_energy(self,indx_defect,charge_type):
        defect = self.defects[indx_defect]
        index = defect.index
        charge = defect.get_charge(state=charge_type)
        env_charge = self.elstat.charge.copy()
        
        self.elstat.charge[index] = charge
        res = self.elstat.get_EnergyShift()
        
        self.elstat.charge = env_charge
        
        with energy_units("AU"):
            res = Energy(res)
        
        return res
        
    
    def _zero_induced_dipoles(self):
        self.diel.dipole = np.zeros((self.diel.Nat,3),dtype='f8')
            
    def _dR_BpA(self, indxA, indxB, poltype, chargeAtype='transition',
                chargeBtype='transition', eps=1):
        ''' function which calculate derivation of interaction energy between defect
        A and defect B defined by index: \n
        d/dR[Sum_{n} E^{(B)}(Rn).(1/2*Polarizability(n)).E^{(A)}(Rn)] \n
        which is -interaction energy ( for derivation of energy, Hamiltonian,
        we need to take negative value of the result)
        
        Parameters
        ----------
        indxA : integer
            Index of the defect A
        indxB : integer
            Index of the defect A
        chargeAtype : string
            Which atomic charges are used for interaction energy calculation on
            defect A. Allowed types are ``'transition'``, ``'ground'`` and 
            ``'excited'`` 
        chargeBtype : string
            Which atomic charges are used for interaction energy calculation on
            defect A. Allowed types are ``'transition'``, ``'ground'`` and 
            ``'excited'`` 
        poltype : str ('AlphaE','Alpha_E','BetaEE')
            Specifies which polarizability is used for calculation of induced
            atomic dipoles
        eps : real (optional - init=1.0)
            Relative dielectric polarizability of medium where the dipoles and 
            molecule is present ( by default vacuum with relative permitivity 
            1.0)

        Notes
        ----------
        **After exiting the function transition charges are the same as in the
        begining**
        
        For calculation of derivation of ApA use ``_dR_BpA(indxA,indxA,
        chargeAtype,chargeAtype,typ,eps=1)``.
        '''

        Nat = self.diel.Nat
        res=np.zeros((Nat,3),dtype='f8')
        
        charge = np.zeros(Nat,dtype='f8')
        index1 = self.defects[indxA].index
        index2 = self.defects[indxB].index
        charge1 = self.defects[indxA].get_charge(state=chargeAtype)
        charge2 = self.defects[indxB].get_charge(state=chargeBtype)

        # FIXME: Other defect might be imputed with ground state charges
# TODO: Add posibility to read charges from self.charges: charge1 = self.charges[index1] and charge2 = self.charges[index2]
# TODO: Read polarizabilities on the defects and when potting charges to zero put also zero polarizabilities       
        

        # calculation of tensors with interatomic distances
        R,RR = self.diel.get_distance_matrixes()
        RR=RR+np.identity(Nat) # only for avoiding ddivision by 0 for diagonal elements
        RR3=np.power(RR,3)
        RR5=np.power(RR,5)
        P=np.zeros((Nat,3),dtype='f8')

        # Initialize T tensor
        T=self.diel.get_T_tensor(R, RR, RR3, RR5)
        
        # Place transition charges only on the first defect (defect A)
        charge[index1] = charge1
        
        # FIXME: This can be done by calc_dipoles (but slower)
        # calculating derivation according to defect B atom displacement
        Q=np.meshgrid(charge,charge)[0]   # in columns same charges
        ELF=np.zeros((Nat,Nat,3),dtype='f8')
        
        # calculate electric field generated by the first defect (defect A)
        for jj in range(3):
            ELF[:,:,jj]=(Q/RR3)*R[:,:,jj]   # ELF[i,j,:]  is electric field at position i generated by atom j 
        for ii in range(Nat):
            ELF[ii,ii,:]=np.zeros(3,dtype='f8')
        
        # calculate induced dipoles induced by the first defect (defect A)
        ELFV=np.array(np.sum(ELF,axis=1),dtype='f8')             # ELFV[i,:]   is electric field at position of atom i
        for ii in range(Nat):
            P[ii,:]=np.dot(self.diel.polar[poltype][ii],ELFV[ii,:])
        
        ELFV=np.zeros((Nat,3),dtype='f8')
        for ii in range(3):
            for jj in range(3):
                ELFV[:,ii]+=np.dot(T[:,:,ii,jj],P[:,jj])
        
        for ii in range(len(index2)):
            res[index2[ii],:] -= charge2[ii]*ELFV[index2[ii],:]
            
        # calculating derivation with respect to displacement of environment atom
        for ii in range(Nat):
            if not (ii in index1 or ii in index2):
                for jj in range(len(index2)):
                    res[ii,:]+=charge2[jj]*np.dot(T[index2[jj],ii,:,:],P[ii,:])
        
        
        # Place transition charges only on the second defect (defect B)
        charge = np.zeros(Nat,dtype='f8')
        charge[index2] = charge2
        
        # calculating derivation according to atom displacement from defect A
        Q=np.meshgrid(charge,charge)[0]   # in columns same charges
        ELF=np.zeros((Nat,Nat,3),dtype='f8')
        
        # Calculate electric field generated by the second defect (defect B)
        for jj in range(3):
            ELF[:,:,jj]=(Q/RR3)*R[:,:,jj]   # ELF[i,j,:]  is electric field at position i generated by atom j 
        for ii in range(Nat):
            ELF[ii,ii,:]=np.zeros(3,dtype='f8')
            
        # Calculate induced dipoles, induced by the second defect (defect B)
        ELFV=np.array(np.sum(ELF,axis=1),dtype='f8')             # ELFV[i,:]   is electric field at position of atom i
        for ii in range(Nat):
            P[ii,:]=np.dot(self.diel.polar[poltype][ii],ELFV[ii,:])
        
        ELFV=np.zeros((Nat,3),dtype='f8')
        for ii in range(3):
            for jj in range(3):
                ELFV[:,ii]+=np.dot(T[:,:,ii,jj],P[:,jj])
        
        for ii in range(len(index1)):
            res[index1[ii],:] -= charge1[ii]*ELFV[index1[ii],:]
            
        # calculating derivation with respect to displacement of environment atom
        for ii in range(Nat):
            if not (ii in index1 or ii in index2):
                for jj in range(len(index1)):
                    res[ii,:]+=charge1[jj]*np.dot(T[index1[jj],ii,:,:],P[ii,:])

        return res.reshape(3*Nat)
    
    
    
    
    
    
    def _dR_BppA(self, indxA, indxB, poltype, chargeAtype='transition',
                chargeBtype='transition', eps=1):
        ''' function which calculate derivation of second order interaction energy
        between defect A and defect B defined by index1 resp. index2: \n
        ``d/dR[Sum_{n} E^{(B)}(Rn).(1/2*Polarizability(n)). Sum_{n'} T(Rn-Rn').(1/2*Polarizability(n')).E^{(A)}(Rn)]`` \n
        which is -interaction energy ( for derivation of energy, Hamiltonian,
        we need to take negative value of the result)
        
        Parameters
        ----------
        indxA : integer
            Index of the defect A
        indxB : integer
            Index of the defect A
        chargeAtype : string
            Which atomic charges are used for interaction energy calculation on
            defect A. Allowed types are ``'transition'``, ``'ground'`` and 
            ``'excited'`` 
        chargeBtype : string
            Which atomic charges are used for interaction energy calculation on
            defect A. Allowed types are ``'transition'``, ``'ground'`` and 
            ``'excited'`` 
        poltype : str ('AlphaE','Alpha_E','BetaEE')
            Specifies which polarizability is used for calculation of induced
            atomic dipoles
        eps : real (optional - init=1.0)
            Relative dielectric polarizability of medium where the dipoles and 
            molecule is present ( by default vacuum with relative permitivity 
            1.0)

        Notes
        ----------
        **After exiting the function transition charges are placed on both defects
        and not only on the first**
        
        For calculation of derivation of AppA use ``_dR_BppA(index1,index1,
        charge1,charge1,typ,eps=1)`` where charges in molecule Dielectric class
        have to be nonzero for defect with ``index1`` **and zero for the other
        defect if present**.
        '''
        
        Nat = self.diel.Nat
        res = np.zeros((Nat,3),dtype='f8')
        
        charge = np.zeros(Nat,dtype='f8')
        index1 = self.defects[indxA].index
        index2 = self.defects[indxB].index
        charge1 = self.defects[indxA].get_charge(state=chargeAtype)
        charge2 = self.defects[indxB].get_charge(state=chargeBtype)

# TODO: Add posibility to read charges from self.charges: charge1 = self.charges[index1] and charge2 = self.charges[index2]
# TODO: Read polarizabilities on the defects and when potting charges to zero put also zero polarizabilities       
        
        # calculation of tensors with interatomic distances
        P=np.zeros((self.diel.Nat,3),dtype='f8')
        R,RR = self.diel.get_distance_matrixes()
        RR=RR+np.identity(Nat) # only for avoiding ddivision by 0 for diagonal elements    
        RR3=np.power(RR,3)
        RR5=np.power(RR,5)
        
        # Initialize T tensor
        T=self.diel.get_T_tensor(R, RR, RR3, RR5)
        
        # definition of S tensor
        S=self.diel.get_S_tensor(R, RR, RR5)
        
        # Place transition charges only on the first defect (defect A)
        charge[index1] = charge1
        
        # calculating derivation according to atom displacement from defect B
        Q=np.meshgrid(charge,charge)[0]   # in columns same charges
        ELF=np.zeros((Nat,Nat,3),dtype='f8')
        
        # Calculate electric field generated by the first defect (defect A)
        for jj in range(3):
            ELF[:,:,jj]=(Q/RR3)*R[:,:,jj]   # ELF[i,j,:]  is electric field at position i generated by atom j 
        for ii in range(Nat):
            ELF[ii,ii,:]=np.zeros(3,dtype='f8')
            
        # Calculate induced dipoles, induced by the first defect (defect A)
        ELFV=np.array(np.sum(ELF,axis=1),dtype='f8')             # ELFV[i,:]   is electric field at position of atom i
        PA=np.zeros((Nat,3),dtype='f8')
        for ii in range(Nat):
            PA[ii,:]=np.dot(self.diel.polar[poltype][ii],ELFV[ii,:])

        for rep in range(2):
            P=np.zeros((Nat,3),dtype='f8')
            for ii in range(Nat):
                P[ii,:]=np.dot(self.diel.polar[poltype][ii],ELFV[ii,:])
            
            ELFV=np.zeros((Nat,3),dtype='f8')
            for ii in range(3):
                for jj in range(3):
                    ELFV[:,ii]+=np.dot(T[:,:,ii,jj],P[:,jj])
                
        for ii in range(len(index2)):
            res[index2[ii],:] += charge2[ii]*ELFV[index2[ii],:]
            
        # calculating derivation with respect to displacement of environment atom
        for ii in range(Nat):
            if not (ii in index1 or ii in index2):
                for jj in range(len(index2)):
                    res[ii,:] -= charge2[jj]*np.dot(T[index2[jj],ii,:,:],P[ii,:])
                    
        # Place transition charges only on the second defect (defect B)
        charge = np.zeros(Nat,dtype='f8')
        charge[index2] = charge2
        
        # calculating derivation according to atom displacement from defect A
        Q=np.meshgrid(charge,charge)[0]   # in columns same charges
        ELF=np.zeros((Nat,Nat,3),dtype='f8')
        
        # Calculate electric field generated by the second defect (defect B)
        for jj in range(3):
            ELF[:,:,jj]=(Q/RR3)*R[:,:,jj]   # ELF[i,j,:]  is electric field at position i generated by atom j 
        for ii in range(Nat):
            ELF[ii,ii,:]=np.zeros(3,dtype='f8')
            
        # Calculate induced dipoles, induced by the second defect (defect B)
        ELFV=np.array(np.sum(ELF,axis=1),dtype='f8')             # ELFV[i,:]   is electric field at position of atom i
        PB=np.zeros((Nat,3),dtype='f8')
        for ii in range(Nat):
            PB[ii,:]=np.dot(self.diel.polar[poltype][ii],ELFV[ii,:])        
        
        for rep in range(2):
            P=np.zeros((Nat,3),dtype='f8')
            for ii in range(Nat):
                P[ii,:]=np.dot(self.diel.polar[poltype][ii],ELFV[ii,:])
            
            ELFV=np.zeros((Nat,3),dtype='f8')
            for ii in range(3):
                for jj in range(3):
                    ELFV[:,ii]+=np.dot(T[:,:,ii,jj],P[:,jj])                
        
        for ii in range(len(index1)):
            res[index1[ii],:] += charge1[ii]*ELFV[index1[ii],:]
        
        # calculating derivation with respect to displacement of environment atom
        for ii in range(Nat):
            if not (ii in index1 or ii in index2):
                for jj in range(len(index1)):
                    res[ii,:] -= charge1[jj]*np.dot(T[index1[jj],ii,:,:],P[ii,:])
        
        # + contribution from S tensor
        for nn in range(Nat):
            for ii in range(3):
                for kk in range(3):
                    res[nn,:]+=3*PB[nn,ii]*np.dot(S[nn,:,ii,:,kk].T,PA[:,kk])
                    res[nn,:]+=3*PA[nn,ii]*np.dot(S[nn,:,ii,:,kk].T,PB[:,kk])
                    
        
        return res.reshape(3*Nat)
    
    def _dR_ApEnv(self, indx, poltype, charge_type='transition'):
        ''' function which calculate derivation of 'interaction energy' between defect
        A defined by index and environment atoms: \n
        d/dR[Sum_{n} E^{(A)}(Rn).(1/2*Polarizability(n)).E^{(env)}(Rn)] \n
        which is -interaction energy ( for derivation of energy, Hamiltonian,
        we need to take negative value of the result)
        
        Parameters
        ----------
        indxA : integer
            Index of the defect
        chargeAtype : string
            Which atomic charges are used for interaction energy calculation on
            defect A. Allowed types are ``'transition'``, ``'ground'`` and 
            ``'excited'`` 
        poltype : str ('AlphaE','Alpha_E','BetaEE')
            Specifies which polarizability is used for calculation of induced
            atomic dipoles
            
        Return
        ----------
        res
        res_env
        

        Notes
        ----------
        **After exiting the function transition charges are the same as in the
        begining**
        '''

# TODO: Add posibility to read charges from self.charges: charge1 = self.charges[index1] and charge2 = self.charges[index2]
# TODO: Read polarizabilities on the defects and when potting charges to zero put also zero polarizabilities       
        
        Nat = self.diel.Nat
        env_Nat = self.elstat.coor.shape[0]
        res=np.zeros((Nat,3),dtype='f8')
        res_env=np.zeros((env_Nat,3),dtype='f8')
        
        index1 = self.defects[indx].index
        charge1 = self.defects[indx].get_charge(state=charge_type)        
        charge = np.zeros(Nat,dtype='f8')
        charge_env = self.elstat.charge

        MASK = np.zeros(Nat,dtype="bool")
        MASK[index1] = True
        
        # Place charges on the defect (defect A)
        charge[index1] = charge1
        # zero charges for defect in the environment 
        charge_env[index1] = 0.0
        
        # calculation of tensors with interatomic distances for polarizability class
        P=np.zeros((Nat,3),dtype='f8')
        R,RR = self.diel.get_distance_matrixes()
        RR=RR+np.identity(Nat) # only for avoiding ddivision by 0 for diagonal elements    
        RR3=np.power(RR,3)
        RR5=np.power(RR,5)
        
        # calculate tensor with interactomic distances between environmnet and polarizability class atoms
        #R_env2pol=np.zeros((self.diel.Nat,env_Nat,3),dtype='f8') # mutual distance vectors
        R_pol = np.tile(self.diel.coor,(env_Nat,1,1))
        R_pol = np.swapaxes(R_pol,0,1)
        R_env = np.tile(self.elstat.coor,(Nat,1,1))
        R_env2pol = R_pol - R_env
        RR_env2pol = np.linalg.norm(R_env2pol,axis=2)
        RR3_env2pol = np.power(RR_env2pol,3)
        RR5_env2pol = np.power(RR_env2pol,5)
        for ii in range(Nat):
            RR3_env2pol[ii,ii] += 1.0
            RR5_env2pol[ii,ii] += 1.0
        
        # Initialize T tensor
        T=self.diel.get_T_tensor(R, RR, RR3, RR5) 
        
        # definition of T tensor between environment and the polarizability class
        T_pol2env=np.zeros((env_Nat,Nat,3,3),dtype='f8') # mutual distance vectors
        for ii in range(3):
            T_pol2env[:,:,ii,ii]=1/(RR3_env2pol.T)[:,:] -3*np.power(R_env2pol[:,:,ii],2).T/(RR5_env2pol.T)
            for jj in range(ii+1,3):
                T_pol2env[:,:,ii,jj] = -3*R_env2pol[:,:,ii].T*R_env2pol[:,:,jj].T/(RR5_env2pol.T)
                T_pol2env[:,:,jj,ii] = T_pol2env[:,:,ii,jj]
        for ii in range(Nat):
            T_pol2env[ii,ii,:,:]=0.0        # no self interaction of atom i with atom i
        
        # calculating derivation according to environment atom displacement
        Q=np.meshgrid(charge,charge)[0]   # in columns same charges
        ELF=np.zeros((Nat,Nat,3),dtype='f8')
        
        # calculate electric field generated by the first defect (defect A)
        for jj in range(3):
            ELF[:,:,jj]=(Q/RR3)*R[:,:,jj]   # ELF[i,j,:]  is electric field at position i generated by atom j 
        for ii in range(Nat):
            ELF[ii,ii,:]=np.zeros(3,dtype='f8')
        
        # calculate induced dipoles induced by the first defect (defect A)
        ELFV=np.array(np.sum(ELF,axis=1),dtype='f8')             # ELFV[i,:]   is electric field at position of atom i
        for ii in range(Nat):
            P[ii,:]=np.dot(self.diel.polar[poltype][ii],ELFV[ii,:])
        
        for ii in range(3):
            for n in range(Nat):
                if not MASK[n]:
                    res[n,ii] += np.dot(np.dot(charge_env,T_pol2env[:,n,ii,:]),P[n,:])
        
        ELFV=np.zeros((env_Nat,3),dtype='f8')
        for ii in range(3):
            for jj in range(3):
                ELFV[:,ii]+=np.dot(T_pol2env[:,:,ii,jj],P[:,jj])
        
        for ii in range(3):
            res_env[:,ii] -= charge_env * ELFV[:,ii]
        
        # calculate induced dipoles induced by the environment ESP atomic charges
        Q=np.meshgrid(charge_env,charge)[0]   # in columns same charges - in rows environment charges
        ELF=np.zeros((Nat,env_Nat,3),dtype='f8')
        for jj in range(3):
            ELF[:,:,jj]=( Q/(RR3_env2pol) )*R_env2pol[:,:,jj]   # ELF[i,j,:]  is electric field at position i generated by atom j 
        for ii in range(Nat):
            ELF[ii,ii,:]=np.zeros(3,dtype='f8')
        
        ELFV=np.array(np.sum(ELF,axis=1),dtype='f8')             # ELFV[i,:]   is electric field at position of atom i
        for ii in range(Nat):
            P[ii,:]=np.dot(self.diel.polar[poltype][ii],ELFV[ii,:])  # induced dipoles by environment charge distribution
        
        P[index1,:]=0.0 # just for sure
        
        for n in range(Nat):
            if not MASK[n]:
                for jj in range(len(index1)):
                    res[n,:]+=charge1[jj]*np.dot(T[index1[jj],n,:,:],P[n,:])
        
        
        # calculating derivation according to atom displacement from defect A
        ELFV=np.zeros((Nat,3),dtype='f8')
        for ii in range(3):
            for jj in range(3):
                ELFV[:,ii]+=np.dot(T[:,:,ii,jj],P[:,jj])
        for ii in range(len(index1)):
            res[index1[ii],:] -= charge1[ii]*ELFV[index1[ii],:]
        
        
        
        return res.reshape(3*Nat),res_env.reshape(3*env_Nat)
            
    
    def get_InteractionEnergyVacuum(self, indxA, indxB, chargeAtype='transition',
                                    chargeBtype='transition'):
        """ Calculate ESP interaction energy for defects (defect-like 
        molecules) in vacuum. Interaction energy calculated from point charges.
        TrEsp interaction energy is obtained when transition charges are used
        for both defects
        
        Parameters
        --------
        indxA : integer
            Index of the defect A
        indxB : integer
            Index of the defect B
        chargeAtype : string
            Which atomic charges are used for interaction energy calculation on
            defect A. Allowed types are ``'transition'``, ``'ground'`` and 
            ``'excited'`` 
        chargeBtype : string
            Which atomic charges are used for interaction energy calculation on
            defect B. Allowed types are ``'transition'``, ``'ground'`` and 
            ``'excited'`` 
            
        Returns
        --------
        E_TrEsp : Energy class
            TrEsp interaction energy with units management
        
        """
        
        defA = self.defects[indxA]
        defB = self.defects[indxB]
        
        defA_coor = self.diel.coor[defA.index]
        defB_coor = self.diel.coor[defB.index]
        defA_charge = defA.get_charge(state=chargeAtype)
        defB_charge = defB.get_charge(state=chargeBtype)

        E_Esp = charge_charge(defA_coor,defA_charge,defB_coor,defB_charge)[0]
        
        with energy_units("AU"):
            E_Esp = Energy(E_Esp) 
        
        return E_Esp
    
    def get_dipole(self, index, dipole_type='transition'):
        dipole_AU = self.defects[index].get_dipole(state=dipole_type)
        return dipole_AU
    
    
    def get_SingleDefectProperties(self, def_index, approx=1.1):
        ''' Calculate effects of environment such as transition energy shift
        and transition dipole change for single defect.
        
        Parameters
        ----------
        def_index : integer
            Index of the defect of interest
        approx : real (optional - init=1.1)
            Specifies which approximation should be used.
            
            * **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)` and 
              `Alpha(-E)`.
            * **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.
            * **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
              `Alpha(E)=Alpha(-E)`, however the second one is not condition 

        Returns
        -------
        Eshift : Energy class
            Transition energy shift for the defect due to the fluorographene
            environment calculated from structure with single defect. Units are
            energy managed
        TrDip : numpy array of real (dimension 3)
            Total transition dipole for the defect with environment effects 
            included calculated from structure with single defect (in ATOMIC 
            UNITS)
        
        **Neglecting `tilde{Beta(E)}` is not valid approximation. It should be
        better to neglect Beta(E,-E) to be consistent with approximation for 
        interaction energy**
        
        Notes
        ----------
        dip = Alpha(E)*El_field_TrCharge + Alpha(-E)*El_field_TrCharge 
        Then final transition dipole of molecule with environment is calculated
        according to the approximation:
        
        **CHANGE - INCORECT**
        **Approximation 1.1:**
            dip_fin = dip - (Vinter-DE)*Beta(E,E)*El_field_TrCharge + dip_init(1-1/4*Ind_dip_Beta(E,E)*El_field_TrCharge)
        **Approximation 1.2:**
            dip_fin = dip - (Vinter-DE)*Beta(E,E)*El_field_TrCharge + dip_init     
        **Approximation 1.3:**
            dip_fin = dip - 2*Vinter*Beta(E,E)*El_field_TrCharge + dip_init
        
        '''

        
        defect = self.defects[def_index]
        index = defect.index
        Nat = self.diel.Nat
        Nat_env = self.elstat.Nat
        
        # Get TrEsp Transition dipole - transition dipole for single defect
        TrDip_TrEsp = defect.get_dipole(state="transition")
        
#        with energy_units("AU"):
#            E01_int = defect.get_transition_energy()
        E01_int = defect.get_transition_energy()
        E01_int = E01_int._value
        
        # Set initial charges for defect
        charge_orig = self.diel.charge[index]
        tr_charge = defect.get_charge(state="transition")
        gr_charge = defect.get_charge(state="ground")
        ex_charge = defect.get_charge(state="excited")
        
        # Set initial charges for environment
        env_charge_orig = self.elstat.charge.copy()
        env_charge = self.elstat.charge.copy()
        self.elstat.charge[index] = ex_charge-gr_charge
        
        # Get electrostatic energy shift
        dAVA = self.elstat.get_EnergyShift() # dAVA = <A|V|A>-<G|V|G>
        self.elstat.charge[index] = tr_charge
        Eelstat_trans = self.elstat.get_EnergyShift() # Elstat inter of transition density with the environment ground state
        
        # return original environment charges back
        self.elstat.charge = env_charge_orig.copy()
        
        # Set distance matrix
        R_elst = np.tile(self.elstat.coor,(Nat,1,1))
        R_pol = np.tile(self.diel.coor,(Nat_env,1,1))
        R = (R_elst - np.swapaxes(R_pol,0,1))            # R[ii,jj,:]=self.coor[jj]-self.coor[ii]
        # TODO: Maybe also exclude connected fluorinesto atoms ii 
        for ii in range(Nat):
            if ii < Nat_env:
                R[ii,ii,:] = 0.0   # self interaction is not permited in potential calculation
        
        
        # Calculate polarization matrixes
        # TODO: Shift this block to separate function
        self._zero_induced_dipoles()
        self.diel.charge[index] = tr_charge
        self._induce_dipoles('AlphaE') #, order=2)
        Polar2_AlphaE = self.diel._get_interaction_energy(index,charge=tr_charge,debug=False)
        dip_AlphaE = np.sum(self.diel.dipole,axis=0)
        self._zero_induced_dipoles()
        self._induce_dipoles('AlphaE', order=1)
        Potential = potential_dipole(self.diel.dipole,R)
        E_Pol1_env_AE_tr = np.dot(env_charge,Potential)
                
        self._zero_induced_dipoles()
        self._induce_dipoles('Alpha_E') #, order=2)
        Polar2_Alpha_E = self.diel._get_interaction_energy(index,charge=tr_charge,debug=False)
        dip_Alpha_E = np.sum(self.diel.dipole,axis=0)
        self._zero_induced_dipoles()
        self._induce_dipoles('Alpha_E', order=1)
        Potential = potential_dipole(self.diel.dipole,R)
        E_Pol1_env_A_E_tr = np.dot(env_charge,Potential)
        
        self._zero_induced_dipoles()
        self.diel.charge[index] = ex_charge
        self._induce_dipoles('Alpha_st') #, order=2)
        Polar2_Alpha_st_ex = self.diel._get_interaction_energy(index,charge=ex_charge,debug=False)
        Potential = potential_dipole(self.diel.dipole,R)
        Polar2_env_Alpha_st_ex = np.dot(env_charge,Potential)
        # TODO: check if this should be used or not
        # dip_Alpha_st_ex = np.sum(self.diel.dipole,axis=0)
        self._zero_induced_dipoles()
        self.diel.charge[index] = gr_charge
        self._induce_dipoles('Alpha_st') #, order=2)
        Polar2_Alpha_st_gr = self.diel._get_interaction_energy(index,charge=gr_charge,debug=False)
        Potential = potential_dipole(self.diel.dipole,R)
        Polar2_env_Alpha_st_gr = np.dot(env_charge,Potential)
        # TODO: check if this should be used or not
        # dip_Alpha_st_gr = np.sum(self.diel.dipole,axis=0)
        
        # TODO: for pol2-env_static second order is twice and first order is only single times - therefore I need to calculate first and second order separately for environmnet efects
        self._zero_induced_dipoles()
        self.diel.charge[index] = ex_charge
        self._induce_dipoles('Alpha_st',order=1)
        dip1_Ast_ex = np.sum(self.diel.dipole,axis=0)
        Potential = potential_dipole(self.diel.dipole,R)
        Polar1_env_Alpha_st_ex = np.dot(env_charge,Potential)
        self._zero_induced_dipoles()
        self.diel.charge[index] = gr_charge
        self._induce_dipoles('Alpha_st',order=1)
        dip1_Ast_gr = np.sum(self.diel.dipole,axis=0)
        Potential = potential_dipole(self.diel.dipole,R)
        Polar1_env_Alpha_st_gr = np.dot(env_charge,Potential)
        self._zero_induced_dipoles()
        self.diel.charge[index] = tr_charge
        self._induce_dipoles('Alpha_st', order=1)
        Potential = potential_dipole(self.diel.dipole,R)
        Pol1_env_Alpha_st_tr = np.dot(env_charge,Potential)
        
        self._zero_induced_dipoles()
        self.diel.charge[index] = tr_charge
        self._induce_dipoles('BetaEE', order=1)
        dip_Beta = np.sum(self.diel.dipole,axis=0)
        Polar1_Beta_EE = self.diel._get_interaction_energy(index,charge=tr_charge,debug=False)
        
        # needed for transition dipole
        self._zero_induced_dipoles()
        self.diel.charge[index] = gr_charge
        self._induce_dipoles('AlphaE', order=1)
        dip1_AE_gr = np.sum(self.diel.dipole,axis=0)
        self._zero_induced_dipoles()
        self.diel.charge[index] = ex_charge
        self._induce_dipoles('Alpha_E', order=1)
        dip1_A_E_ex = np.sum(self.diel.dipole,axis=0)
  
        # Set the variables to initial state
        self._zero_induced_dipoles()
        self.diel.charge[index] = charge_orig
        
        if approx==1.1:
            # Calculate transition energy shift
            Eshift = dAVA + Polar2_AlphaE - Polar2_Alpha_E
            Eshift -= (self.diel.VinterFG - dAVA)*Polar1_Beta_EE
            Eshift += Polar2_Alpha_st_ex - Polar2_Alpha_st_gr
            Eshift += Polar1_env_Alpha_st_ex - Polar1_env_Alpha_st_gr
            Eshift += 2*(Polar2_env_Alpha_st_ex - Polar1_env_Alpha_st_ex - Polar2_env_Alpha_st_gr + Polar1_env_Alpha_st_gr)
            Eshift += Eelstat_trans/E01_int * (2*E_Pol1_env_AE_tr + 4*Pol1_env_Alpha_st_tr + 2*E_Pol1_env_A_E_tr)

            # Calculate transition dipoles for every defect
            TrDip = TrDip_TrEsp*(1 + Polar1_Beta_EE/2 - 2*(Eelstat_trans/E01_int)*(Eelstat_trans/E01_int) )
            TrDip += dip_AlphaE + dip_Alpha_E
            TrDip -= (self.diel.VinterFG - dAVA)*dip_Beta
            TrDip += (Eelstat_trans/E01_int)*(dip1_Ast_gr - dip1_Ast_ex)
            TrDip += (Eelstat_trans/E01_int)*(dip1_AE_gr - dip1_A_E_ex)
            # TODO: Add term for polarization of environment by environment itself
            
            # Change to energy class
            with energy_units('AU'):
                Eshift = Energy(Eshift)
                dAVA = Energy(dAVA)
            
                res_Energy = {'dE_0-1': Eshift, 'dE_elstat(exct-grnd)': dAVA}
                res_Energy['E_pol2_Alpha(E)'] = Energy(Polar2_AlphaE)
                res_Energy['E_pol2_Alpha(-E)'] = Energy(Polar2_Alpha_E)
                res_Energy['E_pol1_Beta(E,E)'] = Energy(Polar1_Beta_EE)
                res_Energy['E_pol2_static_(exct-grnd)'] = Energy(Polar2_Alpha_st_ex - Polar2_Alpha_st_gr)
                res_Energy['Pol1-env_static_(exct-grnd)'] = Energy(Polar1_env_Alpha_st_ex - Polar1_env_Alpha_st_gr)
                res_Energy['Pol2-env_static_(exct-grnd)'] = Energy(Polar2_env_Alpha_st_ex - Polar2_env_Alpha_st_gr)
                res_Energy['Pol1-env_Alpha(E)_(trans)'] = Energy(E_Pol1_env_AE_tr)
                res_Energy['Pol1-env_Alpha(-E)_(trans)'] = Energy(E_Pol1_env_A_E_tr)
                res_Energy['Pol1-env_static_(trans)'] = Energy(Pol1_env_Alpha_st_tr)
            

            return Eshift, res_Energy, TrDip
        else:
            raise IOError('Unsupported approximation')
    
    
    
    def get_SingleDefect_derivation(self, def_index, approx=1.1):
        ''' Calculate derivative of single defect property
        
        '''
        
        defect = self.defects[def_index]
        index = defect.index
        Nat = self.diel.Nat
        Nat_env = self.elstat.Nat
        
        with energy_units("AU"):
            E01_int = defect.get_transition_energy()
            E01_int = E01_int._value
        
        # Set initial charges for defect
        charge_orig = self.diel.charge[index]
        tr_charge = defect.get_charge(state="transition")
        gr_charge = defect.get_charge(state="ground")
        ex_charge = defect.get_charge(state="excited")
        
        # Set initial charges for environment
        env_charge_orig = self.elstat.charge.copy()
        self.elstat.charge[index] = ex_charge-gr_charge
        
        # Get electrostatic energy shift
        dAVA, dR_dAVA = self.elstat.get_EnergyShift_and_Derivative() # dAVA = <A|V|A>-<G|V|G>
        self.elstat.charge[index] = tr_charge
        Eelstat_trans = self.elstat.get_EnergyShift() # Elstat inter of transition density with the environment ground state
        
        # return original environment charges back
        self.elstat.charge = env_charge_orig.copy()
        
        # Set distance matrix - polarizable atoms x electrostatic atoms
        R_elst = np.tile(self.elstat.coor,(Nat,1,1))
        R_pol = np.tile(self.diel.coor,(Nat_env,1,1))
        R_pol_elst = (R_elst - np.swapaxes(R_pol,0,1))            # R[ii,jj,:]=self.coor[jj]-self.coor[ii]
        # TODO: Maybe also exclude connected fluorinesto atoms ii 
        for ii in range(Nat):
            R_pol_elst[ii,ii,:] = 0.0   # self interaction is not permited in potential calculation
        
        # Negative values are there because we want to calculate dE/dR and not d(El(Rn)*1/2*Alpha*El(Rn))/dR
        # calculate first order derivation - Polar1_Alpha(E)
        dR_pol1_AlphaE = -self._dR_BpA(def_index, def_index, 'AlphaE', 
                            chargeAtype='transition', chargeBtype='transition') 
        # calculate first order derivation - Polar1_Alpha(-E)
        dR_pol1_Alpha_E = -self._dR_BpA(def_index, def_index, 'Alpha_E', 
                            chargeAtype='transition', chargeBtype='transition')
        # calculate second order derivation - Polar2_Alpha(E)
        dR_pol2_AlphaE = -self._dR_BppA(def_index, def_index, 'AlphaE',
                            chargeAtype='transition', chargeBtype='transition')
        # calculate second order derivation - Polar2_Alpha(-E)
        dR_pol2_Alpha_E = -self._dR_BppA(def_index, def_index, 'Alpha_E',
                            chargeAtype='transition', chargeBtype='transition')
        # calculate first order derivation - Polar1_static for excited and ground charges
        dR_pol1_static_grnd = -self._dR_BpA(def_index, def_index, 'Alpha_st', 
                                    chargeAtype='ground', chargeBtype='ground')
        dR_pol1_static_exct = -self._dR_BpA(def_index, def_index, 'Alpha_st',
                                  chargeAtype='excited', chargeBtype='excited')
        # calculate second order derivation - Polar2_static for excited and ground charges
        dR_pol2_static_grnd = -self._dR_BppA(def_index, def_index, 'Alpha_st', 
                                    chargeAtype='ground', chargeBtype='ground')
        dR_pol2_static_exct = -self._dR_BppA(def_index, def_index, 'Alpha_st', 
                                  chargeAtype='excited', chargeBtype='excited')
        # calculate first order derivation - Polar1_Beta(E,E)
        dR_pol1_BetaEE = -self._dR_BpA(def_index, def_index, 'BetaEE',
                            chargeAtype='transition', chargeBtype='transition')
        # calculate first order derivation of Polar1-env with static polarizability
        self.elstat.charge[index] = 0.0
        dR_pol1_env_static_ex_gr, dR_pol1_env_static_ex_gr_env = \
            self._dR_ApEnv(def_index, 'Alpha_st', charge_type='excited-ground')
        dR_pol1_env_static_ex_gr = - dR_pol1_env_static_ex_gr
        dR_pol1_env_static_ex_gr_env = - dR_pol1_env_static_ex_gr_env 
        
        # this could be maybe left out:
        dR_pol1_env_AlphaE_tr, dR_pol1_env_AlphaE_tr_env = \
                self._dR_ApEnv(def_index, 'AlphaE', charge_type='transition')
        dR_pol1_env_AlphaE_tr = -dR_pol1_env_AlphaE_tr
        dR_pol1_env_AlphaE_tr_env = -dR_pol1_env_AlphaE_tr_env
        
        dR_pol1_env_Alpha_E_tr, dR_pol1_env_Alpha_E_tr_env = \
                self._dR_ApEnv(def_index, 'Alpha_E', charge_type='transition')
        dR_pol1_env_Alpha_E_tr = -dR_pol1_env_Alpha_E_tr
        dR_pol1_env_Alpha_E_tr_env = -dR_pol1_env_Alpha_E_tr_env
        dR_pol1_env_static_tr, dR_pol1_env_static_tr_env = \
                self._dR_ApEnv(def_index, 'Alpha_st', charge_type='transition')
        dR_pol1_env_static_tr = -dR_pol1_env_static_tr
        dR_pol1_env_static_tr_env = -dR_pol1_env_static_tr_env
        
        # calculate Beta polarizability 
        self.dipole = np.zeros((Nat,3),dtype='f8')
        self.diel.charge[index] = tr_charge
        self._induce_dipoles('BetaEE', order=1)
        Polar1_Beta_EE = self.diel._get_interaction_energy(index,charge=tr_charge,debug=False)
        
        # return environment charges back to initial state
        self.elstat.charge = env_charge_orig.copy()
        # Return polarization charges back to initial state
        self.diel.charge[index] = charge_orig
        
        # TODO: Add derivation of pol2-env
        # TODO: Split environment contribution and polarizable atoms contribution - both different dimensions

        if approx==1.1:
            # Calculate transition energy shift
            dR_Eshift_env = dR_dAVA
            dR_Eshift = (dR_pol1_AlphaE + dR_pol2_AlphaE)
            dR_Eshift -= (dR_pol1_Alpha_E + dR_pol2_Alpha_E)
            dR_Eshift -= (self.diel.VinterFG - dAVA)*dR_pol1_BetaEE
            dR_Eshift_env += dR_dAVA*Polar1_Beta_EE
            dR_Eshift += (dR_pol1_static_exct + dR_pol2_static_exct)
            dR_Eshift -= (dR_pol1_static_grnd + dR_pol2_static_grnd)
            dR_Eshift += dR_pol1_env_static_ex_gr
            dR_Eshift_env += dR_pol1_env_static_ex_gr_env

            # this could be maybe left out
            dR_Eshift += Eelstat_trans/E01_int * ( 2*dR_pol1_env_AlphaE_tr +
                                                     4*dR_pol1_env_static_tr +
                                                     2*dR_pol1_env_Alpha_E_tr)
            dR_Eshift_env += Eelstat_trans/E01_int * ( 2*dR_pol1_env_AlphaE_tr_env +
                                                     4*dR_pol1_env_static_tr_env +
                                                     2*dR_pol1_env_Alpha_E_tr_env)
            
            
            
#            Eshift += 2*(Polar2_env_Alpha_st_ex - Polar1_env_Alpha_st_ex - Polar2_env_Alpha_st_gr + Polar1_env_Alpha_st_gr)


            return dR_Eshift, dR_Eshift_env
        else:
            raise IOError('Unsupported approximation')
    
    
    def get_gmm(self, def_index, int2cart, freq, red_mass, approx=1.1):
        """ Calculate coupling strength of the site energy to atomic coordinates. 
        The reult is dimensionless coupling strength and resulting spectral
        density is defined as \sum_xi {gmm_xi*gmm_xi*\delta(omega-omega_xi)}
        
        Parameters
        ----------
        gr_charge : numpy array of real (dimension Natoms_defect)
            Ground state ESP charges for every atom from the defect
        ex_charge : numpy array of real (dimension Natoms_defect)
            Excited state ESP charges for every atom from the defect
        FG_elstat : Electrostatics class
            Electrostatic definition of the system (atomic charges, positions, 
            ...). It is possible to use it for calculation of electrostatic
            interaction energy between defect and environment
        struc : Structure class
            Structure definition of the molecule (needed for calculation of 
            derivative of the hamiltonian with respect to atomic coordinates).
        index : list of integer (dimension Natoms_defect)
            Indexes of all atoms from the defect (starting from 0)
        E01 : Energy class
            Transition energy of isolated defect without environment (calculated
            by quantum chemistry). Needed for calculation of derivative of
            hamiltonian with respect to atomic coordinates
        int2cart : numpy array of real (dimension 3*Nat x Nnormal_modes)
            transformation matrix from internal to cartesian coordinates.
            In columns there are normalized normal mode vectors in cartesian 
            coordinates ordered as [dx1,dy1,dz1,dx2,dy2,dz2,dx3,...]. Norm
            of the whole vector is 1.0 and it is dimensionless
        freq : numpy array of real (dimension Nnormal_modes)
            Wavenumbers of individual normal modes (frequency/speed of light 
            - default output from gaussian and AMBER - in both called frequency)
            in inverse centimeters 
        red_mass : numpy array of real (dimension Nnormal_modes)
            Reduced masses for every normal mode in AMU (atomic mass units)
        order : integer (optional - init = 2)
            Specify how many SCF steps shoudl be used in calculation of induced
            dipoles - according to the used model it should be 2
        CoarseGrain : string (optional - init = "C")
            Possible values are: "plane","C","CF" and "all_atom". Define which 
            level of coarse grained model should be used. If ``CoarseGrain="plane"``
            then all atoms are projected on plane defined by nvec and C-F atoms
            are treated as single atom - for this case polarizabilities defined
            only in 2D by two numbers. If ``CoarseGrain="C"`` then carbon atoms
            are center for atomic polarizability tensor and again C-F are treated
            as a single atom. If ``CoarseGrain="CF"`` then center of C-F bonds 
            are used as center for atomic polarizability tensor and again C-F 
            are treated as a single atom. If ``CoarseGrain="all_atom"`` all atoms
            are used as centers polarizability tensor.
        approx : real (optional - init=1.1)
            Specifies which approximation should be used.
            
            * **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)` and 
              `Alpha(-E)`.
            * **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.
            * **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
              `Alpha(E)=Alpha(-E)`, however the second one is not condition 
        
        """
        
        CoarseGrain = self.diel.coarse_grain
        
        dR_Hmm,dR_env_Hmm = self.get_SingleDefect_derivation(def_index, approx=approx)

        # CONVERT ALL TO ATOMIC UNITS (internal units)
        # freq is actualy wavenumber (= frequency/speed_of_light).
        # therefore angluar frequency omega in atomic units = 100/(2*Rydberg_inf) wavenumber in [cm-1]
        omega_au = freq/conversion_facs_energy["1/cm"]
        RedMass_au = red_mass/conversion_facs_mass["AMU"]
        # in atomic units hbar = 1.0, m_e = 1.0, elementary_charge = 1.0, 1/(4*pi*eps_0) = 1.0, speed_of_light = 137 ( fine-structure constant)
        
        # pick only carbon atoms from eigenvectors of normal modes (assume that fluorine atoms doesn't influent the result) - in needed
        if CoarseGrain in ["C","plane"] :
            at_type = []
            for ii in range(self.elstat.Nat):
                at_type.append(self.elstat.at_type[ii][0])
            indxC = np.where(np.array(at_type) == 'C')[0]
            index = np.zeros((len(indxC),3),dtype='i8')
            for ii in range(3):
                index[:,ii] = indxC*3+ii  # index*3+ii
            index=index.reshape(3*len(indxC))
            int2cart_loc = int2cart[index,:]
        else:
            int2cart_loc = int2cart.copy()
            
        g_mm = np.dot(int2cart_loc.T,dR_Hmm) + np.dot(int2cart.T,dR_env_Hmm)
        g_mm = g_mm/(np.sqrt(omega_au*omega_au*omega_au))
        g_mm = g_mm/(2*np.sqrt(RedMass_au))
        
        return g_mm
    
    def get_HeterodimerProperties(self, indxA, indxB, EngA = None, EngB = None, approx=1.1):
        ''' Calculate effects of the environment for structure with two different
        defects such as interaction energy, site transition energy shifts and 
        changes in transition dipoles

        Parameters
        ----------
        index1 : list of integer (dimension Natoms_defect1)
            Indexes of all atoms from the first defect (starting from 0)
        index2 : list of integer (dimension Natoms_defect2)
            Indexes of all atoms from the second defect (starting from 0)
        Eng1 : float 
            Vacuum transition energy of the first defect in ATOMIC UNITS (Hartree)
        Eng2 : float 
            Vacuum transition energy of the second defect in ATOMIC UNITS (Hartree)
        dAVA : float
            **dAVA = <A|V|A> - <G|V|G>** Difference in electrostatic 
            interaction energy between first defect the and environment for the 
            defect in excited state <A|V|A> and in ground state <G|V|G>.
        dBVB : float
            **dBVB = <B|V|B> - <G|V|G>** Difference in electrostatic 
            interaction energy between second defect and the environment for the 
            defect in excited state <B|V|B> and in ground state <G|V|G>.
        order : integer (optional - init = 80)
            Specify how many SCF steps shoudl be used in calculation of induced
            dipoles - according to the used model it should be 2
        approx : real (optional - init=1.1)
            Specifies which approximation should be used.
            
            * **Approximation 1.1**: Neglect of `Beta(-E,-E)` and `Beta(-E,E)` and 
              `Alpha(-E)`.
            * **Approximation 1.2**: Neglect of `Beta(-E,-E)` and `tilde{Beta(E)}`.
            * **Approximation 1.3**: `Beta(E,E)=Beta(-E,E)=Beta(-E,-E)` and also
              `Alpha(E)=Alpha(-E)`, however the second one is not condition 
        
        Returns
        -------
        J_inter : Energy class
            Interaction energy with effects of environment included. Units are 
            energy managed
        Eshift1 : Energy class
            Transition energy shift for the first defect due to fluorographene
            environment calculated from heterodymer structure. Units are energy
            managed
        Eshift2 : Energy class
            Transition energy shift for the second defect due to fluorographene
            environment calculated from heterodymer structure. Units are energy
            managed
        TrDip1 : numpy array of real (dimension 3)
            Total transition dipole for the first defect with environment effects 
            included calculated from heterodimer structure (in ATOMIC UNITS)
        TrDip2 : numpy array of real (dimension 3)
            Total transition dipole for the first defect with environment effects 
            included calculated from heterodimer structure (in ATOMIC UNITS)
        AllDipAE : numpy array of float (dimension Natoms x 3)
            Induced atomic dipole moments for all atoms in the environment by 
            the first defect with Alpha(E) atomic polarizability
        AllDipA_E : numpy array of float (dimension Natoms x 3)
            Induced atomic dipole moments for all atoms in the environment by 
            the first defect with Alpha(-E) atomic polarizability
        AllDipBE : numpy array of float (dimension Natoms x 3)
            Induced atomic dipole moments for all atoms in the environment by 
            the first defect with Beta(E,E) atomic polarizability
        '''

        res = {}
        defect1 = self.defects[indxA]
        defect2 = self.defects[indxB]
        index1 = defect1.index
        index2 = defect2.index
        Nat = self.diel.Nat
        Nat_env = self.elstat.Nat
        
        # Get transition energies
        E01_int1 = defect1.get_transition_energy()
        E01_int1 = E01_int1._value
        E01_int2 = defect2.get_transition_energy()
        E01_int2 = E01_int2._value
        eps_int = (E01_int1+E01_int2)/2.0
        
        # Set initial charges for defect
        charge_orig = self.diel.charge.copy()
        tr_charge1 = defect1.get_charge(state="transition")
        gr_charge1 = defect1.get_charge(state="ground")
        ex_charge1 = defect1.get_charge(state="excited")
        tr_charge2 = defect2.get_charge(state="transition")
        gr_charge2 = defect2.get_charge(state="ground")
        ex_charge2 = defect2.get_charge(state="excited")
        
        # Set initial charges for environment
        env_charge_orig = self.elstat.charge.copy()
        env_charge = self.elstat.charge.copy()
        
        if EngA is None and EngB is None: 
            # When site energies not specified, 
            # site energy = vacuum transition energy + electroctatic interaction
            # energy with the environment
            
            # Get electrostatic energy shift
            self.elstat.charge[index1] = ex_charge1-gr_charge1
            self.elstat.charge[index2] = gr_charge2
            dAVA = self.elstat.get_EnergyShift() # dAVA = <A|V|A>-<G|V|G>
            self.elstat.charge = env_charge_orig.copy()
            self.elstat.charge[index1] = gr_charge1
            self.elstat.charge[index2] = ex_charge2-gr_charge2
            dBVB = self.elstat.get_EnergyShift() # dBVB = <B|V|B>-<G|V|G>
            self.elstat.charge = env_charge_orig.copy()
            # dAVA and dBVB in internal atomic units
            
            # Get transition energies
            EngA = E01_int1
            EngB = E01_int2
        else:
            # When site energies are specified they are used without electrostatic
            # interaction shift, because all interactions with the environment 
            # are allready included in site energies.
            dAVA = 0.0
            dBVB = 0.0
            # FIXME:
            # For this case interaction energy should be calculated differently
            # and also eps_int sould be maybe (EngA+EngB)/2.0
        
        # Set distance matrix for interaction of defects with the environmnet 
        R_elst = np.tile(self.elstat.coor,(Nat,1,1))
        R_pol = np.tile(self.diel.coor,(Nat_env,1,1))
        R = (R_elst - np.swapaxes(R_pol,0,1))            # R[ii,jj,:]=self.coor[jj]-self.coor[ii]
        # if normaly ordered first are carbon atoms and then are fluorine atoms - for carbon atoms same indexes in pol_mol as in struc
# TODO: Maybe also exclude connected fluorinesto atoms ii 
        for ii in range(Nat):
            if ii < Nat_env:
                R[ii,ii,:] = 0.0   # self interaction is not permited in potential calculation
        
        # Get vaccuum interaction energies (V0100 != V0001) - ground state electron density symmetric to inversion and transition density antisymmetric to inversion - change of sign for some cases
        
        E_TrEsp = self.get_InteractionEnergyVacuum(indxA, indxB, 
                             chargeAtype='transition',chargeBtype='transition')
        VAB_0101 = E_TrEsp
        VAB_1101 = self.get_InteractionEnergyVacuum(indxA, indxB, 
                             chargeAtype='excited',chargeBtype='transition')
        VAB_0001 = self.get_InteractionEnergyVacuum(indxA, indxB, 
                             chargeAtype='ground',chargeBtype='transition')
        VAB_0111 = self.get_InteractionEnergyVacuum(indxA, indxB, 
                             chargeAtype='transition',chargeBtype='excited')
        VAB_0100 = self.get_InteractionEnergyVacuum(indxA, indxB, 
                             chargeAtype='transition',chargeBtype='ground')
        VAB_0000 = self.get_InteractionEnergyVacuum(indxA, indxB, 
                             chargeAtype='ground',chargeBtype='ground')
        VAB_1100 = self.get_InteractionEnergyVacuum(indxA, indxB, 
                             chargeAtype='excited',chargeBtype='ground')
        VAB_1111 = self.get_InteractionEnergyVacuum(indxA, indxB, 
                             chargeAtype='excited',chargeBtype='excited')
        VAB_0011 = self.get_InteractionEnergyVacuum(indxA, indxB, 
                             chargeAtype='ground',chargeBtype='excited')
        # transform energies to internal units
        VAB_0101 = VAB_0101._value
        VAB_1101 = VAB_1101._value
        VAB_0001 = VAB_0001._value
        VAB_0111 = VAB_0111._value
        VAB_0100 = VAB_0100._value
        VAB_0000 = VAB_0000._value
        VAB_1100 = VAB_1100._value
        VAB_1111 = VAB_1111._value
        VAB_0011 = VAB_0011._value
        
        # get electroctatic interaction energy of defects with environment 
        self.elstat.charge[index1] = gr_charge1
        EA_grnd=self.elstat.get_EnergyShift()
        self.elstat.charge[index1] = ex_charge1
        EA_exct=self.elstat.get_EnergyShift()
        self.elstat.charge[index1] = tr_charge1
        EA_trans=self.elstat.get_EnergyShift()
        self.elstat.charge = env_charge_orig.copy()
        
        self.elstat.charge[index2] = gr_charge2
        EB_grnd=self.elstat.get_EnergyShift()
        self.elstat.charge[index2] = ex_charge2
        EB_exct=self.elstat.get_EnergyShift()
        self.elstat.charge[index2] = tr_charge2
        EB_trans=self.elstat.get_EnergyShift()
        self.elstat.charge = env_charge_orig.copy()
        
        # Calculate polarization matrixes for the second order contributions
        self.diel.charge[index1] = tr_charge1
        self.diel.charge[index2] = tr_charge2
        PolarMat_AlphaE, dip_AlphaE1, dip_AlphaE2, AllDipAE1, AllDipAE2 = self.diel._fill_Polar_matrix(index1,index2,typ='AlphaE',order=self.order)
        PolarMat_Alpha_E, dip_Alpha_E1, dip_Alpha_E2, AllDipA_E1, AllDipA_E2 = self.diel._fill_Polar_matrix(index1,index2,typ='Alpha_E',order=self.order)
        self.diel.charge[index1] = gr_charge1
        self.diel.charge[index2] = ex_charge2
        PolarMat_Alpha_st_gr_ex, dip_Alpha_st1_gr, dip_Alpha_st2_ex, AllDipA_st1_gr, AllDipA_st2_ex = self.diel._fill_Polar_matrix(index1,index2,typ='Alpha_st',order=self.order)
        self.diel.charge[index1] = ex_charge1
        self.diel.charge[index2] = gr_charge2
        PolarMat_Alpha_st_ex_gr, dip_Alpha_st1_ex, dip_Alpha_st2_gr, AllDipA_st1_ex, AllDipA_st2_gr = self.diel._fill_Polar_matrix(index1,index2,typ='Alpha_st',order=self.order)
        
        res["E_pol2_A(E)"] = PolarMat_AlphaE
        res["E_pol2_A(-E)"] = PolarMat_Alpha_E

        PolarMat_Alpha_st = np.zeros((2,2),dtype='f8')
        PolarMat_Alpha_st[0,0] = np.sum(PolarMat_Alpha_st_ex_gr) # PolarMat_Alpha_st_ex_gr[0,0] + PolarMat_Alpha_st_ex_gr[1,1] + 2*PolarMat_Alpha_st_ex_gr[0,1]
        PolarMat_Alpha_st[1,1] = np.sum(PolarMat_Alpha_st_gr_ex) # PolarMat_Alpha_st_gr_ex[0,0] + PolarMat_Alpha_st_gr_ex[1,1] + 2*PolarMat_Alpha_st_gr_ex[0,1]
        
        # Add Alpha static pol-env contribution
        pot2_A_dipole_Alpha_st_gr = potential_dipole(AllDipA_st1_gr,R)
        pot2_A_dipole_Alpha_st_ex = potential_dipole(AllDipA_st1_ex,R)
        pot2_B_dipole_Alpha_st_gr = potential_dipole(AllDipA_st2_gr,R)
        pot2_B_dipole_Alpha_st_ex = potential_dipole(AllDipA_st2_ex,R)
        EA_Pol2_env_static_gr_FG = np.dot(env_charge,pot2_A_dipole_Alpha_st_gr)
        EA_Pol2_env_static_ex_FG = np.dot(env_charge,pot2_A_dipole_Alpha_st_ex)
        EB_Pol2_env_static_gr_FG = np.dot(env_charge,pot2_B_dipole_Alpha_st_gr)
        EB_Pol2_env_static_ex_FG = np.dot(env_charge,pot2_B_dipole_Alpha_st_ex)
        PolarMat_Alpha_st[0,0] = 2*( EA_Pol2_env_static_ex_FG + EB_Pol2_env_static_gr_FG )
        PolarMat_Alpha_st[1,1] = 2*( EA_Pol2_env_static_gr_FG + EB_Pol2_env_static_ex_FG )
        res["E_pol2_A_static"] = PolarMat_Alpha_st
        
        
        # first order electrostatic contribution
        ElstatMat_1 = np.zeros((2,2), dtype='f8')
        ElstatMat_1[0,0] = (EA_trans + VAB_0100)**2 - (EB_trans + VAB_1101)**2
        ElstatMat_1[1,1] = (EB_trans + VAB_0001)**2 - (EA_trans + VAB_0111)**2
        ElstatMat_1[0,1] = (EA_trans + VAB_0100)*(EB_trans + VAB_0001) - (EA_trans + VAB_0111)*(EB_trans + VAB_1101)
        ElstatMat_1[1,0] = ElstatMat_1[0,1]
        ElstatMat_1 = ElstatMat_1/eps_int
        res['E_elstat_1'] = ElstatMat_1
# TODO: This electrostatic contribution should be small print and see if it could be neglected
        
        # calculate polarization matrixes for contriutions containing only first order polarizabilities
        self.diel.charge[index1] = tr_charge1
        self.diel.charge[index2] = tr_charge2
        PolarMat_Beta, dip_Beta1, dip_Beta2, AllDipBE1, AllDipBE2 = self.diel._fill_Polar_matrix(index1,index2,typ='BetaEE',order=1)   
        PolarMat_Beta_scaled = ( (VAB_1100 - VAB_0000 + VAB_0011 - VAB_0000 + EA_exct - EA_grnd + EB_exct - EB_grnd)/2 - self.diel.VinterFG)*PolarMat_Beta
        res["E_pol2_B(E,E)"] = PolarMat_Beta
        res["E_pol2_B(E,E)_scaled"] = PolarMat_Beta_scaled
# TODO: Calculate and check contribution from d_epsilon
        
        # calculate contribution from 0-1 ground interaction with alpha(E) polarizability 
        self._zero_induced_dipoles()
        self.diel.charge[index1] = tr_charge1
        self.diel.charge[index2] = np.zeros(len(index2),dtype='f8')
        self._induce_dipoles('AlphaE',order=1)
        self.diel.charge[index1] = np.zeros(len(index1),dtype='f8')
        E_AB_pol1_tr_gr = self.diel._get_interaction_energy(index2,charge=gr_charge2,debug=False)
        E_A_pol1_tr_gr = self.diel._get_interaction_energy(index1,charge=gr_charge1,debug=False)
        Potential = potential_dipole(self.diel.dipole,R)
        E_A_pol1_env_tr = np.dot(env_charge,Potential)
        self._zero_induced_dipoles()
        self.diel.charge[index1] = np.zeros(len(index1),dtype='f8')
        self.diel.charge[index2] = tr_charge2
        self._induce_dipoles('AlphaE',order=1)
        self.diel.charge[index2] = np.zeros(len(index1),dtype='f8')
        E_AB_pol1_gr_tr = self.diel._get_interaction_energy(index1,charge=gr_charge1,debug=False)
        E_B_pol1_tr_gr = self.diel._get_interaction_energy(index2,charge=gr_charge2,debug=False)
        Potential = potential_dipole(self.diel.dipole,R)
        E_B_pol1_env_tr = np.dot(env_charge,Potential)
        self._zero_induced_dipoles()
        
        PolarMat_Alpha_tr_gr = np.zeros((2,2),dtype='f8')
        PolarMat_Alpha_tr_gr[0,0] = E_A_pol1_tr_gr + E_AB_pol1_tr_gr + E_A_pol1_env_tr
        PolarMat_Alpha_tr_gr[0,1] = E_B_pol1_tr_gr + E_AB_pol1_gr_tr + E_B_pol1_env_tr
        PolarMat_Alpha_tr_gr[1,0] = PolarMat_Alpha_tr_gr[0,0]
        PolarMat_Alpha_tr_gr[1,1] = PolarMat_Alpha_tr_gr[0,1]
        PolarMat_Alpha_tr_gr[0,:] = PolarMat_Alpha_tr_gr[0,:]*( EA_trans + VAB_0100 )/eps_int
        PolarMat_Alpha_tr_gr[1,:] = PolarMat_Alpha_tr_gr[1,:]*( EB_trans + VAB_0001 )/eps_int
        res["E_pol2_A(E)_(trans,grnd)"] = PolarMat_Alpha_tr_gr
        
        # calculate contribution from 0-1 ground and 0-1 excited interaction with alpha_static polarizability
        self._zero_induced_dipoles()
        self.diel.charge[index1] = tr_charge1
        self.diel.charge[index2] = np.zeros(len(index2),dtype='f8')
        self._induce_dipoles('Alpha_st',order=1)
        self.diel.charge[index1] = np.zeros(len(index1),dtype='f8')
        E_AB_st_pol1_tr_gr = self.diel._get_interaction_energy(index2,charge=gr_charge2,debug=False)
        E_AB_st_pol1_tr_ex = self.diel._get_interaction_energy(index2,charge=ex_charge2,debug=False)
        E_A_st_pol1_tr_gr = self.diel._get_interaction_energy(index1,charge=gr_charge1,debug=False)
        E_A_st_pol1_tr_ex = self.diel._get_interaction_energy(index1,charge=ex_charge1,debug=False)
        Potential = potential_dipole(self.diel.dipole,R)
        E_A_st_pol1_env_tr = np.dot(env_charge,Potential)
        self._zero_induced_dipoles()
        self.diel.charge[index1] = np.zeros(len(index1),dtype='f8')
        self.diel.charge[index2] = tr_charge2
        self._induce_dipoles('Alpha_st',order=1)
        self.diel.charge[index2] = np.zeros(len(index1),dtype='f8')
        E_AB_st_pol1_gr_tr = self.diel._get_interaction_energy(index1,charge=gr_charge1,debug=False)
        E_AB_st_pol1_ex_tr = self.diel._get_interaction_energy(index1,charge=ex_charge1,debug=False)
        E_B_st_pol1_tr_gr = self.diel._get_interaction_energy(index2,charge=gr_charge2,debug=False)
        E_B_st_pol1_tr_ex = self.diel._get_interaction_energy(index2,charge=gr_charge2,debug=False)
        Potential = potential_dipole(self.diel.dipole,R)
        E_B_st_pol1_env_tr = np.dot(env_charge,Potential)
        self._zero_induced_dipoles()
        
        PolarMat_static_tr_gr_ex = np.zeros((2,2),dtype='f8')
        PolarMat_static_tr_gr_ex[0,0] = (EA_trans + VAB_0100)/eps_int * (E_A_st_pol1_tr_ex + E_AB_st_pol1_tr_gr + E_A_st_pol1_env_tr)
        PolarMat_static_tr_gr_ex[1,0] = (EB_trans + VAB_0001)/eps_int * (E_A_st_pol1_tr_ex + E_AB_st_pol1_tr_gr + E_A_st_pol1_env_tr)
        PolarMat_static_tr_gr_ex[0,1] = (EA_trans + VAB_0100)/eps_int * (E_B_st_pol1_tr_ex + E_AB_st_pol1_gr_tr + E_B_st_pol1_env_tr)
        PolarMat_static_tr_gr_ex[1,1] = (EB_trans + VAB_0001)/eps_int * (E_B_st_pol1_tr_ex + E_AB_st_pol1_gr_tr + E_B_st_pol1_env_tr)
        PolarMat_static_tr_gr_ex[0,0] -= (EB_trans + VAB_1101)/eps_int * (E_AB_st_pol1_ex_tr + E_B_st_pol1_tr_gr + E_B_st_pol1_env_tr)
        PolarMat_static_tr_gr_ex[0,1] -= (EA_trans + VAB_0111)/eps_int * (E_AB_st_pol1_ex_tr + E_B_st_pol1_tr_gr + E_B_st_pol1_env_tr)
        PolarMat_static_tr_gr_ex[1,0] -= (EB_trans + VAB_1101)/eps_int * (E_AB_st_pol1_tr_ex + E_A_st_pol1_tr_gr + E_A_st_pol1_env_tr)
        PolarMat_static_tr_gr_ex[1,1] -= (EA_trans + VAB_0111)/eps_int * (E_AB_st_pol1_tr_ex + E_A_st_pol1_tr_gr + E_A_st_pol1_env_tr)
        res["E_pol1_A_static"] = PolarMat_static_tr_gr_ex
        
        
        # return charges to original values    
        self.elstat.charge = env_charge_orig.copy()
        self.diel.charge = charge_orig.copy()
        self._zero_induced_dipoles()

        # calculate new eigenstates and energies
        HH=np.zeros((2,2),dtype='f8')
        if EngA<EngA:
            HH[0,0] = EngA+dAVA
            HH[1,1] = EngB+dBVB
        else:
            HH[1,1] = EngA+dAVA
            HH[0,0] = EngB+dBVB
        HH[0,1] = E_TrEsp._value
        HH[1,0] = HH[0,1]
        EE,Coeff=np.linalg.eigh(HH)
        
        d_esp=np.sqrt( E_TrEsp._value**2 + ((EngB-EngA+dBVB-dAVA)/2)**2 )          # sqrt( (<A|V|B>)**2 + ((Eng2-Eng1+dBVB-dAVA)/2)**2  )
        
        
        # Calculate interaction energies
        if approx==1.1:
            # Calculate Total polarizability matrix
            PolarMat = PolarMat_AlphaE + PolarMat_Alpha_E + PolarMat_Alpha_st 
            PolarMat += PolarMat_Beta_scaled + ElstatMat_1 + 2*PolarMat_Alpha_tr_gr
            PolarMat += 2*PolarMat_static_tr_gr_ex
            
            # Calculate interaction energies
            C1 = Coeff.T[0]
            E1 = EE[0] + np.dot(C1, np.dot(PolarMat - d_esp*PolarMat_Beta, C1.T))
            
            C2 = Coeff.T[1]
            E2 = EE[1] + np.dot(C2, np.dot(PolarMat + d_esp*PolarMat_Beta, C2.T))
            
            J_inter = np.sqrt( (E2 - E1)**2 - (EngB - EngA)**2 )/2*np.sign(E_TrEsp._value)
            
            # Calculate energy shifts for every defect
            Eshift1 = dAVA + PolarMat_AlphaE[0,0] - PolarMat_Alpha_E[1,1]
            Eshift1 -= (self.diel.VinterFG - dAVA)*PolarMat_Beta[0,0]
            
            Eshift2 = dBVB + PolarMat_AlphaE[1,1] - PolarMat_Alpha_E[0,0]
            Eshift2 -= (self.diel.VinterFG - dBVB)*PolarMat_Beta[1,1]
            
            # Calculate transition dipoles for every defect
            TrDip1 = np.dot(tr_charge1,self.diel.coor[index1,:]) # vacuum transition dipole for single defect
            TrDip1 = TrDip1*(1 + PolarMat_Beta[0,0]/4) + dip_AlphaE1 + dip_Alpha_E1
            TrDip1 -= (self.diel.VinterFG - dAVA)*dip_Beta1
            
            TrDip2 = np.dot(tr_charge2,self.diel.coor[index2,:]) # vacuum transition dipole for single defect
            TrDip2 = TrDip2*(1 + PolarMat_Beta[1,1]/4) + dip_AlphaE2 + dip_Alpha_E2
            TrDip2 -= (self.diel.VinterFG - dBVB)*dip_Beta2
            
        
            # Change to energy class
            with energy_units('AU'):
                J_inter = Energy(J_inter)
                Eshift1 = Energy(Eshift1)
                Eshift2 = Energy(Eshift2)
                res["E_pol2_A(E)"] = Energy(res["E_pol2_A(E)"])
                res["E_pol2_A(-E)"] = Energy(res["E_pol2_A(-E)"])
                res["E_pol2_A_static"] = Energy(res["E_pol2_A_static"])
                res["E_pol2_B(E,E)_scaled"] = Energy(res["E_pol2_B(E,E)_scaled"])
                res["E_pol2_A(E)_(trans,grnd)"] = Energy(res["E_pol2_A(E)_(trans,grnd)"])
                res["E_pol1_A_static"] = Energy(res["E_pol1_A_static"])
                res["E_elstat_1"] = Energy(res["E_elstat_1"])
                res["E_pol2_B(E,E)"] = Energy(res["E_pol2_B(E,E)"])

            return J_inter, res #, Eshift1, Eshift2, TrDip1, TrDip2, AllDipAE1, AllDipA_E1, AllDipBE1 
        else:
            raise IOError('Unsupported approximation')
    
def _assign_phase(coor,Type,at_type=None):
    from ..QuantumChem.calc import GuessBonds
    
    Nat = len(coor)
    phase = np.zeros(Nat,dtype="f8")
    bonds = GuessBonds(coor,bond_length=4.0)
    connected = []
    for ii in range(Nat):
        connected.append([])
    
    for ii in range(len(bonds)):
        atom1 = bonds[ii][0]
        atom2 = bonds[ii][1]
        connected[atom1].append(atom2)
        connected[atom2].append(atom1)
    
    pairs = np.zeros((2,Nat),dtype='i8')
    for ii in range(Nat):
        pairs[:,ii] = [ii,connected[ii][0]] 
    
    vecs = coor[pairs[1]] - coor[pairs[0]]
    phase = np.arctan2(vecs[:,1],vecs[:,0])
#    for ii in range(20):
#            print(ii,np.rad2deg(phase[ii]),vecs[ii])
    
#    norm = np.linalg.norm(vecs,axis=1)
#    norm = np.tile(norm,(3,1))
#    nvecs = vecs / norm.T
#    
#    if Type in ["plane","C","CF"]: 
#        # calculate angle of interatom vector
#        phase = np.arctan2(vecs[:,1],vecs[:,0])
#        for ii in range(20):
#            print(ii,np.rad2deg(phase[ii]),vecs[ii])
#        
#        for ii in range(Nat):
#            nvec = nvecs[ii]
#            if (np.isclose(nvec[1],0,atol=1e-7) and np.isclose(nvec[0],0,atol=1e-7)) or np.isclose(abs(nvec[2]),1.0,atol=1e-4):
#                Phi=0.0
#            elif np.isclose(nvec[0],0,atol=1e-3) or np.isclose(abs(nvec[1]),1.0,atol=1e-3) :
#                if nvec[1]<0:
#                    Phi = -np.pi/2
#                else:
#                    Phi = np.pi/2
#            else:
#                #Phi=np.arctan(nvec[1]/nvec[0])
#                Phi=np.arctan2(nvec[1],nvec[0])
#            
#            Phi=np.arctan2(nvec[1],nvec[0])
#            phase[ii] = Phi
#            
#            if ii<20:
#                print(ii,np.rad2deg(Phi),nvec)
#                print(coor[ii],coor[connected[ii][0]])
            
    # For all atom calculation phase is so far set to zero 
    # FIXME: Calculate phase for all atom simulations
    return phase

def _prepare_polar_structure(struc,Type,verbose=False,use_bonds=False):
    """
    Type = "plane","C","CF","all_atom"
    
    """
    
    from ..QuantumChem.positioningTools import project_on_plane, fit_plane
    from ..QuantumChem.calc import GuessBonds
    
    if not Type in ["plane","C","CF","all_atom"]:
        raise Warning("Unsupported type of coarse graining.")
    
    if verbose:
        print(Type)
        
    indx_defects = constrainsFG(struc,border=False,defect=True)
    #connected = struc.get_bonded_atoms() # Would be faster if this is done only 
    
    # Assign pol types
    PolType=[]
    Polcharge=[]
    PolCoor=[]
    if Type == "plane" or Type == "C": 
        indx_C = np.where(np.array(struc.at_type)=="C")[0]
        NC = indx_C.shape[0]
        
        PolType = np.char.array(["CF"]*NC)
        PolType[indx_defects] = "C"
        PolCoor = struc.coor._value[indx_C]
        Polcharge = np.zeros(NC,dtype='f8')
    
        if Type == "plane":
            # project molecule whole system to plane defined carbon system
            nvec,origin = fit_plane(PolCoor)
            PolCoor=project_on_plane(PolCoor,nvec,origin)
            
    elif Type == "all_atom":
        indx_C = np.where(np.array(struc.at_type)=="C")[0]
        
        PolCoor = struc.coor._value.copy()
        PolType = np.char.array(["FC"]*struc.nat)
        PolType[indx_C] = "CF"
        PolType[indx_defects] = "C"
        Polcharge = np.zeros(struc.nat,dtype='f8')
        
    elif Type == "CF":
        indx_C = np.where(np.array(struc.at_type)=="C")[0]
        NC = indx_C.shape[0]
        connectivity = []
        for ii in range(struc.nat):
            connectivity.append([])
        if struc.bonds is None:
            struc.guess_bonds()
        for ii in range(len(struc.bonds)):
            indx1=struc.bonds[ii][0]
            at1=struc.at_type[indx1]
            indx2=struc.bonds[ii][1]
            at2=struc.at_type[indx2]
            if at1=="C" and at2=="F":    
                connectivity[indx1].append(indx2)
            elif at2=="C" and at1=="F":
                connectivity[indx2].append(indx1)
        
        PolType = np.char.array(["CF"]*NC)
        PolType[indx_defects] = "C"
        Polcharge = np.zeros(NC,dtype='f8')
        PolCoor = []
        for ii in range(struc.nat):
            if struc.at_type[ii]=='C':
                # polarizabiliy center will be located at center of C-F bond (or F-C-F for border carbons)
                count = 1
                position = struc.coor._value[ii]
                for jj in range(len(connectivity[ii])):
                    position += struc.coor._value[ connectivity[ii][jj] ]
                    count += 1
                position = position / count
                PolCoor.append(position)
        PolCoor=np.array(PolCoor,dtype='f8')
        
    if use_bonds:
        bonds = GuessBonds(PolCoor,bond_length=4.0)
        for ii in range(len(bonds)):
            indxA = bonds[ii][0]
            indxB = bonds[ii][1]
            rr = (PolCoor[indxA] + PolCoor[indxB])/2.0
            PolCoor = np.vstack((PolCoor,rr))
            PolType = np.append(PolType,'BOND')
        Polcharge = np.zeros(len(PolType),dtype='f8')
            
    PolPhase = _assign_phase(PolCoor,Type)
    
    return PolCoor,Polcharge,PolType,PolPhase


def _get_elstat_params(charge_type):
    if charge_type == 'Hirshfeld':
        CF_charge=0.08125
        CF2_charge=0.171217
    elif charge_type == "ESPfit":
        CF_charge = -0.0522
        CF2_charge = 2*CF_charge
    elif charge_type == 'CM5':
        CF_charge=0.0608
        CF2_charge=0.1352
    elif charge_type == 'zero':
        CF_charge=0.0
        CF2_charge=0.0
    else:
        raise Warning("Unknown type of charge.")
        
    FG_charges={'CF': CF_charge,'CF2': CF2_charge,'CD': 0.0,'C': 0.0}
    FG_charges['FC'] = -FG_charges['CF']
    FG_charges['F2C'] = -FG_charges["CF2"]/2.0
    
    return FG_charges

def _get_diel_params(charge_type,coarse_grain,approx,use_VinterFG,symm=False):
    
    AlphaE = np.zeros((3,3),dtype='f8')
    BetaE = np.zeros((3,3),dtype='f8')
    Alpha_E = np.zeros((3,3),dtype='f8')
    Alpha_st = np.zeros((3,3),dtype='f8')
    FAlphaE = np.zeros((3,3),dtype='f8')
    FBetaE = np.zeros((3,3),dtype='f8')
    FAlpha_E = np.zeros((3,3),dtype='f8')
    AlphaF_st = np.zeros((3,3),dtype='f8')
    ZeroM = np.zeros((3,3),dtype='f8')
    VinterFG=0.0
    func = None
    
    #print(coarse_grain,charge_type,use_VinterFG)
    
    if approx==1.1 and symm:
        if coarse_grain == "plane":
            if charge_type == 'Hirshfeld':
                AlphaE[0,0] = 6.9334582685
                AlphaE[1,1] = AlphaE[0,0]
                Alpha_E[0,0] = 1.23269450881
                Alpha_E[1,1] = Alpha_E[0,0]
                BetaE[0,0] = 0.00544878525234
                BetaE[1,1] = BetaE[0,0]
                VinterFG = 0.70082877
                func = 6.67459980317
            elif charge_type == 'ESPfit':
                AlphaE[0,0] = 7.01281780259
                AlphaE[1,1] = AlphaE[0,0]
                Alpha_E[0,0] = 0.525187717686
                Alpha_E[1,1] = Alpha_E[0,0]
                BetaE[0,0] = 0.328834053345
                BetaE[1,1] = BetaE[0,0]
                VinterFG = -0.23631988
                func = None
        elif coarse_grain == "C":
            if charge_type == "ESPfit":
                AlphaE[0,0] = 7.94759774636
                AlphaE[1,1] = AlphaE[0,0]
                AlphaE[2,2] = 3.21505490572
                Alpha_E[0,0] = 0.254952962056
                Alpha_E[1,1] = Alpha_E[0,0]
                Alpha_E[2,2] = 0.95693711133
                BetaE[0,0] = 0.256050965586
                BetaE[1,1] = BetaE[0,0]
                BetaE[2,2] = 0.98908237962
                VinterFG = 0.27908823332
                func = 5.9239635569
            if charge_type == 'Hirshfeld':
                AlphaE[0,0] = 7.55920919103
                AlphaE[1,1] = AlphaE[0,0]
                AlphaE[2,2] = 2.93279376194
                Alpha_E[0,0] = 0.84652632656
                Alpha_E[1,1] = Alpha_E[0,0]
                Alpha_E[2,2] = 0.86248644931
                BetaE[0,0] = 0.0085244841293
                BetaE[1,1] = BetaE[0,0]
                BetaE[2,2] = 1.02447353074
                VinterFG = 0.697280495448
                func = 6.637565434920049
                
    elif approx==1.1 and not symm:
        if coarse_grain == "plane":
            if charge_type == 'Hirshfeld':
                if not use_VinterFG:  # VinterFG = 0 (not included to fitting) (fun: 4.7181867436994045)
                    AlphaE[0,0]=7.3041890051 
                    AlphaE[1,1]=6.0241602626 
                    BetaE[0,0]=0.175182978222 
                    BetaE[1,1]= 0.103982443245 
                    Alpha_E[0,0]=0.0442315153377 
                    Alpha_E[1,1]=1.02933572591 
                    VinterFG=0.0
                    func = 4.7181867436994045
                else: # VinterFG included to fitting  (fun: 4.682778032798102)
                    AlphaE[0,0] = 7.478133264
                    AlphaE[1,1] = 6.046092258
                    BetaE[0,0] = 0.1592294
                    BetaE[1,1] = 0.0372992 
                    Alpha_E[0,0] = 0.01102153
                    Alpha_E[1,1] = 1.15007728
                    VinterFG = 0.54212481
                    func = 4.682778032798102
            elif charge_type == 'ESPfit':
                if not use_VinterFG:  # VinterFG = 0 (not included to fitting) (fun: 4.146799018918191)
                    AlphaE[0,0] = 7.53538330517 
                    AlphaE[1,1] = 6.50272559274 
                    BetaE[0,0] = 0.129161747387 
                    BetaE[1,1] = 0.0704009774178
                    Alpha_E[0,0] = 0.00737173071024
                    Alpha_E[1,1] = 0.505521019116
                    VinterFG = 0.0
                    func = 4.146799018918191
                else: # VinterFG included to fitting  (fun: 4.231930402706717)
                    AlphaE[0,0] = 7.666761627 
                    AlphaE[1,1] = 6.630444858 
                    BetaE[0,0] = 0.29803452 
                    BetaE[1,1] = 0.21500474 
                    Alpha_E[0,0] = 3.42354547e-04
                    Alpha_E[1,1] = 0.308054437
                    VinterFG = 0.929730408
                    func = 4.231930402706717
            elif charge_type == 'zero':
                if not use_VinterFG: # VinterFG = 0 (not included to fitting)
                    AlphaE[0,0] = 7.53252167197
                    AlphaE[1,1] = 6.33607655915
                    BetaE[0,0] = 0.0139090968952 
                    BetaE[1,1] = 0.0247518066045 
                    Alpha_E[0,0] = 0.00550346578145
                    Alpha_E[1,1] = 0.765093171532
                    VinterFG = 0.0
                else: # VinterFG included to fitting  (fun: 4.338780983932878)
                    AlphaE[0,0] = 7.57549868087 
                    AlphaE[1,1] = 6.34597075543
                    BetaE[0,0] = 0.0415620092596 
                    BetaE[1,1] = 0.333900795923 
                    Alpha_E[0,0] = 0.00152286902323
                    Alpha_E[1,1] = 0.768882724257
                    VinterFG = 0.148722203031
                    func = 4.338780983932878
            else:
                AlphaE[0,0]=7.09007854  
                AlphaE[1,1]=5.22989017 
                BetaE[0,0]=1.36615233 
                BetaE[1,1]= 0.32496846 
                Alpha_E[0,0]=0.0
                Alpha_E[1,1]=1.67121228
                VinterFG=0.0
            
            # Static polarizability / 2  (for energy shift of single chromophore alpha/2 is needed)
            Alpha_st[0,0]=2.30828107       # 4.61656214/2
            Alpha_st[1,1]=2.226315085      # 4.45263017/2
        
        elif coarse_grain == "C":
            if charge_type == 'ESPfit':
                if not use_VinterFG: # VinterFG = 0 (not included to fitting)
                    AlphaE[0,0] = 8.51210234419 
                    AlphaE[1,1] = 7.55308163377 
                    AlphaE[2,2] = 3.9652063838
                    Alpha_E[0,0] = 0.210269932532 
                    Alpha_E[1,1] = 0.498265152817 
                    BetaE[0,0] = 0.0618634656118 
                    BetaE[1,1] = 0.0387699553924
                    BetaE[2,2] = 0.34996428
                    VinterFG=0.0
                else: # VinterFG included to fitting  (fun: 4.303720899345645)
                    AlphaE[0,0] = 8.53950038458
                    AlphaE[1,1] = 7.49602283181
                    AlphaE[2,2] = 3.67719406572
                    Alpha_E[0,0] = 0.04505067863
                    Alpha_E[1,1] = 0.357157704568
                    Alpha_E[2,2] = 0.486405508174
                    BetaE[0,0] = 0.0752287116204 
                    BetaE[1,1] = 0.119328862239
                    BetaE[2,2] = 0.291115962688
                    VinterFG = 1.0
                    func = 4.303720899345645
            elif charge_type == 'Hirshfeld':
                if not use_VinterFG: # VinterFG = 0 (not included to fitting)
                    AlphaE[0,0] = 8.319204747
                    AlphaE[1,1] = 6.837981516
                    AlphaE[2,2] = 1.032203502
                    Alpha_E[0,0] = 0.00127885831
                    Alpha_E[1,1] = 1.17300541
                    BetaE[0,0] = 0.149163228
                    BetaE[1,1] = 0.000339873104
                    BetaE[2,2] = 0.310064431  
                    VinterFG=0.0
                else: # VinterFG included to fitting  (fun: 4.782278482925838)
                    AlphaE[0,0] = 8.45537962035
                    AlphaE[1,1] = 6.94911831601
                    AlphaE[2,2] = 4.06507435588
                    Alpha_E[0,0] = 0.0740550967058
                    Alpha_E[1,1] = 1.07705615554
                    Alpha_E[2,2] = 0.485345581054
                    BetaE[0,0] = 0.105673114257
                    BetaE[1,1] = 0.188114171514
                    BetaE[2,2] = 0.00657596713326 
                    VinterFG=0.775775109894     
                    func = 4.782278482925838
            elif charge_type == 'zero':
                if not use_VinterFG: # VinterFG = 0 (not included to fitting)
                    AlphaE[0,0] = 1.31648311 * 6.3
                    AlphaE[1,1] = 1.26990778 * 5.7
                    AlphaE[2,2] = 1.00051114 * 3.97
                    Alpha_E[0,0] = 0.21445315
                    Alpha_E[1,1] = 0.51064236
                    BetaE[0,0] = 0.149163228
                    BetaE[1,1] = 0.0625896
                    BetaE[2,2] = 0.34996506  
                    VinterFG=0.0
                else: # VinterFG included to fitting  (fun: 4.894367499008075) 
                    AlphaE[0,0] = 8.5263903736
                    AlphaE[1,1] = 7.55712544593
                    AlphaE[2,2] = 3.96990587054
                    Alpha_E[0,0] = 0.213862807965
                    Alpha_E[1,1] = 0.502103795464
                    Alpha_E[2,2] = 0.501654736106
                    BetaE[0,0] = 0.0333263826629
                    BetaE[1,1] = 0.259896505052
                    BetaE[2,2] = 6.97955987078 
                    VinterFG = 0.999766774316
                    func = 4.894367499008075
            else:
                AlphaE[0,0]=7.9009888
                AlphaE[1,1]=5.9624178
                AlphaE[2,2]=3.9745420
                BetaE[0,0]=1.3603038 
                BetaE[1,1]= 0.3168220 
                BetaE[2,2]= 0.3497237
                Alpha_E[0,0]=0.0
                Alpha_E[1,1]=1.67121228
                VinterFG=0.0

            # FIXME: This is from fitting coarse_grain="plane"            
            # Static polarizability / 2  (for energy shift of single chromophore alpha/2 is needed)
            Alpha_st[0,0]=2.30828107       # 4.61656214/2
            Alpha_st[1,1]=2.226315085      # 4.45263017/2
            
    else:
        raise Warning("Unknown type of approximation")
    
    polar = { 'CF': [AlphaE,Alpha_E,BetaE,Alpha_st], 'C': [ZeroM,ZeroM,ZeroM,ZeroM]}
    polar['FC'] = [FAlphaE,FAlpha_E,FBetaE,AlphaF_st]
    params_polar={"VinterFG": VinterFG,"coarse_grain": coarse_grain,
                  "polarizability": polar} 
    return params_polar
    

#    def _dRcB_BpA(self,index2,charge2,typ,c,eps=1):
#        ''' function which calculate derivation of interaction energy between defect
#        A and defect B defined by index2:
#        d/dRc^{(B)}[Sum_{n} E^{(B)}(Rn).(1/2*Polarizability(n)).E^{(A)}(Rn)]
#        
#        Parameters
#        ----------
#        index2 : list or numpy.array of integer (dimension N_def_atoms)
#            Atomic indexes of atoms which coresponds to defect B (defect with zero charges)
#        charge2 : numpy array of real (dimension N_def_atoms)
#            Vector of transition charge for every atom of defect B (listed in `index2`)
#        typ : str ('AlphaE','Alpha_E','BetaEE')
#            Specifies which polarizability is used for calculation of induced
#            atomic dipoles
#        c : integer
#            Atomic index specifying along which atom displacement should we calculate 
#            derivation
#        eps : real (optional - init=1.0)
#            Relative dielectric polarizability of medium where the dipoles and 
#            molecule is present ( by default vacuum with relative permitivity 1.0)
#
#        Notes
#        ----------
#        In initial structure transition charges are placed only on atoms from 
#        first defect (defect A defined by index1) and zero charges are placed 
#        on second defect (defect B defined by index2)
#        '''
#        
#        # check if atom with index c is in defect B
#        if c in index2:
#            c_indx=np.where(index2==c)[0][0]
#        else:
#            raise IOError('Defined index c is not in defect B')
#                
#        
#        R=np.zeros((self.diel.Nat,self.diel.Nat,3),dtype='f8') # mutual distance vectors
#        P=np.zeros((self.diel.Nat,3),dtype='f8')
#        for ii in range(self.diel.Nat):
#            for jj in range(ii+1,self.diel.Nat):
#                R[ii,jj,:]=self.coor[ii]-self.coor[jj]
#                R[jj,ii,:]=-R[ii,jj,:]
#        RR=np.sqrt(np.power(R[:,:,0],2)+np.power(R[:,:,1],2)+np.power(R[:,:,2],2))  # mutual distances
#        unit=np.diag([1]*self.diel.Nat)
#        RR=RR+unit   # only for avoiding ddivision by 0 for diagonal elements     
#        RR3=np.power(RR,3)
#        RR5=np.power(RR,5)
#        
#        # definition of T tensor
#        T=np.zeros((self.diel.Nat,self.diel.Nat,3,3),dtype='f8') # mutual distance vectors
#        for ii in range(3):
#            T[:,:,ii,ii]=1/RR3[:,:]-3*np.power(R[:,:,ii],2)/RR5
#            for jj in range(ii+1,3):
#                T[:,:,ii,jj] = -3*R[:,:,ii]*R[:,:,jj]/RR5
#                T[:,:,jj,ii] = T[:,:,ii,jj]
#        for ii in range(self.diel.Nat):
#            T[ii,ii,:,:]=0.0        # no self interaction of atom i with atom i
#        
#        # calculating derivation according to atom displacement from defect B
#        Q=np.meshgrid(self.charge,self.charge)[0]   # in columns same charges
#        ELF=np.zeros((self.diel.Nat,self.diel.Nat,3),dtype='f8')
#        
#        # Calculation of electric field generated by defect A
#        for jj in range(3):
#            ELF[:,:,jj]=(Q/RR3)*R[:,:,jj]   # ELF[i,j,:]  is electric field at position i generated by atom j 
#        for ii in range(self.diel.Nat):
#            ELF[ii,ii,:]=np.zeros(3,dtype='f8')
#            
#        # calculate induced dipoles induced by defect A
#        ELFV=np.array(np.sum(ELF,axis=1),dtype='f8')             # ELFV[i,:]   is electric field at position of atom i
#        for ii in range(self.diel.Nat):
#            P[ii,:]=np.dot(self.polar[typ][ii],ELFV[ii,:])
#        
#        ELFV=np.zeros((self.diel.Nat,3),dtype='f8')
#        for ii in range(3):
#            for jj in range(3):
#                ELFV[:,ii]+=np.dot(T[:,:,ii,jj],P[:,jj])
#        
## TODO: check if it shouldnt be res = - charge2[c_indx]*ELFV[c,:]
#        res=charge2[c_indx]*ELFV[c,:]
#        
#        return res
        
