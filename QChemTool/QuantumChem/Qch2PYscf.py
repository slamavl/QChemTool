# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:30:08 2016

@author: slamav
"""
# TODO: write documentation for the module itself

from pyscf import gto, scf
from pyscf.gto import mole 
import numpy as np
import scipy

from ..General.UnitsManager import position_units
from ..General.units import conversion_facs_position


def str2l(orbit):
    """
    Transform orbital type into orbital qauntum number: 's'->0, 'p'->1, ...
    
    Parameters
    ----------
    orbit : string 
        Orbital type as a string. For example `orbit` = 's', 'p', 'd', '5d', 
        'f', '7f'
    
    Returns
    ----------
    l_orbit : integer
        Orbital quantum number. For the same ordering of `orbit` orbital 
        quantum numbers will be: 0, 1, 2, 2, 3, 3 
    """
    
    if orbit=='s':
        return 0
    elif orbit=='p':
        return 1
    elif orbit=='d' or orbit=='5d':
        return 2
    elif orbit=='f' or orbit=='7f':
        return 3
            
def TransMat(orbit):
    """
    Transformation matrix for different ordering of orientations in AO basis 
    in PyScf and Gaussian. Works only for 5d and 7f orbitals (not d and f).
    In gaussian calculation has to be run with **5D 7F**  keyword to obtain this 
    basis function.
    
    Parameters
    ----------
    orbit : string 
        Orbital type as a string. Allowed `orbit` values are: 's', 'p', '5d' 
        and '7f'
        
    Returns
    ----------
    mat : numpy array of float
        Transformation matrix between bassis ordering from PyScf into ordering
        in Gaussian 09
    
    Notes
    ----------
    Matrix **M** calculated in PyScf orbital basis will be **np.dot(mat,np.dot(M,mat))**
    matrix in Gaussian 09 (5d 7f) orbital basis.     
    """
    
    if orbit=='s':
        return 1
    elif orbit=='p':
        return np.identity(3)
    elif orbit=='5d':
        mat=np.zeros((5,5))
        mat[0,2]=1.0
        mat[1,3]=1.0
        mat[2,1]=1.0
        mat[3,4]=1.0
        mat[4,0]=1.0
        return mat
    elif orbit=='7f':
        mat=np.zeros((7,7))
        mat[0,3]=1.0
        mat[1,4]=1.0
        mat[2,2]=1.0
        mat[3,5]=1.0
        mat[4,1]=1.0
        mat[5,6]=1.0
        mat[6,0]=1.0
        return mat
    elif orbit=='f':
        raise IOError("For PyScf software 'f' functions are not supported. Please"
                      "repeat the Gaussian calculation with '7F' keyword or do"
                      "not transform '7f' to 'f' orbitals before using PyScf")
    elif orbit=='d':
        raise IOError("For PyScf software 'd' functions are not supported. Please"
                      "repeat the Gaussian calculation with '5D' keyword or do"
                      "not transform '5d' to 'd' orbitals before using PyScf")

def molecule_to_PYscf_Pot(mol):
    """
    Transforms the molecule defined in `molecule` class into definition required
    for PyScf calculation of electrostatic potential. For this calculation extra
    `He` atom is added to the molecule. This atom will be then shifted and potential
    will be calculated at this point. There is no other way how to define point 
    for calculation of electrostatic potential in PyScf
    
    Parameters
    ----------
    mol : molecule class
        Molecule information which contains all nessesary information for 
        creating molecule with user defined basis on every atom in PyScf
    
    Returns
    ----------
    mol_scf : PyScf molecule
        Contains all information about molecule needed for calculation of
        electrostatic potential
    TrMat : numpy array of float
        Transition matrix from PyScf ordering of atomic orbitals into Gaussian 09
        ordering of atomic orbitals. More information about transformation matrix
        are in :func:`TransMat`.
    
    """
    
    print('Using molecule_to_PYscf_Pot')
    mol_scf = gto.Mole(unit = 'angstrom')    
    mol_scf.verbose = 0

# TODO: this part is completly the same as in previous function - it would be better to put this code to separate function than repeat it many times
#------------------------------------------------------------------------------------------------    
    # Nacteni atomu
    mol_scf.atom=[]
    with position_units("Angstrom"):
        for ii in range(mol.struc.nat):
            mol_scf.atom.append(["".join([mol.struc.at_type[ii],'@',str(ii)]),mol.struc.coor.value[ii]])
            
    i_atom_old=0
    # nacteni baze:
    for ii in range(mol.ao.nao):
        i_atom=mol.ao.atom[ii].indx
        AtomName="".join([mol.struc.at_type[i_atom],'@',str(i_atom)])
        if ii == 0:
            counter=0
            #mol_scf.basis = { AtomName: [[str2l(mol.ao.type[ii][0])]]}
            mol_scf.basis = { AtomName: [[str2l(mol.ao.type[ii])]]}
            for jj in range(len(mol.ao.exp[ii])):
                mol_scf.basis[AtomName][counter].append((mol.ao.exp[ii][jj],mol.ao.coeff[ii][jj]))
        else:
            if i_atom_old==i_atom:
                counter+=1
                # mol_scf.basis[AtomName].append([str2l(mol.ao.type[ii][0])])
                mol_scf.basis[AtomName].append([str2l(mol.ao.type[ii])])
                for jj in range(len(mol.ao.exp[ii])):
                    mol_scf.basis[AtomName][counter].append((mol.ao.exp[ii][jj],mol.ao.coeff[ii][jj]))
            else:
                counter=0
                #mol_scf.basis[AtomName]=[[str2l(mol.ao.type[ii][0])]]
                mol_scf.basis[AtomName]=[[str2l(mol.ao.type[ii])]]
                for jj in range(len(mol.ao.exp[ii])):
                    mol_scf.basis[AtomName][counter].append((mol.ao.exp[ii][jj],mol.ao.coeff[ii][jj]))                                   
        i_atom_old=np.copy(i_atom)
#-------------------------------------------------------------------------------------------------        
    mol_scf.atom.append(['He',np.array([1.23456789,1.23456789,1.23456789])*conversion_facs_position["Angstrom"]])    
    
    #building imput molecule for PYscf program
    mol_scf.build()
    mol_scf._atm[len(mol_scf._atm)-1,0]=1
    for ii in range(mol.ao.nao):
        if ii==0:
            TrMat=TransMat(mol.ao.type[ii])
        else:
            Mat=TransMat(mol.ao.type[ii])
            TrMat=scipy.linalg.block_diag(TrMat,Mat)
            
    return mol_scf,TrMat

def molecule_to_PYscf(mol):
    """
    Transforms the molecule defined in `molecule` class into definition required
    for PyScf. THIS CANNOT BE USED FOR CALCULATION OF ELECTROCTATIC POTENTIAL. 
    For this purpose use :func:`molecule_to_PYscf_Pot`
    
    Parameters
    ----------
    mol : molecule class
        Molecule information which contains all nessesary information for 
        creating molecule with user defined basis on every atom in PyScf
    
    Returns
    ----------
    mol_scf : PyScf molecule
        Contains all information about molecule needed for PyScf
    TrMat : numpy array of float
        Transition matrix from PyScf ordering of atomic orbitals into Gaussian 09
        ordering of atomic orbitals. More information about transformation matrix
        are in :func:`TransMat`.
    """
    
    print('Using molecule_to_PYscf')
    mol_scf = gto.Mole(units = 'Angstrom')    
    mol_scf.verbose = 3
    
    
    # Nacteni atomu
    mol_scf.atom=[]
    with position_units("Angstrom"):
        for ii in range(mol.struc.nat):
            mol_scf.atom.append([mol.struc.at_type[ii],mol.struc.coor.value[ii]])
    
    i_atom_old=0
    # nacteni baze:
    for ii in range(mol.ao.nao):
        i_atom=mol.ao.atom[ii].indx
        AtomName=mol.struc.at_type[i_atom]
        if ii == 0:
            counter=0
            mol_scf.basis = { AtomName: [[str2l(mol.ao.type[ii])]]}
            for jj in range(len(mol.ao.exp[ii])):
                mol_scf.basis[AtomName][counter].append((mol.ao.exp[ii][jj],mol.ao.coeff[ii][jj]))
        else:
            if i_atom_old==i_atom:
                counter+=1
                mol_scf.basis[AtomName].append([str2l(mol.ao.type[ii])])
                for jj in range(len(mol.ao.exp[ii])):
                    mol_scf.basis[AtomName][counter].append((mol.ao.exp[ii][jj],mol.ao.coeff[ii][jj]))
            else:
                counter=0
                mol_scf.basis[AtomName]=[[str2l(mol.ao.type[ii])]]
                for jj in range(len(mol.ao.exp[ii])):
                    mol_scf.basis[AtomName][counter].append((mol.ao.exp[ii][jj],mol.ao.coeff[ii][jj]))                                    
        i_atom_old=np.copy(i_atom)
    
    #building imput molecule for PYscf program
    mol_scf.build()
    for ii in range(mol.ao_spec['Nao']):
        if ii==0:
            TrMat=TransMat(mol.ao.type[ii])
        else:
            Mat=TransMat(mol.ao.type[ii])
            TrMat=scipy.linalg.block_diag(TrMat,Mat)
    
    
    return mol_scf,np.mat(TrMat)

def overlap_PYscf(mol):
    """
    Calculation of overlap matrix between atomic orbitals in spherical harmonics
    (5d and 7f). Correspond to Gaussian basis functions outputed with keywords 
    **5D 7F**
    
    Parameters
    ----------
    mol : molecule class
        Molecule information which contains all nessesary information for 
        creating molecule with user defined basis on every atom in PyScf
    
    Returns
    ----------
    over_transf : numpy array of float
        Overlap matrix between atomic orbitals in spherical harmonics (5d 7f).
        Atomic orbital ordering corresponds to the one in Gaussian 09
    
    """
    
    mol_scf,TMat=molecule_to_PYscf(mol)
    overlap=gto.getints('cint1e_ovlp_sph', mol_scf._atm, mol_scf._bas, mol_scf._env)
    over_transf=np.dot(TMat,np.dot(overlap,TMat.T))
    return over_transf


""" This is correctly described but use rather potential_basis_PYscf_grid"""
#def potential_basis_PYscf(mol,rr):
#    print('Transformation of molecule into PYscf')
#
#    mol_scf,TrMat=molecule_to_PYscf_Pot(mol)
#    
#    print('Molecule transformed')
#    
#    indx_min=mol_scf._atm[len(mol_scf._atm)-1,1]
#    mol_scf._env[indx_min]=np.copy(rr[0])
#    mol_scf._env[indx_min+1]=np.copy(rr[1])
#    mol_scf._env[indx_min+2]=np.copy(rr[2])
#    
#    # nyni vypocitat jadernou interakci dokud mam jeste info o nabojich
#    potential_nuc=0
#    q2 = mol_scf._atm[len(mol_scf._atm)-1,0]
#    r2 = np.copy(rr)
#
#    for i in range(len(mol_scf._atm)-1):
#        q1 = mol_scf._atm[i,0]
#        indx_env=mol_scf._atm[i,1]
#        r1 = np.zeros(3)
#        r1[0]=mol_scf._env[indx_env]
#        r1[1]=mol_scf._env[indx_env+1]
#        r1[2]=mol_scf._env[indx_env+2]
#        r = np.linalg.norm(r1-r2)
#        potential_nuc += q1 * q2 / r
#        
#    print('Nuclear potential calculated')
#    
#    for ii in range(len(mol_scf._atm)-1):
#        mol_scf._atm[ii,0]=0    
#    potential_el=gto.getints('cint1e_nuc_sph', mol_scf._atm, mol_scf._bas, mol_scf._env)
#    potential_el=np.dot(TrMat,np.dot(potential_el,TrMat.T))
#    
#    print('Electronic Potential calculated')
##    print(potential_el)
##    print(potential_nuc)
#    
#    return potential_el,potential_nuc # Matice Norb*Norb, cislo

def potential_basis_PYscf_grid(mol,points,M_mat,Msize=1):
    """
    Calculates molecular electrostatic potential on given grid.
    
    Parameters
    ----------
    mol : molecule class
        Molecule information which contains all nessesary information for 
        creating molecule with user defined basis on every atom in PyScf
    points : numpy array of float (dimension Npoints x 3)
        Positions in ATOMIC UNITS where potential will be calculated.
    M_mat : list of numpy arrays or numpy array
        Density matrix in atomic orbitals. For ground state: 
        ``M_mat[mu,nu]=Sum_{n_occ}{2*C_n,mu*C_n,nu}``. Where `C_n,mu` are expansion 
        coefficients of molecular orbital `n` into atomic orbital `mu` (contribution
        of atomic orbital `mu` in molecular orbital `n`). For transition density
        it would be ``M_mat[mu,nu]=Sum_{i->j}{c_ij*C_i,mu*C_j,nu}``. If more potential
        types should be calculated e.g. ground state, excited state, transition
        electrostatic potential ``M_mat=[M_mat_grnd,M_mat_exct,M_mat_tras]``
    Msize : integer
        Number of density matrixes imputed into ``M_mat``
        
    Return
    ----------
    potential_el_grid : numpy array of float ( dimension Npoints x Msize)
        Electronic electrostatic potential of the molecule in ATOMIC UNITS 
        (Hartree/e) for every point defined in `points` and every defined 
        density matrix `M_mat`(ordering is the same as ordering of `M_mat`) 
    potential_nuc_grid : numpy array of float ( dimension Npoints)
        Nuclear electrostatic potential of the molecule in ATOMIC UNITS 
        (Hartree/e) for every point defined in `points`. 
    
    Notes
    ----------
    Total ground and excited state electrostatic potential is sum of 
    electronic potential **potential_el_grid** and nuclear potential **potential_nuc_grid**.
    For transition electrostatic potential total potential is only electrostatic
    potential **potential_el_grid**.
    
    """

# TODO: Points should be maybe coordinate class and position units managed
    
    mol_scf,TrMat=molecule_to_PYscf_Pot(mol)
    
    print(len(points))
    indx_min=mol_scf._atm[len(mol_scf._atm)-1,1]
    
    # first calculate nuclear interaction energy
    potential_nuc_grid=np.zeros(len(points),dtype='f8')
    q2 = mol_scf._atm[len(mol_scf._atm)-1,0]
    for j in range(len(points)):
        r2 = np.copy(points[j,:])
        for i in range(len(mol_scf._atm)-1):
            q1 = mol_scf._atm[i,0]
            indx_env=mol_scf._atm[i,1]
            r1 = np.zeros(3)
            r1[0]=mol_scf._env[indx_env]
            r1[1]=mol_scf._env[indx_env+1]
            r1[2]=mol_scf._env[indx_env+2]
            r = np.linalg.norm(r1-r2)
            potential_nuc_grid[j] += q1 * q2 / r
    
    # then "delete" all other atoms than the testing one
    for ii in range(len(mol_scf._atm)-1):
        mol_scf._atm[ii,0]=0    
    
    
    # Then calculate electronic potential at every point
    if Msize==1:
        potential_el_grid=np.zeros(len(points))
        for j in range(len(points)):
            mol_scf._env[indx_min]=np.copy(points[j,0])
            mol_scf._env[indx_min+1]=np.copy(points[j,1])
            mol_scf._env[indx_min+2]=np.copy(points[j,2])
            potential_el=gto.getints('cint1e_nuc_sph', mol_scf._atm, mol_scf._bas, mol_scf._env)
            potential_el=np.dot(TrMat,np.dot(potential_el,TrMat.T))
            potential_el_grid[j]=np.trace(np.dot(potential_el,M_mat.T))
    elif Msize>1:
        potential_el_grid=np.zeros((len(points),Msize),dtype='f8')
        for j in range(len(points)):
            mol_scf._env[indx_min]=np.copy(points[j,0])
            mol_scf._env[indx_min+1]=np.copy(points[j,1])
            mol_scf._env[indx_min+2]=np.copy(points[j,2])
            potential_el=gto.getints('cint1e_nuc_sph', mol_scf._atm, mol_scf._bas, mol_scf._env)
            potential_el=np.dot(TrMat,np.dot(potential_el,TrMat.T))
            for ii in range(Msize):
                potential_el_grid[j,ii]=np.trace(np.dot(potential_el,M_mat[ii].T))
        
    return potential_el_grid,potential_nuc_grid





#==============================================================================
# TESTS
#==============================================================================
       
'''----------------------- TEST PART --------------------------------'''
if __name__=="__main__":
    
    def NormG(exp,index):
        res=(2*exp/np.pi)**(3/4)*((8*exp)**(np.sum(index))*scipy.special.factorial(index[0])*scipy.special.factorial(index[1])*scipy.special.factorial(index[2])/(scipy.special.factorial(2*index[0])*scipy.special.factorial(2*index[1])*scipy.special.factorial(index[2])))**(1/2)
        return res
    
    from ..General.functions import are_similar
    from ..QuantumChem.Classes.molecule import Molecule
    
    
    print(' ')
    print('TESTS')
    print('-------')

    MolDir='/mnt/sda2/Dropbox/PhD/Programy/Python/Test/'

    mol=Molecule('N7-Polyene')
    mol.load_Gaussian_fchk("".join([MolDir,'N7_exct_CAM-B3LYP_LANL2DZ_geom_CAM-B3LYP_LANL2DZ.fchk']))
    #mol.load_Gaussian_fchk('/mnt/sda2/PhD/Python/TestNewOverlapDip/NormalMode1/RRAstaxanthin_PDB_inv_TDDFT_wB97XD_6-31G(d)_NormalMode1_Real0.fchk')
    #mol.load_Gaussian_fchk('/mnt/sda2/PhD/Python/TestNewOverlapDip/N7_exct_CAM-B3LYP_LANL2DZ_geom_CAM-B3LYP_LANL2DZ.fchk')
    #mol.load_Gaussian_fchk('/mnt/sda2/PhD/Python/TestNewOverlapDip/AcrylicAcid_spherical.fchk')
    
    if 0:
        mol_scf=molecule_to_PYscf(mol)
        print(mol_scf.atom)
        #print(mol_scf.basis)
        overlap=gto.getints('cint1e_ovlp_sph', mol_scf._atm, mol_scf._bas, mol_scf._env)
        print(overlap[31,13:18])
        print(overlap[32,13:18])
        print(overlap[33,13:18])
    
    if 0:
        mol_scf=molecule_to_PYscf(mol)
        print(mol_scf.atom)
        #print(mol_scf.basis)
        SS_PYscf=gto.getints('cint1e_ovlp_sph', mol_scf._atm, mol_scf._bas, mol_scf._env)
        #SS_PYscf = mol_scf.intor('cint1e_ovlp_cart')
        mol.get_atomic_overlap()
        SS=mol.mol_spec['AO_overlap']
        if are_similar(SS_PYscf,SS,do_print=False) :
            print('Overlaps are THE same')
        
        for ii in range(len(mol.ao_spec['Type'])):
            #Mat=calc.get_cart2sph(mol.ao_spec['Type'][ii][0])
            Mat=mole.cart2sph(str2l(mol.ao_spec['Type'][ii][0]))
            if ii==0:
                Mc2s=np.copy(Mat)
            else:
                Mc2s=scipy.linalg.block_diag(Mc2s,Mat)
        
        SS_tr=np.dot(Mc2s.T,np.dot(SS,Mc2s))
        if are_similar(SS_PYscf,SS_tr,do_print=False) :
            print('Transposed cart ok')
        SS_PYscf_tr=np.dot(Mc2s,np.dot(SS_PYscf,Mc2s.T))
        if are_similar(SS_PYscf_tr,SS,do_print=False) :
            print('Transposed pyscf ok')
        
    if 0:
        from .qch_functions import norm_GTO 
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = 'CHO.log'
        mol.atom = [[6,(0, 0, 0)], ['H',(0, 1, 0)],['H',(0, -1, 0)], ['O',(0, 0, 1)]]
        mol.basis = { 'C': '6-31G','H': '6-31G','O': '6-31G'}
        mol.build()
        print(mol._atm)
        print(mol._bas)
        print(len(mol._env))
        jj=1
        for ii in range(len(mol._basis['C'][jj])-1):
            value=norm_GTO(np.zeros(3),mol._basis['C'][jj][ii+1][0],[0,0,0])
            print(value,NormG(mol._basis['C'][jj][ii+1][0],[0,0,0]),mole.gto_norm(0,mol._basis['C'][jj][ii+1][0]),mol._basis['C'][jj][ii+1][0],value*mol._basis['C'][jj][ii+1][1])
    
       
        
        
    if 0:
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = 'H2O.log'
        mol.atom = [[8,(0, 0, 0)], ['H',(0, 1, 0)], ['H@265',(0, 0, 1)]]
        #mol.atom = [[8,(0, 0, 0)], ['H',(0, 1, 0)], ['H',(0, 0, 1)]]
        #mol.basis = { 'O': 'ccpvdz','H': 'ccpvdz','H@265': '6-31G'}
        mol.basis = { 'O': 'ccpvdz','H': 'ccpvdz'}
        mol.basis['H@265']='6-31G'
        print(mol.atom)
        #mol.atom = '''GHOST:0 0 0; h:1 0 1 0;h: 0 0 1'''
        #mol.basis = { 'GHOST': gto.basis.load('ccpvdz', 'O'),'h': 'ccpvdz','h': 'ccpvdz'}
        mol.build()
        m = scf.RHF(mol)
        print('E(HF) = %g' % m.kernel())
        ovlp = mol.intor('cint1e_ovlp_cart')
        Vnuc_all = mol.intor('cint1e_nuc_cart')
        Vnuc_test=gto.getints('cint1e_nuc_cart', mol._atm, mol._bas, mol._env)
        if are_similar(Vnuc_test,Vnuc_all):
            print(' ')
            print('Two approaches provides same results')
        print(Vnuc_all[0,0])
        #print(ovlp)
        print(mol.basis)
        
        mol2 = gto.Mole()
        mol2.verbose = 5
        mol2.output = 'H2O.log'
        mol2.atom = [['GHOST',(0, 0, 0)], ['H',(0, 1, 0)], ['H@2',(0, 0, 1)]]
        print(mol2.atom)
        mol2.basis = { 'GHOST': gto.basis.load('ccpvdz', 'O'),'H': 'ccpvdz','H@2': '6-31G'}
        mol2.build()
        #Vnuc=gto.getints('cint1e_ipnuc_sph', mol2._atm, mol2._bas, mol2._env)
        Vnuc=mol2.intor('cint1e_nuc_cart')
        print(np.shape(Vnuc))
        m = scf.RHF(mol2)
        print('E(HF) = %g' % m.kernel())
        ovlp = mol2.intor('cint1e_ovlp_cart')
        #print(ovlp)
        print(mol2.basis)
        
        mol3 = gto.Mole()
        mol3.verbose = 5
        mol3.output = 'H2O.log'
        mol3.atom = [['O',(0, 0, 0)], ['H',(0, 1, 0)], ['H@2',(0, 0, 1)]]
        print(mol3.atom)
        mol3.basis = {'O': 'ccpvdz','H': 'ccpvdz','H@2': '6-31G'}
        print(mol3.basis['O'])
        mol3.build()
        print(mol3.basis)
        print(mol3._atm)
        print(mol3._bas)
        mol3._atm[0,0]=8
        mol3._atm[1,0]=0
        mol3._atm[2,0]=0
        Vnuc=gto.getints('cint1e_nuc_cart', mol3._atm, mol3._bas, mol3._env)
        #Vnuc=mol3.intor('cint1e_nuc_cart')
        print(Vnuc[0,0])
        mol3._atm[0,0]=0
        mol3._atm[1,0]=1
        Vnuc2=gto.getints('cint1e_nuc_cart', mol3._atm, mol3._bas, mol3._env)
        print(Vnuc2[0,0])
        mol3._atm[0,0]=0
        mol3._atm[1,0]=0
        mol3._atm[2,0]=1
        # Pokud pouzijme tento zpusob muzeme efektivne odstranit vsechny atomy a 
        Vnuc3=gto.getints('cint1e_nuc_cart', mol3._atm, mol3._bas, mol3._env)
        print(Vnuc3[0,0])
        print((Vnuc+Vnuc2+Vnuc3)[0,0])
        print(Vnuc_all[0,0])
        if are_similar(Vnuc+Vnuc2+Vnuc3,Vnuc_all):
            print(' ')
            print('Calculation of nuclear atraction energy is right')
        print(mol3._bas)
        
