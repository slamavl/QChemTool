# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:55:05 2017

@author: Vladislav Sláma
"""
import numpy as np
import os
import subprocess
from copy import deepcopy

from ...General.UnitsManager import position_units
from ..read_mine import read_amber_restart, read_AMBER_prepc, read_AMBER_NModes
from ..vibration import Proces_AMBER_Nmodes
from ..Classes.structure import Structure
from ..calc import rsmd
from ..positioningTools import SolveAngle

def deleteFG(struc,add_hydrogen=False,Hdist=2.0031,Fragmentize=False):
    """ Delete fluorographene atoms from structure and keeps only defect atoms.
    
    Parameters
    ----------
    struc : Structure class
        Contains information about structure of fluorographen sheet with defects
    add_hydrogen : logical (optional init = False)
        If hydrogens should be added to replace cutted bonds between defect 
        carbons and fluorographene carbons
    Hdist : float (optional init = 2.0031 Bohr)
        Distance between hydrogen and connected carbon atom in Bohr ( C-H distance).
    Fragmentize : logical (optional init = False)
        If ``True`` output will be list of separated structures corresponding 
        to individual defects. If ``Talse`` output will be single structure
        with all defects
    
    Returns
    --------
    new_struc : Structure class or list of Structure class
        Structure or list of structures with defects cutted from fluorographene
        with exactly the same coordinates as in fluorographene
    """
    
    if struc.bonds is None:
        struc.guess_bonds()
    is_FG=np.zeros(struc.nat,dtype='bool')
    for ii in range(len(struc.bonds)):
        Atom1=struc.bonds[ii,0]
        Atom2=struc.bonds[ii,1]
        if struc.at_type[Atom1]=='C' and struc.at_type[Atom2]=='F':
            is_FG[Atom1]=True
            is_FG[Atom2]=True
        elif struc.at_type[Atom1]=='F' and struc.at_type[Atom2]=='C':
            is_FG[Atom1]=True
            is_FG[Atom2]=True
    inxed_FG=np.arange(struc.nat)
    inxed_FG=list(inxed_FG[is_FG])
    
    new_struc=struc.delete_by_indx(indx=inxed_FG)
    if add_hydrogen:
        count=np.zeros(new_struc.nat,dtype='i8')
        new_struc.guess_bonds()
        for bond in new_struc.bonds:
            count[bond[0]] += 1
            count[bond[1]] += 1
        H_missing = (count==2)
        indx2addH = np.arange(new_struc.nat)[H_missing]   # index of atoms to which hydrogens will be added
        connections=[]
        for ii in range(new_struc.nat):
            connections.append([])
        for bond in new_struc.bonds:
            if bond[0] in indx2addH or bond[1] in indx2addH:                
                connections[bond[0]].append(bond[1])
                connections[bond[1]].append(bond[0])
        for ii in range(len(indx2addH)):
            indx0=indx2addH[ii]
            indx1=connections[indx0][0]
            indx2=connections[indx0][1]
            Hvec = 2*new_struc.coor._value[indx0] - new_struc.coor._value[indx1] - new_struc.coor._value[indx2]
            Hvec = Hvec*Hdist/np.linalg.norm(Hvec)
            Hvec = Hvec + new_struc.coor._value[indx0]
            with position_units('Bohr'):
                new_struc.add_atom(Hvec,'H')
    if Fragmentize:
        return new_struc.fragmentize()
    else:
        return new_struc

def constrainsFG(struc,border=True,defect=False):
    """ Output indexes of atoms to be frozen during geometry optimization for 
    fluorographene systems
    
    Parameters
    ----------
    struc : Structure class
        Contains information about structure of fluorographen sheet with defects
    border : logical (optional init = True)
        Specify if border carbon atoms (of fluorographene sheet) should be
        frozen or not (if should be included to constrain indexes)
    defect : logical (optional init = False)
        Specify if defect carbons (the one without fluorines) should be frozen
         or not (if should be included to constrain indexes). Usefull for 
         example for geometry optimization of single defect in geometry from
         two defect structure.
         
    Returns
    -------
    constrain_indx : list of integer (dimension Nconstrain)
        Indexes of atoms which should be frozen during geometry optimization
        (starting from 0).
    
    """
    
    if struc.bonds is None:
        struc.guess_bonds()
    
    # border carbons - conected to only 2 other carbons + cabons conected to these
    if border:
        constrains=np.zeros(struc.nat,dtype='bool')
        Nbonds = np.zeros(struc.nat,dtype='i8')
        connect_CC=[]
        connect_CF=[]
        for ii in range(struc.nat):
            connect_CC.append([])
            connect_CF.append([])
        for bond in struc.bonds:
            if struc.at_type[bond[0]]=='C' and struc.at_type[bond[1]]=='C':
                Nbonds[bond[0]] += 1
                Nbonds[bond[1]] += 1
                connect_CC[bond[0]].append(bond[1])
                connect_CC[bond[1]].append(bond[0])
            elif struc.at_type[bond[0]]=='C' and struc.at_type[bond[1]]=='F':
                connect_CF[bond[0]].append(bond[1])
                connect_CF[bond[1]].append(bond[0])
            elif struc.at_type[bond[0]]=='F' and struc.at_type[bond[1]]=='C':
                connect_CF[bond[0]].append(bond[1])
                connect_CF[bond[1]].append(bond[0])
            
        BorderC1=np.where(Nbonds==2)[0]
        
        for ii in BorderC1:
            constrains[ii]=True
            for jj in connect_CC[ii]:
                constrains[jj]=True
    
    # Defect carbons - carbons without fluorines
    if defect:
        for ii in range(struc.nat):
            if len(connect_CF[ii])==0:
                constrains[ii]=True
    
    constrain_indx=list(np.arange(struc.nat)[constrains])
    
    return constrain_indx

def get_border_carbons_FG(struc):
    """ Output indexes of carbons with two fluorines
    
    Parameters
    ----------
    struc : Structure class
        Contains information about structure of fluorographen sheet with defects
   
    Returns
    -------
    border_C_indx : list of integer (dimension Nborder_C)
        Indexes of carbons with two fluorine atoms
    border_F_indx : list of integer (dimension 2*Nborder_C)
        Fluorines connected carbons with two fluorines
    
    """
    
    if struc.bonds is None:
        struc.guess_bonds()
    
    # border carbons - conected to only 2 other carbons + cabons conected to these
    Nbonds = np.zeros(struc.nat,dtype='i8')
    connect_CC=[]
    connect_CF=[]
    for ii in range(struc.nat):
        connect_CC.append([])
        connect_CF.append([])
    for bond in struc.bonds:
        if struc.at_type[bond[0]]=='C' and struc.at_type[bond[1]]=='C':
            Nbonds[bond[0]] += 1
            Nbonds[bond[1]] += 1
            connect_CC[bond[0]].append(bond[1])
            connect_CC[bond[1]].append(bond[0])
        elif struc.at_type[bond[0]]=='C' and struc.at_type[bond[1]]=='F':
            connect_CF[bond[0]].append(bond[1])
            connect_CF[bond[1]].append(bond[0])
        elif struc.at_type[bond[0]]=='F' and struc.at_type[bond[1]]=='C':
            connect_CF[bond[0]].append(bond[1])
            connect_CF[bond[1]].append(bond[0])
        
    border_C_indx = list(np.where(Nbonds==2)[0])
    border_F_indx = []
    for ii in border_C_indx:
        for jj in range(len(connect_CF[ii])):
            border_F_indx.append(connect_CF[ii][jj])
    
    if len(border_F_indx) != 2*len(border_C_indx):
        raise Warning('Number of fluorine border atoms should be twice larger than carbon atoms.')
    
    
    return border_C_indx,border_F_indx

def create_AMBER_frcmod(filename,**kwargs):
    """ create frcmod file for AMBER MD simulation or normal mode analysis 
    of fluorographene cluster
    
    Parameters
    ----------
    filename : string
        Name of frcmod file (.frcmod ending)
    **kwargs : dictionary of dictionaries (optional):
        
        * **kwargs['equilibrium']** : dictionary
          Dictionary with equilibrium values of defied property. For example
          kwargs['equilibrium'] = {'c3-c3-c3': 90.0, 'cb-c3': 1.4} for setting
          the ``c3-c3-c3`` angle to 90 deg and ``bc-c3`` bond to 1.4 Angstrom. 
        * **kwargs['force']** : dictionary
          Dictionary with force constant for defined property.
    """
    
    param = default_AMBER_param.copy()
    if kwargs:
        equil=kwargs['equilibrium']
        force=kwargs['force']
        for ii in equil:
            param[ii][1] = equil[ii]
        for ii in force:
            param[ii][0] = force[ii]

    with open(filename, "wt") as f:
        # Vypis hlavicky
        f.write("remark goes here\n")
        f.write("MASS\n")
        f.write("cb 12.01         0.878               Same as c3\n")
        f.write("fb 19.00         0.320               Same as f\n")
        f.write("\n")
        f.write("BOND\n")
        f.write("cb-cb    {:8.4f}    {:8.6f}     Same as c3-c3 (for initial guess)\n".format( param['cb-cb'][0], param['cb-cb'][1] ))
        f.write("cb-fb    {:8.4f}    {:8.6f}     Same as c3-f (for initial guess)\n".format( param['cb-fb'][0], param['cb-fb'][1] ))
        f.write("cb-c3    {:8.4f}   {:8.6f}     Same as c3-c3 (for initial guess)\n".format( param['cb-c3'][0], param['cb-c3'][1] ))
        f.write("c3-c3    {:8.4f}    {:8.6f}     Same as c3-c3 (for initial guess)\n".format( param['c3-c3'][0], param['c3-c3'][1] ))
        f.write("c3-f     {:8.4f}   {:8.6f}     Same as c3-f (for initial guess)\n".format(  param['c3-f'][0], param['c3-f'][1] ))
        f.write("\n")
        f.write("ANGLE\n")
        f.write("cb-cb-cb   {:9.5f}     {:8.4f}  Same as c3-c3-c3 (for initial guess)\n".format( param['cb-cb-cb'][0], param['cb-cb-cb'][1] ))
        f.write("cb-cb-fb   {:9.5f}     {:8.4f}  Same as c3-c3-f (for initial guess)\n".format( param['cb-cb-fb'][0], param['cb-cb-fb'][1] ))
        f.write("cb-cb-c3   {:9.5f}     {:8.4f}  Same as c3-c3-c3 (for initial guess)\n".format( param['cb-cb-c3'][0], param['cb-cb-c3'][1] ))
        f.write("cb-c3-c3   {:9.5f}     {:8.4f}  Same as c3-c3-c3 (for initial guess)\n".format( param['cb-c3-c3'][0], param['cb-c3-c3'][1] ))
        f.write("cb-c3-cb   {:9.5f}     {:8.4f}  Same as c3-c3-c3 (for initial guess)\n".format( param['cb-c3-cb'][0], param['cb-c3-cb'][1] ))
        f.write("c3-cb-c3   {:9.5f}     {:8.4f}  Same as c3-c3-c3 (for initial guess)\n".format( param['c3-cb-c3'][0], param['c3-cb-c3'][1] ))
        f.write("cb-c3-f    {:9.5f}     {:8.4f}  Same as c3-c3-f (for initial guess)\n".format( param['cb-c3-f'][0], param['cb-c3-f'][1] ))
        f.write("c3-cb-fb   {:9.5f}     {:8.4f}  Same as c3-c3-f (for initial guess)\n".format( param['c3-cb-fb'][0], param['c3-cb-fb'][1] ))
        f.write("fb-cb-fb   {:9.5f}     {:8.4f}  Same as f -c3-f (for initial guess)\n".format( param['fb-cb-fb'][0], param['fb-cb-fb'][1] ))
        f.write("c3-c3-c3   {:9.5f}     {:8.4f}  from gaff_fgrph.dat\n".format( param['c3-c3-c3'][0], param['c3-c3-c3'][1] ))
        f.write("c3-c3-f    {:9.5f}     {:8.4f}  from gaff_fgrph.dat\n".format( param['c3-c3-f'][0], param['c3-c3-f'][1] ))
        f.write("f -c3-f    {:9.5f}     {:8.4f}  from gaff_fgrph.dat\n".format( param['f-c3-f'][0], param['f-c3-f'][1] ))
        
        # for now other parameters will reamin the same
        f.write("\n")
        f.write("DIHE\n")
        f.write("cb-cb-cb-cb   1    0.18        {:7.3f}          -3.         Same as c3-c3-c3-c3 (for initial guess)\n".format( param['cb-cb-cb-cb_3'][1] ))
        f.write("cb-cb-cb-cb   1    0.25        {:7.3f}          -2.         Same as c3-c3-c3-c3 (for initial guess)\n".format( param['cb-cb-cb-cb_2'][1] ))
        f.write("cb-cb-cb-cb   1    0.20        {:7.3f}           1.         Same as c3-c3-c3-c3 (for initial guess)\n".format( param['cb-cb-cb-cb_1'][1] ))
        f.write("cb-cb-c3-cb   1    0.18          0.0            -3.         Same as c3-c3-c3-c3 (for initial guess) - Section added by me START \n")
        f.write("cb-cb-c3-cb   1    0.25        180.0            -2.         Same as c3-c3-c3-c3 (for initial guess)\n")
        f.write("cb-cb-c3-cb   1    0.20        180.0             1.         Same as c3-c3-c3-c3 (for initial guess)\n")
        f.write("cb-c3-cb-cb   1    0.18          0.0            -3.         Same as c3-c3-c3-c3 (for initial guess)\n")
        f.write("cb-c3-cb-cb   1    0.25        180.0            -2.         Same as c3-c3-c3-c3 (for initial guess)\n")
        f.write("cb-c3-cb-cb   1    0.20        180.0             1.         Same as c3-c3-c3-c3 (for initial guess)\n")
        f.write("cb-c3-c3-cb   1    0.18          0.0            -3.         Same as c3-c3-c3-c3 (for initial guess)\n")
        f.write("cb-c3-c3-cb   1    0.25        180.0            -2.         Same as c3-c3-c3-c3 (for initial guess)\n")
        f.write("cb-c3-c3-cb   1    0.20        180.0             1.         Same as c3-c3-c3-c3 (for initial guess) - Section added by me END \n")
        f.write("cb-cb-cb-fb   9    1.400       {:7.3f}           3.000      Same as X -c3-c3-X (for initial guess)\n".format( param['cb-cb-cb-fb'][1] ))
        f.write("cb-cb-cb-c3   1    0.18          0.0            -3.         Same as c3-c3-c3-c3 (for initial guess)\n")
        f.write("cb-cb-cb-c3   1    0.25        180.0            -2.         Same as c3-c3-c3-c3 (for initial guess)\n")
        f.write("cb-cb-cb-c3   1    0.20        180.0             1.         Same as c3-c3-c3-c3 (for initial guess)\n")
        f.write("cb-cb-c3-c3   1    1.20         60.0             3.000      Same as c3-c3-c3-c3 (for initial guess)\n")
        f.write("cb-cb-c3-f    9    1.400       {:7.3f}           3.000      Same as X -c3-c3-X (for initial guess)\n".format( param['cb-cb-c3-f'][1] ))
        f.write("c3-cb-cb-fb   9    1.400       {:7.3f}           3.000      Same as X -c3-c3-X (for initial guess)\n".format( param['c3-cb-cb-fb'][1] ))
        f.write("c3-c3-cb-fb   9    1.400       {:7.3f}           3.000      Same as X -c3-c3-X (for initial guess)\n".format( param['c3-c3-cb-fb'][1] ))
        f.write("fb-cb-cb-fb   1    0.00          0.0            -3.         Same as f -c3-c3-f (for initial guess)\n")
        f.write("fb-cb-cb-fb   1    1.20        {:7.3f}           3.000      Same as f -c3-c3-f (for initial guess)\n".format( param['fb-cb-cb-fb'][1] ))
        f.write("fb-cb-c3-f    1    0.00          0.0            -3.         Same as f -c3-c3-f (for initial guess)\n")
        f.write("fb-cb-c3-f    1    1.20        {:7.3f}           3.000      Same as f -c3-c3-f (for initial guess)\n".format( param['fb-cb-c3-f'][1] ))
        f.write("cb-c3-cb-fb   1    1.20        {:7.3f}           3.000      (for initial guess)\n".format( param['cb-c3-cb-fb'][1] ))
        f.write("c3-cb-c3-f    1    1.20        {:7.3f}           3.000      (for initial guess)\n".format( param['c3-cb-c3-f'][1] ))
        f.write("cb-c3-cb-c3   1    0.20        {:7.3f}           3.000      (for initial guess)\n".format( param['cb-c3-cb-c3'][1] ))
        f.write("c3-cb-cb-c3   1    0.20        {:7.3f}           3.000      (for initial guess)\n".format( param['c3-cb-cb-c3'][1] ))
        f.write("c3-cb-c3-c3   1    0.20        {:7.3f}           3.000      (for initial guess)\n".format( param['c3-cb-c3-c3'][1] ))
        f.write("\n")
        f.write("IMPROPER\n")
        f.write("\n")
        f.write("NONBON\n")
        f.write("cb          1.9080  0.1094             Same as c3 (for initial guess)\n")
        f.write("fb          1.75    0.061              Same as f  (for initial guess)\n")
        f.write("\n")
        f.write("\n")
    f.close()
    
    
def create_AMBER_imput(struc,filename_frcmod,filename_prepc,filename_prmtop,state='Ground',**kwargs):
    
    directory = os.path.dirname(filename_frcmod)
    if directory is not "":
        directory += "/"
    
    # generate frcmod file
    create_AMBER_frcmod(filename_frcmod,**kwargs)
    
    #names of amber imput files
    filename_inpcrd = "".join([filename_prmtop[:-6],"inpcrd"])
    
    # names of auxiliary files
    pdb_filename = "".join([directory,"init_structure.pdb"])
    prepc_tmp_filename = "".join([directory,"init_structure_tmp.prepc"])
    struc.output_to_pdb(pdb_filename)
    
    # generate file with charges for every atom
    if state=='Ground':
        if struc.esp_grnd is None:
            raise Warning("Atomic charges has to be defined for all atoms before minimization")
        charges = struc.esp_grnd.copy()
    elif state=='Excited':
        if struc.esp_exct is None:
            raise Warning("Atomic charges has to be defined for all atoms before minimization")
        charges = struc.esp_exct.copy()
    elif state=='Transition':
        if struc.esp_trans is None:
            raise Warning("Atomic charges has to be defined for all atoms before minimization")
        charges = struc.esp_trans.copy()

    charge_filename = "".join([directory,"qin"])
    with open("qin", "wt") as f:
        for ii in range(struc.nat):
            if (ii+1)%8 == 0:
                f.write("{:10.6f}\n".format(charges[ii]))
            else:
                f.write("{:10.6f}".format(charges[ii]))
    
    # generate prepc file by antechamber - atoms will be reordered to be able to use nab for NM analysis
    cmd = "".join(["antechamber -i ",pdb_filename," -fi pdb -o ",prepc_tmp_filename," -fo prepc -rn FGR -c rc -cf ",charge_filename])
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", cmd])
    sp.communicate()
    
    # remove antechamber files
    os.system("rm *.AC *.AC0 *.INF")
    
    # reorder atoms according to prepc file and change FF types for border atoms
    struc_prepc = Structure()
    struc_prepc.load_prepc(prepc_tmp_filename,state=state)
    borderC_indx,borderF_indx = get_border_carbons_FG(struc_prepc)
    fid    = open(prepc_tmp_filename,'r')   # Open the file
    flines = fid.readlines()      # Read the WHOLE file into RAM
    fid.close()                   # Close the file
    
    for ii in borderC_indx:
        line = list(flines[ii+10])
        line[13] = "b"
        flines[ii+10] = "".join(line)
    for ii in borderF_indx:
        line = list(flines[ii+10])
        line[13] = "b"
        flines[ii+10] = "".join(line)
    
    with open(filename_prepc, "wt") as f:
        for ii in range(len(flines)):
            f.write(flines[ii])
    
    # generate tleap input file
    tlep_filename = "".join([directory,"tleap.in"])
    lib_filename = "".join([filename_prmtop[:-6],"lib"])
    with open(tlep_filename, "wt") as f:
        f.write("source leaprc.gaff \n")
        f.write("loadAmberPrep {:} \n".format(filename_prepc))
        f.write("loadamberparams {:} \n".format(filename_frcmod))
        f.write("saveoff FGR {:} \n".format(lib_filename))
        f.write("saveamberparm FGR {:} {:} \n".format(filename_prmtop,filename_inpcrd))
        f.write("savepdb FGR {:} \n".format("".join( [filename_prmtop[:-6],"pdb"] )))
        f.write("quit \n")
    
    # generate Amber imput files
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", "tleap -f tleap.in >> tleap.out"])
    sp.communicate()
    
##        Cindx = np.where(np.array(struc_prepc.at_type)=="C")
##        Ccoor = struc_prepc.coor._value[Cindx]
##        A_mat = Ccoor.copy()
##        A_mat[:,2] = 1.0
##        B_mat = Ccoor[:,2]
##        res = np.dot(np.dot(np.dot(A_mat.T,A_mat),A_mat.T),B_mat)
##        vec = np.array([res[0],res[1],1.0,res[2]])
##        vec = vec / np.linalg.norm(vec[0:3])
##        dist = np.dot(struc_prepc.coor._value,vec[0:3]) + vec[3]
##        borderF_indx = np.where(np.abs(dist)<=1.0 and np.array(struc_prepc.at_type)=="F")
#        # This is probably not needed. First check for border carbons (cb) and all connected fluorines would be fb
##        borderC_indx,borderF_indx = get_border_carbons_FG(struc_prepc)
##        struc_prepc.ff_type[borderC_indx] = 'cb'
##        struc_prepc.ff_type[borderF_indx] = 'fb'
##        struc_prepc.output_mol2(mol2_filename,state=state,ch_method="ESP",Name="FGR")
##        cmd = "".join(["antechamber -i ",mol2_filename," -fi mol2 -o ",prepc_filename," -fo prepc -rn FGR"])
##        sp = subprocess.Popen(["/bin/bash", "-i", "-c", cmd])
##        sp.communicate()
    
def Optimize_MD_AMBER_structure(filename,struc,state='Ground',prepc_filename=None,gen_input=False,struc_out=False,**kwargs):
    """ Optimize structure by AMBER MD software
    
    Parameters
    ----------
    filename : string
        Name of the frcmod file with aditional forcefield parameters
    struc : Structure class
        Imput structure before optimization
    state : string (optional init = 'Ground')
        Which charges should be used for MD structure optimization.
        If ``state='Ground'`` ground state charges located in 
        ``struc.esp_grnd`` are used (default). 
        If ``state='Excited'`` excited state charges located in 
        ``struc.esp_exct`` are used. If ``state='Transition'`` transition 
        charges located in ``struc.esp_trans`` are used.
    gen_input : logical (optional)
        If ``gen_input=True`` new imput files are generated for AMBER MD
        minimization. if ``gen_input=False`` old files for minimization 
        are used.
    struc_out : logical (optional)
        If ``True`` also optimized structure together with RMSD will be 
        outputed.
    **kwargs : dictionary of dictionaries (optional):
        
        * **kwargs['equilibrium']** : dictionary
          Dictionary with equilibrium values of defied property. For example
          kwargs['equilibrium'] = {'c3-c3-c3': 90.0, 'cb-c3': 1.4} for setting
          the ``c3-c3-c3`` angle to 90 deg and ``bc-c3`` bond to 1.4 Angstrom. 
        * **kwargs['force']** : dictionary
          Dictionary with force constant for defined property.
    
    Return
    -------
    RMSD : real
        Root mean square deviation in ATOMIC UNITS between input structure 
        and optimized structure by AMBER MD.
    struc_new : Structure class (optional)
        Struture class wit AMBER MD optimized and aligned structure (with
        input structure)
    
    """
    
    # check for requirements sander, antechamber, tleap, charges
#    try:
#        subprocess.call(["tleap", "-h"])
#    except OSError as e:
#        raise IOError("AmberTools required by this function are not present.")
#    
#    try:
#        subprocess.call(["sander","--version"])
#    except OSError as e:
#        raise IOError("Sander (part of AMBER) required by this function is not present.")
    
    directory = os.path.dirname(filename)
    if directory is not "":
        directory += "/"
    
    #names of amber imput files
    prmtop_filename = "".join([filename[:-6],"prmtop"])
    inpcrd_filename = "".join([filename[:-6],"inpcrd"])
    prepc_filename = "".join([directory,"init_structure.prepc"])
    
    # generate imput structure (optional)
    if gen_input:
        create_AMBER_imput(struc,filename,prepc_filename,prmtop_filename,state=state,**kwargs)           
        with open("01_Min.in", "wt") as f:
            f.write("minimize\n &cntrl\n  imin=1,\n  ntb=0,\n  ntx=1,\n  irest=0,\n")
            f.write("  maxcyc=10000,\n  ncyc=4000,\n  ntpr=100,\n  ntwx=10,\n")
            f.write("  cut=999.0,\n / \n")
    else:
        # generate frcmod file
        create_AMBER_frcmod(filename,**kwargs)
        
        # generate tleap imput file and MD minimization file
        tlep_filename = "".join([directory,"tleap.in"])
        lib_filename = "".join([filename[:-6],"lib"])
        with open(tlep_filename, "wt") as f:
            f.write("source leaprc.gaff \n")
            f.write("loadAmberPrep {:} \n".format(prepc_filename))
            f.write("loadamberparams {:} \n".format(filename))
            f.write("saveoff FGR {:} \n".format(lib_filename))
            f.write("saveamberparm FGR {:} {:} \n".format(prmtop_filename,inpcrd_filename))
            f.write("savepdb FGR {:} \n".format("".join( [filename[:-6],"pdb"] )))
            f.write("quit \n") 
        
        sp = subprocess.Popen(["/bin/bash", "-i", "-c", "tleap -f tleap.in >> tleap.out"])
        sp.communicate()
    
    # run AMBER structure optimization
    cmd =  "".join(["sander -O -i 01_Min.in -o 01_Min.out -p ",prmtop_filename," -c ",inpcrd_filename," -x 01_Min.mdcrd -r 01_Min.rst -inf 01_Min.mdinfo"])
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", cmd])
    sp.communicate()
    
    # read prepc file (or atom order)
    struc_prepc = Structure()
    struc_prepc.load_prepc(prepc_filename)
    
    # read restart file from amber minimization
    restartname = "".join([directory,'01_Min.rst'])
    struc_new = struc_prepc.copy()
    with position_units("Angstrom"):
        struc_new.coor.value = read_amber_restart(restartname)
    
    # center and align new geometry with the old one (trough SolveAngle)
    Rcom = struc_new.get_com()
    struc_new.move(-Rcom.value[0],-Rcom.value[1],-Rcom.value[2])
    Rcom = struc_prepc.get_com()
    struc_prepc.move(-Rcom.value[0],-Rcom.value[1],-Rcom.value[2])
    phi,psi,chi = SolveAngle(struc_prepc.coor._value,struc_new.coor._value,struc_prepc.mass,'minimize',50)
    struc_new.rotate_1(phi,psi,chi)
    
    # calculate RMSD
    RSMD = rsmd(struc_prepc.coor._value,struc_new.coor._value)
    
    # output RMSD (and optionaly also structure)
    if struc_out:
        return RSMD, struc_new, struc_prepc
    else:
        return RSMD
        

def get_AMBER_MD_normal_modes(struc,state='Ground',gen_input=False,**kwargs):
    # define file names 
    frcmod_filename = "nab_input.frcmod"
    prmtop_filename = "nab_input.prmtop"
    inpcrd_filename = "nab_input.inpcrd"
    prepc_filename = "nab_input.prepc"
    pdb_filename = "nab_input.pdb"
    pdb_out_filename = "nab_opt_structure.pdb"
    
    # create structure files - if needed - else create only tleap file and run tleap again to obtain prmtop file
    if gen_input:
        create_AMBER_imput(struc,frcmod_filename,prepc_filename,prmtop_filename,state=state,**kwargs)
    else:
        # generate frcmod file
        create_AMBER_frcmod(frcmod_filename,**kwargs)
        
        # generate tleap imput file and MD minimization file
        tlep_filename = "tleap.in"
        lib_filename = "nab_input.lib"
        with open(tlep_filename, "wt") as f:
            f.write("source leaprc.gaff \n")
            f.write("loadAmberPrep {:} \n".format(prepc_filename))
            f.write("loadamberparams {:} \n".format(frcmod_filename))
            f.write("saveoff FGR {:} \n".format(lib_filename))
            f.write("saveamberparm FGR {:} {:} \n".format(prmtop_filename,inpcrd_filename))
            f.write("savepdb FGR {:} \n".format(pdb_filename))
            f.write("quit \n") 
        
        sp = subprocess.Popen(["/bin/bash", "-i", "-c", "tleap -f tleap.in >> tleap.out"])
        sp.communicate()
    
    # create nab input file - optimization and normal mode analysis
    with open("nmode.nab", "wt") as f:
        f.write('molecule m; \n')
        f.write('float x[{:}], fret; \n\n'.format(struc.nat*3) )
        f.write('m = getpdb( "{:}"); \n'.format(pdb_filename) )
        f.write('readparm( m, "{:}" ); \n'.format(prmtop_filename) )
        f.write('mm_options( "cut=999.0, ntpr=50" ); \n')
        f.write('setxyz_from_mol(m, NULL, x); \n')
        f.write('mme_init( m, NULL, "::ZZZ", x, NULL); \n\n')
        f.write('//conjugate gradient minimization \n')
        f.write('conjgrad(x, 3*m.natoms, fret, mme, 0.001, 0.0001, 30000); \n\n')
        f.write('//Newton-Raphson minimization \n')
        f.write('mm_options( "ntpr=10" ); \n')
        f.write('newton( x, 3*m.natoms, fret, mme, mme2, 0.00000001, 0.0, 500 ); \n\n')
        f.write('//Output minimized structure \n')
        f.write('setmol_from_xyz( m, NULL, x); \n')
        f.write('putpdb( "{:}", m ); \n\n'.format(pdb_out_filename))
        f.write('//get the normal modes: \n')
        f.write('nmode( x, 3*m.natoms, mme2, 990, 0, 0.0, 0.0, 0); \n\n') 
        f.write('// instructions how to compile and run nab normal mode analysis. \n')
        f.write('// nab nmode.nab  \n')
        f.write('// ./a.out > freq.txt \n')
        f.write('// eigenvectors are stored in vecs file \n')
        f.write('// reduced masses and information about frequencies and minimization is stored in freq.txt \n')
    
    # run nab optimization and normal mode analysis
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", "nab nmode.nab"])
    sp.communicate()
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", "./a.out > freq.txt"])
    sp.communicate()
    
    struc_prepc = Structure()
    indx_orig = struc_prepc.load_prepc(prepc_filename)
    
    # find corespondence between original ordering of atoms and atoms in prepc file
    indx_struc = {}
    type_unq = np.unique(struc.at_type)
    for ii in type_unq:
        indx_struc[ii] = []
    for ii in range(struc.nat):
        indx_struc[struc.at_type[ii]].append( ii )
    
    indx_prepc = deepcopy(indx_struc)
    for ii in range(struc.nat):
        indx_prepc[struc_prepc.at_type[ii]][indx_orig[ii] - 1] = ii
        
    indx_orig2new = np.zeros(struc.nat,dtype='i8')

    for jj in indx_prepc.keys():
        for ii in range(len(indx_prepc[jj])):
            indx_orig2new[indx_prepc[jj][ii]] = indx_struc[jj][ii] 
    
    
    
    # organise files
    if not os.path.exists("Input_files"):
        os.makedirs("Input_files")
        
    if gen_input:
        os.rename("init_structure_tmp.prepc", "Input_files/init_structure_tmp.prepc")
        os.rename("init_structure.pdb", "Input_files/init_structure.pdb")
        os.rename("NEWPDB.PDB", "Input_files/NEWPDB.PDB")
        os.rename("qin", "Input_files/qin")
    os.rename("a.out", "Input_files/a.out")
    os.rename("leap.log", "Input_files/leap.log")
    os.rename("nab_input.inpcrd", "Input_files/nab_input.inpcrd")
    os.rename("nab_input.lib", "Input_files/nab_input.lib")
    os.rename("nab_input.prmtop", "Input_files/nab_input.prmtop")
    os.rename("nab_input.pdb", "Input_files/nab_input.pdb")
    os.rename("nmode.c", "Input_files/nmode.c")
    os.rename("nmode.nab", "Input_files/nmode.nab")
    os.rename("tleap.in", "Input_files/tleap.in")
    os.rename("tleap.out", "Input_files/tleap.out")
    
    # read nab results (maybe from hessian calculate normal modes)
    AM_geom,AM_Freq,AM_NormalModes = read_AMBER_NModes("vecs")
    with position_units("Angstrom"):
        struc_prepc.coor.value = AM_geom
    Freqcm1,RedMass,ForcesCm1Agstrom2,InternalToCartesian,CartesianToInternal,Units = Proces_AMBER_Nmodes(AM_geom,AM_Freq,AM_NormalModes,struc_prepc.mass)
    
    # output normal modes, frequencies, reduced mass ...
    NM_info = {}
    NM_info["int2cart"] = InternalToCartesian
    NM_info["cart2int"] = CartesianToInternal
    NM_info["freq"] = Freqcm1
    NM_info["RedMass"] = RedMass
    NM_info['force'] = ForcesCm1Agstrom2
    NM_info['units'] = {"freq": "1/cm", "RedMass": "AMU(atomic mass units)",
           "force": "1/(cm * Angstrom^2)", "int2cart": "dimensionles",
           'cart2int': "dimensionles"}
    return NM_info, indx_orig2new
    
    # write function for comparison of normal modes - gaussian vs AMBER
    
    # create separate function for creating AMBER input files
    
    # write function  for plotting histogram of vibrational frequencies
    
    print('test')

        
# Default gaff parameters
default_AMBER_param={'c3-c3': [303.1,1.5350], 'cb-cb': [303.1,1.5350], 
                     'cb-c3': [303.1,1.5350], 'c3-f': [363.8,1.3440],
                     'cb-fb': [363.8,1.3440 ], 'c3-c3-c3': [63.21,110.63],
                     'cb-c3-c3': [63.21,110.63], 'cb-cb-c3': [63.21,110.63],
                     'cb-c3-cb': [63.21,110.63], 'c3-cb-c3': [63.21,110.63],
                     'cb-cb-cb': [63.21,110.63], 'cb-cb-fb': [66.22,109.41],
                     'cb-c3-f': [66.22,109.41], 'c3-cb-fb': [66.22,109.41], 
                     'fb-cb-fb': [71.260,120.0], 'c3-c3-f': [66.22,109.41],
                     'f-c3-f': [71.260,210.0], 
                     'cb-cb-cb-cb_3': [0.18,0.0], 'cb-cb-cb-cb_2': [0.25,180.0],
                     'cb-cb-cb-cb_1': [0.20,180.0], 'cb-cb-cb-fb': [ 1.400,0.0],
                     'cb-cb-c3-f': [ 1.400,0.0], 'c3-cb-cb-fb': [ 1.400,0.0],
                     'c3-c3-cb-fb': [ 1.400,0.0], 'fb-cb-cb-fb': [ 1.2,180.0],
                     'fb-cb-c3-f': [ 1.2,180.0], 'cb-c3-cb-fb' : [ 1.2,60.0],
                     
                     'cb-c3-cb-c3': [ 0.2,0.0],'c3-cb-c3-f': [ 1.2,60.0],
                     'c3-cb-cb-c3': [ 0.2,0.0],'c3-cb-c3-c3': [ 1.2,0.0],
                     }










