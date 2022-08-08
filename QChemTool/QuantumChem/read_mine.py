import numpy
import operator
import pickle

from ..General.units import conversion_facs_position

from .Classes.general import Coordinate,Energy
from .Classes.atomic_orbital import AO,Atom,l_orient
from .Classes.molecular_orbital import MO
from .Classes.excitation import Excitation
from ..General.UnitsManager import energy_units,position_units
from .Classes.mdinfo import MDinfo   

# TODO: All coordinates should be position units managed

def read_gaussian_fchk(filename, **kwargs):
    ''' Reads usefull information from gaussian 09 fchk file - so far only for
    restricted calculations

    Parameters
    ----------
    filename : str
        Specifies the filename for the input file (including the path if needed)
    **kwargs : dictionary
        Specifies which density should be read from checkpoint file. If not present
        ground state density is read and stored in variable mo.densmat_grnd and if
        also excited state density is present it will be written in exct_dens
        If non optimized excited state density is needed \n
        **kwargs has to be specified as follows:
             **{'CI_rho_density': True}
  
    Returns
    -------
    struc : Structure class
        Contains the information about molecule structure: coordinates,
        atom types, charges... For more information look at the class documentation
    ao : AO class
        Contains the information about atomic orbitals: expansion coefficients
        into gaussian orbitals, exponents of the functions in expansion, orbital 
        types... For more information look at the class documentation
    mo : MO class
        Contains the information about molecular orbitals: expansion coefficients
        into atomic orbitals, energies of the orbitals, ground state density 
        matrix... For more information look at the class documentation
    TotalEnergy : Energy class
        Total ground state energy of the molecule 
    exct_dens : numpy array of float (dimension Nao_orient x Nao_orient)
        Excited state density matrix. If not present in the fchk file it will
        have zero length
    hess_fchk : numpy array of float (dimension 3Natom x 3Natom)
        Hessian matrix for the molecule in *Hartree/(Bohr^2)*. Second derivatives 
        of total energy in respect to coordinate displacements of individual
        atoms. If not present in the fchk file it will have zero length
    
    '''
    
    from .Classes.structure import Structure,get_atom_symbol
  
    fid    = open(filename,'r')   # Open the file
    flines = fid.readlines()      # Read the WHOLE file into RAM
    fid.close()                   # Close the file

    use_CI_rho_density=False

    for key in list(kwargs.keys()):
        if key=='CI_rho_density':
            use_CI_rho_density=kwargs['CI_rho_density']
  
    # Is this an unrestricted calculation?
    has_beta = False
    is_6D = False
    is_10F = False
    for line in flines:
        if 'beta mo coefficients' in line.lower():
            has_beta = True
        if 'Pure/Cartesian d shells' in line:
            is_6D = int(line.split()[-1]) == 1
        if 'Pure/Cartesian f shells' in line:
            is_10F = int(line.split()[-1]) == 1
    cartesian_basis = (is_6D and is_10F)


#    if spin is not None:
#        if spin != 'alpha' and spin != 'beta':
#            raise IOError('`spin=%s` is not a valid option' % spin)
#        elif has_beta:
#            print('Reading only molecular orbitals of spin %s.' % spin)
#        else:
#            raise IOError('The keyword `spin` is only supported for unrestricted calculations.')
    restricted = (not has_beta)
  
    # inicializace promennych
    sec_flag = None
  
    el_num = [0,0]
    mo_i0 = {'alpha': 0, 'beta': 0}
    what = 'alpha'
    index = 0
    at_num = 0
    ao_num = 0 
    
    ao=AO()
    struc=Structure()
    mo=MO()
    mo_spec=[]
    gr_dens=[]
    exct_dens=[]
    hess_fchk=[]
    
    # Set a counter for the AOs 
    basis_count = 0
    num_cons=None
  
    with energy_units('Ha'):
        with position_units('Bohr'):
            # Go through the file line by line 
            for il in range(len(flines)):
                line = flines[il]         # The current line as string
                thisline = line.split()   # The current line split into segments
                
                # Check the file for keywords 
                if 'Number of alpha electrons' in line:
                  el_num[0] = int(thisline[5])     # pocet alfa el.
                elif 'Number of beta electrons' in line:
                  el_num[1] = int(thisline[5])      # pocet beta el.
                elif 'Number of basis functions' in line:   
                  basis_number = int(thisline[5])   # pocet bazovych fci
                elif 'Atomic numbers'  in line:
                  sec_flag = 'geo_info'
                  index = 0
                  at_num = int(thisline[-1])    #pocet atomu
                  count = 0
                  ncharge=[]
                elif 'Nuclear charges' in line:
                  sec_flag = 'geo_info'
                  index = 2
                  at_num = int(thisline[-1])
                  count = 0
                elif 'Total Energy' in line:
                    TotalEnergy=Energy(float(thisline[-1]))
                elif 'Current cartesian coordinates' in line:
                  at_num = int(thisline[-1])/3
                  sec_flag = 'geo_pos'
                  coor=Coordinate(None)
                  count = 0
                  xyz = []
                elif 'Real atomic weights' in line:
                  sec_flag = 'at_wights' 
                  at_num = int(thisline[-1])
                  count=0
                  mass=numpy.zeros(at_num,dtype='f8')
                elif ('Shell types' in line) and (not ('Density' in line)):
                  sec_flag = 'ao_info'
                  index = 'type'
                  ao_num = int(thisline[-1])
                  count = 0
                  ao_types=[]
                elif ('Number of primitives per shell' in line)  and (not ('Density' in line)):
                  sec_flag = 'ao_info'
                  index = 'pnum'
                  ao_num = int(thisline[-1])
                  count = 0
                  ao_prim_number=[]
                elif ('Shell to atom map' in line) and (not ('Density' in line)):
                  sec_flag = 'ao_info'
                  index = 'atom'
                  ao_num = int(thisline[-1])
                  count = 0
                elif ('Primitive exponents' in line) and (not ('Density' in line)):
                  sec_flag = 'ao_exps'
                  ao_num = int(thisline[-1])
                  count = 0
                  index = 0
                  exps=[]
                elif ('Contraction coefficients' in line) and (not ('P(S=P) Contraction coefficients' in line) and (not ('Density' in line))):    #((not is_6D) and is_10F)
                  sec_flag = 'ao_coeffs'
                  ao_num = int(thisline[-1])
                  count = 0
                  index = 0
                  coeffs=[]
                elif ('P(S=P) Contraction coefficients' in line) and (not ('Density' in line)):
                  sec_flag = 'ao_coeffs_p'
                  ao_num = int(thisline[-1])
                  count = 0
                  index = 0
                  index_all = 0
                elif 'Density Number of contracted shells' in line:
                  sec_flag=None
                elif 'Coordinates of each shell' in line:
                  sec_flag=None
                elif 'Cartesian Force Constants' in line:
                  sec_flag='hessian'
                  num_cons=int(thisline[-1])
                  MatHess=numpy.zeros(num_cons)
                  count=0
                elif 'Orbital Energies' in line:
                  sec_flag = 'mo_eorb'
                  mo_num = int(thisline[-1])      
                  mo_i0[thisline[0].lower()] = len(mo_spec)
                  if restricted:
                      if el_num[0] == el_num[1]:
                          i = el_num[0]
                          occ = 2
                      else:
                          i = el_num[0 if 'Alpha' in line else 1]
                          occ = 1
                  else:
                      i = el_num[0 if 'Alpha' in line else 1]
                      occ = 1      
                  for ii in range(mo_num):
                      mo_spec.append({'coeffs': numpy.zeros(basis_number,dtype='f8'), #basis_count),
                                    'energy': 0.0,
                                    'occ_num': float(occ if ii < i else 0),
                                    'sym': '%i.1' % (ii+1),
                                    'spin':thisline[0].lower()
                                    })
                elif 'MO coefficients' in line:
                    sec_flag = 'mo_coeffs'
                    count = 0
                    index = 0
                    mo_num = int(thisline[-1])
                    what = thisline[0].lower()
                elif 'Total SCF Density' in line:
                    sec_flag = 'tot_scf_dens'
                    dens_num = int(thisline[-1])
                    if (basis_number*(basis_number+1)/2)!=dens_num:
                        raise IOError('basis_number does not represent number of basis functions')
                    gr_dens=numpy.zeros((basis_number,basis_number),dtype='f8')
                    denscount_line=0
                    denscount_row=0
                elif 'Total CI Density' in line:
                  sec_flag = 'tot_ci_dens'
                  dens_num = int(thisline[-1])
                  if (basis_number*(basis_number+1)/2)!=dens_num:
                      raise IOError('basis_number does not represent number of basis functions')
                  exct_dens=numpy.zeros((basis_number,basis_number),dtype='f8')
                  denscount_line=0
                  denscount_row=0
                elif ("Total CI Rho(1) Density" in line) and  use_CI_rho_density:
                  # This density is more precise than
                  sec_flag = 'tot_scf_dens'
                  dens_num = int(thisline[-1])
                  if (basis_number*(basis_number+1)/2)!=dens_num:
                      raise IOError('basis_number does not represent number of basis functions')
                  gr_dens=numpy.zeros((basis_number,basis_number),dtype='f8')
                  denscount_line=0
                  denscount_row=0
                elif sec_flag=='tot_scf_dens' and 'Charges' in line:
                  sec_flag = None
                else:
                  if sec_flag == 'geo_info':
                      if index==2:
                          for ii in thisline:
                              ncharge.append(int(float(ii)))
                              count += 1
                              if count == at_num:
                                sec_flag = None
                  elif sec_flag == 'geo_pos':
                    for ii in thisline:
                      xyz.append(float(ii))
                      if len(xyz) == 3:
                          coor.add_coor(numpy.array(xyz,dtype='f8'))
                          xyz = []
                          count += 1
                          if count == at_num:
                              sec_flag = None
                  elif sec_flag == 'at_wights':
                    for ii in thisline:
                        mass[count]=float(ii)
                        count+=1
                        if count == at_num:
                          sec_flag = None
                  elif sec_flag == 'ao_info':
                      for ii in thisline:
# TODO: Correct changing ii - correct but not nice
                          ii = int(ii)
                          if index is 'type':
                              ao_types.append(ii)
                              if ii == -1:
                                  ii='sp'
                                  ao.type.append('s')
                                  ao.type.append('p')
                                  l = lquant['p']
                                  basis_count += l_deg(l,cartesian_basis=True)  # za p orbital z sp hybridizace
                                  basis_count += 1                                         # za s orbital z sp hybridizace
                              elif ii == -2:
                                  ii='5d'
                                  ao.type.append('5d')
                                  basis_count += l_deg(ii,cartesian_basis=True)
                              elif ii == -3:
                                  ii='7f'
                                  ao.type.append('7f')
                                  basis_count += l_deg(ii,cartesian_basis=True)
                              else:
                                  ao.type.append(orbit[abs(ii)])
                                  ii = orbit[abs(ii)]
                                  l = lquant[ii]
                                  basis_count += l_deg(l,cartesian_basis=True)
                                  #print(ii,l_deg(l,cartesian_basis=True),basis_count)
#                              if not cartesian_basis:
#                                  for m in (range(0,l+1) if l != 1 else [1,0]):
#                                    qc.ao_spherical.append([count,(l,m)])
#                                    if m != 0:
#                                      qc.ao_spherical.append([count,(l,-m)])
                          elif index is 'atom':
                              if ao_types[count]==-1:
                                  ao.atom.append(Atom(get_atom_symbol(ncharge[ii-1]),ii-1))  # for sp orbital we have to add two atoms (one for s-orbital and one for p-orbital)
                                  ao.atom.append(Atom(get_atom_symbol(ncharge[ii-1]),ii-1))
                              else:
                                  ao.atom.append(Atom(get_atom_symbol(ncharge[ii-1]),ii-1))
                          elif index is 'pnum':
                              ao_prim_number.append(ii)
#                          qc.ao_spec[count][index] = ii
                          count += 1
                          if count == ao_num:
                            sec_flag = None
                  elif sec_flag == 'ao_exps':
                      for ii in thisline:
                          exps.append(float(ii))
                          count += 1
                          if count==ao_prim_number[index]:
                              if ao_types[index]==-1:        # for sp orbital we have the same exponents 
                                  ao.exp.append(numpy.array(exps,dtype='f8'))
                                  ao.exp.append(numpy.array(exps,dtype='f8'))
                              else:
                                  ao.exp.append(numpy.array(exps,dtype='f8'))
                              index += 1
                              count = 0
                              exps=[]
                  elif sec_flag == 'ao_coeffs':
                      for ii in thisline:
                          coeffs.append(float(ii))
                          count += 1
                          if count==ao_prim_number[index]:
                              ao.coeff.append(numpy.array(coeffs,dtype='f8'))
                              if ao_types[index]==-1: 
                                  ao.coeff.append([])      # for sp orbital expansion coefficients for p orbitals will be loaded later
                              index += 1
                              count = 0
                              coeffs = []
                  elif sec_flag == 'ao_coeffs_p':
                      for ii in thisline:
                          coeffs.append(float(ii))
                          count += 1
                          if count==ao_prim_number[index]:
                              if ao_types[index]==-1:
                                  index_all += 1
                                  if len(ao.coeff[index_all])==0:
                                      ao.coeff[index_all]=numpy.array(coeffs,dtype='f8')
                                  else:
                                      raise IOError('Trying to write p part of sp expansion coefficients into allready allocated coefficients')
                              index += 1
                              index_all += 1
                              count = 0 
                              coeffs = []
                          ao_num -= 1
                      if not ao_num:
                          sec_flag = None
                  elif sec_flag == 'mo_eorb':
                      for ii in thisline:
                          mo_spec[count]['energy'] = float(ii)
                          count += 1
                          if index != 0 and not count % basis_count:
                              sec_flag = None
                  elif sec_flag == 'mo_coeffs':
                     for ii in thisline:   
                          mo_spec[mo_i0[what]+index]['coeffs'][count] = float(ii)
                          count += 1
                          if count == basis_count:
                              count = 0
                              index += 1
                          if index != 0 and not index % basis_count:
                              sec_flag = None
                  elif sec_flag=='hessian':
                      for ii in thisline:
                          MatHess[count]=float(ii)
                          count+=1
                          if count==num_cons:
                              sec_flag=None
                  elif sec_flag=='tot_scf_dens':
                      for ii in thisline:
                          gr_dens[denscount_line,denscount_row]=float(ii)
                          denscount_row+=1
                          if denscount_row>denscount_line:
                              denscount_row=0
                              denscount_line+=1
                          if denscount_line==basis_number:
                              sec_flag=None
                  elif sec_flag=='tot_ci_dens':
                      for ii in thisline:
                          exct_dens[denscount_line,denscount_row]=float(ii)
                          denscount_row+=1
                          if denscount_row>denscount_line:
                              denscount_row=0
                              denscount_line+=1
                          if denscount_line==basis_number:
                              sec_flag=None

    # Transfor lower triangular matrix into full one if defined:            
    if len(gr_dens)!=0:
          for ii in range(len(gr_dens)):
              for jj in range(ii+1,len(gr_dens)):
                  gr_dens[ii,jj]=gr_dens[jj,ii]
              
    if len(exct_dens)!=0:
        for ii in range(len(exct_dens)):
            for jj in range(ii+1,len(exct_dens)):
                exct_dens[ii,jj]=exct_dens[jj,ii]
  
    # Transform lower triangular hessian matrix writen in vector into full hessian matrix
    if num_cons!=None:
        Nmodes=int(round((numpy.sqrt(num_cons*8+1)-1)/2))
        hess_fchk=numpy.zeros((Nmodes,Nmodes))
        count=0
        for ii in range(Nmodes):
            for jj in range(ii+1):
                hess_fchk[ii,jj]=MatHess[count]
                count+=1
        for ii in range(Nmodes):
            for jj in range(ii+1,Nmodes):
                hess_fchk[ii,jj]=hess_fchk[jj,ii]
  
  
    # Only molecular orbitals of one spin requested?
#    if spin is not None:
#        for i in range(len(mo_spec))[::-1]:
#            if mo_spec[i]['spin'] != spin:
#                del mo_spec[i]
  
    if restricted:
        # Closed shell calculation
        for _mo in mo_spec:
            del _mo['spin']
    else:
        # Rename MOs according to spin
        for _mo in mo_spec:
            _mo['sym'] += '_%s' % _mo['spin'][0]
  
    # Check for natural orbital occupations
    energy_sum = sum([abs(i['energy']) for i in mo_spec])
    if energy_sum < 0.0000001:
        print('Attention!\n\tThis FChk file contains natural orbitals. '+
            '(There are no energy eigenvalues.)\n\t' + 
            'In this case, Gaussian does not print the respective natural' +
            'occupation numbers!' )

    # Fill structure class
    at_type=[]
    for ii in range(len(ncharge)):
        at_type.append(get_atom_symbol(ncharge[ii]))
    struc.add_coor(coor.value,at_type,ncharge,mass)

    # Fill atomic orbitals - the missing parts
    ao.nao=len(ao.type)
    if ao.nao != len(ao.coeff) or ao.nao != len(ao.exp):
        print("Atomic types:",ao.nao,"  n. of coefficients:",len(ao.coeff)," and exponents:",len(ao.exp))
        #print(ao.type)
        raise Warning('Wrong read of atomic orbitals')
    for ii in range(ao.nao):
        orient=l_orient(ao.type[ii])
        ao.orient.append(orient)
        ao.nao_orient+=len(orient)
        for jj in range(len(orient)):
            ao.indx_orient.append([ii,orient[jj]])
    # for coordinates assuming atomic centered basis
    ao_coor=Coordinate(None)
    for ii in range(ao.nao):
        at_indx=ao.atom[ii].indx
        ao_coor.add_coor(struc.coor.value[at_indx])
    ao.coor=ao_coor
    ao.init=True
    #repair ao.type
    if False:
        count={}
        for ii in range(ao.nao):
            ao_type=''.join([i for i in ao.type[ii] if not i.isdigit()])
            if not (ao_type in count.keys()):
                count[ao_type]=0
            if ao.atom[ii].indx!=ao.atom[ii-1].indx and ii>0:
                for key in count:
                    count[key]=0
            count[ao_type] += 1
            ao.type[ii]=[ao.type[ii],count[ao_type]+lquant[ao_type]]
    
    # Fill molecular orbitals
    mo.name='All Atom MO'
    NMO=len(mo_spec)
    with energy_units('AU'):
        for ii in range(NMO):
            coeff_i=numpy.array(dict(mo_spec[ii])['coeffs'],dtype='f8')
            energy_i=float(mo_spec[ii]['energy'])
            occ_i=float(mo_spec[ii]['occ_num'])
            symm_i=mo_spec[ii]['sym']
            mo.add_all(coeff_i,energy_i,occ_i,symm_i)
        
    if mo.nmo!=NMO:
        raise IOError('Wrong read of molecular orbitals')
    if len(gr_dens)!=0:
        mo.densmat_grnd=gr_dens

    return struc,ao,mo,TotalEnergy,exct_dens,hess_fchk

      
def read_gaussian_log(filename,verbose=False):
    ''' Reads usefull information from gaussian 09 log file.
    (excited state information, 1electron and 2electrons integrals, SAC and
    SAC-CI calculations, ...)

    Parameters
    ----------
    filename : str
        Specifies the filename for the input file (including the path if needed)
  
    Returns
    -------
    job_info : list of strings
        Contains information about gaussian 09 calculation (calculation settings)
    Excitation_list : list of Excitation class
        Contains information about excited state properties in Excitation class.
        Excitation_list[0] correspond to transition 0->1, Excitation_list[1] 
        correspond to transition 0->2, etc.
    Normal_mode_info : list (dimension 3)
        Information about normal mode analysis if it was performed, if not 
        Normal_mode_info = None. 
            
        * Normal_mode_info[0] : numpy array or float (dimension 3Natom x Nmodes)
            Vibration normal mode displacements in cartesian displacemenst
            in columns. Transformation matrix from normal mode displacements
            into cartesian ones.
        * Normal_mode_info[1] : list of dictionary (dimension Nmodes)
            * Normal_mode_info[1][i]['frequency_cm-1'] = frequency of normal mode i in inverse centimeters
            * Normal_mode_info[1][i]['red_mass_AMU'] = reduced mass of normal moce i in atomic mass units
            * Normal_mode_info[1][i]['force_cons_mDyne/A'] = force constants of normal mode i in mDyne/A
            * Normal_mode_info[1][i]['IR_intens_KM/Mole'] = infrared intensity if normal mode i in KM/Mole
        * Normal_mode_info[2] : numpy array of foat (dimension 3*Natom x 3*Natom)
            Hessian matrix read from log file
            
    Single_el_prop : dictionary
        Information about single electron integrals in atomic orbital basis if
        requested by **IOp=('3/33=1')**.
        
        * Single_el_prop['overlap'] : numpy array of float (dimension Nao_orient x Nao_orient)
            Overlap matrix between atomic orbitals
        * Single_el_prop['kinetic'] : numpy array of float (dimension Nao_orient x Nao_orient)
            Kinetic energy matrix in atomic orbital basis
        * Single_el_prop['potential'] : numpy array of float (dimension Nao_orient x Nao_orient)
            Potential energy matrix in atomic orbital basis. (Nuclei - electron interaction)
        * Single_el_prop['core'] : numpy array of float (dimension Nao_orient x Nao_orient)
            Core hamiltonian in atomic orbital basis. Core hamiltonian = Kinetic energy - Potential energy
        * Single_el_prop['fermi'] : numpy array of float (dimension Nao_orient x Nao_orient)
            fermi contact integrals in atomic orbital basis. 
    
    Double_el_integrals : list of integer and float
        Double electron integrals if requested by **extralinks=l316**
        Double_el_integrals[a]=[i,j,k,l,int_value] where int_value = (ij|kl)
        where (ij|kl) = int{dr1 int{dr2} {AO_i(r1)AO_j(r1) 1/r12 AO_k(r2)AO_l(r2)}}
    SAC_spec : dictionary
        Information from CAS calculation of ground state
        
        * SAC_spec['coeff_single'] : list of integer and float (dimension Nexct x 3)
            Contribution of single excitations in ground state. 
            SAC_spec['coeff_single'][n] = [MO_i, MO_j, coeff] correspond to
            excitation MO_i -> MO_j with expansion coefficient coeff
        * SAC_spec['coeff_double'] : list of integer and float (dimension Nexct x 5)
            Contribution of double excitations in ground state. 
        * SAC_spec['energy_hartree'] : float
            SAC ground state energy in Hartree
            
    SAC_CI_spec : list of dictionary
        Information from CAS-CI calculation of excited state
        
        * SAC_CI_spec[a]['coeff_single'] : list of integer and float (dimension Nexct x 3)
            Contribution of single excitations for (a+1)th excited state. 
            SAC_CI_spec[a]['coeff_single'][n] = [MO_i, MO_j, coeff] correspond to
            excitation MO_i -> MO_j with expansion coefficient coeff
        * SAC_CI_spec[a]['coeff_double'] : list of integer and float (dimension Nexct x 5)
            Contribution of double excitations for (a+1)th excited state. 
        * SAC_CI_spec[a]['energy_hartree'] : float
            SAC-CI (a+1)th excited state energy in Hartree
        * SAC_CI_spec[a]['energy_eV'] : float
            SAC-CI transition energy in eV of (a+1)th excited state
        * SAC_CI_spec[a]['symmetry'] : string
            Symmertry of (a+1)th excited state.

    '''
    
    fid    = open(filename,'r')      # Open the file
    flines = fid.readlines()         # Read the WHOLE file into RAM
    fid.close()                      # Close the file
    sec_flag=None
    ex_allocated=False
    excit=False
    root_def=False
    is_single_el=False
    is_double_el=False
    is_init=False
    int_keyword=[]
    is_HPModes=False
    counterNM=0
    is_TwoElInt=False
    
    Nel_Alpha_Beta=[]  # Nel_Alpha_Beta[0]=number of Alpha electrons
                       # Nel_Alpha_Beta[1]=number of Beta electrons
    # Nat : Number of atoms in the  molecule
    # E_Nuc : Nuclear repulsion energy
    # N_Basis : number of AO basiss functions
    # sel_over : overlap matrix of AO basis (only if single electron information is outputed)
    # sel_kin : kinetic energy matrix in AO basis (only if single electron information is outputed)
    # sel_pot : potential matrix in AO basis (only if single electron information is outputed)
    # sel_core : core hamiltonian in AO basis = Kinetic energy - Potential energy (only if single electron information is outputed)
    # sel_fermi : fermi contact integrals in AO basis
    # del : Double electron integrals ordered 
    Excitation_list=[] # List with excitation information
    exc_info=[]
    nm_info = [] # Information about vibrational analysis - list of dictionaries (dimension Nmodes)
                 # 'frequency_cm-1'
                 # 'red_mass_AMU'
                 # 'force_cons_mDyne/A'
                 # 'IR_intens_KM/Mole'
    nm_coef = [] # vibrational normal modes in column (dimension 3*NAtoms x Nmodes)
    hess_log = [] # Hessian matrix stored in log file - different from the one stored in fchk file.
                  # USE THE ONE from fchk file
    
    SAC_spec={}
    SAC_CI_spec=[]   
    
    with energy_units('AU'):
        with position_units('Bohr'):
            for il in range(len(flines)):
              line = flines[il]            # The current line as string
              thisline = line.split()      # The current line split into segments
              if line.rstrip():
                  blank_line=False
              else:
                  blank_line=True
            	# Nacteni hlavicky musi probehnout uplne nejdrive jelikoz vypis se muze lisist podle metody      
              if ('#' in line) and (not is_init):
                  if sec_flag == None:
                      sec_flag = 'Job_desc'
                      job_info=[]
              if (' ----' in line) and (not is_init) and sec_flag == 'Job_desc':
                  job_info=''.join(job_info)
                  job_info=(job_info.replace('=(','(').replace('(',' ').replace('/',' / ').replace(',',' ').replace(')','')).split()
                  #print('Job info from log file:',job_info)
                  sec_flag = None
                  is_init=True
              elif 'alpha electrons' in line:
                  Nel_Alpha_Beta.append(int(thisline[0]))
                  Nel_Alpha_Beta.append(int(thisline[3]))
              elif ' Calculate overlap and kinetic energy integrals' in line:
                  sec_flag = 'Basis information'
              elif ' NAtoms=' in line:
                  thisline = line.split("=")[1].split()
                  NAtoms = int(thisline[0])
                  Nat=NAtoms
              elif 'nuclear repulsion energy' in line:
                  E_Nuc=Energy(float(thisline[3]))
              elif ('    NBasis' in line) and sec_flag == 'Basis information':
                  sec_flag = None
                  N_Basis= int(thisline[2])
              elif ' *** Overlap *** ' in line:
                  sec_flag = 'Single_el_prop'
                  index = 'ao_overlap'
                  counter=0
                  MM=numpy.zeros((N_Basis,N_Basis))
                  for jj in range(N_Basis//5+1):
                      for ii in range(5*jj,N_Basis+1):
                          if ii!=5*jj:
                              line = flines[il+1+counter]
                              thisline = line.split()
                              for kk in range(5):
                                  if kk+5*jj+1<=ii:
                                      MM[ii-1,kk+5*jj]=float(thisline[kk+1].replace('D', 'e'))
                                #print(5*jj,ii-1,flines[counter])
                          counter+=1
                  for ii in range(N_Basis):
                      for jj in range(ii+1,N_Basis):
                          MM[ii,jj]=MM[jj,ii]
                  # print(MM)
                  # Zapsat MM do qc.
                  sel_over=numpy.array(MM)
              elif ' *** Kinetic Energy *** ' in line:
                  sec_flag = 'Single_el_prop'
                  index = 'ao_kinetic'
                  counter=0
                  MM=numpy.zeros((N_Basis,N_Basis))
                  for jj in range(N_Basis//5+1):
                      for ii in range(5*jj,N_Basis+1):
                          if ii!=5*jj:
                              line = flines[il+1+counter]
                              thisline = line.split()
                              for kk in range(5):
                                  if kk+5*jj+1<=ii:
                                      MM[ii-1,kk+5*jj]=float(thisline[kk+1].replace('D', 'e'))
                                #print(5*jj,ii-1,flines[counter])
                          counter+=1
                  for ii in range(N_Basis):
                      for jj in range(ii+1,N_Basis):
                          MM[ii,jj]=MM[jj,ii]
                  #print(MM)
                  # Zapsat MM do qc.
                  sel_kin=numpy.array(MM)
              elif (' Entering OneElI...' in line) and sec_flag == 'Single_el_prop' and index == 'ao_kinetic':
                  sec_flag = 'Single_el_prop'
                  index = None
              elif ' ***** Potential Energy ***** ' in line:
                  sec_flag = 'Single_el_prop'
                  index = 'ao_potential'
                  counter=0
                  MM=numpy.zeros((N_Basis,N_Basis))
                  for jj in range(N_Basis//5+1):
                      for ii in range(5*jj,N_Basis+1):
                          if ii!=5*jj:
                              line = flines[il+1+counter]
                              thisline = line.split()
                              for kk in range(5):
                                  if kk+5*jj+1<=ii:
                                      MM[ii-1,kk+5*jj]=float(thisline[kk+1].replace('D', 'e'))
                                #print(5*jj,ii-1,flines[counter])
                          counter+=1
                  for ii in range(N_Basis):
                      for jj in range(ii+1,N_Basis):
                          MM[ii,jj]=MM[jj,ii]
                  # print(MM)
                  # Zapsat MM do qc.
                  sel_pot=numpy.array(MM)
              elif ' ****** Core Hamiltonian ****** ' in line:
                  # (Kinetic energy - Potential energy)
                  sec_flag = 'Single_el_prop'
                  index = 'ao_core'
                  counter=0
                  MM=numpy.zeros((N_Basis,N_Basis))
                  for jj in range(N_Basis//5+1):
                      for ii in range(5*jj,N_Basis+1):
                          if ii!=5*jj:
                              line = flines[il+1+counter]
                              thisline = line.split()
                              for kk in range(5):
                                  if kk+5*jj+1<=ii:
                                      MM[ii-1,kk+5*jj]=float(thisline[kk+1].replace('D', 'e'))
                                #print(5*jj,ii-1,flines[counter])
                          counter+=1
                  for ii in range(N_Basis):
                      for jj in range(ii+1,N_Basis):
                          MM[ii,jj]=MM[jj,ii]
                  # print(MM)
                  # Zapsat MM do qc.
                  sel_core=numpy.array(MM)
              elif (' SVDSVc' in line) and sec_flag == 'Single_el_prop' and index == 'ao_core':
                  sec_flag = None
                  index = None
              elif ' Orthogonalized basis functions: ' in line:
                  sec_flag = 'Single_el_prop'
                  index = 'ao_orthog_basis'
              elif (' NBasis=' in line) and sec_flag == 'Single_el_prop' and index == 'ao_orthog_basis':
                  sec_flag = None
                  index = None
              elif ' Fermi contact integrals:' in line:
                  sec_flag = 'Single_el_prop'
                  index = 'ao_fermi_int'
                  #counter1 = 0
                  #counter2 = 0
                  counter=0
                  MM=numpy.zeros((N_Basis,NAtoms))
                  for jj in range(NAtoms//5+1): #  Pocet bloku ktere se musi nacist
                      for ii in range(N_Basis+1):
                          if ii!=0: # Prvni radek se necte
                              line = flines[il+1+counter]
                              thisline = line.split()
                              for kk in range(5):
                                  if kk+5*jj+1<=NAtoms:
                                      MM[ii-1,kk+5*jj]=float(thisline[kk+1].replace('D', 'e'))
                                #print(5*jj,ii-1,flines[counter])
                          counter+=1
                  # print(MM)
                  # Zapsat MM do qc.
                  sel_fermi=numpy.array(MM)
              elif (' Leave Link  303' in line) and sec_flag == 'Single_el_prop' and index == 'ao_fermi_int':
                  sec_flag = None
                  index = None
              elif ' *** Dumping Two-Electron integrals ***' in line:
                  sec_flag = 'Two_el_prop'
                  index = None
              elif (' IntCnt=' in line) and sec_flag == 'Two_el_prop':
                  N_int=int(thisline[1])
                  sec_flag = 'Two_el_prop'
                  index = 'ao_int'
                  counter=0
                  TwoElInt = []
                  is_TwoElInt=True
              elif (' Leave Link  316' in line) and sec_flag == 'Two_el_prop' and index == 'ao_int':
                  sec_flag = None
                  index = None
              elif ' Ground to excited state transition electric dipole moments (Au):' in line:
                  sec_flag = 'Excited_state_info'
                  index = 'el_trans_dip'
                  # Az se bude prochazet pres tyto dipolove momenty je potreba allokovat qc.exc_spec.append({})
              elif ' Ground to excited state transition velocity dipole moments (Au):' in line:
                  sec_flag = 'Excited_state_info'
                  index = 'vel_trans_dip'
              elif ' Ground to excited state transition magnetic dipole moments (Au):' in line:
                  sec_flag = 'Excited_state_info'
                  index = 'mag_trans_dip'
              elif ' Ground to excited state transition velocity quadrupole moments (Au):' in line:
                  sec_flag = 'Excited_state_info'
                  index = 'vel_trans_quad'
              elif '  <0|del|b> * <b|rxdel|0> + <0|del|b> * <b|delr+rdel|0>' in line:
                  sec_flag = 'rot_info'
              elif ' Excitation energies and oscillator strengths:' in line:
                  sec_flag = 'Excited_state_eng_osc'
                  index = None 
              elif (' Excited State' in line) and (not (' Excited state symmetry could not be determined.' in line)) and (sec_flag=='Excited_state_eng_osc'):
                  index = 'coef'
                  # split line into individual words and remove double dot
                  thisline2 = (line.replace(':', ' ').replace('=','= ').replace('-',' ')).split()
                  Nexc = int(thisline2[2])
                  energy_eV = float(thisline2[5])
                  #energy_nm = float(thisline2[7])
                  
                  if ex_allocated:
                      with energy_units('eV'):
                          Excitation_list[Nexc-1].energy.value=energy_eV
                      Excitation_list[Nexc-1].coeff=[]
                      Excitation_list[Nexc-1].symm=thisline2[4]
                  else:
                      Excitation_list.append(Excitation(energy_eV,numpy.zeros(3,dtype='f8'),coeff=[],symm=thisline2[4]))
                  
              elif (sec_flag=='Excited_state_eng_osc') and blank_line:
                  index = None
              elif (' This state for optimization' in line) and (sec_flag=='Excited_state_eng_osc'):     
                  index=None
              elif (' SavETr:' in line) and (sec_flag=='Excited_state_eng_osc'):
                  sec_flag=None
              elif '    ENERGY AND WAVE FUNCTION OF SAC METHOD' in line:
                  sec_flag='SAC'   # ground state properties
                  index = None
                  SAC_spec={}
              elif (' *SINGLE EXCITATION' in line) and sec_flag=='SAC':
                  sec_flag='SAC'   # ground state properties
                  index = 'single_exc'
                  SAC_spec['coeff_single'] = []
              elif (' *DOUBLE EXCITATION' in line) and sec_flag=='SAC':
                  sec_flag='SAC'   # ground state properties
                  index = 'double_exc'
                  SAC_spec['coeff_double'] = []
              elif (' Hartree-Fock configuration' in line) and sec_flag=='SAC' and (index=='double_exc' or index=='single_exc'):
                  sec_flag=None
                  index=None
              elif '    ENERGY AND WAVE FUNCTION OF SAC-CI METHOD' in line:
                  sec_flag='SAC-CI'     # excited state properties
                  index=None
              elif (' *******************************************************************************' in line) and sec_flag=='SAC-CI':
                  index=None
              elif ('###' in line) and sec_flag=='SAC-CI':
                  sec_flag='SAC-CI'
                  index=None
                  # nacist energie a cislo excit. stavu
                  # ###   1-st  ###           ---  11th state in this spin multiplicity ---
                  # ###   1-st  ###           ---   1st state in this spin multiplicity ---
                  # ###   1-st  ###           ---   2nd state in this spin multiplicity ---
                  thisline2 = (line.replace('-', ' ').replace('st',' ').replace('nd',' ').replace('rd',' ').replace('th',' ')).split()          
                  Nexc=int(thisline2[3])
                  #print(thisline2,Nexc)
              elif ('   *SINGLE EXCITATION' in line) and sec_flag=='SAC-CI':
                  sec_flag='SAC-CI'
                  index = 'single_exc'
                  ii=len(SAC_CI_spec)-1
                  SAC_CI_spec[ii]['coeff_single'] = []
              elif ('   *DOUBLE EXCITATION' in line) and sec_flag=='SAC-CI':
                  sec_flag='SAC-CI'
                  index = 'double_exc'
                  ii=len(SAC_spec)-1
                  SAC_CI_spec[ii]['coeff_double'] = []
              elif (' ================================================================' in line) and sec_flag=='SAC-CI':
                  index = None
              elif (' Transition dipole moment of' in line) and sec_flag=='SAC-CI':
                  sec_flag='SAC-CI'
                  index = 'el_trans_dip'
                  counter=0
              elif (' Dipole(Magnetic)' in line) and sec_flag=='SAC-CI':
                  sec_flag='SAC-CI'
                  index = 'mag_trans_dip'
              elif (' Dipole(Velocity)' in line) and sec_flag=='SAC-CI':
                  sec_flag='SAC-CI'
                  index = 'vel_trans_dip'
              elif (' Rotatory Strengths(Length-gauge)' in line) and sec_flag=='SAC-CI':
                  sec_flag='SAC-CI'
                  index = 'rot_str'
              elif (' Rotatory Strengths(Velocity-gauge)' in line) and sec_flag=='SAC-CI':
                  sec_flag='SAC-CI'
                  index = 'rot_vel'
              elif (' Rotatory Strengths are given in') and sec_flag=='SAC-CI' and index=='rot_vel':
                  sec_flag=None            
                  index = None
              elif ' and normal coordinates' in line:
                  if (is_HPModes) and (counterNM==0):
                      sec_flag='NormalModesHP'
                      index=None
                      Nmodes=NAtoms*3-6
                      ModeNum=0
                      for ii in range(Nmodes):
                          nm_info.append({})
                      nm_coef=numpy.zeros((3*NAtoms,Nmodes))
                      elements=numpy.zeros(NAtoms)
                  elif (counterNM==0) and (not is_HPModes):
                      sec_flag='NormalModes'
                      index=None
                      Nmodes=NAtoms*3-6
                      ModeNum=0
                      for ii in range(Nmodes):
                          nm_info.append({})
                      nm_coef=numpy.zeros((3*NAtoms,Nmodes))
                      elements=numpy.zeros(NAtoms)
                  else: 
                      sec_flag=None            
                      index = None
                  counterNM+=1
              elif ('Frequencies ---' in line) and (sec_flag=='NormalModesHP'):
                  index = None
                  for ii in range(5*ModeNum,5*ModeNum+5):
                      if ii<=Nmodes-1:
                          nm_info[ii]['frequency_cm-1'] = float(thisline[2+ii%5])
              elif ('Reduced masses ---' in line) and (sec_flag=='NormalModesHP'):
                  index = None
                  for ii in range(5*ModeNum,5*ModeNum+5):
                      if ii<=Nmodes-1:
                          nm_info[ii]['red_mass_AMU'] = float(thisline[3+ii%5]) # transform AMU to something better
              elif ('Force constants --' in line) and (sec_flag=='NormalModesHP'):
                  index = None
                  for ii in range(5*ModeNum,5*ModeNum+5):
                      if ii<=Nmodes-1:
                          nm_info[ii]['force_cons_mDyne/A'] = float(thisline[3+ii%5]) # transform mDyne/A to something better            
              elif ('IR Intensities --' in line) and (sec_flag=='NormalModesHP'):
                  index = None
                  for ii in range(5*ModeNum,5*ModeNum+5):
                      if ii<=Nmodes-1:
                          nm_info[ii]['IR_intens_KM/Mole'] = float(thisline[3+ii%5])
              elif ('Coord Atom Element:' in line) and (sec_flag=='NormalModesHP'):
                  index = 'Modes'
                  indx=0
                  if (Nmodes-5*ModeNum)>5:
                      NNM=5
                  else:
                      NNM=(Nmodes-5*ModeNum)
              elif ('Frequencies --' in line) and (sec_flag=='NormalModes'):      
                  index = None
                  for ii in range(3*ModeNum,3*ModeNum+3):
                      if ii<=Nmodes-1:
                          nm_info[ii]['frequency_cm-1'] = float(thisline[2+ii%3])
              elif ('Red. masses --' in line) and (sec_flag=='NormalModes'):      
                  index = None
                  for ii in range(3*ModeNum,3*ModeNum+3):
                      if ii<=Nmodes-1:
                          nm_info[ii]['red_mass_AMU'] = float(thisline[3+ii%3]) # transform AMU to something better
              elif ('Frc consts  --' in line) and (sec_flag=='NormalModes'):      
                  index = None
                  for ii in range(3*ModeNum,3*ModeNum+3):
                      if ii<=Nmodes-1:
                          nm_info[ii]['force_cons_mDyne/A'] = float(thisline[3+ii%3]) # transform mDyne/A to something better                       
              elif ('IR Inten    --' in line) and (sec_flag=='NormalModes'):      
                  index = None
                  for ii in range(3*ModeNum,3*ModeNum+3):
                      if ii<=Nmodes-1:
                          nm_info[ii]['IR_intens_KM/Mole'] = float(thisline[3+ii%3])
              elif ('  Atom  AN' in line) and (sec_flag=='NormalModes'):
                  index = 'Modes'
                  indx=0
                  if (Nmodes-3*ModeNum)>3:
                      NNM=3
                  else:
                      NNM=(Nmodes-3*ModeNum)
              elif ' Harmonic frequencies (cm**-1)' in line:
                  sec_flag=None            
                  index = None
              elif ' Force constants in Cartesian coordinates:' in line:
                  sec_flag='NormalModes'
                  index='Hessian'
                  counter=0
                  Ndegrees=NAtoms*3
                  MM=numpy.zeros((Ndegrees,Ndegrees))
                  for jj in range(Ndegrees//5+1):
                      for ii in range(5*jj,Ndegrees+1):
                          if ii!=5*jj:
                              line = flines[il+1+counter]
                              thisline = line.split()
                              for kk in range(5):
                                  if kk+5*jj+1<=ii:
                                      MM[ii-1,kk+5*jj]=float(thisline[kk+1].replace('D', 'e'))
                                #print(5*jj,ii-1,flines[counter])
                          counter+=1
                  for ii in range(Ndegrees):
                      for jj in range(ii+1,Ndegrees):
                          MM[ii,jj]=MM[jj,ii]
                  hess_log=numpy.array(MM)        
              else:
                  if sec_flag == 'Job_desc':
                      # read job description
                      job_info.append(line)
                      thisline3 = (line.replace('=(','(').replace('(',' ').replace('/',' / ').replace(',',' ').replace(')','')).split()
                      
                      #print(thisline3)
                      for ii in thisline3:
                          #print(ii.lower())
                          #print(ii.lower().replace('=',' '))
                          #if ii.lower() == '/':
                          #    print('I like /')
                          #if ii.lower() == 'aug-cc-pvtz':
                          #    print('aug-cc-pvtz is best basis ever')
                          if ii.lower() == 'td':
                              #print('excitation type TD-DFT')
                              exc_info.append(['calc_type','TD-DFT'])
                              #qc.exc_info['excit_type'] = 'TD-DFT'
                              excit=True
                          elif ii.lower()=='cis':
                              exc_info.append(['calc_type','CIS'])
                              excit=True
                          elif ii.lower()=='eomccsd':
                              exc_info.append(['calc_type','EOM-CCSD'])
                              excit=True
                          elif ii.lower()=='casscf':
                              exc_info.append(['calc_type','CASSCF'])
                              excit=True
                          elif ii.lower()=='sac-ci':
                              exc_info.append(['calc_type','SAC-CI'])
                              excit=True
                          elif ii.lower()=='zindo':
                              exc_info.append(['calc_type','ZIndo'])
                              excit=True
                          elif ii.lower()=='singlets' or ii.lower()=='singlet':
                              exc_info.append(['exct_type','Singlets'])
                          elif ii.lower()=='triplets' or ii.lower()=='triplet':
                              exc_info.append(['exct_type','Triplets'])
                          elif 'root' in ii.lower():
                              li=ii.lower().split('=')
                              exc_info.append(['root',int(li[1])])
                              root_def=True
                          elif 'noint' in (ii.lower().replace('=',' ')):
                              int_keyword.append('symmetry=noint')
                              #print('symmetry=noint     OK')
                          elif 'conventional' in ii.lower().replace('=',' '):
                              int_keyword.append('conventional')
                              #print('conventional     OK')
                          elif 'noint' in ii.lower().replace('=',' '):
                              int_keyword.append('noint')
                          elif 'extralinks=l316' in ii.lower():
                              is_double_el=True
                              #print('Double integrals will be printed')
                          elif 'noraff' in ii.lower().replace('=',' '):
                              int_keyword.append('noraff')
                              #print('noraff     OK')
                          elif 'hpmodes' in ii.lower().replace('=',' '):
                              is_HPModes=True
                              #print('HPModes    OK')
                      if '3/33=1' in line:
                          is_single_el=True
                          #print('Single integrals will be printed')
                  elif sec_flag == 'Single_el_prop':
                      if index == 'ao_orthog_basis':
                          a='Not implemented yet'
                      elif index == 'ao_fermi_int':
                          # counter1
                          # counter2
                          # for ii in range(5):
                          #    ii+counter
                          a='Not implemented yet'
                  elif sec_flag == 'Two_el_prop':
                      if index == 'ao_int':
                          #if counter<N_int:
                          TwoElInt.append([int(thisline[1]),int(thisline[3]),int(thisline[5]),int(thisline[7]),float(thisline[9].replace('D', 'e'))])
                          #counter+=1
                  elif sec_flag == 'Excited_state_info':
                      if index == 'el_trans_dip':
                          if not ('       state          X           Y           Z        Dip. S.      Osc.' in line):
                              tr_dip=(numpy.array(thisline[1:4], dtype='|S11')).astype(numpy.float)
                              oscil=float(thisline[-1])
                              Excitation_list.append(Excitation(0.0,tr_dip,oscil=oscil))
#                              qc.exc_spec.append({})
#                              Nexc = int(thisline[0])
#                              qc.exc_spec[Nexc-1]['excitation'] = Nexc
#                              qc.exc_spec[Nexc-1]['trans_dip'] = (numpy.array(thisline[1:4], dtype='|S11')).astype(numpy.float)
#                              qc.exc_spec[Nexc-1]['oscillator'] = float(thisline[-1])
                              ex_allocated=True
                      if index == 'vel_trans_dip':
                          if not ('       state          X           Y           Z        Dip. S.      Osc.' in line):
                              # read transition velocity dipole moments
                              a='Not implemented yet' 
                      if index == 'mag_trans_dip':
                          if not ('       state          X           Y           Z' in line):
                              # read transition magnetic dipole moments
                              a='Not implemented yet'
                      if index == 'vel_trans_quad':
                          if not ('       state          XX          YY          ZZ          XY          XZ          YZ' in line):
                              # read transition velocity quadrupole moments
                              a='Not impemented yet'
                  elif sec_flag == 'Excited_state_eng_osc':
                      calc_type=dict(exc_info)['calc_type']
                      if index == 'coef':
                          # read coefficients for transitions Nexc -> index zatisu je tedy Nexc-1
                          #coef=[]
                          
                          if calc_type!='SAC-CI' and calc_type!='CASSCF':
                              thisline3 = (line.replace('->', ' -> ').replace('<-',' <- ')).split()
                              if thisline3[1]=='->':
                                  #coef.append([int(thisline3[0]),int(thisline3[2]),float(thisline3[3])])
                                  #qc.exc_spec[Nexc-1]['coefficients'].append([int(thisline3[0]),int(thisline3[2]),float(thisline3[3])])
                                  Excitation_list[Nexc-1].coeff.append([int(thisline3[0]),int(thisline3[2]),float(thisline3[3])])
                              elif thisline3[1]=='<-':
                                  #coef.append([int(thisline3[2]),int(thisline3[0]),float(thisline3[3])])
                                  #qc.exc_spec[Nexc-1]['coefficients'].append([int(thisline3[2]),int(thisline3[0]),float(thisline3[3])])
                                  Excitation_list[Nexc-1].coeff.append([int(thisline3[2]),int(thisline3[0]),float(thisline3[3])])
                                   # if TD-DFT deexcitation first coef>second if normal excitation first coef<second 
                                    
                  elif sec_flag == 'SAC':
                      if index==None:
                          # read energy
                          if 'TOTAL ENERGY' in line:
                              energy=float(thisline[3])
                              SAC_spec['energy_hartree'] = energy
                              SAC_spec['energy_eV'] = None
                              SAC_spec['excitation'] = 0  # ground state
                          elif 'SAC-NV  coefficients' in line:
                              print('Cutoff for written coefficients is:',thisline[2])
                          # vypis energie do nejake promenne
                              
                      elif index=='single_exc':
                          # nacist koeficienty single excitaci pro ground state
                          if len(thisline)==6:
                              SAC_spec['coeff_single'].append([int(thisline[0]),int(thisline[1]),float(thisline[2])])
                              SAC_spec['coeff_single'].append([int(thisline[3]),int(thisline[4]),float(thisline[5])])
                          elif len(thisline)==3:
                              SAC_spec['coeff_single'].append([int(thisline[0]),int(thisline[1]),float(thisline[2])])
                          else:
                              raise IOError('Wrong number of single excitation properties (must be 3 or 6)')
                      elif index=='double_exc':
                          # nacist koeficienty double excitaci pro ground state
                          if len(thisline)==10:
                              SAC_spec['coeff_double'].append([int(thisline[0]),int(thisline[1]),int(thisline[2]),int(thisline[3]),float(thisline[4])])
                              SAC_spec['coeff_double'].append([int(thisline[5]),int(thisline[6]),int(thisline[7]),int(thisline[8]),float(thisline[9])])
                          elif len(thisline)==5:
                              SAC_spec['coeff_double'].append([int(thisline[0]),int(thisline[1]),int(thisline[2]),int(thisline[3]),float(thisline[4])])
                          else:
                              raise IOError('Wrong number of double excitation properties (must be 5 or 10)')
                  elif sec_flag == 'SAC-CI':
                      if index==None:
                          if ' Singlet' in line:
                              symm=thisline[1]
                              # vypis symetrie pro kazdou excitaci ele nacteni vzdy jen jednou pro dany blok excitaci
                          elif ' Triplet' in line:
                              symm=thisline[1]
                              # vypis symetrie pro kazdou excitaci ele nacteni vzdy jen jednou pro dany blok excitaci
                          elif 'Excitation energy' in line:
                              energy_hartree=thisline[5]
                              energy_eV=thisline[9]
                              # nyni muzu udelat vypis energie i excitace pro excitaci c. Nexc 
                              SAC_CI_spec.append({})
                              indx=len(SAC_CI_spec)-1
                              SAC_CI_spec[indx]['energy_eV'] = energy_eV
                              SAC_CI_spec[indx]['energy_hartree'] = energy_hartree
                              SAC_CI_spec[indx]['symmetry'] = symm
                              SAC_CI_spec[indx]['excitation'] = Nexc
                      elif index=='single_exc':
                          # nacist koeficienty single excitaci
                          if len(thisline)==6:
                              SAC_CI_spec[indx]['coeff_single'].append([int(thisline[0]),int(thisline[1]),float(thisline[2])])
                              SAC_CI_spec[indx]['coeff_single'].append([int(thisline[3]),int(thisline[4]),float(thisline[5])])
                          elif len(thisline)==3:
                              SAC_CI_spec[indx]['coeff_single'].append([int(thisline[0]),int(thisline[1]),float(thisline[2])])
                          else:
                              raise IOError('Wrong number of single excitation properties (must be 3 or 6)')
                      elif index=='double_exc':
                          # nacist koeficienty double excitaci
                          if len(thisline)==10:
                              SAC_CI_spec[indx]['coeff_double'].append([int(thisline[0]),int(thisline[1]),int(thisline[2]),int(thisline[3]),float(thisline[4])])
                              SAC_CI_spec[indx]['coeff_double'].append([int(thisline[5]),int(thisline[6]),int(thisline[7]),int(thisline[8]),float(thisline[9])])
                          elif len(thisline)==5:
                              SAC_CI_spec[indx]['coeff_double'].append([int(thisline[0]),int(thisline[1]),int(thisline[2]),int(thisline[3]),float(thisline[4])])
                          else:
                              raise IOError('Wrong number of double excitation properties (must be 5 or 10)')
                      elif index=='el_trans_dip':
                          if not (' -------------------------------------------------------------------------------' in line) and not (' Symmetry  Solution' in line) and not ('energy' in line) and not blank_line:
                              if counter==0:
                                  SAC_spec['symmetry'] = thisline [0]
                                  counter += 1
                              else:
                                  SAC_CI_spec[counter]['trans_dip'] = numpy.array([float(thisline[3]),float(thisline[4]),float(thisline[5])])
                                  SAC_CI_spec[counter]['oscillator'] = float(thisline[6])
                                  counter += 1
                  elif sec_flag=='NormalModesHP':
                      if index=='Modes':
                          # Nmodes
                          # ModeNum
                          # indx=0
                          # nm_coef=numpy.zeros((3*NAtoms,Nmodes))
                          # elements=numpy.zeros(NAtoms)
                          # NNM  Number of normal modes wich has to be red
                          for ii in range(NNM):
                              nm_coef[indx,5*ModeNum+ii]=float(thisline[3+ii])
                          elements[indx//3]=int(thisline[2])
                          indx+=1
                          if indx==3*NAtoms:
                              ModeNum+=1
                              index=None
                          
                  elif sec_flag=='NormalModes':
                      if index=='Modes':
                          # Nmodes
                          # ModeNum
                          for ii in range(NNM):
                              nm_coef[3*indx+0,3*ModeNum+ii]=float(thisline[2+3*ii]) # x displacements
                              nm_coef[3*indx+1,3*ModeNum+ii]=float(thisline[3+3*ii]) # y displacements
                              nm_coef[3*indx+2,3*ModeNum+ii]=float(thisline[4+3*ii]) # x displacements
                          elements[indx]=int(thisline[2])
                          indx+=1
                          if indx==NAtoms:
                              ModeNum+=1
                              index=None
                          
    def sort_SACCI_excit_state(SAC_CI_spec):
        '''Sorts excited state by number of excited state (should be same as energy). Only for SAC-CI 
        '''
        tmp=[]
        for i_exct in range(len(SAC_CI_spec)):
          tmp.append({})
        for i_exct in range(len(SAC_CI_spec)):
          tmp[int(SAC_CI_spec[i_exct]['excitation'])]=SAC_CI_spec[i_exct]
        #for i_exct in range(len(tmp)):
        #  print(i_exct,int(tmp[i_exct]['excitation']))
          #qc.exc_spec[int(tmp[i_exct]['excitation'])]=tmp[i_exct]
          #print(i_exct,int(tmp[i_exct]['excitation']))
        return tmp
    
    # Function needed for formating double electron integrals
    def sort_table(table, cols):
        """ sort a table by multiple columns
            table: a list of lists (or tuple of tuples) where each inner list 
                   represents a row
            cols:  a list (or tuple) specifying the column numbers to sort by
                   e.g. (1,0) would sort by column 1, then by column 0
        """
        for col in reversed(cols):
            table = sorted(table, key=operator.itemgetter(col))
        return table
    
    def add_zero_integrals(TwoElInt):
        ''' Add zero integralls'''
        counter=0
        for ii in range(N_Basis):
            for jj in range(ii+1):
                for kk in range(ii+1):
                    for ll in range(kk+1):
                        if TwoElInt[counter][0]!=ii+1 or TwoElInt[counter][1]!=jj+1 or TwoElInt[counter][2]!=kk+1 or TwoElInt[counter][3]!=ll+1:
                            TwoElInt.append([ii+1,jj+1,kk+1,ll+1,0.0])
                        else:
                            #print(ii+1,jj+1,kk+1,ll+1,TwoElInt[counter][0],TwoElInt[counter][1],TwoElInt[counter][2],TwoElInt[counter][3])
                            counter+=1
                        
        print(counter)
        
    # Sorting two electron integrals and adding missing zero integrals 
    if is_TwoElInt:
        TwoElInt=sort_table(TwoElInt, (0,1,2,3))
        add_zero_integrals(TwoElInt)
        del_int=sort_table(TwoElInt, (0,1,2,3))    
        
    #print(TwoElInt)               
    #if not excit:
    #   raise IOError('No supported type of excitation calculation was found')
    if excit:
        if dict(exc_info)['calc_type']=='SAC-CI':
            print('Sorting')
            SAC_CI_spec = sort_SACCI_excit_state(SAC_CI_spec)
        if verbose:
            print('Excited states found in gaussian log file')
        if not root_def:
            exc_info.append(['root',1])
    if is_single_el:
        print('Single electron integrals found in gaussian log file')
    if is_double_el:
        print('Double electron integrals found in gaussian log file')
        counter=0
        for ii in range(len(int_keyword)):
            if int_keyword[ii]=='conventional':
                counter+=1
            elif int_keyword[ii]=='symmetry=noint':
                counter+=1
            elif int_keyword[ii]=='noraff':
                counter+=1
        #if counter>=2:
        #    raise IOError('For right output of 2 electron integrals EXTRALINKS=l316 has to be followed with scf=Conventional Symmetry=Noint Int=NoRaff')
        
    
    # Reorder data to have nice output
    for ii in range(len(Excitation_list)):
        Excitation_list[ii].multiplicity = dict(exc_info)['exct_type']
        Excitation_list[ii].root = dict(exc_info)['root']
        Excitation_list[ii].method = dict(exc_info)['calc_type']
    
    Single_el_prop={}  # Single electron properties
    if is_single_el:
        Single_el_prop['overlap']=sel_over  # overlap matrix of AO basis
        Single_el_prop['kinetic']=sel_kin   # kinetic energy matrix in AO basis
        Single_el_prop['potential']=sel_pot # potential matrix in AO basis
        Single_el_prop['core']=sel_core     # core hamiltonian in AO basis = Kinetic energy - Potential energy
        Single_el_prop['fermi']=sel_fermi   # fermi contact integrals in AO basis
    
    Double_el_integrals = None  # Double electron integrals
    if is_double_el:
        Double_el_integrals = del_int
    
    if len(nm_coef)==0:
        Normal_mode_info = None
    else:
        Normal_mode_info = [nm_coef, nm_info, hess_log] 
    
    # Nel_Alpha_Beta[0]=number of Alpha electrons
    # Nel_Alpha_Beta[1]=number of Beta electrons
    # Nat : Number of atoms in the  molecule
    # E_Nuc : Nuclear repulsion energy
    # N_Basis : number of AO basiss functions

    return job_info, Excitation_list, Normal_mode_info, Single_el_prop, Double_el_integrals, SAC_spec, SAC_CI_spec      

def read_gaussian_gjf(filename,verbose=True):
    ''' Reads gaussian imput file - under development 

    Parameters
    ----------
    filename : str
        Specifies the filename for the input gaussian log file (including the
        path if needed)
    
    Returns
    ---------
    Coor : numpy array of real (dimension Natoms x 3)
        Coordinates of every atom in the system in ANGSTROMS
    AtType : list of string
        Atom types for every atom in the system e.g. ['C','F','H','C',...]
      
    Notes
    ----------
        Applicability of this function will be extended in future according to 
        actual requirements .
        
    TODO
    ----------
    ** Rewrite reading job information. read all lines merge them and then remove
    brackets and commas **
    '''
    import re
    
    fid    = open(filename,'r')      # Open the file
    flines = fid.readlines()         # Read the WHOLE file into RAM
    fid.close() 
    
    # Job specification default:
    Run_spec={'ncpu': 12,'ram': 10,'chk': 'Default.chk','oldchk': None} # $RunGauss, %NProcShared=12, %mem=10GB
    Job_spec={}
    job_tmp=[]
    Info=[]
    Coor=[]
    AtType=[]
    
    sec_flag='run_spec'
    for line in flines:
        if '#' in line:
            sec_flag='job_spec'
            job_tmp.append(line)
        elif sec_flag=='job_spec' and (not line.strip()):
            sec_flag='info'
            job_tmp="".join(job_tmp)
        elif sec_flag=='info' and (not line.strip()):
            sec_flag='multiplicity'
        
        elif sec_flag=='job_spec':
            job_tmp.append(line)
        elif sec_flag=='info':
            Info.append(line)
        elif sec_flag=='multiplicity':
            Job_spec['charge']=int(line.split()[0])
            Job_spec['multiplicity']=int(line.split()[1])
            sec_flag='geometry'
        elif sec_flag=='geometry' and (not line.strip()):
            if 'ModRedundant' in job_tmp:
                sec_flag='constrains'
                constrains=[]
            else:
                sec_flag='aditional'
        elif sec_flag=='constrains' and (not line.strip()):
            sec_flag='aditional'
        elif sec_flag=='aditional' and (not line.strip()):
            sec_flag='end'
        
        elif sec_flag=='run_spec':
            if '%NProcShared' in line:
                Run_spec['ncpu']=int(re.findall('\d+', line)[0])
            elif '%mem' in line:
                if 'GB' in line or 'gb' in line:
                    Run_spec['ram']=int(re.findall('\d+', line)[0])
                elif 'MB' in line or 'mb' in line:
                    Run_spec['ram']=int(numpy.ceil(float(re.findall('\d+', line)[0])/100))
            elif '%chk=' in line:
                Run_spec['chk']=line[5:]
            elif '%oldchk=' in line:
                Run_spec['oldchk']=line[8:]
        elif sec_flag=='geometry':
            thisline=line.split()
            if len(thisline)!=4:
                raise IOError('So far only supported coordinate type is xyz')
#            else:
#                print('Coordinale format is xyztype')
            Coor.append([float(thisline[1]),float(thisline[2]),float(thisline[3])])
            AtType.append(thisline[0])
        elif sec_flag=='constrains':
            thisline=line.split()
            if len(thisline)!=3:
                raise IOError('So far only supported constrains are single atom constrains')
            constrains.append([thisline[0],int(thisline[1]),thisline[2]])
        
        if verbose:
            print("Warning: read_gaussian_gjf using Angstroms as default units")
            
    return Coor,AtType 

def read_VMD_pdb(filename):
    ''' Reads structure and simulation snapshots from pdb file generated by VMD
    or AMBER software.
    
    Parameters
    ----------
    filename : str
          Specifies the filename for the input file (including the path if needed)
  
    Returns
    -------
    md : MDinfo class  
          Contains most important information from MD output:
          md.NAtom   .... number of atoms in the system
          md.at_name .... atom names from PDB file
          md.NStep   .... number of steps of MD simulation
          md.geom    .... atom coordinates of every timestep in ANGSTROMS
          md.geom[:,:,i] .... atom coordinates of i-th timestep in ANGSTROMS
      
    Notes
    ----------
          ** Change units from ANGSTROMS to ATOMIC UNITS - all scripts using this 
          function have to be corrected too **
    
    '''
    
    fid    = open(filename,'r')      # Open the file
    flines = fid.readlines()         # Read the WHOLE file into RAM
    fid.close()                      # Close the file

    md = MDinfo()
    Nstep=0
    IAtom=0
    Geometry=[]     
    
    for il in range(len(flines)):
      line = flines[il]            # The current line as string
      thisline = line.split()      # The current line split into segments
      if line.rstrip():
          blank_line=False
      else:
          blank_line=True
    	# Nacteni hlavicky musi probehnout uplne nejdrive jelikoz vypis se muze lisist podle metody      
      if 'CRYST1' in line:
          PBCbox=numpy.zeros(6)
          for ii in range(6):
              PBCbox[ii]=float(thisline[ii+1])
          IAtom=0
          Geometry.append([])
      elif il==0:
          IAtom=0
          Geometry.append([])
      elif 'END' in line or 'TER' in line:
          Nstep+=1
          md.NAtom=IAtom
          IAtom=0
          Geometry.append([])
      else:
          Geometry[Nstep].append([])
          Geometry[Nstep][IAtom]=[float(thisline[-5]),float(thisline[-4]),float(thisline[-3])]
          if Nstep==0:
              md.at_name.append(thisline[2])
          IAtom+=1
              
    md.NStep=Nstep
    md.geom=numpy.zeros((md.NAtom,3,md.NStep))
    for ii in range(md.NStep):
        md.geom[:,:,ii]=numpy.array(Geometry[ii])
    print('Warning: Function read_VMD_pdb uses ANGSTROMS as default units')
    return md     
    
def read_xyz(filename,verbose=True):
    ''' Reads molecule structure from xyz file.
    
    Parameters
    ----------
    filename : str
          Specifies the filename for the input file (including the path if needed)
    verbose : logical (optional - init=True)
        Controls if warning about units is printed
  
    Returns
    -------
    Geom : numpy.array of real (dimension Nx3)
        Atomic coordinates of all atoms in ANGSTROMS
    At_type : list of characters (dimension N)
        List of atomic types of first molecule
        (for example AtType1=['C','N','C','C',...])
      
    Notes
    ----------
          ** Change units from ANGSTROMS to ATOMIC UNITS - all scripts using this 
          function have to be corrected too **
    
    '''
    
    fid    = open(filename,'r')      # Open the file
    flines = fid.readlines()         # Read the WHOLE file into RAM
    fid.close()                      # Close the file
    
    NAtom=int(flines[0])
    Geom=numpy.zeros((NAtom,3))
    At_type=[]
    for ii in range(NAtom):
        line = flines[ii+2]            # The current line as string
        thisline = line.split()      # The current line split into segments
        At_type.append(thisline[0])
        At_type[ii]=At_type[ii].capitalize()
        Geom[ii,0]=numpy.float(thisline[1])
        Geom[ii,1]=numpy.float(thisline[2])
        Geom[ii,2]=numpy.float(thisline[3])
    return Geom,At_type

def read_amber_restart(filename):
    ''' Reads molecular coordinates from AMBER restart file
    
    Parameters
    ----------
    filename : str
          Specifies the filename for the input file (including the path if needed)
  
    Returns
    -------
    Coor : numpy.array of real (dimension Nx3)
        Atomic coordinates of all atoms most probably in ANGSTROMS
   
    Notes
    ----------
          ** Change units from ANGSTROMS to ATOMIC UNITS - all scripts using this 
          function have to be corrected too **
    
    '''
    
    fid    = open(filename,'r')      # Open the file
    flines = fid.readlines()         # Read the WHOLE file into RAM
    fid.close()                      # Close the file
    line=flines[0]
    thisline = line.split()
    molname=thisline[0]
    line=flines[1]
    thisline = line.split()
    Nat=int(thisline[0])
    
    Coor=numpy.zeros((Nat,3),dtype='f8')
    for ii in range(Nat//2):
        line=flines[ii+2]
        thisline = line.split()
        Coor[2*ii,0]=float(thisline[0])
        Coor[2*ii,1]=float(thisline[1])
        Coor[2*ii,2]=float(thisline[2])
        Coor[2*ii+1,0]=float(thisline[3])
        Coor[2*ii+1,1]=float(thisline[4])
        Coor[2*ii+1,2]=float(thisline[5])
    if Nat%2==1:
        line=flines[Nat//2+2]
        thisline = line.split()
        Coor[Nat-1,0]=float(thisline[0])
        Coor[Nat-1,1]=float(thisline[1])
        Coor[Nat-1,2]=float(thisline[2])
    
    print('Warning: Function read_amber_restart uses most probably ANGSTROMS as default units')
    return Coor

def read_mol2(filename):
    ''' information about molecule from mol2 file
    
    Parameters
    ----------
    filename : str
          Specifies the filename for the input file (including the path if needed)
  
    Returns
    -------
    Coor : numpy.array of real (dimension Nx3)
        Atomic coordinates of all atoms in ANGSTROMS
    Bond : numpy.array of integer (dimension Nbond x 3)
        Specifies which pairs of atoms are bonded and also type of the bond
        Bond[ii]=[atom1_indx,atom2_indx,bond_type] where bond_type=1 for single 
        or conjugated bond and 2 for double bond and 3 for triple bond.
    Charge : numpy.array of real (dimension N)
        Array of ground state charges (default but it could be also excited state
        or transition charges for special cases) for every atom
    AtName : numpy.array of characters (dimension N)
        Atomic names - should correspond to pdb names or to atomic types from 
        gaussian input file with additional number after character which corresponds
        to position of atom in original file used for generation of mol2 file.
        For example "C1" name means first carbon in original file, 'N3' means 
        third nitrogen in original file,...
    AtType : numpy.array of characters (dimension N)
        Forcefield atom type for every atom (for example for GAFF AtType=['ca','c3','ca','ha',...])
    Molecule : numpy.array of integer (dimension N)
        Integer for every atom to which molecule it corresponds. If there is only
        single molecule in mol2 file all atoms would have index 1.
        ** So far not needed. i could be only aditional output **
    MolNameAt : numpy.array of characters (dimension N)
        For every atom 3 characters specifying name of the molecule to which it
        belongs.
        ** So far not needed. i could be only aditional output **
    Aditional : list (size 6)
        [MolName,Nat,Nbond,Ninfo,ChargeMethod,Info] where MolName is name of molecule
        writen in the begining of mol2 file (3characters). Nat is number of all 
        atoms in mol2 file. Nbond is total number of bonds defined in mol2 file.
        Ninfo is number of aditional information about molecule. ChargeMethod
        is method how the atomic charges were calculated and Info contains
        aditional information about molecule (usualy text)
   
    Notes
    ----------
          ** Change units from ANGSTROMS to ATOMIC UNITS - all scripts using this 
          function have to be corrected too **
          
          ** Change output to more basic one and write everything only if asked**
    
    '''
    fid    = open(filename,'r')      # Open the file
    flines = fid.readlines()         # Read the WHOLE file into RAM
    fid.close()                      # Close the file
    
    line=flines[1]
    thisline = line.split()
    MolName=thisline[0]
    line=flines[2]
    thisline = line.split()
    Nat=int(thisline[0])
    Nbond=int(thisline[1])
    Ninfo=int(thisline[2])
    line=flines[4]
    thisline = line.split()
    ChargeMethod=thisline[0]
    
    AtName=[]
    AtType=[]
    Coor=numpy.zeros((Nat,3),dtype='f8')
    Molecule=numpy.zeros(Nat,dtype='i8')
    MolNameAt=[]
    Charge=numpy.zeros(Nat,dtype='f8')
    # loading atom information
    for ii in range(Nat):
        line=flines[8+ii]
        thisline = line.split()
        AtName.append(thisline[1])
        Coor[ii,0]=float(thisline[2])
        Coor[ii,1]=float(thisline[3])
        Coor[ii,2]=float(thisline[4])
        AtType.append(thisline[5])
        Molecule[ii]=int(thisline[6])
        MolNameAt.append(thisline[7])
        Charge[ii]=float(thisline[8])
    
    Bond=numpy.zeros((Nbond,4),dtype='i8')
    for ii in range(Nbond):
        line=flines[9+Nat+ii]
        thisline = line.split()
        Bond[ii,0] = int(thisline[0])
        Bond[ii,1] = int(thisline[1])
        Bond[ii,2] = int(thisline[2])
        if thisline[3]=='ar':
            Bond[ii,3]=1
        else:
            Bond[ii,3] = int(thisline[3])
    
    Info=[]
    for ii in range(Ninfo): 
        line=flines[10+Nat+Nbond+ii]
        Info.append(line)
    
    print('Warning: Function read_mol2 uses ANGSTROMS as default units')
    return Coor,Bond,Charge,numpy.array(AtName),numpy.array(AtType),Molecule,numpy.array(MolNameAt),[MolName,Nat,Nbond,Ninfo,ChargeMethod,Info]

def read_AMBER_prepc(filename):
    ''' Information about molecule from AMBER prepc file
    
    Parameters
    ----------
    filename : str
          Specifies the file name for the input file (including the path if needed)
  
    Returns
    -------
    Coor : numpy.array of real (dimension Nx3)
        Atomic coordinates of all atoms in ANGSTROMS
    Charge : numpy.array of real (dimension N)
        Array of ground state charges (default but it could be also excited state
        or transition charges for special cases) for every atom
    AtType : numpy.array of characters (dimension N)
        List of atomic types (for example `AtType=['C','N','C','C',...]`)
    FFType : numpy.array of characters (dimension N)
        List of forcefield atom type for every atom. 
        (for example for GAFF `FFType=['ca','c3','ca','ha',...]`)
    MolName : string
        3 character name of the molecule. For example `MolName='CLA'` for chlorophyll a.
    INDX : numpy.array of integer
        list of integers specifying position of every atom in original file
        from which prepc file was generated. For every atom type `INDX` is from 1
        to number of atoms of that type.
   
    Notes
    ----------
          ** Change units from ANGSTROMS to ATOMIC UNITS - all scripts using this 
          function have to be corrected too **
    '''
    
    
    fid    = open(filename,'r')      # Open the file
    flines = fid.readlines()         # Read the WHOLE file into RAM
    fid.close()                      # Close the file
    
    line=flines[4]
    thisline = line.split()
    MolName=thisline[0]
    nline=10    

    AtType=[]
    FFType=[]
    Coor=[]
    Charge=[]
    Indx=[]
    Nat=0    
    
    while True:
        line=flines[nline]
        thisline = line.split()
        if not line.rstrip():
            break
        
        nline+=1
        Nat+=1

        name_tmp=thisline[1]
        name_tmp=list(name_tmp)
        NAME=[]
        INDX=[]
        for jj in range(len(name_tmp)):
            if not name_tmp[jj].isdigit():
                NAME.append(name_tmp[jj])
            else:
                INDX.append(name_tmp[jj])
        NAME="".join(NAME)
        INDX=int("".join(INDX))
        
        AtType.append(NAME)
        Indx.append(INDX)
        FFType.append(thisline[2])
        Coor.append([float(thisline[4]),float(thisline[5]),float(thisline[6])])
        Charge.append(thisline[7])
        
    Coor=numpy.array(Coor,dtype='f8')
    Indx=numpy.array(Indx,dtype='i8')
    AtType=numpy.array(AtType)
    FFType=numpy.array(FFType) 
        
    print('Warning: Function read_AMBER_prepc uses ANGSTROMS as default units')
    return Coor,Charge,AtType,FFType,MolName,Indx

def SaveVariable(filename,Variable):
    f = open(filename, 'w')
    pickle.dump(Variable, f)
    f.close()
    
def LoadVariable(filename):
    f = open(filename)
    Variable = pickle.load(f)
    f.close()
    return Variable
    
def read_AMBER_NModes(filename="vecs"):
    ''' Reads eigenvectors and eigenvalues from MD normal mode analysis from 
    AMBER program.
    
    Parameters
    ----------
    filename : str (default name is: vecs)
          Specifies the file name for the input file (including the path if needed)
          
    Returns
    -------
    Geometry : numpy.array of real (dimension Nx3)
        Array of atomic coordinates for optimal structure (for which normal mode 
        analysis is done) in ANGSTROMS
    Freq : numpy.array of real (dimension 3*N)
        Frequency of normal modes in INVERSE CENTIMETERS. First six frequencies
        correspond to translational and rotation normal mode frequency.
    NormalModes : numpy.array of real (dimension 3*Nx3*N)
        In columns are written cartessian displacements for individual normal
        modes. First 6 (5 for linear molecule) columns correspond to 
        translational and rotation eigenvectors. This matrix also transforms 
        normal mode displacements to cartesian displacement. It should be 
        dimensionless matrix. For normal mode displacement in atomic units
        we should get cartesian displacements in atomic units. The same is true
        for displacements in angstroms. If first 6(5) columnst woudl be deleted 
        it would be the same matrix as InternalToCartesian from `vibration` module.
          
    Notes
    ----------
          ** Change units from ANGSTROMS to ATOMIC UNITS - all scripts using this 
          function have to be corrected too **
          
          ** Maybe also change frequency units form INVERSE CENTIMETERS into 
          ATOMIC UNITS **
    
    AMBER input file
    ----------
    molecule m;
    float x[`3*Natoms`], fret;
    
    m = getpdb( "pdb_filename.pdb");
    readparm( m, "prmtop_filename.prmtop" );
    mm_options( "cut=999.0, ntpr=50" );
    
    setxyz_from_mol(m, NULL, x);
    mme_init( m, NULL, "::ZZZ", x, NULL);
    
    //conjugate gradient minimization
    conjgrad(x, 3*m.natoms, fret, mme, 0.001, 0.0001, 30000);
    
    //Newton-Raphson minimization
    mm_options( "ntpr=10" );
    newton( x, 3*m.natoms, fret, mme, mme2, 0.00000001, 0.0, 500 );
    
    //Output minimized structure
    setmol_from_xyz( m, NULL, x);
    putpdb( "pdb_filename_min.pdb", m );
    
    //get the normal modes:
    nmode( x, 3*m.natoms, mme2, `3*Natoms`, 0, 0.0, 0.0, 0);
    
    
    // run as:
    // nab nmode.nab 
    // ./a.out > freq.txt
    // eigenvectors are stored in vecs file and also reduced masses and information about frequencies and minimization is stored in freq.txt

    '''
    
    #--------------- Read eigenvectors-----------------------------------------
    fid    = open(filename,'r')      # Open the file
    flines = fid.readlines()         # Read the WHOLE file into RAM
    fid.close()                      # Close the file
    NAtom=int(flines[1].split()[0])//3
#    print('NAtom: ',NAtom)
    Nlines=(3*NAtom)//7
#    print('Nlines: ',Nlines)
    Opt_geom=numpy.zeros(3*NAtom)
    # geometry will be Opt_geom.reshape((NAtom,3))
    
    # Read geometry
    CLine=2
    for ii in range(Nlines):
        line=flines[CLine+ii]
        thisline = line.split()
        for jj in range(7):
            Opt_geom[ii*7+jj]=numpy.float(thisline[jj])
    CLine+=Nlines
    line=flines[CLine]
    thisline = line.split()
    for ii in range((3*NAtom)%7):
        Opt_geom[(Nlines)*7+ii]=numpy.float(thisline[ii])
    if NAtom%7!=0:
        CLine+=1
    
    # Read Normal modes and energy - first 6 normal modes eigenvectors are rotational and translational eigenvectors
        # rot and trans eigenvectors
    NormalModes=numpy.zeros((3*NAtom,3*NAtom))
    Freq=numpy.zeros(3*NAtom)
#    CLine+=12+6*Nlines
#    if NAtom%7!=0:
#        CLine+=6
        
        # vibrational eigenvectors
    for kk in range(3*NAtom):
        #first line is ****
        CLine+=1
        Freq[kk]=numpy.float(flines[CLine].split()[1])
        CLine+=1
        for ii in range(Nlines):
            line=flines[CLine+ii]
            thisline = line.split()
            for jj in range(7):
                NormalModes[ii*7+jj,kk]=numpy.float(thisline[jj])
        CLine+=Nlines
        line=flines[CLine]
        thisline = line.split()
        for ii in range((3*NAtom)%7):
            NormalModes[(Nlines)*7+ii,kk]=numpy.float(thisline[ii])
        if NAtom%7!=0:
            CLine+=1
            
    Geometry=numpy.zeros((NAtom,3))
    Geometry=Opt_geom.reshape((NAtom,3))
        
        # Freq is normal mode frequency in inverse centimeters cm-1 
        # InternalToCartesian is matrix where cartesian displacements belonging to individual normal modes are written in columns and this is also matrix which transform vector displacement in normal coordinates into cartesian displacement
        # geometry is optimal geometry of molecule in Angstroms        
    
    print('Warning: Function read_AMBER_NModes uses ANGSTROMS as default units for geometry and inverse centimeters for frequency.')
    return Geometry,Freq,NormalModes
    
def read_AMBER_Energy_Scan(filename="EnergyScan.dat"):
    ''' Read energy dependence on given coordinate generated from amber NAB program
    
        Parameters
    ----------
    filename : str (default name is: EnergyScan.dat)
          Specifies the file name for the input file (including the path if needed)
          
    Returns
    -------
    Eng : numpy.array of real (dimension N)
        List with energies along coordinate q        
    q : numpy.array of real (dimension N)
        List of coorninate displacements in ANGSTROMS
          
    Notes
    ----------
          ** Check energy units **
          ** Change units from ANGSTROMS to ATOMIC UNITS - all scripts using this 
          function have to be corrected too **
    
    '''
    
    #--------------- Read eigenvectors-----------------------------------------
    fid    = open(filename,'r')      # Open the file
    flines = fid.readlines()         # Read the WHOLE file into RAM
    fid.close()                      # Close the file
    
    Eng=[]
    q=[]
    
    for il in range(len(flines)):
      line = flines[il]            # The current line as string
      thisline = line.split()      # The current line split into segments
      if 'Energy of ' in line:
          Eng.append(numpy.float(thisline[-2]))
          q.append(numpy.float(thisline[3]))
    
    Eng=numpy.array(Eng)
    q=numpy.array(q)
    
    print('Warning: Function read_AMBER_Energy_Scan uses ANGSTROMS as default units')
    return Eng,q

def read_amber_mdout(filename):

# TODO: Write documentation
    
    fid    = open(filename,'r')   # Open the file
    flines = fid.readlines()      # Read the WHOLE file into RAM
    fid.close()
    
    section = None 
    part = None
    
    step = []
    time = []
    temp = []
    press = []
    Etot = []
    Ekin = []
    Epot = []
    Eelec = []
    vdW = []
    Vol = []
    Dens = []

    for line in flines:
        if "   2.  CONTROL  DATA  FOR  THE  RUN" in line:
            section = "input"
        elif "   4.  RESULTS" in line:
            section = "results"
        elif "      A V E R A G E S   O V E R" in line:
            section = "averages"
            count = 0
        elif "      R M S  F L U C T U A T I O N S" in line:
            section =  "fluctuations"
            count = 0
        elif "   5.  TIMINGS" in line:
            section = None
        elif "Nature and format of output:" in line and section == "input":
            part = "output"
            count = 0
        elif "Potential function:" in line and section == "input":
            part = "potential"
            count = 0
        elif "Molecular dynamics:" in line and section == "input":
            part = "MD"
            count = 0
        elif "Langevin dynamics temperature regulation:" in line and section == "input":
            part = "Temperature"
            count = 0
        elif "Pressure regulation:" in line and section == "input":
            part = "Pressure"
            count = 0
        elif " NSTEP = " in line and section == "results":
            part = "Single_step_res"
            thisline = line.split()
            step.append(int(thisline[2]))
            time.append(float(thisline[5])) # time in ps
            temp.append(float(thisline[8])) # temperature in kelvins
            press.append(float(thisline[11])) # pressure in bars
            count = 1
        else:
            if section == "input":
                thisline = line.split(",")
                if part == "output":
                    if count == 0:
                        info = thisline[1].split()
                        ntpr = int(info[2]) # frequency of write to mdout file
                        info = thisline[3].split()
                        ntwr = int(info[2]) # write frequency to restart file in steps
                    elif count == 1:
                        info = thisline[1].split()
                        ntwx = int(info[2]) # frequency of write to trajectory file
                    count += 1
                    
                elif part == "potential":
                    if count == 2:
                        info = thisline[1].split()
                        cutoff = float(info[2]) # frequency of write to mdout file
                    count += 1
                
                elif part == "MD":
                    if count == 0:
                        nstep = int(thisline[0].split()[2]) # number of MD steps 
                    elif count == 1:
                        dt = float(thisline[1].split()[2])
                        dt = dt * 1000 # timestep for the simulation if fs
                    count += 1

                elif part == "Temperature":
                    if count == 1:
                        temperature = float(thisline[0].split()[2])
                        try:
                            gamma = float(thisline[2].split()[1])
                        except:
                            gamma = None
                    count += 1
                
                elif part == "Pressure":
                    if count == 1:
                        pressure = float(thisline[0].split()[2]) # pressure in bar = 0.987atmosphere
                        taup = float(thisline[2].split()[2]) # pressure relaxation time in ps
                    count += 1 
                    
            elif section == "results" and part == "Single_step_res":
                thisline = line.split()
                if count == 1:
                    Etot.append(float(thisline[2]))
                    Ekin.append(float(thisline[5]))
                    Epot.append(float(thisline[8]))
                elif count == 3:
                    vdW.append(float(thisline[-1]))
                elif count == 4:
                    Eelec.append(float(thisline[2]))
                elif count == 5:
                    Vol.append(float(thisline[8]))
                elif count == 6:
                    Dens.append(float(thisline[2]))
                    count = -1
                    part = None
                count += 1
                
            elif section == "averages":
                thisline = line.split()
                if count == 2:
                    temp_avrg = float(thisline[8]) # temperature in kelvins
                    press_avrg = float(thisline[11]) # pressure in bars
                elif count == 3:
                    Etot_avrg = float(thisline[2])
                    Ekin_avrg = float(thisline[5])
                    Epot_avrg = float(thisline[8])
                elif count == 5:
                    vdW_avrg = float(thisline[-1])
                elif count == 6:
                    Eelec_avrg = float(thisline[2])
                elif count == 7:
                    Vol_avrg = float(thisline[8])
                elif count == 8:
                    Dens_avrg = float(thisline[2])
                    section = None
                    count = -1
                count += 1
                
            elif section == "fluctuations":
                thisline = line.split()
                if count == 2:
                    temp_rmsd = float(thisline[8]) # temperature in kelvins
                    press_rmsd = float(thisline[11]) # pressure in bars
                elif count == 3:
                    Etot_rmsd = float(thisline[2])
                    Ekin_rmsd = float(thisline[5])
                    Epot_rmsd = float(thisline[8])
                elif count == 5:
                    vdW_rmsd = float(thisline[-1])
                elif count == 6:
                    Eelec_rmsd = float(thisline[2])
                elif count == 7:
                    Vol_rmsd = float(thisline[8])
                elif count == 8:
                    Dens_rmsd = float(thisline[2])
                    section = None
                    count = -1
                count += 1
                    
    
    step = numpy.array(step, dtype='i8')
    time = numpy.array(time, dtype='f8')
    temp = numpy.array(temp, dtype='f8')
    press = numpy.array(press, dtype='f8')
    Etot = numpy.array(Etot, dtype='f8')
    Ekin = numpy.array(Ekin, dtype='f8')
    Epot = numpy.array(Epot, dtype='f8')
    Eelec = numpy.array(Eelec, dtype='f8')
    vdW = numpy.array(vdW, dtype='f8')
    Vol = numpy.array(Vol, dtype='f8')
    Dens = numpy.array(Dens, dtype='f8')

    inp = {"ntpr": ntpr, "ntwr": ntwr, "ntwx": ntwx, "cutoff": cutoff,
           "Nsteps": nstep, "time_step": dt, "temperature": temperature,
           "gamma_t": gamma, "pressure": pressure, "taup": taup}
    
    res = {"step": step, "time": time, "temperature": temp, "pressure": press,
           "Etotal": Etot, "Ekinetic": Ekin, "Epotential": Epot, 
           "Eelstat": Eelec, "EvdW": vdW, "volume": Vol, "density": Dens}
    
    avrg = {"temperature": temp_avrg, "pressure": press_avrg, 
            "Etotal": Etot_avrg, "Ekinetic": Ekin_avrg, 
            "Epotential": Epot_avrg, "Eelstat": Eelec_avrg, "EvdW": vdW_avrg,
            "volume": Vol_avrg, "density": Dens_avrg}
    
    rmsd = {"temperature": temp_rmsd, "pressure": press_rmsd, 
            "Etotal": Etot_rmsd, "Ekinetic": Ekin_rmsd, 
            "Epotential": Epot_rmsd, "Eelstat": Eelec_rmsd, "EvdW": vdW_rmsd,
            "volume": Vol_rmsd, "density": Dens_rmsd}
    # temp_avrg, press_avrg, Etot_avrg, Ekin_avrg, Epot_avrg, vdW_avrg, Eelec_avrg, Vol_avrg, Dens_avrg
    # temp_rmsd, press_rmsd, Etot_rmsd, Ekin_rmsd, Epot_rmsd, vdW_rmsd, Eelec_rmsd, Vol_rmsd, Dens_rmsd

    return res, inp, avrg, rmsd

def read_gaussian_esp(filename,output_charge=False,output_AtType=False):
    ''' Function for reading the output from gaussian Electrostatic potential
    calculation (with Pop=MK IOp(6/50=1) option)
    
    Parameters
    ----------
    filename : str
        Name of .esp file with electrostatic potential from gaussian output
    output_charge : logical (optional - init=False)
        If `output_charge=True` also fitted charges by Gaussian will be written
    
    Returns
    ----------
    Points : numpy.array (dimension Npointsx3)
        Coordiantes in Bohrs with positions of points where potential is calculated
    ESP : numpy.array (dimension Npoints)
        Vector of calculated ESP values at position specified in Points variable
        from gaussian calculation
    Coor : numpy.array (dimension Natomsx3)
        Coordinates of individual atoms of the molecule for which electrostatic
        potential was calculated
    Charge_esp : numpy.array - optional (dimension Natoms)
        List of fited charges by gaussian software from ESP calculation - these
        chares are different than charges fitted by AMBER. Charges givent in times
        of electron charge.
        
    Notes
    ----------
    Example of gaussian input:
        $RunGauss
        %NProcShared=8
        %mem=2GB
        %oldchk=chk_from_previous_calculation.chk
        %chk=chk_filename.chk              
        #p HF/6-31G* SCF=Tight 5D 7F Pop=MK IOp(6/50=1) Symmetry=(Loose,Follow) geom=checkpoint GFPrint GFInput

        Some aditional info

        0 1

        esp_filename-ini.esp

        esp_filename.esp
    '''
    print('        Reading Gaussien esp grid from file:',filename)
    fid    = open(filename,'r')   # Open the file
    flines = fid.readlines()      # Read the WHOLE file into RAM
    fid.close()
            
    line=flines[1]
    thisline = line.split()
    charge= int(thisline[2])
    multiplicity = thisline[-1]
    line=flines[2]
    thisline = line.split()
    Nat = int(thisline[-1])
            
    Coor=numpy.zeros((Nat,3),dtype='f8')
    Charge_esp=numpy.zeros(Nat,dtype='f8')
    AtType=[]
    for ii in range(Nat):
        line=flines[3+ii]
        thisline = line.split()
        AtType.append(thisline[0])
        for kk in range(3):
            if 'D' in thisline[kk+1]:
                Coor[ii,kk] = float(thisline[kk+1].replace('D', 'e'))
            else:
                Coor[ii,kk] = float(thisline[kk+1])
        kk=3
        if 'D' in thisline[kk+1]:
            Charge_esp[ii] = float(thisline[kk+1].replace('D', 'e'))
        else:
            Charge_esp[ii] = float(thisline[kk+1])
            
    #line[3+Nat],line[4+Nat] - DIPOLE MOMENT
    #line[5+Nat],line[6+Nat],line[7+Nat] -QUADRUPOLE MOMENT
    line=flines[8+Nat]
    thisline = line.split()
    Npoints=int(thisline[-1])
    ESP=numpy.zeros(Npoints,dtype='f8')
    Points=numpy.zeros((Npoints,3),dtype='f8')
    for ii in range(Npoints):
        line=flines[9+Nat+ii]
        thisline = line.split()
        for kk in range(3):
            if 'D' in thisline[kk+1]:
                Points[ii,kk] = float(thisline[kk+1].replace('D', 'e'))
            else:
                Points[ii,kk] = float(thisline[kk+1])
                    
        if 'D' in thisline[0]:
            ESP[ii] = float(thisline[0].replace('D', 'e'))
        else:
            ESP[ii] = float(thisline[0])
    
    if output_charge:
        if output_AtType:
            return Points,ESP,Coor,Charge_esp,AtType
        else:  
            return Points,ESP,Coor,Charge_esp
    else:
        if output_AtType:
            return Points,ESP,Coor,AtType
        else:
            return Points,ESP,Coor
    
def read_qchem_esp(filename):
    ''' Function for reading the output from qchem Electrostatic potential
    calculation
    
    Parameters
    ----------
    filename : str
        Name of .esp file with electrostatic potential from qchem output 
        (default is plot.esp)
    
    Returns
    ----------
    Points : numpy.array (dimension Npointsx3)
        Coordiantes in ATOMIC UNITS (Bohrs) with positions of points where 
        potential is calculated
    ESP : numpy.array (dimension Npoints)
        Vector of calculated ESP values at position specified in Points variable
        from qchem calculation
        
    Notes
    ----------
    Example of qchem input:
        
    '''
    print('        Reading Q-chem esp grid from file:',filename)
    fid    = open(filename,'r')   # Open the file
    flines = fid.readlines()      # Read the WHOLE file into RAM
    fid.close()
            
    ESP=[]
    Points=[]
    Npoints=0
    for ii in range(4,len(flines)):
        line=flines[ii]
        if line.rstrip():   # if not blank line
            Npoints+=1
    ESP=numpy.zeros(Npoints,dtype='f8')
    Points=numpy.zeros((Npoints,3),dtype='f8')
    for ii in range(Npoints):
        line=flines[ii+4]
        thisline = line.split()
        for kk in range(3):
            if 'D' in thisline[kk]:
                Points[ii,kk] = float(thisline[kk].replace('D', 'e'))
            else:
                Points[ii,kk] = float(thisline[kk])
                    
        if 'D' in thisline[3]:
            ESP[ii] = float(thisline[3].replace('D', 'e'))
        else:
            ESP[ii] = float(thisline[3])
    
    return Points,ESP
    
def read_TrEsp_charges(filename,verbose=True,dipole=False):
    ''' Read information about fited charges and geometry from charge fitting 
    of electrostatic potential from qchem calcualtion. Default output 
    file fitted_charges.out

    Parameters
    ----------
    filename : str
        Name of .out file with fited charges
    verbose : logical (optional - init=False)
        if `True` print line with fited and quantum chemistry dipole
    dipole : logical (optional - init=False)
        If `True` also calculate and output dipole calculated from fitted charges
    
    
    Returns
    ----------
    coor,charge,at_type(,DipoleTrESP)
    
    coor : numpy.array of real (dimension Nx3 where N is number of atoms)
        Atomic coordiantes in ATOMIC UNITS (Bohrs) 
    charge : numpy.array of real (dimension N)
        Vector of fited atomic ESP charges
    at_type: numpy.array of str (dimension N)
        List of atomic types (for example `AtType=['C','N','C','C',...]`)
    DipoleTrESP: numpy.array of real - optional (dimension 3)
        Dipole vector calculated from coordinates and positions which were read
        from the file (from  fitted charges) in ATOMIC UNITS (e*Bohr)
    
    '''
    
    fid    = open(filename,'r')   # Open the file
    flines = fid.readlines()      # Read the WHOLE file into RAM
    fid.close()  
    
    coor=[]
    charge=[]
    at_type=[]
    section=None
    for line in flines:
        if 'atom' in line:
            section='geom_charge'
        elif 'Total charge:' in line:
            section='None'
        elif 'Exact:' in line:
            thisline=line.split()
            DipoleExact=numpy.array([float(thisline[1]),float(thisline[2]),float(thisline[3])])
            #DipoleExact=DipoleExact/const.AuToDebye
        else:
            if section=='geom_charge':
                thisline=line.split()
                at_type.append( thisline[1].capitalize() )
                coor.append([float(thisline[2]),float(thisline[3]),float(thisline[4])])
                charge.append(float(thisline[5]))
    coor=numpy.array(coor,dtype='f8')
    charge=numpy.array(charge,dtype='f8')
    
    if verbose:
        print('Warning: Function read_TrEsp_charges uses ANGSTROMS as default units')

    if dipole:
        DipoleTrESP=numpy.dot(charge,coor/conversion_facs_position["Angstrom"]) # transformation into Bohrs           
        return coor,charge,at_type,DipoleTrESP,DipoleExact
    
    return coor,charge,at_type

# TODO: add comments
def read_qchem_grid(filename):
    
    fid    = open(filename,'r')   # Open the file
    flines = fid.readlines()      # Read the WHOLE file into RAM
    fid.close()

    Npoints=len(flines)
    positions=numpy.zeros((Npoints,3),dtype='f8')
    for ii in range(len(flines)):
        line=flines[ii]
        thisline = line.split()
        positions[ii,0]=float(thisline[0])
        positions[ii,1]=float(thisline[1])
        positions[ii,2]=float(thisline[2])
# TODO: read as coordinate type:
    
    return positions
    
#--- Support Code ---# 

# Assign the quantum number l to every AO symbol (s,p,d,etc.) 
strg='abcdefghijklmnopqrstuvwxyz'
orbit = 'spd' + strg[5:].replace('s','').replace('p','')
lquant = dict([(j, i) for i,j in enumerate(orbit)])
#del i,j

def l_deg(l=0,ao=None,cartesian_basis=True):
  '''Calculates the degeneracy of a given atomic orbitals.
  
  **Options:**
  
  Works with the molpro output nomenclature for Cartesian Harmonics:
    s->'s', p->['x','y','z'], d-> ['xx','yy', etc.], etc.
    e.g., l_deg(ao='xxy')
  
  Works with quantum number l for the Cartesian Harmonic:
    e.g., l_deg(l=1)
  
  Works with name of the Cartesian Harmonic:
    e.g., l_deg(l='p')
  ''' 
  if l=='5d':
      l=-2
      return 5
  elif l=='7f':
      l=-3
      return 7
  else:
      if ao != None:
        if ao == 's':
          return 1
        else:
          l = len(ao)
      elif isinstance(l,str):
          l = lquant[l]
  return int((l+1)*(l+2)/2) if (cartesian_basis and l>0) else (2*abs(l)+1)
  # l_deg   
