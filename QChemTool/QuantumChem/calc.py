import numpy
    
def GuessBonds(rr,bond_length=4.0,**kwargs):
    ''' Function guesses pairs of atoms between which bond might occure.
    rr  ...............  Atomic positions in Bohr (AtomicUnits) (or atomic orbital position) shape=(Natoms,3)
    bond_length .......  Bond length offset for pair of atoms in Bohr (AtomicUnits). Should be dependent on atomic types (H-... bond reuire smaller offset than f.e. C-C bond)
    **kwargs ..........  for example: AtType={['C','N','C','C',...]} (dictionary type) (
    
    example of **kwargs: dictionary={'AtType': numpy.array([1,2,3,4,5])},dictionary['AtType'] -> array([1, 2, 3, 4, 5]),  list(dictionary.keys()) -> ['AtType']
    '''    
    Nat=len(rr)
    
    is_AtType=False
    for key in list(kwargs.keys()):
        if key=='AtType':
            AtType=kwargs['AtType']
            is_AtType=True
    
    Bonds=[]
    if is_AtType:
        for ii in range(Nat):
            for jj in range(ii+1,Nat):
                dr=rr[jj,:]-rr[ii,:]
                if AtType[ii]=='H' or AtType[jj]=='H':
                    if numpy.sqrt(numpy.dot(dr,dr))<bond_length/1.8:
                        Bonds.append([ii,jj])
                else:
                    if numpy.sqrt(numpy.dot(dr,dr))<bond_length:
                        Bonds.append([ii,jj])
    else:
        for ii in range(Nat):
            for jj in range(ii+1,Nat):
                dr=rr[jj,:]-rr[ii,:]
                if numpy.sqrt(numpy.dot(dr,dr))<bond_length:
                    Bonds.append([ii,jj])
    
    return numpy.array(Bonds,dtype='i8')


def center_grid(ac):
  '''Centers the grid to the point ac and to the origin (0,0,0).
  '''
  # All grid related variables should be globals 
  global x, y, z, d3r, min_, max_, N_, delta_
  
  P=[numpy.zeros((3,1)), numpy.reshape(ac,(3,1))]
  
  d_tilde = numpy.abs(P[0] - P[1])
  N_tilde = numpy.round(numpy.abs(d_tilde / delta_))
  
  for ii in range(3): 
    if N_tilde[ii] != 0:
      delta_[ii] = d_tilde[ii] / N_tilde[ii]
  
  grid = [x, y, z]
  
  for ii in range(3):
    if len(grid[ii]) != 1:
      position = numpy.nonzero(ac[ii] <= grid[ii])[0][0]
      g = numpy.abs(grid[ii][position] - ac[ii]);
      c = 1/2.*delta_[ii] - g;
      grid[ii] += c;
  
  x = grid[0]  
  y = grid[1]  
  z = grid[2]
  d3r = numpy.product(delta_)
  
  min_ = [min(grid[0]), min(grid[1]), min(grid[2])]
  max_ = [max(grid[0]), max(grid[1]), max(grid[2])]
  N_   = [len(grid[0]), len(grid[1]), len(grid[2])]
  
  for ii in range(3):
    if len(numpy.nonzero(0. == numpy.round(grid[ii]*10000))[0])!= 0: 
      print('Warning!\n\tAt least one grid point is equal to zero.\n')
  
  return 0
  
def rsmd(coor1,coor2):
    if len(coor1)!=len(coor2):
        raise IOError('Both coordiantes have to have same shape')
    res=0.0
    for ii in range(len(coor1)):
        dr=coor1[ii]-coor2[ii]
        res+=numpy.dot(dr,dr)
    res=res/len(coor1)
    res=numpy.sqrt(res)
    return res

def identify_molecule(struc,struc_test,indx_center1,indx_x1,indx_y1,indx_center_test,indx_x_test,indx_y_test,onlyC=False):
    import Program_Test.QuantumChem.positioningTools as pos
   
    if struc.coor.units!=struc_test.coor.units:
        struc.coor.Angst2Bohr()
        struc_test.coor.Angst2Bohr()
        
    
    Coor_alligned=pos.AlignMolecules(struc.coor.value,struc_test.coor.value,indx_center1,indx_x1,indx_y1,indx_center_test,indx_x_test,indx_y_test)
    # Toto by melo posunout optimalizovanou molekulu do pozice te na fluorofrafenu
    
    # By comparison distance between same atomic types we can guess correspondding atoms
    if onlyC:
        counter=0
        for i in range(struc_test.nat):
            if struc_test.at_type[i]=='C':
                counter+=1
        index1=numpy.zeros(counter,dtype='i8')
        #print('Number of carbons in testing molecule:',len(index1))
        counter=0
        for i in range(struc_test.nat):
            r1=Coor_alligned[i]
            type1=struc_test.at_type[i]
            if type1=='C':
                min_distance=20.0
                for j in range(struc.nat):
                    r2=struc.coor.value[j]
                    type2=struc.at_type[j]
                    if type1==type2:
                        dist=numpy.sqrt(numpy.dot(r1-r2,r1-r2))
                        if dist<min_distance:
                            min_distance=numpy.copy(dist)
                            index1[counter]=j
                counter+=1
    else:
        index1=numpy.zeros(struc_test.nat,dtype='i8')
        #print('Number of atoms in testing molecule:',len(index1))
        for i in range(struc_test.nat):
            #r1=mol_test.at_spec['Coor'][i]
            r1=Coor_alligned[i]
            type1=struc_test.at_type[i]
            min_distance=20.0
            for j in range(struc.nat):
                r2=struc.coor.value[j]
                type2=struc.at_type[j]
                if type1==type2:
                    dist=numpy.sqrt(numpy.dot(r1-r2,r1-r2))
                    if dist<min_distance:
                        min_distance=numpy.copy(dist)
                        index1[i]=j
    return index1

def molecule_osc_3D(rr,bond,factor,NMN,TrDip,centered,nearest_neighbour,verbose=False):
    from Program.QuantumChem.interaction import dipole_dipole 
    cutoff_dist=4.0
    TrDip_norm=numpy.linalg.norm(TrDip)
    
    Ndip=len(bond)
    ro=numpy.zeros((Ndip,3),dtype='f8')
    do=numpy.zeros((Ndip,3),dtype='f8')
    
    if centered=="Bonds":
        # Place unity dipole moments in centers of all bonds
        for ii in range(Ndip):
            do[ii,:]=rr[bond[ii,1],:]-rr[bond[ii,0],:]
            norm=numpy.sqrt(numpy.dot(do[ii,:],do[ii,:]))
            do[ii,:]=do[ii,:]/norm*factor[ii]
            ro[ii,:]=(rr[bond[ii,1],:]+rr[bond[ii,0],:])/2
    elif centered=="Atoms":
        # Place unity dipole moments in position of atoms
        raise IOError("Atom centered dipoles not yet supported - no clear way how to orient them")
    else:
         raise IOError("Unknown type of dipole centering. Alowed types are: 'Bonds' or 'Atoms'.")
    
    
    # Calculate dipole-dipole interaction energy between all dipoles
    hh=numpy.zeros((Ndip,Ndip),dtype='f8')
    for ii in range(Ndip):
        for jj in range(ii+1,Ndip):
            if nearest_neighbour:
                dr=numpy.linalg.norm(ro[ii,:]-ro[jj,:])
                if dr<=cutoff_dist:
                    hh[ii,jj]=dipole_dipole(ro[ii,:],do[ii,:],ro[jj,:],do[jj,:])
                    hh[jj,ii]=hh[ii,jj]
            else:
                hh[ii,jj]=dipole_dipole(ro[ii,:],do[ii,:],ro[jj,:],do[jj,:])
                hh[jj,ii]=hh[ii,jj]
  
    
    # Calculate normal modes (eigenvectors and eigenvalues of hh)
    val,vec=numpy.linalg.eigh(hh) # val= vector with eigenvalues and vec= matrix with eigenvectors in columns

    if verbose:
        for jj in range(6):
            Dip=numpy.dot(vec[:,jj],do)
            print('state',jj,'dipole:',Dip,'dx/dy:',Dip[0]/Dip[1])
    
    # rescaling normal mode to have specified transition dipole
    Dip=numpy.dot(vec[:,NMN],do)
    norm=numpy.sqrt(numpy.dot(Dip,Dip))
    
    for ii in range(Ndip):
        do[ii,:]=do[ii,:]*vec[ii,NMN]*TrDip_norm/norm
    
    return ro,do     