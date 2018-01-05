# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:09:34 2018

@author: Vladislav Sláma
"""

import numpy as np
from QChemTool.Polarizable_atoms.Polarization_module_HeteroDimer import Dielectric

'''----------------------- TEST PART --------------------------------'''
if __name__=="__main__":

    print('                TESTS')
    print('-----------------------------------------')    
    
    ''' Test derivation of energy d/dR ApB '''

    # SETUP VERY SIMPLE SYSTEM OF TWO DEFECT ATOMS AND ONE ENVIRONMENT ATOM:
    
    coor=np.array([[-1.0,0.0,0.0],[0.0,0.0,0.0],[1.0,0.0,0.0]],dtype='f8')
    charge_pol=np.zeros(3,dtype='f8')
    dipole=np.zeros((len(coor),3),dtype='f8')
    
    # definition of defect atoms and corresponding charges     
    charge1=np.array([1.0],dtype='f8')
    index1=[0]
    charge2=np.array([1.0],dtype='f8')
    index2=[2]
    
    charge_pol[index1]=charge1
    charge_pol[index2]=charge2
    
    # polarizability only on environmnet atom
    AlphaE=np.array([np.zeros((3,3)),[[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,0.0]],np.zeros((3,3))],dtype='f8')
    
    pol_mol=Dielectric(coor,charge_pol,dipole,AlphaE,AlphaE,AlphaE,0.0)
    
    res_general=pol_mol._dR_BpA(index1,index2,charge1,charge2,'AlphaE')
    
    result=np.zeros((3,3),dtype='f8')
    result2=np.array([[-4.0,0.0,0.0],[0.0,0.0,0.0],[4.0,0.0,0.0]],dtype='f8').reshape(3*len(coor))
    R01=coor[1,:]-coor[0,:]
    RR01=np.sqrt(np.dot(R01,R01))
    R21=coor[1,:]-coor[2,:]
    RR21=np.sqrt(np.dot(R21,R21))
    dn=np.dot(AlphaE[1],R21/(RR21**3))
    result[0,:]=charge1[0]*charge2[0]*(3*np.dot(R01/(RR01**5),dn)*R01-1/(RR01**3)*dn)
    dn=np.dot(AlphaE[1],R01/(RR01**3))
    result[2,:]=charge1[0]*charge2[0]*(3*np.dot(R21/(RR21**5),dn)*R21-1/(RR21**3)*dn)
    result = result.reshape(3*len(coor))

    if np.allclose(res_general,result2):
        print('Symm _dR_BpA simple system      ...    OK')
    else:
        print('Symm _dR_BpA simple system      ...    Error')
        print('     General result:   ',res_general)
        print('     Analytical result:',result2)
    

    charge_pol=np.zeros(3,dtype='f8')
    charge_pol[index2]=charge2          # Charge only on defect B
    pol_mol=Dielectric(coor,charge_pol,dipole,AlphaE,AlphaE,AlphaE,0.0)
    result3=np.array([[8.0,0.0,0.0],[-8.0,0.0,0.0]],dtype='f8').reshape(6)
    res_general=pol_mol._dR_BpA(index2,index2,charge2,charge2,'AlphaE')
    if np.allclose(res_general[3:9],result3):
        print('Symm _dR_BpB simple system      ...    OK')
    else:
        print('Symm _dR_BpB simple system      ...    Error')
        print('     General result:   ',res_general)
        print('     Analytical result:',result3)
        
    
    # SETUP NON-SYMETRIC SIMPLE SYSTEM OF TWO DEFECT ATOMS AND ONE ENVIRONMENT ATOM:
    coor=np.array([[-1.0,0.0,0.0],[0.0,0.0,0.0],[1.0,2.0,0.0]],dtype='f8')
    charge_pol=np.zeros(3,dtype='f8')
    dipole=np.zeros((len(coor),3),dtype='f8')
    AlphaE=np.array([np.zeros((3,3)),[[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,0.0]],np.zeros((3,3))],dtype='f8')

    # definition of defect atoms and corresponding charges     
    charge1=np.array([1.0],dtype='f8')
    charge2=np.array([1.0],dtype='f8')
    index1=[0]
    index2=[2]
    charge_pol[index1]=charge1
    charge_pol[index2]=charge2
    
    pol_mol=Dielectric(coor,charge_pol,dipole,AlphaE,AlphaE,AlphaE,0.0)
    
    res_general=pol_mol._dR_BpA(index1,index2,charge1,charge2,'AlphaE')
#    
#    result=np.zeros((3,3),dtype='f8')
    result2=np.array([[-4.0/np.sqrt(5)**3,4.0/np.sqrt(5)**3,0.0],
                      [6*(1/np.sqrt(5)**3-1/np.sqrt(5)**5),-4/np.sqrt(5)**3-12/np.sqrt(5)**5,0.0],
                      [6/np.sqrt(5)**5-2/np.sqrt(5)**3,12/np.sqrt(5)**5,0.0]],dtype='f8').reshape(3*len(coor))
    
    result=np.zeros((3,3),dtype='f8')
    R01=coor[1,:]-coor[0,:]
    RR01=np.sqrt(np.dot(R01,R01))
    R21=coor[1,:]-coor[2,:]
    RR21=np.sqrt(np.dot(R21,R21))
    dn=np.dot(AlphaE[1],R21/(RR21**3))
    result[0,:]=charge1[0]*charge2[0]*(3*np.dot(R01/(RR01**5),dn)*R01-1/(RR01**3)*dn)
    dn=np.dot(AlphaE[1],R01/(RR01**3))
    result[2,:]=charge1[0]*charge2[0]*(3*np.dot(R21/(RR21**5),dn)*R21-1/(RR21**3)*dn)
    #print(result2)
    #print(result)
    if np.allclose(res_general,result2):
        print('non-Symm _dR_BpA simple system  ...    OK')
    else:
        print('non-Symm _dR_BpA simple system  ...    Error')
        print('     General result:   ',res_general)
        print('     Analytical result:',result2)
            
    # For calculation of d_BpB charges on defect A has to be zero
    pol_mol.charge[index1]=0.0
    # or
    #charge1=np.array([0.0],dtype='f8')
    #charge2=np.array([1.0],dtype='f8')
    #index1=[0]
    #index2=[2]
    #charge_pol[index1]=charge1
    #charge_pol[index2]=charge2
    #pol_mol=Dielectric(coor,charge_pol,dipole,AlphaE,AlphaE,AlphaE,0.0)
    
    result3=np.array([[0.064,0.128,0.0],[-0.064,-0.128,0.0]],dtype='f8').reshape(6)
    res_general=pol_mol._dR_BpA(index2,index2,charge2,charge2,'AlphaE')
    
    if np.allclose(res_general[3:9],result3):
        print('non-Symm _dR_BpB simple system  ...    OK')
    else:
        print('non-Symm _dR_BpB simple system  ...    Error')
        print('     General result:   ',res_general)
        print('     Analytical result:',result3)
        
        
    
    # SETUP LITTLE BIT MORE COMPLICATED SYSTEM OF 2 DEFECT ATOMS AND 2ENVIRONMENT ATOMS
    for kk in range(2): 
        if kk==0:
            coor=np.array([[-2.0,0.0,0.0],[-2.0,-1.0,0.0],[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[2.0,1.0,0.0]],dtype='f8')
        else:
            coor=np.array([[-2.0,0.0,0.0],[-2.0,1.0,0.0],[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[2.0,1.0,0.0]],dtype='f8')
        charge_pol=np.zeros(len(coor),dtype='f8')
        dipole=np.zeros((len(coor),3),dtype='f8')
        AlphaE=np.array([np.zeros((3,3)),np.zeros((3,3)),
                         [[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,0.0]],
                         [[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,0.0]],
                         np.zeros((3,3)),np.zeros((3,3))],dtype='f8')
    
        # definition of defect atoms and corresponding charges     
        charge1=np.array([1.0,-1.0],dtype='f8')
        index1=[0,1]
        charge2=np.array([1.0,-1.0],dtype='f8')
        index2=[4,5]
        charge_pol[index1]=charge1
        charge_pol[index2]=charge2
        
        pol_mol=Dielectric(coor,charge_pol,dipole,AlphaE,AlphaE,AlphaE,0.0)
        
        res_general=pol_mol._dR_BpA(index1,index2,charge1,charge2,'AlphaE')
        
        if kk==0:
            # for coor[1]=[-2.0,-1.0,0.0]
            result2=np.array([[-0.1313271490,-0.04854981982,0.0],[0.04798957640,0.07411449339,0.0],
                              [0.0,0.0,0.0],[-0.04637925945,-0.08345754376,0.0],
                              [0.1005284061,0.08560623298,0.0],
                              [0.02918842589,-0.02771336278,0.0]],dtype='f8').reshape(3*len(coor))  
        else:
            # for coor[1]=[-2.0,1.0,0.0]
            result2=np.array([[-0.131327,-0.0485498,0.0],[0.126639,-0.0300095,0.0],
                          [0.0,0.0624526,0.0],[-0.0195464,0.138987,0.0],
                          [0.100528,-0.0856062,0.0],[-0.0762936,-0.037274,0.0]],dtype='f8').reshape(3*len(coor))
        if np.allclose(res_general,result2):
            print('non-Symm _dR_BpA system',kk+1,'      ...    OK')
        else:
            print('non-Symm _dR_BpA system',kk+1,'      ...    Error')
            print('     General result:   ',res_general)
            print('     Analytical result:',result2)
    

        if kk==1:
            pol_mol.charge[index2]=0.0 # For calculation of d_ApA charges on defect B has to be zero
            res_general=pol_mol._dR_BpA(index1,index1,charge1,charge1,'AlphaE')
            result3=np.array([[0.0759272,-0.0494062,0.0],[0.00288743,0.0479804,0.0],
                          [-0.0738948,0.0013901,0.0],[-0.00491991,0.00003574515217,0.0]],dtype='f8').reshape(12)
            if np.allclose(res_general[0:12],result3):
                print('non-Symm _dR_ApA system',kk+1,'      ...    OK')
            else:
                print('non-Symm _dR_ApA system',kk+1,'      ...    Error')
                print('     General result:   ',res_general)
                print('     Analytical result:',result3)

                    
    ''' Test derivation of energy d/dR BppA '''
    # SETUP NON-SYMETRIC SIMPLE SYSTEM OF TWO DEFECT ATOMS AND TWO ENVIRONMENT ATOM:
    coor=np.array([[-1.0,0.0,0.0],[0.0,0.0,0.0],[0.0,1.0,0.0],[1.0,0.0,0.0]],dtype='f8')
    charge_pol=np.zeros(len(coor),dtype='f8')
    dipole=np.zeros((len(coor),3),dtype='f8')
    AlphaE=np.array([np.zeros((3,3)),[[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,0.0]],
                     [[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,0.0]],np.zeros((3,3))],dtype='f8')
    
    # definition of defect atoms and corresponding charges     
    charge1=np.array([1.0],dtype='f8')
    index1=[0]
    charge2=np.array([1.0],dtype='f8')
    index2=[3]
    charge_pol[index1]=charge1
    charge_pol[index2]=charge2
    
    pol_mol=Dielectric(coor,charge_pol,dipole,AlphaE,AlphaE,AlphaE,0.0)
    
    res_general=pol_mol._dR_BppA(index1,index2,charge1,charge2,'AlphaE')

    result2=np.array([[3.535533906,-0.7071067812,0.0],[0.0,14.14213562,0.0],
                      [0.0,-12.72792206,0.0],[-3.535533906,-0.7071067812,0.0],
                      ],dtype='f8').reshape(3*len(coor))

    if np.allclose(res_general,result2):
        print('non-Symm _dR_BppA simple system ...    OK')
    else:
        print('non-Symm _dR_BppA simple system ...    Error')
        print('     General result:   ',res_general)
        print('     Analytical result:',result2)

    pol_mol.charge[index2]=0.0 # For calculation of d_ApA charges on defect B has to be zero
    res_general=pol_mol._dR_BppA(index1,index1,charge1,charge1,'AlphaE')
    result3=np.array([[-7.071067812,-9.899494937,0.0],[-2.8284271247,-2.8284271247,0.0],
                      [9.899494937,12.72792206,0.0],
                      ],dtype='f8').reshape(9)
    if np.allclose(res_general[0:9],result3):
        print('non-Symm _dR_AppA simple system ...    OK')
    else:
        print('non-Symm _dR_AppA simple system ...    Error')
        print('     General result:   ',res_general[0:9])
        print('     Analytical result:',result3)
        
# TODO: TEST heterodimer - diferent charges, diferent number of atoms in the defect A-2 B-1...