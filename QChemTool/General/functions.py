# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:54:57 2016

@author: User
"""

import numpy as np

def are_similar(A,B,treshold=0.00001,do_print=True,min_number=1e-5):
    ''' Function which check if two variables are simmilar
    
    Parameters
    ----------
    A,B : real,integer,list,numpy.array
        Variables which we would like to compare.
    treshold : real (optional parameter - init=0.00001)
        How big difference between two numbers is acceptable (in absolute value)
    do_print : logical (optional parameter - init=True)
        If `do_print`=True then print all pairs of A and B which are different.
        If `do_print`=False no written output
    min_number : real (optional parameter - init=1e-5)
        The smallest number which is treated as nonzero (in absolute value), 
        therefore numbers (-min_number,min_number) are treated as zeros
        
    Returns
    -------
    result : logical
        `True` if two variables are simmilar, `False` if not
    
    Notes
    ------- 
    '''
    result=True
    if not isinstance(A,(list,np.ndarray)):
        if isinstance(B,(list,np.ndarray)):
            result=False
            print('A is a number but B is a list')
            return result
        if A==0.0 and np.abs(B)>=treshold/100:
            result=False
            if do_print:
                print('A and B:',str(A),str(B))
            return result
        elif B==0.0 and np.abs(A)>=treshold/100:
            result=False
            if do_print:
                print('A and B:',str(A),str(B))
            return result    
        elif np.abs(A-B)>=np.abs(A)*treshold or np.abs(A-B)>=np.abs(B)*treshold:
            result=False
            if do_print:
                print('A and B:',A,B)
            return result
    else:
        if not isinstance(B,(list,np.ndarray)):
            result=False
            print('A is a list but B is a number')
            return result
        
        if len(np.shape(A))!=len(np.shape(B)):
            result=False
            print('Arrays have different shapes')
            return result
        for ii in range(len(np.shape(A))):
            if np.shape(A)[ii]!=np.shape(B)[ii]:
                result=False
                print('Arrays have different dimensions. A has:',str(np.shape(A)),'and B has:',str(np.shape(B)))
                return result
        
        if len(np.shape(A))==1:
            for ii in range(np.shape(A)[0]):
                if np.abs(A[ii]) < min_number :
                    if np.abs(B[ii])>=min_number*1.1:
                        result=False
                        if do_print:
                            print('indx, A, B:',str(ii),str(A[ii]),str(B[ii]))
                elif np.abs(B[ii]) < min_number :
                    if np.abs(A[ii])>=min_number*1.1:
                        result=False
                        if do_print:
                            print('indx, A, B:',str(ii),str(A[ii]),str(B[ii]))
                elif np.abs(A[ii]-B[ii])>=np.abs(A[ii])*treshold or np.abs(A[ii]-B[ii])>=np.abs(B[ii])*treshold:
                    result=False
                    if do_print:
                        print('indx, A, B:',str(ii),str(A[ii]),str(B[ii]))
        elif len(np.shape(A))==2:
            for ii in range(np.shape(A)[0]):
                for jj in range(np.shape(A)[1]):
                    if np.abs(A[ii,jj]) < min_number:
                        if np.abs(B[ii,jj])>=min_number*1.1:
                            result=False
                            if do_print:
                                print('indx, A, B:',str(ii),str(jj),str(A[ii,jj]),str(B[ii,jj]))
                    elif np.abs(B[ii,jj]) < min_number:
                        if np.abs(A[ii,jj])>=min_number*1.1:
                            result=False
                            if do_print:
                                print('indx, A, B:',str(ii),str(jj),str(A[ii,jj]),str(B[ii,jj]))
                    elif np.abs(A[ii,jj]-B[ii,jj])>=np.abs(A[ii,jj])*treshold or np.abs(A[ii,jj]-B[ii,jj])>=np.abs(B[ii,jj])*treshold:
                        result=False
                        if do_print:
                            print('indx, A, B:',str(ii),str(jj),str(A[ii,jj]),str(B[ii,jj]),np.abs(A[ii,jj]-B[ii,jj]),np.abs(A[ii,jj])*treshold,np.abs(B[ii,jj])*treshold)
        elif len(np.shape(A))==3:
            for ii in range(np.shape(A)[0]):
                for jj in range(np.shape(A)[1]):
                    for kk in range(np.shape(A)[2]):
                        if np.abs(A[ii,jj,kk]) < min_number:
                            if np.abs(B[ii,jj,kk])>=treshold/100:
                                result=False
                                if do_print:
                                    print('indx, A, B:',str(ii),str(jj),str(kk),str(A[ii,jj,kk]),str(B[ii,jj,kk]))
                        elif np.abs(B[ii,jj,kk]) < min_number:
                            if np.abs(A[ii,jj,kk])>=treshold/100:
                                result=False
                                if do_print:
                                    print('indx, A, B:',str(ii),str(jj),str(kk),str(A[ii,jj,kk]),str(B[ii,jj,kk]))
                        elif np.abs(A[ii,jj,kk]-B[ii,jj,kk])>=np.abs(A[ii,jj,kk])*treshold or np.abs(A[ii,jj,kk]-B[ii,jj,kk])>=np.abs(B[ii,jj,kk])*treshold:
                            result=False
                            if do_print:
                                print('indx, A, B:',str(ii),str(jj),str(kk),str(A[ii,jj,kk]),str(B[ii,jj,kk]))
        elif len(np.shape(A))==4:
            for ii in range(np.shape(A)[0]):
                for jj in range(np.shape(A)[1]):
                    for kk in range(np.shape(A)[2]):
                        for ll in range(np.shape(A)[2]):
                            if np.abs(A[ii,jj,kk,ll]) < min_number:
                                if np.abs(B[ii,jj,kk,ll])>=treshold/100:
                                    result=False
                                    if do_print:
                                        print('indx, A, B:',str(ii),str(jj),str(kk),str(ll),str(A[ii,jj,kk,ll]),str(B[ii,jj,kk,ll]))
                            elif np.abs(B[ii,jj,kk,ll]) < min_number:
                                if np.abs(A[ii,jj,kk,ll])>=treshold/100:
                                    result=False
                                    if do_print:
                                        print('indx, A, B:',str(ii),str(jj),str(kk),str(ll),str(A[ii,jj,kk,ll]),str(B[ii,jj,kk,ll]))
                            elif np.abs(A[ii,jj,kk,ll]-B[ii,jj,kk,ll])>=np.abs(A[ii,jj,kk,ll])*treshold or np.abs(A[ii,jj,kk,ll]-B[ii,jj,kk,ll])>=np.abs(B[ii,jj,kk,ll])*treshold:
                                result=False
                                if do_print:
                                    print('indx, A, B:',str(ii),str(jj),str(kk),str(ll),str(A[ii,jj,kk,ll]),str(B[ii,jj,kk,ll]))
    return result
                
