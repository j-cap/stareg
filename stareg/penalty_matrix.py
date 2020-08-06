#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.sparse import diags
from scipy.linalg import block_diag
from scipy.signal import find_peaks

class PenaltyMatrix():
    """Implementation of the various penalty matrices for penalized B-Splines."""
    
    def __init__(self):
        """Initialization.
              
        """
        pass
                
    def d1_difference_matrix(self, n_param=0):
        """Create the first order difference matrix.  
        
        Parameters
        ----------
        n_param : int
            Dimension of the difference matrix.
        
        Returns
        -------
        d1 : np.ndarray
            D1 penalty matrix of size (n_param-1 x n_param), 

        """        

        assert (n_param != 0), "Include n_param!!!"
        d = np.array([-1*np.ones(n_param), np.ones(n_param)])
        offset=[0,1]
        d1 = diags(d,offset, dtype=np.int).toarray()
        d1 = d1[:-1,:]
        return d1    

    def d2_difference_matrix(self, n_param=0):
        """Create the second order difference matrix. 

        Parameters
        ----------
        n_param : int
            Dimension of the difference matrix.

        Returns
        -------
        d2 : np.ndarray 
            D2 penalty matrix of size (n_param-1 x n_param), 

        """

        assert (n_param != 0), "Include n_param!!!"
        d = np.array([np.ones(n_param), -2*np.ones(n_param), np.ones(n_param)])
        offset=[0,1,2]
        d2 = diags(d,offset, dtype=np.int).toarray()
        d2 = d2[:-2,:]
        return d2
    
    def smoothness_matrix(self, n_param=0):
        """Create the smoothness penalty matrix according to Hofner 2012.
        
        Parameters
        ------------
        n_param : int
            Dimension of the smoothnes matrix.
        
        Returns
        -------
        s : np.ndarray
            Smoothnes constraint matrix of size (n_param-2 x n_param).

        """

        assert (n_param != 0), "Include n_param!!!"
        n_param = int(np.product(n_param))
        s = np.array([np.ones(n_param), -2*np.ones(n_param), np.ones(n_param)])
        offset=[0,1,2]
        smoothness = diags(s,offset, dtype=np.int).toarray()
        smoothness = smoothness[:-2,:]        
        return smoothness
            
    
    def peak_matrix(self, n_param=0, y_data=None, basis=None):
        """Create the peak constraint matrix. 
        
        Note
        ---
        Monotonic increasing till the peak, then monotonic decreasing.
        
        Parameters
        ----------
        n_param : int
            Dimension of the peak matrix.
        y_data : array
            Array of data to find the peak location.
        basis : ndarray
            BSpline basis for the X data. 
        
        Returns
        -------
        peak : np.ndarray
            Peak constraint matrix of size (n_param-1 x n_param)

        """
        # TODO:
        # - [ ] boundary cases if peak is far left or far right
        
        assert (y_data is not None), "Include real y_data!!!"
        assert (basis is not None), "Include basis!"
        assert (n_param != 0), "Include n_param!!!"
           
        peak, _ = find_peaks(x=y_data, distance=int(len(y_data)))
        border = np.argwhere(basis[peak,:] > 0)
        peak_idx = border[-3][1]
        inc_matrix = self.d1_difference_matrix(n_param=peak_idx+2)
        dec_matrix = -1 * self.d1_difference_matrix(n_param= n_param - peak_idx-1)
        peak = block_diag(inc_matrix[:,:-1],  dec_matrix)
        peak[peak_idx, peak_idx] = 0

        return peak
        
    def valley_matrix(self, n_param=0, y_data=None, basis=None):
        """Create the valley constraint matrix. 
        
        Note
        ---
        Monotonic decreasing till the valley, then monotonic increasing.
        
        Parameters
        ----------
        n_param : int
            Dimension of the valley constraint matrix.
        y_data : array
            Array of data to find the valley location.
        basis : np.ndarray
            BSpline basis for the X data.
            
        Returns
        -------
        valley : np.ndarray
            Valley constraint matrix of size (n_param-1 x n_param)

        """
        # TODO:
        # - [ ] boundary cases if valley is far left or far right
                
        assert (y_data is not None), "Include real y_data!!!"
        assert (basis is not None), "Include basis!"
        assert (n_param != 0), "Include n_param!!!"

        # here are some coments
        #with are pretty useless
        valley, _ = find_peaks(x=-1*y_data, distance=int(len(y_data)))
        border = np.argwhere(basis[valley,:] > 0)
        valley_idx = border[-3][1]
        dec_matrix = self.d1_difference_matrix(n_param=valley_idx+2)
        inc_matrix = -1 * self.d1_difference_matrix(n_param= n_param - valley_idx-1)
        valley = block_diag(inc_matrix[:,:-1],  dec_matrix)
        valley[valley_idx, valley_idx] = 0
        return valley
        