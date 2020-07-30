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
           
        # find the peak index
        peak, properties = find_peaks(x=y_data, distance=int(len(y_data)))
        # find idx of affected splines
        border = np.argwhere(basis[peak,:] > 0)
        left_border_spline_idx = int(border[0][1])
        right_border_spline_idx = int(border[-1][1])
        # create inc, zero and dec penalty matrices for the corresponding ares
        inc_matrix = self.d1_difference_matrix(n_param=left_border_spline_idx)
        plateu_matrix = np.zeros((len(border), len(border)), dtype=np.int)
        dec_matrix = -1 * self.d1_difference_matrix(n_param= n_param - right_border_spline_idx)

        peak = block_diag(*[inc_matrix[:,:-1], plateu_matrix, dec_matrix])
        peak[left_border_spline_idx-2, left_border_spline_idx-1] = 1
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

        # find the valley index
        valley, properties = find_peaks(x=-y_data, distance=int(len(y_data)))
        # find idx of affected splines
        border = np.argwhere(basis[valley,:] > 0)
        left_border_spline_idx = int(border[0][1])
        right_border_spline_idx = int(border[-1][1])
        # create dec, zero and inc penalty matrices for the corresponding ares
        inc_matrix = -1* self.d1_difference_matrix(n_param=left_border_spline_idx)
        plateu_matrix = np.zeros((len(border), len(border)), dtype=np.int)
        dec_matrix = self.d1_difference_matrix(n_param= n_param - right_border_spline_idx)
        valley = block_diag(*[inc_matrix[:,:-1], plateu_matrix, dec_matrix])
        valley[left_border_spline_idx-2, left_border_spline_idx-1] = -1
        return valley
        