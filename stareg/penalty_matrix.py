#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.sparse import diags
from scipy.linalg import block_diag
from scipy.signal import find_peaks

class PenaltyMatrix():
    """Implementation of the various penalty matrices for penalized B-Splines."""
    
    msg_inp = "Include n_param."

    def __init__(self):
        """Initialization.
              
        """
        self.msg_include_nparam = "Include n_param."
        self.msg_include_ydata = "Include real y_data."
        self.msg_include_basis = "Include basis."
                
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

        assert (n_param != 0), self.msg_include_nparam
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

        assert (n_param != 0), self.msg_include_nparam
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

        assert (n_param != 0), self.msg_include_nparam
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
        peak_idx : int
            Index of the peak in the data.
            
        """
        # TODO:
        # - [ ] boundary cases if peak is far left or far right
        
        assert (y_data is not None), self.msg_include_ydata
        assert (basis is not None), self.msg_include_basis
        assert (n_param != 0), self.msg_include_nparam
        peak, _ = find_peaks(x=y_data, distance=int(len(y_data)))
        assert (len(peak) == 1), "Peak not found!"
        border = np.argwhere(basis[peak,:] > 0)
        peak_idx = border[-3][1]
        inc_matrix = self.d1_difference_matrix(n_param=peak_idx+2)
        dec_matrix = -1 * self.d1_difference_matrix(n_param= n_param - peak_idx-1)
        peak = block_diag(inc_matrix[:,:-1],  dec_matrix)
        peak[peak_idx, peak_idx] = 0
        return peak, peak_idx
        
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
            
        assert (y_data is not None), self.msg_include_ydata
        assert (basis is not None), self.msg_include_basis
        assert (n_param != 0), self.msg_include_nparam
        valley, _ = find_peaks(x=-1*y_data, distance=int(len(y_data)))
        assert (len(valley) == 1), "Valley not found!"
        border = np.argwhere(basis[valley,:] > 0)
        valley_idx = border[-3][1]
        dec_matrix = -1*self.d1_difference_matrix(n_param=valley_idx+2)
        inc_matrix = self.d1_difference_matrix(n_param= n_param - valley_idx-1)
        valley = block_diag(dec_matrix[:,:-1],  inc_matrix)
        valley[valley_idx, valley_idx] = 0
        return valley
        
    def multi_peak_matrix(self, n_param=0, y_data=None, basis=None):
        """Find 2 peaks in the data and generate the penalty matrix.
        
        Note
        ----
        Monotonic increasing till the first peak, then decreasing to the 
        valley between the two peaks. Then again increasing till peak 2 and
        decreasing after peak 2.

        Parameters
        ----------
        n_param : int
            Dimension of the multi-peak constraint matrix.
        y_data : array
            Array of data to find the multi-peak locations.
        basis : np.ndarray
            BSpline basis for the X data.
            
        Returns
        -------
        P : np.ndarray
            Multi-Peak constraint matrix of size (n_param-1 x n_param)

        """
        assert (y_data is not None), self.msg_include_ydata
        assert (basis is not None), self.msg_include_basis
        assert (n_param != 0), self.msg_include_nparam
        peaks, _ = find_peaks(x=y_data, prominence=np.std(y_data), distance=int(len(y_data)/3))
        assert (len(peaks) == 2), "2 distinct peaks not found!"
        local_valley, _ = find_peaks(x=-1*y_data[peaks[0]:peaks[1]], distance=int(len(y_data[peaks[0]:peaks[1]])))
        assert (len(local_valley) == 1), "Local valley not found!"
        valley_idx = np.argwhere(basis[peaks[0]+local_valley[0], :] > 0)[1][0]
        P1 = self.peak_matrix(n_param=valley_idx, y_data=y_data[:peaks[0]+local_valley[0]], basis=basis)
        P2 = self.peak_matrix(n_param=n_param-valley_idx, y_data=y_data[peaks[0]+local_valley[0]:], basis=basis)
        P = block_diag(P1[:,:-1], 0, P2)
        P[valley_idx-2, valley_idx-1] = -1
        return P

    def multi_valley_matrix(self, n_param=0, y_data=None, basis=None):
        """Find 2 valleys in the data and generate the penalty matrix.
        
        Note
        ----
        Monotonic decreasing till the valley 1, then increasing till the 
        peak between the two valleys. Then again decreasing till valley 2 and
        increasing after valley 2.

        Parameters
        ----------
        n_param : int
            Dimension of the multi-valley constraint matrix.
        y_data : array
            Array of data to find the multi-valley locations.
        basis : np.ndarray
            BSpline basis for the X data.
            
        Returns
        -------
        V : np.ndarray
            Multi-valley constraint matrix of size (n_param-1 x n_param)

        """
        assert (y_data is not None), self.msg_include_ydata
        assert (basis is not None), self.msg_include_basis
        assert (n_param != 0), self.msg_include_nparam
        valleys, _ = find_peaks(x=-1*y_data, prominence=np.std(y_data), distance=int(len(y_data)/3))
        assert (len(valleys) == 2), "2 distinct valleys not found!"
        local_peak, _ = find_peaks(x=y_data[valleys[0]:valleys[1]], distance=int(len(y_data[valleys[0]:valleys[1]])))
        assert (len(local_peak) == 1), "Local peak not found!"
        peak_idx = np.argwhere(basis[valleys[0]+local_peak[0], :] > 0)[1][0]
        V1 = self.valley_matrix(n_param=peak_idx, y_data=y_data[:valleys[0]+local_peak[0]], basis=basis)
        V2 = self.valley_matrix(n_param=n_param-peak_idx, y_data=y_data[valleys[0]+local_peak[0]:], basis=basis)
        V = block_diag(V1[:,:-1], 0, V2)
        V[peak_idx-2,peak_idx-1] = 1
        return V

    def multi_extremum_matrix(self, n_param=0, y_data=None, basis=None):
        """Find one peak and one valley in the data and generate the penalty matrix.
        
        Note
        ----
        Either increasing to the peak, then decreasing to the valley and increasing afterwards
        or decreasing to the valley, then increasing to the peak and decreasing afterwards. 

        Parameters
        ----------
        n_param : int
            Dimension of the multi-extremum constraint matrix.
        y_data : array
            Array of data to find the extremum locations.
        basis : np.ndarray
            BSpline basis for the X data.
            
        Returns
        -------
        mulit_extremum : np.ndarray
            Multi-extremum constraint matrix of size (n_param-1 x n_param)

        """        
        assert (y_data is not None), self.msg_include_ydata
        assert (basis is not None), self.msg_include_basis
        assert (n_param != 0), self.msg_include_nparam
        peak, _ = find_peaks(x=y_data, distance=len(y_data))
        valley, _ = find_peaks(x=-1*y_data, distance=len(y_data))
        assert (len(peak) == 1), "Peak not found!"
        assert (len(valley) == 1), "Valley not found!"

        peak = np.argwhere(basis[peak[0], :] > 0)[2][0]
        valley = np.argwhere(basis[valley[0], :] > 0)[2][0]
        middle_spline = int(np.mean([peak, valley]))

        if peak < valley:
            inc_1 = self.d1_difference_matrix(n_param=peak)
            dec_1 = -1*self.d1_difference_matrix(n_param=middle_spline-peak)
            dec_2 = -1*self.d1_difference_matrix(n_param=valley-middle_spline)
            inc_2 = self.d1_difference_matrix(n_param=n_param-valley)
            E = block_diag(inc_1[:,:-1], [0], dec_1[:,:-1], [0], dec_2[:,:-1], [0], inc_2)
            E[valley-2, valley-1] = -1
            E[middle_spline-2, middle_spline-1] = -1
            E[peak-2, peak-1] = 1
        elif peak > valley:
            dec_1 = -1*self.d1_difference_matrix(n_param=valley)
            inc_1 = self.d1_difference_matrix(n_param=middle_spline-valley)
            inc_2 = self.d1_difference_matrix(n_param=peak-middle_spline)
            dec_2 = -1*self.d1_difference_matrix(n_param=n_param-peak)
            E = block_diag(dec_1[:,:-1], [0], inc_1[:,:-1], [0], inc_2[:,:-1], [0], dec_2)
            E[valley-2, valley-1] = 1
            E[middle_spline-2, middle_spline-1] = 1
            E[peak-2, peak-1] = -1
        return E


