#!/usr/bin/env python3

"""
Fit lifetime decays
"""

import numpy as np
import scipy.optimize
import os
import argparse
import re
from uncertainties import ufloat


VERSION = 1.0

package_directory = os.path.dirname(os.path.abspath(__file__))


def parseCmd():
    """
    Parse the command line to get the experimental decay and instrument reponse function (IRF) file.

    Returns
    -------
    fluor_file : str
                 filename of the fluorescence decay
    irf_file : str 
               filename of the IRF (if None then the IRF is approximated by a Gaussian)
    """
    parser = argparse.ArgumentParser(
        description='Fit a series of exponential decays to an ')
    parser.add_argument('--version', action='version',
                        version='%(prog)s ' + str(VERSION))
    parser.add_argument(
        '-f', '--fluor', help='fluorescence decay file (.txt / .dat)', required=True)
    parser.add_argument('-i', '--irf',
                        help='Instrument response function (IRF) file (.txt / .dat)', required=False, default=None)
    parser.add_argument('-g', '--gauss',
                        help='Use Gaussian IRF', required=False, default=True)
    args = parser.parse_args()
    fluor_file = args.fluor
    irf_file = args.irf
    return fluor_file, irf_file


def read_decay(filepath_or_buffer, fileformat='Horiba'):
    """
    Read TCSPC decay file from HORIBA or another data format

    Parameters
    ----------
    filepath_or_buffer : str, os.PathLike, StringIO
                         filename of the decay or StringIO object
    fileformat : str, optional
                 currently implemented formats: {'HORIBA'}

    Returns
    -------
    decay_data : ndarray
                 n x 2 decay containing numbered channels and intensity counts for instrument reponse function (IRF)
    ns_per_chan : float
    """
    if isinstance(filepath_or_buffer, str):
        with open(filepath_or_buffer, 'r') as decay_file:
            decay_data, ns_per_chan = parse_file(decay_file)
    else:
        decay_data, ns_per_chan = parse_file(filepath_or_buffer, fileformat)        
    return decay_data, ns_per_chan


def parse_file(decay_file, fileformat='Horiba'):
    """
    Parse the decay file
    
    Parameters
    ----------
    decay_file : StringIO
    fileformat : str, optional
                 currently implemented formats: {'HORIBA'}

    Returns
    -------
    decay_data : ndarray
                 n x 2 decay containing numbered channels and intensity counts for instrument reponse function (IRF)
    ns_per_chan : float
    """
    if fileformat.lower() == 'horiba':
        for i, line in enumerate(decay_file):
            if 'Time' in line:
                time_found = re.search('\\d+\\.?\\d*E?-?\\d*', line)
            if 'Chan' in line:
                headerlines = i + 1
                break
        try:
            ns_per_chan = float(time_found.group())
        except (AttributeError, NameError):
            print('Timestep not defined')
            ns_per_chan = None
        try:
            decay_data = np.loadtxt(decay_file, skiprows=headerlines)
        except NameError:
            print('Number of headerlines not defined')
            decay_data = None
    elif fileformat == 'customName':
        # implement custom file reader here

        # make sure to define the following variables:
        # ns_per_chan = ...
        # headerlines = ...
        # decay_data = ...
        pass
    else:
        raise ValueError('The specified format is not available. You may define your own format in the `read_decay` function')
    return decay_data, ns_per_chan


def fit(fun, x_data, y_data, p0, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), sigma=None):
    """
    Wrapper for the curve_fit function of the scipy.optimize module
    The curve_fit optimizes the decay parameters (tau1, tau2, etc.) 
    while the nnls weights the exponential decays.

    Parameters
    ----------
    fun : callable
          The model function f(x,...) taking x values as a first argument followed by the function parameters
    x_data : array_like
             array of the independent variable
    y_data : array_like
             array of the dependent variable
    p0 : array_like
         start values for the fit model 
    bounds : 2-tuple of float or 2-tuple of array_like, optional
             lower and upper bounds for each parameter in p0. Can be either a tuple of two scalars 
             (same bound for all parameters) or a tuple of array_like with the same length as p0.
             To deactivate parameter bounds set: `bounds=(-np.inf, np.inf)`
    sigma : array_like, optional
            uncertainty of the decay (same length as y_data)

    Returns
    -------
    p : ndarray
        optimized fit parameters
    p_std : ndarray
            standard deviation of optimized fit parameters 
    """
    p, cov = scipy.optimize.curve_fit(fun, x_data, y_data, p0, bounds=bounds, sigma=sigma)
    p_std = np.sqrt(np.diag(cov))
    return p, p_std


class Lifetime:
    """
    Create lifetime class

    Parameters
    ----------
    fluor_decay : ndarray
                  n x 2 array containing numbered channels and intensity counts of the fluorescence decay
    fluor_ns_per_chan : float
                        nanoseconds per channel
    irf_decay : ndarray, optional
                n x 2 array containing numbered channels and intensity counts for instrument reponse function (IRF)
                if `None`, then IRF is approximated by a Gaussian

    Attributes
    ----------
    ns_per_chan : float
                  nanoseconds per channel
    fluor : ndarray
            n x 4 array containing time, channel number, intensity counts and associated Poissonian weights 
            of the fluorescence decay
    irf : ndarray
          n x 3 array containing time, channel number and intensity counts of the IRF
    irf_type : str
               type of IRF: {'Gaussian', 'experimental'}
    fit_param : ndarray
    fit_param_std : ndarray

    Example
    -------
    
    >>> fluor, fluor_nsperchan = lf.tcspc.read_decay(pathToFluorDecay)
    >>> irf, irf_nsperchan = lf.tcspc.read_decay(pathToIRF)
    >>> lf.tcspc.Lifetime(fluor, fluor_nsperchan, irf)
    
    """

    def __init__(self, fluor_decay, fluor_ns_per_chan, irf_decay=None, gauss_sigma=None, gauss_amp=None):
        # compute Poisson weights
        self.ns_per_chan = fluor_ns_per_chan
        time = self._get_time(fluor_decay[:, 0], fluor_ns_per_chan)
        weights = self._get_weights(fluor_decay[:, 1])
        self.fluor = np.hstack((time, fluor_decay, weights))
        if irf_decay is None:
            if gauss_sigma is None:
                self.gauss_sigma = 0.01
            else:
                self.gauss_sigma = gauss_sigma
            if gauss_amp is None:
                self.gauss_amp = np.max(fluor_decay[:, 1])
            else:
                self.gauss_amp = gauss_amp
            laser_pulse_time = self.fluor[np.argmax(self.fluor[:, 2]), 0]
            irf = self.gauss_irf(self.fluor[:, 0], laser_pulse_time, self.gauss_sigma, self.gauss_amp)
            self.irf = np.hstack((self.fluor[:, 0:2], np.array(irf, ndmin=2).T))
            self.irf_type = 'Gaussian'
        else:
            self.irf = np.hstack((np.array(irf_decay[:, 0] * fluor_ns_per_chan, ndmin=2).T, irf_decay))
            self.irf_type = 'experimental'
        self.fit_param = None
        self.fit_param_std = None
        self.fit_y = None


    @classmethod
    def from_filenames(cls, fluor_file, irf_file=None, fileformat='HORIBA', gauss_sigma=None, gauss_amp=None):
        """
        Alternative constructor for the Lifetime class by reading in filename for the fluorophore and IRF decay

        Parameters
        ----------
        fluor_file : str
                     filename of the fluorophore decay
        irf_file : str
                   filename of the IRF decay
        fileformat : str, optional
                     currently implemented formats: {'HORIBA'}

        Example
        --------

        >>> lf.tcspc.Lifetime.from_filenames(pathToFluorDecay, pathToIRFDecay)

        """
        fluor_decay, ns_per_chan = read_decay(fluor_file, fileformat='HORIBA')
        if irf_file:
            irf_decay, _ = read_decay(irf_file, fileformat='HORIBA')
        else:
            irf_decay = None
        return cls(fluor_decay, ns_per_chan, irf_decay, gauss_sigma, gauss_amp)


    @staticmethod
    def _get_time(channel, fluor_ns_per_chan):
        """
        Convert channel to time bins

        Parameters
        ----------
        channel : array_like
                  array of channel bins
        fluor_ns_per_chan = float
                            nanoseconds per channel bin

        Returns
        -------
        time : ndarray
               array pf time bins
        """
        time = np.array(channel * fluor_ns_per_chan, ndmin=2).T
        return time

    @staticmethod
    def _get_weights(decay, bg=1):
        """
        Compute Poissonian weights

        Parameters
        ----------
        decay : array_like
                array of intensity values (counts) of the decay
        bg : int, optional
             background count

        Returns
        -------
        weights: ndarray
                 array of Poissonian weights of the decay points
        """
        weights = np.array(1 / np.sqrt(decay + bg), ndmin=2).T
        return weights

    @staticmethod
    def gauss_irf(time, mu, sigma=0.01, A=10000):
        """
        Calculate a Gaussian-shaped instrument response function (IRF)

        Parameters
        ----------
        time : ndarray
               time bins
        mu : float
             mean of the Gaussian distribution
        sigma : float, optional
                standard deviation of the Gaussian distribution
        A : float, optional
            amplitude of the Gaussian distribution

        Returns
        -------
        irf : ndarray
              Gaussian shaped instrument response function (IRF)
        """
        irf = A * np.exp(-(time - mu)**2 / (2 * sigma**2)).T
        return irf

    @staticmethod
    def convolution(irf, sgl_exp):
        """
        Compute convolution of irf with a single exponential decay

        Parameters
        ----------
        irf : array_like
              intensity counts of the instrument reponse function (experimental of Gaussian shaped)
        sgl_exp : array_like
                  single-exponential decay

        Returns
        -------
        convolved : ndarray
                    convoluted signal of IRF and exponential decay
        """
        exp_fft = np.fft.fft(sgl_exp)
        irf_fft = np.fft.fft(irf)
        convolved = np.real(np.fft.ifft(exp_fft * irf_fft))
        return convolved

    @staticmethod
    def exp_decay(time, tau):
        """
        Create a single-exponential decay

        Parameters
        ----------
        time : array_like 
               time bins
        tau : float
              fluorescence lifetime

        Returns
        -------
        sgl_exp : array_like
                  single-exponential decay
        """
        sgl_exp = np.exp(-time / tau)
        return sgl_exp

    def nnls_convol_irfexp(self, x_data, p0):
        """
        Solve non-negative least squares for series of IRF-convolved single-exponential decays. 
        First, the IRF is shifted, then convolved with each exponential decay individually (decays 1,...,n), 
        merged into an m x n array (=A) and finally plugged into scipy.optimize.nnls(A, experimental y-data) to 
        compute `argmin_x || Ax - y ||_2`. This optimizes the relative weight of the exponential decays 
        whereas the curve_fit function optimizes the decay parameters (tau1, taus2, etc.)

        Parameters
        ----------
        x_data : array_like
                 array of the independent variable
        p0 : array_like
             start values for the fit model

        Returns
        -------
        A : ndarray
            matrix containing irf-convoluted single-exponential decays in the first n columns 
            and ones in the last column (background counts)
        x : ndarray
            vector that minimizes `|| Ax - y ||_2`
        y : ndarray
            fit vector computed as `y = Ax`

        """
        irf = self._irf_scaleshift(self.irf[:, 1], self.irf[:, 2], p0[0])
        decays = []
        tau0 = p0[1:]
        for t in tau0:
            decays.append(self.convolution(irf, self.exp_decay(x_data, t)))
        decays.append([1] * len(x_data))
        A = np.array(decays).T
        x, rnorm = scipy.optimize.nnls(A, self.fluor[:, 2])
        y = np.dot(A, np.array(x))
        return A, x, y

    def _model_func(self, x_data, *p0):
        """
        Wrapper function for tcspc.nnls_irfshift_convol

        Parameters
        ----------
        x_data : array_like
                 array of the independent variable
        p0 : array_like
             start values for the fit model

        See also
        --------
        nnls_convol_irfexp : Calculate non-linear least squares of IRF-convolved single-exponential decays

        Returns
        -------
        y : ndarray
            fit vector computed as `y = Ax`
        """
        A, x, y = self.nnls_convol_irfexp(x_data, p0)
        return y

    @staticmethod
    def _irf_scaleshift(channel, irf, irf_shift):
        """
        Shift IRF by n-channels (n = irf_shift)

        Parameters
        ----------
        channel : array_like
                  array of channel bins
        irf : array_like
              intensity counts of the instrument reponse function (experimental of Gaussian shaped)
        irf_shift : int
                    shift of the IRF on the time axis (in channel units)

        Returns
        -------
        irf_shifted : array_like
                      time-shifted IRF

        References
        ----------
        .. [2] J. Enderlein, *Optics Communications* **1997**
        """
        n = len(irf)
        # adapted from tcspcfit (J. Enderlein)
        irf_shifted = (1 - irf_shift + np.floor(irf_shift)) * irf[np.fmod(np.fmod(channel - np.floor(irf_shift) - 1, n) + n, n).astype(int)] + (irf_shift - np.floor(irf_shift)) * irf[np.fmod(np.fmod(channel - np.ceil(irf_shift) - 1, n) + n, n).astype(int)]
        return irf_shifted

    @staticmethod
    def average_lifetime(a, tau_val, tau_std):
        """
        Calculate average lifetime according to [1]_

        Parameters
        ----------
        a : array_like
            weighting factors of tau
        tau_val : array_like
                  fluorescence lifetimes
        tau_std : array_like
                  standard deviation of the fluorescence lifetimes

        Returns
        -------
        av_lt : tuple
                average lifetime and associated standard deviation

        References
        ----------
        .. [1] J. Lakowicz, *Principles of Fluorescence*, 3rd ed., Springer, **2010**.
        """
        tau = np.array([ufloat(val, std) for val, std in zip(tau_val, tau_std)])
        av_lt = sum(a * tau**2) / sum(a * tau)
        return av_lt.n, av_lt.s

    def reconvolution_fit(self, tau0=[1], tau_bounds=(0, np.inf), irf_shift=0, sigma=None, verbose=True):
        """
        Fit the experimental lifetime decay to a series of exponentials
        via interative reconvolution with the instrument reponse function (IRF). 

        Parameters
        ----------
        tau0 : int or array_like
               start value(s) of the fluorescence lifetime(s)
        tau_bounds : 2-tuple of float or 2-tuple of array_like, optional
                     lower and upper bounds for each parameter in tau0. Can be either a tuple of two scalars 
                     (same bound for all parameters) or a tuple of array_like with the same length as tau0.
                     To deactivate parameter bounds set: `bounds=(-np.inf, np.inf)`
        irf_shift : int, optional
                    shift of the IRF on the time axis (in channel units)
        sigma : array_like , optional
                uncertainty of the decay (same length as y_data)
        verbose : bool, optional
                  print lifetime fit result

        Example
        -------

        >>> obj.reconvolution_fit([1,5])

        """
        if type(tau0) is int or type(tau0) is float:
            tau0 = [tau0]
        n_param = len(tau0)
        p0 = [irf_shift, *[t0/self.ns_per_chan for t0 in tau0]]
        shift_bounds = [-np.inf, np.inf]
        bounds = []
        for i in range(2):
            if type(tau_bounds[i]) is int or type(tau_bounds[i]) is float:
                bounds.append([shift_bounds[i], *[tau_bounds[i]/self.ns_per_chan] * n_param])  # if bounds are specified as int/floats, i.e the same for all taus
            else:
                bounds.append([shift_bounds[i], *[tb/self.ns_per_chan for tb in tau_bounds[i]]])  # if bounds are specified as arrays, i.e individual for each tau
        p, p_std = fit(self._model_func, self.fluor[:, 1], self.fluor[:, 2], p0, bounds=bounds, sigma=sigma)
        A, x, y = self.nnls_convol_irfexp(self.fluor[:, 1], p)
        ampl = x[:-1] / sum(x[:-1])
        offset = x[-1]
        irf_shift = p[0] * self.ns_per_chan
        tau = p[1:] * self.ns_per_chan
        tau_std = p_std[1:] * self.ns_per_chan
        irf_shift_std = p_std[0] * self.ns_per_chan
        param = {'ampl': ampl, 'offset': offset, 'irf_shift': irf_shift, 'tau': tau}
        param_std = {'tau': tau_std, 'irf_shift': irf_shift_std}

        self.av_lifetime, self.av_lifetime_std = self.average_lifetime(ampl, tau, tau_std)

        if verbose:
            print('=======================================')
            print('Reconvolution fit with {} IRF'.format(self.irf_type))
            for i, (t, t_std, a) in enumerate(zip(tau, tau_std, ampl)):
                print('tau{:d}: {:0.2f} ± {:0.2f} ns ({:0.0f}%)'.format(i, t, t_std, a * 100))
            print('mean tau: {:0.2f} ± {:0.2f} ns'.format(self.av_lifetime, self.av_lifetime_std))
            print('')
            print('irf shift: {:0.2f} ns'.format(irf_shift))
            print('offset: {:0.0f} counts'.format(offset))
            print('=======================================')

        self.fit_param = param
        self.fit_param_std = param_std
        self.fit_y = y


class Anisotropy:
    """
    Create an Anisotropy object with four polarization resolved lifetime decays

    Parameters
    ----------
    VV : ndarray
         vertical excitation - vertical emission
    VH : ndarray
         vertical excitation - horizontal emission
    HV : ndarray
         horizontal excitation - vertical emission
    HH : ndarray
         horizontal excitation - horizontal emission

    Example
    --------

    >>> lf.tcspc.Anisotropy(decay['VV'], decay['VH'], decay['HV'],decay['HH'])
    
    """

    def __init__(self, VV, VH, HV, HH):
        self.VV = VV
        self.VH = VH
        self.HV = HV
        self.HH = HH
        self.G = self.G_factor(self.HV.fluor[:, 2], self.HH.fluor[:, 2])
        self.raw_r = self.aniso_decay(self.VV.fluor[:, 2], self.VH.fluor[:, 2], self.G)
        self.raw_time = self.VV.fluor[:, 0]

    @staticmethod
    def aniso_decay(VV, VH, G):
        """
        Parameters
        ----------
        VV : ndarray
             vertical excitation - vertical emission
        VH : ndarray
             vertical excitation - horizontal emission
        G : float
            G-factor

        Returns
        -------
        r: ndarray
           anisotropy decay

        Notes
        -----
        The anisotropy decay is calculated from the parallel and perpendicular lifetime decays as follows:

        .. math::
            r(t) = \\frac{I_\\text{VV} - GI_\\text{VH}}{I_\\text{VV} + 2GI_\\text{VH}}
        """
        r = np.array([(vv - G * vh) / (vv + 2 * G * vh) if (vv != 0) & (vh != 0) else np.nan for vv, vh in zip(VV, VH)])
        return r

    @staticmethod
    def G_factor(HV, HH):
        """
        Compute G-factor to correct for differences in transmittion effiency of the horizontal and vertical polarized light

        Parameters
        ----------
        HV : ndarray
             horizontal excitation - vertical emission
        HH : ndarray
             horizontal excitation - horizontal emission
        Returns
        -------
        G : float
            G-factor 

        Notes
        -----
        The G-factor is defined as follows:

        .. math::
            G = \\frac{\\int HV}{\\int HH}
        """
        G = sum(HV) / sum(HH)
        return G

    @staticmethod
    def one_rotation(time, r0, tau):
        """
        Single rotator model

        Parameters
        ----------
        time : array_like
               time bins
        r0 : float
             fundamental anisotropy
        tau_r : float
                rotational correlation time


        Returns
        -------
        ndarray
            two-rotation anisotropy decay
        """
        return r0 * np.exp(-time / tau)

    @staticmethod
    def two_rotations(time, r0, b, tau_r1, tau_r2):
        """
        Two-rotator model


        Parameters
        ----------
        time : array_like
                time bins
        r0 : float
             fundamental anisotropy
        b : float
            amplitude of second decay
        tau_r1 : float
                 first rotational correlation time
        tau_r2 : float
                 second rotational correlation time

        Returns
        -------
        ndarray
            two-rotation anisotropy decay
        """
        return r0 * np.exp(-time / tau_r1) + (r0 - b) * np.exp(-time / tau_r2)

    @staticmethod
    def hindered_rotation(time, r0, tau_r, r_inf):
        """
        Hindered rotation in-a-cone model

        Parameters
        ----------
        time : array_like
               time bins
        r0 : float
             fundamental anisotropy
        tau_r : float
                rotational correlation time
        r_inf : float
                residual anisotropy at time->inf

        Returns
        -------
        ndarray
            hindered rotation anisotropy decay
        """
        return (r0 - r_inf) * np.exp(-time / tau_r) + r_inf

    @staticmethod
    def local_global_rotation(time, r0, tau_rloc, r_inf, tau_rglob):
        """
        Local-global rotation in-a-cone model

        Parameters
        ----------
        time : array_like
               time bins
        r0 : float
             fundamental anisotropy
        tau_rloc : float
                   local rotational correlation time
        r_inf : float
                residual anisotropy at time->inf
        tau_rglob : float
                    global rotational correlation time

        Returns
        -------
        ndarray
            local-global rotation anisotropy decay
        """
        return ((r0 - r_inf) * np.exp(-time / tau_rloc) + r_inf) * np.exp(-time / tau_rglob)

    def _aniso_fitinterval(self, r, ns_before_VVmax, signal_percentage):
        """
        Determine interval for tail-fit of anisotropy decay. Outside of the fit interval the data is uncorrelated.

        Parameters
        ----------
        r : array_like
            anisotropy decay
        ns_before_VVmax : float, optional
                          how many nanoseconds before the maximum of the VV decay should the search for r0 start
        signal_percentage : float, optional
                            percentage of the VV decay serving as a treshold to define the end of the anisotropy fit interval 
        Returns
        -------
        channel_start : int
                        channel from where to start the tail fit
        channel_stop : int
                       channel where the fit should end
        """
        channel_VVmax = np.argmax(self.VV.fluor[:, 2])
        channel_start = channel_VVmax - int(ns_before_VVmax / self.VV.ns_per_chan)
        channel_stop = channel_VVmax + np.argmax(self.VV.fluor[channel_VVmax:, 2] < signal_percentage * max(self.VV.fluor[:, 2]))
        channel_start = channel_start + np.argmax(r[channel_start:channel_stop])
        return channel_start, channel_stop

    def rotation_fit(self, p0=[0.4, 1], model='one_rotation', manual_interval=None, bounds=(0, np.inf), verbose=True, ns_before_VVmax=1, signal_percentage=0.01):
        """
        Fit rotation model to anisotropy decay.

        Parameters
        ----------
        p0 : array_like
             start values of the chosen anisotropy fit model
        model : str
                one of the following anisotropy models: {'one_rotation', 'two_rotations', 'hindered_rotation', 'local_global_rotation'}
        manual_interval : 2-tuple of float, optional
        bounds : 2-tuple of float or array_like
                 lower and upper bounds for each parameter in p0. Can be either a tuple of two scalars 
                 (same bound for all parameters) or a tuple of array_like with the same length as p0.
                 To deactivate parameter bounds set: `bounds=(-np.inf, np.inf)`
        verbose : bool
                  print anisotropy fit result
        ns_before_VVmax : float, optional
                          how many nanoseconds before the maximum of the VV decay should the search for r0 start
        signal_percentage : float, optional
                            percentage of the VV decay serving as a treshold to define the end of the anisotropy fit interval 

        Example
        --------

        >>> obj.rotation_fit(p0=[0.4, 1, 10, 1], model='two_rotations')

        """
        if model == 'two_rotations':
            aniso_fn = self.two_rotations
        elif model == 'hindered_rotation':
            aniso_fn = self.hindered_rotation
        elif model == 'local_global_rotation':
            aniso_fn = self.local_global_rotation
        else:
            aniso_fn = self.one_rotation
        self.model = model

        param_names = {'one_rotation': ['r0', 'tau_r'],
                       'two_rotations': ['r0', 'b', 'tau_r', 'tau2'],
                       'hindered_rotation': ['r0', 'tau_r', 'rinf'],
                       'local_global_rotation': ['r0', 'tau_rloc', 'rinf', 'tau_rglob']}
        try:
            if not len(p0) == len(param_names[model]):
                raise ValueError
        except ValueError:
            print('Number of start parameters p0 is not consistent with the model \"{}\"'.format(model))
        else:

            if manual_interval is None:
                self.VV.reconvolution_fit(verbose=False)
                self.VH.reconvolution_fit(verbose=False)
                _fit_raw_r = self.aniso_decay(self.VV.fit_y, self.VH.fit_y, self.G)
                start, stop = self._aniso_fitinterval(_fit_raw_r, ns_before_VVmax, signal_percentage)
            else:
                start, stop = (np.argmin(abs(self.VV.fluor[:, 0] - manual_interval[i])) for i in range(2))
            self.time = self.raw_time[start:stop] - self.raw_time[start]
            self.r = self.raw_r[start:stop]
            self.fit_param, self.fit_param_std = fit(aniso_fn, self.time, self.r, p0, bounds=bounds)
            self.fit_r = aniso_fn(self.time, *self.fit_param)
            

            if verbose:
                print('====================')
                print('Anisotropy fit')
                print('model: {}'.format(self.model))
                for i, p in enumerate(param_names[model]):
                    if 'tau' in p:
                        print('{}: {:0.2f} ± {:0.2f} ns'.format(p, self.fit_param[i], self.fit_param_std[i]))
                    else:
                        print('{}: {:0.2f} ± {:0.2f}'.format(p, self.fit_param[i], self.fit_param_std[i]))
                if model == 'local_global_rotation' or model == 'hindered_rotation':
                    self.aniso_fraction = self._fraction_freeStacked(self.fit_param[0], self.fit_param[2])
                    print('free: {:0.0f}%, stacked: {:0.0f}%'.format(self.aniso_fraction[0]*100,self.aniso_fraction[1]*100))
                print('====================')

    @staticmethod
    def _fraction_freeStacked(r0, r_inf):
        """
        Calculate the fraction of free and stacked dyes based on the residual anisotropy

        Parameters
        ----------
        r0 : float
        r_inf : float

        Returns
        -------
        weights : tuple
                  fraction of free and stacked dye components
        """
        w_free = (r0-r_inf)/r0
        w_stacked = 1-w_free
        return (w_free, w_stacked)


if __name__ == "__main__":
    fluor_file, irf_file = parseCmd()
    fluor_data, fluor_nsperchan = read_decay(fluor_file)
    if irf_file is not None:
        irf_data = read_decay(irf_file)
    decay = Lifetime(fluor_data, fluor_nsperchan, irf_data)
    decay.reconvolution_fit([1, 10])
