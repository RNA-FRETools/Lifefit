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
    fluor_file : string
    irf_file : string
    use_gauss : bool
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
    if irf_file is not None:
        use_gauss = False
    else:
        use_gauss = args.gauss
    return fluor_file, irf_file, use_gauss


def read_decay(decay_file, fileformat='HORIBA'):
    """
    Read TCSPC decay file from HORIBA

    Parameters
    ----------
    decay_file : str
    fileformat : str
                 currently implemented formats: {'HORIBA'}

    Returns
    -------
    decay_data : ndarray
                 n x 2 decay containing numbered channels and intensity counts for instrument reponse function (IRF)
    ns_per_chan : float
    """
    with open(decay_file, 'r') as dec:
        if fileformat == 'HORIBA':
            for i, line in enumerate(dec):
                if 'Time' in line:
                    ns_per_chan = float(re.findall('\d*\.?\d+E?-?\d*', line)[0])
                if 'Chan' in line:
                    headerlines = i + 1
                    break
            decay_data = np.loadtxt(decay_file, skiprows=headerlines)
        elif fileformat == 'customName':
            # implement custom file reader here

            # make sure to define the following variables:
            # ns_per_chan = ...
            # headerlines = ...
            # decay_data = ...
            pass

    return decay_data, ns_per_chan


def fit(fun, x_data, y_data, p0, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), sigma=None):
    """
    Wrapper for the curve_fit function of the scipy.optimize module

    Parameters
    ----------
    fun : callable
          The model function f(x,...) taking x values as a first argument followed by the function parameters
    x_data : array_like
    y_data : array_like
    p0 : array_like
    bounds : 2-tuple of float or array_like
             lower and upper bounds for each parameter in p0. Can be either a tuple of two scalars 
             (same bound for all parameters) or a tuple of array_like with the same length as p0.
             To deactivate parameter bounds set: `bounds=(-np.inf, np.inf)`
    sigma : None or array_like (same length as y_data)
            weights of the y_data

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
    Create lifetime class ss

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

    """

    def __init__(self, fluor_decay, fluor_ns_per_chan, irf_decay=None):
        # compute Poisson weights
        self.ns_per_chan = fluor_ns_per_chan
        time = self._get_time(fluor_decay[:, 0], fluor_ns_per_chan)
        weights = self._get_weights(fluor_decay[:, 1])
        self.fluor = np.hstack((time, fluor_decay, weights))
        if irf_decay is None:
            laser_pulse_time = self.fluor[np.argmax(self.fluor[:, 2]), 0]
            irf = self.gauss_irf(self.fluor[:, 0], laser_pulse_time)
            self.irf = np.hstack((self.fluor[:, 0:2], np.array(irf, ndmin=2).T))
            self.irf_type = 'Gaussian'
        else:
            self.irf = np.hstack((np.array(irf_decay[:, 0] * fluor_ns_per_chan, ndmin=2).T, irf_decay))
            self.irf_type = 'experimental'
        self.fit_param = None
        self.fit_param_std = None
        self.fit_y = None

    @staticmethod
    def _get_time(channel, fluor_ns_per_chan):
        """
        Convert channel to time bins

        Parameters
        ----------
        channel : array_like
        fluor_ns_per_chan = float

        Returns
        -------
        ndarray
            time
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
        bg : int

        Returns
        -------
        weights: ndarray
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
        mu : float
        sigma : float
        A : float

        Returns
        -------
        irf : ndarray
        """
        irf = A * np.exp(-(time - mu)**2 / (2 * sigma**2)).T
        return irf

    @staticmethod
    def convolution(irf, exp_fn):
        """
        Compute convolution of irf with exponential decay

        Parameters
        ----------
        irf : array_like
        exp_fn : array_like

        Returns
        -------
        convolved : ndarray
        """
        exp_fft = np.fft.fft(exp_fn)
        irf_fft = np.fft.fft(irf)
        convolved = np.real(np.fft.ifft(exp_fft * irf_fft))
        return convolved

    @staticmethod
    def exp_decay(time, tau):
        """
        Single exponential decay function

        Parameters
        ----------
        time : array_like 
        tau : float

        Returns
        -------
        decay : array_like
        """
        decay = np.exp(-time / tau)
        return decay

    def nnls_convol_irfexp(self, x_data, p0):
        """
        Solve non-negative least squares for series of IRF-convolved single-exponential decays. 
        First, the IRF is shifted, then convolved with each exponential decay individually (decays 1,...,n), 
        merged into an m x n array (=A) and finally plugged into scipy.optimize.nnls(A, experimental y-data) to which 
        compute `argmin_x || Ax - y ||_2`.

        Parameters
        ----------
        x_data : array_like
        p0 : array_like

        Returns
        -------
        A : ndarray
            matrix containing irf-convoluted single-exponential decays in the first n columns 
            and ones in the last column (background counts)
        x : vector that minimizes `|| Ax - y ||_2`
        y : fit vector computed as `y = Ax`
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
        Parameters
        ----------
        Wrapper function for tcspc.nnls_irfshift_convol

        See also
        --------
        nnls_convol_irfexp : Calculate non-linear least squares of IRF-convolved single-exponential decays

        Returns
        -------
        y : fit vector computed as `y = Ax`
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
        irf : array_like
        irf_shift : int

        Returns
        -------
        irf_shifted : array_like

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
        a : float
        tau_val : float
        taus_std : float

        Returns
        -------
        av_lt = tuple
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
        via interative reconvolution with the instrument reponse function (IRF)

        Parameters
        ----------
        tau0 : int or array_like
        tau_bounds : 2-tuple of float or array_like
        irf_shift : int
        sigma : array_like
        verbose : bool
        """
        if type(tau0) is int or type(tau0) is float:
            tau0 = [tau0]
        n_param = len(tau0)
        p0 = [irf_shift, *tau0]
        shift_bounds = [-np.inf, np.inf]
        bounds = []
        for i in range(2):
            if type(tau_bounds[i]) is int or type(tau_bounds[i]) is float:
                bounds.append([shift_bounds[i], *[tau_bounds[i]] * n_param])  # if bounds are specified as int/floats, i.e the same for all taus
            else:
                bounds.append([shift_bounds[i], *tau_bounds[i]])  # if bounds are specified as arrays, i.e individual for each tau

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

        av_lt, av_lt_std = self.average_lifetime(ampl, tau, tau_std)

        if verbose:
            print('=======================================')
            print('Reconvolution fit with {} IRF'.format(self.irf_type))
            for i, (t, t_std, a) in enumerate(zip(tau, tau_std, ampl)):
                print('tau{:d}: {:0.2f} ± {:0.2f} ns ({:0.0f}%)'.format(i, t, t_std, a * 100))
            print('mean tau: {:0.2f} ± {:0.2f} ns'.format(av_lt, av_lt_std))
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
        VV : array_like
        VH : array_like
        G : float
            G-factor

        Returns
        -------
            ndarray
                Anisotropy decay

        Notes
        -----
        The anisotropy decay is calculated from the parallel and perpendicular lifetime decays as follows:
        
        .. math::
            r(t) = \\frac{I_\\text{VV} - GI_\\text{VH}}{I_\\text{VV} + 2GI_\\text{VH}}
        """
        return np.array([(vv - G * vh) / (vv + 2 * G * vh) if (vv != 0) & (vh != 0) else np.nan for vv, vh in zip(VV, VH)])

    @staticmethod
    def G_factor(HV, HH):
        """
        Compute G-factor to correct for differences in transmittion effiency of the horizontal and vertical polarized light

        Parameters
        ----------
        VV : array_like
                asdf
        VH : array_like

        Returns
        -------
        float
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
    def one_rotation(x, r0, tau):
        """
        Single rotator model

        Parameters
        ----------
        x : array_like
        r0 : float
             fundamental anisotropy
        tau : float

        Returns
        -------
        ndarray
            two-rotation anisotropy decay
        """
        return r0 * np.exp(-x / tau)

    @staticmethod
    def two_rotations(x, r0, b, tau, tau2):
        """
        Two-rotator model

        Parameters
        ----------
        x : array_like
        r0 : float
             fundamental anisotropy
        b : float
        tau : float
        tau2 : float

        Returns
        -------
        ndarray
            two-rotation anisotropy decay
        """
        return r0 * np.exp(-x / tau) + (r0 - b) * np.exp(-x / tau2)

    @staticmethod
    def hindered_rotation(x, r0, tau, rinf):
        """
        Hindered rotation in-a-cone model

        Parameters
        ----------
        x : array_like
        r0 : float
             fundamental anisotropy
        tau : float
        rinf : float

        Returns
        -------
        ndarray
            hindered rotation anisotropy decay
        """
        return (r0 - rinf) * np.exp(-x / tau) + rinf

    def _aniso_fitinterval(self, r, ns_before_VVmax=1, signal_percentage=0.01):
        """
        Determine interval for tail-fit of ansiotropy decay. Outside of the fit interval the data is uncorrelated.

        Parameters
        ----------
        r : array_like
        ns_before_VVmax : float, optional
                          how many nanoseconds before the maximum of the VV decay should the search for r0 start
        signal_percentage : float, optional
                            percentage of the VV decay serving as a treshold to define the end of the anisotropy fit interval 
        Returns
        -------
        channel_start : int
        channel_stop : int
        """
        channel_VVmax = np.argmax(self.VV.fluor[:, 2])
        channel_start = channel_VVmax - int(ns_before_VVmax / self.VV.ns_per_chan)
        channel_stop = channel_VVmax + np.argmax(self.VV.fluor[channel_VVmax:, 2] < signal_percentage * max(self.VV.fluor[:, 2]))
        channel_start = channel_start + np.argmax(r[channel_start:channel_stop])
        return channel_start, channel_stop

    def rotation_fit(self, p0=[0.4, 1], model='one_rotation', manual_interval=None, bounds=(0, np.inf), verbose=True):
        """
        Fit rotation model to anisotropy decay.

        Parameters
        ----------
        p0 : array_like
        model : str
                one of the following anisotropy models: {'one_rotation', 'two_rotations', 'hindered_rotation'}
        manual_interval : 2-tuple of float, optional
                          start and stop time (in ns) for anisotropy fit
        bounds : 2-tuple of float or array_like
        verbose : bool

        Examples
        --------

        >>> 

        """
        if model == 'two_rotations':
            aniso_fn = self.two_rotations
        elif model == 'hindered_rotation':
            aniso_fn = self.hindered_rotation
        else:
            aniso_fn = self.one_rotation
        self.model = model

        param_names = {'one_rotation': ['r0', 'tau'],
                       'two_rotations': ['r0', 'b', 'tau', 'tau2'],
                       'hindered_rotation': ['r0', 'tau', 'rinf']}
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
                start, stop = self._aniso_fitinterval(_fit_raw_r)
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
                    print('{}: {:0.2f} ± {:0.2f} ns'.format(p, self.fit_param[i], self.fit_param_std[i]))
                print('====================')


if __name__ == "__main__":
    fluor_file, irf_file, use_gauss = parseCmd()
    fluor_data, fluor_nsperchan = read_decay(fluor_file)
    if irf_file is not None:
        irf_data = read_decay(irf_file)
    decay = Lifetime(fluor_data, fluor_ns_per_chan, irf_data)
    decay.reconvolution_fit([1, 10])
