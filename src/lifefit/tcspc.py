#!/usr/bin/env python3

"""
Python module to fit time-correlated single photon counting (TCSPC) data
"""

import numpy as np
import scipy.optimize
from pathlib import Path
import argparse
import re
from uncertainties import ufloat
import json
from typing import Union, Callable, Optional
import io

from lifefit import VERSION


def parseCmd() -> tuple[str, str]:
    """Parse the command line to get the experimental decay and instrument reponse function (IRF) file.

    Returns:
        fluor_file : filename of the fluorescence decay
        irf_file : filename of the IRF (if None then the IRF is approximated by a Gaussian)
    """
    parser = argparse.ArgumentParser(description="Fit a series of exponential decays to an ")
    parser.add_argument("--version", action="version", version="%(prog)s " + str(VERSION))
    parser.add_argument("-f", "--fluor", help="fluorescence decay file (.txt / .dat)", required=True)
    parser.add_argument(
        "-i", "--irf", help="Instrument response function (IRF) file (.txt / .dat)", required=False, default=None
    )
    parser.add_argument("-g", "--gauss", help="Use Gaussian IRF", required=False, default=True)
    args = parser.parse_args()
    fluor_file = args.fluor
    irf_file = args.irf
    return fluor_file, irf_file


def read_decay(filepath_or_buffer: Union[str, io.StringIO], fileformat: str = "Horiba") -> tuple[np.ndarray, float]:
    """Read TCSPC decay file from HORIBA or another data format. The 'time_intensity' format assumes a two-column file
    with time and intensity counts.

    Args:
        filepath_or_buffer : filename of the decay or StringIO object
        fileformat : currently implemented formats: {'horiba', 'time_intensity'}

    Returns:
        decay_data : n x 2 decay containing numbered channels and intensity counts for instrument response function (IRF)
        ns_per_chan : timsteps in nanoseconds between channels
    """
    if isinstance(filepath_or_buffer, (str, Path)):
        with open(filepath_or_buffer, "r") as decay_file:
            decay_data, ns_per_chan = _parse_file(decay_file, fileformat)
    else:
        decay_data, ns_per_chan = _parse_file(filepath_or_buffer, fileformat)
    return decay_data, ns_per_chan


def _parse_file(decay_file: io.StringIO, fileformat: str = "Horiba") -> tuple[np.ndarray, float]:
    """Parse the decay file. The 'time_intensity' format assumes a two-column file with time and intensity counts.

    Args:
        decay_file : StringIO of the decay
        fileformat : currently implemented formats: {'horiba', 'time_intensity'}

    Returns:
        decay_data : n x 2 decay containing numbered channels and intensity counts for instrument reponse function (IRF)
        ns_per_chan : timsteps in nanoseconds between channels
    """
    if fileformat.lower() == "horiba":
        ns_per_chan = None
        for i, line in enumerate(decay_file):
            if "Time" in line:
                ns_per_chan = float(re.search("\\d+\\.?\\d*E?-?\\d*", line).group())
            if "Chan" in line:
                break
        else:
            raise ValueError("Beginning of data section not defined. Please add headers 'Chan' and 'Data'")
        if ns_per_chan is None:
            raise ValueError("Timestep not defined")
        decay_data = np.loadtxt(decay_file, skiprows=0, dtype="int")
    elif fileformat == "time_intensity":
        decay_data = np.loadtxt(decay_file, skiprows=1)
        ns_per_chan = decay_data[1, 0] - decay_data[0, 0]
        decay_data[:, 0] /= ns_per_chan
    elif fileformat == "customName":
        # implement custom file reader here

        # make sure to define the following variables:
        # ns_per_chan = ...
        # headerlines = ...
        # decay_data = ...
        pass
    else:
        raise ValueError(
            "The specified format is not available. You may define your own format in the `read_decay` function"
        )
    return decay_data, ns_per_chan


def fit(
    fun: Callable,
    x_data: np.ndarray,
    y_data: np.ndarray,
    p0: np.ndarray,
    bounds: tuple[list[float]] = ([0, 0, 0], [np.inf, np.inf, np.inf]),
    sigma: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrapper for the curve_fit function of the scipy.optimize module
    The curve_fit optimizes the decay parameters (tau1, tau2, etc.)
    while the nnls weights the exponential decays.

    Args:
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

    Returns:
        p : optimized fit parameters
        p_std : standard deviation of optimized fit parameters
    """
    p, cov = scipy.optimize.curve_fit(fun, x_data, y_data, p0, bounds=bounds, sigma=sigma)
    p_std = np.sqrt(np.diag(cov))
    return p, p_std


class Lifetime:
    """Create lifetime class

    Args:
        fluor_decay : n x 2 array containing numbered channels and intensity counts of the fluorescence decay
        fluor_ns_per_chan : nanoseconds per channel
        irf_decay : n x 2 array containing numbered channels and intensity counts for instrument reponse function (IRF).
            If `None`, then IRF is approximated by a Gaussian
        gauss_sigma : standard deviation of the IRF Gaussian
        gauss_amp : amplitude of the IRF Gaussian
        shift_time : whether to shift time point 0 to the maximum of the decay

    Attributes:
        ns_per_chan : nanoseconds per channel
        fluor : n x 4 array containing time, channel number, intensity counts and associated Poissonian weights
            of the fluorescence decay
        irf : n x 3 array containing time, channel number and intensity counts of the IRF
        irf_type : type of IRF: {'Gaussian', 'experimental'}
        fit_param : ndarray of fit parameters
        fit_param_std : ndarray of standard deviations of fit parameters

    Example:

        >>> fluor, fluor_nsperchan = lf.tcspc.read_decay(pathToFluorDecay)
        >>> irf, irf_nsperchan = lf.tcspc.read_decay(pathToIRF)
        >>> lf.tcspc.Lifetime(fluor, fluor_nsperchan, irf)
    """

    def __init__(
        self,
        fluor_decay: np.ndarray,
        fluor_ns_per_chan: float,
        irf_decay: Optional[np.ndarray] = None,
        gauss_sigma=None,
        gauss_amp=None,
        shift_time=False,
    ):
        # compute Poisson weights
        self.ns_per_chan = fluor_ns_per_chan
        time = self._get_time(fluor_decay[:, 0], fluor_ns_per_chan)
        if shift_time:
            time = self._shift_time(time, fluor_decay[:, 1])
        weights = self._get_weights(fluor_decay[:, 1])
        self.fluor = np.hstack((time, fluor_decay, weights))
        self.is_counts = all([f.is_integer() for f in self.fluor[:, 2]])
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
            self.irf_type = "Gaussian"
        else:
            time = self._get_time(irf_decay[:, 0], fluor_ns_per_chan)
            if shift_time:
                time = self._shift_time(time, fluor_decay[:, 1])
            self.irf = np.hstack((time, irf_decay))
            self.irf_type = "experimental"
        self.fit_param = None
        self.fit_param_std = None
        self.fit_y = None

    @classmethod
    def from_filenames(
        cls,
        fluor_file: str,
        irf_file: str = None,
        fileformat: str = "HORIBA",
        gauss_sigma: Optional[float] = None,
        gauss_amp: Optional[float] = None,
        shift_time: bool = False,
    ):
        """Alternative constructor for the Lifetime class by reading in filename for the fluorophore and IRF decay

        Args:
            fluor_file : filename of the fluorophore decay
            irf_file : filename of the IRF decay
            fileformat : currently implemented formats: {'HORIBA'}
            gauss_sigma : standard deviation of the IRF Gaussian
            gauss_amp : amplitude of the IRF Gaussian
            shift_time : whether to shift time point 0 to the maximum of the decay

        Example:

            >>> lf.tcspc.Lifetime.from_filenames(pathToFluorDecay, pathToIRFDecay)
        """
        fluor_decay, ns_per_chan = read_decay(fluor_file, fileformat="HORIBA")
        if irf_file:
            irf_decay, _ = read_decay(irf_file, fileformat="HORIBA")
        else:
            irf_decay = None
        return cls(fluor_decay, ns_per_chan, irf_decay, gauss_sigma, gauss_amp, shift_time)

    @staticmethod
    def _get_time(channel: np.ndarray, fluor_ns_per_chan: float) -> np.ndarray:
        """Convert channel to time bins

        Args:
            channel : array of channel bins
            fluor_ns_per_chan = nanoseconds per channel bin

        Returns:
            array of time bins
        """
        time = np.array(channel * fluor_ns_per_chan, ndmin=2).T.round(3)
        return time

    @staticmethod
    def _shift_time(time: np.ndarray, decay: np.ndarray):
        """Shift time point 0 to the maximum of the decay

        Args:
            time : array of time bins
            decay : array of intensity values (counts) of the decay

        Returns:
            shifted array of time bins
        """
        return time - time[np.argmax(decay)]

    @staticmethod
    def _get_weights(decay: np.ndarray, bg: int = 1) -> np.ndarray:
        """Compute Poissonian weights

        Args:
            decay : array of intensity values (counts) of the decay
            bg : background count

        Returns:
            array of Poissonian weights of the decay points
        """
        weights = np.array(1 / np.sqrt(decay + bg), ndmin=2).T.round(3)
        return weights

    @staticmethod
    def gauss_irf(time: np.ndarray, mu: float, sigma: float = 0.01, A: float = 10000) -> np.ndarray:
        """Calculate a Gaussian-shaped instrument response function (IRF)

        Args:
            time : time bins
            mu : mean of the Gaussian distribution
            sigma : standard deviation of the Gaussian distribution
            A : amplitude of the Gaussian distribution

        Returns:
            A Gaussian-shaped instrument response function
        """
        irf = A * np.exp(-((time - mu) ** 2) / (2 * sigma**2)).T
        return irf

    @staticmethod
    def convolution(irf: np.ndarray, sgl_exp: np.ndarray) -> np.ndarray:
        """Compute convolution of irf with a single exponential decay

        Args:
            irf : intensity counts of the instrument reponse function (experimental of Gaussian shaped)
            sgl_exp : single-exponential decay

        Returns:
            convoluted signal of IRF and exponential decay
        """
        exp_fft = np.fft.fft(sgl_exp)
        irf_fft = np.fft.fft(irf)
        convolved = np.real(np.fft.ifft(exp_fft * irf_fft))
        return convolved

    @staticmethod
    def exp_decay(time: np.ndarray, tau: float) -> np.ndarray:
        """Create a single-exponential decay

        Args:
            time : time bins
            tau : fluorescence lifetime

        Returns:
            single-exponential decay

        Note:
            Single-exponential decay
            $$
            f(t) = \\exp(-t / \\tau)
            $$
        """
        sgl_exp = np.exp(-time / tau)
        return sgl_exp

    def nnls_convol_irfexp(self, x_data: np.ndarray, p0: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve non-negative least squares for series of IRF-convolved single-exponential decays.
        First, the IRF is shifted, then convolved with each exponential decay individually (decays 1,...,n),
        merged into an m x n array (=A) and finally plugged into `scipy.optimize.nnls(A, experimental y-data)` to
        compute `argmin_x || Ax - y ||_2`. This optimizes the relative weight of the exponential decays
        whereas the curve_fit function optimizes the decay parameters (tau1, taus2, etc.).
        See also [scipy.optimize.nnls](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html).


        Args:
            x_data : array of the independent variable
            p0 : start values for the fit model

        Returns:
            A = matrix containing irf-convoluted single-exponential decays in the first n columns
                and ones in the last column (background counts)
            x = vector that minimizes the 2-norm `|| Ax - y ||_2`
            y = fit vector computed as `y = Ax`
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

    def _model_func(self, x_data: np.ndarray, *p0: list[float]) -> np.ndarray:
        """Wrapper function for tcspc.nnls_irfshift_convol

        Args:
            x_data : array of the independent variable
            p0 : start values for the fit model

        See also:
            nnls_convol_irfexp : Calculate non-linear least squares of IRF-convolved single-exponential decays

        Returns:
            fit vector computed as `y = Ax`
        """
        A, x, y = self.nnls_convol_irfexp(x_data, p0)
        return y

    @staticmethod
    def _irf_scaleshift(channel: np.ndarray, irf: np.ndarray, irf_shift: int) -> np.ndarray:
        """Shift IRF by n-channels (n = irf_shift)

        Args:
            channel : array of channel bins
            irf : intensity counts of the instrument reponse function (experimental of Gaussian shaped)
            irf_shift : whether to shift of the IRF on the time axis (in channel units)

        Returns:
            time-shifted IRF

        References:
            [2] J. Enderlein, *Optics Communications* **1997**
        """
        n = len(irf)
        # adapted from tcspcfit (J. Enderlein)
        irf_shifted = (1 - irf_shift + np.floor(irf_shift)) * irf[
            np.fmod(np.fmod(channel - np.floor(irf_shift) - 1, n) + n, n).astype(int)
        ] + (irf_shift - np.floor(irf_shift)) * irf[
            np.fmod(np.fmod(channel - np.ceil(irf_shift) - 1, n) + n, n).astype(int)
        ]
        return irf_shifted

    @staticmethod
    def average_lifetime(a: np.ndarray, tau_val: np.ndarray, tau_std: np.ndarray) -> tuple[float, float]:
        """Calculate average lifetime according to Lakowicz (2010) [1]

        Args:
            a : weighting factors of tau
            tau_val : fluorescence lifetimes
            tau_std : standard deviation of the fluorescence lifetimes

        Returns:
            average lifetime and associated standard deviation

        References:
            [1] J. Lakowicz, *Principles of Fluorescence*, 3rd ed., Springer, **2010**.
        """
        tau = np.array([ufloat(val, std) for val, std in zip(tau_val, tau_std)])
        av_lt = sum(a * tau**2) / sum(a * tau)
        return av_lt.n, av_lt.s

    def reconvolution_fit(
        self,
        tau0: list[int] = [1],
        tau_bounds: tuple[float] = (0, np.inf),
        irf_shift: int = 0,
        sigma: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """Fit the experimental lifetime decay to a series of exponentials
        via interative reconvolution with the instrument reponse function (IRF).

        Args:
            tau0 : start value(s) of the fluorescence lifetime(s)
            tau_bounds : lower and upper bounds for each parameter in tau0. Can be either a tuple of two scalars
                (same bound for all parameters) or a tuple of array_like with the same length as tau0.
                To deactivate parameter bounds set: `bounds=(-np.inf, np.inf)`
            irf_shift : shift of the IRF on the time axis (in channel units)
            sigma : uncertainty of the decay (same length as y_data)
            verbose : whether to print lifetime fit result

        Example:

            >>> obj.reconvolution_fit([1,5])
        """
        if type(tau0) is int or type(tau0) is float:
            tau0 = [tau0]
        n_param = len(tau0)
        p0 = [irf_shift, *[t0 / self.ns_per_chan for t0 in tau0]]
        shift_bounds = [-np.inf, np.inf]
        bounds = []
        for i in range(2):
            if type(tau_bounds[i]) is int or type(tau_bounds[i]) is float:
                bounds.append(
                    [shift_bounds[i], *[tau_bounds[i] / self.ns_per_chan] * n_param]
                )  # if bounds are specified as int/floats, i.e the same for all taus
            else:
                bounds.append(
                    [shift_bounds[i], *[tb / self.ns_per_chan for tb in tau_bounds[i]]]
                )  # if bounds are specified as arrays, i.e individual for each tau
        p, p_std = fit(self._model_func, self.fluor[:, 1], self.fluor[:, 2], p0, bounds=bounds, sigma=sigma)
        A, x, self.fit_y = self.nnls_convol_irfexp(self.fluor[:, 1], p)
        if self.is_counts:
            self.fit_y = self.fit_y.round(0).astype(int)
        ampl = x[:-1] / sum(x[:-1])
        offset = x[-1]
        irf_shift = p[0] * self.ns_per_chan
        tau = p[1:] * self.ns_per_chan
        tau_std = p_std[1:] * self.ns_per_chan
        irf_shift_std = p_std[0] * self.ns_per_chan
        self.fit_param = {"ampl": ampl, "offset": offset, "irf_shift": irf_shift, "tau": tau}
        self.fit_param_std = {"tau": tau_std, "irf_shift": irf_shift_std}
        self.av_lifetime, self.av_lifetime_std = self.average_lifetime(ampl, tau, tau_std)

        if verbose:
            print("=======================================")
            print("Reconvolution fit with {} IRF".format(self.irf_type))
            for i, (t, t_std, a) in enumerate(zip(tau, tau_std, ampl)):
                print("tau{:d}: {:0.2f} ± {:0.2f} ns ({:0.0f}%)".format(i, t, t_std, a * 100))
            print("mean tau: {:0.2f} ± {:0.2f} ns".format(self.av_lifetime, self.av_lifetime_std))
            print("")
            print("irf shift: {:0.2f} ns".format(irf_shift))
            print("offset: {:0.0f} counts".format(offset))
            print("=======================================")

    def export(self, filename):
        data, parameters = self._serialize()
        with open("{}_{}.json".format(filename.split(".", 1)[0], "data"), "w") as f:
            json.dump(data, f, indent=2)

        with open("{}_{}.json".format(filename.split(".", 1)[0], "parameters"), "w") as f:
            json.dump(parameters, f, indent=2)

    def _serialize(self):
        data = {}
        dtype = int if self.is_counts else float
        if self.fit_param is not None:
            data["time"] = self.fluor[:, 0].tolist()
            data["irf_counts"] = self.irf[:, 2].astype(dtype).tolist()
            data["fluor_counts"] = self.fluor[:, 2].astype(dtype).tolist()
            data["fit_counts"] = self.fit_y.astype(dtype).tolist()
            data["residuals"] = (self.fluor[:, 2] - self.fit_y).astype(dtype).tolist()
            parameters = {}
            parameters["irf_type"] = self.irf_type
            for i, (t, t_std, a) in enumerate(
                zip(self.fit_param["tau"], self.fit_param_std["tau"], self.fit_param["ampl"])
            ):
                parameters["tau{:d}".format(i)] = {
                    "units": "ns",
                    "value": round(t, 2),
                    "error": round(t_std, 2),
                    "fraction": round(a, 2),
                }
                parameters["mean_tau"] = {
                    "units": "ns",
                    "value": round(self.av_lifetime, 2),
                    "error": round(self.av_lifetime_std, 2),
                }
            parameters["irf_shift"] = {"units": "ns", "value": round(self.fit_param["irf_shift"], 2)}
            parameters["offset"] = {"units": "counts", "value": round(self.fit_param["offset"], 0)}
            return data, parameters
        else:
            raise ValueError("Fit parameters are not available. Please run a reconvolution_fit before exporting")


class Anisotropy:
    """Create an Anisotropy object with four polarization resolved lifetime decays

    Args:
        VV : vertical excitation - vertical emission
        VH : vertical excitation - horizontal emission
        HV : horizontal excitation - vertical emission
        HH : horizontal excitation - horizontal emission

    Example:

        >>> lf.tcspc.Anisotropy(decay['VV'], decay['VH'], decay['HV'],decay['HH'])
    """

    def __init__(self, VV: np.ndarray, VH: np.ndarray, HV: np.ndarray, HH: np.ndarray):
        self.VV = VV
        self.VH = VH
        self.HV = HV
        self.HH = HH
        self.G = self.G_factor(self.HV.fluor[:, 2], self.HH.fluor[:, 2])
        self.raw_r = self.aniso_decay(self.VV.fluor[:, 2], self.VH.fluor[:, 2], self.G)
        self.raw_time = self.VV.fluor[:, 0]

    @staticmethod
    def aniso_decay(VV: np.ndarray, VH: np.ndarray, G: float) -> np.ndarray:
        """Compute the anisotropy decay

        Args:
            VV : vertical excitation - vertical emission
            VH : vertical excitation - horizontal emission
            G : G-factor

        Returns:
           anisotropy decay

        Note:
            The anisotropy decay is calculated from the parallel and perpendicular lifetime decays as follows
            $$
            r(t) = \\frac{I_\\text{VV} - GI_\\text{VH}}{I_\\text{VV} + 2GI_\\text{VH}}
            $$
        """
        r = np.array(
            [(vv - G * vh) / (vv + 2 * G * vh) if (vv != 0) & (vh != 0) else np.nan for vv, vh in zip(VV, VH)]
        ).round(3)
        return r

    @staticmethod
    def G_factor(HV: np.ndarray, HH: np.ndarray) -> float:
        """Compute G-factor to correct for differences in transmittion effiency of the horizontal and vertical polarized light

        Args:
            HV : horizontal excitation - vertical emission
            HH : horizontal excitation - horizontal emission

        Returns:
            G-factor

        Note:
            The G-factor is defined as follows
            $$
            G = \\frac{\\int HV}{\\int HH}
            $$
        """
        G = sum(HV) / sum(HH)
        return G

    @staticmethod
    def one_rotation(time: np.ndarray, r0: float, tau_r: float) -> np.ndarray:
        """Single rotator model

        Args:
            time : time bins
            r0 : fundamental anisotropy
            tau_r : rotational correlation time

        Returns:
            two-rotation anisotropy decay

        Note: One-rotation model
            $$
            f(t) = r_0 \\exp(-t / \\tau_r)
            $$
        """
        return r0 * np.exp(-time / tau_r)

    @staticmethod
    def two_rotations(time: np.ndarray, r0: float, b: float, tau_r1: float, tau_r2: float) -> np.ndarray:
        """Two-rotator model

        Args:
            time : time bins
            r0 : fundamental anisotropy
            b : amplitude of second decay
            tau_r1 : first rotational correlation time
            tau_r2 : second rotational correlation time

        Returns:
            two-rotation anisotropy decay

        Note: Two-rotation model
            $$
            f(t) = r_0 \\exp(-t / \\tau_{r1}) + (r_0 - b) \\exp(-t / \\tau_{r2})
            $$
        """
        return r0 * np.exp(-time / tau_r1) + (r0 - b) * np.exp(-time / tau_r2)

    @staticmethod
    def hindered_rotation(time: np.ndarray, r0: float, tau_r: float, r_inf: float) -> np.ndarray:
        """Hindered rotation in-a-cone model

        Args:
            time : time bins
            r0 : fundamental anisotropy
            tau_r : rotational correlation time
            r_inf : residual anisotropy at time->inf

        Returns:
            hindered rotation anisotropy decay

        Note: Hindered-rotation model
            $$
            f(t) = (r_0 - r_\\infty) \\exp(-t / \\tau_r) + r_\\infty
            $$
        """
        return (r0 - r_inf) * np.exp(-time / tau_r) + r_inf

    @staticmethod
    def local_global_rotation(
        time: np.ndarray, r0: float, tau_rloc: float, r_inf: float, tau_rglob: float
    ) -> np.ndarray:
        """Local-global rotation in-a-cone model

        Args:
            time : time bins
            r0 : fundamental anisotropy
            tau_rloc : local rotational correlation time
            r_inf : residual anisotropy at time->inf
            tau_rglob : global rotational correlation time

        Returns:
            local-global rotation anisotropy decay

        Note:
            Local-global rotation in-a-cone model
            $$
            f(t) = ((r_0 - r_\\infty) \\exp(-t / \\tau_{r,\\text{loc}}) + r_\\infty) \\exp(-t / \\tau_{r,\\text{glob}})
            $$
        """
        return ((r0 - r_inf) * np.exp(-time / tau_rloc) + r_inf) * np.exp(-time / tau_rglob)

    def _aniso_fitinterval(self, r: np.ndarray, ns_before_VVmax: float, signal_percentage: float) -> tuple[int, int]:
        """Determine interval for tail-fit of anisotropy decay. Outside of the fit interval the data is uncorrelated.

        Args:
            r : anisotropy decay
            ns_before_VVmax : how many nanoseconds before the maximum of the VV decay should the search for r0 start
            signal_percentage : percentage of the VV decay serving as a treshold to define the end of the anisotropy fit interval

        Returns:
            Tuple indicating the channel from where to start the tail fit and the channel where the fit should end
        """
        channel_VVmax = np.argmax(self.VV.fluor[:, 2])
        channel_start = channel_VVmax - int(ns_before_VVmax / self.VV.ns_per_chan)
        channel_stop = channel_VVmax + np.argmax(
            self.VV.fluor[channel_VVmax:, 2] < signal_percentage * max(self.VV.fluor[:, 2])
        )
        channel_start = channel_start + np.argmax(r[channel_start:channel_stop])
        return channel_start, channel_stop

    def rotation_fit(
        self,
        p0: list[float] = [0.4, 1],
        model: str = "one_rotation",
        manual_interval: Optional[tuple[float]] = None,
        bounds: tuple[float] = (0, np.inf),
        verbose: bool = True,
        ns_before_VVmax: float = 1,
        signal_percentage: float = 0.01,
    ):
        """Fit rotation model to anisotropy decay.

        Args:
            p0 : start values of the chosen anisotropy fit model
            model : one of the following anisotropy models: {'one_rotation', 'two_rotations', 'hindered_rotation', 'local_global_rotation'}
            manual_interval : Time interval in which to fit the anisotropy decay
            bounds : lower and upper bounds for each parameter in p0. Can be either a tuple of two scalars
                (same bound for all parameters) or a tuple of array_like with the same length as p0.
                To deactivate parameter bounds set: `bounds=(-np.inf, np.inf)`
            verbose : whether to print anisotropy fit result
            ns_before_VVmax : how many nanoseconds before the maximum of the VV decay should the search for r0 start
            signal_percentage : percentage of the VV decay serving as a treshold to define the end of the anisotropy fit interval

        Example:

            >>> obj.rotation_fit(p0=[0.4, 1, 10, 1], model='two_rotations')
        """
        if model == "two_rotations":
            aniso_fn = self.two_rotations
        elif model == "hindered_rotation":
            aniso_fn = self.hindered_rotation
        elif model == "local_global_rotation":
            aniso_fn = self.local_global_rotation
        else:
            aniso_fn = self.one_rotation
        self.model = model

        self.param_names = {
            "one_rotation": ["r0", "tau_r"],
            "two_rotations": ["r0", "b", "tau_r1", "tau_r2"],
            "hindered_rotation": ["r0", "tau_r", "rinf"],
            "local_global_rotation": ["r0", "tau_rloc", "rinf", "tau_rglob"],
        }
        try:
            if not len(p0) == len(self.param_names[model]):
                raise ValueError
        except ValueError:
            print('Number of start parameters p0 is not consistent with the model "{}"'.format(model))
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
                print("====================")
                print("Anisotropy fit")
                print("model: {}".format(self.model))
                for i, p in enumerate(self.param_names[model]):
                    if "tau" in p:
                        print("{}: {:0.2f} ± {:0.2f} ns".format(p, self.fit_param[i], self.fit_param_std[i]))
                    else:
                        print("{}: {:0.2f} ± {:0.2f}".format(p, self.fit_param[i], self.fit_param_std[i]))
                if self.model == "local_global_rotation" or self.model == "hindered_rotation":
                    self.aniso_fraction = self._fraction_freeStacked(self.fit_param[0], self.fit_param[2])
                    print(
                        "free: {:0.0f}%, stacked: {:0.0f}%".format(
                            self.aniso_fraction[0] * 100, self.aniso_fraction[1] * 100
                        )
                    )
                print("====================")

    @staticmethod
    def _fraction_freeStacked(r0: float, r_inf: float) -> tuple[float, float]:
        """Calculate the fraction of free and stacked dyes based on the residual anisotropy

        Args:
            r0 : fundamental anisotropy
            r_inf : residual anisotropy

        Returns:
            fraction of free and stacked dye components

        Note:
            Weights of free and stacked components
            $$
            \\omega_\\text{free} = (r_0 - r_\\infty) / r_0
            \\omega_\\text{stacked} = 1 - \\omega_\\text{free}
            $$
        """
        w_free = (r0 - r_inf) / r0
        w_stacked = 1 - w_free
        return (w_free, w_stacked)

    def export(self, filename):
        """Export the data and the fit parameters to a json file

        Args:
            filename : Name of the JSON file
        """
        data, parameters = self._serialize()
        with open("{}_{}.json".format(filename.split(".", 1)[0], "data"), "w") as f:
            json.dump(data, f, indent=2)

        with open("{}_{}.json".format(filename.split(".", 1)[0], "parameters"), "w") as f:
            json.dump(parameters, f, indent=2)

    def _serialize(self) -> tuple[dict, dict]:
        """Convert the numpy arrays to lists and package all data into a dictionary"""
        data = {}
        try:
            data["time"] = list(self.time)
            data["anisotropy"] = list(self.r)
            data["fit"] = list(self.fit_r)
            data["residuals"] = list(self.r - self.fit_r)
        except TypeError:
            print("Data is not complete. Please refit")
        else:
            parameters = {}
            parameters["model"] = self.model
            for i, p in enumerate(self.param_names[self.model]):
                if "tau" in p:
                    units = "ns"
                else:
                    units = None
                parameters[p] = {
                    "units": units,
                    "value": round(self.fit_param[i], 2),
                    "error": round(self.fit_param_std[i], 2),
                }

            if self.model == "local_global_rotation" or self.model == "hindered_rotation":
                parameters["free"] = {"units": units, "value": round(self.aniso_fraction[0], 2), "error": None}
                parameters["stacked"] = {"units": units, "value": round(self.aniso_fraction[1], 2), "error": None}
        return data, parameters


if __name__ == "__main__":
    fluor_file, irf_file = parseCmd()
    fluor_data, fluor_nsperchan = read_decay(fluor_file)
    if irf_file is not None:
        irf_data = read_decay(irf_file)
    decay = Lifetime(fluor_data, fluor_nsperchan, irf_data)
    decay.reconvolution_fit([1, 10])
