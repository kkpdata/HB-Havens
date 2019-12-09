import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

import logging
logger = logging.getLogger(__name__)

class Spectrum2D():
    """
    Class for 2D spectra. Array shape is temporal series for a list of points.
    Example:
    >> Sp2 = Spec2(f=np.linspace(0.03,.3,100),direction=np.arange(0,24)*15)
    generates 2D energy array with nans
    """

    def __init__(self, frequencies, directions, energy):
        """
        Constructor
        """
        # frequency, always implicit Hz
        self.f = frequencies
        self.direction = directions

        # Check bin width of directions
        absdiff = np.diff(directions) % 360
        if not np.isclose(absdiff[0], absdiff).all():
            raise NotImplementedError('Variable directional bin widths are found. Wave parameters can only be calculated for equal size bins.')

        self.dir_bin_width = abs(np.diff(self.direction[:2]))

        self.energy = energy

    def __repr__(self):

        if   len(self.energy.shape) == 2:
            txt = '<Spectrum2D ' + ' shape:[nt: 0' + \
            ',nx: 0 ' + ',nf:' + str(self.energy.shape[0]) + \
            ',nd:' + str(self.energy.shape[1]) + ']>'
        elif len(self.energy.shape) == 3:
            txt = '<Spectrum2D ' + ' shape:[nt: 0' + \
            ',nx:' + str(self.energy.shape[0]) + ',nf:' + str(self.energy.shape[1]) + \
            ',nd:' + str(self.energy.shape[2]) + ']>'
        elif len(self.energy.shape) == 4:
            txt = '<Spectrum2D ' + ' shape:[nt:' + str(self.energy.shape[0]) + \
            ',nx:' + str(self.energy.shape[1]) + ',nf:' + str(self.energy.shape[2]) + \
            ',nd:' + str(self.energy.shape[3]) + ']>'

        return txt

    
    def Hm0(self):
        """
        Integrate Hm0 (Hs) from wave spectra:
        >> S = ui.models.SwanSpectrum2D()
        >> S.Hm0()
        """
        return 4 * np.sqrt(self.m0())

    def m0(self):
        """Calculate zeroth order moment"""
        return np.trapz(self.energy.sum(axis=-1) * self.dir_bin_width, self.f, axis=-1)

    def m1(self):
        """Calculate first order moment"""
        return np.trapz(self.energy.sum(axis=-1) * self.dir_bin_width * self.f, self.f, axis=-1)

    def m2(self):
        """Calculate second order moment"""
        return np.trapz(self.energy.sum(axis=-1) * self.dir_bin_width * self.f**2, self.f, axis=-1)

    def mm1(self):
        """Calculate minus first order moment"""
        return np.trapz(self.energy.sum(axis=-1) * self.dir_bin_width / self.f, self.f, axis=-1)

    def Theta0(self):
        """
        theta0 from wave spectra:
        """

        sT = np.sin(np.deg2rad(self.direction))
        cT = np.cos(np.deg2rad(self.direction))

        m0Sin = np.sum(np.dot(self.energy, sT), axis=2)
        m0Cos = np.sum(np.dot(self.energy, cT), axis=2)

        theta0 = np.rad2deg(np.arctan2(m0Sin, m0Cos)) % 360
        return theta0

    def Tm01(self):
        """
        Integrate Tm01 (Tm) from wave spectra:
        S = ui.models.SwanSpectrum2D(), S.Tm01()
        """
        return self.m0() / self.m1()


    def Tmm10(self):
        """
        Integrate Tmm10 (Tm) from wave spectra:
        S = ui.models.SwanSpectrum2D(), S.Tmm10()
        """
        m0 = self.m0()
        nonzero = m0 != 0.0
        Tmm10 = np.zeros_like(m0)
        Tmm10[nonzero] = self.mm1()[nonzero] / m0[nonzero]
        return Tmm10

    def Tm02(self):
        """
        Integrate Hm0 (Hs) from wave spectra: S = ui.models.SwanSpectrum2D(), S.Tm02()
        """
        return np.sqrt(self.m0() / self.m2())

    def Tp(self):
        """
        Get peak period: S = swanui.models.SwanSpectrum2D(), S.Tp()
        """
        # 1/ the frequency where the energy per direction is largest.
        Tp = 1 / self.f[np.argmax(np.max(self.energy,axis=-1), axis=-1)]
        
        return Tp

    def Tp_smooth(self):
        """
        Calculate the peak periode by fitting a parabola
        """

        # Create arrays for energy and frequencies
        if np.ma.is_masked(self.energy):
            Ep = self.energy.data.copy()
        else:
            Ep = self.energy.copy()
        fp = np.zeros_like(Ep)
        fp[:, :, 0, :] = self.f[0]
        fp[:, :, -1, :] = self.f[-1]

        # For the non boundary frequencies, calculate the optimum
        Ep[:, :, 1:-1, :], fp[:, :, 1:-1, :] = calc_optimum(
            f=[
                self.f[np.newaxis, np.newaxis, :-2, np.newaxis],
                self.f[np.newaxis, np.newaxis, 1:-1, np.newaxis],
                self.f[np.newaxis, np.newaxis, 2:, np.newaxis]],
            E=[self.energy[:, :, :-2, :], self.energy[:, :, 1:-1, :], self.energy[:, :, 2:, :]],
            check_bounds=True
        )

        # Determine the maximum directions
        idx = (
            np.arange(Ep.shape[0])[:, None, None],
            np.arange(Ep.shape[1])[None, :, None],
            np.arange(Ep.shape[2])[None, None, :],
            np.argmax(Ep, axis=-1)
        )

        # Calculate the TP_smooth by finding the maximum energy per max direction
        Tp_smooth = 1./ fp[idx][
            np.arange(Ep.shape[0])[:, None],
            np.arange(Ep.shape[1])[None, :],
            np.argmax(Ep[idx], axis=-1)
        ]

        # Set all zero energy spectra to zero peak period
        Tp_smooth[self.m0() == 0.0] = 0.0


        return Tp_smooth

    def pdir(self):
        """
        Get peak period: S = ui.models.SwanSpectrum2D(), S.pdir()
        """

        pdir = self.direction[np.argmax(np.max(self.energy,axis=-2))]

        return pdir

    #@static
    def plot(self, fname=None, description='', figsize=(10,10), fontsize=7, dpi=400, it=0, ix=0):

        """
        plot 2D spectrum (to file) (assumes directions in degrees_north if empty).
        """

        ic = list(range(len(self.direction))) + [0] # circular indices

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(figsize=figsize, subplot_kw={'projection':'polar'})

        if self.direction_units == 'degrees_north':
            axs.pcolormesh(np.radians(self.direction)[ic], self.f, self.energy[it,ix,:,ic].T)
        else: # degrees_true or empty
            axs.pcolormesh(90 - self.direction[ic]         , self.f, self.energy[it,ix,:,ic].T)

        if description:
            axs.set_title(description)

        axs.set_aspect('equal')
        axs.set_theta_offset(np.radians(90))
        axs.set_theta_direction(-1)

        if not fname is None:
            fig.savefig(fname, fontsize=fontsize, dpi=dpi)
            plt.close()
        else:
            return axs

def calc_parabola_vertex(x, y):
    '''
    Adapted and modifed to get the unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    '''
    x1, x2, x3 = x
    y1, y2, y3 = y
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    A = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B = (x3**2 * (y1-y2) + x2**2 * (y3-y1) + x1**2 * (y2-y3)) / denom
    C = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom

    return A, B, C

def calc_optimum(f, E, check_bounds=False):
    """
    Create a parabolic fit trough the top 3 points
    It is possible to check whether the optimum is within bounds
    """
    A, B, C = calc_parabola_vertex(np.atleast_1d(f), np.atleast_1d(E))
    idx = A != 0.0
    
    f_max = np.zeros_like(A)
    E_max = np.zeros_like(A)
    
    f_max[idx] = -B[idx] / (2 * A[idx])
    E_max[idx] = A[idx] * f_max[idx] ** 2 + B[idx] * f_max[idx] + C[idx]
    
    if check_bounds:
        bounds_idx = (f_max > f[-1]) | (f_max < f[0])
        E_max[bounds_idx] = 0.0
        f_max[bounds_idx] = np.nan
        
    f_max[~idx] = np.nan
    E_max[~idx] = 0.0
    
    return E_max, f_max

class JONSWAPSpectrum:

    def __init__(self, f, theta):
        """
        Spectrum Class

        Parameters
        ----------
        f : numpy.ndarray
            Array of angular frequency bins [Hz]
        theta : numpy.ndarray [deg]
            Array of directionally bins
        """
        self.f = f
        # Radialen per seconde:
        self.w = self.f * 2 * np.pi
        self.theta = theta


    def jonswap(self, Hm0, Tp, gamma=3.3, sigma_low=.07, sigma_high=.09, g=9.81, method='goda', normalize=False):
        """
        Generate JONSWAP spectrum

        Parameters
        ----------
        Hm0 : float
            Required zeroth order moment wave height
        Tp : float
            Required peak wave period
        gamma : float
            JONSWAP peak-enhancement factor (default: 3.3)
        sigma_low : float
            Sigma value for frequencies <= ``1/Tp`` (default: 0.07)
        sigma_high : float
            Sigma value for frequencies > ``1/Tp`` (default: 0.09)
        g : float
            Gravitational constant (default: 9.81)
        method : str
            Method to compute alpha (default: yamaguchi)
        normalize : bool
            Normalize resulting spectrum to match ``Hm0``
    
        Returns
        -------
        E : numpy.ndarray
            Array of shape ``f`` with wave energy densities 
    
        """

        # Pierson-Moskowitz
        if method.lower() == 'yamaguchi':
            alpha = 1. / (.06533 * gamma ** .8015 + .13467) / 16.
        elif method.lower() == 'goda':
            alpha = 1. / (.23 + .03 * gamma - .185 / (1.9 + gamma)) / 16.
        else:
            raise ValueError('Unknown method: {}'.format(method))

        # TODO radialen per seconde
        # E_pm(f) = alpha_pm * g^2 (2pi)^-4 f^-5 exp(-(5/4) * (f/f_PM)^-4 )
        E_pm = alpha * Hm0 ** 2 * Tp ** -4 * self.f ** -5 * np.exp(-1.25 * (Tp * self.f) ** -4)
        
        # JONSWAP
        sigma = np.ones(self.f.shape) * sigma_low
        sigma[self.f > 1. / Tp] = sigma_high

        E_js = E_pm * gamma ** np.exp(-0.5 * (Tp * self.f - 1) ** 2. / sigma ** 2.)
    
        if normalize:
            E_js *= Hm0 ** 2. / (16. * np.trapz(E_js, self.f))
        
        return E_js

    def directional_spreading(self, theta_peak=0., s=20., units='deg', normalize=True):
        """
        Generate wave spreading
    
        Parameters
        ----------
        theta_peak : float
            Peak direction (default: 0)
        s : float
            Exponent in cosine law (default: 20)
        units : str
            Directional units (deg or rad, default: deg)
        normalize : bool
            Normalize resulting spectrum to unity
        Returns
        -------
        p_theta : numpy.ndarray
           Array of directional weights
        """
    
        from math import gamma
    
        theta = np.asarray(self.theta, dtype=np.float)

        # convert units radians
        if units.lower().startswith('deg'):
            theta = np.deg2rad(theta)
            theta_peak = np.deg2rad(theta_peak)
        elif units.lower().startswith('rad'):
            pass
        else:
            raise ValueError('Unknown units: %s')

        # compute directional spreading
        p_theta = np.maximum(0., np.cos(theta - theta_peak)) ** s

        # convert to original units
        if units.lower().startswith('deg'):
            theta = np.degrees(theta)
            theta_peak = np.degrees(theta_peak)
            p_theta = np.degrees(p_theta)
    
        # normalize directional spreading
        if normalize:
            p_theta /= np.trapz(p_theta, theta - theta_peak)

        return p_theta

    def jonswap_2D(self, hydraulic_loads, spread, gamma=3.3, sigma_low=.07, sigma_high=.09, g=9.81, method='goda', Smin=0.0, normalize=False):
        """
        Calculates a 2d JONSWAP spectral density

        Parameters
        ----------
        hydraulic_loads : pandas.DataFrame
            Dataframe with hydraulic loads
        spread : float
            Determines the spreading [-]
        gamma : float
            JONSWAP peak-enhancement factor (default: 3.3)
        sigma_low : float
            Sigma value for frequencies <= ``1/Tp`` (default: 0.07)
        sigma_high : float
            Sigma value for frequencies > ``1/Tp`` (default: 0.09)
        g : float
            Gravitational constant (default: 9.81)
        method : str
            Method to compute alpha (default: goda)
        normalize : bool
            Normalize resulting spectrum to match ``Hm0``

        Returns
        -------
        flags : numpy.array
            Contribution flags
        """

        nf = len(self.f)
        nd = len(self.theta)
        
        # add bin flags
        self.flags = np.zeros((nf, nd), dtype=bool)

        for idx, hydraulic_load in hydraulic_loads.iterrows():

            Hm0 = hydraulic_load['Hs']
            if 'Tp' in hydraulic_load.index:
                Tp = hydraulic_load['Tp']
            else:
                Tp = hydraulic_load['Tm-1,0'] * self.settings
            Theta = hydraulic_load['Wave direction']

            # print('Hs : {}, Tp: {}, Theta: {}'.format(Hm0, Tp, Theta))

            if Hm0 != 0 and Tp != 0:

                S = jonswap_2d(
                    f=self.f,
                    theta=self.theta,
                    S_1d=jonswap(self.f, Hm0, Tp),
                    Hm0=Hm0,
                    Tp=Tp,
                    Theta=Theta,
                    spread=spread
                )

                ind = (S > Smin)
                self.flags = self.flags | ind
        
        return self.flags

    def jonswap_2d(self, Hm0, Tp, Theta, spread, gamma=3.3, sigma_low=.07, sigma_high=.09, g=9.81, method='goda', normalize=False):
        """
        Calculates a 2d JONSWAP spectral density

        Parameters
        ----------
        Hm0 : float
            Required zeroth order moment wave height
        Tp : float
            Required peak wave period
        Theta : float
            mean wave direction [deg] w.r.t. North
        spread : float
            Determines the spreading [-]
        gamma : float
            JONSWAP peak-enhancement factor (default: 3.3)
        sigma_low : float
            Sigma value for frequencies <= ``1/Tp`` (default: 0.07)
        sigma_high : float
            Sigma value for frequencies > ``1/Tp`` (default: 0.09)
        g : float
            Gravitational constant (default: 9.81)
        method : str
            Method to compute alpha (default: goda)
        normalize : bool
            Normalize resulting spectrum to match ``Hm0``
        """

        # Calculate 1d jonswap
        S_1d = jonswap(self.f, Hm0, Tp, gamma, sigma_low, sigma_high, g, method, normalize)

        # Calculate 2d spreading
        S_2d = jonswap_2d(self.f, self.theta, S_1d, Hm0, Tp, Theta, spread, g)

        self.energy = S_2d

        return S_2d

    def spread_jonswap_2d(self, S_1d, Hm0, Tp, Theta, spread, g=9.81):
        """
        Calculates a 2d JONSWAP spectral density

        Parameters
        ----------
        S_1d : numpy.ndarray
            Array with 1d wave energy densities 
        Hm0 : float
            Required zeroth order moment wave height
        Tp : float
            Required peak wave period
        Theta : float
            mean wave direction [deg] w.r.t. North
        spread : float
            Determines the spreading [-]
        g : float
            Gravitational constant (default: 9.81)

        Returns
        -------
        s2d : numpy.ndarray
            the spectral density
        """

        # peak frequency
        fp = 1.0 / Tp
    
        # Determine the stepsize in the directions
        dtheta = np.deg2rad(np.diff(self.theta[:2]))
        bins = np.deg2rad(self.theta % 360)
        theta0 = np.deg2rad(Theta % 360)

        nf = len(self.f)
        nd = len(self.theta)

        # Calculate directional spreading
        # Allocate memory
        s = np.zeros_like(self.f)
        G0 = np.zeros_like(self.f)
        G = np.zeros((nf,nd))
        S_2d = np.zeros((nf,nd))


        # If the spread is not given, compute it
        if spread is None:
            # deep water wave steepness
            steepness = Hm0 / (g * Tp ** 2. / (2. * np.pi))
            # computation of parameter smax
            steepness_points = [0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 1.0]
            smax_points = [160, 160, 70, 35, 20, 7, 2.001, 2.0]
            # Interpolate linear
            smax = np.interp(steepness, steepness_points, smax_points)
            # Calculate spread for all frequencies
            power = np.ones_like(self.f) * -2.5
            power[self.f <= fp] = 5.0
            spread = np.power((self.f / fp) , power * smax(steepness))

        # If given as input, create array of the number
        else:
            spread = np.ones_like(self.f) * spread
            
        # Convert to range -np.pi to np.pi relative to the mean wave direction
        Az = ( ((bins - Theta) + np.pi) % (2 * np.pi) - np.pi ) / 2

        # calculate G0
        for i_f, s in enumerate(spread):
            # computation of G0 (Equation 2.23)
            G0[i_f] = 1 / (np.sum((np.cos(Az)) ** (2 * s)))

            # computation of G (Equation 2.22)
            for i_d, d in enumerate(Az):
                
                # If the direction is (smaller than half pi) AND (larger than minus half pi), so
                # if the direction is from the 'right' half of the circle
                # if d <= np.pi / 2 and d >= -np.pi / 2.0:
                if abs(d) <= np.pi / 2.0:
                    G[i_f, i_d] = G0[i_f] * np.power(np.cos(d), 2.0 * s) / dtheta
                    S_2d[i_f, i_d] = S_1d[i_f] * G[i_f, i_d]
                
                # If it is from the other side of the spectrum
                else:
                    S_2d[i_f, i_d] = 0.
        

        ## Test to check Hm0
        # dw = np.ones_like(self.f) * np.max(self.f) / self.f.size
        df = np.diff(self.f[:2])
        Hm0_test = 4 * np.sqrt((S_2d * df * dtheta).sum())

        logger.debug('Hm0_test', Hm0_test)

        return S_2d

    def n_angle0(self, theta):
        return np.mod(theta + np.pi, 2 * np.pi) - np.pi


def jonswap(f, Hm0, Tp, gamma=3.3, sigma_low=.07, sigma_high=.09, g=9.81, method='yamaguchi', normalize=False):
    """
    Generate JONSWAP spectrum

    Parameters
    ----------
    Hm0 : float
        Required zeroth order moment wave height
    Tp : float
        Required peak wave period
    gamma : float
        JONSWAP peak-enhancement factor (default: 3.3)
    sigma_low : float
        Sigma value for frequencies <= ``1/Tp`` (default: 0.07)
    sigma_high : float
        Sigma value for frequencies > ``1/Tp`` (default: 0.09)
    g : float
        Gravitational constant (default: 9.81)
    method : str
        Method to compute alpha (default: yamaguchi)
    normalize : bool
        Normalize resulting spectrum to match ``Hm0``

    Returns
    -------
    E : numpy.ndarray
        Array of shape ``f`` with wave energy densities 

    """

    # Pierson-Moskowitz
    if method.lower() == 'yamaguchi':
        alpha = 1. / (.06533 * gamma ** .8015 + .13467) / 16.
    elif method.lower() == 'goda':
        alpha = 1. / (.23 + .03 * gamma - .185 / (1.9 + gamma)) / 16.
    else:
        raise ValueError('Unknown method: {}'.format(method))
        
    # E_pm(f) = alpha_pm * g^2 (2pi)^-4 f^-5 exp(-(5/4) * (f/f_PM)^-4 )
    E_pm = alpha * Hm0 ** 2 * Tp ** -4 * f ** -5 * np.exp(-1.25 * (Tp * f) ** -4)
    
    # JONSWAP
    sigma = np.ones(f.shape) * sigma_low
    sigma[f > 1. / Tp] = sigma_high

    E_js = E_pm * gamma ** np.exp(-0.5 * (Tp * f - 1) ** 2. / sigma ** 2.)

    if normalize:
        E_js *= Hm0 ** 2. / (16. * np.trapz(E_js, f))
    
    return E_js

def jonswap_2d(f, theta, S_1d, Hm0, Tp, Theta, spread, g=9.81, test=False):
    """
    Calculates a 2d JONSWAP spectral density

    Parameters
    ----------
    S_1d : numpy.ndarray
        Array with 1d wave energy densities 
    Hm0 : float
        Required zeroth order moment wave height
    Tp : float
        Required peak wave period
    Theta : float
        mean wave direction [deg] w.r.t. North
    spread : float
        Determines the spreading [-]
    g : float
        Gravitational constant (default: 9.81)
    test : bool
        Whether to check the resulting wave height with the input waveheight

    Returns
    -------
    s2d : numpy.ndarray
        the spectral density
    """
    
    # peak frequency
    fp = 1.0 / Tp

    # Determine the stepsize in the directions
    dtheta = abs(np.deg2rad(np.diff(theta[:2])[0]))
    bins = np.deg2rad(theta % 360)
    theta0 = np.deg2rad(Theta % 360)

    nf = len(f)
    nd = len(theta)

    # If the spread is not given, compute it
    if spread is None:
        # deep water wave steepness
        steepness = Hm0 / (g * Tp ** 2. / (2. * np.pi))
        # computation of parameter smax
        steepness_points = [0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 1.0]
        smax_points = [160, 160, 70, 35, 20, 7, 2.001, 2.0]
        # Interpolate linear
        smax = np.interp(steepness, steepness_points, smax_points)
        # Calculate spread for all frequencies
        power = np.ones_like(f) * -2.5
        power[f <= fp] = 5.0
        spread = (f / fp) ** (power * smax)

    # If given as input, create array of the number
    else:
        spread = np.ones_like(f) * spread
        
    # Convert to range -np.pi to np.pi relative to the mean wave direction
    Az = ( ((bins - theta0) + np.pi) % (2 * np.pi) - np.pi ) / 2
    
    # computation of G0 (Equation 2.23) for all frequencies
    G0 = 1 / np.sum(np.cos(Az[:, np.newaxis]) ** (2 * spread[np.newaxis, :]), axis=0)
    
    # If the direction is (smaller than half pi) AND (larger than minus half pi), so
    # if the direction is from the 'right' half of the circle
    G = G0[:, np.newaxis] * np.cos(Az[np.newaxis, :]) ** (2.0 * spread[:, np.newaxis]) / dtheta
    S_2d = S_1d[:, np.newaxis] * G
            
    # If it is from the other side of the spectrum
    S_2d[:, np.abs(Az) > np.pi / 2.0] = 0.

    if test:
        # Multiply each energy density bin width its area and sum. Calculate wave height from this
        Hm0_1D_test = 4 * np.sqrt(np.trapz(S_1d, f))
        Hm0_2D_test = 4 * np.sqrt(np.trapz(S_2d.sum(axis=1) * dtheta, f).sum())

        logger.debug('Input Hm0: {:.3f} m\nJONSWAP 1D Hm0: {:.3f} m\nJONSWAP 2D Hm0: {:.3f} m'.format(
            Hm0, Hm0_1D_test, Hm0_2D_test
        ))
    

    return S_2d

def incoming_wave_factors(directions, dike_normal):
    """
    Calculate the incoming wave factors

    Parameters
    ----------
    directions : array of floats
        Wave directions for which the factor is calculated
    dike_normal : float
        Angle in degrees of the dike normal
    """
    # Convert float or int to array is necessary
    if isinstance(dike_normal, (float, int, np.float, np.int)):
        dike_normal = np.array([dike_normal])

    # Initialize empty array for factors
    fac = np.zeros((len(dike_normal), len(directions)))

    # Calculate (half) bin width
    halfbin = 0.5 * abs(np.diff(directions[:2])[0])

    # For each normal, interpolate the factors
    for i, normal in enumerate(dike_normal):
        # Determine interpolation points
        angles_p = [(normal + angle) % 360 for angle in [-90 - halfbin, -90 + halfbin, 90 - halfbin, 90 + halfbin]]
        factors_p = [0, 1, 1, 0]
        # Interpolate factors
        fac[i] = np.interp(directions, angles_p, factors_p, period=360)

    # Return the factors, squeeze in case the input was a single value
    return fac
