import numpy as np
from scipy.integrate import simps
from scipy.optimize import basinhopping
from time import time


twopi = np.pi * 2


def By_surface(x, w, d, j):
    """Magnetic field directed perpendicular to the
    surface of a conducting slab at the slabs surface,
    uniform current:

    Field at this top surface along +y (up)

         -----------------------------  y = d/2
         |                           |
         |             Slab          | y = 0
         |                           |
         ----------------------------- y = -d/2
      x = -w/2        x = 0        x = w/2

    ^+y
    |
    |
    ---->+x

    z out of monitor towards keyboard

    Current density is uniform and along +z

    The equation is from this paper:
    "Distribution of the magnetic field induced by a current
    passing through slabs in the superconducting and normal states"
    D.D. Prokof'ev
    Technical Physics June 2006, Volume 51, Issue 6, pp 675–682
    doi:10.1134/S1063784206060016

    Arguments:
        x (float): coordinate where field is computed (meters)
        w (float): width of slab in x direction (meters).
        d (float): depth of slab in y direction (meters).
        j (float): current density in slab (Amps/meters**2)

    Returns:
        By (float): Magnetic field at x (Tesla).
    """
    A = w - 2*x
    B = w + 2*x
    C = 2*d
    mu0_over_4pi = 1e-7
    return mu0_over_4pi * j * (-A * np.arctan(C / A) +
                               B * np.arctan(C / B) +
                               C/2 * np.log((B**2 + C**2) / (A**2 + C**2)))


def By_2d_approximation(x, w, d, j):
    """Approximation of By_surface valid except near edges of slab."""
    mu0_over_4pi = 1e-7
    return 2e-7 * j * d * np.log((w/2 + x) / (w/2 - x))


def g1d(x, x0, beam_fwhm):
    """1d Gaussian centered at x0 and with FWHM beam_fwhm."""
    s = beam_fwhm / 2.3548
    return twopi**-0.5 / s * np.exp(-(x - x0)**2/(2*s**2))


def box1d(x, x0, d):
    """Box function. Equal 1.0 for x less than d from x0 and 0 otherwise."""
    return (abs(x - x0) <= d).astype(int)


class SlabscanFitParams(object):
    """Object that keeps track of all fit parameters, their 
    normalizing scale factors and pretty printing of the parameters.

    All constructor parameters must be tuples like 
    (parameter_val, normalizing_scale_factor). The scale factor is so 
    that minimizers run smoothly 
    """
    def __init__(self, center, width, height, thickness, fwhm, current):
        self.center, self.center_norm  = center
        self.width, self.width_norm = width
        self.height, self.height_norm = height
        self.thickness, self.thickness_norm = thickness
        self.fwhm, self.fwhm_norm = fwhm
        self.current, self.current_norm = current

    def pp(self):
        fmt_str = ("center:     {:.2f} um   ({:.2f}) \n"
                   "width:      {:.2f} um   ({:.2f}) \n"
                   "height:     {:.2f} mV   ({:.2f}) \n"
                   "thickness:  {:.2f} nm   ({:.2f}) \n"
                   "fwhm:       {:.2f} um   ({:.2f}) \n"
                   "current:    {:.2f} mA   ({:.2f}) \n")
        print(fmt_str.format(self.center * 1e6, self.center_norm * 1e6,
                             self.width * 1e6, self.width_norm * 1e6,
                             self.height, self.height_norm,
                             self.thickness * 1e9, self.thickness_norm * 1e9,
                             self.fwhm * 1e6, self.fwhm_norm * 1e6,
                             self.current * 1e3, self.current_norm * 1e3))


    def ndarray_of(self, names):
        return np.array([getattr(self, n) for n in names])

    def ndarray_of_norms_for(self, names):
        return np.array([getattr(self, n + '_norm') for n in names])

    def normalized_ndarray_of(self, names):
        return self.ndarray_of(names)/self.ndarray_of_norms_for(names)


def get_box_fit_err_func(xdata, ydata, param_normalizers, 
                         cutoff_fwhm=3, npoints=400):
    """Create a function that finds the squared error between a set
    of data and a gaussian convoluted box function.

    Args:
        xdata (ndarray): Independent axis data
        ydata (ndarray): Dependent axis data
        param_normalizers: (center, height, width, fwhm) normalizing
            scale factors.
        fwhm_cutoff (float): Number of fwhm to integrate over in the
            convolution.
        npoints (int): Number of integration points in the convolution.
    """
    def err_func(normalized_params):
        """Error funcdtion to be minimzed.

        Args:
            normalized_params: (center, height, width, fwhm)
        """
        params = normalized_params * param_normalizers
        yfit = box_fit_func(xdata, params, cutoff_fwhm=cutoff_fwhm,
                            npoints=npoints)
        errvec = (yfit - ydata)
        return errvec.dot(errvec)

    return err_func


def get_oer_fit_err_func(xdata, ydata, param_normalizers, 
                         cutoff_fwhm=3, npoints=400):
    """Create a function that finds the squared error between a set
    of data and a gaussian convoluted box function.

    Args:
        xdata (ndarray): Independent axis data
        ydata (ndarray): Dependent axis data
        param_normalizers: (center, height, width, fwhm) normalizing
            scale factors.
        fwhm_cutoff (float): Number of fwhm to integrate over in the
            convolution.
        npoints (int): Number of integration points in the convolution.
    """
    def err_func(normalized_params):
        """Error funcdtion to be minimzed.

        Args:
            normalized_params: (center, height, width, fwhm)
        """
        params = normalized_params * param_normalizers
        yfit = oer_fit_func(xdata, params, cutoff_fwhm=cutoff_fwhm,
                            npoints=npoints)
        errvec = (yfit - ydata)
        return errvec.dot(errvec)

    return err_func


def box_fit_func(x, params, cutoff_fwhm=3, npoints=400):
    """A convoluted box function.

    Args:
        x (ndarray): Points to evaluate function at.
        params (ndarray): Box parameters in absolute terms, not
            normalized. In order (center, height, width, fwhm).
        fwhm_cutoff (float): Number of fwhm to integrate over in the
            convolution.
        npoints (int): Number of integration points in the convolution.
    """
    center, height, width, fwhm = params
    window_halfwidth = cutoff_fwhm * fwhm / 2
    xx = np.linspace(-window_halfwidth, window_halfwidth, npoints)
    y = []
    for xi in x:
        xxi = xx + xi
        box = height * box1d(xxi, center, width/2)
        y.append(simps(box * g1d(xxi, xi, fwhm), xx))
    return np.array(y)


def oersted_fit_func(x, params, cutoff_fwhm=3, npoints=400):
    """A convoluted boxed By function

    Args:
        x (ndarray): Points to evaluate function at.
        params (ndarray): Box parameters in absolute terms, not
            normalized. In order (center, height, width, fwhm).
        fwhm_cutoff (float): Number of fwhm to integrate over in the
            convolution.
        npoints (int): Number of integration points in the convolution.
    """
    center, height, width, thickness, fwhm, current = params
    window_halfwidth = cutoff_fwhm * fwhm / 2
    y = []
    j = current / (width * thickness)
    xx = np.linspace(-window_halfwidth, window_halfwidth, npoints)
    for xi in x:
        xxi = xx + xi
        box = box1d(xxi, center, width/2)
        By = height * By_surface(xxi - center, width, thickness, j)
        y.append(simps(box * g1d(xxi, xi, fwhm) * By, xx))
    return np.array(y)


def boxfit_ps2str(params):
    fmt_str = ("center: {:.2f} um    height: {:.2f} mV    "
               "width: {:.2f} um    fwhm: {:.2f} um")
    return fmt_str.format(*(params * (1e6, 1, 1e6, 1e6)))


def get_bh_callback(param_normalizers, verbose=False):
    def bh_callback(normd_params, f, accept):
        if verbose:
            x_str = boxfit_ps2str(normd_params * param_normalizers)
            print(x_str + "    {:.3f}    {}".format(f, accept))
    return bh_callback


def bhop_boxfit(data, initial_params, cutoff_fwhm=3, npoints=200, 
                niter=100, verbose=False):
    """Use scipy basinhopping to fit a box profile convolved
    with a gaussian beam.

    Args:
        data: Tuple of (xdata, ydata) to be fitted
        initial_params: SlabscanFitParams object for this fit.
        fwhm_cutoff (float): Number of fwhm to integrate over in the
            convolution.
        npoints (int): Number of integration points in the convolution.

    Returns:
        A dict with the following keys: Time (t), minima (xmin),
        basinhopping result object (res), xmin as a nice string
        (xmin_as_str).
    """
    xdata, ydata = data
    param_names = ('center', 'height', 'width', 'fwhm')
    params_tuple = initial_params.ndarray_of(param_names)
    param_normalizers = initial_params.ndarray_of_norms_for(param_names)

    err_func = get_box_fit_err_func(xdata, ydata, param_normalizers,
                                    cutoff_fwhm, npoints)
    # bh_callback = get_bh_callback(param_normalizers, verbose)
    t0 = time()

    res = basinhopping(
        func=err_func,
        x0=initial_params.normalized_ndarray_of(param_names),
        T=1.0,
        stepsize=0.1,
        niter=niter)

    if verbose:
        print('Seconds', (time()-t0))
        print('Initial params: ')
        initial_params.pp()
        print('Final params  : ', boxfit_ps2str(res.x * param_normalizers))
    return {
        't': time() - t0,
        'res': res,
        'xmin': res.x * param_normalizers,
        'xmin_as_str': boxfit_ps2str(res.x * param_normalizers)
    }


def bhop_oerfit(data, initial_params, cutoff_fwhm=3, npoints=200, 
                niter=100, verbose=False):
    """Use scipy basinhopping to fit a box profile convolved
    with a gaussian beam.

    Args:
        data: Tuple of (xdata, ydata) to be fitted
        initial_params: SlabscanFitParams object for this fit.
        fwhm_cutoff (float): Number of fwhm to integrate over in the
            convolution.
        npoints (int): Number of integration points in the convolution.

    Returns:
        A dict with the following keys: Time (t), minima (xmin),
        basinhopping result object (res).
    """
    xdata, ydata = data
    param_names = ('center', 'height', 'width', 'fwhm')
    params_tuple = initial_params.ndarray_of(param_names)
    param_normalizers = initial_params.ndarray_of_norms_for(param_names)

    err_func = get_oer_fit_err_func(xdata, ydata, param_normalizers,
                                    cutoff_fwhm, npoints)
    # bh_callback = get_bh_callback(param_normalizers, verbose)
    t0 = time()

    res = basinhopping(
        func=err_func,
        x0=initial_params.normalized_ndarray_of(param_names),
        T=1.0,
        stepsize=0.1,
        niter=niter)

    if verbose:
        print('Seconds', (time()-t0))
        print('Initial params: ')
        initial_params.pp()
        print('Final params  : ', boxfit_ps2str(res.x * param_normalizers))
    return {
        't': time() - t0,
        'res': res,
        'xmin': res.x * param_normalizers
    }
