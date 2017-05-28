import numpy as np
from scipy.integrate import simps
from scipy.optimize import basinhopping, brute
import copy
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


def box_func(x, center, width, height, fwhm, cutoff_fwhm, npoints):
    """Gaussian smoothed box."""
    window_halfwidth = cutoff_fwhm * fwhm / 2
    xx = np.linspace(-window_halfwidth, window_halfwidth, npoints)
    y = []
    for xi in x:
        xxi = xx + xi
        box = height * box1d(xxi, center, width/2)
        y.append(simps(box * g1d(xxi, xi, fwhm), xx))
    return np.array(y)


def oersted_func(x, center, width, height, fwhm, thickness, current, 
                 cutoff_fwhm, npoints):
    """Gaussian smoothed oersted field profile from a slab."""
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


def get_box_err_func(xdata, ydata, cutoff_fwhm, npoints, squared=True):
    """Error between data and box_func."""
    def err_func(params):
        """Generated error funcion. Params=(center, width, height, fwhm)"""
        other_params = np.array((cutoff_fwhm, npoints))
        params = np.concatenate((params, other_params))
        yfit = box_func(xdata, *params)
        errvec = yfit - ydata
        error = errvec.dot(errvec)
        if squared:
            return error
        else:
            return np.sqrt(error)
    return err_func


def get_oersted_err_func(xdata, ydata, thickness, current, offset, cutoff_fwhm,
                         npoints, squared=True):
    """Error between data and box_func."""
    def err_func(params):
        """Generated error funcion. Params=(center, width, height, fwhm)"""
        other_params = np.array((thickness, current, cutoff_fwhm, npoints))
        params = np.concatenate((params, other_params))
        yfit = oersted_func(xdata, *params) + offset
        errvec = yfit - ydata 
        error = errvec.dot(errvec)
        if squared:
            return error
        else:
            return np.sqrt(error)
    return err_func


def brute_box_fit(data, ranges, Ns, cutoff_fwhm, npoints, verbose=False, 
                  finish=True):
    """Brute force box func fit.

    initial_params = (center, width, height, fwhm, thickness, current)
    """
    xdata, ydata = data
    err_func = get_box_err_func(xdata, ydata, cutoff_fwhm, npoints)
    t0 = time()

    if not finish:
        res = brute(func=err_func, ranges=ranges, Ns=Ns, finish=None)
    else: 
        res = brute(func=err_func, ranges=ranges, Ns=Ns)

    if verbose:
        t1 = time()
        dt = t1 - t0
        per_iter = 1000 * dt / (Ns**4)
        print("Runtime: {:.1f} s   ({:.3f} ms / iter.)".format(dt, per_iter))
        pp_params_ranges(res, ranges)
    
    final_params = {
        'center': res[0], 
        'width': res[1], 
        'height': res[2], 
        'fwhm': res[3], 
        'cutoff_fwhm': cutoff_fwhm, 
        'npoints': npoints}
    res = dict(xmin=res, final_params_dict=final_params)
    return res


def brute_oersted_fit(data, ranges, Ns, thickness, current, cutoff_fwhm, 
                      npoints, verbose=False, finish=True):
    """Brute force oersted func fit.

    initial_params = (center, width, height, fwhm, thickness, current)
    """
    xdata, ydata = data
    offset = (max(ydata) + min(ydata)) / 2
    err_func = get_oersted_err_func(xdata, ydata, thickness, current, offset,
                                    cutoff_fwhm, npoints)
    t0 = time()

    if not finish:
        res = brute(func=err_func, ranges=ranges, Ns=Ns, finish=None)
    else: 
        res = brute(func=err_func, ranges=ranges, Ns=Ns)

    if verbose:
        t1 = time()
        dt = t1 - t0
        per_iter = 1000 * dt / (Ns**4)
        print("Runtime: {:.1f} s   ({:.3f} ms / iter.)".format(dt, per_iter))
        pp_params_ranges(res, ranges)

    final_params = {
        'center': res[0], 
        'width': res[1], 
        'height': res[2], 
        'fwhm': res[3], 
        'thickness': thickness,
        'current': current,
        'cutoff_fwhm': cutoff_fwhm, 
        'npoints': npoints}
    res = dict(xmin=res, final_params_dict=final_params)

    return res


def pp_brute_res(res, oneline=True):
    """Print a formatted version of the fit params."""
    fmt_str = ("center: {:.2f} um    height: {:.2e}     "
               "width: {:.2f} um    fwhm: {:.2f} um")
    if not oneline:
        fmt_str = ("center: {:.2f} um \nheight: {:.2e}  \n"
                   "width: {:.2f} um  \nfwhm: {:.2f} um \n")
    return fmt_str.format(*(res * (1e6, 1, 1e6, 1e6)))


def compare_params(title1, ps1, title2, ps2, str_only=False):
    fmt_str = ("            {:10s}{:10s}\n"
               "center:     {:5.2f} um   {:5.2f} um \n"
               "width:      {:5.2f} um   {:5.2f} um \n"
               "height:     {:5.2e}      {:5.2e}  \n"
               "thickness:  {:5.2f} nm   {:5.2f} nm \n"
               "fwhm:       {:5.2f} um   {:5.2f} um \n"
               "current:    {:5.2f} mA   {:5.2f} mA \n")
    center1, widht1, height1, fwhm1 = ps1[:4]
    center2, widht2, height2, fwhm2 = ps2[:4]
    try:
        thickness1, current1 = ps1[4:]
        thickness2, current2 = ps2[4:]
    except:
        thickness1, current1 = 0.0, 0.0
        thickness2, current2 = 0.0, 0.0
    res_str = fmt_str.format(title1, title2,
                             center1 * 1e6, center2 * 1e6,
                             width1 * 1e6, width2 * 1e6,
                             height1, height2,
                             thickness1 * 1e9, thickness2 * 1e9,
                             fwhm1 * 1e6, fwhm2 * 1e6,
                             current1 * 1e3, current2 * 1e3)
    if str_only:
        return res_str
    else:
        print(res_str)
        return res_str


def pp_params_ranges(ps, ranges, str_only=False):
    fmt_str = ("center:     {:5.2f} um   ({:5.2f}, {:5.2f}) um \n"
               "width:      {:5.2f} um   ({:5.2f}, {:5.2f}) um \n"
               "height:     {:5.2e}      ({:5.2e}, {:5.2e})  \n"
               "fwhm:       {:5.2f} um   ({:5.2f}, {:5.2f}) um \n")

    ranges_noslices = []
    for r in ranges:
        if isinstance(r, slice):
            ranges_noslices.append((r.start, r.stop))
        else:
            ranges_noslices.append(r)

    center1, width1, height1, fwhm1 = ps[:4]
    (cl, cu), (wl, wu), (hl, hu), (fl, fu) = ranges_noslices
    res_str = fmt_str.format(center1 * 1e6, cl * 1e6, cu * 1e6,
                             width1 * 1e6, wl * 1e6, wu * 1e6,
                             height1, hl, hu,
                             fwhm1 * 1e6, fl * 1e6, fu * 1e6)

    if str_only:
        return res_str
    else:
        print(res_str)
        return res_str


#     def compare(self, title, other, othertitle):
#         fmt_str = ("            {:10s}{:10s}\n"
#                    "center:     {:5.2f} um   {:5.2f} um \n"
#                    "width:      {:5.2f} um   {:5.2f} um \n"
#                    "height:     {:5.2e} mV   {:5.2e} mV \n"
#                    "thickness:  {:5.2f} nm   {:5.2f} nm \n"
#                    "fwhm:       {:5.2f} um   {:5.2f} um \n"
#                    "current:    {:5.2f} mA   {:5.2f} mA \n")
#         print(fmt_str.format(title, othertitle,
#                              self.center * 1e6, other.center * 1e6,
#                              self.width * 1e6, other.width * 1e6,
#                              self.height, other.height,
#                              self.thickness * 1e9, other.thickness * 1e9,
#                              self.fwhm * 1e6, other.fwhm * 1e6,
#                              self.current * 1e3, other.current * 1e3))

# def boxfit_ps2str(params, oneline=False):
#     fmt_str = ("center: {:.2f} um    height: {:.2f} mV    "
#                "width: {:.2f} um    fwhm: {:.2f} um")
#     if not oneline:
#         fmt_str = ("center: {:.2f} um \nheight: {:.2f} mV \n"
#                    "width: {:.2f} um  \nfwhm: {:.2f} um \n")
#     return fmt_str.format(*(params * (1e6, 1, 1e6, 1e6)))

# class SlabscanFitParams(object):
#     """Object that keeps track of all fit parameters, their 
#     normalizing scale factors and pretty printing of the parameters.

#     All constructor parameters must be tuples like 
#     (parameter_val, normalizing_scale_factor). The scale factor is so 
#     that minimizers run smoothly 
#     """
#     def __init__(self, center, width, height, thickness, fwhm, current):
#         self.center, self.center_norm  = center
#         self.width, self.width_norm = width
#         self.height, self.height_norm = height
#         self.thickness, self.thickness_norm = thickness
#         self.fwhm, self.fwhm_norm = fwhm
#         self.current, self.current_norm = current

#     def pp(self):
#         fmt_str = ("center:     {:5.2f} um   ({:5.2f}) \n"
#                    "width:      {:5.2f} um   ({:5.2f}) \n"
#                    "height:     {:5.2e} mV   ({:5.2e}) \n"
#                    "thickness:  {:5.2f} nm   ({:5.2f}) \n"
#                    "fwhm:       {:5.2f} um   ({:5.2f}) \n"
#                    "current:    {:5.2f} mA   ({:5.2f}) \n")
#         print(fmt_str.format(self.center * 1e6, self.center_norm * 1e6,
#                              self.width * 1e6, self.width_norm * 1e6,
#                              self.height, self.height_norm,
#                              self.thickness * 1e9, self.thickness_norm * 1e9,
#                              self.fwhm * 1e6, self.fwhm_norm * 1e6,
#                              self.current * 1e3, self.current_norm * 1e3))

#     def compare(self, title, other, othertitle):
#         fmt_str = ("            {:10s}{:10s}\n"
#                    "center:     {:5.2f} um   {:5.2f} um \n"
#                    "width:      {:5.2f} um   {:5.2f} um \n"
#                    "height:     {:5.2e} mV   {:5.2e} mV \n"
#                    "thickness:  {:5.2f} nm   {:5.2f} nm \n"
#                    "fwhm:       {:5.2f} um   {:5.2f} um \n"
#                    "current:    {:5.2f} mA   {:5.2f} mA \n")
#         print(fmt_str.format(title, othertitle,
#                              self.center * 1e6, other.center * 1e6,
#                              self.width * 1e6, other.width * 1e6,
#                              self.height, other.height,
#                              self.thickness * 1e9, other.thickness * 1e9,
#                              self.fwhm * 1e6, other.fwhm * 1e6,
#                              self.current * 1e3, other.current * 1e3))


#     def ndarray_of(self, names):
#         return np.array([getattr(self, n) for n in names])

#     def ndarray_of_norms_for(self, names):
#         return np.array([getattr(self, n + '_norm') for n in names])

#     def normalized_ndarray_of(self, names):
#         return self.ndarray_of(names)/self.ndarray_of_norms_for(names)

#     def updated_copy(self, **kwargs):
#         """Create a new (deepcopied) params object with updated 
#         values but not norms.

#         Args:
#             kwargs: any of the original parameters, but only pass the value
#                 not the norm. For example `updated_copy(center=5e-6)`.
#         """

#         new_params = copy.deepcopy(self)
#         for k, v in kwargs.items():
#             setattr(new_params, k, v)
#         return new_params


# def get_box_fit_err_func(xdata, ydata, param_normalizers, 
#                          cutoff_fwhm=3, npoints=400):
#     """Create a function that finds the squared error between a set
#     of data and a gaussian convoluted box function.

#     Args:
#         xdata (ndarray): Independent axis data
#         ydata (ndarray): Dependent axis data
#         param_normalizers: (center, height, width, fwhm) normalizing
#             scale factors.
#         fwhm_cutoff (float): Number of fwhm to integrate over in the
#             convolution.
#         npoints (int): Number of integration points in the convolution.
#     """
#     def err_func(normalized_params):
#         """Error funcdtion to be minimzed.

#         Args:
#             normalized_params: (center, height, width, fwhm)
#         """
#         params = normalized_params * param_normalizers
#         yfit = box_fit_func(xdata, params, cutoff_fwhm=cutoff_fwhm,
#                             npoints=npoints)
#         errvec = (yfit - ydata)
#         return errvec.dot(errvec)

#     return err_func


# def get_box_brute_fit_err_func(xdata, ydata, cutoff_fwhm=3, npoints=400):
#     """Create a function that finds the squared error between a set
#     of data and a gaussian convoluted box function.

#     Args:
#         xdata (ndarray): Independent axis data
#         ydata (ndarray): Dependent axis data
#         fwhm_cutoff (float): Number of fwhm to integrate over in the
#             convolution.
#         npoints (int): Number of integration points in the convolution.
#     """
#     def err_func(params):
#         """Error funcdtion to be minimzed.

#         Args:
#             normalized_params: (center, height, width, fwhm)
#         """
#         yfit = box_fit_func(xdata, params, cutoff_fwhm=cutoff_fwhm,
#                             npoints=npoints)
#         errvec = (yfit - ydata)
#         return errvec.dot(errvec)

#     return err_func


# def get_oer_fit_err_func(xdata, ydata, nonfit_params, param_normalizers, 
#                          cutoff_fwhm=3, npoints=400):
#     """Create a function that finds the squared error between a set
#     of data and a gaussian convoluted box function.

#     Args:
#         xdata (ndarray): Independent axis data
#         ydata (ndarray): Dependent axis data
#         nonfit_params: (thickness, current)
#         param_normalizers: (center, height, width, fwhm) normalizing
#             scale factors.
#         fwhm_cutoff (float): Number of fwhm to integrate over in the
#             convolution.
#         npoints (int): Number of integration points in the convolution.
#     """
#     thickness, current = nonfit_params

#     def err_func(normalized_params):
#         """Error funcdtion to be minimzed.

#         Args:
#             normalized_params: (center, height, width, fwhm)
#         """
#         params = normalized_params * param_normalizers
#         center, height, width, fwhm = params
#         params = np.array((center, height, width, thickness, fwhm, current))
        
#         yfit = oersted_fit_func(xdata, params, cutoff_fwhm=cutoff_fwhm,
#                                 npoints=npoints)
#         errvec = (yfit - ydata)
#         return errvec.dot(errvec)

#     return err_func


# def get_oer_brute_fit_err_func(xdata, ydata, nonfit_params, cutoff_fwhm=3,    
#                                npoints=400):
#     """Create a function that finds the squared error between a set
#     of data and a gaussian convoluted box function.

#     Args:
#         xdata (ndarray): Independent axis data
#         ydata (ndarray): Dependent axis data
#         nonfit_params: (thickness, current)
#         fwhm_cutoff (float): Number of fwhm to integrate over in the
#             convolution.
#         npoints (int): Number of integration points in the convolution.
#     """
#     thickness, current = nonfit_params

#     def err_func(params):
#         """Error funcdtion to be minimzed.

#         Args:
#             normalized_params: (center, height, width, fwhm)
#         """
#         center, height, width, fwhm = params
#         params = np.array((center, height, width, thickness, fwhm, current))
#         yfit = oersted_fit_func(xdata, params, cutoff_fwhm=cutoff_fwhm,
#                                 npoints=npoints)
#         errvec = (yfit - ydata)
#         return errvec.dot(errvec)

#     return err_func


# def box_fit_func(x, params, cutoff_fwhm=3, npoints=400):
#     """A convoluted box function.

#     Args:
#         x (ndarray): Points to evaluate function at.
#         params (ndarray): Box parameters in absolute terms, not
#             normalized. In order (center, height, width, fwhm).
#         fwhm_cutoff (float): Number of fwhm to integrate over in the
#             convolution.
#         npoints (int): Number of integration points in the convolution.
#     """
#     center, height, width, fwhm = params
#     window_halfwidth = cutoff_fwhm * fwhm / 2
#     xx = np.linspace(-window_halfwidth, window_halfwidth, npoints)
#     y = []
#     for xi in x:
#         xxi = xx + xi
#         box = height * box1d(xxi, center, width/2)
#         y.append(simps(box * g1d(xxi, xi, fwhm), xx))
#     return np.array(y)


# def oersted_fit_func(x, params, cutoff_fwhm=3, npoints=400):
#     """A convoluted boxed By function

#     Args:
#         x (ndarray): Points to evaluate function at.
#         params (ndarray): Box parameters in absolute terms, not
#             normalized. In order (center, height, width, thickness, 
#             fwhm, current).
#         fwhm_cutoff (float): Number of fwhm to integrate over in the
#             convolution.
#         npoints (int): Number of integration points in the convolution.
#     """
#     center, height, width, thickness, fwhm, current = params
#     window_halfwidth = cutoff_fwhm * fwhm / 2
#     y = []
#     j = current / (width * thickness)
#     xx = np.linspace(-window_halfwidth, window_halfwidth, npoints)
#     for xi in x:
#         xxi = xx + xi
#         box = box1d(xxi, center, width/2)
#         By = height * By_surface(xxi - center, width, thickness, j)
#         y.append(simps(box * g1d(xxi, xi, fwhm) * By, xx))
#     return np.array(y)


# def boxfit_ps2str(params, oneline=False):
#     fmt_str = ("center: {:.2f} um    height: {:.2f} mV    "
#                "width: {:.2f} um    fwhm: {:.2f} um")
#     if not oneline:
#         fmt_str = ("center: {:.2f} um \nheight: {:.2f} mV \n"
#                    "width: {:.2f} um  \nfwhm: {:.2f} um \n")
#     return fmt_str.format(*(params * (1e6, 1, 1e6, 1e6)))


# def get_bh_callback(param_normalizers, verbose=False):
#     def bh_callback(normd_params, f, accept):
#         if verbose:
#             x_str = boxfit_ps2str(normd_params * param_normalizers, 
#                                   oneline=True)
#             print(x_str + "    {:.3f}    {}".format(f, accept))
#     return bh_callback


# def bhop_boxfit(data, initial_params, cutoff_fwhm=3, npoints=200, 
#                 niter=100, verbose=False, usecallback=False):
#     """Use scipy basinhopping to fit a box profile convolved
#     with a gaussian beam.

#     Args:
#         data: Tuple of (xdata, ydata) to be fitted
#         initial_params: SlabscanFitParams object for this fit.
#         fwhm_cutoff (float): Number of fwhm to integrate over in the
#             convolution.
#         npoints (int): Number of integration points in the convolution.

#     Returns:
#         A dict with the following keys: Time (t), minima (xmin),
#         basinhopping result object (res), xmin as a nice string
#         (xmin_as_str).
#     """
#     xdata, ydata = data
#     param_names = ('center', 'height', 'width', 'fwhm')
#     params_tuple = initial_params.ndarray_of(param_names)
#     param_normalizers = initial_params.ndarray_of_norms_for(param_names)

#     err_func = get_box_fit_err_func(xdata, ydata, param_normalizers,
#                                     cutoff_fwhm, npoints)
#     if usecallback:
#         bh_callback = get_bh_callback(param_normalizers, verbose)
#     else:
#         bh_callback = None
#     t0 = time()

#     res = basinhopping(
#         func=err_func,
#         x0=initial_params.normalized_ndarray_of(param_names),
#         T=1.0,
#         stepsize=0.1,
#         callback=bh_callback,
#         niter=niter)

#     kwargs = {k: v for k, v in zip(param_names, res.x * param_normalizers) }
#     fps = initial_params.updated_copy(**kwargs)

#     if verbose:
#         print()
#         print('Seconds', (time()-t0), '\n')
#         initial_params.compare('initial', fps, 'final')

#     return {
#         't': time() - t0,
#         'res': res,
#         'xmin': res.x * param_normalizers,
#         'xmin_as_str': boxfit_ps2str(res.x * param_normalizers),
#         'final_params': fps,
#         'err_func': err_func,
#         'param_normalizers': param_normalizers
#     }


# def bhop_oerfit(data, initial_params, cutoff_fwhm=3, npoints=200, 
#                 niter=100, verbose=False):
#     """Use scipy basinhopping to fit a box profile convolved
#     with a gaussian beam.

#     Args:
#         data: Tuple of (xdata, ydata) to be fitted
#         initial_params: SlabscanFitParams object for this fit.
#         fwhm_cutoff (float): Number of fwhm to integrate over in the
#             convolution.
#         npoints (int): Number of integration points in the convolution.

#     Returns:
#         A dict with the following keys: Time (t), minima (xmin),
#         basinhopping result object (res).
#     """
#     xdata, ydata = data
#     param_names = ('center', 'height', 'width', 'fwhm')
#     params_tuple = initial_params.ndarray_of(param_names)
#     param_normalizers = initial_params.ndarray_of_norms_for(param_names)
#     nonfit_params = initial_params.ndarray_of(('thickness', 'current'))

#     err_func = get_oer_fit_err_func(xdata, ydata, nonfit_params,
#                                     param_normalizers, cutoff_fwhm, npoints)
#     # bh_callback = get_bh_callback(param_normalizers, verbose)
#     t0 = time()

#     res = basinhopping(
#         func=err_func,
#         x0=initial_params.normalized_ndarray_of(param_names),
#         T=1.0,
#         stepsize=0.1,
#         niter=niter)

#     if verbose:
#         print('Seconds', (time()-t0))
#         print('Initial params: ')
#         initial_params.pp()
#         print('Final params  : ', boxfit_ps2str(res.x * param_normalizers))
#     return { 't': time() - t0, 'res': res, 'xmin': res.x * param_normalizers}

# def brute_oerfit(data, ranges, initial_params, cutoff_fwhm=3, npoints=200, 
#                  brute_kwargs=None, verbose=False):
#     """Use scipy basinhopping to fit a box profile convolved
#     with a gaussian beam.

#     Args:
#         data: Tuple of (xdata, ydata) to be fitted
#         ranges (tuple): Tuple of the grid to check for (center, height,
#             width, fwhm)
#         initial_params: SlabscanFitParams object for this fit.
#         fwhm_cutoff (float): Number of fwhm to integrate over in the
#             convolution.
#         npoints (int): Number of integration points in the convolution.

#     Returns:
#         A dict with the following keys: Time (t), minima (xmin),
#         basinhopping result object (res).
#     """
#     xdata, ydata = data
#     nonfit_params = initial_params.ndarray_of(('thickness', 'current'))

#     err_func = get_oer_fit_err_func(xdata, ydata, nonfit_params,
#                                     param_normalizers, cutoff_fwhm, npoints)
#     t0 = time()
#     res = brute(func=err_func, ranges=ranges, Ns=7)
#     xmin = np.array(res)

#     param_names = ('center', 'height', 'width', 'fwhm')
#     kwargs = {k: v for k, v in zip(param_names, xmin) }
#     fps = initial_params.updated_copy(**kwargs)

#     if verbose:
#         print()
#         print('Seconds', (time()-t0), '\n')
#         initial_params.compare('initial', fps, 'final')

#     return { 't': time() - t0, 'res': res, 'xmin': xmin}


# def brute_boxfit(data, ranges, initial_params, cutoff_fwhm=3, npoints=200, 
#                  brute_kwargs=None, verbose=False, Ns=1):
#     """Use scipy basinhopping to fit a box profile convolved
#     with a gaussian beam.

#     Args:
#         data: Tuple of (xdata, ydata) to be fitted
#         ranges (tuple): Tuple of the grid to check for (center, height,
#             width, fwhm)
#         initial_params: SlabscanFitParams object for this fit.
#         fwhm_cutoff (float): Number of fwhm to integrate over in the
#             convolution.
#         npoints (int): Number of integration points in the convolution.

#     Returns:
#         A dict with the following keys: Time (t), minima (xmin),
#         basinhopping result object (res).
#     """
#     xdata, ydata = data

#     err_func = get_box_brute_fit_err_func(xdata, ydata, cutoff_fwhm, npoints)
#     t0 = time()
#     res = brute(func=err_func, ranges=ranges, Ns=Ns)
#     xmin = np.array(res)

#     param_names = ('center', 'height', 'width', 'fwhm')
#     kwargs = {k: v for k, v in zip(param_names, xmin) }
#     fps = initial_params.updated_copy(**kwargs)

#     if verbose:
#         print()
#         print('Seconds', (time()-t0), '\n')
#         initial_params.compare('initial', fps, 'final')

#     return {'t': time() - t0, 
#             'res': res, 
#             'xmin': xmin,
#             'err_func': err_func,
#             'final_params': fps}
