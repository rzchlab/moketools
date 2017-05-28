import numpy as np
from scipy.integrate import simps
from scipy.optimize import basinhopping, brute
from time import time, strftime
import numpy as np
import holoviews as hv
import matplotlib.pyplot as plt
import pandas as pd
import uncertainties as un
from uncertainties import unumpy as unp
from types import SimpleNamespace
from pprint import pprint as pp
import pint
from functools import reduce
from operator import add
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.optimize import curve_fit, root, minimize, basinhopping
from moketools.fitting import slabscan as ss
from copy import deepcopy
import copy


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

def load_sotpmoke_data(datapath):
    """Load sot pmoke datafile from `datapath` and put useful columns into 
    a namespace object. Attributes are:
        - [d](X|Y|T)(p|m|sym|asym)
            - (X|Y|T) is X, Y or Theta from the lockin
            - [d] Preadding the 'd' means std-dev of whatever follows
            - (p|m) 'p' means B//-J and 'm' means B//J asssuming the usualy 
              experimental geometry.
        - xum
            - scan displacement in um
    """
    data = pd.read_csv(datapath, sep='\t')
    res = SimpleNamespace()
    
    xcols = ('X+mean(V)', 'X+std(V)', 'X-mean(V)', 'X-std(V)')
    res.Xp, res.dXp, res.Xm, res.dXm = [data[col] for col in xcols]
    Xsym = (unp.uarray(res.Xp, res.dXp) + unp.uarray(res.Xm, res.dXm))/2
    res.Xsym = unp.nominal_values(Xsym)
    res.dXsym = unp.std_devs(Xsym)
    Xasym = (unp.uarray(res.Xp, res.dXp) - unp.uarray(res.Xm, res.dXm))/2
    res.Xasym = unp.nominal_values(Xasym)
    res.dXasym = unp.std_devs(Xasym)
    
    ycols = ('Y+mean(V)', 'Y+std(V)', 'Y-mean(V)', 'Y-std(V)')
    res.Yp, res.dYp, res.Ym, res.dYm = [data[col] for col in ycols]
    Ysym = (unp.uarray(res.Yp, res.dYp) + unp.uarray(res.Ym, res.dYm))/2
    res.Ysym = unp.nominal_values(Xsym)
    res.dYsym = unp.std_devs(Ysym)
    Yasym = (unp.uarray(res.Yp, res.dYp) - unp.uarray(res.Ym, res.dYm))/2
    res.Yasym = unp.nominal_values(Yasym)
    res.dYasym = unp.std_devs(Yasym)

    tcols = ('T+mean(deg)', 'T+std(deg)', 'T-mean(deg)', 'T-std(deg)')
    res.Tp, res.dTp, res.Tm, res.dTm = [data[col] for col in tcols]
    res.Tsym = (res.Tp + res.Tm)/2
    res.Tasym = (res.Tp - res.Tm)/2
    
    res.xum = data['displ(um)']
    
    res.I = data['I(mv)']
    res.Imv = res.I * 1e3
    
    return res


def plot_raw_data(d):
    layout = (hv.Curve((d.xum, d.Xp), label='X B+') + 
              hv.Curve((d.xum, d.Xm), label='X B-') + 
              hv.Curve((d.xum, d.Xsym), label='X Sym') + 
              hv.Curve((d.xum, d.Xasym), label='X Asym'))
    return layout()


def plot_intensity(d):
    return hv.Curve((d.xum, d.Imv), label='Intensity', kdims=['um'], 
                    vdims=['I (mV)'])


def ebar_plot_raw_data(d, fs=None):
    fig, ax = plt.subplots(ncols=4, figsize=(16, 4))
    title = ('B+', 'B-', 'Sym', 'Asym')
    ylabel = ('X B+ (uV)', 'X B- (uV)', 'uV', 'uV')
    xlabel = ['x (um)' for x in range(4)]
    for x, t, yl, xl in zip(ax, title, ylabel, xlabel):
        x.set_title(t, fontsize=fs)
        x.set_xlabel(xl, fontsize=fs)
        x.set_ylabel(yl, fontsize=fs)
    opts = dict(ecolor='k', barsabove=False, capthick=20, elinewidth=1)
    ax[0].errorbar(d.xum, 1e6*d.Xp, yerr=1e6*d.dXp, **opts)
    ax[1].errorbar(d.xum, 1e6*d.Xm, yerr=1e6*d.dXm, **opts)
    ax[2].errorbar(d.xum, 1e6*d.Xsym, yerr=1e6*d.dXsym, **opts)
    ax[3].errorbar(d.xum, 1e6*d.Xasym, yerr=1e6*d.dXasym, **opts)
    return fig, ax


def ebar_plot_raw_data_overlap(d, fs=None):
    fig, ax = plt.subplots(ncols=3, figsize=(16, 4))
    title = ('B+-', 'Sym', 'Asym')
    ylabel = ['uV' for x in range(4)]
    xlabel = ['x (um)' for x in range(4)]
    for x, t, yl, xl in zip(ax, title, ylabel, xlabel):
        x.set_title(t, fontsize=fs)
        x.set_xlabel(xl, fontsize=fs)
        x.set_ylabel(yl, fontsize=fs)
    opts = dict(ecolor='k', barsabove=False, capthick=100, elinewidth=1)
    ax[0].errorbar(d.xum, 1e6*d.Xp, yerr=1e6*d.dXp, label='B+', **opts)
    ax[0].errorbar(d.xum, 1e6*d.Xm, yerr=1e6*d.dXm, label='B-', color='r', 
                   **opts)
    ax[1].errorbar(d.xum, 1e6*d.Xsym, yerr=1e6*d.dXsym, **opts)
    ax[2].errorbar(d.xum, 1e6*d.Xasym, yerr=1e6*d.dXasym, **opts)
    ax[0].legend(loc='best')
    return fig, ax


def intensity_fit_plot(d, params):
    """Quick intensity plot. Compare data with params generated curve.
    Args:
        params: (center, height, width, fwhm, cutoff_fwhm, npoints)
    """
    Imv = d.Imv - min(d.Imv)
    dataplot = hv.Scatter((d.xum, Imv), label='data', kdims=['um'], 
                          vdims=['I (mV)'])
    fitx = np.linspace(min(d.xum), max(d.xum), 200)
    fit = box_func(fitx/1e6, **params)
    fitplot = hv.Curve((fitx, fit), label='fit', kdims=['um'], 
                       vdims=['I (mV)'])
    return dataplot * fitplot

def asymmetric_fit_plot(d, params):
    """Quick intensity plot. Compare data with params generated curve.
    Args:
        params: (center, height, width, fwhm, cutoff_fwhm, npoints)
    """
    dataplot = hv.Scatter((d.xum, d.Xasym * 1e6), label='data', kdims=['um'], 
                          vdims=['uV'])
    fitx = np.linspace(min(d.xum), max(d.xum), 200)
    fit = box_func(fitx/1e6, **params) * 1e6
    fitplot = hv.Curve((fitx, fit), label='fit', kdims=['um'], vdims=['uV'])
    return dataplot * fitplot

def symmetric_fit_plot(d, params, add_offset=True):
    """Quick intensity plot. Compare data with params generated curve.
    Args:
        params: (center, height, width, fwhm, cutoff_fwhm, npoints)
    """
    dataplot = hv.Scatter((d.xum, d.Xsym * 1e6), label='data', kdims=['um'], 
                          vdims=['uV'])
    fitx = np.linspace(min(d.xum), max(d.xum), 200)
    fit = oersted_func(fitx/1e6, **params) * 1e6
    if add_offset:
        fit += (max(d.Xsym) + min(d.Xsym)) / 2.0
    fitplot = hv.Curve((fitx, fit), label='fit', kdims=['um'], vdims=['uV'])
    return dataplot * fitplot


def symmetric_field_plot(d, params):
    """Quick intensity plot. Compare data with params generated curve.
    Args:
        params: dict of (center, height, width, fwhm, cutoff_fwhm, npoints)
    """
    newparams = deepcopy(params)
    newparams['height'] = 1
    fitx = np.linspace(min(d.xum), max(d.xum), 200)
    fit_smoothed = oersted_func(fitx/1e6, **newparams) * 1e3
    newparams['fwhm'] = 1e-8
    fit_notsmoothed = oersted_func(fitx/1e6, **newparams) * 1e3
    fitplot_s = hv.Curve((fitx, fit_smoothed), kdims=['um'], vdims=['mT'])
    fitplot_ns = hv.Curve((fitx, fit_notsmoothed), kdims=['um'], vdims=['mT'])
    return fitplot_s * fitplot_ns


def pp_params(params, noprint=False):
    params = deepcopy(params)
    params['center'] *= 1e6
    params['width'] *= 1e6
    params['fwhm'] *= 1e6
    fmt_str = ("center: {center:.2f} um\n"
               "width:  {width:.2f} um\n"
               "height:  {height:.2e} \n"
               "fwhm:  {fwhm:.2f} um\n")
    for key in ('thickness', 'current', 'cutoff_fwhm', 'npoints'):
        if key in params.keys():
            fmt_str += "{key}:  {{{key}}}\n".format(**dict(key=key))
    if noprint:
        return fmt_str.format(**params)
    else:
        print(fmt_str.format(**params))
        
def find_oersted_offset(ydata):
    return (max(ydata) + min(ydata))/2
