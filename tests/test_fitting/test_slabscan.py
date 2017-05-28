from moketools.fitting import slabscan as ss
import matplotlib.pyplot as plt
import numpy as np

SHOW_PLOTS = True

np.random.seed(seed=17576)

N = 100
xdata = np.linspace(-50e-6, 50e-6, N)
xdata_um = xdata * 1e6

# initial_params = ss.SlabscanFitParams(
#     center=(5e-6, 10e-6),
#     width=(40e-6, 50e-6),
#     height=(15, 10),
#     thickness=(10e-9, 10e-9),
#     fwhm=(7.5e-6, 4.5e-6),
#     current=(0.01, 0.01))

# actual_params = ss.SlabscanFitParams(
#     center=(0, 10e-6),
#     width=(50e-6, 50e-6),
#     height=(10, 10),
#     thickness=(10e-9, 10e-9),
#     fwhm=(4.5e-6, 4.5e-6),
#     current=(0.01, 0.01))


# initial_params = np.array((
#         3e-6,    # center
#         52e-6,   # width
#         12,   # height
#         3e-6,    # fwhm
#         30e-9,   # thickness
#         10e-3   # current

# ))

actual_params = np.array((
        0e-6,    # center
        50e-6,   # width
        10,   # height
        5e-6,    # fwhm
        30e-9,   # thickness
        10e-3   # current

))

cutoff_fwhm = 3
npoints = 300

# bf_param_names = ('center', 'height', 'width', 'fwhm')
# ydata_bf = ss.box_fit_func(
#         xdata, 
#         actual_params.ndarray_of(bf_param_names), 
#         cutoff_fwhm=cutoff_fwhm, 
#         npoints=npoints
# ) 
# oer_param_names = ('center', 'height', 'width', 'thickness', 'fwhm', 'current')
# ydata_oer = ss.oersted_fit_func(
#                 xdata,
#                 actual_params.ndarray_of(oer_param_names),
#                 cutoff_fwhm=cutoff_fwhm,
#                 npoints=npoints)
# ydata_oer += np.random.randn(N)/120000

center, width, height, fwhm, thickness, current = actual_params

ps = (center, width, height, fwhm, cutoff_fwhm, npoints)
ydata_box = ss.box_func(xdata, *ps) + np.random.randn(N)

ps = (center, width, height, fwhm, thickness, current, cutoff_fwhm, npoints)
ydata_oer = ss.oersted_func(xdata, *ps) + np.random.randn(N)/20000


def test_fake_boxfit_data():
    params = (center, width, height, fwhm, cutoff_fwhm, npoints)
    plt.plot(xdata_um, ydata_box, '-',
             xdata_um, ss.box_func(xdata, *params), '-')
    if SHOW_PLOTS:
        plt.title('Fake Box data plot')
        plt.tight_layout()
        plt.show()


def test_fake_oerfit_data():
    params = (center, width, height, fwhm, thickness, current, 
              cutoff_fwhm, npoints)
    plt.plot(xdata_um, ydata_oer, '-',
             xdata_um, ss.oersted_func(xdata, *params), '-')
    # plt.plot(xdata_um, ss.oersted_func(xdata, *params), '-')
    if SHOW_PLOTS:
        plt.title('Fake Oer data plot')
        plt.tight_layout()
        plt.show()


def test_brute_box():
    Ns = 3
    ranges = (
            (-5e-6, 5e-6),    # center
            (45e-6, 55e-6),   # width
            (5, 15),       # height
            (2.5e-6, 7.5e-6)  # fwhm
    )
    data = (xdata, ydata_box)
    final_params = ss.brute_box_fit(data, ranges, Ns, cutoff_fwhm, npoints, 
                                    verbose=True)['xmin']
    final_params = np.concatenate((final_params, (cutoff_fwhm, npoints)))
    plt.plot(xdata_um, ydata_box, '-',
             xdata_um, ss.box_func(xdata, *final_params))
    if SHOW_PLOTS:
        plt.title('Brute box')
        plt.tight_layout()
        plt.show()

def test_pp_params_ranges():
    ranges = (
            (-5e-6, 5e-6),    # center
            (45e-6, 55e-6),   # width
            (5, 15),       # height
            (2.5e-6, 7.5e-6)  # fwhm
    )
    params = actual_params
    ss.pp_params_ranges(params, ranges)



def test_brute_oersted():
    Ns = 1
    ranges = (
            (-5e-6, 5e-6),    # center
            (45e-6, 55e-6),   # width
            (5, 15),       # height
            (2.5e-6, 7.5e-6)  # fwhm
    )
    thickness, current = actual_params[-2:]
    data = (xdata, ydata_oer)
    final_params = ss.brute_oersted_fit(data, ranges, Ns, thickness, current,
                                        cutoff_fwhm, npoints, verbose=True,
                                        finish=True)['xmin']
    final_params = np.concatenate((final_params, 
                                  (thickness, current, cutoff_fwhm, npoints)))
    plt.plot(xdata_um, ydata_oer, '-',
             xdata_um, ss.oersted_func(xdata, *final_params))

    if SHOW_PLOTS:
        plt.title('Brute Oer')
        plt.tight_layout()
        plt.show()


def test_brute_oersted_with_slices():
    Ns = 1
    ranges = (
            (-5e-6, 5e-6),    # center
            (45e-6, 55e-6),   # width
            slice(5, 15, 1),       # height
            (2.5e-6, 7.5e-6)  # fwhm
    )
    thickness, current = actual_params[-2:]
    data = (xdata, ydata_oer)
    final_params = ss.brute_oersted_fit(data, ranges, Ns, thickness, current,
                                        cutoff_fwhm, npoints, verbose=True,
                                        finish=True)['xmin']
    final_params = np.concatenate((final_params, 
                                  (thickness, current, cutoff_fwhm, npoints)))
    plt.plot(xdata_um, ydata_oer, '-',
             xdata_um, ss.oersted_func(xdata, *final_params))

    if SHOW_PLOTS:
        plt.title('Brute Oer')
        plt.tight_layout()
        plt.show()


# def test_boxfit_basinhop():
#     data = (xdata, ydata_bf)
#     res = ss.bhop_boxfit(data, initial_params, verbose=True, niter=1)
#     print('Actual Params')
#     actual_params.pp()
#     fit_params = res['xmin']

#     plt.plot(xdata_um, ydata_bf, '-', 
#              xdata_um, ss.box_fit_func(xdata, fit_params), '-')
#     if SHOW_PLOTS:
#         plt.title('Bhop Fit Plot')
#         plt.tight_layout()
#         plt.show()


# def test_boxfit_brute():
#     data = (xdata, ydata_bf)
#     param_names = ('center', 'height', 'width', 'fwhm')
#     # initial_params.height = -10e-6
#     # initial_params.height_norm = 10e-6
#     params = initial_params.ndarray_of(param_names)

#     # ranges = [np.linspace(x - (0.2 * x), (x + (0.2 * x)), 5) 
#     #           for x in params]
#     # ranges[0] = np.linspace(-6e-6, 9e-6, 5)

#     ranges = [(x - (0.2 * x), (x + (0.2 * x))) for x in params]
#     ranges[0] = (-6e-6, 9e-6)

#     res = ss.brute_boxfit(data, ranges, initial_params, verbose=True)
#     # import pdb; pdb.set_trace()
#     print('Actual Params')
#     actual_params.pp()
#     center, height, width, fwhm = res['xmin']
#     thickness, current = actual_params.ndarray_of(('thickness', 'current'))
#     fit_params = (center, height, width, thickness, fwhm, current)

#     plt.plot(xdata_um, ydata_oer, '-', 
#              xdata_um, ss.oersted_fit_func(xdata, fit_params), '-')
#     if SHOW_PLOTS:
#         plt.title('Bhop Fit Plot')
#         plt.tight_layout()
#         plt.show()


# def test_oerfit_basinhop():
#     data = (xdata, ydata_oer)
#     res = ss.bhop_oerfit(data, initial_params, verbose=True, niter=50)
#     print('Actual Params')
#     actual_params.pp()
#     center, height, width, fwhm = res['xmin']
#     thickness, current = initial_params.ndarray_of(('thickness', 'current'))
#     fit_params = (center, height, width, thickness, fwhm, current)

#     plt.plot(xdata_um, ydata_oer, '-', 
#              xdata_um, ss.oersted_fit_func(xdata, fit_params), '-')
#     if SHOW_PLOTS:
#         plt.title('Bhop Fit Plot')
#         plt.tight_layout()
#         plt.show()

# def test_oerfit_brute():
#     data = (xdata, ydata_oer)
#     param_names = ('center', 'height', 'width', 'fwhm')
#     params = initial_params.ndarray_of(param_names)

#     # ranges = [np.linspace(x - (0.2 * x), (x + (0.2 * x)), 5) 
#     #           for x in params]
#     # ranges[0] = np.linspace(-6e-6, 9e-6, 5)

#     ranges = [(x - (0.2 * x), (x + (0.2 * x))) for x in params]
#     ranges[0] = (-6e-6, 9e-6)

#     res = ss.brute_oerfit(data, ranges, initial_params, verbose=True)
#     # import pdb; pdb.set_trace()
#     print('Actual Params')
#     actual_params.pp()
#     center, height, width, fwhm = res['xmin']
#     thickness, current = actual_params.ndarray_of(('thickness', 'current'))
#     fit_params = (center, height, width, thickness, fwhm, current)

#     plt.plot(xdata_um, ydata_oer, '-', 
#              xdata_um, ss.oersted_fit_func(xdata, fit_params), '-')
#     if SHOW_PLOTS:
#         plt.title('Bhop Fit Plot')
#         plt.tight_layout()
#         plt.show()
