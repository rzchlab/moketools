from moketools.fitting import slabscan as ss
import matplotlib.pyplot as plt
import numpy as np

SHOW_PLOTS = True

np.random.seed(seed=17576)

N = 100
xdata = np.linspace(-50e-6, 50e-6, N)
xdata_um = xdata * 1e6

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
