from moketools.fitting import slabscan as ss
import matplotlib.pyplot as plt
import numpy as np

SHOW_PLOTS = True

np.random.seed(seed=17576)

N = 100
xdata = np.linspace(-50e-6, 50e-6, N)
xdata_um = xdata * 1e6

initial_params = ss.SlabscanFitParams(
    center=(5e-6, 10e-6),
    width=(40e-6, 50e-6),
    height=(15, 10),
    thickness=(10e-9, 10e-9),
    fwhm=(7.5e-6, 4.5e-6),
    current=(0.01, 0.01))

actual_params = ss.SlabscanFitParams(
    center=(0, 10e-6),
    width=(50e-6, 50e-6),
    height=(10, 10),
    thickness=(10e-9, 10e-9),
    fwhm=(4.5e-6, 4.5e-6),
    current=(0.01, 0.01))

cutoff_fwhm = 3
npoints = 200

bf_param_names = ('center', 'height', 'width', 'fwhm')
ydata_bf = ss.box_fit_func(
        xdata, 
        actual_params.ndarray_of(bf_param_names), 
        cutoff_fwhm=cutoff_fwhm, 
        npoints=npoints
) 
ydata_bf += np.random.randn(N)

oer_param_names = ('center', 'height', 'width', 'thickness', 'fwhm', 'current')
ydata_oer = ss.oersted_fit_func(
                xdata,
                actual_params.ndarray_of(oer_param_names),
                cutoff_fwhm=cutoff_fwhm,
                npoints=npoints)
ydata_oer += np.random.randn(N)/12000

sample_profile = (actual_params.height * 
                  ss.box1d(xdata, actual_params.center, actual_params.width/2))


def test_fake_boxfit_data():
    plt.plot(xdata_um, ydata_bf, '-', xdata_um, sample_profile, '-')
    if SHOW_PLOTS:
        plt.title('Fake Box data plot')
        plt.tight_layout()
        plt.show()


def test_fake_oerfit_data():
    plt.plot(xdata_um, ydata_oer, '-')
    if SHOW_PLOTS:
        plt.title('Fake Oer data plot')
        plt.tight_layout()
        plt.show()


def test_boxfit_basinhop():
    data = (xdata, ydata_bf)
    res = ss.bhop_boxfit(data, initial_params, verbose=True, niter=50)
    print('Actual Params')
    actual_params.pp()
    fit_params = res['xmin']

    plt.plot(xdata_um, ydata_bf, '-', 
             xdata_um, ss.box_fit_func(xdata, fit_params), '-')
    if SHOW_PLOTS:
        plt.title('Bhop Fit Plot')
        plt.tight_layout()
        plt.show()

def test_oerfit_basinhop():
    data = (xdata, ydata_bf)
    res = ss.bhop_boxfit(data, initial_params, verbose=True, niter=1)
    print('Actual Params')
    actual_params.pp()
    fit_params = res['xmin']

    plt.plot(xdata_um, ydata_bf, '-', 
             xdata_um, ss.box_fit_func(xdata, fit_params), '-')
    if SHOW_PLOTS:
        plt.title('Bhop Fit Plot')
        plt.tight_layout()
        plt.show()
