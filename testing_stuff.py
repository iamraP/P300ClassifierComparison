#!/usr/bin/python

import numpy as np
import sys
sys.path.append('..')

from BCpy2000.BCI2000Tools.FileReader import bcistream, ParseParam
from BCpy2000.BCI2000Tools.DataFiles import load

__all__ = ['load_data', 'load_weights']

SUPPORTED = ['standard', 'pickle']
from scipy import integrate
from scipy.special import erf

def convolve(f, g, t, epsabs = 1e-8, epsrel = 1e-8):
    return integrate.quad(lambda tau: f(tau) * g(t - tau), -np.Inf, np.Inf,
        epsabs = epsabs, epsrel = epsrel)[0]

def max_gauss_pdf(mu, sigma, n, x):
    x = x - mu
    var = sigma ** 2.
    sqrt2 = np.sqrt(2.)
    return (sqrt2 / (2. ** n)) * np.exp(-(x ** 2.) / (2. * var)) / \
        (np.sqrt(np.pi) * sigma) * n * \
        (1. + erf(x / (sqrt2 * sigma))) ** (n - 1.)

def max_gauss_cdf(mu, sigma, n, x):
    x = x - mu
    return ((1 + erf(x / (np.sqrt(2.) * sigma))) / 2.) ** n



def accuracy(nt_mean, nt_var, nt_count, t_mean, t_var, repetitions):
    nt_mean *= repetitions
    nt_var *= repetitions
    nt_std = np.sqrt(nt_var)
    t_mean *= repetitions
    t_var *= repetitions
    t_std = np.sqrt(t_var)
    return convolve(
        lambda t: max_gauss_cdf(nt_mean - t_mean, nt_std, nt_count, t),
        lambda t: max_gauss_pdf(0, t_std, 1, t),
        0
    )

def test_weights(responses, type, classifier, matrixshape, repetitions):
    responses = np.asarray(responses, dtype = float) \
        [:, :classifier.shape[0], :classifier.shape[1]]
        # If the weights do not include the last channel or two (or more),
        # then the dense classifier matrix created will not be the right
        # dimensions. Since this only occurs for the last channels, this
        # can be corrected by throwing out the channels that are not in
        # the classification matrix, as done by [..., :classifier.shape[1]]
    type = np.asarray(type, dtype = bool)
    if responses.shape[1] != classifier.shape[0]:
        return 'Response window not long enough.'
    target = type.nonzero()[0]
    nontarget = (~type).nonzero()[0]
    target_scores = (responses[target] * classifier). \
        sum(axis = 1).sum(axis = 1)
    target_mean = target_scores.mean()
    target_var = target_scores.var(ddof = 1)
    nontarget_scores = (responses[nontarget] * classifier). \
        sum(axis = 1).sum(axis = 1)
    nontarget_mean = nontarget_scores.mean()
    nontarget_var = nontarget_scores.var(ddof = 1)
    if target_mean <= nontarget_mean:
        return 'These weights are so bad that they actually ' + \
            'classify *against* the target.'
    if np.isscalar(matrixshape):
        matrixshape = [matrixshape]
    if np.isscalar(repetitions):
        repetitions = [repetitions]
    correctness = []
    for reps in repetitions:
        total_accuracy = 1
        for dimension in matrixshape:
            total_accuracy *= \
                accuracy(
                    nontarget_mean,
                    nontarget_var,
                    dimension - 1,
                    target_mean,
                    target_var,
                    reps
                )
        correctness.append(total_accuracy)
    c_mean = target_mean - nontarget_mean
    target_std = np.sqrt(target_var) / c_mean
    nontarget_std = np.sqrt(nontarget_var) / c_mean
    return (target_std, nontarget_std), correctness


def removeAnomalies(data, type, cutoff_std = 6):
    target = data[type]
    nontarget = data[~type]
    bad_target_indices = \
        np.unique(
            (
                abs(target - target.mean(axis = 0)) > \
                    cutoff_std * target.std(axis = 0)
            ).nonzero()[0]
        )
    good_target_indices = np.ones(target.shape[0], dtype = bool)
    good_target_indices[bad_target_indices] = False
    bad_nontarget_indices = \
        np.unique(
            (
                abs(nontarget - nontarget.mean(axis = 0)) > \
                    cutoff_std * nontarget.std(axis = 0)
            ).nonzero()[0]
        )
    good_nontarget_indices = np.ones(nontarget.shape[0], dtype = bool)
    good_nontarget_indices[bad_nontarget_indices] = False
    data = np.concatenate(
        (target[good_target_indices],
        nontarget[good_nontarget_indices])
    )
    type = np.zeros(data.shape[0], dtype = bool)
    type[:good_target_indices.size] = True
    return data, type

def load_weights(fname):
    f = open(fname, 'rb')
    prefices = [
        'Filtering:LinearClassifier',
        'Filtering:SpatialFilter',
        'Source:Online%20Processing',
    ]
    params = {
        'Classifier': None,
        'SpatialFilter': None,
        'TransmitChList': None,
    }
    for line in f:
        # if '\0' in line:
        #     break
        for prefix in prefices:
            if line.startswith(bytes(prefix, encoding= 'utf-8')):
                info = ParseParam(r"{}".format(line))
                if info['name'] in params:
                    params[info['name']] = info['val']


    samples = (np.array(params['Classifier'])[:, 1]).astype(int) - 1
    channels = np.array(params['TransmitChList'])[(np.array(params['Classifier'])[:, 0]).astype(int) - 1].astype(int)-1
    classifier = np.zeros((samples.max() + 1, channels.max() + 1))
    classifier[samples, channels] = np.array(params['Classifier'])[:,3]

#modified version, doesn't check for the errors
    return classifier



def get_state_changes(state_array, to_value = None, from_value = None):
    flattened = state_array.ravel()
    candidates = (flattened[1:] != flattened[:-1]).nonzero()[0] + 1
    if to_value != None:
        mask = (flattened[candidates] == to_value).nonzero()[0]
        candidates = candidates[mask]
    if from_value != None:
        mask = (flattened[candidates - 1] == from_value).nonzero()[0]
        candidates = candidates[mask]
    return candidates

def load_standard_data(fname, window, window_in_samples):
    dat = bcistream(fname)
    signal, states = dat.decode('all')
    samplingrate = dat.samplingrate()
    if window_in_samples:
        window = np.arange(int(window[1]))
    else:
        window = np.arange(int(np.round(window[1] * samplingrate / 1000)))
    signal = np.asarray(signal).transpose()
    if 'Flashing' in states:
        stimulusbegin = get_state_changes(states['Flashing'], from_value = 0)
    elif 'StimulusBegin' in states:
        stimulusbegin = get_state_changes(states['StimulusBegin'],
            from_value = 0)
    elif 'StimulusCode' in states:
        stimulusbegin = get_state_changes(states['StimulusCode'],
            from_value = 0)
    elif 'Epoch' in states:
        stimulusbegin = get_state_changes(states['Epoch'])
        stimulusbegin[(states['Epoch'][:, stimulusbegin] == 0).ravel()] = 0
    else:
        return 'Data file does not seem to have a record of stimulus times.'
    if 'StimulusType' in states:
        type = states['StimulusType'].ravel()[stimulusbegin] > 0
    elif 'TargetBitValue' in states:
        type = states['TargetBitValue'].ravel()[stimulusbegin] == 1
    else:
        return 'Data file does not seem to have a record of stimulus type.'
    if 'EventOffset' in states:
        #signal -= signal.mean(axis = 0)
        zero = 1 << (dat.statedefs['EventOffset']['length'] - 1)
        offsets = states['EventOffset'].ravel()[stimulusbegin] - zero
        stimulusbegin -= offsets
    data = np.zeros((stimulusbegin.size, window.size, signal.shape[1]))
    for i in range(stimulusbegin.size):
        index = stimulusbegin[i] - 1
        data[i] = signal[window + index, :]
    return data, type, samplingrate

def load_pickle_data(fname, window, window_in_samples):
    pickle = load(fname)
    samplingrate = int(pickle['fs'])
    if window_in_samples:
        window = int(window[1])
    else:
        window = int(np.round(window[1] * samplingrate / 1000))
    type = pickle['y'] > 0
    data = np.swapaxes(pickle['x'], 1, 2)[:, :window, :]
    if data.shape[1] != window:
        largest_window = int((data.shape[1] + 1) * 1000 // samplingrate) + 1
        while np.round(largest_window * samplingrate / 1000) > data.shape[1]:
            largest_window -= 1
        return 'Not enough data to fill window. Window is too big.\n' + \
            'Maximum window size would be [0 %i].' % largest_window
    return data, type, samplingrate

def load_data(fname, window, ftype = None, window_in_samples = False,
    removeanomalies = False):
    #reload(__import__('testweights')) #TODO!!!
    if ftype == None:
        if fname.lower().endswith('.dat'):
            ftype = 'standard'
        elif fname.lower().endswith('.pk'):
            ftype = 'pickle'
    if ftype == 'standard':
        loader = load_standard_data
    elif ftype == 'pickle':
        loader = load_pickle_data
    else:
        return '%s file type not supported.' % str(ftype)
    try:
        result = loader(fname, window, window_in_samples)
        if isinstance(result, str):
            return result
        data, type, samplingrate = result
        if removeanomalies:
            data, type = removeAnomalies(data, type)
        return data, type, samplingrate
    except SyntaxError:
        return 'Data could not be loaded. Wrong file type selected?'

data_paths =[r"C:\Users\map92fg\Documents\Software\P300_Classification\data_thesis\dat_files\tacCalibS001R01.dat",r"C:\Users\map92fg\Documents\Software\P300_Classification\data_thesis\dat_files\tacCalibS001R02.dat",r"C:\Users\map92fg\Documents\Software\P300_Classification\data_thesis\dat_files\tacCalibS001R03.dat"]

#data = load_data(r"D:\dat_files\all_files\tacFreeS001R01.dat",[0,800])
classifier = load_weights(r"D:\test_weights_bci2000.prm")

result = load_data(r"C:\Users\map92fg\Documents\Software\P300_Classification\data_thesis\dat_files\tacFreeS001R01.dat", [0, classifier.shape[0]],None, True)
result1 = test_weights(result[0], result[1], classifier,[6,6], 8)

result = testweights.test_weights(data, type, classifier, matrixshape, repetitions)