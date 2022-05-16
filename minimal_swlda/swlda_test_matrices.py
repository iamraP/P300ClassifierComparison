

import py3gui as p3
import pickle
import numpy as np
import pandas as pd


with open(r"C:\Users\map92fg\Documents\Software\P300_Classification\data_thesis\Classsifier_Results\swlda_weights.pickle", "rb") as file:
    python_swlda = np.array(pickle.load(file))




sess = 2
#data = load_data(r"D:\dat_files\all_files\tacFreeS001R01.dat",[0,800])
classifier = p3.load_weights(r"data/test_weights_bci2000_sess" + str(sess)+".prm")

bci2k = classifier[0::24].flatten('F')
python = python_swlda[sess-1]
python_restored = python.reshape(classifier[0::24].shape)
bci2k_nonflat = classifier[0::24]

only_bci2k = np.array((abs(bci2k - python) == bci2k) & (bci2k != 0))
only_python = np.array((abs(bci2k - python) == python) & (python != 0))
both = np.array((bci2k != 0) & (python != 0))

columns=["bci2k","python","difference","only_bci2k","only_python","both"]
df = pd.DataFrame(np.array([bci2k, python, abs(bci2k - python), only_bci2k, only_python, both]).T,columns=["bci2k","python","difference","only_bci2k","only_python","both"])


commpare_sparse = compare[compare.sum(axis=1) != 0]
test = np.c_[commpare_sparse,np.array((commpare_sparse[:,2]==commpare_sparse[:,0]) |( commpare_sparse[:,2]==commpare_sparse[:,1]))]

data = p3.load_data(r"data/tacFreeS001R01.dat", [0, classifier.shape[0]],None, True) #TODO: load multiple files
result_swlda = p3.test_weights(data[0], data[1], classifier,[1,6], 8)
print("Accuracy: %s "%(result_swlda[1]))