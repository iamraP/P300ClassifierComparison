import matplotlib.pyplot as plt

import py3gui as p3
import pickle
import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt



with open(r"data\swlda_weights.pickle", "rb") as file:
    python_swlda = np.array(pickle.load(file))

with open(r"data\swlda_weights_resampled_by_py3gui_dec_freq26_later.pickle", "rb") as file:
    py3gui = np.array(pickle.load(file))
with open(r"data\swlda_channels_resampled_by_py3gui_dec_freq26_later.pickle", "rb") as file:
    channels = np.array(pickle.load(file))

excluded_sessions = np.array([4,7,10,11,12,16,18,19,20,32])-1
sess_list = np.delete(range(1,50),excluded_sessions)

for i,sess in enumerate(sess_list):

    #data = load_data(r"D:\dat_files\all_files\tacFreeS001R01.dat",[0,800])
    classifier,channels_bci = p3.load_weights(r"data/P3Classifier_sess" + str(sess)+".prm")
    channels_bci = [int(i)-1 for i in channels_bci]
    print("Session {} : {}".format(sess,classifier.shape))
    print("Py3 channels: {}".format((channels[i])))
    print("BCI channels: {}".format((channels_bci)))

    if classifier.shape != (390,12):
        classifier_full = np.zeros((390 ,12)) # create empty matrix
        classifier = np.append(classifier,np.zeros((390 -classifier.shape[0],classifier.shape[1])),axis=0) # stretch matrix to full size, size was predetermined by latest weighted sample so it should be fine to fill up with zeros
        classifier_full[:, channels_bci] = classifier[:,classifier.sum(axis=0) != 0]  # take only the channels which have weights to build classifier matrix
        classifier = classifier_full


    weights = py3gui[i]
    full_weight_matrix = np.zeros((classifier.shape))
    try:
        full_weight_matrix[weights[:, 1].astype(int)-(390-classifier.shape[0]+1),channels[i][weights[:, 0].astype(int) - 1]] = weights[:, 3]
    except IndexError:
        full_weight_matrix[weights[:, 1].astype(int) - (390-classifier.shape[0]+1), weights[:, 0].astype(int) - 1] = weights[:, 3]
    test = classifier - full_weight_matrix
    test1 = test[abs(test) > 1]
    print("Numbers of features:\nBCI2000: {} \nPy3GUI: {}\ndiffered >1: {} ".format(len(np.unique(classifier))-1,len(np.unique(full_weight_matrix))-1,len(test1)/26))
    print("___________________")

    fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(20,10))
    seaborn.heatmap(abs(classifier),ax=ax1,cmap='Reds')
    ax1.set_title("BCI2000")
    seaborn.heatmap(abs(full_weight_matrix),ax=ax2,cmap='Blues')
    ax2.set_title("Py3GUI")
    seaborn.heatmap(abs(classifier),ax=ax3,alpha =0.5,cmap='Reds')
    seaborn.heatmap(abs(full_weight_matrix),ax=ax3, alpha=0.5,cmap='Blues')
    ax3.set_title("Overlay")
    fig.suptitle("Session"+str(sess))
    plt.savefig(r"D:\Google Drive\Master\Masterarbeit\Graphics\compare_matrices\matrix_new"+str(sess))


bci2k = classifier[0::24].flatten('F')
python = python_swlda[i]
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

sess40 = np.array([1,261,1,10,1,262,1,10,1,263,1,10,1,264,1,10,1,265,1,10,1,266,1,10,1,267,1,10,1,268,1,10,1,269,1,10,1,270,1,10,1,271,1,10,1,272,1,10,1,273,1,10,1,274,1,10,1,275,1,10,1,276,1,10,1,277,1,10,1,278,1,10,1,279,1,10,1,280,1,10,1,281,1,10,1,282,1,10,1,283,1,10,1,284,1,10,1,285,1,10,1,286,1,10,2,365,1,8.23716,2,366,1,8.23716,2,367,1,8.23716,2,368,1,8.23716,2,369,1,8.23716,2,370,1,8.23716,2,371,1,8.23716,2,372,1,8.23716,2,373,1,8.23716,2,374,1,8.23716,2,375,1,8.23716,2,376,1,8.23716,2,377,1,8.23716,2,378,1,8.23716,2,379,1,8.23716,2,380,1,8.23716,2,381,1,8.23716,2,382,1,8.23716,2,383,1,8.23716,2,384,1,8.23716,2,385,1,8.23716,2,386,1,8.23716,2,387,1,8.23716,2,388,1,8.23716,2,389,1,8.23716,2,390,1,8.23716])
sess40 = sess40.reshape(-1,4)
#session 40, 2 features - 26 samples  channel 1 und 2

sess1= np.array([1,235,1,-8.48605,1,236,1,-8.48605,1,237,1,-8.48605,1,238,1,-8.48605,1,239,1,-8.48605,1,240,1,-8.48605,1,241,1,-8.48605,1,242,1,-8.48605,1,243,1,-8.48605,1,244,1,-8.48605,1,245,1,-8.48605,1,246,1,-8.48605,1,247,1,-8.48605,1,248,1,-8.48605,1,249,1,-8.48605,1,250,1,-8.48605,1,251,1,-8.48605,1,252,1,-8.48605,1,253,1,-8.48605,1,254,1,-8.48605,1,255,1,-8.48605,1,256,1,-8.48605,1,257,1,-8.48605,1,258,1,-8.48605,1,259,1,-8.48605,1,260,1,-8.48605,1,365,1,3.09092,1,366,1,3.09092,1,367,1,3.09092,1,368,1,3.09092,1,369,1,3.09092,1,370,1,3.09092,1,371,1,3.09092,1,372,1,3.09092,1,373,1,3.09092,1,374,1,3.09092,1,375,1,3.09092,1,376,1,3.09092,1,377,1,3.09092,1,378,1,3.09092,1,379,1,3.09092,1,380,1,3.09092,1,381,1,3.09092,1,382,1,3.09092,1,383,1,3.09092,1,384,1,3.09092,1,385,1,3.09092,1,386,1,3.09092,1,387,1,3.09092,1,388,1,3.09092,1,389,1,3.09092,1,390,1,3.09092,2,79,1,1.70965,2,80,1,1.70965,2,81,1,1.70965,2,82,1,1.70965,2,83,1,1.70965,2,84,1,1.70965,2,85,1,1.70965,2,86,1,1.70965,2,87,1,1.70965,2,88,1,1.70965,2,89,1,1.70965,2,90,1,1.70965,2,91,1,1.70965,2,92,1,1.70965,2,93,1,1.70965,2,94,1,1.70965,2,95,1,1.70965,2,96,1,1.70965,2,97,1,1.70965,2,98,1,1.70965,2,99,1,1.70965,2,100,1,1.70965,2,101,1,1.70965,2,102,1,1.70965,2,103,1,1.70965,2,104,1,1.70965,2,105,1,-3.27471,2,106,1,-3.27471,2,107,1,-3.27471,2,108,1,-3.27471,2,109,1,-3.27471,2,110,1,-3.27471,2,111,1,-3.27471,2,112,1,-3.27471,2,113,1,-3.27471,2,114,1,-3.27471,2,115,1,-3.27471,2,116,1,-3.27471,2,117,1,-3.27471,2,118,1,-3.27471,2,119,1,-3.27471,2,120,1,-3.27471,2,121,1,-3.27471,2,122,1,-3.27471,2,123,1,-3.27471,2,124,1,-3.27471,2,125,1,-3.27471,2,126,1,-3.27471,2,127,1,-3.27471,2,128,1,-3.27471,2,129,1,-3.27471,2,130,1,-3.27471,2,209,1,-5.89723,2,210,1,-5.89723,2,211,1,-5.89723,2,212,1,-5.89723,2,213,1,-5.89723,2,214,1,-5.89723,2,215,1,-5.89723,2,216,1,-5.89723,2,217,1,-5.89723,2,218,1,-5.89723,2,219,1,-5.89723,2,220,1,-5.89723,2,221,1,-5.89723,2,222,1,-5.89723,2,223,1,-5.89723,2,224,1,-5.89723,2,225,1,-5.89723,2,226,1,-5.89723,2,227,1,-5.89723,2,228,1,-5.89723,2,229,1,-5.89723,2,230,1,-5.89723,2,231,1,-5.89723,2,232,1,-5.89723,2,233,1,-5.89723,2,234,1,-5.89723,2,235,1,9.72923,2,236,1,9.72923,2,237,1,9.72923,2,238,1,9.72923,2,239,1,9.72923,2,240,1,9.72923,2,241,1,9.72923,2,242,1,9.72923,2,243,1,9.72923,2,244,1,9.72923,2,245,1,9.72923,2,246,1,9.72923,2,247,1,9.72923,2,248,1,9.72923,2,249,1,9.72923,2,250,1,9.72923,2,251,1,9.72923,2,252,1,9.72923,2,253,1,9.72923,2,254,1,9.72923,2,255,1,9.72923,2,256,1,9.72923,2,257,1,9.72923,2,258,1,9.72923,2,259,1,9.72923,2,260,1,9.72923,3,235,1,6.51668,3,236,1,6.51668,3,237,1,6.51668,3,238,1,6.51668,3,239,1,6.51668,3,240,1,6.51668,3,241,1,6.51668,3,242,1,6.51668,3,243,1,6.51668,3,244,1,6.51668,3,245,1,6.51668,3,246,1,6.51668,3,247,1,6.51668,3,248,1,6.51668,3,249,1,6.51668,3,250,1,6.51668,3,251,1,6.51668,3,252,1,6.51668,3,253,1,6.51668,3,254,1,6.51668,3,255,1,6.51668,3,256,1,6.51668,3,257,1,6.51668,3,258,1,6.51668,3,259,1,6.51668,3,260,1,6.51668,4,183,1,-5.53082,4,184,1,-5.53082,4,185,1,-5.53082,4,186,1,-5.53082,4,187,1,-5.53082,4,188,1,-5.53082,4,189,1,-5.53082,4,190,1,-5.53082,4,191,1,-5.53082,4,192,1,-5.53082,4,193,1,-5.53082,4,194,1,-5.53082,4,195,1,-5.53082,4,196,1,-5.53082,4,197,1,-5.53082,4,198,1,-5.53082,4,199,1,-5.53082,4,200,1,-5.53082,4,201,1,-5.53082,4,202,1,-5.53082,4,203,1,-5.53082,4,204,1,-5.53082,4,205,1,-5.53082,4,206,1,-5.53082,4,207,1,-5.53082,4,208,1,-5.53082,4,209,1,9.68747,4,210,1,9.68747,4,211,1,9.68747,4,212,1,9.68747,4,213,1,9.68747,4,214,1,9.68747,4,215,1,9.68747,4,216,1,9.68747,4,217,1,9.68747,4,218,1,9.68747,4,219,1,9.68747,4,220,1,9.68747,4,221,1,9.68747,4,222,1,9.68747,4,223,1,9.68747,4,224,1,9.68747,4,225,1,9.68747,4,226,1,9.68747,4,227,1,9.68747,4,228,1,9.68747,4,229,1,9.68747,4,230,1,9.68747,4,231,1,9.68747,4,232,1,9.68747,4,233,1,9.68747,4,234,1,9.68747,4,313,1,2.35331,4,314,1,2.35331,4,315,1,2.35331,4,316,1,2.35331,4,317,1,2.35331,4,318,1,2.35331,4,319,1,2.35331,4,320,1,2.35331,4,321,1,2.35331,4,322,1,2.35331,4,323,1,2.35331,4,324,1,2.35331,4,325,1,2.35331,4,326,1,2.35331,4,327,1,2.35331,4,328,1,2.35331,4,329,1,2.35331,4,330,1,2.35331,4,331,1,2.35331,4,332,1,2.35331,4,333,1,2.35331,4,334,1,2.35331,4,335,1,2.35331,4,336,1,2.35331,4,337,1,2.35331,4,338,1,2.35331,4,365,1,-5.07327,4,366,1,-5.07327,4,367,1,-5.07327,4,368,1,-5.07327,4,369,1,-5.07327,4,370,1,-5.07327,4,371,1,-5.07327,4,372,1,-5.07327,4,373,1,-5.07327,4,374,1,-5.07327,4,375,1,-5.07327,4,376,1,-5.07327,4,377,1,-5.07327,4,378,1,-5.07327,4,379,1,-5.07327,4,380,1,-5.07327,4,381,1,-5.07327,4,382,1,-5.07327,4,383,1,-5.07327,4,384,1,-5.07327,4,385,1,-5.07327,4,386,1,-5.07327,4,387,1,-5.07327,4,388,1,-5.07327,4,389,1,-5.07327,4,390,1,-5.07327,5,365,1,-2.7244,5,366,1,-2.7244,5,367,1,-2.7244,5,368,1,-2.7244,5,369,1,-2.7244,5,370,1,-2.7244,5,371,1,-2.7244,5,372,1,-2.7244,5,373,1,-2.7244,5,374,1,-2.7244,5,375,1,-2.7244,5,376,1,-2.7244,5,377,1,-2.7244,5,378,1,-2.7244,5,379,1,-2.7244,5,380,1,-2.7244,5,381,1,-2.7244,5,382,1,-2.7244,5,383,1,-2.7244,5,384,1,-2.7244,5,385,1,-2.7244,5,386,1,-2.7244,5,387,1,-2.7244,5,388,1,-2.7244,5,389,1,-2.7244,5,390,1,-2.7244,6,183,1,10,6,184,1,10,6,185,1,10,6,186,1,10,6,187,1,10,6,188,1,10,6,189,1,10,6,190,1,10,6,191,1,10,6,192,1,10,6,193,1,10,6,194,1,10,6,195,1,10,6,196,1,10,6,197,1,10,6,198,1,10,6,199,1,10,6,200,1,10,6,201,1,10,6,202,1,10,6,203,1,10,6,204,1,10,6,205,1,10,6,206,1,10,6,207,1,10,6,208,1,10,6,235,1,-8.46372,6,236,1,-8.46372,6,237,1,-8.46372,6,238,1,-8.46372,6,239,1,-8.46372,6,240,1,-8.46372,6,241,1,-8.46372,6,242,1,-8.46372,6,243,1,-8.46372,6,244,1,-8.46372,6,245,1,-8.46372,6,246,1,-8.46372,6,247,1,-8.46372,6,248,1,-8.46372,6,249,1,-8.46372,6,250,1,-8.46372,6,251,1,-8.46372,6,252,1,-8.46372,6,253,1,-8.46372,6,254,1,-8.46372,6,255,1,-8.46372,6,256,1,-8.46372,6,257,1,-8.46372,6,258,1,-8.46372,6,259,1,-8.46372,6,260,1,-8.46372,7,209,1,-3.86964,7,210,1,-3.86964,7,211,1,-3.86964,7,212,1,-3.86964,7,213,1,-3.86964,7,214,1,-3.86964,7,215,1,-3.86964,7,216,1,-3.86964,7,217,1,-3.86964,7,218,1,-3.86964,7,219,1,-3.86964,7,220,1,-3.86964,7,221,1,-3.86964,7,222,1,-3.86964,7,223,1,-3.86964,7,224,1,-3.86964,7,225,1,-3.86964,7,226,1,-3.86964,7,227,1,-3.86964,7,228,1,-3.86964,7,229,1,-3.86964,7,230,1,-3.86964,7,231,1,-3.86964,7,232,1,-3.86964,7,233,1,-3.86964,7,234,1,-3.86964,7,365,1,2.46892,7,366,1,2.46892,7,367,1,2.46892,7,368,1,2.46892,7,369,1,2.46892,7,370,1,2.46892,7,371,1,2.46892,7,372,1,2.46892,7,373,1,2.46892,7,374,1,2.46892,7,375,1,2.46892,7,376,1,2.46892,7,377,1,2.46892,7,378,1,2.46892,7,379,1,2.46892,7,380,1,2.46892,7,381,1,2.46892,7,382,1,2.46892,7,383,1,2.46892,7,384,1,2.46892,7,385,1,2.46892,7,386,1,2.46892,7,387,1,2.46892,7,388,1,2.46892,7,389,1,2.46892,7,390,1,2.46892,8,157,1,0.0020091,8,158,1,0.0020091,8,159,1,0.0020091,8,160,1,0.0020091,8,161,1,0.0020091,8,162,1,0.0020091,8,163,1,0.0020091,8,164,1,0.0020091,8,165,1,0.0020091,8,166,1,0.0020091,8,167,1,0.0020091,8,168,1,0.0020091,8,169,1,0.0020091,8,170,1,0.0020091,8,171,1,0.0020091,8,172,1,0.0020091,8,173,1,0.0020091,8,174,1,0.0020091,8,175,1,0.0020091,8,176,1,0.0020091,8,177,1,0.0020091,8,178,1,0.0020091,8,179,1,0.0020091,8,180,1,0.0020091,8,181,1,0.0020091,8,182,1,0.0020091])
sess1= sess1.reshape(-1,4)
np.unique(sess1[:,0])
# Session 1 , 17 feature, 26 samples, channel: 1., 2., 3., 4., 5., 6., 7., 8.])
