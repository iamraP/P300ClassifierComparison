import py3gui as p3

#data = load_data(r"D:\dat_files\all_files\tacFreeS001R01.dat",[0,800])
classifier = p3.load_weights(r"data/test_weights_bci2000_sess1.prm")

data = p3.load_data(r"data/tacFreeS001R01.dat", [0, classifier.shape[0]],None, True) #TODO: load multiple files
result_swlda = p3.test_weights(data[0], data[1], classifier,[1,6], 8)
print("Accuracy: %s "%(result_swlda[1]))