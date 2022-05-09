#from __future__ import print_function
import data_prep as dp
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif,SelectKBest
from pyriemann.classification import FgMDM,MDM
from sklearn import metrics
import time
import swlda as sw
import numpy as np
from scipy.spatial import distance
from scipy import stats
import matplotlib.pyplot as plt
import py3gui
from sklearn import preprocessing
import swlda_class as sc

'''This script was used for testing the classifiers, usining the '''
################################################
'''SETTINGS '''
################################################
excluded_sessions = [4,7,10,11,12,16,18,19,20,32] #some sessions were training session which followed after the standard protocol (they happend on the same day and are excluded from analysis)
sessionwise_calibration = True #False for transfer classifiers
trans_calib_sess = [1,2,3] # only relevant when  sessionwise_calibration == False
calib_runs=3 # how many of the runs should be used for calibration? (max=3)
electrode_list = "all" # which electrodes to include, options: 'all' or ["Fz","Fc1","Fc2","C3","Cz","C4","Pz"]== set of unchanged electrodes
classification = True #if only analysis is done don't load data in the complex way it's needed for classification
investigate_sessionwise = False  #used for evaluation, figure plotting etc
investigate_several_sessions = False #set range in script, used for averaging of several sessions
sess_list = range(1,14)  #range(1,50) #list(range(1,50)) #neues Setup ab Session 21 (12. Messtag)
data_origin = r"D:\Google Drive\Master\Masterarbeit\Data\mne_raw_pickle"
################################################
'''SETTINGS STOP'''
################################################

################################################
'''Investigation'''
################################################
if investigate_sessionwise:
    for sess in sess_list:
        data_path = []
        if sess in excluded_sessions:
            continue
        data_path_calib = data_origin + r"\Calib_S" + str(sess).zfill(3) + ".pickle"
        data_path_free = data_origin + r"\Free_S" + str(sess).zfill(3) + ".pickle"
        data_path.append(data_path_free)
        data_path.append(data_path_calib)
        dp.investigate(data_path, sess_name=sess)
if investigate_several_sessions:
    data_path = []
    for sess in sess_list:
        if sess in excluded_sessions:
            continue
        data_path.append(data_origin + r"\Calib_S" + str(sess).zfill(3) + ".pickle")
        data_path.append(data_origin + r"\Free_S" + str(sess).zfill(3) + ".pickle")
    dp.investigate(data_path, sess_name='all')

################################################
'''Classification'''
################################################

classifier_results = pd.DataFrame(columns=["Accuracy","Classifier","Session","Ep2Avg"])
dates = []

roc_values = {"LDA":[],
              "LDA60":[],
              "shrinkLDA":[],
              "LDA_xDawn":[],
              "LDA60_xDawn":[],
              "shrinkLDA_xDawn":[],
              "MDM":[],
              'FGMDM':[],
              'MDM_res':[],
              'FGMDM_res':[],
              'MDM_xDawn':[],
              'FGMDM_xDawn':[],
              'MDM_res_xDawn':[],
              'FGMDM_res_xDawn':[]}
first_session = True

for resampled_riemann in [False]: #TODO rechange to [True,False]
    for xDawn in [False]:
        mdm_name = "MDM"
        fgmdm_name = "FGMDM"
        lda_name = "LDA"
        swlda_name = "SWLDA"
        #lda60_name = "LDA60"
        shrinklda_name = "shrinkLDA"

        if resampled_riemann:
            mdm_name = mdm_name + "_res"
            fgmdm_name = fgmdm_name + "_res"
        if xDawn:
            mdm_name = mdm_name + "_xDawn"
            fgmdm_name = fgmdm_name + "_xDawn"
            lda_name = lda_name + "_xDawn"
            swlda_name = swlda_name + "_xDawn"
            #lda60_name =lda60_name +"_xDawn"
            shrinklda_name = shrinklda_name +"_xDawn"

        for sess in sess_list:
            data_path = []
            if sess in excluded_sessions:
                continue

            if not sessionwise_calibration and sess==21: # recalibrate the classifier after the switch of the electrode set
                first_session =True

            print("Working on sesssion {}".format(str(sess).zfill(3)))
            # dates.append(SESS_DATES[sess])
            ################################################
            '''Load Data to Classify and Investigate '''
            ################################################
            #load data
            if classification:  #load data for classification only if classification is applied
                if not sessionwise_calibration:
                    data_path_calib = []
                    for calib_sess in trans_calib_sess:
                        data_path_calib.append(data_origin + r"\Calib_S"+str(calib_sess).zfill(3)+".pickle")
                else:
                    data_path_calib = data_origin + r"\Calib_S"+str(sess).zfill(3)+".pickle"
                if sessionwise_calibration or first_session:
                    X_train, y_train, epochs_train, epoch_info_train,cov_estimator, cov_matrices_train, spatial_filter = dp.load_epochs(data_path_calib,
                                                                                                                picks=electrode_list,
                                                                                                                resampled_riemann = resampled_riemann,
                                                                                                                xDawn = xDawn,
                                                                                                                calib_runs=calib_runs)
                data_path_free = data_origin + r"\Free_S"+str(sess).zfill(3)+".pickle"
                X_test, y_test,epochs_test, epoch_info_test,cov_estimator, cov_matrices_test,spatial_filter =  dp.load_epochs(data_path_free,
                                                                                                        cov_estimator=cov_estimator,
                                                                                                        picks=electrode_list,
                                                                                                        resampled_riemann=resampled_riemann,
                                                                                                        xDawn=xDawn,
                                                                                                        spatial_filter=spatial_filter)




            ################################################
            '''Initiate, Train and Evlauate Classifiers'''
            ################################################


            #data_train_res classifiers and evaluate classifiers
            if not resampled_riemann:

            #SWLDA
                swlda = sc.swlda()
                swlda.fit(X_train,y_train)
                swlda_y_pred_prob = swlda.predict_proba(X_test)
                ac_swlda = dp.evaluate_independent_epochs(swlda_y_pred_prob, epoch_info_test)

            # get feature weights
                data_train = epochs_train.copy().get_data()*1e6
                data_test = epochs_test.copy().get_data() * 1e6

                swlda = swlda_class.swlda()

                #take resampled data from mne, calculate feature weights and finally calculate classification results without upsampling
                channels, weights, weights_resampled, downsampled = sw.swlda(data_train.swapaxes(1, 2), y_train, 512,[0,800],20,  mne_res=X_train)

                # get feature weights from raw data
                #channels, weights, weights_resampled,downsampled = sw.swlda(data_train.swapaxes(1,2),y_train,512,[0,800],20,mne_res = None)

                #get resampled responses
                data_train_res = epochs_train.copy().resample(20).get_data()*1e6
                data_test_res = epochs_test.copy().resample(20).get_data()*1e6

                #evaluate using the py3GUI algorithm with the resampled weights and data
                #create full matrix from sparse matrix
                weights_resampled_full = np.zeros((data_test_res.shape[2], data_test_res.shape[1]))
                weights_resampled_full[weights_resampled[:, 1].astype(int), weights_resampled[:, 0].astype(int)-1] = weights_resampled[:,3] # channels are 1-based therefore -1!

                ac_swlda = py3gui.test_weights(data_test_res.swapaxes(1, 2), y_test, weights_resampled_full, [1, 6], 8)

                #evaluate using the py3GUI and reversing the resampling (as in the original PY3GUI script)
                #create full matrix from sparse matrix
                full_weight_matrix = np.zeros((data_test.shape[2], data_test.shape[1]))
                full_weight_matrix[weights[:, 1].astype(int), weights[:, 0].astype(int)-1] = weights[:,3]

                ac_swlda = py3gui.test_weights(data_test.swapaxes(1, 2), y_test, full_weight_matrix, [1, 6], 8)


                # extract weighted features
                X_swTrain = np.zeros((weights_resampled.shape[0], data_train_res.shape[0]))
                for i, feature in enumerate(weights_resampled):
                    X_swTrain[i, :] = data_train_res[:, feature[0].astype(int), feature[1].astype(int)] * feature[3]
                X_swTrain = X_swTrain.swapaxes(0, 1)

                X_swTest= np.zeros((weights_resampled.shape[0], data_test_res.shape[0]))
                for i, feature in enumerate(weights_resampled):
                    X_swTest[i, :] = data_test_res[:, feature[0].astype(int), feature[1].astype(int)] * feature[3]
                X_swTest=X_swTest.swapaxes(0,1)

                # sum weighted features:
                # so far the "best" results if the features are not weighted?! - accuracy declines, the more epochs are averaged - this seems terribly wrong
                swlda_result = X_swTest.sum(axis=1)
                swlda_y_pred_prob =np.array([swlda_result,swlda_result*-1]).swapaxes(0,1)
                ac_swlda =  dp.evaluate_independent_epochs(swlda_y_pred_prob, epoch_info_test)

                #sum weighted features, get targets
                swlda_result = X_swTest.sum(axis=1)>0
                swlda_y_pred_prob = np.array([X_swTest.sum(axis=1)>0, X_swTest.sum(axis=1)<0]).swapaxes(0, 1)*1
                ac_swlda = dp.evaluate_independent_epochs(swlda_y_pred_prob, epoch_info_test)


                # train target/non targets mean + std
                tar_train = X_swTrain[y_train == 1]
                nontar_train = X_swTrain[y_train == 0]

                # PDF for Target/non targets

                P_X_tar = stats.norm.pdf(X_swTest, loc=tar_train.mean(),
                                         scale=tar_train.std())  # probability to have feature, given target
                P_X_ntar = stats.norm.pdf(X_swTest, loc=nontar_train.mean(),
                                          scale=nontar_train.std())  # probability to have feature, given nontarget

                #Bayes for prediction
                P_tar_X = (P_X_tar * (1 / 2)) / (P_X_tar * (1 / 2) + P_X_ntar * (1 / 2))
                swlda_y_pred_prob = np.array([1-P_tar_X,P_tar_X]).swapaxes(0, 1).sum(axis=2)/17
                ac_swlda = dp.evaluate_independent_epochs(swlda_y_pred_prob, epoch_info_test)
                print(ac_swlda)



            #calculate distance (tried this with whitend data, didn't work) / maybe try malhanobis

                #Whiten Data?
                scaler = preprocessing.Normalizer()
                scaler = scaler.fit(X_swTrain,y_train)

                X_swTrain = scaler.transform(X_swTrain)
                X_swTest = scaler.transform(X_swTest)

                #find target/ non targets
                target = X_swTrain[y_train==1]
                non_target = X_swTrain[y_train==0]

                distance_to_target = np.array([np.linalg.norm(X_swTest[i] - target.mean(axis=0)) for i in range(X_swTest.shape[0])])
                distance_to_nontarget = np.array([np.linalg.norm(X_swTest[i] - non_target.mean(axis=0)) for i in range(X_swTest.shape[0])])
                result = np.array((distance_to_nontarget / (distance_to_target + distance_to_nontarget), distance_to_target / (distance_to_target + distance_to_nontarget))).swapaxes(0, 1)
                ac_swlda = dp.evaluate_independent_epochs(result, epoch_info_test)


                #try to use a normal LDA with the selected features - try to add the feature weights? # intercept not known? looking at the sklearn documentation coef_ seems to be not the feature weights
                swlda = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None,
                                                   store_covariance=None, tol=0.0001, covariance_estimator=None)
                swlda.fit(X_swTrain, y_train)
                swlda.coef_ = weights_resampled[:, 3].reshape(1, 17)
                swlda_y_pred_prob = swlda.predict_proba(X_swTest)  # returns arrray : [[nontarget,target],[nt,t],[nt,t]...]
                ac_swlda = dp.evaluate_independent_epochs(swlda_y_pred_prob, epoch_info_test)


                # # trying to load feature weights into lda by scikit learn / changing parameters manually - also didn't work properly, also i don't know what i'm actually doing here
                swlda = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None,
                                                 store_covariance=None, tol=0.0001, covariance_estimator=None)
                swlda.classes_ = np.array((0,1))
                swlda.coef_ = weights_resampled[:, 3].reshape(1, 17)
                #explained_variance_ratio = 1 ? vermutlich falsch
                swlda.means_ = np.array((target.mean(axis=0), non_target.mean(axis=0)))
                swlda._max_components = 17
                swlda.n_features_in_ = None
                #priors =None
                #priors_ = [0.8,0,2] ? vermutlich auch falsch
                #estimate priors from sample:
                _, y_t = np.unique(y_train, return_inverse=True)  # non-negative ints
                swlda.priors_ = np.bincount(y_t) / float(len(y_train))
                swlda.scalings_ = (17,1)
                #shrinkage = None
                #solver = svd
                #store_covariance = None
                #tol =0.0001
                trials, ch,  samples = data_train_res.shape
                swlda._solve_svd(data_train_res.reshape(trials,samples*ch),y_train)
                coef = np.dot(swlda.means_ - swlda.xbar_, swlda.scalings_)
                swlda.intercept_ = -0.5 * np.sum(coef ** 2, axis=1) + np.log(swlda.priors_)
                swlda.intercept_ -= np.dot(swlda.xbar_, swlda.coef_.T)

                trials, ch, samples = data_test_res.shape
                swlda_prob_pred = swlda.predict_proba(data_test_res.reshape(trials,samples*ch))
                ac_swlda = dp.evaluate_independent_epochs(swlda_prob_pred, epoch_info_test)

                #plot distributions
                plt_swlda_train = X_swTrain.sum(axis=1)
                plt_swlda_train = np.array([plt_swlda_train[y_train==0],plt_swlda_train[y_train==1]])
                fig,ax = plt.subplots()
                ax.violinplot(plt_swlda_train,[0,1])
                plt.show()

                plt_swlda_test = X_swTest.sum(axis=1)
                plt_swlda_test = np.array([plt_swlda_test[y_test == 0], plt_swlda_test[y_test == 1]])
                fig, ax = plt.subplots()
                ax.violinplot(plt_swlda_test, [0, 1])
                plt.show()



                tar_for_pdf = plt_swlda_train[y_train==1]
                tar_test_pdf = plt_swlda_test[y_test==1]
                pdf_t = stats.norm.pdf(tar_test_pdf,loc=tar_for_pdf.mean(),scale=tar_for_pdf.std())

                plt.scatter(tar_for_pdf, pdf_t)
                plt.xlabel('x-data')
                plt.ylabel('pdf_value')
                plt.title("PDF of Target Normal Distribution with mean={} and sigma={}".format(round(tar_for_pdf.mean(), 2),
                                                                                          round(tar_for_pdf.std(), 2)))
                plt.show()

                nontar_for_pdf = plt_swlda_train[y_train == 0]
                pdf_nt = stats.norm.pdf(nontar_for_pdf, loc=nontar_for_pdf.mean(), scale=nontar_for_pdf.std())

                plt.scatter(nontar_for_pdf, pdf_nt)
                plt.xlabel('x-data')
                plt.ylabel('pdf_value')
                plt.title("PDF of  NonTarget Normal Distribution with mean={} and sigma={}".format(round(nontar_for_pdf.mean(),2),round(nontar_for_pdf.std(),2)))
                plt.show()




            #LDA
                lda = LinearDiscriminantAnalysis(solver='svd', shrinkage=None , priors=None, n_components=None, store_covariance=None, tol=0.0001, covariance_estimator=None)
                lda.fit(X_train,y_train)
                lda_y_pred_prob = lda.predict_proba(X_test) #returns list : [[nontarget,target],[nt,t],[nt,t]...]
                ac_lda = dp.evaluate_independent_epochs(lda_y_pred_prob,epoch_info_test)
                temp_df = dp.results_template(ac_lda,lda_name,sess)
                classifier_results =classifier_results.append(temp_df,ignore_index=True)

                roc_values[lda_name].append(metrics.roc_curve(y_test,lda_y_pred_prob[:,1]))
                print("The Accuracy with " + lda_name + " is: {}".format(ac_lda))
                continue
            # LDA with feature selection

            # #select 60 features
            #     selector = SelectKBest(f_classif, k=60)
            #     X_train_select = selector.fit_transform(X_train, y_train)
            #     X_test_select = selector.transform(X_test)
            #
            #     lda_select = LinearDiscriminantAnalysis(solver='svd', shrinkage=None , priors=None, n_components=None, store_covariance=None, tol=0.0001, covariance_estimator=None)
            #     lda_select.fit(X_train_select, y_train)
            #     lda_select_y_pred_prob = lda_select.predict_proba(X_test_select) #returns array : [[nontarget,target],[nt,t],[nt,t]...]
            #     ac_lda_select = dp.evaluate_independent_epochs(lda_select_y_pred_prob,epoch_info_test)
            #     temp_df = dp.results_template(ac_lda_select, lda60_name, sess)
            #     classifier_results =classifier_results.append(temp_df, ignore_index=True)
            #     roc_values[lda60_name].append(metrics.roc_curve(y_test,lda_select_y_pred_prob[:,1]))
            #     print("The Accuracy with " + lda60_name+ " is: {}".format(ac_lda_select))

            #shrinkage LDA
                lda_shrinkage = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto' , priors=None, n_components=None, store_covariance=None, tol=0.0001, covariance_estimator=None)
                lda_shrinkage.fit(X_train,y_train)
                lda_shrinkage_y_pred_prob = lda_shrinkage.predict_proba(X_test) #returns list : [[nontarget,target],[nt,t],[nt,t]...]
                ac_shrink_lda = dp.evaluate_independent_epochs(lda_shrinkage_y_pred_prob[1],epoch_info_test)
                temp_df = dp.results_template(ac_shrink_lda, shrinklda_name, sess)
                classifier_results =classifier_results.append(temp_df, ignore_index=True)
                roc_values[shrinklda_name].append(metrics.roc_curve(y_test, lda_shrinkage_y_pred_prob[:, 1]))
                print("The Accuracy with " +shrinklda_name + " is: {}".format(ac_shrink_lda))


            #Riemann MDM

            t0=time.time()
            mdm = MDM()
            mdm.fit(cov_matrices_train, y_train)
            mdm_y_pred_prob = mdm.predict_proba(cov_matrices_test)
            ac_mdm = dp.evaluate_independent_epochs(mdm_y_pred_prob[1],epoch_info_test)
            #accuracy[5].append(ac_mdm)
            temp_df = dp.results_template(ac_mdm, mdm_name, sess)
            classifier_results =classifier_results.append(temp_df, ignore_index=True)
            roc_values[mdm_name].append(metrics.roc_curve(y_test, mdm_y_pred_prob[:, 1]))
            print("The Accuracy with " +mdm_name+ " is: {}".format(ac_mdm))


            #MDM with FGDA in tangentspace
            fgmdm = FgMDM()
            fgmdm.fit(cov_matrices_train, y_train)
            fgmdm_y_pred_prob = fgmdm.predict_proba(cov_matrices_test)
            ac_fgmdm = dp.evaluate_independent_epochs(fgmdm_y_pred_prob[1],epoch_info_test)
            temp_df = dp.results_template(ac_fgmdm, fgmdm_name, sess)
            classifier_results =classifier_results.append(temp_df, ignore_index=True)
            roc_values[fgmdm_name].append(metrics.roc_curve(y_test, fgmdm_y_pred_prob[:, 1]))

            print("The Accuracy with " +fgmdm_name+ " is: {}".format(ac_fgmdm))

        df = classifier_results.loc[classifier_results["Ep2Avg"] == 8]
        classifier_results.to_csv(r"D:\Google Drive\Google Drive\Google Drive\Master\Masterarbeit\Data\Classifier_Results\accuracies_trans3.csv")
        with open(r"D:\Google Drive\Google Drive\Google Drive\Master\Masterarbeit\Data\Classifier_Results\roc_values_trans3.pickle","wb") as file:
            pickle.dump(roc_values, file)

