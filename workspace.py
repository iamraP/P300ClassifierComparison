#from __future__ import print_function
import numpy as np

import data_prep as dp
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif,SelectKBest
from pyriemann.classification import FgMDM,MDM
from sklearn import metrics
import time
import swlda as sw

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
sess_list = range(1,50)  #range(1,50) #list(range(1,50)) #neues Setup ab Session 21 (12. Messtag)
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
              "SWLDA":[],
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
                swlda = sw.swlda()
                swlda.fit(X_train,y_train)
                swlda_y_pred_prob = swlda.predict_proba(X_test)
                ac_swlda = dp.evaluate_independent_epochs(swlda_y_pred_prob, epoch_info_test)
                print("The Accuracy with " + swlda_name + " averaged after classification is: {}".format(ac_swlda[-1]))

                ep2avg = 10
                X_train_mean = np.zeros((int(X_train.shape[0] / ep2avg), X_train.shape[1]))
                y_train_mean = np.zeros((int(X_train.shape[0] / ep2avg)))
                tactilo_mean = np.zeros((int(X_train.shape[0] / ep2avg)))
                trial_mean = np.zeros((int(X_train.shape[0] / ep2avg)))
                idx = 0
                for tactilo in range(6):
                    trial_idx = np.split(epoch_info_train["epoch"].loc[(epoch_info_train["tactilo"] == tactilo + 1)].unique(),
                                         18)  # 18 weil 3 Runs a 6 "Runden" pro Taktilo (1*Target, 5*Non-Target)
                    # trial idx gibt an welche epoch zu einem Trial gehören (von dem aktuellen Taktilo)
                    for trial in range(len(trial_idx)):
                        ground_truth = epoch_info_train["condition"].loc[epoch_info_train["epoch"].isin(
                            trial_idx[trial])].mean()  # bildet den mittelwert aus ep2avg epochen (single value)
                        assert ground_truth == 1 or ground_truth == 0  # make sure the epochs didn't mix
                        epoch_mean = X_train.loc[epoch_info_train["epoch"].isin(trial_idx[trial])][
                                     :ep2avg].mean()  # mittelwert aus ep2avg Epochen (should be a single epoch)

                        X_train_mean[idx, :] = epoch_mean
                        y_train_mean[idx] = ground_truth
                        tactilo_mean[idx] = tactilo
                        trial_mean[idx] = trial
                        idx += 1


                swlda = sw.swlda()
                swlda.fit(X_train_mean,y_train_mean)

                # test data



                ep2avg = 8
                X_test_mean= np.zeros((int(X_test.shape[0]/ep2avg),X_test.shape[1]))
                y_test_mean = np.zeros(18)
                tactilo_mean = np.zeros((int(X_test.shape[0]/ep2avg)))
                trial_mean = np.zeros((int(X_test.shape[0]/ep2avg)))
                idx=0
                for tactilo in range(6):
                    trial_idx = np.split(epoch_info_test["epoch"].loc[(epoch_info_test["tactilo"] == tactilo + 1)].unique(),
                                         18)  # 18 weil 3 Runs a 6 "Runden" pro Taktilo (1*Target, 5*Non-Target)
                    # trial idx gibt an welche epoch zu einem Trial gehören (von dem aktuellen Taktilo)
                    for trial in range(len(trial_idx)):
                        ground_truth = epoch_info_test["condition"].loc[epoch_info_test["epoch"].isin(
                            trial_idx[trial])].mean()  # bildet den mittelwert aus ep2avg epochen (single value)
                        assert ground_truth == 1 or ground_truth == 0  # make sure the epochs didn't mix
                        epoch_mean = X_test.loc[epoch_info_test["epoch"].isin(trial_idx[trial])][
                                     :ep2avg].mean()  # mittelwert aus ep2avg Epochen (should be a single epoch)

                        X_test_mean[idx,:] = epoch_mean
                        if ground_truth ==1:
                            y_test_mean[trial] = tactilo
                        tactilo_mean [idx] = tactilo
                        trial_mean[idx] = trial
                        idx +=1
                swlda_y_pred_prob = swlda.predict_proba(X_test_mean)
                eval_array = np.zeros((6, 18))
                for trial in range(18):

                    for tactilo in range(6):
                        eval_array[tactilo,trial] = swlda_y_pred_prob[((tactilo_mean==tactilo) &  (trial_mean==trial))][0][1]



                    ac_swlda = (eval_array.argmax(axis=0)== y_test_mean).sum()/len(y_test_mean)

                # ac_swlda = dp.evaluate_independent_epochs(swlda_y_pred_prob, epoch_info_test)
                # temp_df = dp.results_template(ac_swlda, swlda_name, sess)
                # classifier_results = classifier_results.append(temp_df, ignore_index=True)
                #
                # roc_values[swlda_name].append(metrics.roc_curve(y_test, swlda_y_pred_prob[:, 1]))
                print("The Accuracy with " + swlda_name + " averaged before classification is: {}".format(ac_swlda))
            # #LDA
            #     lda = LinearDiscriminantAnalysis(solver='svd', shrinkage=None , priors=None, n_components=None, store_covariance=None, tol=0.0001, covariance_estimator=None)
            #     lda.fit(X_train,y_train)
            #     lda_y_pred_prob = lda.predict_proba(X_test) #returns list : [[nontarget,target],[nt,t],[nt,t]...]
            #     ac_lda = dp.evaluate_independent_epochs(lda_y_pred_prob,epoch_info_test)
            #     temp_df = dp.results_template(ac_lda,lda_name,sess)
            #     classifier_results =classifier_results.append(temp_df,ignore_index=True)
            #
            #     roc_values[lda_name].append(metrics.roc_curve(y_test,lda_y_pred_prob[:,1]))
            #     print("The Accuracy with " + lda_name + " is: {}".format(ac_lda))
            #     continue
            # # LDA with feature selection
            #
            # # #select 60 features
            # #     selector = SelectKBest(f_classif, k=60)
            # #     X_train_select = selector.fit_transform(X_train, y_train)
            # #     X_test_select = selector.transform(X_test)
            # #
            # #     lda_select = LinearDiscriminantAnalysis(solver='svd', shrinkage=None , priors=None, n_components=None, store_covariance=None, tol=0.0001, covariance_estimator=None)
            # #     lda_select.fit(X_train_select, y_train)
            # #     lda_select_y_pred_prob = lda_select.predict_proba(X_test_select) #returns array : [[nontarget,target],[nt,t],[nt,t]...]
            # #     ac_lda_select = dp.evaluate_independent_epochs(lda_select_y_pred_prob,epoch_info_test)
            # #     temp_df = dp.results_template(ac_lda_select, lda60_name, sess)
            # #     classifier_results =classifier_results.append(temp_df, ignore_index=True)
            # #     roc_values[lda60_name].append(metrics.roc_curve(y_test,lda_select_y_pred_prob[:,1]))
            # #     print("The Accuracy with " + lda60_name+ " is: {}".format(ac_lda_select))
            #
            # #shrinkage LDA
            #     lda_shrinkage = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto' , priors=None, n_components=None, store_covariance=None, tol=0.0001, covariance_estimator=None)
            #     lda_shrinkage.fit(X_train,y_train)
            #     lda_shrinkage_y_pred_prob = lda_shrinkage.predict_proba(X_test) #returns list : [[nontarget,target],[nt,t],[nt,t]...]
            #     ac_shrink_lda = dp.evaluate_independent_epochs(lda_shrinkage_y_pred_prob[1],epoch_info_test)
            #     temp_df = dp.results_template(ac_shrink_lda, shrinklda_name, sess)
            #     classifier_results =classifier_results.append(temp_df, ignore_index=True)
            #     roc_values[shrinklda_name].append(metrics.roc_curve(y_test, lda_shrinkage_y_pred_prob[:, 1]))
            #     print("The Accuracy with " +shrinklda_name + " is: {}".format(ac_shrink_lda))
            #
            #
            # #Riemann MDM
            #
            # t0=time.time()
            # mdm = MDM()
            # mdm.fit(cov_matrices_train, y_train)
            # mdm_y_pred_prob = mdm.predict_proba(cov_matrices_test)
            # ac_mdm = dp.evaluate_independent_epochs(mdm_y_pred_prob[1],epoch_info_test)
            # #accuracy[5].append(ac_mdm)
            # temp_df = dp.results_template(ac_mdm, mdm_name, sess)
            # classifier_results =classifier_results.append(temp_df, ignore_index=True)
            # roc_values[mdm_name].append(metrics.roc_curve(y_test, mdm_y_pred_prob[:, 1]))
            # print("The Accuracy with " +mdm_name+ " is: {}".format(ac_mdm))
            #
            #
            # #MDM with FGDA in tangentspace
            # fgmdm = FgMDM()
            # fgmdm.fit(cov_matrices_train, y_train)
            # fgmdm_y_pred_prob = fgmdm.predict_proba(cov_matrices_test)
            # ac_fgmdm = dp.evaluate_independent_epochs(fgmdm_y_pred_prob[1],epoch_info_test)
            # temp_df = dp.results_template(ac_fgmdm, fgmdm_name, sess)
            # classifier_results =classifier_results.append(temp_df, ignore_index=True)
            # roc_values[fgmdm_name].append(metrics.roc_curve(y_test, fgmdm_y_pred_prob[:, 1]))
            #
            # print("The Accuracy with " +fgmdm_name+ " is: {}".format(ac_fgmdm))

        df = classifier_results.loc[classifier_results["Ep2Avg"] == 8]
        classifier_results.to_csv(r"D:\Google Drive\Google Drive\Google Drive\Master\Masterarbeit\Data\Classifier_Results_current\accuracies_swlda.csv")
        with open(r"D:\Google Drive\Google Drive\Google Drive\Master\Masterarbeit\Data\Classifier_Results\roc_values_swlda.pickle","wb") as file:
            pickle.dump(roc_values, file)

