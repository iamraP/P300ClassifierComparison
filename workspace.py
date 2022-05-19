#from __future__ import print_function
import numpy as np
import time
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
import spaghettimonster as sp
import swlda_old as sw_old
from tqdm.auto import tqdm

t =time.time()
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
sess_list = range(1,50)  #range(1,50) #list(range(1,50)) #neues Setup ab Session 21 (12. Messtag) # for PUG range(1,27)
data_origin = r"C:\Users\map92fg\Documents\Software\P300_Classification\data_thesis\mne_raw_pickled"
################################################
'''SETTINGS STOP'''
################################################
total = 0
pbar_total = tqdm(desc="Total", total=624)
# loop for getting every condition in one run - spagetthi-deluxe
classifier_results = pd.DataFrame(columns=["Accuracy", "Classifier", "Session","Condition", "Ep2Avg"])
conditions = ["sess3","sess1","single3_A","single3_B","single1_A","single1_B"]
for condition in conditions:
    if condition == "sess3":
        sessionwise_calibration = True  # False for transfer classifiers
        calib_runs = 3  # how many of the runs should be used for calibration? (max=3)
        sess_list = range(1,50)  # range(1,50) #list(range(1,50)) #neues Setup ab Session 21 (12. Messtag) # for PUG range(1,27)
    elif condition == "sess1":
        sessionwise_calibration = True  # False for transfer classifiers
        calib_runs = 1  # how many of the runs should be used for calibration? (max=3)
        sess_list = range(1,50)  # range(1,50) #list(range(1,50)) #neues Setup ab Session 21 (12. Messtag) # for PUG range(1,27)
    elif condition == "single1_A":
        sessionwise_calibration = False  # False for transfer classifiers
        trans_calib_sess = [1]  # only relevant when  sessionwise_calibration == False
        calib_runs = 1  # how many of the runs should be used for calibration? (max=3)
        sess_list = range(1,21)  # range(1,50) #list(range(1,50)) #neues Setup ab Session 21 (12. Messtag) # for PUG range(1,27)
    elif condition == "single1_B":
        sessionwise_calibration = False  # False for transfer classifiers
        trans_calib_sess = [1]  # only relevant when  sessionwise_calibration == False
        calib_runs = 1  # how many of the runs should be used for calibration? (max=3)
        sess_list = range(21,50)  # range(1,50) #list(range(1,50)) #neues Setup ab Session 21 (12. Messtag) # for PUG range(1,27)
    elif condition == "single3_A":
        sessionwise_calibration = False  # False for transfer classifiers
        trans_calib_sess = [1,2,3]  # only relevant when  sessionwise_calibration == False
        calib_runs = 1  # how many of the runs should be used for calibration? (max=3)
        sess_list = range(1,21)  # range(1,50) #list(range(1,50)) #neues Setup ab Session 21 (12. Messtag) # for PUG range(1,27)
    elif condition == "single3_B":
        sessionwise_calibration = False  # False for transfer classifiers
        trans_calib_sess = [1,2,3]  # only relevant when  sessionwise_calibration == False
        calib_runs = 1  # how many of the runs should be used for calibration? (max=3)
        sess_list = range(21,50)  # range(1,50) #list(range(1,50)) #neues Setup ab Session 21 (12. Messtag) # for PUG range(1,27)

    condition_tqdm = 0
    pbar_condition = tqdm(desc=condition, total=156)

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


    dates = []

    roc_values = {"LDA":[],
                  "SWLDA":[],
                  "shrinkLDA":[],
                  "LDA_xDawn":[],
                  "SWLDA_xDawn":[],
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

    swlda_weights = []
    channel_list = []
    for resampled_riemann in [False,True]: #TODO rechange to [True,False]
        for xDawn in [True,False]:
            mdm_name = "MDM"
            fgmdm_name = "FGMDM"
            lda_name = "LDA"
            swlda_name = "SWLDA"
            shrinklda_name = "shrinkLDA"

            if resampled_riemann:
                mdm_name = mdm_name + "_res"
                fgmdm_name = fgmdm_name + "_res"
            if xDawn:
                mdm_name = mdm_name + "_xDawn"
                fgmdm_name = fgmdm_name + "_xDawn"
                lda_name = lda_name + "_xDawn"
                swlda_name = swlda_name + "_xDawn"
                shrinklda_name = shrinklda_name +"_xDawn"

            for sess in sess_list:
                data_path = []
                if sess in excluded_sessions:
                    continue

                if not sessionwise_calibration and sess==21: # recalibrate the classifier after the switch of the electrode set
                    first_session =True

                #print("===============================\nWorking on sesssion {}\n===============================".format(str(sess).zfill(3)))
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
                    swlda.fit(X_train, y_train)
                    if swlda.weights is None:
                        ac_swlda = np.NaN
                    else:
                        swlda_y_pred_prob = swlda.predict_proba(X_test)
                        ac_swlda = dp.evaluate_independent_epochs(swlda_y_pred_prob, epoch_info_test)
                        roc_values[swlda_name].append(metrics.roc_curve(y_test, swlda_y_pred_prob[:, 1]))
                    temp_df = dp.results_template(ac_swlda, swlda_name, sess, condition)
                    classifier_results = classifier_results.append(temp_df, ignore_index=True)
                    #print("The Accuracy with " + swlda_name + " is: {}".format(ac_swlda))

                #LDA
                    lda = LinearDiscriminantAnalysis(solver='svd', shrinkage=None , priors=None, n_components=None, store_covariance=None, tol=0.0001, covariance_estimator=None)
                    lda.fit(X_train,y_train)
                    lda_y_pred_prob = lda.predict_proba(X_test) #returns list : [[nontarget,target],[nt,t],[nt,t]...]
                    ac_lda = dp.evaluate_independent_epochs(lda_y_pred_prob,epoch_info_test)
                    temp_df = dp.results_template(ac_lda,lda_name, sess, condition)
                    classifier_results =classifier_results.append(temp_df,ignore_index=True)

                    roc_values[lda_name].append(metrics.roc_curve(y_test,lda_y_pred_prob[:,1]))
                    #print("The Accuracy with " + lda_name + " is: {}".format(ac_lda))

                #shrinkage LDA
                    lda_shrinkage = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto' , priors=None, n_components=None, store_covariance=None, tol=0.0001, covariance_estimator=None)
                    lda_shrinkage.fit(X_train,y_train)
                    lda_shrinkage_y_pred_prob = lda_shrinkage.predict_proba(X_test) #returns list : [[nontarget,target],[nt,t],[nt,t]...]
                    ac_shrink_lda = dp.evaluate_independent_epochs(lda_shrinkage_y_pred_prob,epoch_info_test)
                    temp_df = dp.results_template(ac_shrink_lda, shrinklda_name, sess, condition)
                    classifier_results =classifier_results.append(temp_df, ignore_index=True)
                    roc_values[shrinklda_name].append(metrics.roc_curve(y_test, lda_shrinkage_y_pred_prob[:, 1]))
                    #print("The Accuracy with " +shrinklda_name + " is: {}".format(ac_shrink_lda))


                #Riemann MDM

                t0=time.time()
                mdm = MDM()
                mdm.fit(cov_matrices_train, y_train)
                mdm_y_pred_prob = mdm.predict_proba(cov_matrices_test)
                ac_mdm = dp.evaluate_independent_epochs(mdm_y_pred_prob,epoch_info_test)
                #accuracy[5].append(ac_mdm)
                temp_df = dp.results_template(ac_mdm, mdm_name, sess, condition)
                classifier_results =classifier_results.append(temp_df, ignore_index=True)
                roc_values[mdm_name].append(metrics.roc_curve(y_test, mdm_y_pred_prob[:, 1]))
                #print("The Accuracy with " +mdm_name+ " is: {}".format(ac_mdm))


                #MDM with FGDA in tangentspace
                fgmdm = FgMDM()
                fgmdm.fit(cov_matrices_train, y_train)
                fgmdm_y_pred_prob = fgmdm.predict_proba(cov_matrices_test)
                ac_fgmdm = dp.evaluate_independent_epochs(fgmdm_y_pred_prob,epoch_info_test)
                temp_df = dp.results_template(ac_fgmdm, fgmdm_name, sess, condition)
                classifier_results =classifier_results.append(temp_df, ignore_index=True)
                roc_values[fgmdm_name].append(metrics.roc_curve(y_test, fgmdm_y_pred_prob[:, 1]))

                #print("The Accuracy with " +fgmdm_name+ " is: {}".format(ac_fgmdm))

                pbar_total.update(total+1)


            classifier_results.to_csv(r"C:\Users\map92fg\Documents\Software\P300_Classification\created_data\Classifier_Results\accuracies_19_05_22_V2.csv",index=False)
            with open(r"C:\Users\map92fg\Documents\Software\P300_Classification\created_data\Classifier_Results\roc_values_19_05_22_V2.pickle","wb") as file:
                pickle.dump(roc_values, file)
            sp.spaghetti_code(2)
            pbar_condition.update(condition_tqdm +156)
    sp.spaghetti_code(1)
print("It took: {} min to run the entire script".format(round((time.time()-t)/60)))