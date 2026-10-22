#from __future__ import print_function
import data_prep as dp
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from pyriemann.classification import FgMDM,MDM,TSclassifier
from sklearn import metrics
import csv
import time
import swlda as sw
from tqdm.auto import tqdm

t =time.time()
################################################
'''SETTINGS ''' #mostly only relevant if loop over all data is commented out
################################################
excluded_sessions = [4,7,10,11,12,16,18,19,20,32] #some sessions were training session which followed after the standard protocol (they happend on the same day and are excluded from analysis)
sessionwise_calibration = True #False for transfer classifiers
trans_calib_sess = [1,2,3] # only relevant when  sessionwise_calibration == False
calib_runs=3 # how many of the runs should be used for calibration? (max=3), this will be overwritten later if classification is selected
electrode_list = 'all' #["Fz","Fc1","Fc2","C3","Cz","C4","Pz"] # which electrodes to include, options: 'all' or ["Fz","Fc1","Fc2","C3","Cz","C4","Pz"]== set of unchanged electrodes
classification = True #if only analysis is done don't load data in the complex way it's needed for classification
investigate_sessionwise = False  #used for evaluation, figure plotting etc
investigate_several_sessions = False #set range in script, used for averaging of several sessions
sess_list =list(range(1,50))#range(1,50)  #range(1,50) #list(range(1,50)) #neues Setup ab Session 21 (12. Messtag) # for PUG range(1,27)
data_origin = r"data_thesis"
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
        dp.investigate(data_path, sess_name=str(sess))
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


roc_values = {"LDA": [],
              "SWLDA": [],
              "shrinkLDA": [],
              "LDA_xDawn": [],
              "SWLDA_xDawn": [],
              "shrinkLDA_xDawn": [],
              "MDM": [],
              'FGMDM': [],
              'TS_LG': [],
              'TS_SVM_lin': [],
              'TS_SVM_rbf': [],
              'MDM_res': [],
              'FGMDM_res': [],
              'TS_LG_res': [],
              'TS_SVM_lin_res': [],
              'TS_SVM_rbf_res': [],
              'MDM_xDawn': [],
              'FGMDM_xDawn': [],
              'TS_LG_xDawn': [],
              'TS_SVM_lin_xDawn': [],
              'TS_SVM_rbf_xDawn': [],
              'MDM_res_xDawn': [],
              'FGMDM_res_xDawn': [],
              'TS_LG_res_xDawn': [],
              'TS_SVM_lin_res_xDawn': [],
              'TS_SVM_rbf_res_xDawn': []}



total = 0
pbar_total = tqdm(desc="Total", total=6)
# loop for getting every condition in one run - spagetthi-deluxe
classifier_accuracy = pd.DataFrame(columns=["Accuracy", "Classifier", "Session","Condition", "Ep2Avg"])
classifier_roc = pd.DataFrame(columns=["Classifier", "Session","Condition", "Treshold","FPR","TPR"]) #fpr: false positive rate, tpr: true positive rate
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
        calib_runs = 3  # how many of the runs should be used for calibration? (max=3)
        sess_list = range(1,21)  # range(1,50) #list(range(1,50)) #neues Setup ab Session 21 (12. Messtag) # for PUG range(1,27)
    elif condition == "single1_B":
        sessionwise_calibration = False  # False for transfer classifiers
        trans_calib_sess = [21]  # only relevant when  sessionwise_calibration == False
        calib_runs = 3  # how many of the runs should be used for calibration? (max=3)
        sess_list = range(21,50)  # range(1,50) #list(range(1,50)) #neues Setup ab Session 21 (12. Messtag) # for PUG range(1,27)
    elif condition == "single3_A":
        sessionwise_calibration = False  # False for transfer classifiers
        trans_calib_sess = [1,2,3]  # only relevant when  sessionwise_calibration == False
        calib_runs = 3  # how many of the runs should be used for calibration? (max=3)
        sess_list = range(1,21)  # range(1,50) #list(range(1,50)) #neues Setup ab Session 21 (12. Messtag) # for PUG range(1,27)
    elif condition == "single3_B":
        sessionwise_calibration = False  # False for transfer classifiers
        trans_calib_sess = [21,22,23]  # only relevant when  sessionwise_calibration == False
        calib_runs = 3  # how many of the runs should be used for calibration? (max=3)
        sess_list = range(21,50)  # range(1,50) #list(range(1,50)) #neues Setup ab Session 21 (12. Messtag) # for PUG range(1,27)

    condition_tqdm = 0
    if "_A" in condition:
        pbar_condition = tqdm(desc=condition, total=144,position=0,leave=True)
    elif "_B" in condition:
        pbar_condition = tqdm(desc=condition, total=392,position=0,leave=True)
    else:
        pbar_condition = tqdm(desc=condition, total=1014, position=0,leave=True) # + 39 per condition (number of sessions?)

    dates = []



    swlda_weights = []
    channel_list = []
    for resampled_riemann in [True,False]:
        for xDawn in [True,False]:

            first_session = True

            mdm_name = "MDM"
            fgmdm_name = "FGMDM"
            lda_name = "LDA"
            swlda_name = "SWLDA"
            shrinklda_name = "shrinkLDA"
            ts_lg_name = "TS_LG"
            ts_svm_lin_name = "TS_SVM_lin"
            ts_svm_rbf_name = "TS_SVM_rbf"

            if resampled_riemann:
                mdm_name = mdm_name + "_res"
                fgmdm_name = fgmdm_name + "_res"
                ts_lg_name = ts_lg_name + "_res"
                ts_svm_lin_name = ts_svm_lin_name + "_res"
                ts_svm_rbf_name = ts_svm_rbf_name + "_res"
            if xDawn:
                mdm_name = mdm_name + "_xDawn"
                fgmdm_name = fgmdm_name + "_xDawn"
                lda_name = lda_name + "_xDawn"
                swlda_name = swlda_name + "_xDawn"
                shrinklda_name = shrinklda_name +"_xDawn"
                ts_lg_name = ts_lg_name + "_xDawn"
                ts_svm_lin_name = ts_svm_lin_name + "_xDawn"
                ts_svm_rbf_name = ts_svm_rbf_name + "_xDawn"

            for sess in sess_list:
                data_path = []
                if sess in excluded_sessions:
                    continue


                should_train = sessionwise_calibration or first_session

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
                    if should_train:
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
                    t0 = time.time()
                    if should_train:
                        swlda = sw.swlda()
                        swlda.fit(X_train, y_train)
                    if swlda.weights is None:
                        ac_swlda = [np.nan] * 8
                    else:
                        swlda_y_pred_prob = swlda.predict_proba(X_test)
                        ac_swlda = dp.evaluate_independent_epochs(swlda_y_pred_prob, epoch_info_test)
                        roc_values[swlda_name].append(metrics.roc_curve(y_test, swlda_y_pred_prob[:, 1]))
                    temp_df = dp.results_template(ac_swlda, swlda_name, sess, condition)
                    classifier_accuracy = classifier_accuracy.append(temp_df, ignore_index=True)

                    t1 = time.time()
                    with open('classifer_times.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([condition,sess,swlda_name, t1 - t0])

                    if swlda.weights is not None:
                        roc_swlda = metrics.roc_curve(y_test, swlda_y_pred_prob[:, 1])
                        temp_roc_df = dp.roc_template(roc_swlda, swlda_name, sess, condition)
                    else:
                        temp_roc_df = pd.DataFrame([{
                            "Classifier": swlda_name,
                            "Session": sess,
                            "Condition": condition,
                            "Treshold": np.nan,
                            "FPR": np.nan,
                            "TPR": np.nan
                        }])

                    classifier_roc = classifier_roc.append(temp_roc_df,ignore_index=True)
                    #print("The Accuracy with " + swlda_name + " is: {}".format(ac_swlda))
                    pbar_condition.update(condition_tqdm + 1)

                #LDA
                    t0 = time.time()
                    if should_train:
                        lda = LinearDiscriminantAnalysis(solver='svd', shrinkage=None , priors=None, n_components=None, store_covariance=None, tol=0.0001, covariance_estimator=None)
                        lda.fit(X_train,y_train)
                    lda_y_pred_prob = lda.predict_proba(X_test) #returns list : [[nontarget,target],[nt,t],[nt,t]...]
                    ac_lda = dp.evaluate_independent_epochs(lda_y_pred_prob,epoch_info_test)
                    temp_df = dp.results_template(ac_lda,lda_name, sess, condition)
                    classifier_accuracy =classifier_accuracy.append(temp_df,ignore_index=True)

                    t1 = time.time()
                    with open('classifer_times.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([condition,sess,lda_name, t1 - t0])

                    roc_lda = metrics.roc_curve(y_test, lda_y_pred_prob[:, 1])
                    temp_roc_df = dp.roc_template(roc_lda, lda_name, sess, condition)
                    classifier_roc = classifier_roc.append(temp_roc_df,ignore_index=True)
                    #print("The Accuracy with " + lda_name + " is: {}".format(ac_lda))
                    pbar_condition.update(condition_tqdm + 1)

                #shrinkage LDA
                    t0 = time.time()
                    if should_train:
                        shrinklda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto' , priors=None, n_components=None, store_covariance=None, tol=0.0001, covariance_estimator=None)
                        shrinklda.fit(X_train,y_train)
                    shrinklda_y_pred_prob = shrinklda.predict_proba(X_test) #returns list : [[nontarget,target],[nt,t],[nt,t]...]
                    ac_shrinklda = dp.evaluate_independent_epochs(shrinklda_y_pred_prob,epoch_info_test)
                    temp_df = dp.results_template(ac_shrinklda, shrinklda_name, sess, condition)
                    classifier_accuracy =classifier_accuracy.append(temp_df, ignore_index=True)

                    t1 = time.time()
                    with open('classifer_times.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([condition,sess,shrinklda_name, t1 - t0])

                    roc_shrinklda = metrics.roc_curve(y_test, shrinklda_y_pred_prob[:, 1])
                    temp_roc_df = dp.roc_template(roc_shrinklda, shrinklda_name, sess, condition)
                    classifier_roc = classifier_roc.append(temp_roc_df,ignore_index=True)
                    #print("The Accuracy with " +shrinklda_name + " is: {}".format(ac_shrink_lda))
                    pbar_condition.update(condition_tqdm + 1)


                #Riemann MDM

                t0 = time.time()
                if should_train:
                    mdm = MDM()
                    mdm.fit(cov_matrices_train, y_train)
                mdm_y_pred_prob = mdm.predict_proba(cov_matrices_test)
                ac_mdm = dp.evaluate_independent_epochs(mdm_y_pred_prob,epoch_info_test)

                t1 = time.time()
                with open('classifer_times.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([condition,sess,mdm_name,t1-t0])
                #accuracy[5].append(ac_mdm)
                temp_df = dp.results_template(ac_mdm, mdm_name, sess, condition)
                classifier_accuracy =classifier_accuracy.append(temp_df, ignore_index=True)
                roc_values[mdm_name].append(metrics.roc_curve(y_test, mdm_y_pred_prob[:, 1]))

                #print("The Accuracy with " +mdm_name+ " is: {}".format(ac_mdm))
                pbar_condition.update(condition_tqdm + 1)



                #MDM with FGDA in tangentspace
                t0 = time.time()
                if should_train:
                    fgmdm = FgMDM()
                    fgmdm.fit(cov_matrices_train, y_train)
                fgmdm_y_pred_prob = fgmdm.predict_proba(cov_matrices_test)
                ac_fgmdm = dp.evaluate_independent_epochs(fgmdm_y_pred_prob,epoch_info_test)
                temp_df = dp.results_template(ac_fgmdm, fgmdm_name, sess, condition)
                classifier_accuracy =classifier_accuracy.append(temp_df, ignore_index=True)


                t1 = time.time()
                with open('classifer_times.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([fgmdm_name,t1-t0])

                roc_fgmdm = metrics.roc_curve(y_test, fgmdm_y_pred_prob[:, 1])
                temp_roc_df = dp.roc_template(roc_fgmdm, fgmdm_name, sess, condition)
                classifier_roc = classifier_roc.append(temp_roc_df,ignore_index=True)

                pbar_condition.update(condition_tqdm + 1)

                #print("The Accuracy with " +fgmdm_name+ " is: {}".format(ac_fgmdm))

                # Classifcation in Tagent Space

                # Logistic Regression with Lasso penalty (stackoverflow suggest l1, but this
                t0 = time.time()
                if should_train:
                    ts_lg = TSclassifier(metric='riemann', tsupdate=False, clf=LogisticRegression(penalty='l1',solver='liblinear', random_state=7)) # ValueError: Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
                    ts_lg.fit(cov_matrices_train, y_train)
                ts_lg_y_pred_prob = ts_lg.predict_proba(cov_matrices_test)
                ac_ts_lg = dp.evaluate_independent_epochs(ts_lg_y_pred_prob, epoch_info_test)
                temp_df = dp.results_template(ac_ts_lg, ts_lg_name, sess, condition)
                classifier_accuracy = classifier_accuracy.append(temp_df, ignore_index=True)

                t1 = time.time()
                with open('classifer_times.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([ts_lg_name, t1 - t0])

                roc_ts_lg = metrics.roc_curve(y_test, ts_lg_y_pred_prob[:, 1])
                temp_roc_df = dp.roc_template(roc_ts_lg, ts_lg_name, sess, condition)
                classifier_roc = classifier_roc.append(temp_roc_df, ignore_index=True)

                pbar_condition.update(condition_tqdm + 1)

                # SVM linear
                t0 = time.time()
                if should_train:
                    ts_svm_lin = TSclassifier(metric='riemann', tsupdate=False, clf=SVC(kernel='linear', probability=True, random_state=7)) # ValueError: Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
                    ts_svm_lin.fit(cov_matrices_train, y_train)
                ts_svm_lin_y_pred_prob = ts_svm_lin.predict_proba(cov_matrices_test)
                ac_ts_svm_lin = dp.evaluate_independent_epochs(ts_svm_lin_y_pred_prob, epoch_info_test)
                temp_df = dp.results_template(ac_ts_svm_lin, ts_svm_lin_name, sess, condition)
                classifier_accuracy = classifier_accuracy.append(temp_df, ignore_index=True)

                t1 = time.time()
                with open('classifer_times.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([ts_svm_lin_name, t1 - t0])

                roc_ts_svm_lin = metrics.roc_curve(y_test, ts_svm_lin_y_pred_prob[:, 1])
                temp_roc_df = dp.roc_template(roc_ts_svm_lin, ts_svm_lin_name, sess, condition)
                classifier_roc = classifier_roc.append(temp_roc_df, ignore_index=True)

                pbar_condition.update(condition_tqdm + 1)


                # SVM rbf
                t0 = time.time()
                if should_train:
                    ts_svm_rbf = TSclassifier(metric='riemann', tsupdate=False, clf=SVC(kernel='rbf', probability=True, random_state=7))
                    ts_svm_rbf.fit(cov_matrices_train, y_train)
                ts_svm_rbf_y_pred_prob = ts_svm_rbf.predict_proba(cov_matrices_test)
                ac_ts_svm_rbf = dp.evaluate_independent_epochs(ts_svm_rbf_y_pred_prob, epoch_info_test)
                temp_df = dp.results_template(ac_ts_svm_rbf, ts_svm_rbf_name, sess, condition)
                classifier_accuracy = classifier_accuracy.append(temp_df, ignore_index=True)

                t1 = time.time()
                with open('classifer_times.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([ts_svm_rbf_name, t1 - t0])

                roc_ts_svm_rbf = metrics.roc_curve(y_test, ts_svm_rbf_y_pred_prob[:, 1])
                temp_roc_df = dp.roc_template(roc_ts_svm_rbf, ts_svm_rbf_name, sess, condition)
                classifier_roc = classifier_roc.append(temp_roc_df, ignore_index=True)

                pbar_condition.update(condition_tqdm + 1)

                first_session = False


            pbar_total.update(total + 1)
            classifier_roc.to_csv(r"roc.csv",index=False)
            classifier_accuracy.to_csv(r"accuracies.csv",index=False)
            #sp.spaghetti_code(2)
    #sp.spaghetti_code(1)
print("It took: {} min to run the entire script".format(round((time.time()-t)/60)))