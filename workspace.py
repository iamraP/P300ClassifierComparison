#from __future__ import print_function
import data_prep as dp
import pandas as pd
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif,SelectKBest
from pyriemann.classification import FgMDM,MDM
from sklearn import metrics
import time
import swlda
import numpy as np
from scipy.spatial import distance
import py3gui

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
data_origin = r"C:\Users\map92fg\Documents\Software\P300_Classification\data_thesis"
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


            #train classifiers and evaluate classifiers
            if not resampled_riemann:

                from sklearn import preprocessing

            #SWLDA
                data_train = epochs_train.get_data().swapaxes(1,2)*1e6
                channels, weights, weights_resampled = swlda.swlda(data_train,y_train,512,[0,800],20)




                train = epochs_train.copy().resample(20).get_data()*1e6


                X_swTrain = np.zeros((weights_resampled.shape[0], train.shape[0]))
                for i, feature in enumerate(weights_resampled):
                    X_swTrain[i, :] = train[:, feature[0].astype(int), feature[1].astype(int)]#* feature[3]

                X_swTrain = X_swTrain.swapaxes(0, 1)
                scaler = preprocessing.StandardScaler().fit(X_swTrain, y_train)
                X_swTrain = scaler.transform(X_swTrain)

                target = X_swTrain[y_train==1]
                non_target = X_swTrain[y_train==0]

                test = epochs_test.copy().resample(20).get_data()*1e6
                X_swTest= np.zeros((weights_resampled.shape[0], test.shape[0]))
                for i, feature in enumerate(weights_resampled):
                    X_swTest[i, :] = test[:, feature[0].astype(int), feature[1].astype(int)]# * feature[3]

                X_swTest=X_swTest.swapaxes(0,1)
                X_swTest = scaler.transform(X_swTest)

                distance_to_target = np.array([np.linalg.norm(X_swTest[i] - target.mean(axis=0)) for i in range(X_swTest.shape[0])])
                distance_to_nontarget = np.array([np.linalg.norm(X_swTest[i] - non_target.mean(axis=0)) for i in range(X_swTest.shape[0])])
                result = np.array((distance_to_nontarget / (distance_to_target + distance_to_nontarget),
                                   distance_to_target / (distance_to_target + distance_to_nontarget))).swapaxes(0, 1)


                test_t = X_swTest - target.mean(axis=0)
                test_nt = X_swTest - non_target.mean(axis=0)
                #test_t_weighted = test_t * weights_resampled[:, 3]
                #test_nt_weighted = test_nt * weights_resampled[:, 3]
                test_t_results = test_t.sum(axis=1)
                test_nt_results = test_nt.sum(axis=1)
                result = np.array((test_nt_results / (test_t_results + test_nt_results),test_t_results / (test_t_results + test_nt_results))).swapaxes(0,1)
                ac_swlda = dp.evaluate_independent_epochs(result, epoch_info_test)


                swlda = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None,
                                                   store_covariance=None, tol=0.0001, covariance_estimator=None)
                swlda.fit(swlda_train.swapaxes(0, 1), y_train)
                # swlda.coef_ = weights_resampled[:, 3].reshape(1, 17)
                swlda_y_pred_prob = swlda.predict_proba(
                    swlda_result.swapaxes(0, 1))  # returns list : [[nontarget,target],[nt,t],[nt,t]...]
                ac_swlda = dp.evaluate_independent_epochs(swlda_y_pred_prob, epoch_info_test)

                ac_swlda = py3gui.test_weights(test, y_test, weights_resampled, [1,6], 8)
            # data = epochs_train.copy().get_data(picks=channels)
                # X_train_swlda = np.zeros((data.shape[0],weights.shape[0]))
                # start_idx = 0
                # for channel  in np.unique(weights[:,0]):
                #     weight = weights[weights[:, 0] == channel][:,[1,3]]
                #     stop_idx = start_idx + weight.shape[0]
                #     X_train_swlda[:,start_idx:stop_idx] = data[:,1,weight[:,0].astype(int)] *weight[:,1]
                #     start_idx = stop_idx




                # # trying to load feature weights into lda by scikit learn
                # swlda = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None,
                #                                  store_covariance=None, tol=0.0001, covariance_estimator=None)
                # swlda.classes_ = np.array((0,1))
                # swlda.coef_ = weights_resampled[:, 3].reshape(1, 17)
                # #explained_variance_ratio = 1 ? vermutlich falsch
                # swlda.means_ = np.array((target.mean(axis=0), non_target.mean(axis=0)))
                # swlda._max_components = 17
                # swlda.n_features_in_ = 17
                # #priors =None
                # #priors_ = [0.8,0,2] ? vermutlich auch falsch
                # #estimate priors from sample:
                # _, y_t = np.unique(y_train, return_inverse=True)  # non-negative ints
                # swlda.priors_ = np.bincount(y_t) / float(len(y_train))
                # swlda.scalings_ = (17,1)
                # #shrinkage = None
                # #solver = svd
                # #store_covariance = None
                # #tol =0.0001
                # swlda._solve_svd(swlda_train.swapaxes(0, 1),y_train)
                # coef = np.dot(swlda.means_ - swlda.xbar_, swlda.scalings_)
                # swlda.intercept_ = -0.5 * np.sum(coef ** 2, axis=1) + np.log(swlda.priors_)
                # swlda.intercept_ -= np.dot(swlda.xbar_, swlda.coef_.T)
                #
                #
                # swlda_prob_pred = swlda.predict_proba(swlda_result.swapaxes(0, 1))
                #
                #
                # result = swlda_result.sum(axis=0)
                # #result = result >0
                # ac_swlda = dp.evaluate_independent_epochs(np.array((result)), epoch_info_test)
                # result = result[result == y_test]
                # print(result.shape[0]/y_test.shape[0])
                #
                #
                #
                # # mean_of_targets =  X_train_swlda[y_train == 1].sum(axis=0)
                # # mean_of_nontargets = X_train_swlda[y_train == 0].sum(axis=0)
                # data_test = epochs_test.copy().get_data()*1e6
                # data_test = epochs_test.copy().get_data(picks=channels)
                # X_test_swlda = np.zeros((data_test.shape[0],weights.shape[0]))
                # start_idx = 0
                # for channel  in np.unique(weights[:,0]):
                #     weight = weights[weights[:, 0] == channel][:,[1,3]]
                #     stop_idx = start_idx + weight.shape[0]
                #     X_test_swlda[:,start_idx:stop_idx] = data_test[:,1,weight[:,0].astype(int)] *weight[:,1]
                #     start_idx = stop_idx
                #
                # test = np.array((X_test_swlda.sum(axis=1)> 0,X_test_swlda.sum(axis=1)< 0))
                # ac_swlda = dp.evaluate_independent_epochs(test.swapaxes(0,1), epoch_info_test)
                #
                # ac_swlda = y_test[y_test==test].sum()/len(y_test)
                # print("The Accuracy with the py3gui " + swlda_name + " is: {}".format(ac_swlda))

                #create full matrix from sparse matrix
                # full_weight_matrix = np.zeros((data_test.shape[1], data_test.shape[2]))
                # full_weight_matrix[weights[:, 1].astype(int), weights[:, 0].astype(int)] = weights[:,3]
                #
                # train = data_train * full_weight_matrix
                # test = data_test * full_weight_matrix
                #
                #
                #
                # train = train.reshape(train.shape[0], train.shape[1] * train.shape[2])
                # test = test.reshape(test.shape[0], test.shape[1] * test.shape[2])
                # ac_swlda = py3gui.test_weights(data_test, y_test, full_weight_matrix, [6,6], 8)
                # print("The Accuracy with the py3gui " + swlda_name + " is: {}".format(ac_swlda))
                # prediction0 = np.zeros((2,X_test_swlda.shape[0]))
                # prediction0[0]= [distance.euclidean(X_test_swlda[i],mean_of_targets) for i in range(X_test_swlda.shape[0])]
                # prediction0[1] = [distance.euclidean(X_test_swlda[i],mean_of_nontargets) for i in range(X_test_swlda.shape[0])]
                #
                # prediction1 = np.zeros((2, X_test_swlda.shape[0]))
                # prediction1[0] = [np.linalg.norm(X_test_swlda[i]-mean_of_targets) for i in range(X_test_swlda.shape[0])]
                # prediction1[1] = [np.linalg.norm(X_test_swlda[i]- mean_of_nontargets) for i in range(X_test_swlda.shape[0])]
                #
                #
                #
                #
                #
                #
                # swlda_y_pred_prob = np.array((X_test_swlda.sum(axis=1)<0,X_test_swlda.sum(axis=1)>0)).swapaxes(0,1)

                # testi = np.array((train.sum(axis=1)>0,train.sum(axis=1)<0))
                # ac_swlda = dp.evaluate_independent_epochs(testi, epoch_info_test)
                #
                # lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None , priors=None, n_components=None, store_covariance=None, tol=0.0001, covariance_estimator=None)
                # lda.fit(train,y_train)
                # swlda_y_pred_prob = lda.predict_proba(test) #returns list : [[nontarget,target],[nt,t],[nt,t]...]
                # ac_swlda = dp.evaluate_independent_epochs(swlda_y_pred_prob,epoch_info_test)
                # # temp_df = dp.results_template(ac_swlda,swlda_name,sess)
                # # classifier_results =classifier_results.append(temp_df,ignore_index=True)
                # #
                # # roc_values[lda_name].append(metrics.roc_curve(y_test,swlda_y_pred_prob[:,1]))
                # print("The Accuracy with the features put into the lda" + lda_name + " is: {}".format(ac_swlda))



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

