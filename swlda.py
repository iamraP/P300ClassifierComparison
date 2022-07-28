import numpy as np
import numpy as np
from stepwise import stepwisefit
from scipy import stats

class swlda:
    def __init__(self, response_window = [0,800],decimation_frequency:int = 20,max_model_features: int = 60, penter: float = 0.1,premove:float = 0.15):
        self.response_window = response_window
        self.decimation_frequency = decimation_frequency # is actually never used, sinde the resampling happens before entering the data
        self.max_model_features = max_model_features
        self.penter = penter
        self.premove = premove

    def fit(self,responses, type):
       # trials, samples, channels = responses.shape
        if not isinstance(responses,np.ndarray):
            responses = responses.to_numpy()
        type = type.astype(bool)
        target = type.nonzero()[0]
        nontarget = (~type).nonzero()[0]
        target_then_nontarget = np.concatenate((target, nontarget))
        sorted = responses[target_then_nontarget]
        labels = type[target_then_nontarget] * 2 - 1
        b, se, pval, inmodel, stats, nextstep, history = stepwisefit(
            sorted, labels, maxiter=self.max_model_features,
            penter=self.penter, premove=self.premove
        )
        if not inmodel.any():
            swlda.weights = None
            return 'Could not find an appropriate model.'


        self.weights = b * 10 / abs(b).max() # scale values between -10 and 10
        self.weights = self.weights.flatten() * inmodel

        weighted_features = responses.dot(self.weights)

       # #get full response
       #  restored_weights = np.repeat(self.weights,round(full_responses.shape[1]/responses.shape[1]))
       #  self.weights = restored_weights[:full_responses.shape[1]]
       #  weighted_features = full_responses.dot(self.weights)


        target = weighted_features[type==1]
        nontarget = weighted_features[type==0]
        self.target_std = target.std()
        self.target_mean = target.mean()
        self.nontarget_std = nontarget.std()
        self.nontarget_mean = nontarget.mean()


    def predict_proba(self,responses):
        if not isinstance(responses,np.ndarray):
            responses = responses.to_numpy()
        weighted_features = responses.dot(self.weights)

        # PDF for Target/non targets

        P_X_tar = stats.norm.pdf(weighted_features, loc=self.target_mean,
                                 scale=self.target_std)  # probability to have feature, given target
        P_X_ntar = stats.norm.pdf(weighted_features, loc=self.nontarget_mean,
                                  scale=self.nontarget_std)  # probability to have feature, given nontarget

        # Bayes for prediction
        #TODO find out if you should use 1/6 or 1/2 for target probability and find out if it makes sense to set NaN values 0

        P_tar_X = (P_X_tar * (1 / 6)) / (P_X_tar * (1 /6) + P_X_ntar * (5 / 6)) # probability of target, given the feature set
        P_tar_X = np.nan_to_num(P_tar_X)
        return np.squeeze(np.array([1 - P_tar_X, P_tar_X])).swapaxes(0, 1) #return probability in the same way as sklearn