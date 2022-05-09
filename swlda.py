import numpy as np
from dataclasses import dataclass,field
import numpy as np
from stepwise import stepwisefit
from scipy import stats

@dataclass
class swlda:
    response_window: list = field(default_factory=lambda: [0,800])
    decimation_frequency:int = 20
    max_model_features: int = 60
    penter: float = 0.1
    premove:float = 0.15


    def fit(self,responses, type):
       # trials, samples, channels = responses.shape
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
            return 'Could not find an appropriate model.'

        self.weights = b * 10 / abs(b).max() # why?

        weighted_features = responses.dot(self.weights)

        target = weighted_features[type==1]
        nontarget = weighted_features[type==0]
        self.target_std = target.std()
        self.target_mean = target.mean()
        self.nontarget_std = nontarget.std()
        self.nontarget_mean = nontarget.mean()


    def predict_proba(self,responses):
        responses = responses.to_numpy()
        weighted_features = responses.dot(self.weights)

        # PDF for Target/non targets

        P_X_tar = stats.norm.pdf(weighted_features, loc=self.target_mean,
                                 scale=self.target_std)  # probability to have feature, given target
        P_X_ntar = stats.norm.pdf(weighted_features, loc=self.nontarget_mean,
                                  scale=self.nontarget_std)  # probability to have feature, given nontarget

        # Bayes for prediction
        #TODO find out if you should use 1/6 or 1/2 for target probability and find out if it makes sense to set NaN values 0
        P_tar_X = (P_X_tar * (1 / 6)) / (P_X_tar * (1 / 6) + P_X_ntar * (5 / 6)) # probability of target, given the feature set
        P_tar_X = np.nan_to_num(P_tar_X)
        return np.array([1 - P_tar_X, P_tar_X]).swapaxes(0, 1).sum(axis=2) #return probability in the same way as numpy