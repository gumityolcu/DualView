from .pytorch_influence_functions import batch_interface as interface
from utils.explainers import Explainer
from torch.utils.data import DataLoader
from torch import any


class InfluenceFunctionExplainer(Explainer):
    name="InfluenceFunctionExplainer"
    def __init__(self, model, dataset, device, recursion_depth, r):
        super(InfluenceFunctionExplainer, self).__init__(model, dataset, device)
        self.dataset=dataset
        self.recursion_depth=recursion_depth
        self.r=r

    def train(self):
        return 0.

    def explain(self,x, preds=None, targets=None):
        gpu = -1 if self.device=="cpu" else 0
        xpl=interface(self.model, DataLoader(self.dataset, batch_size=1,shuffle=False),x,preds,gpu,self.recursion_depth,self.r)
        if any(xpl.isnan()):
            print("ISNAN!")
        xpl = -xpl # the external library returns Influence as the "change in loss as the result of an infinitesimal UPWEIGHTING" as defined in the paper. We need to multiply with -1 because we want to downweight training datapoints
        return xpl.to(self.device)