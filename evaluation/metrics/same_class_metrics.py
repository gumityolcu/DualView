import torch
from utils.data import GroupLabelDataset
from utils import Metric


class SameClassMetric(Metric):
    name = "SameClassMetric"

    def __init__(self, train, test, device="cuda"):
        if isinstance(train.targets,list):
            train.targets=torch.tensor(train.targets,device=device)
        self.train_labels = train.targets.to(device)
        self.test_labels = test.test_targets.to(device)
        self.scores = torch.empty(0, dtype=torch.float, device=device)
        self.device = device

    def __call__(self, xpl, start_index):
        xpl.to(self.device)
        most_influential_labels = self.train_labels[xpl.argmax(axis=-1)]
        test_labels = self.test_labels[start_index:start_index + xpl.shape[0]]
        is_equal = (test_labels == most_influential_labels) * 1.
        self.scores = torch.cat((self.scores, is_equal), dim=0)

    def get_result(self, dir=None, file_name=None):
        self.scores = self.scores.to('cpu').numpy()
        score = self.scores.sum() / self.scores.shape[0]
        resdict = {'metric': self.name, 'all_scores': self.scores, 'avg_score': score,
                   'num_examples': self.scores.shape[0]}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict


class SameSubclassMetric(SameClassMetric):
    name = "SameSubclassMetric"

    def __init__(self, train, test, device="cuda"):
        assert isinstance(train, GroupLabelDataset)
        assert isinstance(test, GroupLabelDataset)
        super().__init__(train.dataset, test.dataset, device)
