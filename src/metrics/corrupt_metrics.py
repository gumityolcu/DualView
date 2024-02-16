import matplotlib.pyplot as plt
import torch
import os
from utils import Metric


class CorruptLabelMetric(Metric):
    name = "CorruptLabelMetric"

    def __init__(self, train, test, coef_root=None, device="cuda"):
        self.corrupt_labels = train.corrupt_labels.to(device)
        self.corrupt_samples = train.corrupt_samples.to(device)
        self.scores = torch.zeros(len(train), dtype=torch.float, device=device)
        self.device = device
        self.coef_root = coef_root
        self.num_test_samples = 0
        self.ds_name = train.dataset.name

    def __call__(self, xpl, start_index):
        xpl.to(self.device)
        xpl = torch.abs(xpl)
        xpl = xpl / xpl.max(dim=-1)[0][:, None]  # Normalize each sample's explanations to have max 1.
        self.num_test_samples += xpl.shape[0]
        self.scores = self.scores + xpl.sum(dim=0)

    def compute_score(self, score_array):
        sorted_indices = torch.argsort(score_array, descending=True)
        detected = []
        det_count = 0
        for id in sorted_indices:
            if id in self.corrupt_samples:
                det_count += 1
            detected.append(det_count)
        detected = torch.tensor(detected, dtype=torch.float, device=self.device)
        detected = detected / self.corrupt_samples.shape[0]
        return detected.sum() / self.scores.shape[0], detected

    def add_coef_evaluation(self, resdict):
        max_score = resdict['max_score']
        file = [f for f in os.listdir(self.coef_root) if "coefs" in f]
        assert len(file) == 1
        file = file[0]
        fname = os.path.join(self.coef_root, file)
        coefs = torch.load(fname, map_location=torch.device(self.device))
        coefs = torch.abs(coefs).sum(axis=-1)
        score, curve = self.compute_score(coefs)
        resdict['coefs_auc_score'] = score / max_score
        resdict['coefs_label_flipping_curve'] = curve
        resdict['num_of_support_vectors'] = torch.sum(1 * (coefs != 0.))
        num_of_corrupt_support_vectors = 0
        for i, v in enumerate(coefs):
            if v != 0. and i in self.corrupt_samples:
                num_of_corrupt_support_vectors += 1
        resdict['num_of_corrupt_support_vectors'] = num_of_corrupt_support_vectors
        return resdict

    def get_result(self, dir, file_name):
        max_score = (self.corrupt_samples.shape[0] + 1) / 2 + self.scores.shape[0] - self.corrupt_samples.shape[0]
        max_score = max_score / self.scores.shape[0]
        score, curve = self.compute_score(self.scores)
        min_score = self.corrupt_samples.shape[0] / (2 * self.scores.shape[0])
        score = (score - min_score) / (max_score - min_score)
        resdict = {'metric': self.name, 'auc_score': score, 'label_flipping_curve': curve,
                   'num_examples': self.num_test_samples, 'num_corrupt_samples': self.corrupt_samples.shape[0],
                   'max_score': max_score}

        plt.figure()
        plt.plot(curve.to("cpu"))
        plt.savefig(f"{self.name}_corrupt_plot")
        if self.coef_root is not None:
            resdict = self.add_coef_evaluation(resdict)
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict
