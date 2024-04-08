import numpy as np
import sys

sys.path.append('.')
from reids.marketr101.models.model import MarketR101
from reids.duker101.models.model import DukeR101
from reids.msmtr101.models.model import MSMTR101


class ReIDEnsemble:

    def initialize(self, max_batch_size):
        MarketR101.initialize(max_batch_size=max_batch_size)
        DukeR101.initialize(max_batch_size=max_batch_size)
        MSMTR101.initialize(max_batch_size=max_batch_size)

        self.models = [
            MarketR101(),
            DukeR101(),
            MSMTR101()
        ]

    def run(self, img, dets, max_batch_size):
        appearance_features = []
        for model in self.models:
            feat = model.run(img, dets, max_batch_size)
            appearance_features.append(feat)
        mean_features = np.mean(appearance_features, axis=0)
        return mean_features

    def finalize(self):
        MarketR101.finalize()
        DukeR101.finalize()
        MSMTR101.finalize()