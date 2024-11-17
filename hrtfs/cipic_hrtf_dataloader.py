import os
import glob
import torch
import numpy as np
import re
import soundfile as sf
from typing import Tuple, Dict, Any, List, Set
import random
from hrtfs.cipic_db import CipicDatabase 

class Sample:
    def __init__(self, Subject, index, laterality):
        self.subject = Subject
        self.idx = index
        self.side = laterality
    
    def __hash__(self):
        return hash((hash(self.subject), self.idx, self.side))

    def getX(self):
        azimuth, elevation = self.subject.getSphericalPositionsFromIndex(self.idx)
        anthroData = self.subject.collateAnthroData(ear=self.side)
        anthroData = np.append(anthroData, azimuth)
        anthroData = np.append(anthroData, elevation)
        return anthroData

    def getY(self):
        if self.side == "Right":
            channel = 0
        else:
            assert self.side == "Left"
            channel = 1
        return self.subject.getHRIRFromIndex(self.idx, channel)

class CipicHRTFs:

    def __init__(self, allSamples: List[Sample] = [], subsetIndices: Set[int] = set()) -> None:
        if len(allSamples) == 0:
            print("Cannot work with empty sample list")
            assert(False)
            return
        self.samples = []
        for idx in subsetIndices:
            self.samples.append(allSamples[idx])

    def __getitem__(self, n: int) -> Tuple[np.ndarray,
                                           Dict[str, Any],
                                           int]:
        """Gets the nth sample from the dataset.

        Parameters
        ----------
        n : int
            Index of the dataset sample.

        Returns
        -------
        np.ndarray
            X data.
        np.ndarray
            Y data.
        Dict
            Sample metadata.
        int
            n, the index
        """
        mySample = self.samples[n]
        return mySample.getX(), mySample.getY(), n

    def __len__(self) -> int:
        """Length of the dataset.
        """
        return len(self.samples)

    def collate_fn(self, batch):
        x, y = [], []

        indices = torch.IntTensor([s[2] for s in batch])

        for sample in batch:
            x += [torch.FloatTensor(sample[0])]
            y += [torch.FloatTensor(sample[1])]

        return torch.stack(x), torch.stack(y), indices
