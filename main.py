#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:27:09 2019

@author: danielrangel
"""

import pandas as pd
from surprise import Dataset
from surprise import Reader


reader = Reader(rating_scale=(1, 5), sep=',')
data = Dataset.load_from_file('data.txt', reader)

from surprise import KNNWithMeans

# Using user based cosine similarity
sim_options = {
    "name": "cosine",
    "user_based": True,
}
algorithm = KNNWithMeans(sim_options=sim_options)




trainingSet = data.build_full_trainset()

algorithm.fit(trainingSet)


prediction = algorithm.predict('1', 150)
prediction.est
