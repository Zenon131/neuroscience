import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv(r"C:\Users\jonoz\Downloads\emotions.csv\emotions.csv")

sample = data.loc[0, 'fft_0_b':'fft_749_b']

