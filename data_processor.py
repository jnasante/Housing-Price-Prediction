import matplotlib.pyplot as plot
import matplotlib.legend_handler as legend_handler
import pandas as pd
import numpy as np

# Suppress warnings we want to ignore
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.options.mode.chained_assignment = None  # default='warn'

# data dictionary to hold all the values we read in, drop month and day
data = pd.read_csv('kc_house_data.csv')

print(data.shape[0])




