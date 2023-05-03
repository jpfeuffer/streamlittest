import streamlit as st
import numpy as np
import pandas as pd

import datetime
import xarray as xr
import datashader as ds
import datashader.transfer_functions as tf
from collections import OrderedDict

from holoviews.operation.datashader import datashade
from holoviews.streams import PlotSize

import hvplot
import hvplot.pandas

import holoviews as hv

st.title('🎈 App Name')

st.write('Hello world!')

# Constants
np.random.seed(2)
n = 100000                               # Number of points
cols = list('abcdefg')                   # Column names of samples
start = datetime.datetime(2010, 10, 1, 0)   # Start time

# Generate a fake signal
signal = np.random.normal(0, 0.3, size=n).cumsum() + 50

# Generate many noisy samples from the signal
noise = lambda var, bias, n: np.random.normal(bias, var, n)
data = {c: signal + noise(1, 10*(np.random.random() - 0.5), n) for c in cols}

# Add some "rogue lines" that differ from the rest 
cols += ['x'] ; data['x'] = signal + np.random.normal(0, 0.02, size=n).cumsum() # Gradually diverges
cols += ['y'] ; data['y'] = signal + noise(1, 20*(np.random.random() - 0.5), n) # Much noisier
cols += ['z'] ; data['z'] = signal # No noise at all

# Pick a few samples from the first line and really blow them out
locs = np.random.choice(n, 10)
data['a'][locs] *= 2

# Create a dataframe
data['Time'] = [start + datetime.timedelta(minutes=1)*i for i in range(n)]

df = pd.DataFrame(data)
df['ITime'] = pd.to_datetime(df['Time']).astype('int64')
# Default plot ranges:
x_range = (df.iloc[0].ITime, df.iloc[-1].ITime)
y_range = (1.2*signal.min(), 1.2*signal.max())

PlotSize.scale=2
hv.extension('bokeh')

opts = hv.opts.RGB(width=600, height=300)
ndoverlay = hv.NdOverlay({c:hv.Curve((df['Time'], df[c]), kdims=['Time'], vdims=['Value']) for c in cols})
st.write(hv.render(datashade(ndoverlay, cnorm='linear', aggregator=ds.count(), line_width=2).opts(opts), backend='bokeh'))


# create sample data
@st.cache
def get_data():
    return pd.DataFrame(data=np.random.normal(size=[50, 2]), columns=['col1', 'col2'])

df = get_data()

# streamlit plotting works
st.line_chart(df)

# creating a holoviews plot
nice_plot = df.hvplot(kind='scatter')

st.write(hv.render(nice_plot, backend='bokeh'))
