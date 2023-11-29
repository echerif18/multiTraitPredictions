import gradio as gr
import os
from fun_module import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker

import numpy as np
import pandas as pd

import time
import multiprocessing

import rasterio
from rasterio.plot import show

import warnings
warnings.filterwarnings("ignore")

Traits = [
'LMA (g/m²)', 
'N content (mg/cm²)', 'LAI (m²/m²)', 'C content (mg/cm²)', 'Chl content (μg/cm²)', 'EWT (mg/cm²)', 
'Carotenoid content (μg/cm²)', 'Phosphorus content (mg/cm²)', 'Lignin (mg/cm²)', 'Cellulose (mg/cm²)', 
'Fiber (mg/cm²)',
'Anthocyanin content (μg/cm²)',
'NSC (mg/cm²)',
'Magnesium content (mg/cm²)',
'Ca content (mg/cm²)',
'Potassium content (mg/cm²)',
'Boron content (mg/cm²)',
'Copper content (mg/cm²)',
'Sulfur content (mg/cm²)',
'Manganese content (mg/cm²)'
]


########## Dir to the trained model ###
path = os.getcwd()
path_model = os.path.join(path, 'models/')
best_model, scaler_list = load_model(path_model)


# def display(tr,src, preds):
#   # Initialize the tr variable
#   # tr = 0

#   # Create a figure and axis
#   plt.rc('font', size=3)
#   fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(3, 2), dpi=300, sharex=True, sharey=True, gridspec_kw={'width_ratios': [1, 1.09]})

#   extent = src.bounds

#   nir = src.read(72)/10000
#   red = src.read(47)/10000
#   green = src.read(28)/10000
#   blue = src.read(6)/10000

#   # Stack bands
#   nrg = np.dstack((nir, red, green)) ## False color

#   ori_im = ax1.imshow(nrg, extent=extent, aspect='auto')

#   preds_tr = pd.DataFrame(np.array(preds.loc[:, Traits[tr]]).reshape(src.shape[0], src.shape[1]))  #pd.DataFrame(np.transpose(test.read(tr+1)))
#   preds_vis = preds_tr.copy()[preds_tr<preds_tr.quantile(0.99)]
#   flag = np.array(preds_vis)
#   maxv = pd.DataFrame(flag).max().max()
#   minv = pd.DataFrame(flag).min().min()

#   pred_im = ax2.imshow(np.array(preds_tr), vmin=minv, vmax=maxv, extent=extent,  aspect='auto')

#   # Add colorbar to the second subplot
#   cbar = plt.colorbar(pred_im, ax=ax2, fraction=0.046, pad=0.04)  # Adjust fraction and pad as needed

#   ax1.set(title='Original scene (False Color)')
#   ax2.set(title='{} map'.format(Traits[tr]))

#   ax1.set_aspect('equal')
#   ax2.set_aspect('equal')

#   ax1.set_axis_off
#   ax2.set_axis_off

#   ax1.xaxis.set_major_locator(ticker.NullLocator())
  # ax1.yaxis.set_major_locator(ticker.NullLocator())
  # return fig


# Function to apply the vector regression model with both image and CSV data
def apply_regression(input_image, input_csv, display_channel=0):
    
    src, df, idx_null = image_processing(input_image, input_csv)
    df_transformed = transform_data(df)

    tf_preds = scaler_list.inverse_transform(best_model.predict(df_transformed, verbose=1, batch_size=128)) #df_transformed
    preds = pd.DataFrame(tf_preds) #, columns=Traits
    preds.loc[idx_null] = np.nan

    # Apply the trained model to the combined input
    preds_tr = pd.DataFrame(np.array(preds.loc[:, display_channel]).reshape(src.shape[0], src.shape[1]))

    preds_vis = preds_tr.copy()[preds_tr<preds_tr.quantile(0.99)]
    flag = np.array(preds_vis)
    maxv = pd.DataFrame(flag).max().max()
    minv = pd.DataFrame(flag).min().min()

    plt.rc('font', size=5)
    fig = plt.figure()

    plt.imshow(preds_tr, aspect='auto')
    plt.colorbar(fraction=0.046, pad=0.04)  # Adjust fraction and pad as needed
    plt.title('{} map'.format('test'))
    plt.show()

    return fig


# Gradio Interface
iface = gr.Interface(
    fn=apply_regression,
    inputs=[
        gr.File(type="filepath", label="Raster"),#gr.Image(type="pil"), #, preprocess=image_processing
        gr.File(type="filepath", label="CSV Metadata"), #
        gr.Number(label="Display Channel")
    ],
    outputs= gr.Plot(),#"text"
    live=True,
    title="Multi-trait from HSI App",
    description="Upload a Hyperspectral scene with a corresponding .CSV for the available bands, and the app will generate the trait prediction.",
)

# Launch the Gradio app
iface.launch()





