import os

dir_path = os.getcwd()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" ### do not use the GPUs

from fun_module import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd

import time
import multiprocessing

import rasterio
from rasterio.plot import show

import warnings
warnings.filterwarnings("ignore")

import argparse

# Create the parser
my_parser = argparse.ArgumentParser(description='Predictions')

# Add the arguments
my_parser.add_argument('--routedata',
                       metavar='routed',
                       type=str, default= os.path.join(path, 'HS_img/enmap_toyExp.tif'),
                       help='Path to HS scene')

my_parser.add_argument('--routemeta',
                       metavar='routem',
                       type=str, default= os.path.join(path, 'HS_img/Enmap_bands.csv'),
                       help='Path to metadata (bands)')

my_parser.add_argument('--modelpath',
                       metavar='path',
                       type=str, default= os.path.join(path, 'models/'),
                       help='the path to transferability models')


my_parser.add_argument('--sceneText',
                       metavar='sceneText',
                       type=str,default='',
                       help='Label of the scene')



# Execute the parse_args() method
args = my_parser.parse_args()

enmap_im_path = args.routedata
bands_path = args.routemeta
path_model = args.modelpath  ## data path

sceneText = args.sceneText

###### We provide somw Toy examples of hyperspectral imagery on the repo ######
src, df, idx_null = image_processing(enmap_im_path, bands_path)
df_transformed = transform_data(df)


########## Load the trained model ###
best_model, scaler_list = load_model(path_model)

######### Model predictions ########
start_t = time.perf_counter()

print("starting predictions")

tf_preds = scaler_list.inverse_transform(best_model.predict(df_transformed, verbose=1, batch_size=64)) #df_transformed
preds = pd.DataFrame(tf_preds, columns=Traits)

end_t = time.perf_counter()
total_duration = end_t - start_t
print(f"etl took {total_duration:.2f}s total")

preds.loc[idx_null] = np.nan

########## Save the produced map as geotiff ###
size = (src.height, src.width, len(Traits))
crs = src.crs #target_crs
transform = src.transform #transform_params[0]
bounds = src.bounds #[[miny_tr[0], minx_tr[0]], [maxy_tr[0], maxx_tr[0]]]

new_image_path = os.path.join(dir_path, 'HS_img/{}_allTraits.tif'.format(sceneText))
new_image = rasterio.open(new_image_path, 'w',
                          driver='GTiff',
                          width=size[1],
                          height=size[0],
                          count=size[2],  # Change count for multiband
                          dtype='float32',  # Change dtype as per your data
                          crs=crs,
                          transform=transform,
                         bounds= bounds
                         )

for i in range(1,size[2]+1):
    # print(i)
    array_data = np.array(preds.loc[:,Traits[i-1]]).reshape((src.height, src.width))
    new_image.write(array_data,i)  # Change band index for multiband
    
# Close the new image
new_image.close()





