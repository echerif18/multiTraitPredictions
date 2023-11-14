import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

import rasterio
import multiprocessing
import time


from pickle import dump,load
from tensorflow.keras.models import model_from_json

Traits = ['LMA (g/m²)', 'N content (mg/cm²)', 'LAI (m²/m²)', 'C content (mg/cm²)', 'Chl content (μg/cm²)', 'EWT (mg/cm²)', 
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
'Manganese content (mg/cm²)']


### Apply savgol filter for a wavelength filter, 
def filter_segment(features_noWtab, order=1,der= False):
    #features_noWtab: Segment of the signal
    #order: Order of the savgol filter
    #der: If with first derivative
    
#     part1 = features_noWtab.loc[:,indx]
    part1 = features_noWtab.copy()
    if (der):
        fr1 = savgol_filter(part1, 65, 1,deriv=1)
    else:
        fr1 = savgol_filter(part1, 65, order)
    fr1 = pd.DataFrame(data=fr1, columns=part1.columns)
    return fr1


###### transformation methods #######
def feature_preparation(features, inval = [1351,1431, 1801, 2051], frmax=2451, order=1,der= False):
    # features: The original reflectance signal
    #order: Order of the savgol filter
    #der: If with first derivative
    
    other = features.copy()
    other.columns = other.columns.astype('int')
    other[other<0] = np.nan
    
    #####Substitute high values with the mean of neighbour values
    other[other>1] = np.nan
    other = (other.ffill() + other.bfill())/2
    other = other.interpolate(method='linear',limit_area=None, axis=1, limit_direction='both')
    
    wt_ab = [i for i in range(inval[0],inval[1])]+[i for i in range(inval[2],inval[3])]+[i for i in range(2451,2501)] 

    features_Wtab = other.loc[:,wt_ab]
    features_noWtab=other.drop(wt_ab,axis=1)
    
    fr1 = filter_segment(features_noWtab.loc[:,:inval[0]-1], order = order, der = der)
    fr2 = filter_segment(features_noWtab.loc[:,inval[1]:inval[2]-1], order = order,der = der)
    fr3 = filter_segment(features_noWtab.loc[:,inval[3]:frmax], order = order,der = der)    
    
    
    inter = pd.concat([fr1,fr2,fr3], axis=1, join='inner')
    inter[inter<0]=0
    
    return inter

## Plot spectra signal for all saomles 
def plot_fig(features, save=False, file=None, figsize=(15, 10)):
    # features: The original reflectance signal
    
    plt.figure(figsize=figsize)
    plt.plot(features.T)
    plt.ylim(0, features.max().max())
    if (save):
        plt.savefig(file + '.pdf', bbox_inches = 'tight', dpi = 1000)
        plt.savefig(file + '.svg', bbox_inches = 'tight', dpi = 1000)
    plt.show()

###################
######## Load imagery methods + Vis ######

####### For Enmap data ###
def image_processing(enmap_im_path, bands_path):
    bands = pd.read_csv((bands_path))['bands'].astype(float)
    src = rasterio.open(enmap_im_path)
    array = src.read()
    sp_px = np.stack([array[i].reshape(-1,1) for i in range(array.shape[0])],axis=0)
    sp_px = np.swapaxes(sp_px.mean(axis=2),0,1) #transpose
    
    assert (sp_px.shape[1] == bands.shape[0]), "The number of bands is not correct. Check the number of spectral bands in the imagery!"
    
    df = pd.DataFrame(sp_px, columns = bands.to_list())
    df[df< df.quantile(0.01).min()+10] = np.nan ## eliminate corrupted pixels and replace with nan
    
    idx_null = df[df.T.isna().all()].index
    return src, df, idx_null

def process_dataframe(veg_spec):
    start_t = time.perf_counter()
    veg_reindex = veg_spec.reindex(columns = sorted(veg_spec.columns.tolist() + [i for i in range(400,2501) if i not in veg_spec.columns.tolist()]))#.interpolate(method='linear',limit_area=None, axis=1, limit_direction='both')

    veg_reindex = veg_reindex/10000
    veg_reindex.columns = veg_reindex.columns.astype(int)
    inter = veg_reindex.loc[:,~veg_reindex.columns.duplicated()] ## remove column duplicates 

    inter = feature_preparation(veg_reindex, order=1)
    ############ Remove duplicated columns #######
    inter = inter.loc[:,~inter.columns.duplicated()] ## remove column duplicates 
    
    return inter.loc[:,400:]

####### Prepare fro multi-processing ##
def transform_data(df):
    # Define the number of CPUs to use
    num_cpus = multiprocessing.cpu_count()
    # Create a multiprocessing pool with the specified number of CPUs
    pool = multiprocessing.Pool(num_cpus)
    # Split the DataFrame into chunks to be processed in parallel
    df_chunks = [chunk for chunk in np.array_split(df, num_cpus)]

    start_t = time.perf_counter()

    print("starting processing")
    with multiprocessing.Pool(num_cpus) as pool:
        results = pool.map(process_dataframe, df_chunks)
        pool.close()
        pool.join()

    end_t = time.perf_counter()
    total_duration = end_t - start_t
    print(f"Image transformation took {total_duration:.2f}s total") 
    
    df_transformed = pd.concat(results).reset_index(drop=True)
    
    return df_transformed


def load_model(dir_data, gp = None):
    if(gp is not None):
        json_file = open(dir_data + 'Model_db{}.json'.format(gp), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        best_model = model_from_json(loaded_model_json)

        scaler_list = load(open(dir_data + 'scaler_db{}.pkl'.format(gp), 'rb'))

        # load weights into new model
        best_model.load_weights(dir_data + 'Trial_db{}_weights.h5'.format(gp))
    else:
        json_file = open(dir_data + 'Model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        best_model = model_from_json(loaded_model_json)

        scaler_list = load(open(dir_data+ 'scaler_global.pkl', 'rb'))

        # load weights into new model
        best_model.load_weights(dir_data+ 'Trial_weights.h5')
    
    return best_model, scaler_list