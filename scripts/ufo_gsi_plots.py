import argparse
import xarray as xr
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import glob
import os
from sklearn.linear_model import LinearRegression

def _get_linear_regression(data1, data2):
    """
    Inputs:
        data1 : data on the x axis
        data2 : data on the y axis
    """

    x = np.array(data1).reshape((-1,1))
    y = np.array(data2)
    
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x,y)
    
    intercept = model.intercept_
    slope = model.coef_[0]
    
    # This is the same as if you calculated y_pred
    # by y_pred = slope * x + intercept
    y_pred = model.predict(x)
    
    return y_pred, r_sq, intercept, slope

def _get_channels(file):
    
    with Dataset(file, mode='r') as f:
        # Grab channels
        variables = list(f.variables.keys())
        channels = sorted([int(x.split('_')[-1].split('@')[0]) for x in variables if x.split('@')[-1] == 'ObsValue'])
        
    return channels

def plot_scatter(df, metadata):
    
    ## Scatter Plot ##
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    plt.scatter(df['ufo'], df['gsi'], s=15, color='darkgray', label='All Data')
    plt.scatter(df['qc_ufo'], df['qc_gsi'], s=15, color='dimgrey', label='QC Data')

    # Plot Regression line
    y_pred, r_sq, intercept, slope = _get_linear_regression(df['ufo'], df['gsi'])
    label = f'Estimated Regression line\ny = {slope:.4f}x + {intercept:.4f}\nR\u00b2 : {r_sq:.4f}'
    plt.plot(df['ufo'], y_pred, color='blue', linewidth=1, label=label)

    # Plot QC data Regression line
    y_pred, r_sq, intercept, slope = _get_linear_regression(df['qc_ufo'].dropna(),
                                                            df['qc_gsi'].dropna())
    label = f'Estimated Regression line - QC\ny = {slope:.4f}x + {intercept:.4f}\nR\u00b2 : {r_sq:.4f}'
    plt.plot(df['qc_ufo'].dropna(), y_pred, color='red', linewidth=1, label=label)

    plt.legend(loc='upper left', fontsize=11)

    plt.grid(linewidth=0.5, color='gray', linestyle='--')
    plt.title('{sensor} {satellite} - H(x)\nAll Channels'.format(**metadata),
              loc='left', fontsize=12)
    plt.title('{cycle}'.format(**metadata), loc='right', fontweight='semibold')

    plt.xlabel('UFO H(x)', fontsize=12)
    plt.ylabel('GSI H(x)', fontsize=12)
    
    save_filename = '{cycle}_{sensor}_{satellite}_All_Channels_scatter.png'.format(**metadata)
    
    plt.savefig(metadata['outdir']+save_filename, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')

    return

def plot_histogram(df, metadata):
    ## Histogram ##

    # Create Figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    stats = df['ufo-gsi'].describe()

    df['ufo-gsi'].plot.hist(ax=ax, bins=9)

    # Plots data mean with red line
    ax.axvline(stats[1], color='r',
                linestyle='solid', linewidth=1)

    text = f'n: {int(stats[0])}\nmean: {stats[1]:.3e}\nmin: {stats[3]:.3e}\nmax: {stats[-1]:.3e}\nstd. dev: {stats[2]:.3e}'
    ax.text(0.04, 0.74, text, fontsize=12,
                bbox=dict(boxstyle="round,pad=.5",
                          facecolor="white",
                          edgecolor="grey",
                          linewidth=.75),
               transform=ax.transAxes)

    plt.grid(linewidth=0.5, color='gray', linestyle='--')
    plt.title('{sensor} {satellite} - H(x)\nAll Channels'.format(**metadata),
              loc='left', fontsize=12)
    plt.title('{cycle}'.format(**metadata), loc='right', fontweight='semibold')
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('UFO H(x) - GSI H(x)', fontsize=12)
    
    save_filename = '{cycle}_{sensor}_{satellite}_All_Channels_histogram.png'.format(**metadata)
    
    plt.savefig(metadata['outdir']+save_filename, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')
    
    return

def plot_obs_count(ufo_data, ufo_qc_data,
                   gsi_data, gsi_qc_data,
                   metadata):
        
    x = np.arange(len(metadata['channels']))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(15,5))
    plt.grid(linewidth=0.5, color='gray', linestyle='--')
    
    rects1 = ax.bar(x - 1.5*width, ufo_data, width, label='All UFO Data')
    rects2 = ax.bar(x - width/2, gsi_data, width, label='All GSI Data')
    rects3 = ax.bar(x + width/2, ufo_qc_data, width, label='UFO Assimilated Data')
    rects4 = ax.bar(x + 1.5*width, gsi_qc_data, width, label='GSI Assimilated Data')
    
    plt.title('{sensor} {satellite} - H(x)'.format(**metadata),
              loc='left', fontsize=12)
    plt.title('{cycle}'.format(**metadata), loc='right', fontweight='semibold')

    plt.xlabel('Channel', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metadata['channels'])
    plt.ylabel('Count', fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', )
    
    save_filename = '{cycle}_{sensor}_{satellite}_nobs.png'.format(**metadata)
    
    plt.savefig(metadata['outdir']+save_filename, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')    
    
    return
    

def get_data_df(file, channels):
    
    ds = xr.open_dataset(file)
    df = ds.to_dataframe()

    #Get Column names by looping through channels
    columns = []

    for i in channels:
        columns.append(f'brightness_temperature_{i}@hofx')
        columns.append(f'brightness_temperature_{i}@EffectiveQC')
        columns.append(f'brightness_temperature_{i}@ObsBias')
        columns.append(f'brightness_temperature_{i}@GsiHofXBc')
        columns.append(f'brightness_temperature_{i}@PreQC')
        columns.append(f'brightness_temperature_{i}@GsiFinalObsError')
        
    # Create dataframe from appropriate column names and indexing
    idx = pd.IndexSlice
    data_df = df.loc[idx[:, 0], columns].reset_index()
          
    return data_df

def generate_figs(inpath, outpath):
    
    obsfiles = glob.glob(inpath+'*')
    obsfiles.sort()

    for file in obsfiles:
        # Get channels
        channels = _get_channels(file)
        
        # Get metadata
        filename = file.split('/')[-1]
        sensor = filename.split('_')[0]
        satellite = filename.split('_')[1]
        cycle = filename.split('_')[3]

        metadata = {'sensor': sensor,
                    'satellite': satellite,
                    'cycle': cycle,
                    'channels': channels,
                    'outdir': './'}

        data_df = get_data_df(file, channels)
        
        # Create Dataframe appropriate for plotting data
        plot_df = pd.DataFrame()
        
        # Get channel counts for bar graph
        ufo_obs_count = []
        ufo_obs_count_qc = []
        gsi_obs_count = []
        gsi_obs_count_qc = []

        for i in channels:
            ufo = data_df[f'brightness_temperature_{i}@hofx'] + data_df[f'brightness_temperature_{i}@ObsBias']
            gsi = data_df[f'brightness_temperature_{i}@GsiHofXBc']

            qc_df = data_df.loc[data_df[f'brightness_temperature_{i}@EffectiveQC'] == 0]

            qc_ufo = qc_df[f'brightness_temperature_{i}@hofx'] + qc_df[f'brightness_temperature_{i}@ObsBias']
            qc_gsi = qc_df[f'brightness_temperature_{i}@GsiHofXBc']
    
            ufo_obs_count.append(len(ufo))
            ufo_obs_count_qc.append(len(qc_ufo))
            gsi_obs_count.append(len(gsi))
            gsi_obs_count_qc.append(len(qc_gsi))
            
            plot_df = plot_df.append(pd.DataFrame({'ufo': ufo, 'gsi': gsi, 'ufo-gsi': ufo-gsi,
                                                   'qc_ufo': qc_ufo, 'qc_gsi': qc_ufo,
                                                   'qc_ufo-qc_gsi': qc_ufo-qc_gsi}))
            
            
        plot_scatter(plot_df, metadata)
        
        plot_histogram(plot_df, metadata)
        
        plot_obs_count(ufo_obs_count, ufo_obs_count_qc,
                       gsi_obs_count, gsi_obs_count_qc,
                       metadata)
        
        

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--diagdir', help='path to UFO netCDF diags', required=True)
    ap.add_argument('-o', '--output', help="path to output directory", default="./")
    MyArgs = ap.parse_args()
    generate_figs(MyArgs.diagdir, MyArgs.output)
