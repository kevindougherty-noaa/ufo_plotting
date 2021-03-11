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

def _create_scatter(x=None, y=None, qc_x=None, qc_y=None, \
                    plot_attributes=None):

    ## Scatter Plot ##
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # Checks to see if all data is included to plot
    if x is not None and y is not None:
        plt.scatter(x, y, s=15, color='darkgray', label= f'All Data: n={x.size}')
        y_pred, r_sq, intercept, slope = _get_linear_regression(x, y)
        label = f'Estimated Regression line\ny = {slope:.4f}x + {intercept:.4f}\nR\u00b2 : {r_sq:.4f}'
        plt.plot(x, y_pred, color='blue', linewidth=1, label=label)
    
    # Plot QC data and Regression line
    plt.scatter(qc_x, qc_y, s=15, color='dimgrey', label= f'QC Data: n={qc_x.dropna().size}')

    y_pred, r_sq, intercept, slope = _get_linear_regression(qc_x.dropna(), qc_y.dropna())
    label = f'Estimated Regression line - QC\ny = {slope:.4f}x + {intercept:.4f}\nR\u00b2 : {r_sq:.4f}'
    plt.plot(qc_x.dropna(), y_pred, color='red', linewidth=1, label=label)

    plt.legend(loc='upper left', fontsize=11)

    plt.grid(linewidth=0.5, color='gray', linestyle='--')
    plt.title(plot_attributes['left_title'], loc='left', fontsize=12)
    plt.title(plot_attributes['date_title'], loc='right', fontweight='semibold')

    plt.xlabel(plot_attributes['xlabel'], fontsize=12)
    plt.ylabel(plot_attributes['ylabel'], fontsize=12)
    
    plt.savefig(plot_attributes['outdir']+plot_attributes['save_filename'],
                bbox_inches='tight', pad_inches=0.1)
    plt.close('all')

def plot_scatter(df, metadata):
    
    ## Plot all H(x) data with qc as EffectiveQC = 0 ##
    left_title = '{sensor} {satellite} - H(x)\nAll Channels - EffectiveQC = 0 QC'.format(**metadata)
    date_title = '{cycle}'.format(**metadata)
    save_filename = '{cycle}_{sensor}_{satellite}_HofX_All_Channels_EffectiveQC_scatter.png'.format(**metadata)
    xlabel = 'UFO H(x)'
    ylabel = 'GSI H(x)'
    
    plot_attributes = {'left_title': left_title,
                       'date_title': date_title,
                       'save_filename': save_filename,
                       'xlabel': xlabel,
                       'ylabel': ylabel,
                       'outdir': metadata['outdir']
                      }
    
    _create_scatter(x=df['ufo'], y=df['gsi'],
                    qc_x=df['qc_flag_ufo'],
                    qc_y=df['qc_flag_gsi'],
                    plot_attributes=plot_attributes)
    
    ## Plot all H(x) data with qc as GSIObservationError < 1e9 ##
    left_title = '{sensor} {satellite} - H(x)\nAll Channels - GSI Observation Error < 1e9 QC'.format(**metadata)
    date_title = '{cycle}'.format(**metadata)
    save_filename = '{cycle}_{sensor}_{satellite}_HofX_All_Channels_GsiObsError_QC_scatter.png'.format(**metadata)
    xlabel = 'UFO H(x)'
    ylabel = 'GSI H(x)'
    
    plot_attributes = {'left_title': left_title,
                       'date_title': date_title,
                       'save_filename': save_filename,
                       'xlabel': xlabel,
                       'ylabel': ylabel,
                       'outdir': metadata['outdir']
                      }
    
    _create_scatter(x=df['ufo'], y=df['gsi'],
                    qc_x=df['err_ufo'],
                    qc_y=df['err_gsi'],
                    plot_attributes=plot_attributes)
    
    ## Plot Effective Error vs. GsiFinalObsError with qc as EffectiveQC = 0 ##
    left_title = '{sensor} {satellite} - Errors\nAll Channels - EffectiveQC = 0 QC'.format(**metadata)
    date_title = '{cycle}'.format(**metadata)
    save_filename = '{cycle}_{sensor}_{satellite}_Errors_All_Channels_EffectiveQC_scatter.png'.format(**metadata)
    xlabel = 'UFO Effective Error'
    ylabel = 'GSI Observation Error'
    
    plot_attributes = {'left_title': left_title,
                       'date_title': date_title,
                       'save_filename': save_filename,
                       'xlabel': xlabel,
                       'ylabel': ylabel,
                       'outdir': metadata['outdir']
                      }
    
    _create_scatter(qc_x=df['qc_flag_ufo_oberr'],
                    qc_y=df['qc_flag_gsi_oberr'],
                    plot_attributes=plot_attributes)
    
    ## Plot Effective Error vs. GsiFinalObsError with qc as GSIObservationError < 1e9 ##
    left_title = '{sensor} {satellite} - Errors\nAll Channels - GSI Observation Error < 1e9 QC'.format(**metadata)
    date_title = '{cycle}'.format(**metadata)
    save_filename = '{cycle}_{sensor}_{satellite}_Errors_All_Channels_GsiObsError_QC_scatter.png'.format(**metadata)
    xlabel = 'UFO Effective Error'
    ylabel = 'GSI Observation Error'
    
    plot_attributes = {'left_title': left_title,
                       'date_title': date_title,
                       'save_filename': save_filename,
                       'xlabel': xlabel,
                       'ylabel': ylabel,
                       'outdir': metadata['outdir']
                      }
    
    _create_scatter(qc_x=df['err_ufo_oberr'],
                    qc_y=df['err_gsi_oberr'],
                    plot_attributes=plot_attributes)
    
    
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
    
    if metadata['satellite'] == 'iasi':
        # Separated into 15 microns CO2 Channels 1-284, Water Vapor Channels 285-465,
        # and 4.3 microns CO2 Channels 466-616
        ufo_data = [np.sum(ufo_data[0:283]), np.sum(ufo_data[284:464]), np.sum(ufo_data[465:-1])]
        gsi_data = [np.sum(gsi_data[0:283]), np.sum(gsi_data[284:464]), np.sum(gsi_data[465:-1])]
        
        ufo_qc_data = [np.sum(ufo_qc_data[0:283]), np.sum(ufo_qc_data[284:464]), np.sum(ufo_qc_data[465:-1])]
        gsi_qc_data = [np.sum(gsi_qc_data[0:283]), np.sum(gsi_qc_data[284:464]), np.sum(gsi_qc_data[465:-1])]
        
        
    elif metadata['satellite'] == 'cris':
        # Separated into 15 microns CO2 Channels 1-263, Water Vapor Channels 264-366,
        # and 4.3 microns CO2 Channels 367-431
        ufo_data = [np.sum(ufo_data[0:262]), np.sum(ufo_data[263:365]), np.sum(ufo_data[366:-1])]
        gsi_data = [np.sum(gsi_data[0:262]), np.sum(gsi_data[263:365]), np.sum(gsi_data[366:-1])]
        
        ufo_qc_data = [np.sum(ufo_qc_data[0:262]), np.sum(ufo_qc_data[263:365]), np.sum(ufo_qc_data[366:-1])]
        gsi_qc_data = [np.sum(gsi_qc_data[0:262]), np.sum(gsi_qc_data[263:365]), np.sum(gsi_qc_data[366:-1])]
        
    elif metadata['satellite'] == 'airs':
        # Separated into 15 microns CO2 Channels 1-162, Water Vapor Channels 163-214,
        # and 4.3 microns CO2 Channels 367-431
        ufo_data = [np.sum(ufo_data[0:161]), np.sum(ufo_data[162:213]), np.sum(ufo_data[214:-1])]
        gsi_data = [np.sum(gsi_data[0:161]), np.sum(gsi_data[162:213]), np.sum(gsi_data[214:-1])]
        
        ufo_qc_data = [np.sum(ufo_qc_data[0:161]), np.sum(ufo_qc_data[162:213]), np.sum(ufo_qc_data[214:-1])]
        gsi_qc_data = [np.sum(gsi_qc_data[0:161]), np.sum(gsi_qc_data[162:213]), np.sum(gsi_qc_data[214:-1])]
        
        
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

    ax.set_xticks(x)
    
    if metadata['satellite'] is in ['iasi', 'cris', 'airs']:
        plt.xlabel('Window', fontsize=12)
        ax.set_xticklabels(['15\u03BCm CO\u2082', 'Water Vapor', '4.3\u03BCm CO\u2082'])
    else:
        plt.xlabel('Channel', fontsize=12)
        ax.set_xticklabels(metadata['channels'])
    
    plt.ylabel('Count', fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', )
    
    save_filename = '{cycle}_{sensor}_{satellite}_nobs.png'.format(**metadata)
    
    len_x = len(x)
    
    ### Get heights of all data rects ###
    all_data = []
    for rect in rects1+rects2:
        height = rect.get_height()
        all_data.append(height)
    
    # Subtract vals every nth index depending on n channels
    diffs = []
    for i, val in enumerate(all_data[0:len_x]):
        diff = all_data[i] - all_data[i+len_x]
        diffs.append(diff)
    
    # Plot the differences
    for i, rect in enumerate(rects1): 
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/1, height,
                 '%d' % diffs[i], ha='center', va='bottom')
        
    ### Get heights of QC rects ###
    all_data = []
    for rect in rects3+rects4:
        height = rect.get_height()
        all_data.append(height)
    
    # Subtract vals every nth index depending on n channels
    diffs = []
    for i, val in enumerate(all_data[0:len_x]):
        diff = all_data[i] - all_data[i+len_x]
        diffs.append(diff)
    
    # Plot the differences
    for i, rect in enumerate(rects3): 
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/1, height,
                 '%d' % diffs[i], ha='center', va='bottom')
    
    
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
        columns.append(f'brightness_temperature_{i}@EffectiveError')
        columns.append(f'brightness_temperature_{i}@ObsBias')
        columns.append(f'brightness_temperature_{i}@GsiHofXBc')
        columns.append(f'brightness_temperature_{i}@PreQC')
        columns.append(f'brightness_temperature_{i}@GsiFinalObsError')
        
    gsi_use_flag = df['gsi_use_flag@VarMetaData'][0]
        
    # Create dataframe from appropriate column names and indexing
    idx = pd.IndexSlice
    data_df = df.loc[idx[:, 0], columns].reset_index()
          
    return data_df, gsi_use_flag

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
                    'outdir': outpath}

        data_df, gsi_use_flag = get_data_df(file, channels)
        
        # Create Dataframe appropriate for plotting data
        plot_df = pd.DataFrame()
        
        # Get channel counts for bar graph
        ufo_obs_count = []
        ufo_obs_count_qc = []
        ufo_obs_count_err = []
        gsi_obs_count = []
        gsi_obs_count_qc = []
        gsi_obs_count_err = []
        

        for i, chan in enumerate(channels):
            ufo = data_df[f'brightness_temperature_{chan}@hofx']
            gsi = data_df[f'brightness_temperature_{chan}@GsiHofXBc']
            ufo_oberr = data_df[f'brightness_temperature_{chan}@EffectiveError']
            gsi_oberr = data_df[f'brightness_temperature_{chan}@GsiFinalObsError']
            
            # Index by EffectiveQC = 0
            qc_df = data_df.loc[(data_df[f'brightness_temperature_{chan}@EffectiveQC'] == 0)]

            qc_flag_ufo = qc_df[f'brightness_temperature_{chan}@hofx']
            qc_flag_gsi = qc_df[f'brightness_temperature_{chan}@GsiHofXBc']
            qc_flag_ufo_oberr = qc_df[f'brightness_temperature_{chan}@EffectiveError']
            qc_flag_gsi_oberr = qc_df[f'brightness_temperature_{chan}@GsiFinalObsError']

            # If the gsi data was assimilated, index by GsiFinalObsError < 1e9
            if gsi_use_flag[i] == 1:
                error_df = data_df.loc[data_df[f'brightness_temperature_{chan}@GsiFinalObsError'] < 1e9]
                
                err_ufo = error_df[f'brightness_temperature_{chan}@hofx']
                err_gsi = error_df[f'brightness_temperature_{chan}@GsiHofXBc']
                err_ufo_oberr = error_df[f'brightness_temperature_{chan}@EffectiveError']
                err_gsi_oberr = error_df[f'brightness_temperature_{chan}@GsiFinalObsError']
            
            # Else replace the columns with NaNs
            else:
                cols = [f'brightness_temperature_{chan}@hofx',
                        f'brightness_temperature_{chan}@GsiHofXBc',
                        f'brightness_temperature_{chan}@EffectiveError',
                        f'brightness_temperature_{chan}@GsiFinalObsError']
                
                data_df[cols] = np.nan
                
                err_ufo = data_df[f'brightness_temperature_{chan}@hofx']
                err_gsi = data_df[f'brightness_temperature_{chan}@GsiHofXBc']
                err_ufo_oberr = data_df[f'brightness_temperature_{chan}@EffectiveError']
                err_gsi_oberr = data_df[f'brightness_temperature_{chan}@GsiFinalObsError']
            
            # .size is more appropriate than len() when using pandas series
            ufo_obs_count.append(ufo.size)
            ufo_obs_count_qc.append(qc_flag_ufo.dropna().size)
            ufo_obs_count_err.append(err_ufo_oberr.dropna().size)
            gsi_obs_count.append(gsi.size)
            gsi_obs_count_qc.append(qc_flag_gsi.dropna().size)
            gsi_obs_count_err.append(err_gsi_oberr.dropna().size)
            

            plot_df = plot_df.append(pd.DataFrame({'ufo': ufo,
                                                   'gsi': gsi,
                                                   'ufo_oberr': ufo_oberr,
                                                   'gsi_oberr': gsi_oberr,
                                                   'qc_flag_ufo': qc_flag_ufo,
                                                   'qc_flag_gsi': qc_flag_ufo,
                                                   'qc_flag_ufo_oberr': qc_flag_ufo_oberr,
                                                   'qc_flag_gsi_oberr': qc_flag_gsi_oberr,
                                                   'err_ufo': err_ufo, 'err_gsi': err_gsi,
                                                   'err_ufo_oberr': err_ufo_oberr,
                                                   'err_gsi_oberr': err_gsi_oberr,
                                                   'ufo-gsi': ufo-gsi,
                                                   'qc_ufo-qc_gsi': qc_flag_ufo-qc_flag_gsi}))
            
            
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
