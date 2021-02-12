import numpy as np
import glob as glob
from netCDF4 import Dataset
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def _get_linear_regression(data1, data2):
    """
    Inputs:
        data1 : data on the x axis
        data2 : data on the y axis
    """

    x = data1.reshape((-1,1))
    y = data2
    
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


def qc_data(data_dict):
    
    ufo_hofx_qc = [] 
    gsi_hofx_qc = []
    
    for i in range(len(data_dict['cycles'])):
        ufo_hofx_qc_tmp = []
        gsi_hofx_qc_tmp = []
        for j in range(len(data_dict['channels'])):

            ufo_qc_idx = np.where(data_dict['ufo qc'][i][j] == 0)
            
            #tmp line bc o bias correction
            ufo_hofx = data_dict['ufo hofx'][i][j][ufo_qc_idx] + data_dict['ufo obs bias'][i][j][ufo_qc_idx]
            #ufo_hofx = data_dict['ufo hofx'][i][j][ufo_qc_idx]
            
            ufo_hofx_qc_tmp.append(ufo_hofx)
    
            if len(ufo_qc_idx[0]) == 0:
                 gsi_hofx_qc_tmp.append(data_dict['gsi hofx'][i][j][ufo_qc_idx])
            
            else:
                gsi_error = data_dict['gsi error'][i][j]
                gsi_error = gsi_error[gsi_error.mask == False]
        
                gsi_qc_idx = np.where(data_dict['gsi error'][i][j])
                gsi_hofx_qc_tmp.append(data_dict['gsi hofx'][i][j][gsi_qc_idx])

        ufo_hofx_qc.append(ufo_hofx_qc_tmp)
        gsi_hofx_qc.append(gsi_hofx_qc_tmp)
    
    data_dict['ufo hofx qc'] = ufo_hofx_qc
    data_dict['gsi hofx qc'] = gsi_hofx_qc
    
    return data_dict


def plot_hofx_scatter(ufo_data, gsi_data,
                      metadata,
                      ufo_data_qc=None,
                      gsi_data_qc=None):

    # Create Figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    plt.scatter(ufo_data, gsi_data, s=15, color='darkgrey', label=f'All Data (n: {len(ufo_data)})')
    y_pred, r_sq, intercept, slope = _get_linear_regression(ufo_data, gsi_data)


    # Plot Regression line
    label = f'Estimated Regression line\ny = {slope:.4f}x + {intercept:.4f}\nR\u00b2 : {r_sq:.4f}'
    plt.plot(ufo_data, y_pred, color='blue', linewidth=1, label=label)
    
    try:
    
        plt.scatter(ufo_data_qc, gsi_data_qc, s=15, color='dimgrey', label=f'QC=0 Data (n: {len(ufo_data_qc)})')
        
        y_pred, r_sq, intercept, slope = _get_linear_regression(ufo_data_qc, gsi_data_qc)

        # Plot Regression line
        label = f'Estimated Regression line - QC\ny = {slope:.4f}x + {intercept:.4f}\nR\u00b2 : {r_sq:.4f}'
        plt.plot(ufo_data_qc, y_pred, color='red', linewidth=1, label=label)
    
    except:
        pass
    
    
    plt.legend(loc='upper left', fontsize=11)
    
    plt.grid(linewidth=0.5, color='gray', linestyle='--')
    plt.title('{sensor} {satellite} - H(x)\nChannel {channel}'.format(**metadata),
              loc='left', fontsize=12)
    plt.title('{cycle}'.format(**metadata), loc='right', fontweight='semibold')
    
    plt.xlabel('UFO H(x)', fontsize=12)
    plt.ylabel('GSI H(x)', fontsize=12)
    
    save_filename = '{cycle}_{sensor}_{satellite}_channel_{channel}_scatter.png'.format(**metadata)
    
    plt.savefig(metadata['outdir']+save_filename, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')

    return


def plot_hofx_histogram(ufo_data, gsi_data,
                      metadata,
                      ufo_data_qc=None,
                      gsi_data_qc=None):
    
    hist_data = ufo_data - gsi_data
    
    #stats
    n = len(hist_data)
    mean = np.mean(hist_data)
    mn = np.min(hist_data)
    mx = np.max(hist_data)
    std = np.std(hist_data)
    
    # Create Figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    binsize=0.05
    bins=np.arange(0-4*0.5, 0+4*0.5, binsize)
    
    plt.hist(hist_data, bins=bins)
    
    text = f'n: {n}\nmean: {mean:.3f}\nmin: {mn:.3f}\nmax: {mx:.3f}\nstd. dev: {std:.3f}'
    ax.text(0.04, 0.74, text, fontsize=12,
            bbox=dict(boxstyle="round,pad=.5",
                      facecolor="white",
                      edgecolor="grey",
                      linewidth=.75),
           transform=ax.transAxes)
    
    plt.grid(linewidth=0.5, color='gray', linestyle='--')
    plt.title('{sensor} {satellite} - H(x)\nChannel {channel}'.format(**metadata),
              loc='left', fontsize=12)
    plt.title('{cycle}'.format(**metadata), loc='right', fontweight='semibold')
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('UFO H(x) - GSI H(x)', fontsize=12)
    
    save_filename = '{cycle}_{sensor}_{satellite}_channel_{channel}_histogram.png'.format(**metadata)
    
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


def read_var(datapath):
    obsfiles = glob.glob(datapath+'*')
    obsfiles.sort()
    
    lats = []
    lons = []
    filenames = []
    cycles = []
    ufo_hofx = []
    ufo_qc_flag = []
    ufo_error = []
    ufo_obs_bias = []
    gsi_hofx = []
    gsi_qc_flag = []
    gsi_error = []
    
    for file in obsfiles:
        try:
            channels = _get_channels(file)
            
            filename = file.split('/')[-1]
            filenames.append(filename)
            cycle = filename.split('_')[3]
            cycles.append(cycle)

            ufo_hofx_tmp = []
            ufo_qc_flag_tmp = []
            ufo_error_tmp = []
            ufo_obs_bias_tmp = []
            gsi_hofx_tmp = []
            gsi_qc_flag_tmp = []
            gsi_error_tmp = []

            with Dataset(file, mode='r') as f:
                lattmp = f.variables['latitude@MetaData'][:]
                lontmp = f.variables['longitude@MetaData'][:]

                for channel in channels:
                    ufo_hofx_chan = f.variables[f'brightness_temperature_{channel}@hofx'][:]
                    ufo_qc_flag_chan = f.variables[f'brightness_temperature_{channel}@EffectiveQC'][:]
                    ufo_error_chan = f.variables[f'brightness_temperature_{channel}@EffectiveError'][:]
                    ufo_obs_bias_chan = f.variables[f'brightness_temperature_{channel}@ObsBias'][:]
                    
                    gsi_hofx_chan = f.variables[f'brightness_temperature_{channel}@GsiHofXBc'][:]
#                     gsi_hofx_chan = f.variables[f'brightness_temperature_{channel}@GsiHofX'][:]
                    gsi_qc_flag_chan = f.variables[f'brightness_temperature_{channel}@PreQC'][:]
                    gsi_error_chan = f.variables[f'brightness_temperature_{channel}@GsiFinalObsError'][:]

                    ufo_hofx_tmp.append(ufo_hofx_chan)
                    ufo_qc_flag_tmp.append(ufo_qc_flag_chan)
                    ufo_error_tmp.append(ufo_error_chan)
                    ufo_obs_bias_tmp.append(ufo_obs_bias_chan)
                    gsi_hofx_tmp.append(gsi_hofx_chan)
                    gsi_qc_flag_tmp.append(gsi_qc_flag_chan)
                    gsi_error_tmp.append(gsi_error_chan)
                    
            lats.append(lattmp)
            lons.append(lontmp)

            ufo_hofx.append(ufo_hofx_tmp)
            ufo_qc_flag.append(ufo_qc_flag_tmp)
            ufo_error.append(ufo_error_tmp)
            ufo_obs_bias.append(ufo_obs_bias_tmp)
            gsi_hofx.append(gsi_hofx_tmp)
            gsi_qc_flag.append(gsi_qc_flag_tmp)
            gsi_error.append(gsi_error_tmp)

        except:
            pass
        
    data_dict = {'lats': lats,
                 'lons': lons,
                 'filename': filenames,
                 'channels': channels,
                 'cycles': cycles,
                 'ufo hofx': ufo_hofx,
                 'ufo qc': ufo_qc_flag,
                 'ufo error': ufo_error,
                 'ufo obs bias': ufo_obs_bias,
                 'gsi hofx': gsi_hofx,
                 'gsi qc': gsi_qc_flag,
                 'gsi error': gsi_error
                }
    
    return data_dict


def generate_figs(inpath, outpath, qc):
    
    data_dict = read_var(inpath)
    
    if qc_data:
        data_dict = qc_data(data_dict)

    # Plot file individually
    for i in range(len(data_dict['cycles'])):
        ufo_obs_count = []
        ufo_obs_count_qc = []
        gsi_obs_count = []
        gsi_obs_count_qc = []

        for j in range(len(data_dict['channels'])):

            ufo_data = data_dict['ufo hofx'][i][j]
            ufo_obs_bias = data_dict['ufo obs bias'][i][j]

            ufo_data = ufo_data + ufo_obs_bias

            gsi_data = data_dict['gsi hofx'][i][j]
            ufo_data_qc = data_dict['ufo hofx qc'][i][j]
            gsi_data_qc = data_dict['gsi hofx qc'][i][j]

            ufo_obs_count.append(len(ufo_data))
            ufo_obs_count_qc.append(len(ufo_data_qc))
            gsi_obs_count.append(len(gsi_data))
            gsi_obs_count_qc.append(len(gsi_data_qc))

            sensor = data_dict['filename'][i].split('_')[0]
            satellite = data_dict['filename'][i].split('_')[1]

            metadata = {'sensor': sensor,
                        'satellite': satellite,
                        'cycle': data_dict['cycles'][i],
                        'channel': data_dict['channels'][j],
                        'outdir': outpath,
                       }                

            plot_hofx_scatter(ufo_data, gsi_data, metadata,
                              ufo_data_qc=ufo_data_qc,
                              gsi_data_qc=gsi_data_qc)

            plot_hofx_histogram(ufo_data, gsi_data, metadata)

        metadata = {'sensor': sensor,
                    'satellite': satellite,
                    'cycle': data_dict['cycles'][i],
                    'channels': data_dict['channels'],
                    'outdir': outpath,
                   }

        plot_obs_count(ufo_obs_count, ufo_obs_count_qc,
                       gsi_obs_count, gsi_obs_count_qc,
                       metadata)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--diagdir', help='path to UFO netCDF diags', required=True)
    ap.add_argument('-o', '--output', help="path to output directory", default="./")
    ap.add_argument('-q', '--qc', help="include good QC check", action='store_true')
    MyArgs = ap.parse_args()
    generate_figs(MyArgs.diagdir, MyArgs.output, MyArgs.qc)
