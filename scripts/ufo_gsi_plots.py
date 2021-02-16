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


def plot_hofx_scatter(ufo_data, gsi_data,
                      metadata,
                      ufo_data_qc=None,
                      gsi_data_qc=None):

    # Create Figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    extend_ufo = []
    extend_gsi = []
    extend_ufo_qc = []
    extend_gsi_qc = []
    
    for i in range(len(metadata['channels'])-1):
        
        plt.scatter(ufo_data[i], gsi_data[i], s=15, color='darkgrey')#, label=f'All Data (n: {len(ufo_data[i])})')

        try:
            plt.scatter(ufo_data_qc[i], gsi_data_qc[i], s=15, color='dimgrey')#, label=f'QC=0 Data (n: {len(ufo_data_qc)})')
        except:
            pass
        
        extend_ufo.extend(ufo_data[i])
        extend_gsi.extend(gsi_data[i])
        extend_ufo_qc.extend(ufo_data_qc[i])
        extend_gsi_qc.extend(gsi_data_qc[i])
        
    extend_ufo.extend(ufo_data[-1])
    extend_gsi.extend(gsi_data[-1])
    extend_ufo_qc.extend(ufo_data_qc[-1])
    extend_gsi_qc.extend(gsi_data_qc[-1])
    
    plt.scatter(ufo_data[-1], gsi_data[-1], s=15, color='darkgrey', label=f'All Data (n: {len(extend_ufo)})')
    try:
        plt.scatter(ufo_data_qc[-1], gsi_data_qc[-1], s=15, color='dimgrey', label=f'QC=0 Data (n: {len(extend_ufo_qc)})')
    except:
        pass    
        

    # Plot Regression line
    y_pred, r_sq, intercept, slope = _get_linear_regression(extend_ufo, extend_gsi)
    label = f'Estimated Regression line\ny = {slope:.4f}x + {intercept:.4f}\nR\u00b2 : {r_sq:.4f}'
    plt.plot(extend_ufo, y_pred, color='blue', linewidth=1, label=label)

    # Plot QC data Regression line
    y_pred, r_sq, intercept, slope = _get_linear_regression(extend_ufo_qc, extend_gsi_qc)
    label = f'Estimated Regression line - QC\ny = {slope:.4f}x + {intercept:.4f}\nR\u00b2 : {r_sq:.4f}'
    plt.plot(extend_ufo_qc, y_pred, color='red', linewidth=1, label=label)
    
    
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


def plot_hofx_histogram(ufo_data, gsi_data,
                      metadata,
                      ufo_data_qc=None,
                      gsi_data_qc=None):
    
    hist_data = []
    for i in range(len(metadata['channels'])):
        data = ufo_data[i] - gsi_data[i]
        hist_data.extend(data)
    
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


def read_var(datapath):
    obsfiles = glob.glob(datapath+'*')
    obsfiles.sort()
    
    lats = []
    lons = []
    filenames = []
    cycles = []
    ufo_hofx = []
    ufo_hofx_qc = []
    gsi_hofx = []
    gsi_hofx_qc = []
    
    for file in obsfiles:
        try:
            channels = _get_channels(file)

            filename = file.split('/')[-1]
            filenames.append(filename)
            cycle = filename.split('_')[3]
            cycles.append(cycle)

            ufo_hofx_tmp = []
            ufo_hofx_qc_tmp = []
            gsi_hofx_tmp = []
            gsi_hofx_qc_tmp = []

            with Dataset(file, mode='r') as f:
                lattmp = f.variables['latitude@MetaData'][:]
                lontmp = f.variables['longitude@MetaData'][:]

                for channel in channels:
                    ufo_hofx_chan = f.variables[f'brightness_temperature_{channel}@hofx'][:]
                    ufo_qc_flag_chan = f.variables[f'brightness_temperature_{channel}@EffectiveQC'][:]
                    ufo_obs_bias_chan = f.variables[f'brightness_temperature_{channel}@ObsBias'][:]

                    ### COMMENT THIS LINE OUT IF DATA HAS BIAS CORRECTION INCLUDED ###
                    ufo_hofx_chan = ufo_hofx_chan + ufo_obs_bias_chan

                    gsi_hofx_chan = f.variables[f'brightness_temperature_{channel}@GsiHofXBc'][:]
                    #gsi_hofx_chan = f.variables[f'brightness_temperature_{channel}@GsiHofX'][:]
                    gsi_qc_flag_chan = f.variables[f'brightness_temperature_{channel}@PreQC'][:]
                    gsi_error_chan = f.variables[f'brightness_temperature_{channel}@GsiFinalObsError'][:]

                    ### Create QC data ###
                    ufo_qc_idx = np.where(ufo_qc_flag_chan == 0)
                    ufo_hofx_qc_chan = ufo_hofx_chan[ufo_qc_idx]

                    # If len of UFO index is 0, GSI is not also 0, so just apply UFO index to GSI data
                    if len(ufo_qc_idx[0]) == 0:
                        gsi_hofx_qc_chan = gsi_hofx_chan[ufo_qc_idx]

                    # Using where GSI error is masked as qc
                    else:
                        gsi_qc_idx = np.where(gsi_error_chan.mask == False)

                        gsi_hofx_qc_chan = gsi_hofx_chan[gsi_qc_idx]

                    ### Append channel data ###
                    ufo_hofx_tmp.append(ufo_hofx_chan)
                    ufo_hofx_qc_tmp.append(ufo_hofx_qc_chan)
                    gsi_hofx_tmp.append(gsi_hofx_chan)
                    gsi_hofx_qc_tmp.append(gsi_hofx_qc_chan)


            lats.append(lattmp)
            lons.append(lontmp)

            ufo_hofx.append(ufo_hofx_tmp)
            ufo_hofx_qc.append(ufo_hofx_qc_tmp)
            gsi_hofx.append(gsi_hofx_tmp)
            gsi_hofx_qc.append(gsi_hofx_qc_tmp)

        except:
            pass
        
    data_dict = {'lats': lats,
                 'lons': lons,
                 'filename': filenames,
                 'channels': channels,
                 'cycles': cycles,
                 'ufo hofx': ufo_hofx,
                 'ufo hofx qc': ufo_hofx_qc,
                 'gsi hofx': gsi_hofx,
                 'gsi hofx qc': gsi_hofx_qc,
                }
    
    return data_dict


def generate_figs(inpath, outpath):
    
    data_dict = read_var(inpath)
    
    ### Plot Cycles ###
    for i in range(len(data_dict['cycles'])):
        
        sensor = data_dict['filename'][i].split('_')[0]
        satellite = data_dict['filename'][i].split('_')[1]
        
        metadata = {'sensor': sensor,
                    'satellite': satellite,
                    'cycle': data_dict['cycles'][i],
                    'channels': data_dict['channels'],
                    'outdir': os.path.join(outpath),
                   }     
        
        ufo_data = data_dict['ufo hofx'][i]
        gsi_data = data_dict['gsi hofx'][i]
        ufo_data_qc = data_dict['ufo hofx qc'][i]
        gsi_data_qc = data_dict['gsi hofx qc'][i]
        
        ### Plot Scatter ###
        plot_hofx_scatter(ufo_data, gsi_data, metadata,
                          ufo_data_qc=ufo_data_qc,
                          gsi_data_qc=gsi_data_qc)
        
        ### Plot Histogram ###
        plot_hofx_histogram(ufo_data, gsi_data, metadata)
        
        
        ### Plot Obs Count ###
        ufo_obs_count = []
        ufo_obs_count_qc = []
        gsi_obs_count = []
        gsi_obs_count_qc = []
        
        for j in range(len(data_dict['channels'])):
            ufo_obs_count.append(len(data_dict['ufo hofx'][i][j]))
            ufo_obs_count_qc.append(len(data_dict['ufo hofx qc'][i][j]))
            gsi_obs_count.append(len(data_dict['gsi hofx'][i][j]))
            gsi_obs_count_qc.append(len(data_dict['gsi hofx qc'][i][j]))
            

        plot_obs_count(ufo_obs_count, ufo_obs_count_qc,
                       gsi_obs_count, gsi_obs_count_qc,
                       metadata)
        
        

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--diagdir', help='path to UFO netCDF diags', required=True)
    ap.add_argument('-o', '--output', help="path to output directory", default="./")
    MyArgs = ap.parse_args()
    generate_figs(MyArgs.diagdir, MyArgs.output)
