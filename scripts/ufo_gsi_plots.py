import argparse
import xarray as xr
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib import rcParams, ticker
import glob
import os
from sklearn.linear_model import LinearRegression
from datetime import datetime

#set figure params one time only.
rcParams['figure.subplot.left'] = 0.1
rcParams['figure.subplot.top'] = 0.85
rcParams['legend.fontsize'] = 12
rcParams['axes.grid'] = True

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


def plot_scatter(plot_df, metadata):
    
    ## Plot all H(x) data with qc as EffectiveQC = 0 ##
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # H(x) data
    plt.scatter(x=plot_df['hofx'], y=plot_df['GsiHofXBc'], s=15, color='darkgray', label=f'All Data: n={plot_df["hofx"].count()}')

    # drop nans to get equal amount of values to get linear regression
    tmp = plot_df[['hofx','GsiHofXBc']].dropna()
    y_pred, r_sq, intercept, slope = _get_linear_regression(tmp['hofx'], tmp['GsiHofXBc'])
    label = f'Estimated Regression line\ny = {slope:.4f}x + {intercept:.4f}\nR\u00b2 : {r_sq:.4f}'
    plt.plot(tmp['hofx'], y_pred, color='blue', linewidth=1, label=label)

    # Effective QC = 0 
    # create df where nans are ignored for EffectiveQC
    qcdf = plot_df[plot_df['EffectiveQC'].notnull()]
    plt.scatter(x=qcdf['hofx'], y=qcdf['GsiHofXBc'], s=15, color='dimgray', label=f'QC Data: n={qcdf["hofx"].count()}')

    # drop nans to get equal amount of values to get linear regression
    qctmp = qcdf[['hofx','GsiHofXBc']].dropna()
    y_pred, r_sq, intercept, slope = _get_linear_regression(qctmp['hofx'], qctmp['GsiHofXBc'])
    label = f'Estimated Regression line - QC\ny = {slope:.4f}x + {intercept:.4f}\nR\u00b2 : {r_sq:.4f}'
    plt.plot(qctmp['hofx'], y_pred, color='red', linewidth=1, label=label)

    plt.legend(loc='upper left', fontsize=11)
    plt.title('{sensor} {satellite} - H(x)\nAll Channels - EffectiveQC = 0 QC'.format(**metadata),
              loc='left', fontsize=12)

    plt.xlabel('UFO H(x)', fontsize=12)
    plt.ylabel('GSI H(x)', fontsize=12)

    if metadata['concatenate']:
        date_title = '{cycle[0]}-\n{cycle[1]}'.format(**metadata)
        save_filename = '{cycle[0]}_{cycle[1]}_{sensor}_{satellite}_HofX_All_Channels_EffectiveQC_scatter.png'.format(**metadata)
    else:
        date_title = '{cycle}'.format(**metadata)
        save_filename = '{cycle}_{sensor}_{satellite}_HofX_All_Channels_EffectiveQC_scatter.png'.format(**metadata)

    plt.title(date_title, loc='right', fontweight='semibold')
    plt.savefig(metadata['outdir']+save_filename,
                bbox_inches='tight', pad_inches=0.1)


    ######################################################

    ## Plot all H(x) data with qc as GSIObservationError < 1e10 ##
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # H(x) data
    plt.scatter(x=plot_df['hofx'], y=plot_df['GsiHofXBc'], s=15, color='darkgray', label=f'All Data: n={plot_df["hofx"].count()}')

    # drop nans to get equal amount of values to get linear regression
    tmp = plot_df[['hofx','GsiHofXBc']].dropna()
    y_pred, r_sq, intercept, slope = _get_linear_regression(tmp['hofx'], tmp['GsiHofXBc'])
    label = f'Estimated Regression line\ny = {slope:.4f}x + {intercept:.4f}\nR\u00b2 : {r_sq:.4f}'
    plt.plot(tmp['hofx'], y_pred, color='blue', linewidth=1, label=label)

    # GSIObservationError < 1e10 
    # create df where nans are ignored for EffectiveQC
    errordf = plot_df[plot_df['GsiFinalObsError'].notnull()]
    plt.scatter(x=errordf['hofx'], y=errordf['GsiHofXBc'], s=15, color='dimgray', label=f'QC Data: n={errordf["hofx"].count()}')

    # drop nans to get equal amount of values to get linear regression
    errortmp = errordf[['hofx','GsiHofXBc']].dropna()
    y_pred, r_sq, intercept, slope = _get_linear_regression(errortmp['hofx'], errortmp['GsiHofXBc'])
    label = f'Estimated Regression line - QC\ny = {slope:.4f}x + {intercept:.4f}\nR\u00b2 : {r_sq:.4f}'
    plt.plot(errortmp['hofx'], y_pred, color='red', linewidth=1, label=label)

    plt.legend(loc='upper left', fontsize=11)
    plt.title('{sensor} {satellite} - H(x)\nAll Channels - GSI Observation Error < 1e10'.format(**metadata),
              loc='left', fontsize=12)

    plt.xlabel('UFO H(x)', fontsize=12)
    plt.ylabel('GSI H(x)', fontsize=12)

    if metadata['concatenate']:
        date_title = '{cycle[0]}-\n{cycle[1]}'.format(**metadata)
        save_filename = '{cycle[0]}_{cycle[1]}_{sensor}_{satellite}_HofX_All_Channels_GsiObsError_QC_scatter.png'.format(**metadata)
    else:
        date_title = '{cycle}'.format(**metadata)
        save_filename = '{cycle}_{sensor}_{satellite}_HofX_All_Channels_GsiObsError_QC_scatter.png'.format(**metadata)

    plt.title(date_title, loc='right', fontweight='semibold')
    plt.savefig(metadata['outdir']+save_filename,
                bbox_inches='tight', pad_inches=0.1)
    
    ######################################################
    
    ## Plot Effective Error vs. GsiFinalObsError with qc as EffectiveQC = 0 ##
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # Effective QC = 0
    # create df where nans are ignored for EffectiveQC
    qcdf = plot_df[plot_df['EffectiveQC'].notnull()]
    plt.scatter(x=qcdf['EffectiveError'], y=qcdf['GsiFinalObsError'], s=15, color='dimgray', label=f'QC Data: n={qcdf["EffectiveError"].count()}')

    # drop nans to get equal amount of values to get linear regression
    qctmp = qcdf[['EffectiveError','GsiFinalObsError']].dropna()
    y_pred, r_sq, intercept, slope = _get_linear_regression(qctmp['EffectiveError'], qctmp['GsiFinalObsError'])
    label = f'Estimated Regression line - QC\ny = {slope:.4f}x + {intercept:.4f}\nR\u00b2 : {r_sq:.4f}'
    plt.plot(qctmp['EffectiveError'], y_pred, color='red', linewidth=1, label=label)

    plt.legend(loc='upper left', fontsize=11)
    plt.title('{sensor} {satellite} - Errors\nAll Channels - EffectiveQC = 0 QC'.format(**metadata),
              loc='left', fontsize=12)

    plt.xlabel('UFO Effective Error', fontsize=12)
    plt.ylabel('GSI Observation Error', fontsize=12)

    if metadata['concatenate']:
        date_title = '{cycle[0]}-\n{cycle[1]}'.format(**metadata)
        save_filename = '{cycle[0]}_{cycle[1]}_{sensor}_{satellite}_Errors_All_Channels_EffectiveQC_scatter.png'.format(**metadata)
    else:
        date_title = '{cycle}'.format(**metadata)
        save_filename = '{cycle}_{sensor}_{satellite}_Errors_All_Channels_EffectiveQC_scatter.png'.format(**metadata)

    plt.title(date_title, loc='right', fontweight='semibold')
    plt.savefig(metadata['outdir']+save_filename,
                bbox_inches='tight', pad_inches=0.1)

    
    ######################################################
    
    ## Plot Effective Error vs. GsiFinalObsError with qc as GSIObservationError < 1e9 ##
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    errordf = plot_df[plot_df['GsiFinalObsError'].notnull()]
    plt.scatter(x=errordf['EffectiveError'], y=errordf['GsiFinalObsError'], s=15, color='dimgray', label=f'QC Data: n={errordf["EffectiveError"].count()}')

    # drop nans to get equal amount of values to get linear regression
    errortmp = errordf[['EffectiveError','GsiFinalObsError']].dropna()
    y_pred, r_sq, intercept, slope = _get_linear_regression(errortmp['EffectiveError'], errortmp['GsiFinalObsError'])
    label = f'Estimated Regression line - QC\ny = {slope:.4f}x + {intercept:.4f}\nR\u00b2 : {r_sq:.4f}'
    plt.plot(errortmp['EffectiveError'], y_pred, color='red', linewidth=1, label=label)

    plt.legend(loc='upper left', fontsize=11)
    plt.title('{sensor} {satellite} - Errors\nAll Channels - GSI Observation Error < 1e9 QC'.format(**metadata),
              loc='left', fontsize=12)

    plt.xlabel('UFO Effective Error', fontsize=12)
    plt.ylabel('GSI Observation Error', fontsize=12)
    
    if metadata['concatenate']:
        date_title = '{cycle[0]}-\n{cycle[1]}'.format(**metadata)
        save_filename = '{cycle[0]}_{cycle[1]}_{sensor}_{satellite}_Errors_All_Channels_GsiObsError_QC_scatter.png'.format(**metadata)
    else:
        date_title = '{cycle}'.format(**metadata)
        save_filename = '{cycle}_{sensor}_{satellite}_Errors_All_Channels_GsiObsError_QC_scatter.png'.format(**metadata)
        
    plt.title(date_title, loc='right', fontweight='semibold')
    plt.savefig(metadata['outdir']+save_filename,
                bbox_inches='tight', pad_inches=0.1)
    
    plt.close('all')
    
    return

def plot_histogram(plot_df, metadata):
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    stats = plot_df['ufo-gsi'].describe()
    plot_df['ufo-gsi'].plot.hist(ax=ax)

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
    plt.title('{sensor} {satellite} - H(x)\nAll Channels\n'.format(**metadata),
              loc='left', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('UFO H(x) - GSI H(x)', fontsize=12)

    if metadata['concatenate']:
        date_title = '{cycle[0]}-\n{cycle[1]}'.format(**metadata)
        save_filename = '{cycle[0]}_{cycle[1]}_{sensor}_{satellite}_HofX_All_Channels_histogram.png'.format(**metadata)
    else:
        date_title = '{cycle}'.format(**metadata)
        save_filename = '{cycle}_{sensor}_{satellite}_HofX_All_Channels_histogram.png'.format(**metadata)

    plt.title(date_title, loc='right', fontweight='semibold')

    plt.savefig(metadata['outdir']+save_filename, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')
    
    return

def plot_obscount(count_dict, metadata):
    
    ufo_data = count_dict['ufo_count']
    gsi_data = count_dict['gsi_count']
    ufo_qc_data = count_dict['qc_ufo_count']
    gsi_qc_data = count_dict['qc_gsi_count']

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


    x = np.array([x+1 for x in np.arange(len(count_dict['ufo_count']))])
    width = 0.2

    fig, ax = plt.subplots(figsize=(15,5))
    plt.grid(linewidth=0.5, color='gray', linestyle='--')

    rects1 = ax.bar(x - 1.5*width, ufo_data, width, label='All UFO Data')
    rects2 = ax.bar(x - width/2, gsi_data, width, label='All GSI Data')
    rects3 = ax.bar(x + width/2, ufo_qc_data, width, label='UFO Assimilated Data')
    rects4 = ax.bar(x + 1.5*width, gsi_qc_data, width, label='GSI Assimilated Data')

    ax.set_xticks(x)

    if metadata['satellite'] in ['iasi', 'cris', 'airs']:
        plt.xlabel('Window', fontsize=12)
        ax.set_xticklabels(['15\u03BCm CO\u2082', 'Water Vapor', '4.3\u03BCm CO\u2082'])
    else:
        plt.xlabel('Channel', fontsize=12)
        ax.set_xticklabels(x)

    plt.ylabel('Count', fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', )

    if metadata['concatenate']:
        date_title = '{cycle[0]}-\n{cycle[1]}'.format(**metadata)
        save_filename = '{cycle[0]}_{cycle[1]}_{sensor}_{satellite}_nobs.png'.format(**metadata)
    else:
        date_title = '{cycle}'.format(**metadata)
        save_filename = '{cycle}_{sensor}_{satellite}_nobs.png'.format(**metadata)

    plt.title(date_title, loc='right', fontweight='semibold')

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


def create_dataframes(obsfiles):
    datadf = pd.DataFrame([])
    for file in obsfiles:
        _str_cycle = os.path.basename(file).split('_')[3]
        cycle = datetime.strptime(_str_cycle, '%Y%m%d%H')
        
        # read into xarray
        with xr.open_dataset(file) as data:
            # load all data from the transformed dataset
            # to ensure we can use it after closing each original file
            data.load()
            
        validvars = [var for var in list(data.keys()) if var.endswith(('@hofx', '@EffectiveQC',
                                                                       '@EffectiveError', '@ObsBias',
                                                                       '@GsiHofXBc', '@PreQC',
                                                                       '@GsiFinalObsError'))]
    
        df = data[validvars].to_dataframe()
        
        if df.shape[0] > 0:
            df['cycle'] = cycle
            datadf = datadf.append(df)
            
    metavars = [var for var in list(data.keys()) if var.endswith(('@VarMetaData'))]
    metadatadf = data[metavars].to_dataframe()
    
    ## Process and filter data ##
    print('Filtering data ...')
    # Find channels that are not used (gsi_use_flag == -1)
    badchans = metadatadf['variable_names@VarMetaData'][0][metadatadf['gsi_use_flag@VarMetaData'][0] != 1].str.decode('utf-8').tolist()
    badchans = [x.strip() for x in badchans]
    
    # Replace large values and bad channels with NaNs
    filtercols = [var for var in datadf.columns if var.endswith(('@hofx', '@GsiHofXBc',
                                                                 '@EffectiveError',
                                                                 '@EffectiveQC', '@GsiFinalObsError'))]
    for col in filtercols:
        datadf.loc[datadf[col] > 1e10, col] = float("NaN")
        if col.startswith(tuple(badchans)):
            datadf[col] = float("NaN")

    # replace where Effective qc is not 0 with NaNs
    qccols = [var for var in datadf.columns if var.endswith(('@EffectiveQC'))]
    for col in qccols:
        datadf.loc[datadf[col] != 0, col] = float("NaN")
    
    return datadf, metadatadf

def get_metadata(obsfiles, outpath, concatenate):
    
    filename = obsfiles[0].split('/')[-1]
    sensor = filename.split('_')[0]
    satellite = filename.split('_')[1]

    # Get first and last file cycle
    s_cycle = filename.split('_')[3]

    if concatenate:
        end_file = obsfiles[-1]
        e_filename = end_file.split('/')[-1]
        e_cycle = e_filename.split('_')[3]


        metadata = {'sensor': sensor,
                    'satellite': satellite,
                    'cycle': [s_cycle, e_cycle],
                    'concatenate': True,
                    'outdir': outpath}
    else:
        metadata = {'sensor': sensor,
                    'satellite': satellite,
                    'cycle': s_cycle,
                    'concatenate': False,
                    'outdir': outpath}
        
    return metadata


def generate_figs(inpath, outpath, concatenate):
    
    # Get files
    print('Fetching Files ...')
    obsfiles = glob.glob(inpath+'*output.nc4')
    obsfiles.sort()

    print('Grabbing metadata ...')
    metadata = get_metadata(obsfiles, outpath, concatenate)

    # just working on concatenation right now
    if concatenate:
        print('Concatenating and creating dataframes ...')
        datadf, metadatadf = create_dataframes(obsfiles)

        ## Create Plotting dataframe ##
        columns = ['hofx', 'EffectiveQC','EffectiveError', 'ObsBias','GsiHofXBc', 'PreQC','GsiFinalObsError']
        plot_df = pd.DataFrame(columns=columns)

        validvars = [var for var in list(datadf.keys()) if var.endswith(('@hofx', '@EffectiveQC',
                                                                        '@EffectiveError', '@ObsBias',
                                                                        '@GsiHofXBc', '@PreQC',
                                                                        '@GsiFinalObsError'))]

        plot_dict = {'hofx': [], 'EffectiveQC': [],'EffectiveError': [], 'ObsBias': [],'GsiHofXBc': [], 'PreQC': [],'GsiFinalObsError': []}

        for col in validvars:
            var = col.split('@')[-1]
            plot_dict[var].extend(datadf[col])

        plot_df = pd.DataFrame.from_dict(plot_dict)
        plot_df['ufo-gsi'] = plot_df['hofx']-plot_df['GsiHofXBc']

        ## Plot Scatter and histogram ##
        print('Plotting ...')
        plot_scatter(plot_df, metadata)
        plot_histogram(plot_df, metadata)

        ## Obs count plot ##
        countcols = [var for var in list(datadf.keys()) if var.endswith(('@hofx', '@EffectiveQC', '@GsiHofXBc'))]
        countdf = datadf[countcols]

        count_dict={'ufo_count': [],
                    'gsi_count': [],
                    'qc_ufo_count': [],
                    'qc_gsi_count': []}

        channels = metadatadf['variable_names@VarMetaData'][0].str.decode('utf-8').to_list()

        for chan in channels:
            chan = chan.strip()
            count_dict['ufo_count'].append(countdf[chan+'@hofx'].count())
            count_dict['gsi_count'].append(countdf[chan+'@GsiHofXBc'].count())

            qctmp = countdf[countdf[chan+'@EffectiveQC'].notnull()]
            count_dict['qc_ufo_count'].append(qctmp[chan+'@hofx'].count())
            count_dict['qc_gsi_count'].append(qctmp[chan+'@GsiHofXBc'].count())

        plot_obscount(count_dict, metadata)
        
    return
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--diagdir', help='path to UFO netCDF diags', required=True)
    ap.add_argument('-o', '--output', help="path to output directory", default="./")
    ap.add_argument('-c', '--concatenate', help="True if all files calculated together.", default=False)
    MyArgs = ap.parse_args()
    
    generate_figs(MyArgs.diagdir, MyArgs.output, MyArgs.concatenate)
    print('done')    
