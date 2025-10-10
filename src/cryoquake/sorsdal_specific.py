"""
A place for functions that are specific to the Sorsdal dataset, so would need modification for other settings.
"""

import numpy as np
from src import visualisation as vis
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
from obspy import Stream, Trace, UTCDateTime
import pyTMD

#temperature and tidal data as an obspy stream
def TemperatureTidalStream(t1,t2,temp_dir,tide_dir='/Users/jmagyar/Documents/tides',lat=-68.70793,lon=78.10155):
    #load the temperature data from file
    all_files = glob.glob(os.path.join(temp_dir, "*.csv"))

    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    df = df.drop(['MAX_OBSERVATION_DATE','MIN_RAINFALL','AVG_RAINFALL','MAX_RAINFALL','MIN_WIND_SPEED',
                  'MAX_WIND_SPEED','AVG_WIND_DIRECTION','MIN_RELATIVE_HUMIDITY','MAX_RELATIVE_HUMIDITY','MIN_AIR_TEMPERATURE',
                  'MAX_AIR_TEMPERATURE','MIN_AIR_PRESSURE','MAX_AIR_PRESSURE'],axis=1)
    df['AVG_RELATIVE_HUMIDITY'][df['AVG_RELATIVE_HUMIDITY'] > 100] = pd.NA

    for index, row in df.iterrows():
        df['MIN_OBSERVATION_DATE_'][index] = pd.to_datetime(df['MIN_OBSERVATION_DATE_'][index],format='%Y-%m-%d %H:%M:%S')

    #now resample at hourly intervals and give NaN values when there is missing hours
    df = df.sort_values('MIN_OBSERVATION_DATE_',axis=0)
    df = df.set_index('MIN_OBSERVATION_DATE_')
    df = df.resample('1H').bfill(limit=1)

    #make traces for the important climate variables
    delta = 60 * 60 #seconds in an hour (sampling distance)
    start = df.index[0]
    starttime = start.year,start.month,start.day,start.hour,start.minute,start.second

    traces = []
    for name in df.columns:
        tr = Trace(data=df[name].to_numpy())
        tr.stats.delta = delta
        tr.stats.starttime = UTCDateTime(*starttime)
        tr.stats.channel = name
        traces.append(tr)

    #combine the traces to make stream with all times
    stream = Stream(traces=traces)
    #slice stream to start and end times
    stream = stream.slice(t1,t2)
    #get the times that it is sampled
    times = stream[0].times()
    #get tides from model at these times
    epoch = stream[0].stats.starttime
    epoch = (epoch.year,epoch.month,epoch.day,epoch.hour,epoch.minute,epoch.second)
    tides = pyTMD.compute_tide_corrections(np.array([lon]),np.array([lat]),times,EPOCH=epoch,DIRECTORY=tide_dir,MODEL='CATS2008',TIME='UTC',TYPE='time series',METHOD='spline',EPSG='4326')
    #add the tides as another trace in this stream
    tide_tr = Trace(data=tides.flatten())
    tide_tr.stats.delta = delta
    tide_tr.stats.starttime = UTCDateTime(t1.year,t1.month,t1.day,t1.hour,t1.minute,t1.second)
    tide_tr.stats.channel = 'TIDES'
    stream += Stream(traces=[tide_tr])

    return stream

def TidalStream(t1,t2,delta,tide_dir='/Users/jmagyar/Documents/tides',lat=-68.70793,lon=78.10155):
    duration = t2 - t1
    starttime = t1.year,t1.month,t1.day,t1.hour,t1.minute,t1.second

    times = np.mgrid[0:duration:delta] #get the times to compute the tides at
    tides = pyTMD.compute_tide_corrections(np.array([lon]),np.array([lat]),times,EPOCH=starttime,DIRECTORY=tide_dir,MODEL='CATS2008',TIME='UTC',TYPE='time series',METHOD='spline',EPSG='4326')

    tide_tr = Trace(data=tides.flatten())
    tide_tr.stats.delta = delta
    tide_tr.stats.starttime = UTCDateTime(*starttime)
    tide_tr.stats.channel = 'TIDES'

    stream = Stream(traces=[tide_tr])
    return stream




#back azimuth sectors based on arrival time at three broadband seismometers.

def SorsdalDirections(t1,t2,c_path,groups=False,area_norm=False):
    """
    Plots the directional distribution of events for the Sorsdal broadband seismometers.
    """
    
    offset = 5 #first sector is 5 degrees off north
    angles = [80,45,55,80,45,55] #opening angle of each sector
    
    right = []
    for angle in angles:
        right.append(offset)
        offset += angle
    
    #now convert everything to radians
    angles = np.deg2rad(np.array(angles))
    right = np.deg2rad(np.array(right))
    
    #we now need the counts for each sector
    chunk = vis.SeismicChunk(t1,t2)
    mod_cat = chunk.get_classification(c_path)
    
    if groups == False:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111,projection='polar')
        ax.set_theta_zero_location('N')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        counts = mod_cat['arrival'].value_counts()
        max_count = np.max(counts.to_numpy())
        ax.set_ylim((0,max_count))
        values = np.array([counts['965'],counts['956'],counts['596'],counts['569'],counts['659'],counts['695']])
        ax.bar(right,values,width=angles,bottom=0.1*max_count,align='edge',color='grey')
    
    if groups == True:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111,projection='polar')
        ax.set_theta_zero_location('N')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        cumulative = np.zeros(6)
        i = 0
        c = ['maroon','red','orange','pink','yellow','purple','blue','green','grey']
        for group in range(1,10):
            group_cat = mod_cat[mod_cat['group']==group]
            counts = group_cat['arrival'].value_counts().to_dict()
            for key in ['965','956','596','569','659','695']:
                counts.setdefault(key,0)
            values = np.array([counts['965'],counts['956'],counts['596'],counts['569'],counts['659'],counts['695']])
            ax.bar(right,values,width=angles,bottom=cumulative,align='edge',color=c[i],alpha=0.75)
            
            cumulative += values
            
            i += 1
    
    if groups == 'separate':
        fig = plt.figure(figsize=(10,4))
        grid_spec = fig.add_gridspec(ncols=5,nrows=2,width_ratios=[3,1,1,1,1])
        comb_ax = fig.add_subplot(grid_spec[:,0],projection='polar')
        comb_ax.set_theta_zero_location('N')
        comb_ax.set_xticks([])
        comb_ax.set_yticks([])
        comb_ax.spines['polar'].set_visible(False)
        comb_ax.set_title('All Groups')
        
        cumulative = np.zeros(6)
        i = 0
        c = ['maroon','red','darkorange','hotpink','gold','purple','blue','darkgreen']
        for group in range(1,9):
            group_cat = mod_cat[mod_cat['group']==group]
            counts = group_cat['arrival'].value_counts().to_dict()
            for key in ['965','956','596','569','659','695']:
                counts.setdefault(key,0)
            values = np.array([counts['965'],counts['956'],counts['596'],counts['569'],counts['659'],counts['695']])
            
            if area_norm:
                values = values.astype(np.float64)
                values /= angles
            comb_ax.bar(right,values,width=angles,bottom=cumulative,align='edge',color=c[i],alpha=0.75)
            
            cumulative += values
            
            sing_ax = fig.add_subplot(grid_spec[i//4,i%4 + 1],projection='polar')
            sing_ax.set_theta_zero_location('N')
            sing_ax.set_xticks([])
            sing_ax.set_yticks([])
            sing_ax.spines['polar'].set_visible(False)
            sing_ax.bar(right,values,width=angles,align='edge',color=c[i],alpha=0.75)
            sing_ax.set_title('Group ' + str(group))
            
            i += 1
            
    return fig
        