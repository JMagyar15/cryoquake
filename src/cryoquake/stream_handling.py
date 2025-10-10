from obspy.signal.detrend import spline
import copy
from obspy import Stream, Trace, UTCDateTime, read
import numpy as np
import glob
import os
import pandas as pd
from obspy.core.inventory import inventory, read_inventory
from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader
import warnings
from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
from obspy.core.util import create_empty_data_chunk


"""
LOADING THE SEISMIC DATA CODE
"""

__chunklength_in_sec = 86400


def get_waveforms(network, station, location, channel, starttime, endtime, event_buffer=3600, waveform_name='waveforms', station_name='stations', providers=['IRIS', 'LMU', 'GFZ'], user=None, password=None, download=True, fill=0,print_s=True):
    """
    Function to download waveform data from an online seismic repository and save to a local directory or external drive. The waveform data are split into files containing a single day of data to enable fast recall and storage across multiple drives. The function checks if data for the requested station, channels (components) and time period are present in the specified local directory before attempting to download new waveform data. The requested waveform data are output as a single obspy Stream object.
        
    Parameters
    ----------
    network : str
        A single SEED or data center defined network code; wildcards are not allowed.
    station : str
        A single SEED station code; wildcards are not allowed.
    location : str
        A single SEED location identifier (often blank or ’0’); wildcards are not allowed.
    channel : str or list
        One or more SEED channel codes; multiple codes are entered as a list; e.g. [’BHZ’, ’HHZ’]; wildcards are not allowed.
    starttime : UTCDateTime
        Start time series on the specified start time (or after that time in the case of a data gap).
    endtime : UTCDateTime
        End time series (one sample) before the specified end time.
    event_buffer : float, optional
        Minimum duration of data to buffer before and after the requested time period; expected units are seconds. This is used to capture the full length of events that extend beyond the time period. The default value is 3600 seconds.
    waveform_name : str or path, optional
        Path to directory to read (check for) and write waveform data (an existing file of the same name will not be overwritten). The default location is a directory named waveforms in the working directory.
    station_name : str or path, optional
        Path to directory to read (check for) and write station data (location coordinates, elevation etc, as provided by data repository). The default location is a directory named stations in the working directory.
    providers : str or list, optional
        One or more clients to use to download the requested waveform data if it does not already exist in the specified local directory. Multiple clients are entered as a list; e.g. [’IRIS’, ’GFZ’]. By default IRIS, LMU and GFZ are queried.
    user : str, optional
        User name of HTTP Digest Authentication for access to restricted data.
    password : str, optional
        Password of HTTP Digest Authentication for access to restricted data.
    download : bool, optional
        Specify whether days with missing waveform data are to be downloaded from client; e.g. True or False, alternatively 1 or 0. Missing data are downloaded by default.
        
    Returns
    -------
    stream : Stream
        A stream object with one or more traces for each component at the requested seismometer.
    """
    # create empty stream to store amplitude of three component waveform
    stream = Stream()

    # read-in waveform data from downloaded files
    if isinstance(channel, list):
        for i in range(0, len(channel)):
            if i == 0:
                stream = __get_waveforms(network, station, location, channel[0], starttime - event_buffer, endtime + event_buffer, waveform_name=waveform_name, station_name=station_name, providers=providers, user=user, password=password, download=download)
            else:
                stream += __get_waveforms(network, station, location, channel[i], starttime - event_buffer, endtime + event_buffer, waveform_name=waveform_name, station_name=station_name, providers=providers, user=user, password=password, download=download)
    else:
        stream = __get_waveforms(network, station, location, channel, starttime - event_buffer, endtime + event_buffer, waveform_name=waveform_name, station_name=station_name, providers=providers, user=user, password=password, download=download)

    # merge different days in the stream at the same seismograph and channel; this prevents masked arrays from being created
    stream.merge(method=0,fill_value=fill)
    # sort channels into ZNE order for use in some obspy functions (only used for some attributes)
    stream.sort(keys=['network', 'station', 'location', 'channel'], reverse=True)
    # truncate stream at requested time window with buffer either side
    #stream = stream.slice(starttime - event_buffer, endtime + event_buffer)
    stream.trim(starttime - event_buffer, endtime + event_buffer,pad=True,fill_value=fill)
    
    # check all traces are the same length and start at the same time
    for i in range(0, len(stream)):
        if not (stream[i].stats.starttime == stream[0].stats.starttime and len(stream[i].data) == len(stream[0].data)):
            warnings.filterwarnings('always', category=UserWarning)
            warnings.warn('Stream has one or more components with inconsistent start and end times! Download the data again if it exists or select a valid time period.', category=UserWarning)
            warnings.filterwarnings('ignore', category=Warning)

    if print_s:
        print(stream)
    return stream

def __get_waveforms(network, station, location, channel, t1, t2, waveform_name='waveforms', station_name='stations', providers=['IRIS', 'LMU', 'GFZ'], user=None, password=None, download=True):
    """
    Private function for get_waveforms() to read existing waveform data on the user computer or download missing waveform data from a client.
    """
    # create empty stream to store waveform
    stream = Stream()

    # set start and end time of each file; these start and end on calendar dates
    start_time = UTCDateTime(t1.year, t1.month, t1.day)
    end_time = start_time + __chunklength_in_sec

    # read-in waveform data from downloaded files
    while (start_time < t2):
        filename = os.path.join(os.path.join(os.getcwd(), waveform_name), network+'.'+station+'.'+location+'.'+channel+'__'+start_time.strftime("%Y%m%dT%H%M%SZ")+'__'+end_time.strftime("%Y%m%dT%H%M%SZ")+'.mseed')

        # if file exists add to stream
        if os.path.isfile(filename):
            stream += read(filename)
        # otherwise attempt to download file then read-in if data exists
        else:
            if download == True:
                __download_waveforms(network, station, location, channel, start_time, end_time, waveform_name=waveform_name, station_name=station_name, providers=providers, user=user, password=password)
                if os.path.isfile(filename):
                    stream += read(filename)
            else:
                # issue warning that file is not available in local directory
                warnings.filterwarnings('always', category=UserWarning)
                #warnings.warn(filename+' not available in local directory.', category=UserWarning)
                warnings.filterwarnings('ignore', category=Warning)

        # update start and end time of each file
        start_time += __chunklength_in_sec
        end_time += __chunklength_in_sec
            
    return stream

def __download_waveforms(network, station, location, channel, t1, t2, waveform_name='waveforms', station_name='stations', providers=['IRIS', 'LMU', 'GFZ'], user=None, password=None):
    """
    Private function for get_waveforms() to download missing waveform data from a client.
    """
    # specify rectangular domain containing any location in the world.
    domain = RectangularDomain(minlatitude=-90, maxlatitude=90, minlongitude=-180, maxlongitude=180)

    # apply restrictions on start/end times, chunk length, station name, and minimum station separation
    restrictions = Restrictions(
        starttime=t1,
        endtime=t2,
        chunklength_in_sec=__chunklength_in_sec,
        network=network, station=station, location=location, channel=channel,
        reject_channels_with_gaps=False,
        minimum_length=0.0,
        minimum_interstation_distance_in_m=100.0)

    # download requested waveform and station data to specified locations
    if isinstance(providers, list):
        if (not user == None) and (not password == None):
            client = []
            for provider in providers:
                client.append(Client(provider, user=user, password=password))
            mdl = MassDownloader(providers=client)
        else:
            mdl = MassDownloader(providers=providers)
    else:
        if (not user == None) and (not password == None):
            mdl = MassDownloader(providers=[Client(providers, user=user, password=password)])
        else:
            mdl = MassDownloader(providers=[providers])
    mdl.download(domain, restrictions, mseed_storage=waveform_name, stationxml_storage=os.path.join(station_name, '{network}.{station}.'+location+'.'+channel+'.xml'))


def inv_to_waveforms(inv,t1,t2,w_path,buffer=3600,print_s=True,fill=0):
    """
    Use an inventory to load pre-downloaded waveforms.
    """
    stream = Stream()
    for net in inv:
            for station in net:
                for channel in station:
                    trace = get_waveforms(net.code,station.code,channel.location_code,channel.code,t1,t2,event_buffer=buffer,waveform_name=w_path,download=False,print_s=print_s,fill=fill)
                    if len(trace) > 0:
                        stream += trace
                    else:
                        stream += Trace(data=create_empty_data_chunk(1,'f'),header={'network':net.code,'station':station.code,'location_code':channel.location_code,'channel':channel.code,'starttime':t1-buffer,'sampling_rate':channel.sample_rate}).trim(t1-buffer,t2+buffer,pad=True,fill_value=None)
    return stream

"""
PRE PROCESSING OF SEISMIC STREAM
"""


def FullProcessing(inv,starttime, endtime,raw_path, process_path,freq=100,pre_filt=[0.001,0.005,45,50]):
    """
    Takes the raw data for a given station between a start and end time and removes the instrument response, resamples, then saves
    as a processed file in the same format as the raw data.
    """
    for network in inv:
        for station in network:
            for channel in station:
                c_id = network.code + '.' + station.code + '..' + channel.code
                stream = __get_stream(c_id,raw_path,starttime,endtime)
                #now do the processing of the entire stream in a single chunk
                __process_chunk(stream,inv,freq,pre_filt)
                __save_chunk(stream,c_id,process_path,starttime,endtime)
    return stream


def RemoveResponse(inv,t1,t2,w_path,p_path,freq=100,pre_filt=[0.5,1.0,45,50]):

    if not os.path.exists(p_path):
        os.mkdir(p_path)

    starttime = UTCDateTime(t1.year,t1.month,t1.day)
    endtime = starttime + __chunklength_in_sec

    while starttime < t2:
        stream = inv_to_waveforms(inv,starttime,endtime,w_path,buffer=3600,print_s=False,fill=None)
        stream.split()
        stream.remove_response(inventory=inv,zero_mean=True,pre_filt=pre_filt)
        stream.resample(sampling_rate=freq)
        stream.trim(starttime,endtime)
        stream.merge()

        for trace in stream:

            filename = os.path.join(p_path, trace.id+'__'+starttime.strftime("%Y%m%dT%H%M%SZ")+'__'+endtime.strftime("%Y%m%dT%H%M%SZ")+'.mseed')
            try:
                trace.write(filename,format='MSEED')
            except:
                print(trace.id,'has no data for',starttime,'-',endtime,': skipping to next time segment.')
        starttime += __chunklength_in_sec
        endtime += __chunklength_in_sec




def __get_stream(c_id,path,t1,t2):
    stream = Stream()

    # set start and end time of each file; these start and end on calendar dates
    start_time = UTCDateTime(t1.year, t1.month, t1.day)
    end_time = start_time + __chunklength_in_sec

    # read-in waveform data from downloaded files
    while (start_time < t2):
        filename = os.path.join(path,c_id+'__'+start_time.strftime("%Y%m%dT%H%M%SZ")+'__'+end_time.strftime("%Y%m%dT%H%M%SZ")+'.mseed')

        if os.path.isfile(filename):
            stream += read(filename)
        
        start_time += __chunklength_in_sec
        end_time += __chunklength_in_sec
    stream.merge().split()
    return stream

def __process_chunk(stream,inv,freq,pre_filt):
    stream.remove_response(inventory=inv,zero_mean=True,pre_filt=pre_filt)
    stream.resample(freq)
    return stream

def __save_chunk(stream,c_id,path,t1,t2):
    if not os.path.exists(path):
        os.mkdir(path)
    start_time = UTCDateTime(t1.year, t1.month, t1.day)
    end_time = start_time + __chunklength_in_sec

    # read-in waveform data from downloaded files
    while (start_time < t2):
        filename = os.path.join(path, c_id+'__'+start_time.strftime("%Y%m%dT%H%M%SZ")+'__'+end_time.strftime("%Y%m%dT%H%M%SZ")+'.mseed')
        day = stream.slice(start_time,end_time)
        try:
            day.write(filename,format='MSEED')
        except:
            print(c_id,'has no data for',start_time,'-',end_time,': skipping to next time segment.')
            
      
        # update start and end time of each file
        start_time += __chunklength_in_sec
        end_time += __chunklength_in_sec

def __process_day(network, station, location, channel, t1, t2, raw_path, process_path, inv, freq, pre_filt):
    """
    Private function for get_waveforms() to read existing waveform data on the user computer or download missing waveform data from a client.
    """
    # create empty stream to store waveform
    stream = Stream()

    # set start and end time of each file; these start and end on calendar dates
    start_time = UTCDateTime(t1.year, t1.month, t1.day)
    end_time = start_time + __chunklength_in_sec

    # read-in waveform data from downloaded files
    while (start_time < t2):
        filename = os.path.join(raw_path, network+'.'+station+'.'+location+'.'+channel+'__'+start_time.strftime("%Y%m%dT%H%M%SZ")+'__'+end_time.strftime("%Y%m%dT%H%M%SZ")+'.mseed')
        new_file = os.path.join(process_path, network+'.'+station+'.'+location+'.'+channel+'__'+start_time.strftime("%Y%m%dT%H%M%SZ")+'__'+end_time.strftime("%Y%m%dT%H%M%SZ")+'.mseed')

        if os.path.isfile(filename):
            day = read(filename)
            #! now process this day of seismic data
            day.detrend(type='simple') #use simple to try and avoid discontinuities.
            day.remove_response(inventory=inv,zero_mean=False,pre_filt=pre_filt)
            #TODO change the data type to make the files smaller (downloaded files are as int so smaller).
            day.resample(freq)
            #! save stream to file with same name but different folder
            try:
                day.write(new_file,format='MSEED')
            except:
                os.mkdir(process_path)
                day.write(new_file,format='MSEED')
            stream += day
      
        # update start and end time of each file
        start_time += __chunklength_in_sec
        end_time += __chunklength_in_sec
            
    return stream

def PreProcess(stream,inv,freq,pre_filt=[0.001,0.005,45,50]):
    """
    Takes a stream and detrends, demeans, tapers, deconvolves the instrument response, and resamples at a given sampling frequency.
    """
    p_stream = Stream()
    for tr in stream:
        print('Processing trace',tr.id)
        splits = tr.split()
        #splits.detrend(type='spline',order=1,dspline=1000)
        splits.remove_response(inventory=inv,zero_mean=True,pre_filt=pre_filt) #TODO try applying simple filter to chunks before removing response and do not use zero-mean to keep continuous.
        splits.merge()
        try:
            splits[0].data[splits[0].data.mask] = np.nan
            splits[0].data.set_fill_value(np.nan) #! should not need to do this twice, but has plotting issues otherwise...
        except:
            print(tr.id,'has no data gaps for this time period.')

        p_stream += splits
    
    p_stream.resample(freq)
    return p_stream

def TemporalCoverage(stream):
    times = pd.DataFrame(columns=['ID','Start','End'])
    for tr in stream:
        splits = tr.split()
        for s_tr in splits:
            start = s_tr.stats.starttime
            end = s_tr.stats.endtime
            tr_id = s_tr.id
            new_row = {'ID':tr_id,'Start':start,'End':end}
            times = times.append(new_row,ignore_index=True)
    return times

def PlotTempCoverage(times,inv,figsize=(10,10)):
    fig, ax = plt.subplots(figsize=figsize)
    i = 0
    labels = []
    coords = []
    for net in inv:
        for station in net:
            i -= 1
            for channel in station:
                c_id = net.code + '.' + station.code + '..' + channel.code
                if c_id[-1] == 'Z':
                    color = 'black'
                if c_id[-1] == 'N':
                    color = 'purple'
                if c_id[-1] == 'E':
                    color = 'darkorange'
                labels.append(c_id)
                subset = times.loc[times['ID'] == c_id]
                start = subset['Start'].to_numpy()
                end = subset['End'].to_numpy()
                duration = end - start

            
                ax.barh(i,duration,left=start,color=color)
                coords.append(i)
                i -= 1
    plt.gca().xaxis_date('UTC')
    plt.gcf().autofmt_xdate()
    ax.set_yticks(coords)
    ax.set_yticklabels(labels)
    return fig, ax


def detrend_stream(stream,dspline=10000):
    """
    Takes raw waveform data and uses splines to detrend. Both original and detrended stream are returned
    """
    o_stream = copy.deepcopy(stream)
    d_stream = stream
    for i, tr in enumerate(stream):
        d_tr = spline(tr.data,order=2,dspline=dspline)
        d_stream[i].data = d_tr
    return d_stream, o_stream


def euclidean_stream(stream,type='amplitude'):
    all_stations = split_seismometers(stream)
    euclid = group_components(all_stations,signal_type=type)
    new_stream = None
    for i in range(0, len(euclid)):
        if i == 0:
            new_stream = euclid[0]
        else:
            new_stream = new_stream + euclid[i]
    return new_stream


def split_seismometers(stream):
    """
    Private function for get_events() to separate each seismometer into a unique Stream comprising all channels recorded at that seismometer.
    """
    # create lists to store streams for each seismometer (if applicable)
    component_list = [] # amplitude of each component

    # create a copy of the input stream with components added in quadrature
    for i in range(0, len(stream)):
        # create new stream object to store combined components
        if i == 0:
            component_list.append(Stream(stream[0].copy()))
        else:
            for j in range(0, len(component_list)):
                # test if current seismometer has a stream in the list
                if (stream[i].stats.network == component_list[j][0].stats.network and stream[i].stats.station == component_list[j][0].stats.station and (stream[i].stats.channel)[0:-1] == (component_list[j][0].stats.channel)[0:-1]):
                    # seismometer in the list
                    component_list[j] += Stream(stream[i].copy())
                    break
            else:
                # seismometer not in the list; so add it
                component_list.append(Stream(stream[i].copy()))
                
    return component_list


def group_components(component_list, signal_type='amplitude'):
    """
    Function to calculate the Euclidean norm of the waveform amplitude for seismometers with multiple component mea- surements. The normal can be returned as an absolute value amplitude waveform or an energy waveform.
    
    Parameters
    ----------
    stream : Stream
        Stream containing waveform data for each component from one or more seismometers between the start and end time of the event.
    signal_type : str, optional
        Specify whether components are grouped as an ‘amplitude’ (i.e. absolute value) or ‘energy’ (i.e. amplitude-squared) waveform. The components are grouped as an amplitude waveform by default.
        
    Returns
    -------
    stream_list : list
        Return a list of streams for each seismometer with the data representing the amplitude or energy of the waveform measured by taking the normal of the signal from each component. The first stream is accessed as group_components(...)[0] and the trace of that stream as group_components(...)[0][0].
    """
    # convert stream to a list if it is a Stream object for a single seismometer
    if not isinstance(component_list, (list, np.ndarray)):
        component_list = [component_list]
    
    # create lists to store streams for each seismometer (if applicable)
    stream_list = [] # total amplitude (or energy)
    
    # combine traces at each seismometer in quadrature as appropriate
    for i in range(0, len(component_list)):
        # find latest start time and earlest stop time across the compenents at the given seismometer
        starttime, endtime = 0, 1e99
        for j in range(0, len(component_list[i])):
            if component_list[i][j].stats.starttime > starttime:
                starttime = component_list[i][j].stats.starttime
            if component_list[i][j].stats.endtime < endtime:
                endtime = component_list[i][j].stats.endtime
                
        # find weighted mean for the seismometer and component in this trace
        mean_list = []
        for j in range(0, len(component_list[i])):
            count = len(component_list[i][j].slice(starttime, endtime).data)
            mean = np.sum(component_list[i][j].slice(starttime, endtime).data)
            mean_list.append(float(mean)/count)
    
        for j in range(0, len(component_list[i])):
            # create new stream object to store combined components
            if j == 0:
                new_stream = Stream(component_list[i][0].slice(starttime, endtime))
                new_stream[0].data = (component_list[i][0].slice(starttime, endtime).data - mean_list[0])**2
                stream_list.append(new_stream)
            else:
                # add additional components to stream in quadrature
                stream_list[i][0].data += (component_list[i][j].slice(starttime, endtime).data - mean_list[j])**2
                # modify trace id to terminate in number of components
                stream_list[i][0].stats.channel = stream_list[i][0].stats.channel + component_list[i][j].stats.channel
                
        # if requested output is amplitude convert data to amplitudes
        if signal_type == 'amplitude':
            stream_list[i][0].data = np.sqrt(stream_list[i][0].data)
        
    return stream_list



def make_windows(t1,t2,days=5):
    """
    Split start and end time of stream into smaller segments so that event detection can be parallelised.
    Once split, each core can deal with a smaller period of time.
    """
    delta = days * 24 * 60 * 60 #number of seconds in each time window
    t = t1
    t_lst = [] #list starts with initial time
    while t < t2:
        t_lst.append(t)
        t += delta
    t_lst.append(t2) #list ends with final time
    return t_lst


def PolesZeros(station):
    inv = inventory.read_inventory(station,level='response')
    resp = inv[0][0][0].response
    paz = resp.get_paz()
    paz_dict = {'gain': paz.stage_gain,
       'poles': paz.poles,
       'sensitivity': resp.instrument_sensitivity.value,
       'zeros': paz.zeros}
    return paz_dict, inv