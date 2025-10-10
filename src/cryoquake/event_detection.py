from cryoquake.data_objects import SeismicChunk
import os
from obspy.core import UTCDateTime, Stream
import numpy as np
import pandas as pd
from obspy.geodetics import locations2degrees, degrees2kilometers
from obspy.signal.trigger import trigger_onset
import warnings
from obspy.signal.cross_correlation import correlation_detector


class ChunkDetection(SeismicChunk):
    #TODO can elliminate the while loop using new SeismicChunk iterator, get filtering back out
    #TODO of here so can be done flexibly in script.

    #? may want to keep while loop for naming convention, even if generally only giving day long chunk.

    def detect_events(self,c_path,**options):

        if not os.path.exists(c_path):
            os.mkdir(c_path) #make the catalogue directory if it has not yet been made

        day_start = UTCDateTime(self.starttime.year,self.starttime.month,self.starttime.day)
        day_end = day_start + 24*60*60

        while day_start < self.endtime:
            print('Detecting events for ',day_start.strftime("%Y-%m-%d"),' - ',day_end.strftime("%Y-%m-%d"))
            if len(self.stream) >= 1: #only try event detection if there is data in this period

                events, traces = get_events(self.stream,day_start,day_end,self.inv,**options)
                events['group'] = 0

                attributes = events[['event_id','group']]
            
                events.to_csv(os.path.join(c_path,'events__'+day_start.strftime("%Y%m%dT%H%M%SZ")+'__'+day_end.strftime("%Y%m%dT%H%M%SZ")+'.csv'))
                traces.to_csv(os.path.join(c_path,'traces__'+day_start.strftime("%Y%m%dT%H%M%SZ")+'__'+day_end.strftime("%Y%m%dT%H%M%SZ")+'.csv'))
                attributes.to_csv(os.path.join(c_path,'attributes__'+day_start.strftime("%Y%m%dT%H%M%SZ")+'__'+day_end.strftime("%Y%m%dT%H%M%SZ")+'.csv'))
            
            day_start += 24*60*60
            day_end += 24*60*60


    def match_template(self,event,thresh=0.5,dist=10):
        """
        Give a SeismicEvent object to use as a template, and search the chunk for events with sufficient correlation.
        """
        from obspy.core import Stream
        from copy import deepcopy
        templates = [Stream([tr]) for tr in event.get_data_window().select(channel='??Z')]
        template_names = [tr.id for tr in event.get_data_window().select(channel='??Z')]
        from obspy.signal.cross_correlation import correlation_detector
        detections, _ = correlation_detector(self.stream, templates, thresh, dist, template_names=template_names)

        matches = []
        for detection in detections:
            #make new SeismicEvents from these matches
            new_event = deepcopy(event)
            new_event.window_start = detection['time']
            new_event.window_end = new_event.window_start + event.window_length
            new_event.triggers = False #have not got trigger times for this event
            new_event.starttime = detection['time'] + event.buffer
            new_event.endtime = new_event.starttime + event.duration
            new_event.event_id = '{:0>4d}'.format(new_event.starttime.year)+'{:0>2d}'.format(new_event.starttime.month)+'{:0>2d}'.format(new_event.starttime.day)+'T'+'{:0>2d}'.format(new_event.starttime.hour)+'{:0>2d}'.format(new_event.starttime.minute)+'{:0>2d}'.format(new_event.starttime.second)+'Z'
            matches.append(new_event)
        return matches
    
    def template_catalogue(self,c_path,template_dict,threshold_dict,dist=10,method='mean',separate_traces=False,num=None):
        """
        Produce an event catalogue in the same format as with STA/LTA but using the given template events and thresholds
        """

        if not os.path.exists(c_path):
            os.mkdir(c_path) #make the catalogue directory if it has not yet been made

        day_start = UTCDateTime(self.starttime.year,self.starttime.month,self.starttime.day)
        day_end = day_start + 24*60*60

        while day_start < self.endtime:
            print('Detecting events for ',day_start.strftime("%Y-%m-%d"),' - ',day_end.strftime("%Y-%m-%d"))
            if len(self.stream) >= 1: #only try event detection if there is data in this period
                template_streams = []
                thresholds = []
                template_names = []

                for template_name, template in template_dict.items(): #TODO make option for dealing with traces in event separately (Sorsdal) or together (Totten).

                    if separate_traces:
                        template_streams += [Stream([tr]) for tr in template.get_data_window().select(channel='??Z')]
                        thresholds += [threshold_dict[template_name] for tr in template.get_data_window().select(channel='??Z')]
                        template_names += [template_name for tr in template.get_data_window().select(channel='??Z')]
                    else:
                        template_streams += [template.get_data_window().select(channel='??Z')]
                        thresholds += [threshold_dict[template_name]]
                        template_names += [template_name]

                #TODO make mean, min, and max cross-correlation methods for the detector.
                split_stream = self.stream.split()

                if method == 'nanmean':
                    detections, _ = correlation_detector(split_stream,template_streams,thresholds,dist,template_names=template_names,similarity_func=_calc_nanmean)
                elif method == 'n_mean':
                    if num == None:
                        num = 3
                    def simf(ccs):
                        return _calc_n_mean(ccs, num)
                    detections, _ = correlation_detector(split_stream,template_streams,thresholds,dist,template_names=template_names,similarity_func=simf)

                else:
                    detections, _ = correlation_detector(split_stream,template_streams,thresholds,dist,template_names=template_names) 

                event_rows = []
                trace_rows = []
                attribute_rows = []
                #now fill in the catalogue dataframes with the results of the template matching - these should match the outputs of the STA/LTA detection.
                for detection in detections:

                    template = template_dict[detection['template_name']]
                    stations = template.stations
                    starttime = detection['time'] + template.buffer
                    duration = template.duration
                    event_id = '{:0>4d}'.format(starttime.year)+'{:0>2d}'.format(starttime.month)+'{:0>2d}'.format(starttime.day)+'T'+'{:0>2d}'.format(starttime.hour)+'{:0>2d}'.format(starttime.minute)+'{:0>2d}'.format(starttime.second)+'Z'

                    if (starttime >= day_start) & (starttime < day_end): #for if a buffer is used for filtering.
                        event_dict = {}
                        event_dict['event_id'] = event_id
                        event_dict['stations'] = stations
                        event_dict['ref_time'] = starttime
                        event_dict['ref_duration'] = duration
                        event_dict['group'] = detection['template_name']
                        event_rows.append(event_dict)

                        attribute_dict = {}
                        attribute_dict['event_id'] = event_id
                        attribute_dict['group'] = detection['template_name']
                        attribute_dict['similarity'] = detection['similarity']
                        attribute_rows.append(attribute_dict)

                        for i, trace in template.trace_rows.iterrows():
                            trace_dict = {}
                            trace_dict['event_id'] = event_id
                            trace_dict['station'] = trace.name
                            trace_dict['components'] = trace['components']
                            trace_dict['time'] = starttime
                            trace_dict['duration'] = duration
                            trace_rows.append(trace_dict)

                events = pd.DataFrame(event_rows,columns=['event_id','stations','ref_time','ref_duration','group'])
                traces = pd.DataFrame(trace_rows,columns=['event_id','station','components','time','duration'])
                attributes = pd.DataFrame(attribute_rows,columns=['event_id','group','similarity'])
            
                events.to_csv(os.path.join(c_path,'template_events__'+day_start.strftime("%Y%m%dT%H%M%SZ")+'__'+day_end.strftime("%Y%m%dT%H%M%SZ")+'.csv'))
                traces.to_csv(os.path.join(c_path,'template_traces__'+day_start.strftime("%Y%m%dT%H%M%SZ")+'__'+day_end.strftime("%Y%m%dT%H%M%SZ")+'.csv'))
                attributes.to_csv(os.path.join(c_path,'template_attributes__'+day_start.strftime("%Y%m%dT%H%M%SZ")+'__'+day_end.strftime("%Y%m%dT%H%M%SZ")+'.csv'))
            
            day_start += 24*60*60
            day_end += 24*60*60



def _calc_nanmean(stream):
    from obspy.core import Trace
    """
    Return trace with nanmean of traces in stream.
    """
    if len(stream) == 0:
        return stream
    matrix = np.array([tr.data for tr in stream])
    header = dict(sampling_rate=stream[0].stats.sampling_rate,
                  starttime=stream[0].stats.starttime)
    return Trace(data=np.nanmean(matrix, axis=0), header=header)

def _calc_n_mean(stream,num):
    from obspy.core import Trace
    """
    Return trace with nanmean of traces in stream.
    """
    if len(stream) == 0:
        return stream
    matrix = np.array([tr.data for tr in stream])
    header = dict(sampling_rate=stream[0].stats.sampling_rate,
                  starttime=stream[0].stats.starttime)
    
    matrix = np.sort(matrix,axis=0)
    top_n = matrix[-num:,:]

    return Trace(data=np.nanmean(top_n, axis=0), header=header)


"""
MINIMAL VERSION OF CATALOGUE GENERATION (WITHOUT AMPLITUDE/ENERGY INFORMATION, DO THIS LATER IN NEW ATTRIBUTE CALCULATION FRAMEWORK)
"""


def get_events(stream, starttime, endtime, inv, signal_type='amplitude', station_name='stations', trigger_type='recstalta', avg_wave_speed=2, thr_event_join=0.5, thr_coincidence_sum=-1, thr_on=5, thr_off=1, **options): # avg_wave_speed in km/s
    """
    Function to detect events from seismic waveform data using STA/LTA-type algorithms. The input waveforms are denoted by seismometer and channel (e.g. ‘BH?’, ‘HH?’, ‘LH?’); the signal from seismometers with multiple components (i.e. ‘Z’, ‘N’, ‘E’) are combined into a single waveform using the Euclidean norm. The triggering algorithm is applied to the resulting amplitude or energy waveform. Small gaps between triggered events are removed before the combined event details are written to the (reference) event catalogue. If multiple seismometers are present the user can specify the minimum number of seismometers on which an event must be detected for that event to be included in the catalogue.
    
    Parameters
    ----------
    stream : Stream
        Stream containing waveform data for each component from one or more seismometers.
    starttime : UTCDateTime
        Limit results to time series samples starting on the specified start time (or after that time in the case of a data gap).
    endtime : UTCDateTime
        Limit results to time series ending (one sample) before the specified end time.
    signal_type : str, optional
        Apply the event detection algorithm to the ‘amplitude’ (i.e. absolute value) or ‘energy’ (i.e. amplitude-squared) waveform. The event detection algorithm is applied to the amplitude waveform by default.
    station_name : str or path, optional
        Path to directory to read station data, specifically the GPS coordinates. The default location is a directory named stations in the working directory.
    trigger_type : str, optional
        The trigger algorithm to be applied (e.g. ‘recstalta’). See e.g. obspy.core.trace.Trace.trigger() for further details. The recursive STA/LTA algorithm is applied by default.
    avg_wave_speed : float, optional
        The speed at which seismic waves propagate through the local medium, contributing to a delay between detections in elements of a seismic array. The distance between the closest set of n (defined by the thr_coincidence_sum parameter) seismometers defines a critical distance the seismic waves must cover for coincidence triggering to detect events simultaneously at n seismometers. The default value for the average wave speed is 2 km/s.
    thr_event_join : float, optional
        The maximum duration of gaps between triggered events before those events are considered separate. Joined events are reported as a single event in the (reference) event catalogue. The maximum gap duration is assumed to be 0.5 seconds by default.
    thr_coincidence_sum : int, optional
        The number of seismometers, n, on which an event must be detected for that event to be included in the (reference) event catalogue. By default an event must be detected at every seismometer.
    thr_on : float, optional
        Threshold for switching single seismometer trigger on. The default value is a threshold of 5.
    thr_off : float, optional
        Threshold for switching single seismometer trigger off. The default value is a threshold of 1.
    options
        Necessary keyword arguments for the respective trigger algorithm that will be passed on. For example ‘sta’ and ‘lta’ for any STA/LTA variant (e.g. sta=3, lta=10). Arguments ‘sta’ and ‘lta’ (seconds) will be mapped to ‘nsta’ and ‘nlta’ (samples) by multiplying by the sampling rate of trace (e.g. sta=3, lta=10 would call the trigger algorithm with 3 and 10 seconds average, respectively).
        
    Returns
    -------
    events : DataFrame
        A pandas dataframe containing the events in the form of a (reference) event catalogue with eight columns including the reference start time and duration of the event based on coincidence triggering. The format of the event catalogue is detailed in the Event and Trace Catalogues section of this documentation.
    traces : DataFrame
        A pandas dataframe containing the trace (metadata) with eight columns including the start time and duration of the triggers for each trace based on single station triggering. The format of the trace (metadata) catalogue is detailed in the Event and Trace Catalogues section of the documentation.
    """
    # create a copy of the input stream separated into streams for each seismometer
    component_list = __group_seismometers(stream)
    # and with components added in quadrature (i.e. energy)
    stream_list = group_components(component_list, signal_type=signal_type)

    
    # create new stream of the quadrature streams from each seismometer
    new_stream = None
    for i in range(0, len(stream_list)):
        if i == 0:
            new_stream = stream_list[0]
        else:
            new_stream = new_stream + stream_list[i]

    if thr_coincidence_sum <= 0:
        thr_coincidence_sum = len(stream_list)
    else:
        thr_coincidence_sum = min(len(stream_list), thr_coincidence_sum)
    
    # get distances between array elements
    distance = __get_distances(stream, starttime, inv, thr_coincidence_sum=thr_coincidence_sum)

    # trigger events using specified event detection algorithm
    events, coincident_events = __coincidence_trigger(trigger_type=trigger_type, thr_on=thr_on, thr_off=thr_off, stream=new_stream, nseismometers=len(stream_list), thr_travel_time=distance/avg_wave_speed, thr_event_join=thr_event_join, thr_coincidence_sum=thr_coincidence_sum, **options)
    events_df, traces_df = __make_catalogues(coincident_events, stream, events, stream_list, starttime, endtime, signal_type, thr_travel_time=distance/avg_wave_speed, thr_coincidence_sum=thr_coincidence_sum)

    return events_df, traces_df

def __get_distances(stream, starttime, inv, thr_coincidence_sum=1):
    """
    Private function for get_events() to calcualte maximum distance between any seismometer in a given array and its closest (n-1) neighbours.
    """
    if thr_coincidence_sum <= 1:
        return 0
    else:
        # create list to store latitude and longitude of each unique seismometer
        coordinates_list = []
        seismometer_list = []
        
        # find coordinates of each unique seismometer, excluding channels
        for i in range(0, len(stream)):
            #filename = os.path.join(station_name, stream[i].stats.network+'.'+stream[i].stats.station+'.'+stream[i].stats.location+'.'+stream[i].stats.channel+'.xml')

            location = stream[i].stats.network+'.'+stream[i].stats.station+'.'+stream[i].stats.location
            if i == 0 or not np.any(location == np.asarray(seismometer_list)):
                try:
                    #inv = read_inventory(filename)
                    coordinates = inv.get_coordinates(stream[i].id) #! removed starttime input
                    coordinates['location'] = location
                    # append to lists
                    coordinates_list.append(coordinates)
                    seismometer_list.append(coordinates['location'])
                except:
                    warnings.filterwarnings('always', category=UserWarning)
                    warnings.warn(stream[i].stats.channel+' channel not available in local directory.', category=UserWarning)
        
        # calculate distances between each pair of seismometers
        distances_list = np.zeros((len(coordinates_list), len(coordinates_list)))
        for i in range(0, len(coordinates_list)):
            for j in range(i + 1, len(coordinates_list)):
                distances_list[i][j] = degrees2kilometers(locations2degrees(coordinates_list[i]['latitude'], coordinates_list[i]['longitude'], coordinates_list[j]['latitude'], coordinates_list[j]['longitude']))
        
        # calculate maximum distance between closest n seismometers
        if thr_coincidence_sum == 2:
            # distance between closest pair
            return np.min(distances_list[distances_list > 0])
        elif thr_coincidence_sum < len(coordinates_list) and thr_coincidence_sum <= 5:
            if thr_coincidence_sum == 3:
                # distance between closest set of three seismometers
                distances_matrix = np.zeros((len(coordinates_list), len(coordinates_list), len(coordinates_list)))
                for i in range(0, len(coordinates_list)):
                    for j in range(i + 1, len(coordinates_list)):
                        for k in range(j + 1, len(coordinates_list)):
                            distances_matrix[i][j][k] = np.max((distances_list[i][j], distances_list[i][k], distances_list[j][k]))
                return np.min(distances_matrix[distances_matrix > 0])
            elif thr_coincidence_sum == 4:
                # distance between closest set of four seismometers
                distances_matrix = np.zeros((len(coordinates_list), len(coordinates_list), len(coordinates_list), len(coordinates_list)))
                for i in range(0, len(coordinates_list)):
                    for j in range(i + 1, len(coordinates_list)):
                        for k in range(j + 1, len(coordinates_list)):
                            for l in range(k + 1, len(coordinates_list)):
                                distances_matrix[i][j][k] = np.max((distances_list[i][j], distances_list[i][k], distances_list[i][l], distances_list[j][k], distances_list[j][l], distances_list[k][l]))
                return np.min(distances_matrix[distances_matrix > 0])
            else:
                # distance between closest set of five seismometers
                distances_matrix = np.zeros((len(coordinates_list), len(coordinates_list), len(coordinates_list), len(coordinates_list), len(coordinates_list)))
                for i in range(0, len(coordinates_list)):
                    for j in range(i + 1, len(coordinates_list)):
                        for k in range(j + 1, len(coordinates_list)):
                            for l in range(k + 1, len(coordinates_list)):
                                for m in range(l + 1, len(coordinates_list)):
                                    distances_matrix[i][j][k] = np.max((distances_list[i][j], distances_list[i][k], distances_list[i][l], distances_list[i][m], distances_list[j][k], distances_list[j][l], distances_list[j][m], distances_list[k][l], distances_list[k][m], distances_list[l][m]))
                return np.min(distances_matrix[distances_matrix > 0])
        else:
            # too computationally inefficient, so use maximum distance in array
            return np.max(distances_list)

def __make_catalogues(events, stream, events_list, stream_list, starttime, endtime, signal_type='amplitude', thr_travel_time=0, thr_coincidence_sum=1):
    """
    Private function for get_events() to create reference and trace catalogues of the identified events.
    """
    ## REFERENCE CATALOGUE
    # output relevant columns to pandas dataframe
    events_df = pd.DataFrame(columns=['event_id', 'stations', 'network_time', 'ref_time', 'ref_duration'])
    if len(events) > 0:
        df = pd.DataFrame(events)[['time', 'duration', 'stations']]
    else:
        #raise Exception('No events found with given start/end times.') #! this has been edited from before to reflect what what raises exception.
        events_df = pd.DataFrame(columns=['event_id', 'stations', 'network_time', 'ref_time', 'ref_duration'])
        traces_df = pd.DataFrame(columns=['event_id', 'stations', 'network_time', 'time', 'duration'])
        return events_df, traces_df

    # remove events outside requested time window and less than 10 times the sampling rate
    df = df[np.logical_and(df['time'] + df['duration'] > starttime, df['time'] < endtime)]
    df = df[df['duration'] > 10./stream[0].stats.sampling_rate]
    df.reset_index(drop=True, inplace=True)
    
    #TODO can probably avoid the for loop here and vectorise this? Not sure whether this is the bottleneck or the trigger sorting as computing the characteristic function is pretty quick...
    # add columns with peak amplitude and energy of top N stations, and the event id
    for i in range(0, len(df['time'])):
        event_id = '{:0>4d}'.format(df['time'][i].year)+'{:0>2d}'.format(df['time'][i].month)+'{:0>2d}'.format(df['time'][i].day)+'T'+'{:0>2d}'.format(df['time'][i].hour)+'{:0>2d}'.format(df['time'][i].minute)+'{:0>2d}'.format(df['time'][i].second)+'Z'
        events_df.loc[i] = list([event_id, df['stations'][i], thr_travel_time, df['time'][i], df['duration'][i]])
    
    ## TRACE CATALOGUE
    # find start time and duration of event at each seismometer
    if isinstance(stream_list, (list, np.ndarray)):
        traces_df = pd.DataFrame(columns=['event_id', 'station', 'components', 'time', 'duration'])
        k = 0
        for i in range(0, len(stream_list)): #loop over each seismometer
            trace_df = pd.DataFrame(columns=['event_id', 'station', 'components', 'time', 'duration'])
            
            # create array of components in stream
            components = []
            for j in range(0, len(stream_list[i][0].stats.channel)):
                if int(j - 2) % 3 == 0:
                    components.append(stream_list[i][0].stats.channel[j - 2:j + 1]) #these are the 3 letter codes of the channels
            
            df = pd.DataFrame(events_list)[['time', 'duration', 'stations']]
            df = df[df['stations'] == stream_list[i][0].stats.network+'.'+stream_list[i][0].stats.station+'.'+stream_list[i][0].stats.location]
            df.reset_index(drop=True, inplace=True)

            l = 0
            # find all events within range of reference event, including times extending beyond reference event
            for j in range(0, len(events_df['ref_time'])):
                index = np.logical_and(np.any(np.asarray(events_df['stations'][j]) == stream_list[i][0].stats.network+'.'+stream_list[i][0].stats.station+'.'+stream_list[i][0].stats.location), np.logical_and(df['time'] <= events_df['ref_time'][j] + events_df['ref_duration'][j] + thr_travel_time/2., df['time'] + df['duration'] + thr_travel_time/2. >= events_df['ref_time'][j]))
                if np.sum(index) > 0:
                    trace_df.loc[l] = list([events_df['event_id'][j], df['stations'][0], components, np.min(df['time'][index]), np.max(df['time'][index] + df['duration'][index]) - np.min(df['time'][index])])
                    l = l + 1

            # add columns with peak amplitude and energy
            if len(trace_df) > 0:
                # append dataframe for this seismometer to final catalogue
                if k == 0:
                    traces_df = trace_df
                else:
                    traces_df = traces_df.append(trace_df, ignore_index=True)
                k = k + 1

    else:
        traces_df = events_df
        # rename columns to match expected format for trace catalogue
        traces_df.rename(columns = {'ref_time': 'time', 'ref_duration': 'duration'}, inplace = True)

    return events_df, traces_df

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
    
def __group_seismometers(stream):
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

def __coincidence_trigger(trigger_type, thr_on, thr_off, stream, nseismometers, thr_travel_time=0, thr_event_join=10, thr_coincidence_sum=0, trigger_off_extension=0, **options):
    """
    Private function for get_events(), based on the obspy coincidence_trigger function, to identify events based on simultaneous detections at n seismometers (with possible short gaps of duration thr_event_join). This function outputs two lists with the time, duration and station for (1) triggers at single seismometers and (2) detections of events.
    """
    st = stream.copy()
    # use all traces ids found in stream
    trace_ids = [tr.id for tr in st]
    # we always work with a dictionary with trace ids and their weights later
    if isinstance(trace_ids, list) or isinstance(trace_ids, tuple):
        trace_ids = dict.fromkeys(trace_ids, 1)

    # the single station triggering
    triggers = []
    single_triggers = []
    # prepare kwargs for trigger_onset
    kwargs = {'max_len_delete': False}
    for tr in st:
        if tr.id not in trace_ids:
            msg = "At least one trace's ID was not found in the " + \
                  "trace ID list and was disregarded (%s)" % tr.id
            warnings.warn(msg, UserWarning)
            continue
        if trigger_type is not None:
            tr.trigger(trigger_type, **options)

        max_trigger_length = 1e6
        kwargs['max_len'] = int(
            max_trigger_length * tr.stats.sampling_rate + 0.5)
        tmp_triggers = trigger_onset(tr.data, thr_on, thr_off, **kwargs)
        # find triggers for given station
        prv_on, prv_off = -1000, -1000
        for on, off in tmp_triggers:
            on = tr.stats.starttime + float(on) / tr.stats.sampling_rate
            off = tr.stats.starttime + float(off) / tr.stats.sampling_rate
            # extend previous event if only small gap
            if prv_on < 0:
                # update on and off times for first event
                prv_on = on
                prv_off = off
            elif on <= prv_off + thr_event_join:
                # update off time assuming continuing event
                prv_off = off
            else:
                # add previous trigger to catalogue
                triggers.append([prv_on.timestamp, prv_off.timestamp, tr.id])
                # add trigger to trace catalogue
                event = {}
                event['time'] = UTCDateTime(prv_on)
                event['stations'] = (tr.id).split(".")[0]+'.'+(tr.id).split(".")[1]+'.'+(tr.id).split(".")[2]
                event['trace_ids'] = tr.id
                event['coincidence_sum'] = 1.0
                event['duration'] = prv_off - prv_on
                single_triggers.append(event)
                # update on and off times
                prv_on = on
                prv_off = off
        # add final trigger to catalogue
        if prv_on > 0:
            triggers.append([prv_on.timestamp, prv_off.timestamp, tr.id])
            # add trigger to event catalogue
            event = {}
            event['time'] = UTCDateTime(prv_on)
            event['stations'] = (tr.id).split(".")[0]+'.'+(tr.id).split(".")[1]+'.'+(tr.id).split(".")[2]
            event['trace_ids'] = tr.id
            event['coincidence_sum'] = 1.0
            event['duration'] = prv_off - prv_on
            single_triggers.append(event)
    triggers.sort()

    # the coincidence triggering and coincidence sum computation
    coincidence_triggers = []
    last_off_time = [0.0]
    while triggers != []:
        # remove first trigger from list and look for overlaps
        on, off, tr_id = triggers.pop(0)
        on = on - thr_travel_time
        sta = (tr.id).split(".")[0]+'.'+(tr.id).split(".")[1]+'.'+(tr.id).split(".")[2]
        # add trigger to event catalogue
        event = {}
        event['time'] = [UTCDateTime(on)]
        event['off_time'] = [UTCDateTime(off)]
        event['stations'] = [tr_id.split(".")[0]+'.'+tr_id.split(".")[1]+'.'+tr_id.split(".")[2]]
        event['trace_ids'] = [tr_id]
        # compile the list of stations that overlap with the current trigger
        k = 0
        for trigger in triggers:
            tmp_on, tmp_off, tmp_tr_id = trigger
            tmp_sta = (tmp_tr_id).split(".")[0]+'.'+(tmp_tr_id).split(".")[1]+'.'+(tmp_tr_id).split(".")[2]
            if np.any(tmp_sta == np.asarray(event['stations'])):
                pass # station already included so do not add travel time again
            else:
                tmp_on = tmp_on - thr_travel_time
            # break if there is a gap in between the two triggers
            if tmp_on > off + trigger_off_extension: # place limit on number of triggers; must be within a small time of the last trigger
                break
            if k == 10*nseismometers**2:
                warnings.filterwarnings('always', category=UserWarning)
                #warnings.warn('Too many triggers joined together; consider looking at a smaller time window to improve computational efficiency.', category=UserWarning)
                warnings.filterwarnings('ignore', category=Warning)
            event['time'].append(UTCDateTime(tmp_on))
            event['off_time'].append(UTCDateTime(tmp_off))
            event['stations'].append(tmp_sta)
            event['trace_ids'].append(tmp_tr_id)
            # allow sets of triggers that overlap only on subsets of all stations (e.g. A overlaps with B and B overlaps w/ C => ABC)
            off = max(off, tmp_off)
            k = k + 1
        
        # find on and off time of first region with multiple triggers
        trigger_times = event['time'] + event['off_time']
        trigger_stations = event['stations'] + event['stations']
        trigger_traces = event['trace_ids'] + event['trace_ids']
        trigger_sum = np.asarray([1]*len(event['time']) + [-1]*len(event['off_time']))
        index = np.argsort(trigger_times.copy())
        
        # initialise variables
        coincidence_sum, event['coincidence_sum'], join_time, first_time = 0, 0, None, True
        event['stations'], event['trace_ids'] = [], []
        for i in range(0, len(index)):
            coincidence_sum = coincidence_sum + trigger_sum[index[i]]
            # coincidence sum region
            if coincidence_sum >= thr_coincidence_sum:
                # set start time if over threshold for the first time
                if first_time:
                    event['time'] = trigger_times[index[i]]
                    first_time = False
                # update end time
                event['off_time'] = trigger_times[index[i]]
                event['duration'] = event['off_time'] - event['time']
                # update maximum coincidence sum for detection
                event['coincidence_sum'] = max(coincidence_sum, event['coincidence_sum'])
                # add station and trace_id to event catalogue
                if trigger_sum[index[i]] > 0:
                    event['stations'].append(trigger_stations[index[i]])
                    event['trace_ids'].append(trigger_traces[index[i]])
                # reset join time if coincidence trigger condition met again
                join_time = None
            else:
                # before coincidence sum region
                if first_time:
                    # add station and trace_id to event catalogue and remove if it detriggers before coincidence sum region
                    if trigger_sum[index[i]] > 0:
                        event['stations'].append(trigger_stations[index[i]])
                        event['trace_ids'].append(trigger_traces[index[i]])
                    else:
                        event['stations'].remove(trigger_stations[index[i]])
                        event['trace_ids'].remove(trigger_traces[index[i]])
                # after coincidence sum region
                else:
                    # update end time
                    event['off_time'] = trigger_times[index[i]]
                    event['duration'] = event['off_time'] - event['time']
                    if join_time == None:
                        join_time = event['off_time']
                    elif (event['off_time'] - join_time) > thr_event_join:
                        # only join if at least one seismometer active
                        break
        # update end time and duration in case coincidence trigger did not join events
        if not join_time == None:
            event['off_time'] = join_time
            event['duration'] = join_time - event['time']
                    
        # remove duplicate stations and trace_ids (as applicable)
        event['stations'] = list(dict.fromkeys(event['stations']))
        event['trace_ids'] = list(dict.fromkeys(event['trace_ids']))

        # skip if both coincidence sum and similarity thresholds are not met
        if event['coincidence_sum'] < thr_coincidence_sum:
            continue
        # skip coincidence trigger if it is just a subset of the previous (determined by a shared off-time, this is a bit sloppy)
        if np.any(np.asarray(last_off_time) - float(event['off_time']) >= 0):
            continue
        
        # add event to catalogue and center times
        event['time'], event['off_time'] = event['time'] + thr_travel_time/2., event['off_time'] + thr_travel_time/2.
        coincidence_triggers.append(event)
        last_off_time.append(event['off_time'])
    
    # remove keys used in computation only
    for trigger in coincidence_triggers:
        trigger.pop('off_time')

    return single_triggers, coincidence_triggers