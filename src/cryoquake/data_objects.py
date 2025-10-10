import os
from cryoquake import stream_handling as sh
import numpy as np
import pandas as pd
from obspy.core import UTCDateTime, Stream, Trace
from obspy.core.util import create_empty_data_chunk
import copy

import warnings
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from matplotlib import rc


class SeismicEvent:
    """
    The base object used to define a single event from a catalogue.
    This is used to initialise the event, and do some basic signal processing
    that all other contexts may need. Regardless of the currect context of
    the event, these methods are always accessible as other contexts are
    children of this parent class.
    """
    def __init__(self,event_row,trace_rows):
        self.event_row = event_row
        trace_rows.set_index('station',inplace=True,drop=True)
        trace_rows = trace_rows[~trace_rows.index.duplicated(keep='first')]
        self.trace_rows = trace_rows
        self.event_id = event_row.name
        self.group = event_row['group']
        
        self.starttime = UTCDateTime(event_row['ref_time'])
        self.endtime = self.starttime + event_row['ref_duration']
        self.duration = event_row['ref_duration']
        
        self.stations = event_row['stations']
        self.triggers = True
        
        trace_starts = []
        trace_ends = []
        for i, trace in trace_rows.iterrows():
            trace_starts.append(UTCDateTime(trace['time']))
            trace_ends.append(UTCDateTime(trace['time'])+trace['duration'])
        
        #the reference event start and endtime are based on when sufficiently many stations are triggered
        #this means it will miss the first arrival if multiple triggers are needed
        #the window start and end are defined using the maximal event extent from each trigger on/off.
        self.window_start = np.min(np.array(trace_starts))
        self.window_end = np.max(np.array(trace_ends))
        self.window_length = self.window_end - self.window_start

        #initialise some attributes that will be used later to show that they have not yet been attached.
        self.stream = None
        self.inv = None

        self.attributes = pd.DataFrame(columns=['Name','Value','Label'])
        
    def context(self,new_context=None):
        """
        Change from the current context to a new one. This method is accessible regardless of the
        present context, with unknown or unspecified context arguments send back to this base class.

        Inputs:
            new_context: the context class to switch to. If none is provided, switches to the base
                SeismicEvent class. Currently implemented options are: 'plot', 'calculation', 
                'spectral', and 'beamforming'.
        """
        if new_context == 'plot':
            from iqvis.visualisation import EventPlot
            self.__class__ = EventPlot
        
        elif new_context == 'calculation':
            from iqvis.event_calculation import EventCalculation
            self.__class__ = EventCalculation #TODO actually make this object for event attribute calculations
        
        elif new_context == 'spectral':
            from iqvis.spec_analysis import EventSpectrum
            self.__class__ = EventSpectrum
        
        elif new_context == 'beamforming':
            from iqvis.spatial_analysis import EventBeampower
            self.__class__ = EventBeampower

        elif new_context == 'polarisation':
            from iqvis.spatial_analysis import EventPolarisation
            self.__class__ = EventPolarisation
        
        else:
            self.__class__ = SeismicEvent
        
    def attach_waveforms(self,inv,w_path,buffer=0,length=None,extra=5):
        """
        Finds the waveform data from a given directory for the stations in the inventory and attaches this to the object for later use.
        This needs to be called before any filtering or plotting can be applied.

        Inputs:
            inv: obspy inventory with stations to load waveforms for. 

            w_path: path to waveform file directory.

            buffer: a buffer time to add to the event catalogue window to help include event information prior to initial trigger time.

            length: closes the data window at a set duration after the event start for consistancy of length between events. If None, data window end
                is the event end plus the buffer time

            extra: additional waveform data to load on each side of the window of interest which is excluded from plots and analysis, but
                included in filtering to avoid edge artifacts. Default is 5 seconds (this is somewhat arbitrary).
        """
        if length == None:
            self.data_window = [self.window_start - buffer, self.window_end + buffer]
        else:
            self.data_window = [self.window_start - buffer, self.window_start + length]
        
        self.stream = sh.inv_to_waveforms(inv,self.data_window[0],self.data_window[1],w_path,print_s=False,buffer=extra)

        self.inv = inv
        self.buffer = buffer
        self.length = length
        self.extra = extra

    def assign_group(self,group_num,c_path):
        """
        Assign a group number to this event, and save this group number in the event catalogue.

        Inputs:
            group_num: the group number to assign to this event (could also potentially use letters or other descriptors)
            
            c_path: path to the catalogue files which will be accessed and modified with the new group number.
        """
        #TODO need to make a group column in the event catalogue that is zero (unassigned) by default and is then updated by this function.
        self.group = group_num
        #now need to load in the file with this event
        starttime = self.starttime #ref time of the event
        t1 = UTCDateTime(starttime.year,starttime.month,starttime.day) #start of day of the event
        t2 = t1 + 24*60*60 #end of day with event

        all_day = pd.read_csv(os.path.join(c_path,'events__'+t1.strftime("%Y%m%dT%H%M%SZ")+'__'+t2.strftime("%Y%m%dT%H%M%SZ")+'.csv'),index_col=0)
        #change the group number for the event
        ind = all_day[all_day['event_id'] == self.event_id].index[0] #index in day corresponding to event
        all_day.at[ind,'group'] = group_num   
        #overwrite the original event catalogue file with the updated event in it.
        all_day.to_csv(os.path.join(c_path,'events__'+t1.strftime("%Y%m%dT%H%M%SZ")+'__'+t2.strftime("%Y%m%dT%H%M%SZ")+'.csv'))

    def get_data_window(self):
        """
        Returns only the stream for the window of interest for plotting (i.e. cuts out the extra buffer data used for filtering)
        """
        stream = self.stream.slice(self.data_window[0],self.data_window[1])
        return stream
    
    def flat_window(self):
        window = self.get_data_window()
        flattened = np.stack([tr.data.astype(np.float64) for tr in window],axis=0)
        return flattened
    
    def flat_response(self,freq):
        resp = np.stack([self.inv.get_response(tr.id,self.starttime).get_evalresp_response_for_frequencies(freq) for tr in self.stream],axis=1)
        return resp
    
    def template_trace(self,component='Z'):
        max_template = 0
        for tr in self.get_data_window().select(component=component):
            energy = np.sum(tr.data**2)
            if energy > max_template:
                max_template = energy
                template_tr = tr
        return template_tr
    
    def filter(self,type,**options):
        """
        Obspy filter functionality applied to the SeismicEvent object. Use exactly as obspy filter, but automatically demeans before
        and after the filtering. See Obspy documentation for all options for filtering.
        """
        self.stream = self.stream.split()
        self.stream.detrend("demean")
        self.stream.taper(None,max_length=self.extra/2,type='blackman')
        self.stream.filter(type,**options)
        self.stream.detrend("demean")
        self.stream = self.stream.merge(fill_value=None)

        self.__fill_gaps()

    def remove_response(self,**options):
        """
        Obspy remove response wrapper - see obspy documentation
        """
        #TODO change the label according to what the output is (e.g. Velocity [m/s] for when output=='VEL').
        self.stream.remove_response(inventory=self.inv,**options)

    def decimate(self,factor):
        self.stream = self.stream.split()
        self.stream.decimate(factor)
        self.stream = self.stream.merge(fill_value=None)
        self.__fill_gaps()

    def remove_sensitivity(self):
        """
        Obspy remove sensitivity wrapper - see obspy documentation
        """
        #TODO change the label from count to the natural measurement of the instrument (VEL for the broadbands).
        self.stream.remove_sensitivity(self.inv)

    def __fill_gaps(self):
        self.stream = self.stream.merge(fill_value=None) #incase this has not been done yet.
        gapless_stream = Stream()
        for network in self.inv:
            for station in network:
                for channel in station:
                    subset = self.stream.select(network=network.code,station=station.code,channel=channel.code)
                    if len(subset) > 0:
                        tr = subset[0].trim(self.data_window[0]-self.extra,self.data_window[1]+self.extra) #want to trim to the window length with padding to make sure all have same number of samples
                    elif len(subset) == 0:
                        tr = Trace(data=create_empty_data_chunk(1,'f'),header={'network':network.code,'station':station.code,'location_code':channel.location_code,'channel':channel.code,'starttime':self.data_window[0]-self.extra,'sampling_rate':channel.sample_rate}).trim(self.data_window[0]-self.extra,self.data_window[1]+self.extra,pad=True,fill_value=None)
                    else:
                        raise(Exception('Non-uniqueness in trace IDs'))
                    
                    gapless_stream += tr
        self.stream = gapless_stream

        


class SeismicChunk:
    def __init__(self,starttime,endtime,time_offset=0):
        self.starttime = starttime
        self.endtime = endtime
        self.ppsd_bool = False
        self.time_offset = time_offset
        self.step = 24*60*60 #default if __call__ is not used for iteration

        self.step_start = starttime

        self.str_name = starttime.strftime("%Y%m%dT%H%M%SZ")+'__'+endtime.strftime("%Y%m%dT%H%M%SZ")

        self.stream = None
        self.inv = None


    def __iter__(self):
        return self
    
    def __next__(self):
        if self.step_start < self.endtime:
            step_end = self.step_start + self.step
            if step_end > self.endtime:
                step_end = self.endtime
            step_chunk = SeismicChunk(self.step_start,step_end,time_offset=self.time_offset)

            #if the chunk has an attached stream, give this to the subchunk
            if self.stream is not None:
                step_chunk.stream = self.stream.copy().trim(step_chunk.starttime,step_chunk.endtime,pad=True,fill_value=None)
                step_chunk.inv = self.inv
            self.step_start += self.step
            return step_chunk
        else:
            #finised iterating through the chunk, reset start time and exit loop.
            self.step_start = self.starttime
            raise StopIteration
        
    def __call__(self,step):
        self.step = step
        return self

    def context(self,new_context=None):
        """
        Change from the current context to a new one. This method is accessible regardless of the
        present context, with unknown or unspecified context arguments send back to this base class.
        """
        if new_context == 'plot':
            from iqvis.visualisation import ChunkPlot
            self.__class__ = ChunkPlot
        
        elif new_context == 'eventplot':
            from iqvis.visualisation import ChunkEventPlot
            self.__class__ = ChunkEventPlot
        
        elif new_context == 'detect':
            from iqvis.event_detection import ChunkDetection
            self.__class__ = ChunkDetection
        
        elif new_context == 'spectral':
            from iqvis.spec_analysis import ChunkSpectrum
            self.__class__ = ChunkSpectrum

        elif new_context == 'timeseries':
            from iqvis.time_series import TimeSeries
            self.__class__ = TimeSeries
        
        else:
            self.__class__ = SeismicChunk
        
    def attach_waveforms(self,inv,w_path,buffer=0):
        """
        Access and attach the seismic waveforms for the chunk time period for the stations and channels provided in the inventory.
        """
        self.stream = sh.inv_to_waveforms(inv,self.starttime,self.endtime,w_path,buffer=buffer,print_s=False,fill=None)
        self.inv = inv

    def download_waveforms(self,w_path,**options):
        """
        Download the waveforms for specified stations and channels over the time period of the chunk. To maintain consistant file
        naming conventions, full days from 00:00:00 UTC time will be downloaded, but then trimmed to the chunk length when automatically
        attached.
        """
        #TODO just make this a wrapper around the seismic_attributes version of get_waveforms, but then trim the data to the chunk
        #TODO length and attach the downloaded inventory. This can then be used as an alternative to attach waveforms but when the
        #TODO data and/or inventory have not been downloaded locally. 
        pass
    
    def filter(self,type,**options):
        """
        Obspy filter functionality applied to the SeismicEvent object. Use exactly as obspy filter, but automatically demeans before
        and after the filtering. It also retains empty traces that would otherwise be lost during obspy filtering.
        """

        filtered = Stream()

        for tr in self.stream:
            start = tr.stats.starttime
            end = tr.stats.endtime
            split_stream = tr.split()
            if len(split_stream) > 0:
                split_stream.detrend("demean")
                split_stream.filter(type,**options)
                split_stream.detrend("demean")
                merged = split_stream.merge(fill_value=None).trim(start,end,pad=True,fill_value=None)
                filtered += merged
            else:
                filtered += tr #if no data, just put the empty trace back in
        
        self.stream = filtered

    def remove_response(self,**options):
        filtered = Stream()

        for tr in self.stream:
            start = tr.stats.starttime
            end = tr.stats.endtime
            split_stream = tr.split()
            if len(split_stream) > 0:
                split_stream.remove_response(self.inv,**options)
                merged = split_stream.merge(fill_value=None).trim(start,end,pad=True,fill_value=None)
                filtered += merged
            else:
                filtered += tr #if no data, just put the empty trace back in
        
        self.stream = filtered

    def remove_sensitivity(self):
        self.stream.remove_sensitivity(self.inv)

    def decimate(self,factor):
        #need this instead of in-built obspy as the obspy version plays funny buggers with empty traces and ignores masked arrays.
        decimated = Stream()

        for tr in self.stream:
            start = tr.stats.starttime
            end = tr.stats.endtime

            if factor > 16:
                msg = "Automatic filter design is unstable for decimation " + \
                      "factors above 16. Manual decimation is necessary."
                raise ArithmeticError(msg)
            
            freq = tr.stats.sampling_rate * 0.5 / float(factor) #nyquist freq and new sampling rate

            split_stream = tr.split()

            if len(split_stream) > 0:
                split_stream.filter('lowpass_cheby_2', freq=freq, maxorder=12)
                tr = split_stream.merge(fill_value=None).trim(start,end,pad=True,fill_value=None)[0]

            #now do the decimation - this works for empty traces too.
            tr.data = tr.data[::factor]
            tr.stats.sampling_rate = tr.stats.sampling_rate / float(factor)
            decimated += tr
        
        self.stream = decimated

    def load_csv(self,path,name):

        files = []
        for daychunk in self:
            filename = os.path.join(path,name + '__' + daychunk.starttime.strftime("%Y%m%dT%H%M%SZ")+'__'+daychunk.endtime.strftime("%Y%m%dT%H%M%SZ") + '.csv')
            if os.path.exists(filename):
                files.append(filename)

        df = pd.concat([pd.read_csv(file,index_col=0) for file in files],ignore_index=False)
        return df


    def get_events(self,c_path):
        """
        Load in the events occuring within the chunk time window.
        Inputs:
            c_path: path to the catalogue file, assumming that UTC naming convention is used.
        Returns:
            event_cat: pandas DataFrame of events occuring in chunk
        """
        __f_length = 86400
        start_time = sh.UTCDateTime(self.starttime.year,self.starttime.month,self.starttime.day)
        end_time = start_time + __f_length

        traces = []

        while (start_time < self.endtime):
                
            try:
                traces.append(pd.read_csv(os.path.join(c_path,'events__'+start_time.strftime("%Y%m%dT%H%M%SZ")+'__'+end_time.strftime("%Y%m%dT%H%M%SZ")+'.csv'),index_col=0,converters={"stations": self.__lst_station}))
            except:
                print('No events or traces for day, skipping to next.')

            start_time += __f_length
            end_time += __f_length
        try:
            full_cat = pd.concat(traces,ignore_index=True)
        except:
            raise(Exception('No events for the given time window'))

        event_cat = full_cat[((full_cat['ref_time'] >= self.starttime) & (full_cat['ref_time'] <= self.endtime))] #clip out the events that do not occur in chunk.
        return event_cat
    
    def get_traces(self,c_path):
        """
        As above for get_events, but loads the trace catalogue for the chunk.
        Inputs:
            c_path: path to the catalogue file, assumming that UTC naming convention is used.
        Returns:
            trace_cat: pandas DataFrame of event traces occuring in chunk
        """
        __f_length = 86400
        start_time = sh.UTCDateTime(self.starttime.year,self.starttime.month,self.starttime.day)
        end_time = start_time + __f_length

        traces = []

        while (start_time < self.endtime):
                
            try:
                traces.append(pd.read_csv(os.path.join(c_path,'traces__'+start_time.strftime("%Y%m%dT%H%M%SZ")+'__'+end_time.strftime("%Y%m%dT%H%M%SZ")+'.csv'),index_col=0,converters={"stations": self.__lst_station}))
            except:
                print('No events or traces for day, skipping to next.')

            start_time += __f_length
            end_time += __f_length
        try:
            full_cat = pd.concat(traces,ignore_index=True)
        except:
            raise(Exception('No events for the given time window'))

        trace_cat = full_cat[((full_cat['time'] >= self.starttime) & (full_cat['time'] <= self.endtime))]
        return trace_cat
    
    def attach_events(self,c_path):
        """
        Creates and attaches a dictionary of events for the chunk by loading in all events and converting
        into SeismicEvent objects. The dictionary keys are the event ids and the items are the corresponding
        SeismicEvent objects. 
        Inputs:
            c_path: path to the catalogue files (both event and trace catalogues needed)
        """
        event_cat = self.get_events(c_path)
        trace_cat = self.get_traces(c_path)

        self.events = {}

        for i, row in event_cat.iterrows():
            trace_rows = trace_cat[trace_cat['event_id']==row['event_id']]
            event_obj = SeismicEvent(row,trace_rows)    
            self.events[event_obj.event_id] = event_obj

        self.event_keys = list(self.events.keys())

    def attach_event_times(self,c_path):
        event_cat = self.get_events(c_path)
        self.event_times = np.array([UTCDateTime(time) for time in event_cat['ref_time'].to_numpy()],dtype=UTCDateTime)


    def pull_events(self,chunk):
        self.events = self.events | chunk.events
        self.event_keys = self.event_keys + chunk.event_keys

    def split_events(self):
        self.grouped_events = {}
        for event_id, event in self.events.items():
            group = event.group
            try:
                self.grouped_events[group][event_id] = event
            except:
                #if a dictionary has not been initialised for this group yet, make it here
                self.grouped_events[group] = {}
                self.grouped_events[group][event_id] = event


    def event_stream(self,event_cat):
        group_ind = event_cat.events.group.unique()
        event_streams = {key:Stream() for key in group_ind} #! this will be empty dictionary if there are no events (no unique groups)
        for event in event_cat:
            event_streams[event.group] += self.stream.slice(event.starttime,event.endtime)
        
        for key, event_stream in event_streams.items():
            event_stream = event_stream.merge(fill_value=None)
            if len(event_stream) > 0: #TODO simplify to what was before as this loop should not be entered if there are no events, as there are no event groups.
                event_streams[key] = event_stream.trim(self.starttime,self.endtime,fill_value=None,pad=True)
            else:
                masked = self.stream.copy()
                for tr in masked:
                    tr.data = np.ma.array(tr.data,mask=True)
                event_streams[key] = masked
        return event_streams

    def teleseism_arrivals(self,phases=('ttall', ),**kwargs):
        #first get the events in the time period over a given magnitude
        client = Client('IRIS')
        cat = client.get_events(starttime=self.starttime,endtime=self.endtime,**kwargs)
        #now loop over the events and compute the arrival time of the chosen phases for each station in the attached inventory

        all_arrivals = []

        for event in cat:
            arrival_times = {}

            depth = event.origins[0].depth/1000
            lat0 = event.origins[0].latitude
            lon0 = event.origins[0].longitude

            event_time = event.origins[0].time

            model = TauPyModel(model='prem')

            for network in self.inv:
                for station in network:
                    rx_lat = station.latitude
                    rx_lon = station.longitude

                    arrivals = model.get_travel_times_geo(source_depth_in_km=depth,
                                                        source_latitude_in_deg=lat0, 
                                                        source_longitude_in_deg=lon0,
                                                        receiver_latitude_in_deg=rx_lat,
                                                        receiver_longitude_in_deg=rx_lon,
                                                        phase_list=phases)
                    
                    travel_times = dict()
                    for i, arrival in enumerate(arrivals):
                        travel_times[arrival.phase.name] = arrival.time
                    
                    arrival_times[station.code] = {key:event_time + tt for key, tt in travel_times.items()}
            all_arrivals.append(arrival_times)
        return all_arrivals

    



    def add_tidal_phase(self,event_cat):
        """
        Takes an existing event catalogue (with or without assigned groups) and interpolates a tidal phase to each
        of the event times, adding this as a new column to the DataFrame.
        """
        #TODO move this to another class and make use of the attached events - loop through dictionary.
        #firstly get the tidal angles and times
        times, angles = self.__model_tide_phases(self.starttime-2*60*60,self.endtime+2*60*60)
        
        #now take the ref time for each event in the catalogue and interpolate the tidal phase...
        time_arr = event_cat['ref_time'].to_numpy()
        time_matplotlib = np.array([UTCDateTime(time).matplotlib_date for time in time_arr])
        event_cat['tide_phase'] = np.interp(time_matplotlib, times, angles)
        return event_cat



    # def get_tides(self,delta,tide_path,lat,lon,buffer=120):
    #     t1 = self.starttime - buffer
    #     t2 = self.endtime + buffer
    #     duration = t2 - t1
    #     starttime = t1.year,t1.month,t1.day,t1.hour,t1.minute,t1.second

    #     times = np.mgrid[0:duration:delta] #get the times to compute the tides at
    #     tides = pyTMD.compute_tide_corrections(np.array([lon]),np.array([lat]),times,EPOCH=starttime,DIRECTORY=tide_path,MODEL='CATS2008',TIME='UTC',TYPE='time series',METHOD='spline',EPSG='4326')

    #     tide_tr = Trace(data=tides.flatten())
    #     tide_tr.stats.delta = delta
    #     tide_tr.stats.starttime = UTCDateTime(*starttime)
    #     tide_tr.stats.channel = 'TIDES'

    #     stream = Stream(traces=[tide_tr])
    #     return stream

    
    def __lst_station(self,string):
        string = string.strip("[]")
        lst = string.split(', ')
        lst = [st.strip("''") for st in lst]
        return lst
    
    def __fill_gaps(self):
        self.stream = self.stream.merge(fill_value=None) #incase this has not been done yet.
        gapless_stream = Stream()
        for network in self.inv:
            for station in network:
                for channel in station:
                    subset = self.stream.select(network=network.code,station=station.code,channel=channel.code)
                    if len(subset) > 0:
                        tr = subset[0].trim(self.data_window[0]-self.extra,self.data_window[1]+self.extra) #want to trim to the window length with padding to make sure all have same number of samples
                    elif len(subset) == 0:
                        tr = Trace(data=create_empty_data_chunk(1,'f'),header={'network':network.code,'station':station.code,'location_code':channel.location_code,'channel':channel.code,'starttime':self.data_window[0]-self.extra,'sampling_rate':channel.sample_rate}).trim(self.data_window[0]-self.extra,self.data_window[1]+self.extra,pad=True,fill_value=None)
                    else:
                        raise(Exception('Non-uniqueness in trace IDs'))
                    
                    gapless_stream += tr
        self.stream = gapless_stream
    


class EventCatalogue:
    """
    A wrapper around the event and trace catalogues that allows iteration over SeismicEvent objects.
    """
    #TODO may want to make event_id the index for all dataframes - would involve changing the code in some scripts as well, have had issues using .at on dataframe when using i as index, works when using subset of catalogue.
    #TODO need a check in here for duplicates of event_id, drop second occurance if this does happen as fail-safe. 
    def __init__(self,t1,t2,c_path,templates=False):
        self.starttime = t1
        self.endtime = t2
        #load the event catalogue for these times...
        start_time = sh.UTCDateTime(t1.year,t1.month,t1.day)
        end_time = start_time + 24*60*60

        events = []
        traces = []
        attributes = []

        while (start_time < t2):
            if templates:
                e_name = os.path.join(c_path,'template_events__'+start_time.strftime("%Y%m%dT%H%M%SZ")+'__'+end_time.strftime("%Y%m%dT%H%M%SZ")+'.csv')
                t_name = os.path.join(c_path,'template_traces__'+start_time.strftime("%Y%m%dT%H%M%SZ")+'__'+end_time.strftime("%Y%m%dT%H%M%SZ")+'.csv')
                a_name = os.path.join(c_path,'template_attributes__'+start_time.strftime("%Y%m%dT%H%M%SZ")+'__'+end_time.strftime("%Y%m%dT%H%M%SZ")+'.csv')
            else:
                e_name = os.path.join(c_path,'events__'+start_time.strftime("%Y%m%dT%H%M%SZ")+'__'+end_time.strftime("%Y%m%dT%H%M%SZ")+'.csv')
                t_name = os.path.join(c_path,'traces__'+start_time.strftime("%Y%m%dT%H%M%SZ")+'__'+end_time.strftime("%Y%m%dT%H%M%SZ")+'.csv')
                a_name = os.path.join(c_path,'attributes__'+start_time.strftime("%Y%m%dT%H%M%SZ")+'__'+end_time.strftime("%Y%m%dT%H%M%SZ")+'.csv')

            if os.path.isfile(e_name):
                events.append(pd.read_csv(e_name,index_col=0,converters={"stations": self.__lst_station}))
            if os.path.isfile(t_name):
                traces.append(pd.read_csv(t_name,index_col=0,converters={"stations": self.__lst_station}))
            if os.path.isfile(a_name):
                attributes.append(pd.read_csv(a_name,index_col=0))
          
            start_time += 24*60*60
            end_time += 24*60*60
        
        if len(events) > 0:
            event_cat = pd.concat(events,ignore_index=True)
            event_cat = event_cat[((event_cat['ref_time'] >= t1) & (event_cat['ref_time'] <= t2))]
            event_cat.set_index('event_id',drop=True,inplace=True)
            event_cat = event_cat[~event_cat.index.duplicated(keep='first')] #if multiple events have the same event id, only keep this first occurance.
            self.N = len(event_cat)
        else:
            event_cat = None
            self.N = 0
            warnings.warn('No events found for time period')
        
        if len(traces) > 0:
            trace_cat = pd.concat(traces,ignore_index=True)
            if event_cat is not None:
                trace_cat.set_index('event_id',drop=True,inplace=True)
                trace_cat = trace_cat[trace_cat.index.isin(event_cat.index)] #this does not deal with duplicates, but only needed for SeismicEvent so deal with that there.
            else:
                trace_cat = None
        else:
            trace_cat = None
            warnings.warn('No traces found for time period')

        if len(attributes) > 0:
            attribute_cat = pd.concat(attributes,ignore_index=True)
            if event_cat is not None:
                attribute_cat.set_index('event_id',drop=True,inplace=True)
                attribute_cat = attribute_cat[attribute_cat.index.isin(event_cat.index)]
                attribute_cat = attribute_cat[~attribute_cat.index.duplicated(keep='first')] #if multiple events have the same event id, only keep this first occurance.
            else:
                attribute_cat = None
        else:
            attribute_cat = None
            warnings.warn('No attributes found for time period, attaching empty attribute catalogue')

        self.events = event_cat
        self.traces = trace_cat
        self.attributes = attribute_cat
        self.attribute_dict = {}

        self.i = 0 #set the current index to zero

        self.event_times = np.array([UTCDateTime(time) for time in event_cat['ref_time'].to_numpy()],dtype=UTCDateTime) #array of the event reference times.


    def __len__(self):
        return self.N
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.i < self.N:
            event_row = self.events.iloc[self.i]
            trace_rows = self.traces.loc[event_row.name]

            if type(trace_rows) == pd.Series:
                trace_rows = trace_rows.to_frame().T

            event_obj = SeismicEvent(event_row,trace_rows)   
            self.i += 1
            return event_obj
        else:
            self.i = 0 #reset the index for next time and cut out of the for loop.
            raise StopIteration
        
    def __add__(self,event_cat):
        merged_cat = copy.deepcopy(self)

        merged_cat.events = pd.concat([self.events,event_cat.events])
        merged_cat.traces = pd.concat([self.traces,event_cat.traces])
        merged_cat.attributes = pd.concat([self.attributes,event_cat.attributes])

        merged_cat.N = len(merged_cat.events)
        return merged_cat
    
    def add_classification(self,class_cat):
        for i, row in class_cat.iterrows():
            event_id = row['event_id']
            group_id = row['group']

            if event_id in self.events.index:
                self.events.at[event_id,'group'] = group_id
                self.attributes.at[event_id,'group'] = group_id

    def group_split(self):
        #firstly get the unique group tags from the event dataframe
        groups = self.events.group.unique()
        catalogues = {}
        for group in groups:
            copy_cat = copy.deepcopy(self)
            sub_event = self.events[self.events['group']==group]
            sub_trace = self.traces[self.traces.index.isin(sub_event.index)]
            sub_att = self.attributes[self.attributes.index.isin(sub_event.index)]
            
            copy_cat.events = sub_event
            copy_cat.traces = sub_trace
            copy_cat.attributes = sub_att

            copy_cat.event_times = np.array([UTCDateTime(time) for time in copy_cat.events['ref_time'].to_numpy()],dtype=UTCDateTime)


            copy_cat.N = len(copy_cat.events)
            copy_cat.i = 0

            catalogues[group] = copy_cat

        return catalogues
    

    def select_event(self,event_id):
        event_row = self.events.loc[event_id]
        trace_rows = self.traces.loc[event_id]
        event_obj = SeismicEvent(event_row,trace_rows)   
        return event_obj

    
        
    def __lst_station(self,string):
        string = string.strip("[]")
        lst = string.split(', ')
        lst = [st.strip("''") for st in lst]
        return lst
