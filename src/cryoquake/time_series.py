"""
FUNCTIONS FOR TIME SERIES ANAYLSIS OF SEISMIC CHUNKS
INCLUDES COMPUTING AMPLITUDE BASED METRICS ACROSS A TIME PERIOD, AND FUNCTIONS WORKING WITH ENVIRONMENTAL DRIVERS
"""

from cryoquake.data_objects import SeismicChunk

import numpy as np
from obspy.core import Stream, Trace


class TimeSeries(SeismicChunk):
    def attach_timeseries(self,window_length,window_overlap):
      
        traces = []
        for network in self.inv:
            for station in network:
                median = []
                rms = []
                t1 = self.starttime
                t2 = self.starttime + window_length
                station_st = self.stream.select(network = network.code, station = station.code)
                while t1 < self.endtime:
                    section = station_st.slice(t1,t2)
                   
                    amp2 = sum([tr.data**2 for tr in section]) #get the squared amplitude time series if multiple components
                    if not isinstance(amp2,np.ndarray):
                        rms.append(np.nan)
                        median.append(np.nan)
                    else:
                        #compute and attach the amplitude metrics
                        rms.append(np.sqrt(np.sum(amp2) / amp2.size))
                        median.append(np.sqrt(np.median(amp2)))


                    #shift the window forward
                    t1 += window_length * (1 - window_overlap)
                    t2 += window_length * (1 - window_overlap)

                #now make a trace for each of the metrics at this station.
                rms_tr = Trace(data=np.array(rms))
                rms_tr.stats.starttime = self.starttime + (window_length/2)
                rms_tr.stats.delta = window_length * (1-window_overlap)
                rms_tr.stats.network = network.code
                rms_tr.stats.station = station.code
                rms_tr.stats.channel = 'RMS'

                med_tr = Trace(data=np.array(median))
                med_tr.stats.starttime = self.starttime + (window_length/2)
                med_tr.stats.delta = window_length * (1-window_overlap)
                med_tr.stats.network = network.code
                med_tr.stats.station = station.code
                med_tr.stats.channel = 'MED'

       
                traces += [rms_tr.copy(),med_tr.copy()]

        stream = Stream(traces=traces)
        self.timeseries = stream


    def attach_environmental(self,env_path,filename='environmental_stream.mseed'):
        import os
        import obspy
        file_path = os.path.join(env_path,filename)
        env_stream = obspy.read(file_path)
        env_stream.trim(self.starttime,self.endtime)
        self.env_stream = env_stream

    def event_interpolation(self):
        for event_id, event in self.events.items():
            ref_time = event.starttime
            local_time = ref_time + self.time_offset * 60 * 60
            time_var = {}
            time_var['local_time'] = local_time
            for tr in self.env_stream:
                starttime = tr.stats.starttime #t=0 for the trace
                dt = ref_time - starttime #seconds after start of trace
                interp = np.interp(dt,tr.times(),tr.data)
                time_var[tr.stats.channel] = interp
            
            for tr in self.timeseries:
                starttime = tr.stats.starttime #t=0 for the trace
                dt = ref_time - starttime #seconds after start of trace
                interp = np.interp(dt,tr.times(),tr.data)
                time_var[tr.stats.channel] = interp


            event.time_var = time_var #attach dictionary of interpolated values.



    #todo put some band integration code for spectrograms here, where they become



    #todo move median spectrogram code here as it is form of time-series analysis and useful for comparing with environmental data.