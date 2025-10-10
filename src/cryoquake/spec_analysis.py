"""
CODE FOR MAKING CONSISTANT SPECTROGRAM OBJECTS FOR DETECTED SEISMIC EVENTS
"""
import numpy as np
from obspy import UTCDateTime
from cryoquake.data_objects import SeismicEvent, SeismicChunk
import os


class EventSpectrum(SeismicEvent):
    def get_spectrograms(self,nperseg,noverlap,freq_range=None):
        """
        """
        if self.stream == None:
            raise(Exception("Waveforms must be attached to SeismicEvent before spectrograms are computed."))
        
        all_specs = {}
        for network in self.inv:
            for station in network:
                spec_obj = Spectrogram(self,station)
                spec_obj.make_spec(nperseg,noverlap,freq_range=freq_range)
                all_specs[station.code] = spec_obj
        self.specs = all_specs

    def get_power_spectrum(self,freq_range=None):
        if self.stream == None:
            raise(Exception("Waveforms must be attached to SeismicEvent before spectrograms are computed."))
        
        all_specs = {}
        for network in self.inv:
            for station in network:
                psd_obj = PowerSpectrum(self,station)
                psd_obj.make_spec(freq_range=freq_range)
                all_specs[station.code] = psd_obj
        self.psds = all_specs



class ChunkSpectrum(SeismicChunk):
    def make_periodograms(self,window_length,window_overlap,path,window='blackman'):

        window_shift = window_length * (1 - window_overlap)

        if not os.path.exists(path):
            os.mkdir(path) #make the catalogue directory if it has not yet been made
        
        from scipy.signal import periodogram
        from obspy.signal.util import nearest_pow_2
        
        for tr in self.stream:
            fs = tr.stats.sampling_rate
            N = int(fs * window_length)
            nfft = nearest_pow_2(int(N))

            day_start = UTCDateTime(self.starttime.year,self.starttime.month,self.starttime.day) #start of first day of stream
            day_end = day_start + 24*60*60 #end of day
            while day_end <= self.endtime: #! this does not allow partial days - want it to be day start before end time. Also try and make it a for loop so that a progress bar can be added.

                day_trace = tr.slice(day_start,day_end) #get trace just for the current day
                t1 = day_start
                t2 = t1 + window_length
                
                
                times = []
                psds = []

                while t2 <= day_end: #need to make usre there are no short segments with different frequencies.
                    section = day_trace.slice(t1,t2)
                    if isinstance(section.data,np.ndarray):
                        full = True
                    elif isinstance(section.data,np.ma.MaskedArray):
                        if not section.data.mask.max():
                            full = True
                        else:
                            full = False
                    else:
                        full = False

                    if full: #only compute the PSD if it is a full section, otherwise skip...
                        f, Pxx = periodogram(section.data,window=window,fs=fs,nfft=nfft)
                        times.append((t1 + window_length/2)) #get centre of the window as time for PSD
                        psds.append(Pxx)
                    t1 += window_shift
                    t2 += window_shift
                
                try:
                    spec = np.stack(psds,axis=-1)
                    times = np.array(times,dtype=UTCDateTime)

                    #now save these results to file
                    filename = os.path.join(path,'psds__'+ tr.id + '__' + day_start.strftime("%Y%m%dT%H%M%SZ")+'__'+day_end.strftime("%Y%m%dT%H%M%SZ")+'.npz')

                    np.savez(filename,spec=spec,t=times,f=f)
                except:
                    print('No PSDs for ' + day_start.strftime("%Y%m%dT%H%M%SZ")+'__'+day_end.strftime("%Y%m%dT%H%M%SZ"))

                day_start += 24*60*60
                day_end += 24*60*60
        

    def load_periodograms(self,inv,path):
        t1 = self.starttime
        t2 = self.endtime
        

        spec_dict = {}
        for network in inv:
            for station in network:
                for channel in station:

                    tr_id = network.code + '.' + station.code + '..' + channel.code
        
                    day_start = UTCDateTime(t1.year,t1.month,t1.day) #start of first day of stream
                    day_end = day_start + 24*60*60 #end of day
                    specs = []
                    times = []
                    while day_start < t2:
                        filename = os.path.join(path,'psds__'+ tr_id + '__' + day_start.strftime("%Y%m%dT%H%M%SZ")+'__'+day_end.strftime("%Y%m%dT%H%M%SZ")+'.npz')
                        try:
                            npz = np.load(filename,allow_pickle=True)
                            spec = npz['spec']
                            f = npz['f']
                            t = npz['t']

                            specs.append(spec)
                            times.append(t)
                        except:
                            print('No file for ' + day_start.strftime("%Y%m%dT%H%M%SZ")+'__'+day_end.strftime("%Y%m%dT%H%M%SZ"))

                        day_start += 24*60*60
                        day_end += 24*60*60
                    specs = np.concatenate(specs,axis=1)
                    times = np.concatenate(times)
                    #TODO trim to start and end of specified time in case of partial days.
                    spec_dict[tr_id] = (times,f,specs)
        return spec_dict


    def get_spectrograms(self,window_length,window_overlap,sample_length,sample_overlap,freq_range=None,window='hann',average='median'):
        """
        Write in similar format to equivalent function for SeismicEvent, but this time use a median probabalistic method.
        """
        if self.stream == None:
            raise(Exception("Waveforms must be attached to SeismicEvent before spectrograms are computed."))
        
        all_specs = {}
        for network in self.inv:
            for station in network:
                spec_obj = ChunkSpectrogram(self,station)
                spec_obj.make_spec(window_length,window_overlap,sample_length,sample_overlap,freq_range=freq_range,window=window,average=average)
                all_specs[station.code] = spec_obj
        self.specs = all_specs

    def get_bandpower(self,band):
        for network in self.inv:
            for station in network:
                spec_obj = self.specs[station.code]
                spec_obj.integrate_spec(band)

    def compute_ppsd(self,ppsd_length,period_limits=(1/30,10)):
        from obspy.signal.spectral_estimation import PPSD

        self.ppsd = PPSD(self.stream[0].stats,self.inv,ppsd_length=ppsd_length,skip_on_gaps=True,period_limits=period_limits)
        self.ppsd_bool = self.ppsd.add(self.stream) #flag as to whether it was successfully computed.
        self.ppsd_length = ppsd_length
    


class Spectrogram:
    """
    Might need this class to attach three spectrogram components, along with the station name, 
    event ID and other metadata.
    """
    def __init__(self,seismic_event,station):
        self.starttime = seismic_event.window_start
        self.endtime = seismic_event.window_end
        self.event_id = seismic_event.event_id
        self.stream = seismic_event.get_data_window()
        self.station = station.code
    
    def make_spec(self,nperseg,noverlap,freq_range=None):
        max_power = 0.0
        for tr in self.stream.select(station=self.station):
            channel = tr.id
            spec, f, t = self.__spec_backend(tr,nperseg,noverlap)

            if np.max(spec) > max_power: 
                max_power = np.max(spec)

            if channel[-1] == 'Z':
                self.Z = spec
            if channel[-1] == 'N':
                self.N = spec
            if channel[-1] == 'E':
                self.E = spec
            
            utc_time = np.full_like(t,fill_value=tr.stats.starttime,dtype=UTCDateTime) + t

        self.f = f
        self.t = utc_time
        self.max_power = max_power
        
        
    def __spec_backend(self,tr,nperseg,noverlap):
        from scipy.signal import stft
        f, t, spec = stft(tr.data,fs=tr.stats.sampling_rate,nperseg=nperseg,noverlap=noverlap)
        spec = np.abs(spec)
        return spec, f, t
    
class PowerSpectrum: #TODO add the frequency range here as this has been removed from the plotting functions.
    def __init__(self,seismic_event,station):
        self.starttime = seismic_event.window_start
        self.endtime = seismic_event.window_end
        self.event_id = seismic_event.event_id
        self.stream = seismic_event.get_data_window()
        self.station = station.code
    
    def make_spec(self,freq_range=None):
        max_power = 0.0
        for tr in self.stream.select(station=self.station):
            channel = tr.id
            psd, f = self.__spec_backend(tr)
            if isinstance(freq_range,list):
                #just pick out the frequency range of interest.
                psd = psd[(f >= freq_range[0]) & (f <= freq_range[1])]
                f = f[(f >= freq_range[0]) & (f <= freq_range[1])]

            if np.max(psd) > max_power: 
                max_power = np.max(psd)

            if channel[-1] == 'Z':
                self.Z = psd
            if channel[-1] == 'N':
                self.N = psd
            if channel[-1] == 'E':
                self.E = psd

        self.f = f
        self.max_power = max_power
        
    def __spec_backend(self,tr):
        from scipy.signal import periodogram, get_window
        from obspy.signal import util
        N = tr.data.size
        window = get_window('blackman',N)
        nfft = util.nearest_pow_2(N)
        windowed = tr.data * window
        windowed -= np.mean(windowed) #dtrend after windowing to eliminate zero frequency...
        f, psd = periodogram(windowed,fs=tr.stats.sampling_rate,nfft=nfft)
        return psd, f


class ChunkSpectrogram:
    def __init__(self,seismic_chunk,station):
        self.starttime = seismic_chunk.starttime
        self.endtime = seismic_chunk.endtime
        self.stream = seismic_chunk.stream.select(station=station.code)
        self.station = station.code

    
    def make_spec(self,window_length,window_overlap,sample_length,sample_overlap,freq_range=None,window='hann',average='median'):
        self.window_length = window_length
        self.window_overlap = window_overlap
        self.sample_length = sample_length
        self.sample_overlap = sample_overlap
        self.window = window
        self.average = average

        for tr in self.stream:
            channel = tr.id
            spec, f, t = self.__spec_backend(tr)
            
            if isinstance(freq_range,list):
                #just pick out the frequency range of interest.
                spec = spec[(f >= freq_range[0]) & (f <= freq_range[1]),:] #need to work out if all columns or rows need to be taken (index of time frequency dependent)
                f = f[(f >= freq_range[0]) & (f <= freq_range[1])]


            if channel[-1] == 'Z':
                self.Z = spec
            if channel[-1] == 'N':
                self.N = spec
            if channel[-1] == 'E':
                self.E = spec

        self.f = f
        self.t = t

    def integrate_spec(self,band):
        """
        Integrate the spectrograms over a particular frequency band to get power within the frequency range.
        """
        from scipy.integrate import trapezoid
        for tr in self.stream:
            network, station, location, channel = tr.id.split('.')
            if channel[-1] == 'Z':
                spec = self.Z
            if channel[-1] == 'N':
                spec = self.N
            if channel[-1] == 'E':
                spec = self.E

            f = self.f
            spec_int = spec[(f >= band[0]) & (f <= band[1]),:]
            f_int = f[(f >= band[0]) & (f <= band[1])]
            power = trapezoid(spec_int,f_int,axis=0)

            if channel[-1] == 'Z':
                self.pZ = power
            if channel[-1] == 'N':
                self.pN = power
            if channel[-1] == 'E':
                self.pE = power

                
    def __spec_backend(self,tr):
        from scipy.signal import welch

        window_length = self.window_length
        window_step = self.window_overlap * window_length
        fs = tr.stats.sampling_rate
        nperseg = int(np.round(fs*self.sample_length))
        noverlap = int(np.round(self.sample_overlap * nperseg))

        #need to do sliding window spectrogram
        t1 = self.starttime
        t2 = t1 + window_length

        t = []
        spec = []

        while t2 < self.endtime: #don't do last segment if it is not full length as this probably gives different frequencies...
            tr_window = tr.copy().trim(t1,t2)
            t.append(t1 + window_length / 2) #define time as centre of current window

            f, psd = welch(tr_window.data,fs=fs,window=self.window,nperseg=nperseg,noverlap=noverlap,average=self.average)
            spec.append(psd)

            t1 += window_step
            t2 += window_step

        spec = np.stack(spec,axis=-1)
        t = np.array(t,dtype=UTCDateTime)

        #now do a dimension check
        if not (spec.shape == (f.size,t.size)):
            raise(Exception('Spectrogram shape does not match the time and frequency arrays.'))
        
        return spec, f, t