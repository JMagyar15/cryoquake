from cryoquake.data_objects import SeismicEvent
import pandas as pd
import numpy as np
from obspy.signal.filter import envelope
from obspy.signal import cross_correlation
from scipy.signal import hilbert


def average_cc(event1,event2):
    cc_traces = []
    for tr1 in event1.get_data_window().select(component='Z'):
        net, sta, loc, cha = tr1.id.split('.')
        tr2 = event2.get_data_window().select(network=net,station=sta,location=loc,channel=cha)[0]

        len1 = tr1.stats.npts
        len2 = tr2.stats.npts

        shift = max(len1,len2) // 2
    
        #now get the cross-correlation function between them
        cc_tr = cross_correlation.correlate(tr1,tr2,shift)
        cc_traces.append(cc_tr)
    ave_cc = np.mean(np.stack(cc_traces,axis=-1),axis=1)
    _ , xcorr = cross_correlation.xcorr_max(ave_cc,abs_max=False)
    return xcorr

def cross_correlate(template,candidate):

    #TODO this seems inefficient...look into speeding it up by vectorising cross correlation with flat arrays.
    #firstly want to get the vertical trace from the station that had the most energy
    max_template = 0
    max_candidate = 0
    for tr in template.get_data_window().select(component='Z'):
        energy = np.sum(tr.data**2)
        if energy > max_template:
            max_template = energy
            template_tr = tr
    
    for tr in candidate.get_data_window().select(component='Z'):
        energy = np.sum(tr.data**2)
        if energy > max_candidate:
            max_candidate = energy
            candidate_tr = tr

    len_template = template_tr.stats.npts
    len_candidate = candidate_tr.stats.npts

    shift = max(len_candidate,len_template) // 2
    
    #now get the cross-correlation function between them
    cc = cross_correlation.correlate(template_tr,candidate_tr,shift)
    shift, xcor = cross_correlation.xcorr_max(cc,abs_max=False)
    return shift, xcor

def xcorr_data(template_tr,candidate_tr):
    len_template = template_tr.stats.npts
    len_candidate = candidate_tr.stats.npts

    shift = max(len_candidate,len_template) // 2
    
    #now get the cross-correlation function between them
    cc = cross_correlation.correlate(template_tr,candidate_tr,shift)
    shift, xcor = cross_correlation.xcorr_max(cc,abs_max=False)
    return shift, xcor


def event_polarisation(event,backazimuth):
    #probably just want stream to have one station which is externally selected to be one with greatest amplitude. This code otherwise selects the first station in the list.
    #backazimuth can either be a single value calculated from beamforming, MFP, etc. or a grid of values to maximise over. 
    #make each of the outputs a tuple with the correlation and the backazimuth it corresponds to.
    if type(backazimuth) != np.ndarray:
        backazimuth = np.array([backazimuth])
    
    xcorr_rayleigh = np.zeros_like(backazimuth)
    xcorr_pwave = np.zeros_like(backazimuth)
    xcorr_swave = np.zeros_like(backazimuth)

    for i, baz in enumerate(backazimuth):
        stream = event.get_data_window().copy()
        stream.rotate(method='NE->RT',back_azimuth=baz)

        radial = stream.select(component='R')[0].data
        vertical = stream.select(component='Z')[0].data
        transverse = stream.select(component='T')[0].data
        
        analytical_signal = hilbert(radial)
        shifted = np.real(np.abs(analytical_signal) * np.exp((np.angle(analytical_signal) + 0.5 * np.pi) * 1j))
        
        xcorr_rayleigh[i] = np.dot(vertical,shifted) / np.sqrt(np.dot(vertical,vertical)*np.dot(shifted,shifted))
        xcorr_pwave[i] = np.dot(vertical,radial) / np.sqrt(np.dot(vertical,vertical)*np.dot(radial,radial))
        xcorr_swave[i] = np.dot(vertical,transverse) / np.sqrt(np.dot(vertical,vertical)*np.dot(transverse,transverse))


    max_rayleigh = np.argmax(xcorr_rayleigh)
    backaz_rayleigh = backazimuth[max_rayleigh]
    corr_rayleigh = np.max(xcorr_rayleigh)

    rayleigh_corr = (corr_rayleigh,backaz_rayleigh)

    max_pwave = np.argmax(xcorr_pwave)
    backaz_pwave = backazimuth[max_pwave]
    corr_pwave = np.max(xcorr_pwave)

    p_corr = (corr_pwave,backaz_pwave)

    max_swave = np.argmax(xcorr_swave)
    backaz_swave = backazimuth[max_swave]
    corr_swave = np.max(xcorr_swave)

    s_corr = (corr_swave, backaz_swave)

    return rayleigh_corr, p_corr, s_corr

def beamforming(event):
    #calculate both backazimuth and horizontal slowness here.
    from obspy.signal.array_analysis import array_processing
    from obspy.core.util import AttribDict
    stream = event.get_data_window()
    for tr in stream:
        network, station, location, channel = tr.id.split('.')
        tr.stats.coordinates = AttribDict({
        'latitude': event.inv.select(station=station)[0][0].latitude,
        'elevation': event.inv.select(station=station)[0][0].elevation/1000,
        'longitude': event.inv.select(station=station)[0][0].longitude})
        

    stime = event.data_window[0]
    etime = event.data_window[1]

    #TODO tune the window length for different length events - test using fraction of event duration for length
    win_length = (etime - stime)
    kwargs = dict(
        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll_x=-3.0, slm_x=3.0, sll_y=-3.0, slm_y=3.0, sl_s=0.03,
        # sliding window properties
        win_len=win_length, win_frac=1.0, 
        # frequency properties
        frqlow=1.0, frqhigh=50.0, prewhiten=0, #todo work out the effect of this highfreq threshold as it affects computation time.
        # restrict output
        semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
        stime=stime, etime=etime
    )
    out = array_processing(stream, **kwargs)
    t, rel_power, abs_power, baz, slow = out.T
    baz[baz < 0.0] += 360
    max_ind = np.argmax(abs_power)
    baz = baz[max_ind]
    slow = slow[max_ind]

    label1 = 'Event Backazimuth [degrees]'
    label2 = 'Horizontal Slowness [s/km]'
    row1 = make_row(7,'backazimuth',baz,label1)
    row2 = make_row(8,'slowness',slow,label2)
    return [row1,row2]

def environment(event):

    row1 = make_row(9,'local_time',event.time_var['local_time'],'Local Time')
    row2 = make_row(10,'temperature',event.time_var['TEM'], 'Air Temperature [$^\circ$C]')
    row3 = make_row(11,'wind_speed',event.time_var['SPE'],'Wind Speed [km/h]')
    row4 = make_row(12,'air_pressure',event.time_var['PRE'],'Air Pressure [hPa]')
    row5 = make_row(13,'rel_humid',event.time_var['HUM'],'Relative Humidity')
    row6 = make_row(14,'tide_height',event.time_var['TID'],'Tidal Height [m]')

    #TODO add row for tidal phase - need to have this as a column in the original .csv file.
    return [row1,row2,row3,row4,row5,row6]


def make_row(number,att_name,value,label):
    row = {'Name':att_name,'Value':value,'Label':label}
    row = pd.Series(data=row,name=number)
    return row


#! new object for dealing with attribute calculations

class Attribute:
    def __init__(self):
        self.att_name = None
        self.att_label = None
        self.att_unit = None

    def prepare_stream(self,event):
        return event
    
    def compute_attribute(self,event):
        return None
    
    def __check_stream(self,event):
        #TODO run a check of the event stream to make sure there are no partial traces, etc.
        pass
    


class WaveformAttributes(Attribute):
    def __init__(self):
        super().__init__()
        self.att_name = ['duration','amplitude']
        self.att_label = ['Duration','Amplitude']
        self.att_unit = ['s','m/s']

    def prepare_stream(self,event):
        #to get good meaure of amplitude for event rather than background, should filter and then remove sensitivity...
        event.filter('highpass',freq=1)
        event.remove_sensitivity()
        return event
    
    def compute_attribute(self,event):
        duration = event.duration

        #calculate max amp of euclidian streams.
        max_amps = []
        window = event.get_data_window() #only look for max within the narrower data window.
        for network in event.inv:
            for station in network:
                sta_stream = window.select(station=station.code) #just get the traces for this station
                euc = np.sqrt(sum([tr.data**2 for tr in sta_stream]))
                max_amps.append(np.max(euc))
        amplitude = np.max(np.array(max_amps))
        
        return [duration,amplitude]

class SpectralAttributes(Attribute):
    def __init__(self):
        super().__init__()
        self.att_name = ['central_frequency','max_frequency','second_spectral_moment','spectral_deviation','hv_ratio']
        self.att_label = ['Central Frequency','Maximum Frequency','Second Frequency Moment','Spectral Deviation','Horizontal/Vertical Energy']
        self.att_unit = ['Hz','Hz','Hz$^2$','Hz','']

    def prepare_stream(self,event):
        event.filter('highpass',freq=1)
        event.remove_sensitivity()
        return event
        
    def compute_attribute(self,event):
        from scipy.signal import periodogram
        from obspy.signal.util import nearest_pow_2

        spectra = []
        vert_spec = []
        hor_spec = []
        centroid = []
        max_f = []

        stream = event.get_data_window() #this is the same window that is plotted for consistancy with manual interpretation.

        for tr in stream:
            f, Pxx = periodogram(tr.data,window='blackman',fs=tr.stats.sampling_rate,nfft=nearest_pow_2(tr.data.size))
            spectra.append(Pxx)
            if tr.id[-1] == 'Z':
                vert_spec.append(Pxx)
            else:
                hor_spec.append(Pxx)
        
        spectrum = sum(spectra)
        vert_sum = np.sum(sum(vert_spec))
        hor_sum = np.sum(sum(hor_spec))
        h_over_v = hor_sum / vert_sum
        spectrum /= np.sum(spectrum)
        centroid = np.sum(spectrum * f)
        second_mom = np.sum(spectrum * (f**2))
        deviation = np.sqrt(second_mom - centroid**2)
        max_f = f[spectrum.argmax()]
      
        return [centroid,max_f,second_mom,deviation,h_over_v]
    


