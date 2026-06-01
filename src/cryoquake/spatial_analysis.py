from cryoquake.data_objects import SeismicEvent, SeismicChunk
import numpy as np
from obspy.signal.filter import envelope
from obspy.signal.util import next_pow_2
from scipy.signal import get_window, hilbert
from obspy.geodetics import degrees2kilometers, kilometers2degrees
from multiprocessing import Pool, shared_memory
import pandas as pd
from numpy.fft import rfft, irfft
from numpy.linalg import eigvalsh, LinAlgError, eigh
import pyproj
import xarray as xr
from scipy.stats import chi2, norm



class EventBeamforming(SeismicEvent):
    def beamforming(self,pre_buffer=2,post_buffer=4): #TODO this will be moved to a child class for location or attribute based calculations...
        from obspy.signal.array_analysis import array_processing
        from obspy.core.util import AttribDict
        for tr in self.stream:
            network, station, location, channel = tr.id.split('.')
            tr.stats.coordinates = AttribDict({
            'latitude': self.inv.select(station=station)[0][0].latitude,
            'elevation': self.inv.select(station=station)[0][0].elevation/1000,
            'longitude': self.inv.select(station=station)[0][0].longitude})
            
        stime = self.window_start - pre_buffer
        etime = self.window_start + post_buffer

        kwargs = dict(
            # slowness grid: X min, X max, Y min, Y max, Slow Step
            sll_x=-3.0, slm_x=3.0, sll_y=-3.0, slm_y=3.0, sl_s=0.03,
            # sliding window properties
            win_len=3.0, win_frac=0.05,
            # frequency properties
            frqlow=1.0, frqhigh=8.0, prewhiten=0,
            # restrict output
            semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
            stime=stime, etime=etime
        )
        out = array_processing(self.stream, **kwargs)
        t, rel_power, abs_power, baz, slow = out.T
        baz[baz < 0.0] += 360
        max_ind = np.argmax(abs_power)
        self.baz = baz[max_ind]
        self.slow = slow[max_ind]

    def rotate_components(self):
        """
        Rotates 3-component stream from ZNE to ZRT coordinates if a back azimuth has been assigned to the event.
        """
        try:
            self.stream.rotate(method='NE->RT',back_azimuth=self.baz)
        except:
            raise(Exception('Back azimuth must be computed before rotating components'))
        

class EventBeampower(SeismicEvent):

    def fourier(self,frqlow=0,frqhigh=500,remove_response=True,frq_step=1):

        flattened = self.flat_window()
        nsamp = flattened.shape[1]
        blackman = get_window("blackman",nsamp) 

        #demean and taper the traces.
        flattened -= np.mean(flattened,axis=1)[:,None]
        flattened *= blackman
        
        sta_ind = np.argwhere(~(np.isnan(flattened).any(axis=1))).flatten()
        self.sta_ind = sta_ind

        #set up some fft lengths, etc.
        nfft = next_pow_2(nsamp)
        fs = self.stream[0].stats.sampling_rate
        deltaf = fs / float(nfft)

        ft = np.fft.rfft(flattened,nfft,axis=1)[sta_ind,:]
        frq = (np.arange(nfft//2+1) * deltaf)
     
        nlow = int(frqlow / float(deltaf) + 0.5)
        nhigh = int(frqhigh / float(deltaf) + 0.5)
        nlow = max(1, nlow)  # avoid using the offset
        nhigh = min(nfft // 2 - 1, nhigh)  # avoid using nyquist

        nf = nhigh - nlow + 1  # include upper and lower frequency
        frq = frq[nlow:nlow+nf]

        #get spectra for each station and compute cross-spectral density matrices.
        ft = (ft[:,nlow:nlow+nf]).T #[freq, station]

        #remove the response if this information is given. 
        if remove_response:
            resp = self.flat_response(frq)[:,sta_ind] #[freq,station]
            ft /= resp  #remove the response -> #! there are obspy functions for stably inverting response spec with water level if needed.
        
        frq = frq[::frq_step]
        ft = ft[::frq_step,:]

        self.frq = frq
        self.ft = ft
        self.frq_range = (frq[0],frq[-1])
 

    def geometry(self,tt_array):
        self.tt_array = tt_array[:,:,:,self.sta_ind]

    def coherence(self):
        B = np.zeros([self.tt_array.shape[0],self.tt_array.shape[1],self.tt_array.shape[2],self.frq.size])
        for i in range(self.frq.size):
            K = (self.ft[i,:,None]) @ (self.ft.conj()[i,None,:]) #[freq,station,station]
        
            #make the steering vector from the travel time grid and freqeuncies.
            #tt_array must have shape [backazimuth,radius,slowness,station]
            d = np.exp(-2*np.pi*1j*self.frq[i]*self.tt_array) #[backazimuth,radius,slowness,station]

            B[:,:,:,i] = np.abs(d.conj()[:,:,:,None,:] @ (K @ d[:,:,:,:,None]))[:,:,:,0,0] #Bartlett processor

        B_mean = np.mean(B,axis=-1) #[backazimuth,radius,slowness]
        
        return B_mean

    def opt_objective(self,point):
        baz_grid = np.array(point[0])
        rad_grid = np.array(point[1])
        slow_grid = np.array(point[2])

        tt_array, _ = radial_grid(baz_grid,rad_grid,slow_grid,self.inv)
        self.geometry(tt_array)
        obj = - np.sum(np.squeeze(self.coherence()))
        return obj
        


    def envelope_power(self,travel_times):
        first = min(travel_times)
        tt_rel = {key:travel_times[key] - travel_times[first] for key in travel_times.keys()}

        shifted = []
        for tr in self.stream: #also loop through stream on each core, but just pointer to object so ok?
            tr_copy = tr.copy()
            net, sta, loc, cha = tr.id.split('.')
            shift = tt_rel[sta]
            samp = round(shift * tr.stats.sampling_rate)
            shift = samp / tr.stats.sampling_rate
            t1 = self.data_window[0] + shift
            t2 = self.data_window[1] + shift
            trimmed = tr_copy.trim(t1,t2,pad=True,fill_value=0,nearest_sample=False).data
            trimmed = envelope(trimmed)
            shifted.append(trimmed)
        beampower = np.sum(sum(shifted)**2)

        return beampower

    
    def vertical_csd_vectorised(self,tt_array,frqlow=0,frqhigh=500,remove_response=False,normalise=False,processor='bartlett'):
        """
        This is all vectorised without needing to be parallelised -> can therefore loop to use multiproccessing over the events
        get additional speedup. Part that cannot be sped up is attachement of waveforms, so can attach waveforms, do this, then do template matching.

        """
        flattened = self.flat_window()
        nsamp = flattened.shape[1]
        blackman = get_window("blackman",nsamp) 

        #demean and taper the traces.
        flattened -= np.mean(flattened,axis=1)[:,None]
        flattened *= blackman
        
        sta_ind = np.argwhere(~(np.isnan(flattened).any(axis=1))).flatten()

        #set up some fft lengths, etc.
        nfft = next_pow_2(nsamp)
        fs = self.stream[0].stats.sampling_rate
        deltaf = fs / float(nfft)
        nlow = int(frqlow / float(deltaf) + 0.5)
        nhigh = int(frqhigh / float(deltaf) + 0.5)
        nlow = max(1, nlow)  # avoid using the offset
        nhigh = min(nfft // 2 - 1, nhigh)  # avoid using nyquist
        nf = nhigh - nlow + 1  # include upper and lower frequency
        frq = (np.arange(nfft//2+1) * deltaf)[nlow:nlow+nf]

        #get spectra for each station and compute cross-spectral density matrices.
        ft = (np.fft.rfft(flattened,nfft,axis=1)[sta_ind,nlow:nlow+nf]).T #[freq, station]

        #remove the response if this information is given. 
        if remove_response:
            resp = self.flat_response(frq)[:,sta_ind] #[freq,station]
            ft /= resp  #remove the response -> #! there are obspy functions for stably inverting response spec with water level if needed.
        
    
        K = (ft[:,:,None]) @ (ft.conj()[:,None,:]) #[freq,station,station]

        if normalise: #divide by product of lengths to get unit complex number for each entry in CSDM -> just retains phase information.
            temp = np.abs(ft)
            norm = ((temp[:,:,None]) @ (temp[:,None,:])) #[freq,station,station]
            K /= norm #[freq,station,station]

        #make the steering vector from the travel time grid and freqeuncies.
        #tt_array must have shape [backazimuth,radius,slowness,station], steering vector adds freq dimension.
        d = np.exp(-2*np.pi*1j*frq[None,None,None,:,None]*tt_array[:,:,:,None,sta_ind]) #[backazimuth,radius,slowness,freq,station]

        if processor == 'bartlett':
            Kc = np.ascontiguousarray(K)
            dc = np.ascontiguousarray(d)

            B = np.squeeze(np.abs(dc.conj()[:,:,:,None,:] @ (Kc @ dc[:,:,:,:,None]))) #Bartlett processor
            #B = np.squeeze(np.abs(d[:,:,:,:,None,:].conj() @ (K @ d[:,:,:,:,:,None]))) #Bartlett processor

        if processor == 'capon':
            K_inv = np.linalg.inv(K) #TODO might be a faster way using linear solver as we are multiplying by a vector.
            B = 1 / np.squeeze(np.abs(d[:,:,:,:,None,:].conj() @ (K_inv @ d[:,:,:,:,:,None]))) #Capon processor

        B_mean = np.mean(B,axis=-1) #[backazimuth,radius,slowness]
        
        return B_mean
    
    def vertical_csd_frq_loop(self,tt_array,frqlow=0,frqhigh=500,remove_response=False,normalise=False,frq_step=1,supress_noise=True):
        """
        This is all vectorised without needing to be parallelised -> can therefore loop to use multiproccessing over the events
        get additional speedup. Part that cannot be sped up is attachement of waveforms, so can attach waveforms, do this, then do template matching.

        """
        flattened = self.flat_window()
        nsamp = flattened.shape[1]
        blackman = get_window("blackman",nsamp) 

        #demean and taper the traces.
        flattened -= np.mean(flattened,axis=1)[:,None]
        flattened *= blackman
        
        sta_ind = np.argwhere(~(np.isnan(flattened).any(axis=1))).flatten()

        #set up some fft lengths, etc.
        nfft = next_pow_2(nsamp)
        fs = self.stream[0].stats.sampling_rate
        deltaf = fs / float(nfft)
        nlow = int(frqlow / float(deltaf) + 0.5)
        nhigh = int(frqhigh / float(deltaf) + 0.5)
        nlow = max(1, nlow)  # avoid using the offset
        nhigh = min(nfft // 2 - 1, nhigh)  # avoid using nyquist
        nf = nhigh - nlow + 1  # include upper and lower frequency
        frq = (np.arange(nfft//2+1) * deltaf)[nlow:nlow+nf]

        #get spectra for each station and compute cross-spectral density matrices.
        ft = (np.fft.rfft(flattened,nfft,axis=1)[sta_ind,nlow:nlow+nf]).T #[freq, station]

        #remove the response if this information is given. 
        if remove_response:
            resp = self.flat_response(frq)[:,sta_ind] #[freq,station]
            ft /= resp  #remove the response -> #! there are obspy functions for stably inverting response spec with water level if needed.
        
        frq = frq[::frq_step]
        ft = ft[::frq_step,:]

        B = np.zeros([tt_array.shape[0],tt_array.shape[1],tt_array.shape[2],frq.size])
        for i in range(frq.size):
            K = (ft[i,:,None]) @ (ft.conj()[i,None,:]) #[freq,station,station]
            if supress_noise:
                np.fill_diagonal(K,0)

            if normalise: #divide by product of lengths to get unit complex number for each entry in CSDM -> just retains phase information.
                temp = np.abs(ft[i,:])
                norm = ((temp[:,None]) @ (temp[None,:])) #[station,station]
                K /= norm #[station,station]

            #make the steering vector from the travel time grid and freqeuncies.
            #tt_array must have shape [backazimuth,radius,slowness,station]
            d = np.exp(-2*np.pi*1j*frq[i]*tt_array[:,:,:,sta_ind]) #[backazimuth,radius,slowness,station]

            Kc = np.ascontiguousarray(K)
            dc = np.ascontiguousarray(d)

            B[:,:,:,i] = np.abs(dc.conj()[:,:,:,None,:] @ (Kc @ dc[:,:,:,:,None]))[:,:,:,0,0] #Bartlett processor
    

        B_mean = np.mean(B,axis=-1) #[backazimuth,radius,slowness]
        
        return B_mean
    
    

    def set_conditions(self,frqlow=0,frqhigh=500,remove_response=False,normalise=False,frq_step=1,supress_noise=True):

        N = len(self.inv)
        locations_ll = np.zeros((N,2))

        i = 0
        for net in self.inv:
            for sta in net:
                locations_ll[i,:] = np.array([sta.latitude,sta.longitude])
                i += 1
        
        centre = np.mean(locations_ll,axis=0)
        d_ll = locations_ll - centre[None,:]

        sta_xy = np.zeros_like(d_ll)
        sta_xy[:,0] = degrees2kilometers(d_ll[:,1],radius=6371.0*np.cos(np.deg2rad(centre[0])))
        sta_xy[:,1] = degrees2kilometers(d_ll[:,0])

        self.sta_xy = sta_xy
        self.centre = centre

        flattened = self.flat_window()
        nsamp = flattened.shape[1]
        blackman = get_window("blackman",nsamp) 

        #demean and taper the traces.
        flattened -= np.mean(flattened,axis=1)[:,None]
        flattened *= blackman
        
        self.sta_ind = np.argwhere(~(np.isnan(flattened).any(axis=1))).flatten()

        #set up some fft lengths, etc.
        nfft = next_pow_2(nsamp)
        fs = self.stream[0].stats.sampling_rate
        deltaf = fs / float(nfft)
        nlow = int(frqlow / float(deltaf) + 0.5)
        nhigh = int(frqhigh / float(deltaf) + 0.5)
        nlow = max(1, nlow)  # avoid using the offset
        nhigh = min(nfft // 2 - 1, nhigh)  # avoid using nyquist
        nf = nhigh - nlow + 1  # include upper and lower frequency
        frq = (np.arange(nfft//2+1) * deltaf)[nlow:nlow+nf]

        #get spectra for each station and compute cross-spectral density matrices.
        ft = (np.fft.rfft(flattened,nfft,axis=1)[self.sta_ind,nlow:nlow+nf]).T #[freq, station]

        #remove the response if this information is given. 
        if remove_response:
            resp = self.flat_response(frq)[:,self.sta_ind] #[freq,station]
            ft /= resp  #remove the response -> #! there are obspy functions for stably inverting response spec with water level if needed.
        
        self.frq = frq[::frq_step]
        self.ft = ft[::frq_step,:]

        self.K = np.zeros([frq.size,ft.shape[1],ft.shape[1]],dtype=np.complex128)

        self.normalise = normalise
        self.supress_noise = supress_noise

        for i in range(self.frq.size):
            K = (ft[i,:,None]) @ (ft.conj()[i,None,:])
            if supress_noise:
                np.fill_diagonal(K,0)
            self.K[i,:,:] = K

            if self.normalise: #divide by product of lengths to get unit complex number for each entry in CSDM -> just retains phase information.
                temp = np.abs(ft[i,:])
                norm = ((temp[:,None]) @ (temp[None,:])) #[station,station]
                self.K[i,:,:] /= norm #[station,station]



    def objective(self,point):

        baz = point[0]
        rad = point[1]
        slow = point[2]
        
        x = rad * np.sin(baz) #Easting, opposite of normal way of doing conversions due to backazimuth defintion
        y = rad * np.cos(baz) #Northing
        travel_time = np.sqrt((x - self.sta_xy[:,0])**2 + (y - self.sta_xy[:,1])**2) * slow
          
        d = np.exp(-2*np.pi*1j*self.frq[:,None]*travel_time[None,:]) #[freq,station]


        B = np.squeeze(np.abs(d.conj()[:,None,:] @ (self.K @ d[:,:,None]))) #Bartlett processor
    

        B_mean = - np.mean(B,axis=0)
        
        return B_mean





def radial_grid(backazimuth_grid,radial_grid,slowness_grid,inv):
    theta, r, slow = np.meshgrid(backazimuth_grid,radial_grid,slowness_grid,indexing='ij')
    x = r * np.sin(theta) #Easting, opposite of normal way of doing conversions due to backazimuth defintion
    y = r * np.cos(theta) #Northing

    N = len(inv)
    locations_ll = np.zeros((N,2))

    i = 0
    for net in inv:
        for sta in net:
            locations_ll[i,:] = np.array([sta.latitude,sta.longitude])
            i += 1
    
    centre = np.mean(locations_ll,axis=0)
    d_ll = locations_ll - centre[None,:]

    sta_xy = np.zeros_like(d_ll)
    sta_xy[:,0] = degrees2kilometers(d_ll[:,1],radius=6371.0*np.cos(np.deg2rad(centre[0])))
    sta_xy[:,1] = degrees2kilometers(d_ll[:,0])

    travel_times = np.sqrt((x[:,:,:,None] - sta_xy[:,0])**2 + (y[:,:,:,None] - sta_xy[:,1])**2) * slow[:,:,:,None]
    return travel_times , centre

def polarisation_grid(backazimuth_grid,radial_grid,inv,centre=None):
    theta, r = np.meshgrid(backazimuth_grid,radial_grid,indexing='ij')
    x = r * np.sin(theta) #Easting, opposite of normal way of doing conversions due to backazimuth defintion
    y = r * np.cos(theta) #Northing

    N = len(inv)
    locations_ll = np.zeros((N,2))

    i = 0
    for net in inv:
        for sta in net:
            locations_ll[i,:] = np.array([sta.latitude,sta.longitude])
            i += 1

    if centre is None:
        centre = np.mean(locations_ll,axis=0)
    
    d_ll = locations_ll - centre[None,:]

    sta_xy = np.zeros_like(d_ll)
    sta_xy[:,0] = degrees2kilometers(d_ll[:,1],radius=6371.0*np.cos(np.deg2rad(centre[0])))
    sta_xy[:,1] = degrees2kilometers(d_ll[:,0])

    diff_x = x[:,:,None] - sta_xy[:,0]
    diff_y = y[:,:,None] - sta_xy[:,1]

    baz = np.arctan2(diff_x,diff_y) #arctan(x/y) due to backazimuth coordinate system
    return baz , centre



def centre_baz_to_station_baz(baz,rad,inv,centre):
    """
    Takes the radial distance and backazimuth relative to the centre of the array estimated from MFP and converts it
    to a backazimuth relative to each station for polarisation analysis
    """
    N = len(inv)
    locations_ll = np.zeros((N,2))

    i = 0
    for net in inv:
        for sta in net:
            locations_ll[i,:] = np.array([sta.latitude,sta.longitude])
            i += 1
    
    d_ll = locations_ll - centre[None,:]

    sta_xy = np.zeros((N,2))
    sta_xy[:,0] = degrees2kilometers(d_ll[:,1],radius=6371.0*np.cos(np.deg2rad(centre[0])))
    sta_xy[:,1] = degrees2kilometers(d_ll[:,0])

    source_xy = np.array([rad*np.sin(baz),rad*np.cos(baz)]) #x,y position of the proposed source

    #now need to make vectors that point from the station to the source, then find the angle of these vectors.
    vec_xy = source_xy[None,:] - sta_xy

    station_baz = np.arctan2(vec_xy[:,0],vec_xy[:,1]) #arctan(x/y) rather than y/x due to north being zero with clockwise positive.

    return station_baz

def centre_rad_to_station_rad(baz,rad,inv,centre):
    N = len(inv)
    locations_ll = np.zeros((N,2))

    i = 0
    for net in inv:
        for sta in net:
            locations_ll[i,:] = np.array([sta.latitude,sta.longitude])
            i += 1
    
    d_ll = locations_ll - centre[None,:]

    sta_xy = np.zeros((N,2))
    sta_xy[:,0] = degrees2kilometers(d_ll[:,1],radius=6371.0*np.cos(np.deg2rad(centre[0])))
    sta_xy[:,1] = degrees2kilometers(d_ll[:,0])

    source_xy = np.array([rad*np.sin(baz),rad*np.cos(baz)]) #x,y position of the proposed source

    #now need to make vectors that point from the station to the source, then find the length of these vectors.
    vec_xy = source_xy[None,:] - sta_xy

    station_rad = np.sqrt(vec_xy[:,0]**2 + vec_xy[:,1]**2)
    return station_rad



class EventPolarisation(SeismicEvent):

    def phase_shift(self):
        window = self.get_data_window()
        north = np.stack([tr.data.astype(np.float64) for tr in window.select(component='N')],axis=0) #[stations,N]
        east = np.stack([tr.data.astype(np.float64) for tr in window.select(component='E')],axis=0) #[stations,N]
        vert = np.stack([tr.data.astype(np.float64) for tr in window.select(component='Z')],axis=0) #[stations,N]

        self.north = north
        self.east = east
        self.vert = vert

        all_data = np.hstack([north,east,vert])
        self.sta_ind = np.argwhere(~(np.isnan(all_data).any(axis=1))).flatten()

        analytical_signal = hilbert(vert[self.sta_ind,:],axis=1) #[backazimuth,radial,stations,N]
        self.shifted = np.real(np.abs(analytical_signal) * np.exp((np.angle(analytical_signal) - 0.5 * np.pi) * 1j)) #[backazimuth,radial,stations,N] #! changed from + to - from before

    def geometry(self,baz_array,centre):
        self.baz_grid = baz_array
        self.centre = centre


    def rotate(self):
        radial = - np.cos(self.baz_grid[:,:,self.sta_ind,None]) * self.north[self.sta_ind,:] - np.sin(self.baz_grid[:,:,self.sta_ind,None])*self.east[self.sta_ind,:] #[backazimuth,radial,stations,N]
        self.radial = radial

    def correlation(self):

        R_corr = np.sum(self.shifted[None,None,:,:] * self.radial,axis=3) #[backazimith,radial,station]
        total_R_corr = np.sum(R_corr,axis=2) #[backazimith,radial]

        R_norm = np.sqrt(np.sum(self.shifted[None,None,:,:]**2,axis=3) * np.sum(self.radial**2,axis=3)) #[backazimith,radial,station]
        R_corr /= R_norm #[backazimuth,radial,station]

        total_R_norm = np.sqrt(np.sum(self.shifted[None,None,:,:]**2,axis=(2,3)) * np.sum(self.radial**2,axis=(2,3))) #[backazimith,radial]
        total_R_corr /= total_R_norm

        return R_corr, total_R_corr
    
    def opt_objective(self,point):
        baz_grid = np.array(point[0])
        rad_grid = np.array(point[1])

        baz_array, _ = polarisation_grid(baz_grid,rad_grid,self.inv,self.centre)
        
        self.phase_shift()
        self.geometry(baz_array,self.centre)
        self.rotate()
        R_corr, total_R_corr = self.correlation()
        obj = - np.sum(np.squeeze(total_R_corr))
        return obj

    def correlation_power(self,baz_grid):

        window = self.get_data_window()
        north = np.stack([tr.data.astype(np.float64) for tr in window.select(component='N')],axis=0) #[stations,N]
        east = np.stack([tr.data.astype(np.float64) for tr in window.select(component='E')],axis=0) #[stations,N]
        vert = np.stack([tr.data.astype(np.float64) for tr in window.select(component='Z')],axis=0) #[stations,N]

        all_data = np.hstack([north,east,vert])
        sta_ind = np.argwhere(~(np.isnan(all_data).any(axis=1))).flatten()

        radial = - np.cos(baz_grid[:,:,sta_ind,None]) * north[sta_ind,:] - np.sin(baz_grid[:,:,sta_ind,None])*east[sta_ind,:] #[backazimuth,radial,stations,N]
        analytical_signal = hilbert(radial,axis=3) #[backazimuth,radial,stations,N]
        shifted = np.real(np.abs(analytical_signal) * np.exp((np.angle(analytical_signal) + 0.5 * np.pi) * 1j)) #[backazimuth,radial,stations,N]

        R_corr = np.sum(shifted * vert[None,None,sta_ind,:],axis=3) #[backazimith,radial,station]

        R_norm = np.sqrt(np.sum(shifted**2,axis=-1) * np.sum(vert[sta_ind,:]**2,axis=-1)) #[backazimith,radial,station]
        total_R_norm = np.sqrt(np.sum(shifted**2,axis=(2,3)) * np.sum(vert[sta_ind,:]**2,axis=(0,1)))

        total_R_corr = np.sum(R_corr,axis=2)
        total_R_corr /= total_R_norm

        R_corr /= R_norm

        P_corr = np.sum(radial * vert[None,None,sta_ind,:],axis=3) #[backazimith,radial,station]
        
        P_norm = np.sqrt(np.sum(radial**2,axis=-1) * np.sum(vert[sta_ind,:]**2,axis=-1)) #[backazimith,radial,station]
        total_P_norm = np.sqrt(np.sum(radial**2,axis=(2,3)) * np.sum(vert[sta_ind,:]**2,axis=(0,1)))

        total_P_corr = np.sum(P_corr,axis=2)
        total_P_corr /= total_P_norm

        P_corr /= P_norm

        return R_corr, total_R_corr, P_corr, total_P_corr
    

    def objective(self,point):

        baz = point[0]
        rad = point[1]
        
        x = rad * np.sin(baz) #Easting, opposite of normal way of doing conversions due to backazimuth defintion
        y = rad * np.cos(baz) #Northing
        travel_time = np.sqrt((x - self.sta_xy[:,0])**2 + (y - self.sta_xy[:,1])**2) * slow
          
        d = np.exp(-2*np.pi*1j*self.frq[:,None]*travel_time[None,:]) #[freq,station]


        B = np.squeeze(np.abs(d.conj()[:,None,:] @ (self.K @ d[:,:,None]))) #Bartlett processor
    

        B_mean = - np.mean(B,axis=0)
        
        return B_mean
    

        

def Cartesian2Coords(x,y,centre):
    lon = centre[1] + kilometers2degrees(x,radius=6371.0*np.cos(np.deg2rad(centre[0])))
    lat = centre[0] + kilometers2degrees(y)
    return lon, lat

def CartesianStations(inv):
    N = len(inv)
    locations_ll = np.zeros((N,3))

    i = 0
    for net in inv:
        for sta in net:
            locations_ll[i,:] = np.array([sta.latitude,sta.longitude,-sta.elevation/1000])
            i += 1

    centre = np.mean(locations_ll,axis=0)

    d_ll = locations_ll - centre[None,:]

    sta_xy = np.zeros_like(d_ll)
    sta_xy[:,0] = degrees2kilometers(d_ll[:,1],radius=6371.0*np.cos(np.deg2rad(centre[0])))
    sta_xy[:,1] = degrees2kilometers(d_ll[:,0])
    sta_xy[:,2] = locations_ll[:,2]

    return sta_xy, centre

def EPSGStations(inv):
    N = len(inv)
    locations_ll = np.zeros((N,3))

    i = 0
    sta_lst = []
    for net in inv:
        for sta in net:
            locations_ll[i,:] = np.array([sta.latitude,sta.longitude,sta.elevation])
            sta_lst.append(sta.code)
            i += 1

    proj_wgs84 = pyproj.Proj(init="epsg:4326")
    proj_epsg3031 = pyproj.Proj(init="epsg:3031")
    x, y = pyproj.transform(proj_wgs84, proj_epsg3031, locations_ll[:,1], locations_ll[:,0])

    sta_xy = np.zeros_like(locations_ll)

    sta_xy[:,0] = x
    sta_xy[:,1] = y
    sta_xy[:,2] = locations_ll[:,2]

    #now make a pairwise distance matrix for the network...
    D = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            D[i,j] = np.sqrt(np.sum((sta_xy[i,:] - sta_xy[j,:])**2))

    return sta_xy, D, sta_lst


def CartesianStationsDF(inv):

    rows = []
    labels = []

    for net in inv:
        for sta in net:
            row = {'Latitude':sta.latitude,'Longitude':sta.longitude,'Depth':-sta.elevation/1000}
            rows.append(row)
            labels.append(sta.code)
    
    df = pd.DataFrame(data=rows,index=labels)

    centre = df.mean(axis=0)

    df['rLatitude'] = df['Latitude'] - centre['Latitude']
    df['rLongitude'] = df['Longitude'] - centre['Longitude']

    df['X'] = degrees2kilometers(df['rLongitude'].to_numpy(),radius=6371.0*np.cos(np.deg2rad(centre['Latitude'])))
    df['Y'] = degrees2kilometers(df['rLatitude'])
    df['Z'] = df['Depth']

    return df, centre


def ArrivalTimes(inv,fit,uncertainties,Vp=3.87,Vs=1.90):
    """
    Compute the P wave window at each station based on the fitted location and origin time from backmigration.
    """
    x, y, z, t, r = fit
    dr = uncertainties[-1]

    sta_xy, _ = CartesianStationsDF(inv)
    sta_xy['diffX'] = sta_xy['X'] - x
    sta_xy['diffY'] = sta_xy['Y'] - y
    sta_xy['diffZ'] = sta_xy['Z'] - z
    sta_xy['R'] = np.sqrt(sta_xy['diffX']**2 + sta_xy['diffY']**2 + sta_xy['diffZ']**2)
    sta_xy['dR'] = dr

    arrivals = sta_xy.copy()

    arrivals['P_start'] = t + arrivals['R']/Vp
    arrivals['P_end'] = t + arrivals['R']/Vs
    arrivals['window'] = arrivals['P_end'] - arrivals['P_start']

    return arrivals



def RollingStreamModulation(Z, N, E, window_len, g=4, smooth=5, overlap=0.5, taper="hann"):
   
    total_length = Z.size
    mod_Z = np.zeros_like(Z, dtype=float)
    mod_N = np.zeros_like(N, dtype=float)
    mod_E = np.zeros_like(E, dtype=float)

    # Taper window
    if taper == "hann":
        taper_win = np.hanning(window_len)
    elif taper == "hamming":
        taper_win = np.hamming(window_len)
    else:
        taper_win = np.ones(window_len)

    step = int(window_len * (1 - overlap))
    if step <= 0:
        raise ValueError("overlap too large: step size <= 0")

    # Overlap-add weighting array (for normalisation)
    weight = np.zeros_like(Z, dtype=float)

    for window_start in range(0, total_length - window_len + 1, step):
        window_end = window_start + window_len

        # Slice windowed time series
        window_Z = Z[window_start:window_end] * taper_win
        window_N = N[window_start:window_end] * taper_win
        window_E = E[window_start:window_end] * taper_win

        # FFT
        Zf = rfft(window_Z)
        Nf = rfft(window_N)
        Ef = rfft(window_E)

        # Shape (n_freq, 3)
        M = np.stack((Zf, Nf, Ef), axis=1)

        n_freq = M.shape[0]
        P = np.zeros(n_freq)

        # For each frequency bin, build cross-spectral matrix averaged over neighbours
        for f in range(n_freq):
            lo = max(0, f - smooth)
            hi = min(n_freq, f + smooth + 1)
            band = M[lo:hi]  # (bandwidth, 3)

            # Cross-spectral density matrix (3x3)
            S = np.einsum("bi,bj->ij", band, band.conj()) / band.shape[0]

            try:
                eig_vals = np.real(eigvalsh(S))
                P[f] = eig_vals[-1] / eig_vals.sum() if eig_vals.sum() > 0 else 0.0
            except LinAlgError:
                P[f] = np.nan
        

        # Apply polarisation weighting
        Zf_mod = Zf * (P**g)
        Nf_mod = Nf * (P**g)
        Ef_mod = Ef * (P**g)

        # IFFT back to time domain, reapply taper for overlap-add
        z_time = irfft(Zf_mod, n=window_len) * taper_win
        n_time = irfft(Nf_mod, n=window_len) * taper_win
        e_time = irfft(Ef_mod, n=window_len) * taper_win

        mod_Z[window_start:window_end] += z_time
        mod_N[window_start:window_end] += n_time
        mod_E[window_start:window_end] += e_time
        weight[window_start:window_end] += taper_win**2

    # Normalise by overlap-add weights to avoid amplitude distortion
    weight[weight == 0] = 1.0
    mod_Z /= weight
    mod_N /= weight
    mod_E /= weight

    return mod_Z, mod_N, mod_E



"""
bed scan using travel time inversion
"""




def DistXr(inv,bed_grid):
    xlin = bed_grid.x.to_numpy()
    ylin = bed_grid.y.to_numpy()
    d = xr.Dataset(coords=dict(x=xlin,y=ylin))
    dz = xr.Dataset(coords=dict(x=xlin,y=ylin))

    proj_wgs84 = pyproj.Proj(init="epsg:4326")
    proj_epsg3031 = pyproj.Proj(init="epsg:3031")

    for network in inv:
        for station in network:
            lat, lon, sta_z, = station.latitude,station.longitude,station.elevation
            sta_x, sta_y = pyproj.transform(proj_wgs84, proj_epsg3031, lon, lat)

            dX = d.x - sta_x
            dY = d.y - sta_y
            dZ = bed_grid.bed_topography - sta_z

            dz = dz.assign({station.code:np.abs(dZ)}) #difference in elevation between station and bed
                    
            dist = (dX**2 + dY**2 + dZ**2)**(0.5)

            d = d.assign({station.code:dist})
    
    sigma_D = bed_grid.bed_uncertainty * (dz / d)
    return d, sigma_D


def BedScan(inv,p_picks,bed_grid,V=3900,sigma_V=100,Delta=10000,bed_error=True,no_model_error=False):
    """
    inv: obspy inventory with the stations
    p_picks: pandas dataframe with the 95% bounds on the p arrival time.
    bed_grid: xarray with the desired x and y coordinates for the grid search, and the bed topography (with uncertainties)
    """

    xlin = bed_grid.x.to_numpy()
    ylin = bed_grid.y.to_numpy()

    d, sigma_D = DistXr(inv,bed_grid)

    

    #h = xr.Dataset(coords=dict(x=xlin,y=ylin))
    #d = xr.Dataset(coords=dict(x=xlin,y=ylin))
    #dz = xr.Dataset(coords=dict(x=xlin,y=ylin))
    gamma = xr.Dataset(coords=dict(x=xlin,y=ylin))

    h = d / V

    if ~bed_error:
        sigma_D = 0.0 * sigma_D

    sta_xy, D, sta_lst = EPSGStations(inv)

    #turn the above picks and uncertainties into arrays and mark missing data
    mu_P = np.zeros(len(sta_lst))
    sigma_P = np.zeros(len(sta_lst))
    pick_ind = np.full(len(sta_lst),fill_value=True)
    pick_sta = []
    for i, sta_code in enumerate(sta_lst):
        mu_P[i] = p_picks['mu'][sta_code]
        sigma_P[i] = p_picks['sigma'][sta_code]

        if np.isnan(p_picks['mu'][sta_code]):
            pick_ind[i] = False
        else:
            pick_sta.append(sta_code)

    mu_P = mu_P[pick_ind]
    sigma_P = sigma_P[pick_ind]
    D = D[pick_ind,:][:,pick_ind]
    N = np.sum(pick_ind) #number of stations where we have picks

    C_t = np.zeros((xlin.size,ylin.size,N,N))
    C_T = np.zeros((xlin.size,ylin.size,N,N))

    for i_s, sta_i in enumerate(pick_sta):

        C_t[:,:,i_s,i_s] += sigma_P[i_s]**2 #include the picking error to the diagonals

        for j_s, sta_j in enumerate(pick_sta):
            
            h_i = h.variables[sta_i].to_numpy()
            h_j = h.variables[sta_j].to_numpy()

            D_i = d.variables[sta_i].to_numpy()
            D_j = d.variables[sta_j].to_numpy()

            sigma_D_i = sigma_D.variables[sta_i].to_numpy()
            sigma_D_j = sigma_D.variables[sta_j].to_numpy()

            rho_D = 1.0
            rho_V = np.exp(-0.5 * D[i_s,j_s]**2 / (Delta**2))

            if i_s == j_s:
                rho_DV = 1.0
            else:
                rho_DV = 0.0
            
            C_T[:,:,i_s,j_s] = h_i * h_j * (((rho_D * sigma_D_i * sigma_D_j) / (D_i * D_j))**2 + 
                                            ((rho_DV * sigma_D_i * sigma_V) / (D_j * V))**2 +
                                            ((rho_DV * sigma_D_j * sigma_V) / (D_i * V))**2 + 
                                            ((rho_V * sigma_V**2)/(V**2))**2)**(0.5)

    

    if no_model_error:
        C = C_t
    else:
        C = C_t + C_T

    P = np.linalg.inv(C) # N x M x S x S

    p = np.sum(P,axis=2)
    K = np.sum(P,axis=(2,3))

    t_ave = np.zeros((xlin.size,ylin.size,N))
    h_ave = np.zeros((xlin.size,ylin.size,N))

    for j, sta_code in enumerate(pick_sta):
        t_ave[:,:,j] = (p[:,:,j] * mu_P[j]) / K

        h_ave[:,:,j] = (p[:,:,j] * h.variables[sta_code].to_numpy()) / K
        
    t_ave = t_ave.sum(axis=2)
    h_ave = h_ave.sum(axis=2)

    t_tilde = np.zeros((xlin.size,ylin.size,N))
    h_tilde = np.zeros((xlin.size,ylin.size,N))

    for i, sta_code in enumerate(pick_sta):
        t_tilde[:,:,i] = mu_P[i] - t_ave
        h_tilde[:,:,i] = h.variables[sta_code].to_numpy() - h_ave

    resid = t_tilde - h_tilde
    res = np.squeeze(P @ resid[:,:,:,None]) # N x M x S
    misfit = np.squeeze(resid[:,:,None,:] @ res[:,:,:,None]) #resid is N x M x S
    gamma_arr = (N-2) * (misfit - misfit.min()) / misfit.min() #compute the contour function from the misfit.

    #convert gamma to a xarray dataarray...

    gamma = gamma.assign({'gamma':(('x','y'),gamma_arr), 'chi2':(('x','y'),misfit)})

    #work out t0 for the optimal solution and attach this to the dataset

    x_ind, y_ind = np.unravel_index(np.argmin(gamma_arr),gamma_arr.shape)
    x0 = xlin[x_ind]
    y0 = ylin[y_ind]    
    
    fit = h_tilde[x_ind,y_ind,:] + t_ave[x_ind,y_ind]

    h_fit = np.zeros(N)


    chi2_norm = (misfit / misfit.min()) * (N-2) 
    pdf = chi2.pdf(chi2_norm,N-2) #get the unnormalised pdf structure
    pdf /= pdf.sum()
    pdf_flat = pdf.flatten()


    for i, sta_code in enumerate(pick_sta):
        gamma.attrs['D_'+sta_code] = d.isel(x=x_ind,y=y_ind).variables[sta_code].values
        gamma.attrs['t_'+sta_code] = d.isel(x=x_ind,y=y_ind).variables[sta_code].values / V
        h_fit[i] = h.isel(x=x_ind,y=y_ind).variables[sta_code].values

        r_i = d.variables[sta_code]

        r_i_flat = r_i.to_numpy().flatten()

        ind_sort = np.argsort(r_i_flat)

        r_i_sort = r_i_flat[ind_sort]
        chi2_sort = pdf_flat[ind_sort]

        cdf_r = np.cumsum(chi2_sort)
        cdf_r /= cdf_r[-1]

        ci = r_i_sort[(cdf_r >= 0.025) & (cdf_r <= 0.975)]
        sigma = (ci[-1] - ci[0]) / 4
        gamma.attrs['dD_'+sta_code] = sigma


    t0 = np.mean(fit - h_fit) #all these values are actually the same (as they should be)

    gamma.attrs['t0'] = t0
    gamma.attrs['x0'] = x0
    gamma.attrs['y0'] = y0


    pdf_x = np.sum(pdf,axis=1)
    cdf_x = np.cumsum(pdf_x)
    ci_x = xlin[(cdf_x >= 0.025) & (cdf_x <= 0.975)]
    sigma_x = (ci_x[-1] - ci_x[0]) / 4

    pdf_y = np.sum(pdf,axis=0)
    cdf_y = np.cumsum(pdf_y)
    ci_y = ylin[(cdf_y >= 0.025) & (cdf_y <= 0.975)]
    sigma_y = (ci_y[-1] - ci_y[0]) / 4

    gamma.attrs['dx'] = sigma_x
    gamma.attrs['dy'] = sigma_y

    gamma.attrs['stations'] = pick_sta


    gamma = gamma.assign({'pdf':(('x','y'),pdf)})

    return gamma