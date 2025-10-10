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
    


"""
BACKMIGRATION FUNCTIONS - eventually want to make these part of a class for SeismicEvent (or StackedEvent) but worry about this later...
"""

        

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

def Rotate():
    #will have inv attached to the object so can do sta_xy
    pass

def Onset_Function(z,r,t,nsta,nlta,mode='radial_transverse'):

    if mode == 'radial_transverse':
        sta = np.cumsum(r**2)
        lta = np.cumsum(t**2)
    elif mode == 'radial':
        sta = np.cumsum(r**2)
        lta = np.cumsum(z**2 + r**2 + t**2)   
    elif mode == 'absolute':
        sta = np.cumsum(z**2 + r**2 + t**2)
        lta = np.cumsum(z**2 + r**2 + t**2)
    elif mode == 'transverse':
        sta = np.cumsum(t**2)
        lta = np.cumsum(z**2 + r**2 + t**2)
    elif mode == 'transverse_radial':
        sta = np.cumsum(t**2)
        lta = np.cumsum(r**2)
    elif mode == 'transverse_self':
        sta = np.cumsum(t**2)
        lta = np.cumsum(t**2)
    elif mode == 'radial_self':
        sta = np.cumsum(r**2)
        lta = np.cumsum(r**2)
    elif mode == 'P':
        sta = np.cumsum(r**2 + z**2)
        lta = np.cumsum(z**2 + r**2 * t**2)
    elif mode == 'S':
        sta = np.cumsum(t**2 + z**2)
        lta = np.cumsum(z**2 + r**2 * t**2)
    elif mode == None:
        ratio = np.zeros_like(z)
        return ratio #return zeros if switching off P or S coalescence
    else:
        sta = np.cumsum(z**2)
        lta = np.cumsum(z**2)


    ratio = np.zeros_like(sta)

    sta[:-nsta] = sta[nsta:] - sta[:-nsta]
    sta /= nsta
    sta[-nsta:] = 0

    #sta[nsta:] = sta[nsta:] - sta[:-nsta]
    #sta /= nsta
    #sta[:nsta - 1] = 0

    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta
    lta[:nlta - 1] = 0

    ratio[nlta:] = sta[nlta:] / lta[nlta:]

    #ratio = np.nan_to_num(ratio)
    return ratio


def CoalescenceSurface(x_grid,y_grid,z_grid,stream,inv,sta=1,lta=5,decimation=10,p_detect='radial_self',s_detect='transverse_self',modulate=False,normalise=True,mod_win=0.2,mod_overlap=0.9,g=5,smooth=10,Vp=3.84,Vs=1.90):

    t_grid = stream[0].times()[::decimation]
    df = stream[0].stats.sampling_rate


    Z_lst = []
    N_lst = []
    E_lst = []
    sta_lst = []

    for network in inv:
        for station in network:
            Z = stream.select(station=station.code,component='Z')[0].data
            N = stream.select(station=station.code,component='1')[0].data
            E = stream.select(station=station.code,component='2')[0].data

            if modulate:
                Z, N, E = RollingStreamModulation(Z,N,E,int(mod_win*df),g=g,smooth=smooth,overlap=mod_overlap)
              
            Z_lst.append(Z)
            N_lst.append(N)
            E_lst.append(E)
            sta_lst.append(station.code)

    Z = np.stack(Z_lst,axis=1)
    N = np.stack(N_lst,axis=1)
    E = np.stack(E_lst,axis=1)

    summed_PS_obj = np.zeros((x_grid.size,y_grid.size,z_grid.size,t_grid.size))
    sta_xy, centre = CartesianStations(inv)


    for i, e in enumerate(x_grid):
        for j, n in enumerate(y_grid):
            #compute the backazimuth and rotate the streams...
            diff_e = e - sta_xy[:,0]
            diff_n = n - sta_xy[:,1]

            baz = np.arctan2(diff_e,diff_n) #uses x/y rather than y/x due to way backaz is defined from North
        
            R = N * np.cos(baz[None,:]) + E * np.sin(baz[None,:])
            T = E * np.cos(baz[None,:]) - N * np.sin(baz[None,:])


            #now get the characteristic functions for the full traces...
            
            P_cfts = []
            S_cfts = []
            for ii in range(len(inv)):
              
                P_cfts.append(Onset_Function(Z[:,ii],R[:,ii],T[:,ii],int(sta*df),int(lta*df),mode=p_detect))
                S_cfts.append(Onset_Function(Z[:,ii],R[:,ii],T[:,ii],int(sta*df),int(lta*df),mode=s_detect))

            P_cft = np.stack(P_cfts,axis=1)
            S_cft = np.stack(S_cfts,axis=1)

            if normalise:
                #normalise the characteristic functions
                norm_P = (np.sum(P_cft,axis=0)[None,:] * 1/df)
                norm_S = (np.sum(S_cft,axis=0)[None,:] * 1/df)

                if (norm_P > 1e-10).any():
                    P_pdf = P_cft / norm_P
                else:
                    P_pdf = P_cft #close to zero, probably switched off S wave

                if (norm_S > 1e-10).any():
                    S_pdf = S_cft / norm_S
                else:
                    S_pdf = S_cft
            
            else:
                P_pdf = P_cft
                S_pdf = S_cft
            


            for k, z in enumerate(z_grid):
                diff_z = z - sta_xy[:,2]

                r = np.sqrt(diff_e**2 + diff_n**2 + diff_z**2)
                P_tt = r / Vp
                S_tt = r / Vs
                P_tt = P_tt
                S_tt = S_tt

                P_tt_samp = (df*P_tt).astype(int) #number of samples for travel time
                S_tt_samp = (df*S_tt).astype(int)  

                shifted_P_cfts = []
                shifted_S_cfts = []

                for ii in range(len(inv)):
                    shifted_P_cfts.append(np.pad(P_pdf[P_tt_samp[ii]:,ii],(0,P_tt_samp[ii])))
                    shifted_S_cfts.append(np.pad(S_pdf[S_tt_samp[ii]:,ii],(0,S_tt_samp[ii])))


                shift_P_cfts = np.stack(shifted_P_cfts,axis=1)
                shift_S_cfts = np.stack(shifted_S_cfts,axis=1)

                stacked_P = np.nansum(shift_P_cfts,axis=1)
                stacked_S = np.nansum(shift_S_cfts,axis=1)
                
                summed_PS_obj[i,j,k,:] = (stacked_P + stacked_S)[::decimation]

    return x_grid, y_grid, z_grid, t_grid, summed_PS_obj, sta_xy, centre



from scipy import ndimage

def ConnectedCoalescenceMask(coal_surface,thresh=0.5):
    p = np.squeeze(coal_surface)
    n_dim = len(p.shape) #number of dimensions for the coalescence surface
    max_coal = p.max()

    # MAP voxel
    imax = np.unravel_index(np.argmax(p), p.shape)

    threshold = thresh * max_coal #half maximum cutoff value

    # Initial mask (may include ties at the threshold)
    mask = p >= threshold

    structure = ndimage.generate_binary_structure(rank=n_dim, connectivity=1)
    labels, nlab = ndimage.label(mask, structure=structure)
    if nlab == 0:
        # Degenerate edge case: no voxels passed the threshold due to numerical weirdness
        comp = np.zeros_like(mask, dtype=bool)
    else:
        comp_label = labels[imax]
        comp = (labels == comp_label)

    mask = comp

    return mask

def interval_from_projection(proj_bool, coords):
    idx = np.flatnonzero(proj_bool)
    if idx.size == 0:
        return (np.nan, np.nan)
    return (coords[idx.min()], coords[idx.max()])

def wstats(coords, weights):
    s = weights.sum()
    if s <= 0:
        return (np.nan, np.nan)
    w = weights / s
    mu = (coords * w).sum()
    var = ((coords - mu)**2 * w).sum()
    return float(mu), float(np.sqrt(max(var, 0.0)))


def UncertaintyQuantBackM(x_grid,y_grid,z_grid,t_grid,inv,coalescence,contours=[0.75,0.5],time_slice=True):
    
    sta_xy, _ = CartesianStationsDF(inv)
    coord_lst = [x_grid,y_grid,z_grid,t_grid]

    if time_slice:
        x_ind, y_ind, z_ind, t_ind = np.unravel_index(np.argmax(coalescence),coalescence.shape)
        coal_surf = coalescence[:,:,:,t_ind]
    else:
        coal_surf = coalescence

    X, Y, Z = np.meshgrid(x_grid,y_grid,z_grid) #for computing the distance to each station

    dX = (sta_xy['X'].to_numpy()[:,None,None,None] - X[None,:,:,:])
    dY = (sta_xy['Y'].to_numpy()[:,None,None,None] - Y[None,:,:,:])
    dZ = (sta_xy['Z'].to_numpy()[:,None,None,None] - Z[None,:,:,:])

    dR = np.sqrt(dX**2 + dY**2 + dZ**2) #[station,X,Y,Z]

    r_fit = dR[:,x_ind,y_ind,z_ind] #get the distance to each station for the optimal location

    fit = (x_grid[x_ind],y_grid[y_ind],z_grid[z_ind],t_grid[t_ind],r_fit)

    centres = {}
    uncertainties = {}

    for thresh in contours:
        centres[thresh] = []
        uncertainties[thresh] = []

        mask = ConnectedCoalescenceMask(coal_surf,thresh=thresh)
        p_mask = np.where(mask, coal_surf, 0.0)


        #do the intervals, projections, and uncertainties on each coordinate from the masked coalescence.
        all_coords = set([i for i in range(len(mask.shape))])

        for coord in all_coords:
            coord_grid = coord_lst[coord]
            sum_coords = tuple(all_coords - {coord}) #drop the current coordinate for the summation axes

            marg = p_mask.sum(axis=sum_coords)

            mu, s = wstats(coord_grid, marg)

            centres[thresh].append(mu)
            uncertainties[thresh].append(s)     

        cR = []
        sR = []
        for i in range(dR.shape[0]): #loop through the stations
            staR = dR[i,:,:,:]

            mu, s = wstats(staR,p_mask) #get the centre and std for the radial distance at this station
            cR.append(mu)
            sR.append(s)
        centres[thresh].append(cR)
        uncertainties[thresh].append(sR)

    return fit, uncertainties, centres


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



def StreamModulation(Z, N, E, window_len, g=4, smooth=5):
    """
    Apply polarisation-based modulation to 3-component seismic data.
    
    Parameters
    ----------
    Z, N, E : np.ndarray
        1D arrays with the three component seismic data.
    window_len : int
        Length of time-domain window in samples.
    g : float
        Exponent for polarisation weighting.
    smooth : int
        Half-width of frequency smoothing in bins (default 5).
    
    Returns
    -------
    mod_Z, mod_N, mod_E : np.ndarray
        Modulated Z, N, E arrays.
    """

    total_length = Z.size
    mod_Z = np.zeros_like(Z)
    mod_Zn = np.zeros_like(Z)
    mod_Ze = np.zeros_like(Z)
    mod_N = np.zeros_like(N)
    mod_E = np.zeros_like(E)

    window_start = 0
    window_end = window_start + window_len

    while window_end <= total_length:
        # Slice windowed time series
        window_Z = Z[window_start:window_end]
        window_N = N[window_start:window_end]
        window_E = E[window_start:window_end]

        # FFT
        Zf = rfft(window_Z)
        Nf = rfft(window_N)
        Ef = rfft(window_E)

        # Shape (n_freq, 3)
        M = np.stack((Zf, Nf, Ef), axis=1)

        n_freq = M.shape[0]
        P = np.zeros(n_freq)
        vN = np.zeros(n_freq)
        vE = np.zeros(n_freq)

        # For each frequency bin, build cross-spectral matrix averaged over neighbours
        for f in range(n_freq):
            lo = max(0, f - smooth)
            hi = min(n_freq, f + smooth + 1)
            band = M[lo:hi]  # (bandwidth, 3)

            # Cross-spectral density matrix (3x3)
            S = np.einsum("bi,bj->ij", band, band.conj()) / band.shape[0]

            try:
                eig_vals, eig_vecs = eigh(S)
                v1 = np.abs(eig_vecs[:,-1]) #principle eigenvector magnitude
                v1_proj = v1[1:] #drop the vertical component
                v1_proj /= np.sqrt(np.sum(v1_proj**2)) #make sure it is normalised
                vN[f] = v1_proj[0]
                vE[f] = v1_proj[1]
                P[f] = eig_vals[-1] / eig_vals.sum() #TODO can we do something with the eigenvector by supressing the other directions? Take each component of normalised principle eigenvalue and have directional dependent P
            except LinAlgError:
                P[f] = np.nan #missing data, fill out with nans
                vN[f] = np.nan
                vE[f] = np.nan
            

        # Apply polarisation weighting
        Zf_mod = Zf * (P**g)
        Znf_mod = Zf * (P**g) * vN
        Zef_mod = Zf * (P**g) * vE
        Nf_mod = Nf * (P**g)
        Ef_mod = Ef * (P**g)

        # IFFT back to time domain
        mod_Z[window_start:window_end] = irfft(Zf_mod, n=window_len)
        mod_Zn[window_start:window_end] = irfft(Znf_mod, n=window_len)
        mod_Ze[window_start:window_end] = irfft(Zef_mod, n=window_len)
        mod_N[window_start:window_end] = irfft(Nf_mod, n=window_len)
        mod_E[window_start:window_end] = irfft(Ef_mod, n=window_len)

        # Advance to next window
        window_start += window_len
        window_end += window_len

    return mod_Z, mod_Zn, mod_Ze, mod_N, mod_E



def RollingStreamModulation(Z, N, E, window_len, g=4, smooth=5, overlap=0.5, taper="hann"):
    """
    Apply polarisation-based modulation to 3-component seismic data.

    Parameters
    ----------
    Z, N, E : np.ndarray
        1D arrays with the three component seismic data.
    window_len : int
        Length of time-domain window in samples.
    g : float
        Exponent for polarisation weighting.
    smooth : int
        Half-width of frequency smoothing in bins (default 5).
    overlap : float
        Fraction of overlap between windows (0–1). Default 0.5 (50%).
    taper : str or None
        Window function to apply before FFT. Options: 'hann', 'hamming', None.

    Returns
    -------
    mod_Z, mod_N, mod_E : np.ndarray
        Modulated Z, N, E arrays.
    """

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