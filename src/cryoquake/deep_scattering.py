import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from obspy.core import UTCDateTime, inventory

from scatseisnet import ScatteringNetwork

from cryoquake.data_objects import SeismicChunk
from matplotlib import dates as mdates

from sklearn.decomposition import FastICA, PCA, NMF, SparsePCA
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn.preprocessing import StandardScaler, normalize, RobustScaler
import umap

from matplotlib.dates import DateFormatter, DayLocator, HourLocator
import copy
import xarray as xr

import fastcluster as fc
from sknetwork.hierarchy import cut_balanced, cut_straight

from scipy.stats.mstats import gmean
from scipy.linalg import helmert

#from spherecluster import SphericalKMeans



class ScatteringSpectrum(SeismicChunk):
    
    def __call__(self,window_length=10,overlap=1.0,reduce=np.max):

        #segments, timestamps, trace_ids = self.__segment_stream(window_length,overlap)

        segments, timestamps, channel_ids = self.__slide_stream(window_length,overlap)

        self.channel_ids = channel_ids
        self.timestamps = timestamps

        scattering = self.__transform(segments,reduce)

        del segments

        dss = self.__make_xarray(scattering)

        del scattering

        self.scattering_coefficients = dss

        self.N_times = len(self.timestamps)
        self.N_channels = len(self.channel_ids)
        self.ind = np.full(self.N_times*self.N_channels,fill_value=True,dtype=bool) #initially want to consider all windows...

    
    def set_network(self,network):
        self.network = network

        freqs = []
        layer_N = []
        for bank in self.network.banks:
            freqs.append(bank.centers)
            layer_N.append(bank.centers.size)
        
        self.freqs = freqs
        self.layer_N = layer_N

    def dim_reduction(self,n_components,method='ica',p=1,log=True,log1=False):
        """
        base this off current ica and transfer relevant bits to nmf for pre
        """

        X = self.__unpack_xarray(self.scattering_coefficients)

        #TODO add a raise statement here for if one tries to use log scaling and nmf...

        X_mod = self.__preprocessX(np.copy(X),p=p,log=log,log1=log1,norm=True)

        if method == 'nmf':
            #scale = StandardScaler(with_mean=False) #need to keep nonnegative for nmf.
            scale = RobustScaler(with_centering=False)
        else:
            #scale = StandardScaler()
            scale = RobustScaler(with_centering=True)

        X_mod = scale.fit_transform(X_mod) #rescale the components...

        if method == 'pca':
            model = PCA(n_components,random_state=0,whiten=True)
        elif method == 'nmf':
            model = NMF(n_components)
        else:
            model = FastICA(n_components,random_state=0) #gives the option of testing exp function for approximating neg-entropy

        model.fit(X_mod)

        X_fit = self.__preprocessX(np.copy(X),p=p,log=log,log1=log1,norm=False)
        X_fit = scale.transform(X_fit) #apply the same rescaling of the components as with the fitting...

        S = model.transform(X_fit)

        #robust_scale = RobustScaler()
        S = scale.fit_transform(S) #now rescale the attributes back to a reasonable range...
        H = model.components_ 

        self.S = S #as the output of the dimension reduction, this has no nan values and is smaller than the original data. It can continue to be shortened with pruning.
        self.H = H #mixing matrix (kind of, some stuff about whittening in the documentation)...
        #to make the xarray, will need to add back in the nans
        S_full = np.full((self.N_times * self.N_channels,n_components),dtype=np.float32,fill_value=np.nan)
        
        S_full[self.ind,:] = S

        self.S_full = S_full #this is the same dimension as the original spectra, and has nan values where we have dropped windows.
        self.S_xr = xr.DataArray(self.S_full.reshape((self.N_times,self.N_channels,-1)),coords={'t':self.scattering_coefficients.t,'channels':self.scattering_coefficients.channels,'i':np.arange(n_components)})
        delattr(self,"scattering_coefficients")

    def nmf(self,n_components,max_iter=1000,p=1,balance=True,beta_loss=1,sens=False):
        #TODO get this in the same form as ICA and PCA - could eventually put all in one function with a switch argument...?
        X = self.__unpack_xarray(self.scattering_coefficients)

        X_mod, sdX = self.__preprocessX(np.copy(X),p=p,norm=True,sens=sens)
        self.sdX = sdX
        X_fit = self.__fitX(np.copy(X),p=p,sens=sens)


        model = NMF(n_components,max_iter=max_iter,beta_loss=beta_loss,solver='mu',alpha_W=0.1,alpha_H=0.0) #gives the option of testing exp function for approximating neg-entropy

        model.fit(X_mod)
        S = model.transform(X_fit)

        #Sf = model.transform(X_fit)
        
        H = model.components_ #* self.sdX[None,:] #multiply by variance to get clearer physical meaning / the difference between the basis functions as they are all multiplied by sensitivity.

        if balance:
            comp_sum = np.sum(H,axis=1)
            #comp_mag = np.linalg.norm(H,axis=1)
            H = H / comp_sum[:,None]
            S = S * comp_sum[None,:]
            #Sf = Sf * comp_sum[None,:]
         

        self.S = S #as the output of the dimension reduction, this has no nan values and is smaller than the original data. It can continue to be shortened with pruning.
        #self.Sf = Sf
        self.H = H #mixing matrix (kind of, some stuff about whittening in the documentation)...
        #to make the xarray, will need to add back in the nans
        S_full = np.full((self.N_times * self.N_channels,n_components),dtype=np.float32,fill_value=np.nan)
        #Sf_full = np.full((self.N_times * self.N_channels,n_components),dtype=np.float32,fill_value=np.nan)
        
        S_full[self.ind,:] = S
        #Sf_full[self.ind,:] = Sf

        self.S_full = S_full #this is the same dimension as the original spectra, and has nan values where we have dropped windows.
        #self.Sf_full = Sf_full
        self.S_xr = xr.DataArray(self.S_full.reshape((self.N_times,self.N_channels,-1)),coords={'t':self.scattering_coefficients.t,'channels':self.scattering_coefficients.channels,'i':np.arange(n_components)})
        #self.Sf_xr = xr.DataArray(self.Sf_full.reshape((self.N_times,self.N_channels,-1)),coords={'t':self.scattering_coefficients.t,'channels':self.scattering_coefficients.channels,'i':np.arange(n_components)})
        delattr(self,"scattering_coefficients")

    def pca(self,n_components,p=1,log=True,log1=False):
        X = self.__unpack_xarray(self.scattering_coefficients)

        X_mod = self.__preprocessX(np.copy(X),p=p,log=log,log1=log1,norm=True)

        scale = StandardScaler()
        #scale = RobustScaler()
        X_mod = scale.fit_transform(X_mod) #rescale the components...

        model = PCA(n_components,random_state=0,whiten=True) #gives the option of testing exp function for approximating neg-entropy
        model.fit(X_mod)

        X_fit = self.__preprocessX(np.copy(X),p=p,log=log,log1=log1,norm=False)
        X_fit = scale.transform(X_fit) #apply the same rescaling of the components as with the fitting...

        S = model.transform(X_fit)

        robust_scale = RobustScaler()
        S = robust_scale.fit_transform(S) #now rescale the attriubtes back to a reasonable range...
        H = model.components_ 

        self.S = S #as the output of the dimension reduction, this has no nan values and is smaller than the original data. It can continue to be shortened with pruning.
        #self.Sf = Sf
        self.H = H #mixing matrix (kind of, some stuff about whittening in the documentation)...
        #to make the xarray, will need to add back in the nans
        S_full = np.full((self.N_times * self.N_channels,n_components),dtype=np.float32,fill_value=np.nan)
        #Sf_full = np.full((self.N_times * self.N_channels,n_components),dtype=np.float32,fill_value=np.nan)
        
        S_full[self.ind,:] = S
        #Sf_full[self.ind,:] = Sf

        self.S_full = S_full #this is the same dimension as the original spectra, and has nan values where we have dropped windows.
        #self.Sf_full = Sf_full
        self.S_xr = xr.DataArray(self.S_full.reshape((self.N_times,self.N_channels,-1)),coords={'t':self.scattering_coefficients.t,'channels':self.scattering_coefficients.channels,'i':np.arange(n_components)})
        #self.Sf_xr = xr.DataArray(self.Sf_full.reshape((self.N_times,self.N_channels,-1)),coords={'t':self.scattering_coefficients.t,'channels':self.scattering_coefficients.channels,'i':np.arange(n_components)})
        delattr(self,"scattering_coefficients")

    def ica(self,n_components,max_iter=1000,p=1,log=True,log1=False):
        X = self.__unpack_xarray(self.scattering_coefficients)

        X_mod = self.__preprocessX(np.copy(X),p=p,log=log,log1=log1,norm=True)

        scale = StandardScaler()
        #scale = RobustScaler()
        X_mod = scale.fit_transform(X_mod) #rescale the components...

        model = FastICA(n_components,max_iter=max_iter,random_state=0) #gives the option of testing exp function for approximating neg-entropy
        model.fit(X_mod)

        X_fit = self.__preprocessX(np.copy(X),p=p,log=log,log1=log1,norm=False)
        X_fit = scale.transform(X_fit) #apply the same rescaling of the components as with the fitting...

        S = model.transform(X_fit)

        robust_scale = RobustScaler()
        S = robust_scale.fit_transform(S) #now rescale the attriubtes back to a reasonable range...
        H = model.components_ 

        self.S = S #as the output of the dimension reduction, this has no nan values and is smaller than the original data. It can continue to be shortened with pruning.
        #self.Sf = Sf
        self.H = H #mixing matrix (kind of, some stuff about whittening in the documentation)...
        self.whiten = model.whitening_
        #to make the xarray, will need to add back in the nans
        S_full = np.full((self.N_times * self.N_channels,n_components),dtype=np.float32,fill_value=np.nan)
        #Sf_full = np.full((self.N_times * self.N_channels,n_components),dtype=np.float32,fill_value=np.nan)
        
        S_full[self.ind,:] = S
        #Sf_full[self.ind,:] = Sf

        self.S_full = S_full #this is the same dimension as the original spectra, and has nan values where we have dropped windows.
        #self.Sf_full = Sf_full
        self.S_xr = xr.DataArray(self.S_full.reshape((self.N_times,self.N_channels,-1)),coords={'t':self.scattering_coefficients.t,'channels':self.scattering_coefficients.channels,'i':np.arange(n_components)})
        #self.Sf_xr = xr.DataArray(self.Sf_full.reshape((self.N_times,self.N_channels,-1)),coords={'t':self.scattering_coefficients.t,'channels':self.scattering_coefficients.channels,'i':np.arange(n_components)})
        delattr(self,"scattering_coefficients")

    def composite_model(self,k=-5):

        n_components = self.S.shape[1]
        S_mean = np.mean(self.S,axis=0)
        eps = (10**k) * S_mean
        Sp = self.S + eps[None,:]

        S_norm = Sp / np.sum(Sp,axis=1)[:,None]

        S_mean = gmean(S_norm,axis=0)
        clr = np.log10(S_norm / S_mean)#[None,:])

        self.clr = clr
        clr_full = np.full((self.N_times * self.N_channels,n_components),dtype=np.float32,fill_value=np.nan)
        clr_full[self.ind] = clr
        self.clr_full = clr_full

        rotation = helmert(n_components,full=False)
        ilr = (clr @ rotation.T)

        self.ilr = ilr
        
        ilr_full = np.full((self.N_times * self.N_channels,n_components-1),dtype=np.float32,fill_value=np.nan)
        ilr_full[self.ind] = ilr

        self.ilr_full = ilr_full

    def log_transform(self,k=-5):

        S = self.S
        n_components = self.S.shape[1]
        S_mean = np.mean(self.S,axis=0)
        eps = (10**k) * S_mean
        Sp = self.S + eps[None,:]


        S_mean = gmean(Sp,axis=0)
        logS = np.log10(Sp / S_mean)#[None,:])

        self.logS = logS
        logS_full = np.full((self.N_times * self.N_channels,n_components),dtype=np.float32,fill_value=np.nan)
        logS_full[self.ind] = logS
        self.logS_full = logS_full

    def normalise_transform(self):
        self.S = normalize(self.S, norm="l2")


    def compute_linkage(self,target='S',method='ward',metric='euclidean'):        
        if target == 'logS':
            X = self.logS
        elif target == 'clr':
            X = self.clr
        else:
            X = self.S

        self.U = fc.linkage_vector(X,method=method,metric=metric)
        self.X = X
    

    def save_linkage(self,path,name):
        filename = os.path.join(path,self.str_name + '__' + name) #include full chunk extent in name as this step cannot be broken into daychunks...
        np.save(filename,self.U)

    def load_linkage(self,path,name):
        filename = os.path.join(path,self.str_name + '__' + name + '.npy') #include full chunk extent in name as this step cannot be broken into daychunks...
        self.U = np.load(filename)



    def save_spectra(self,path,name='scattering_spectra'):
        #use the chunk start/end times to make a file name
        filename = os.path.join(path,name + '__' + self.str_name)
        self.scattering_coefficients.to_netcdf(filename)


    def load_spectra(self,path,name='scattering_spectra'):
        #reverse process of above, just so we can then use flatten output to get back to the array form...
        #load in the scattering coefficients
        day_spectra = []
        for daychunk in self:
            filename = os.path.join(path,name + '__' + daychunk.str_name)
            if os.path.exists(filename):
                day = xr.load_dataarray(filename)
                day_spectra.append(day)
        
        dss = xr.concat(day_spectra,dim='t')

        self.scattering_coefficients = dss

        self.timestamps = [t for t in dss.t.values]
        self.channel_ids = [channel_id for channel_id in dss.channels.values]

        self.N_times = len(self.timestamps)
        self.N_channels = len(self.channel_ids)
        self.ind = np.full(self.N_times*self.N_channels,fill_value=True,dtype=bool) #initially want to consider all windows...


    def save_dim_reduction(self,path,name='scattering_features',spec_name='scattering_spectra'):

        for daychunk in self:

            filename = os.path.join(path,name + '__features__' + daychunk.str_name)

            daychunk.context('scattering')
            daychunk.load_spectra(path,name=spec_name)

            timestamps = daychunk.scattering_coefficients.t
            channels = daychunk.scattering_coefficients.channels

            delattr(daychunk,"scattering_coefficients")

            S = self.S_xr.sel(t=timestamps,channels=channels)
            S.to_netcdf(filename)

        #now also save the components...
        filename = os.path.join(path,name + '__components')
        np.save(filename,self.H)

        filename = os.path.join(path,name + '__dev')
        np.save(filename,self.sdX)




    def load_dim_reduction(self,path,name='scattering_features'):

        day_features = []
        for daychunk in self:
            filename = os.path.join(path,name + '__features__' + daychunk.str_name)
            if os.path.exists(filename):
                day = xr.load_dataarray(filename)
                day_features.append(day)

        
        self.S_xr = xr.concat(day_features,dim='t')

        self.timestamps = [t for t in self.S_xr.t.values]
        self.channel_ids = [channel_id for channel_id in self.S_xr.channels.values]

        self.N_times = len(self.timestamps)
        self.N_channels = len(self.channel_ids)

        self.S_full = self.S_xr.values.reshape((self.N_times*self.N_channels,-1))
        
        drop_ind = np.isnan(self.S_full).any(axis=1)

        self.ind = np.full(self.N_times*self.N_channels,True,dtype=bool)
        self.ind[drop_ind] = False

        self.S = self.S_full[self.ind,:] #as the output of the dimension reduction, this has no nan values and is smaller than the original data. It can continue to be shortened with pruning.
        
        self.H = np.load(os.path.join(path,name + '__components.npy'))
        self.sdX = np.load(os.path.join(path,name + '__dev.npy'))
        #self.H = model.components_ #mixing matrix (kind of, some stuff about whittening in the documentation)...


    def __transform(self,segments,reduce):
        scattering_coefficients = []

        for j in range(len(self.channel_ids)):
            channel_segments = [segments[i,j,:] for i in range(len(self.timestamps))]
            channel_bool = [np.isnan(seg).any() for seg in channel_segments] #TODO make the second layer sum to the previous layer for consistency of energy / amplitude...
            print(self.channel_ids[j],np.sum(np.array(channel_bool)))
            all_layers = self.network.transform(channel_segments,reduce_type=reduce) #list of the layers of the scattering spectra...

            layer1 = all_layers[0] #has shape N_times, f1
            norm = layer1.sum(axis=-1)

            for layer in all_layers: #layer has shape N_times, f1, f2,...,fn where n is number of layers in the network
                end_sum = layer.sum(axis=-1) #sum over the new frequency axis
                fact = norm / end_sum #these should be the same shape as have summed over the additional axis...
                layer *= fact[...,None]
                norm = layer

            spectra = layer #just take the last layer... #self.network.transform(channel_segments,reduce_type=reduce)[-1] #just take the last layer for now...
            spectra[channel_bool,...] = np.nan
            scattering_coefficients.append(spectra) 
        
        scattering_coefficients = np.stack(scattering_coefficients,axis=1)

        return scattering_coefficients
    
    def __slide_stream(self,window_length,overlap):
        timestamps = []
        segments = []
        channel_ids = []

        for tr in self.stream:
            channel_ids.append(tr.id)

        stream = self.stream.trim(self.starttime,self.endtime)

        for segment in stream.slide(window_length,window_length * overlap):
            #want to make a boolean array of length channels to then set these rows to nan for missing data...
            keep_station = np.array([np.ma.is_masked(tr.data[:-1]) for tr in segment])

            timestamps.append(segment[0].stats.starttime.datetime)
            data = np.stack([tr.data[:-1] for tr in segment],axis=0)
            data[keep_station,:] = np.nan
            segments.append(data) #each entry to this list has shape channels x window_len
        
        segments = np.stack(segments,axis=0) #should have shape time x channels x window_len

        return segments, timestamps, channel_ids
    

    def __make_xarray(self,scattering):

        freq_dict = {}
        for i, freq in enumerate(self.freqs):
            freq_dict['f' + str(i)] = freq
        coord_dict = {'t':self.timestamps,'channels':self.channel_ids} | freq_dict
        dss = xr.DataArray(scattering,coords=coord_dict)

        return dss
    
    def __unpack_xarray(self,spectra):
        #take the xarray dss and convert it to a matrix form without nans that can be used for clustering
        #fill need to keep track of indices which can then be ravelled back to the xarray coordinates...
        spec_arr = spectra.values

        #now flatten along frequency and station axes.
        spec_flat = spec_arr.reshape(self.N_times*self.N_channels,-1) #so just keep the windows as the first dimension

        drop_ind = np.isnan(spec_flat).any(axis=1)
        self.ind[drop_ind] = False #get rid of the locations with missing data

        X = spec_flat[self.ind,...]

        return X
    
    
    def __preprocessX(self,X,p=1,log=True,log1=False,norm=False):
        """
        To be used before finding the basis functions from NMF.
        """
        X = X**(1/p) #take the power first
        
        if norm:
            X = X / np.linalg.norm(X,axis=1)[:,None] #normalise each spectrum

        if log:
            #X = np.log1p(X) #maps zero to zero, but compresses the higher ampltudes more that p=2...
            X = np.log10(X) #from neg inf to inf but accounts for small noise fluctuations?
        
        elif log1:
            X = np.log1p(X)
        
        return X
    
    def __fitX(self,X,p=1,log=True,log1=False,norm=False):
        """
        Same a preprocessing but uses the precomputed standard deviation, and does not normalise the spectra at the end.
        """
        X = X**(1/p) #take the power first

        return X


    def __g_tanh(self,x, alpha=1.0):
        gx = np.tanh(alpha * x)
        g_x = alpha * (1 - np.tanh(alpha * x)**2).mean(axis=1)
        return gx, g_x

    


class SpectralClustering:
    def __init__(self,spectra):#,timestamps,trace_ids):
        self.spectra = spectra
        self.timestamps = spectra.t.to_numpy()
        self.channels = spectra.channels.to_numpy()

        self.N_times = self.timestamps.size
        self.N_channels = self.spectra.shape[1]
        self.ind = np.full(self.N_times*self.N_channels,fill_value=True,dtype=bool) #initially want to consider all windows...


    def reduce_dim(self,n_components,fun='logcosh',fun_args=None,max_iter=200,sqrt=False,norm=False,log=False):

        X = self.__unpack_xarray(self.spectra)

        X = self.__adjustX(X,sqrt,norm,log)

        model = FastICA(n_components,fun=fun,fun_args=fun_args,max_iter=max_iter) #gives the option of testing exp function for approximating neg-entropy
        S = model.fit_transform(X)

        self.S = S #as the output of the dimension reduction, this has no nan values and is smaller than the original data. It can continue to be shortened with pruning.
        self.H = model.components_ #mixing matrix (kind of, some stuff about whittening in the documentation)...

        self.trained_ica = model

        #to make the xarray, will need to add back in the nans
        S_full = np.full((self.N_times * self.N_channels,n_components),dtype=np.float32,fill_value=np.nan)
        S_full[self.ind,:] = S.copy()

        self.S_full = S_full #this is the same dimension as the original spectra, and has nan values where we have dropped windows.
        self.S_xr = xr.DataArray(self.S_full.reshape((self.N_times,self.N_channels,-1)),coords={'t':self.spectra.t,'channels':self.spectra.channels,'i':np.arange(n_components)})

    def nnmf_dim(self,n_components,max_iter=200,sqrt=False,norm=False):

        X = self.__unpack_xarray(self.spectra)

        X = self.__adjustX(X,sqrt,norm,False) #can't do log transformation with NMF

        model = NMF(n_components,max_iter=max_iter) #gives the option of testing exp function for approximating neg-entropy
        S = model.fit_transform(X)

        self.S = S #as the output of the dimension reduction, this has no nan values and is smaller than the original data. It can continue to be shortened with pruning.
        self.H = model.components_ #mixing matrix (kind of, some stuff about whittening in the documentation)...

        self.trained_nmf = model

        #to make the xarray, will need to add back in the nans
        S_full = np.full((self.N_times * self.N_channels,n_components),dtype=np.float32,fill_value=np.nan)
        S_full[self.ind,:] = S.copy()

        self.S_full = S_full #this is the same dimension as the original spectra, and has nan values where we have dropped windows.
        self.S_xr = xr.DataArray(self.S_full.reshape((self.N_times,self.N_channels,-1)),coords={'t':self.spectra.t,'channels':self.spectra.channels,'i':np.arange(n_components)})


    def prune_clustering(self,n_clusters=10,min_size=10,threshold=1,branching_factor=1000,linkage='ward',metric='euclidean'):
        #take the above result and remove windows from sington clusters, and then rerun without these so they don't take up a whole cluster...
        #can motivate this by saying that these sections are either glitches, or so rare that we can't look at any trends, want at least sample size of ~10 to work with...
        
        pruning = True
        while pruning:
            print('Pruning...')
            self.__birch_agglom_cluster(n_clusters=n_clusters,threshold=threshold,branching_factor=branching_factor,linkage=linkage,metric=metric)
            pruning = self.__prune(min_size=min_size)

        group_times = {}
        group_channels = {}

        #when doing this, also want to make a link back to the timestamps and channels names of the original xarray for further analysis of windows...
        for name in self.cluster_names:
    
            i_ind = np.argwhere(self.full_labels==name).flatten()

            i, j = np.unravel_index(i_ind,(self.N_times,self.N_channels))

            group_times[name] = self.timestamps[i]
            group_channels[name] = self.channels[j]

        self.group_times = group_times
        self.group_channels = group_channels


    def compute_centroids(self,target='S'):
        centroids = {}
        dist = {}

        if target == 'logS':
            X = self.logS_full
        elif target == 'clr':
            X = self.clr_full
        else:
            X = self.S_full

        for name in self.cluster_names:

            ind = (self.full_labels==name)
            locs = X[ind,:]

           
            centroids[name] = np.mean(locs,axis=0)
            dist[name] = np.sqrt(np.sum((centroids[name][None,:] - locs)**2,axis=1)) #(1 x 10) - (N x 10)

        
        self.centroids = centroids
        self.dist = dist


    def __birch_agglom_cluster(self,n_clusters=10,threshold=1,branching_factor=1000,linkage='ward',metric='euclidean'):
        sub_model = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage,metric=metric)
        birch = Birch(n_clusters=sub_model, threshold=threshold,branching_factor=branching_factor).fit(self.S.copy()) #! issue here - need higher threshold in some cases, otherwise too many subclusters for Agglomerative. Try using proportion of IQR to capture scale...?
        self.labels = birch.labels_ + 1
        self.cluster_names = np.unique(self.labels)
        self.n_clusters = self.cluster_names.size

        self.full_labels = np.zeros(self.N_times * self.N_channels,dtype=int)
        self.full_labels[self.ind] = self.labels #fill the points where there was data with the appropriate cluster name...

    
    def __prune(self,min_size=3):
        pruning = True
        unique, counts = np.unique(self.labels, return_counts=True)
        if (counts >= min_size).all():
            pruning = False
        else:
            keep_labels = unique[counts >= min_size]

            #now find the indices of the windows we want to keep...
            drop_ind = ~np.isin(self.full_labels, keep_labels)

            self.ind[drop_ind] = False

            self.S = self.S_full[self.ind,:]

        return pruning


    def __unpack_xarray(self,spectra):
        #take the xarray dss and convert it to a matrix form without nans that can be used for clustering
        #fill need to keep track of indices which can then be ravelled back to the xarray coordinates...
        spec_arr = spectra.values

        #now flatten along frequency and station axes.
        spec_flat = spec_arr.reshape(self.N_times*self.N_channels,-1) #so just keep the windows as the first dimension

        drop_ind = np.isnan(spec_flat).any(axis=1)
        self.ind[drop_ind] = False #get rid of the locations with missing data

        X = spec_flat[self.ind,...]

        return X
    

    def __adjustX(self,X,sqrt,norm,log):

        if norm:
            X = X / np.sum(X,axis=1)[:,None] #normalise the columns to get unit sum...
        if sqrt:
            X = np.sqrt(X)
        if log:
            X = np.log10(X)

        return X


class LinkageClustering:
    """
    Separate class for dealing with the linkage matrix, which is the key output from the scattering spectra calculations...
    Want to be able to chose number of clusters and properties (e.g. min cluster size) and then convert these results back into the timestamps/channels...
    """
    def __init__(self,chunk):
        self.X = chunk.X
        self.U = chunk.U
        self.ind = chunk.ind
        self.timestamps = chunk.timestamps
        self.channel_ids = chunk.channel_ids
        self.ind = chunk.ind

        self.N_times = chunk.N_times
        self.N_channels = chunk.N_channels

    def k_means_composite(self,n_clusters):
        S = self.ilr

        labels = KMeans(n_clusters=n_clusters).fit_predict(S)
        self.labels = np.full(self.N_times*self.N_channels,fill_value=-1,dtype=np.int64)
        self.labels[self.ind] = labels + 1

        self.cluster_names = np.arange(1,n_clusters+1)

        group_times = {}
        group_channels = {}

        #when doing this, also want to make a link back to the timestamps and channels names of the original xarray for further analysis of windows...
        for name in self.cluster_names:
    
            i_ind = np.argwhere(self.labels==name).flatten()

            i, j = np.unravel_index(i_ind,(self.N_times,self.N_channels))


            group_times[name] = np.array(self.timestamps)[i]
            group_channels[name] = np.array(self.channel_ids)[j]

        self.group_times = group_times
        self.group_channels = group_channels

        return self.labels


    def k_means(self,n_clusters=10,unit_sum=True):
        S = self.S[self.ind]

        if unit_sum:
            S /= np.sum(S,axis=1)[:,None]
        labels = KMeans(n_clusters=n_clusters).fit_predict(S)
        self.labels = np.full(self.N_times*self.N_channels,fill_value=-1,dtype=np.int64)
        self.labels[self.ind] = labels + 1

        self.cluster_names = np.arange(1,n_clusters+1)

        group_times = {}
        group_channels = {}

        #when doing this, also want to make a link back to the timestamps and channels names of the original xarray for further analysis of windows...
        for name in self.cluster_names:
    
            i_ind = np.argwhere(self.labels==name).flatten()

            i, j = np.unravel_index(i_ind,(self.N_times,self.N_channels))


            group_times[name] = np.array(self.timestamps)[i]
            group_channels[name] = np.array(self.channel_ids)[j]

        self.group_times = group_times
        self.group_channels = group_channels

        return self.labels

    def prune_cluster(self,n_clusters=10,min_size=5):

        distances = self.U[::-1,2] #get distances of merges in reverse order to iterate through
        pruning = True

        i = n_clusters - 2 #minimum number of iterations it could take...

        self.labels = np.full(self.N_times*self.N_channels,fill_value=-1,dtype=np.int64)

        while pruning:
            labels, dendrogram = cut_straight(self.U,n_clusters=None,threshold=distances[i],return_dendrogram=True)
            unique, counts = np.unique(labels, return_counts=True)

            print(counts)

            large_clusters = unique[(counts >= min_size)]

            drop_ind = ~np.isin(labels, large_clusters)
            labels[drop_ind] = -1 #will become cluster zero later...
    
            current_clusters = large_clusters.size

            if (current_clusters >= n_clusters):
                pruning = False

            self.labels[self.ind] = labels + 1

            i += 1

        unique, counts = np.unique(self.labels, return_counts=True)

        self.cluster_names = np.arange(1,n_clusters+1)

        group_times = {}
        group_channels = {}

        #when doing this, also want to make a link back to the timestamps and channels names of the original xarray for further analysis of windows...
        for name in self.cluster_names:
    
            i_ind = np.argwhere(self.labels==name).flatten()

            i, j = np.unravel_index(i_ind,(self.N_times,self.N_channels))


            group_times[name] = np.array(self.timestamps)[i]
            group_channels[name] = np.array(self.channel_ids)[j]

        self.group_times = group_times
        self.group_channels = group_channels

        self.dendrogram = dendrogram

        return self.labels, self.dendrogram

    def balanced_cluster(self,max_size):
        self.labels = np.full(self.N_times*self.N_channels,fill_value=-1,dtype=np.int64)

        labels, dendrogram = cut_balanced(self.U,max_cluster_size=max_size,return_dendrogram=True)

        self.labels[self.ind] = labels + 1

        unique, counts = np.unique(labels, return_counts=True)
        n_clusters = unique.size
        self.cluster_names = np.arange(1,n_clusters+1)

        group_times = {}
        group_channels = {}

        #when doing this, also want to make a link back to the timestamps and channels names of the original xarray for further analysis of windows...
        for name in self.cluster_names:
    
            i_ind = np.argwhere(self.labels==name).flatten()

            i, j = np.unravel_index(i_ind,(self.N_times,self.N_channels))


            group_times[name] = np.array(self.timestamps)[i]
            group_channels[name] = np.array(self.channel_ids)[j]

        self.group_times = group_times
        self.group_channels = group_channels

        self.dendrogram = dendrogram

        return self.labels, self.dendrogram
        

    def compute_centroids(self,target='S'):
        centroids = {}
        dist = {}

        for name in self.cluster_names:

            ind = (self.labels==name)[self.ind]
            locs = self.X[ind,:]

           
            centroids[name] = np.mean(locs,axis=0)
            dist[name] = np.sqrt(np.sum((centroids[name][None,:] - locs)**2,axis=1)) #(1 x 10) - (N x 10)

        
        self.centroids = centroids
        self.dist = dist

