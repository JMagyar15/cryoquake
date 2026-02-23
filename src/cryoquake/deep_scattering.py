import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from obspy.core import UTCDateTime, inventory

from scatseisnet import ScatteringNetwork

from cryoquake.data_objects import SeismicChunk
from matplotlib import dates as mdates

from sklearn.decomposition import FastICA, PCA, NMF
from sklearn.cluster import AgglomerativeClustering, Birch
import umap

from matplotlib.dates import DateFormatter, DayLocator, HourLocator
import copy
import xarray as xr

import fastcluster as fc



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

        self.N_times = self.timestamps.size
        self.N_channels = self.spectra.shape[1]
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

    def reduce_dimension(self,n_components,max_iter=1000,sqrt=False,norm=False):

        X = self.__unpack_xarray(self.scattering_coefficients)

        X = self.__adjustX(X,sqrt,norm,False) #can't do log transformation with NMF

        model = NMF(n_components,max_iter=max_iter) #gives the option of testing exp function for approximating neg-entropy
        S = model.fit_transform(X)

        self.S = S #as the output of the dimension reduction, this has no nan values and is smaller than the original data. It can continue to be shortened with pruning.
        self.H = model.components_ #mixing matrix (kind of, some stuff about whittening in the documentation)...

        #to make the xarray, will need to add back in the nans
        S_full = np.full((self.N_times * self.N_channels,n_components),dtype=np.float32,fill_value=np.nan)
        S_full[self.ind,:] = S.copy()

        self.S_full = S_full #this is the same dimension as the original spectra, and has nan values where we have dropped windows.
        self.S_xr = xr.DataArray(self.S_full.reshape((self.N_times,self.N_channels,-1)),coords={'t':self.spectra.t,'channels':self.spectra.channels,'i':np.arange(n_components)})

        delattr(self,"scattering_coefficients")


    def compute_linkage(self,method='ward',metric='euclidean'):        
        self.U = fc.linkage_vector(self.S,method=method,metric=metric)
    
    
    def save_linkage(self,path,name):
        filename = os.path.join(path,self.str_name + '__' + name) #include full chunk extent in name as this step cannot be broken into daychunks...
        np.save(filename,self.U)



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

            filename = os.path.join(path,name + '__' + daychunk.str_name)

            daychunk.context('scattering')
            daychunk.load_spectra(path,name=spec_name)

            timestamps = daychunk.scattering_coefficients.t
            channels = daychunk.scattering_coefficients.channels

            delattr(daychunk,"scattering_coefficients")

            S = self.S_xr.sel(t=timestamps,channels=channels)
            S.to_netcdf(filename)


    def load_dim_reduction(self,path,name='scattering_features'):

        day_features = []
        for daychunk in self:
            filename = os.path.join(path,name + '__' + daychunk.str_name)
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
        
        #TODO save the components to file and reload them here...
        #self.H = model.components_ #mixing matrix (kind of, some stuff about whittening in the documentation)...

    

    def __transform(self,segments,reduce):
        scattering_coefficients = []

        for j in range(len(self.channel_ids)):
            channel_segments = [segments[i,j,:] for i in range(len(self.timestamps))]
            channel_bool = [np.isnan(seg).any() for seg in channel_segments]
            print(self.channel_ids[j],np.sum(np.array(channel_bool)))
            spectra = self.network.transform(channel_segments,reduce_type=reduce)[-1] #just take the last layer for now...
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
    

    def __adjustX(self,X,sqrt,norm,log):

        if norm:
            X = X / np.sum(X,axis=1)[:,None] #normalise the columns to get unit sum...
        if sqrt:
            X = np.sqrt(X)
        if log:
            X = np.log10(X)

        return X
    

    #TODO move dimension reduction to here - I think it makes much more sense wrt data structures...


#!probably move all of these functions to the above object, it's a bit messy with loading/saving to have two...
#?or at least have the linkage in the above, and then feed the linkage to a new object for interpretation...

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


    def compute_centroids(self,cosine=False):
        centroids = {}
        dist = {}
        centroid_specs = {}

        for name in self.cluster_names:

            ind = (self.full_labels==name)
            locs = self.S_full[ind,:]

            if cosine:
                #for now make centroid the mean of the normalised spectra? The magnitude of this is meaningless, but only used for getting cosine distance where direction matters...
                locs_norm = np.linalg.norm(locs,axis=1)
                locs_unit = locs / locs_norm[:,None]  #M x n
                centroids[name] = np.mean(locs_unit,axis=0) #1 x n
                cent_norm = np.linalg.norm(centroids[name]) #float
                uv = locs @ centroids[name].T #M vector
                
                dist[name] = 1 - (uv / (locs_norm * cent_norm)) #1 - (u.v / ||u||||v||)

            else:
                centroids[name] = np.mean(locs,axis=0)
                dist[name] = np.sqrt(np.sum((centroids[name][None,:] - locs)**2,axis=1)) #(1 x 10) - (N x 10)

            centroid_specs[name] = centroids[name][None,:] @ self.H
        
        self.centroids = centroids
        self.centroid_specs = centroid_specs
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