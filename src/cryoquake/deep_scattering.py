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
import umap

from matplotlib.dates import DateFormatter, DayLocator, HourLocator
import copy
import xarray as xr

import fastcluster as fc
from sknetwork.hierarchy import cut_balanced, cut_straight



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

    def nmf(self,n_components,max_iter=1000,p=1,balance=True):
        X = self.__unpack_xarray(self.scattering_coefficients)

        X_mod, var = self.__preprocessX(np.copy(X),p=p)
        self.var = var

        model = NMF(n_components,solver='mu',max_iter=max_iter,beta_loss=1) #gives the option of testing exp function for approximating neg-entropy

        model.fit(X_mod) #! this is doing fit_transform in one go so would be clustering the normalised spectra rather than the unnormalised where a separation would need to be made...

        X_fit = self.__fitX(np.copy(X),p=p)
        S = model.transform(X_fit)
        #X = self.__adjustX(X,sqrt,norm)
        #S = model.transform(X)
        
        H = model.components_ * self.var[None,:] #multiply by variance to get clearer physical meaning / the difference between the basis functions as they are all multiplied by sensitivity.

        if balance:
            comp_sum = np.sum(H,axis=1)
            H = H / comp_sum[:,None]
            S = S * comp_sum[None,:]
         

        self.S = S #as the output of the dimension reduction, this has no nan values and is smaller than the original data. It can continue to be shortened with pruning.
        self.H = H #mixing matrix (kind of, some stuff about whittening in the documentation)...
        #to make the xarray, will need to add back in the nans
        S_full = np.full((self.N_times * self.N_channels,n_components),dtype=np.float32,fill_value=np.nan)
        S_full[self.ind,:] = S.copy()

        self.S_full = S_full #this is the same dimension as the original spectra, and has nan values where we have dropped windows.
        self.S_xr = xr.DataArray(self.S_full.reshape((self.N_times,self.N_channels,-1)),coords={'t':self.scattering_coefficients.t,'channels':self.scattering_coefficients.channels,'i':np.arange(n_components)})

        delattr(self,"scattering_coefficients")



    def reduce_dimension(self,n_components,method='NMF',max_iter=1000,fit_norm=True,sqrt=False,norm=False,balance=True):

        X = self.__unpack_xarray(self.scattering_coefficients)

        X_mod = self.__adjustX(np.copy(X),sqrt,fit_norm)

        if method == 'NMF':
            model = NMF(n_components,max_iter=max_iter,solver='mu',init='random',beta_loss=1) #gives the option of testing exp function for approximating neg-entropy
        else:
            model = FastICA(n_components,max_iter=max_iter)

        model.fit(X_mod)

        X = self.__adjustX(X,sqrt,norm)
        S = model.transform(X)
        H = model.components_

        if balance:
            if method == 'NMF':
                #comp_length = np.linalg.norm(H,axis=1) #get length of each component
                comp_sum = np.sum(H,axis=1)
                H = H / comp_sum[:,None]
                S = S * comp_sum[None,:]
            else:
                comp_length = np.linalg.norm(H,axis=1) #get length of each component
                #comp_sum = np.sum(H,axis=1)
                H = H / comp_length[:,None]
                S = S * comp_length[None,:]

        self.S = S #as the output of the dimension reduction, this has no nan values and is smaller than the original data. It can continue to be shortened with pruning.
        self.H = H #mixing matrix (kind of, some stuff about whittening in the documentation)...

        #to make the xarray, will need to add back in the nans
        S_full = np.full((self.N_times * self.N_channels,n_components),dtype=np.float32,fill_value=np.nan)
        S_full[self.ind,:] = S.copy()

        self.S_full = S_full #this is the same dimension as the original spectra, and has nan values where we have dropped windows.
        self.S_xr = xr.DataArray(self.S_full.reshape((self.N_times,self.N_channels,-1)),coords={'t':self.scattering_coefficients.t,'channels':self.scattering_coefficients.channels,'i':np.arange(n_components)})

        delattr(self,"scattering_coefficients")


    def compute_linkage(self,method='ward',metric='euclidean',norm=False):        
        S = self.S
        if norm:
            S /= np.sum(S,axis=1)[:,None]
        self.U = fc.linkage_vector(S,method=method,metric=metric)
        #self.U = fc.linkage(self.S,method=method,metric=metric)
    

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

        filename = os.path.join(path,name + '__variance')
        np.save(filename,self.var)




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
        self.var = np.load(os.path.join(path,name + '__variance.npy'))
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
    
    def __normaliseX(self,X):
        mod_X = X / np.sum(X,axis=1)[:,None]
        return mod_X
    
    def __powerX(self,X,p=2):
        mod_X = X**(1/p)
        return mod_X
    
    def __preprocessX(self,X,p=1,norm=True):
        """
        To be used before finding the basis functions from NMF.
        """
        X = X**(1/p) #take the power first
        if norm:
            X = X / np.sum(X,axis=1)[:,None] #normalise each spectrum
        var = np.var(X,axis=0)
        X = X / var[None,:] #rescale by variance of each scattering element

        if norm:
            X = X / np.sum(X,axis=1)[:,None] #renormalise again
        return X, var
    
    def __fitX(self,X,p=1):
        X = X**(1/p) #take the power first
        X = X / self.var[None,:] #rescale by variance of each scattering element
        return X
    

    def __adjustX(self,X,sqrt,norm):
        #TODO make these separate functions as I want to make sure it is always done in the right order, but make it for general p...
        if sqrt:
            X = np.sqrt(X)
        
        if norm:
            X = X / np.sum(X,axis=1)[:,None] #normalise the columns to get unit sum...

        return X
    


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
    


class LinkageClustering:
    """
    Separate class for dealing with the linkage matrix, which is the key output from the scattering spectra calculations...
    Want to be able to chose number of clusters and properties (e.g. min cluster size) and then convert these results back into the timestamps/channels...
    """
    def __init__(self,chunk):
        self.U = chunk.U
        self.S = chunk.S_full
        self.H = chunk.H
        self.ind = chunk.ind
        self.timestamps = chunk.timestamps
        self.channel_ids = chunk.channel_ids
        self.ind = chunk.ind

        self.N_times = chunk.N_times
        self.N_channels = chunk.N_channels


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

        i = n_clusters - 1 #minimum number of iterations it could take...

        self.labels = np.full(self.N_times*self.N_channels,fill_value=-1,dtype=np.int64)

        while pruning:
            labels, dendrogram = cut_straight(self.U,n_clusters=None,threshold=distances[i],return_dendrogram=True)
            unique, counts = np.unique(labels, return_counts=True)

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
        

    

    def compute_centroids(self,cosine=False):
        centroids = {}
        dist = {}
        centroid_specs = {}

        for name in self.cluster_names:

            ind = (self.labels==name)
            locs = self.S[ind,:]

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