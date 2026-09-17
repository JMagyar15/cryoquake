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
from sklearn.mixture import GaussianMixture

import xarray as xr

import fastcluster as fc
from sknetwork.hierarchy import cut_balanced, cut_straight

from scipy.stats.mstats import gmean
from scipy.linalg import helmert


class ScatteringSpectrum(SeismicChunk):
    
    def __call__(self,window_length=10,overlap=1.0,reduce=np.max):

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
            freqs.append(bank.centers[-bank.size:])
            layer_N.append(bank.size)
        
        self.freqs = freqs
        self.layer_N = layer_N


    def dim_reduction(self,n_components,p=1,log1=False,alpha_H=0.0,alpha_W=0.0):
        """
        base this off current ica and transfer relevant bits to nmf for pre
        """

        X = self.__unpack_xarray(self.scattering_coefficients)


        X_mod = self.__preprocessX(np.copy(X),p=p,log=False,log1=log1)
        
        X_sum = np.sum(X_mod,axis=1)
        X_mod /= X_sum[:,None] #rescale to unit sum

        self.X_len = X_sum

        scale = StandardScaler(with_mean=False) #need to keep nonnegative for nmf. #! can also try with robust scalar...
        X_mod = scale.fit_transform(X_mod) #rescale the components...

        self.sdX = np.sqrt(scale.var_)
        self.X = X_mod

        model = NMF(n_components,max_iter=10000,alpha_H=alpha_H,alpha_W=alpha_W,l1_ratio=0.0,tol=1e-4,random_state=42) #can encourage dense solutions for clustering with regularisation

        S = model.fit_transform(X_mod)
        
        H = model.components_ 

        self.S = S #as the output of the dimension reduction, this has no nan values and is smaller than the original data. It can continue to be shortened with pruning.
        self.H = H #mixing matrix (kind of, some stuff about whittening in the documentation)...
        #to make the xarray, will need to add back in the nans
        S_full = np.full((self.N_times * self.N_channels,n_components),dtype=np.float32,fill_value=np.nan)
        
        S_full[self.ind,:] = S

        self.S_full = S_full #this is the same dimension as the original spectra, and has nan values where we have dropped windows.
        self.S_xr = xr.DataArray(self.S_full.reshape((self.N_times,self.N_channels,-1)),coords={'t':self.scattering_coefficients.t,'channels':self.scattering_coefficients.channels,'i':np.arange(n_components)})
        delattr(self,"scattering_coefficients")


    def embed_hellinger(self):
        H = self.H
        S = self.S

        H_s = H * self.sdX[None,:]

        H_sum = np.sum(H_s,axis=1,keepdims=True)

        #H = H / H_norm[:,None]
        H_s = H_s / H_sum #make unit length basis functions
        S_s = S * H_sum.T #readjust S to account for rescaling

        self.H_s = H_s
        self.S_s = S_s

        G = H_s @ H_s.T #for defining the metric #! long term it might be worth considering applying this with the sensitivity applied so distance is more important for sensitive elements of the spectrum.

        L = np.linalg.cholesky(G) #decomposition of this metric for the embedding in euclidean space

        self.S_re = S_s / np.sum(S_s,axis=1,keepdims=True) #enforce closure of composition

        S_emb = np.sqrt(self.S_re) @ L
        S_t = S_emb #/ np.linalg.norm(S_emb,axis=1,keepdims=True) #!NOT fully decided on this, could go either way...

        self.S_emb = S_emb
        self.S_t = S_t
        self.L = L
        self.G = G

   

    # def embed_unit(self):
    #     H = self.H
    #     S = self.S

    #     H_norm = np.linalg.norm(H,axis=1,keepdims=True)

    #     #H = H / H_norm[:,None]
    #     H_s = H / H_norm #make unit length basis functions
    #     S_s = S * H_norm.T #readjust S to account for rescaling

    #     self.H_s = H_s
    #     self.S_s = S_s

    #     G = H_s @ H_s.T #for defining the metric #! long term it might be worth considering applying this with the sensitivity applied so distance is more important for sensitive elements of the spectrum.

    #     L = np.linalg.cholesky(G) #decomposition of this metric for the embedding in euclidean space


    #     S_emb = S_s @ L
    #     S_t = S_emb / np.linalg.norm(S_emb,axis=1,keepdims=True) #!NOT fully decided on this, could go either way...

    #     self.S_emb = S_emb
    #     self.S_t = S_t
    #     self.L = L
    #     self.G = G


    # def embed_nmf(self):
    #     H = self.H

    #     H_s = H * self.sdX[None,:]

    #     #H_norm = np.linalg.norm(H,axis=1)
    #     H_sum = np.sum(H_s,axis=1,keepdims=True)

    #     #H = H / H_norm[:,None]
    #     H_s /= H_sum

    #     S = self.S

    #     S = S * H_sum.T #readjust S to account

    #     G = H_s @ H_s.T #for defining the metric #! long term it might be worth considering applying this with the sensitivity applied so distance is more important for sensitive elements of the spectrum.

    #     L = np.linalg.cholesky(G) #decomposition of this metric for the embedding in euclidean space

    #     S_norm = np.sum(S,axis=1,keepdims=True)

    #     S_re = S / S_norm #ensure this closure which was approximate

    #     S_t = S @ L

    #     self.S_re = S_re
    #     self.S_t = S_t
    #     self.L = L


    # def embed_nmf_mod(self):
    #     H = self.H

    #     H_s = H * self.sdX[None,:]

    #     H_norm = np.linalg.norm(H_s,axis=1)
    #     #H_sum = np.sum(H_s,axis=1,keepdims=True)

    #     H_s = H_s / H_norm[:,None]
    #     #H_s /= H_sum

    #     S = self.S

    #     S_s = S * H_norm.T #readjust S to account

    #     #now reapply the sensitivity to get metric
    #     #H_sens = H_s / self.sdX[None,:]

    #     G = H_s @ H_s.T #for defining the metric #! long term it might be worth considering applying this with the sensitivity applied so distance is more important for sensitive elements of the spectrum.

    #     L = np.linalg.cholesky(G) #decomposition of this metric for the embedding in euclidean space

    #     #S_norm = np.sum(S,axis=1,keepdims=True)

    #     #S_re = S / S_norm #ensure this closure which was approximate

    #     S_emb = S_s @ L
    #     S_t = S_emb / np.linalg.norm(S_emb,axis=1,keepdims=True)

    #     self.H_s = H_s
    #     self.S_s = S_s
    #     self.S_emb = S_emb
    #     self.S_t = S_t
    #     self.L = L
    #     self.G = G


    def compute_linkage(self,method='centroid',metric='euclidean',alpha=1.0):

        X = self.S_t * (self.X_len[:,None]**alpha)# / np.std(self.S_re,axis=0,keepdims=True)
        self.X = X
        self.U = fc.linkage_vector(X,method=method,metric=metric)


    def amp_linkage(self,method='ward',metric='euclidean'):

        H = self.H

        H_s = H * self.sdX[None,:]

        #H_norm = np.linalg.norm(H,axis=1)
        H_sum = np.sum(H_s,axis=1)

        #H = H / H_norm[:,None]
        H_s /= H_sum[:,None]

        S = self.S
        #S = S * H_norm[None,:]
        S *= H_sum[None,:]

        S_norm = np.sum(S,axis=1,keepdims=True)

        self.S_re = S / S_norm
        self.H_re = H_s

        X = self.S_re * self.X_sum[:,None] / np.std(self.S_re,axis=0,keepdims=True)

        self.U = fc.linkage_vector(X,method=method,metric=metric)


    def amp_linkage_norm(self,method='centroid',metric='euclidean',alpha=1.0):
        X = self.S_t * (self.X_norm[:,None]**alpha)# / np.std(self.S_re,axis=0,keepdims=True)

        self.U = fc.linkage_vector(X,method=method,metric=metric)
    

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
    
    
    def __preprocessX(self,X,p=1,log=True,log1=False):
        """
        To be used before finding the basis functions from NMF.
        """
        X = X**(1/p) #take the power first

        if log:
            #X = np.log1p(X) #maps zero to zero, but compresses the higher ampltudes more that p=2...
            X = np.log10(X) #from neg inf to inf but accounts for small noise fluctuations?
        
        elif log1:
            X = np.log1p(X)
        
        return X
    

    
class LinkageClustering:
    """
    Separate class for dealing with the linkage matrix, which is the key output from the scattering spectra calculations...
    Want to be able to chose number of clusters and properties (e.g. min cluster size) and then convert these results back into the timestamps/channels...
    """
    def __init__(self,chunk,alpha):
        X = chunk.S_t * (chunk.X_len[:,None]**alpha)
        self.L = chunk.L
        self.X = X
        self.X_len = chunk.X_len
        self.H = chunk.H
        self.sdX = chunk.sdX
        self.U = chunk.U
        self.ind = chunk.ind
        self.timestamps = chunk.timestamps
        self.channel_ids = chunk.channel_ids
        self.ind = chunk.ind

        self.N_times = chunk.N_times
        self.N_channels = chunk.N_channels



    def prune_cluster(self,n_clusters=10,min_size=5):

        distances = self.U[::-1,2] #get distances of merges in reverse order to iterate through
        pruning = True

        i = n_clusters - 2 #minimum number of iterations it could take...

        self.labels = np.full(self.N_times*self.N_channels,fill_value=-1,dtype=np.int64)

        while pruning:
            labels = cut_straight(self.U,n_clusters=None,threshold=distances[i],return_dendrogram=False)
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


        return self.labels

        

    def compute_centroids(self):

        #! need to get the amplitude information in here through chunk.X_sum
        centroids = {}
        dist = {}

        #scale = RobustScaler(with_centering=False)
        #X = scale.fit_transform(self.X)

        for name in self.cluster_names:

            ind = (self.labels==name)[self.ind]
            locs = self.X[ind,:]

           
            centroids[name] = np.mean(locs,axis=0)
            dist[name] = np.sqrt(np.sum((centroids[name][None,:] - locs)**2,axis=1)) #(1 x 10) - (N x 10)

        
        self.centroids = centroids
        self.dist = dist

    def hellinger_centroids(self):
        
        centroids_emb = {}
        centroids = {}
        dist = {}

        #scale = RobustScaler(with_centering=False)
        #X = scale.fit_transform(self.X)

        for name in self.cluster_names:

            ind = (self.labels==name)[self.ind]
            locs = self.X[ind,:]

            centroid_emb = np.mean(locs,axis=0) #centroid in the embedded space
            centroids_emb[name] = centroid_emb

            centroid = (np.linalg.solve(self.L.T,centroid_emb)) #centroids in feature space.
            centroids[name] = centroid
           
            dist[name] = np.sqrt(np.sum((centroid_emb[None,:] - locs)**2,axis=1)) #(1 x 10) - (N x 10)
        
        self.centroids = centroids
        self.centroids_emb = centroids_emb
        self.dist = dist


class DirectClustering:
    def __init__(self,chunk):
        self.X = chunk.X
        self.S = chunk.S
        self.X_sum = chunk.X_sum
        self.H = chunk.H
        self.sdX = chunk.sdX

        self.ind = chunk.ind
        self.timestamps = chunk.timestamps
        self.channel_ids = chunk.channel_ids

        self.N_times = chunk.N_times
        self.N_channels = chunk.N_channels

        self.str_name = chunk.str_name

        H = self.H

        H_s = H * self.sdX[None,:]

        #H_norm = np.linalg.norm(H,axis=1)
        H_sum = np.sum(H_s,axis=1)

        #H = H / H_norm[:,None]
        H_s /= H_sum[:,None]

        S = self.S
        #S = S * H_norm[None,:]
        S *= H_sum[None,:]

        S_norm = np.sum(S,axis=1,keepdims=True)

        self.S_re = S / S_norm
        self.H_re = H_s


    def direct_cluster(self):

        labels = np.argmax(self.S_re,axis=1)        
        self.labels = np.full(self.N_times*self.N_channels,fill_value=-1,dtype=np.int64)
        self.labels[self.ind] = labels + 1

        self.cluster_names = np.arange(1,self.H.shape[0]+1)

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
    
    def amp_linkage(self,method='ward',metric='euclidean'):

        X = self.S_re * self.X_sum[:,None] / np.std(self.S_re,axis=0,keepdims=True)

        self.U = fc.linkage_vector(X,method=method,metric=metric)


    def save_linkage(self,path,name):
        filename = os.path.join(path,self.str_name + '__' + name) #include full chunk extent in name as this step cannot be broken into daychunks...
        np.save(filename,self.U)


    def compute_linkage(self,method='ward',metric='euclidean',pca_comp=3,eps=1e-6,alpha=0.0):
        H_smooth = (self.S_re + eps)
        H_smooth /= H_smooth.sum(axis=1, keepdims=True)
        logH = np.log(H_smooth)
        mean_log = logH.mean(axis=1, keepdims=True)
        clr =  logH - mean_log + alpha*np.log(self.X_sum)[:,None]
        self.clr = clr

        #try PCA on the clr to get rid of correlations and sparsity problems?
        pca = PCA(n_components=pca_comp,random_state=0,whiten=True).fit(clr)

        clr_low = pca.transform(clr)

        self.clr_low = clr_low

        self.U = fc.linkage_vector(clr_low,method=method,metric=metric)

    def kmeans(self,n_clusters=10,weighted=True,alpha=1.0):

        X = self.S_re * self.X_sum[:,None]# / np.std(self.S_re,axis=0,keepdims=True)


        if weighted:
            model = KMeans(n_clusters=n_clusters,random_state=0).fit(X,sample_weight=self.X_sum**alpha)
        else:
            model = KMeans(n_clusters=n_clusters,random_state=0).fit(X)
        #model = GaussianMixture(n_components=n_clusters,random_state=0).fit(clr)
        labels = model.labels_    
        #labels = model.predict(clr)

        self.labels = np.full(self.N_times*self.N_channels,fill_value=-1,dtype=np.int64)
        self.labels[self.ind] = labels + 1

        self.cluster_names = np.arange(1,n_clusters+1)
        self.centroids = model.cluster_centers_
        #self.centroids = model.means_

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
    
    def gmm(self,n_clusters=10,weighted=True,alpha=1.0):

        X = self.S_re * self.X_sum[:,None]# / np.std(self.S_re,axis=0,keepdims=True)


        if weighted:
            model = GaussianMixture(n_components=n_clusters,random_state=0).fit(X,sample_weight=self.X_sum**alpha)
        else:
            model = GaussianMixture(n_components=n_clusters,random_state=0).fit(X)
        #model = GaussianMixture(n_components=n_clusters,random_state=0).fit(clr)
        #labels = model.labels_    
        labels = model.predict(X)

        self.labels = np.full(self.N_times*self.N_channels,fill_value=-1,dtype=np.int64)
        self.labels[self.ind] = labels + 1

        self.cluster_names = np.arange(1,n_clusters+1)
        #self.centroids = model.cluster_centers_
        self.centroids = model.means_

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


    def log_cluster(self,n_clusters=10,pca_comp=3,eps=1e-6,alpha=1.0):

        H_smooth = (self.S_re + eps)
        H_smooth /= H_smooth.sum(axis=1, keepdims=True)
        H_smooth = H_smooth ** alpha
        H_smooth /= np.sum(H_smooth,axis=1,keepdims=True)
        logH = np.log(H_smooth)
        mean_log = logH.mean(axis=1, keepdims=True)
        clr =  logH - mean_log


        self.clr = clr

        #try PCA on the clr to get rid of correlations and sparsity problems?
        pca = PCA(n_components=pca_comp,random_state=0).fit(clr)
        #pca = SparsePCA(n_components=pca_comp,random_state=0).fit(clr)

        clr_low = pca.transform(clr)

        self.clr_low = clr_low


        #now do K-means on these ratios...
        model = KMeans(n_clusters=n_clusters,random_state=0).fit(clr_low)
        #model = GaussianMixture(n_components=n_clusters,random_state=0).fit(clr)
        labels = model.labels_    
        #labels = model.predict(clr)

        self.labels = np.full(self.N_times*self.N_channels,fill_value=-1,dtype=np.int64)
        self.labels[self.ind] = labels + 1

        self.cluster_names = np.arange(1,n_clusters+1)
        self.centroids_low = model.cluster_centers_
        self.centroids = pca.inverse_transform(model.cluster_centers_)
        #self.centroids = model.means_

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


    def centroid_dist(self):
        dist = {}

        #scale = RobustScaler(with_centering=False)
        #X = scale.fit_transform(self.X)

        X = self.S_re * self.X_sum[:,None]

        for name in self.cluster_names:

            ind = (self.labels==name)[self.ind]
            locs = X[ind,:]

            centroid = self.centroids[name-1,:] #this is the component

            dist[name] = np.linalg.norm(locs - centroid[None,:],axis=1)

        self.dist = dist

    def alpha_dist(self):
        dist = {}

        #scale = RobustScaler(with_centering=False)
        #X = scale.fit_transform(self.X)

        for name in self.cluster_names:

            ind = (self.labels==name)[self.ind]
            locs = self.S_a[ind,:]

            centroid = self.centroids[name-1,:] #this is the component

            dist[name] = np.linalg.norm(locs - centroid[None,:],axis=1)

        self.dist = dist


    def hellinger_dist(self):
        dist = {}

        #scale = RobustScaler(with_centering=False)
        #X = scale.fit_transform(self.X)

        for name in self.cluster_names:

            ind = (self.labels==name)[self.ind]
            locs = self.clr[ind,:]

            centroid = self.centroids[name-1,:] #this is the component

           
            dist[name] = np.linalg.norm(locs - centroid[None,:],axis=1) / np.sqrt(2)

        self.dist = dist

    

    def spectral_dist(self):
        dist = {}

        #scale = RobustScaler(with_centering=False)
        #X = scale.fit_transform(self.X)

        for name in self.cluster_names:

            ind = (self.labels==name)[self.ind]
            locs = self.X[ind,:]

            H = self.H[name-1,:] #this is the component

           
            dist[name] = np.linalg.norm(locs - H[None,:],axis=1)

        self.dist = dist


    def compute_dist(self):
        dist = {}

        #scale = RobustScaler(with_centering=False)
        #X = scale.fit_transform(self.X)

        for name in self.cluster_names:

            ind = (self.labels==name)[self.ind]
            locs = self.S_re[ind,:]

           
            dist[name] = 1 - locs[:,name-1] #(1 x 10) - (N x 10)

        
        self.dist = dist


    def prune_cluster(self,n_clusters=10,min_size=5):

        distances = self.U[::-1,2] #get distances of merges in reverse order to iterate through
        pruning = True

        i = n_clusters - 2 #minimum number of iterations it could take...

        self.labels = np.full(self.N_times*self.N_channels,fill_value=-1,dtype=np.int64)

        while pruning:
            labels = cut_straight(self.U,n_clusters=None,threshold=distances[i],return_dendrogram=False)
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


        return self.labels
    
    def compute_centroids(self):
        centroids = {}
        dist = {}

        X = self.S_re * self.X_sum[:,None]

        #scale = RobustScaler(with_centering=False)
        #X = scale.fit_transform(self.X)

        for name in self.cluster_names:

            ind = (self.labels==name)[self.ind]
            locs = X[ind,:]

           
            centroids[name] = np.median(locs,axis=0)
            dist[name] = np.sqrt(np.sum((centroids[name][None,:] - locs)**2,axis=1)) #(1 x 10) - (N x 10)

        
        self.centroids = centroids
        self.dist = dist

    

class ModDirectClustering:
    def __init__(self,chunk,std=True):
        self.X = chunk.X
        self.S = chunk.S
        self.X_sum = chunk.X_sum
        self.H = chunk.H
        self.sdX = chunk.sdX

        self.ind = chunk.ind
        self.timestamps = chunk.timestamps
        self.channel_ids = chunk.channel_ids

        self.N_times = chunk.N_times
        self.N_channels = chunk.N_channels

        H = self.H

        #H_norm = np.linalg.norm(H,axis=1)
        H_sum = np.sum(H,axis=1)

        #H = H / H_norm[:,None]
        H /= H_sum[:,None]

        S = self.S
        #S = S * H_norm[None,:]
        S *= H_sum[None,:]

        S_norm = np.sum(S,axis=1,keepdims=True)

        self.S_re = S / S_norm
        self.H_re = H


    def direct_cluster(self):

        labels = np.argmax(self.S_re,axis=1)        
        self.labels = np.full(self.N_times*self.N_channels,fill_value=-1,dtype=np.int64)
        self.labels[self.ind] = labels + 1

        self.cluster_names = np.arange(1,self.H.shape[0]+1)

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
    

    def spectral_dist(self):
        dist = {}

        #scale = RobustScaler(with_centering=False)
        #X = scale.fit_transform(self.X)

        for name in self.cluster_names:

            ind = (self.labels==name)[self.ind]
            locs = self.X[ind,:]

            H = self.H[name-1,:] #this is the component

           
            dist[name] = np.linalg.norm(locs - H[None,:],axis=1)

        self.dist = dist


    def compute_dist(self):
        dist = {}

        #scale = RobustScaler(with_centering=False)
        #X = scale.fit_transform(self.X)

        for name in self.cluster_names:

            ind = (self.labels==name)[self.ind]
            locs = self.S_re[ind,:]

           
            dist[name] = 1 - locs[:,name-1] #(1 x 10) - (N x 10)

        
        self.dist = dist
