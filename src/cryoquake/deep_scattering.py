import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from obspy.core import UTCDateTime, inventory

from scatseisnet import ScatteringNetwork

from cryoquake.data_objects import SeismicChunk
from matplotlib import dates as mdates

from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import AgglomerativeClustering, Birch
import umap

from matplotlib.dates import DateFormatter, DayLocator, HourLocator
import copy



class ScatteringSpectrum(SeismicChunk):

    #! will probably do all of this inside the __iter__ of SeismicChunk, so just get the spectra for the section, leave clustering etc. for other functions/script...
    #! don't worry about nans in this object, that is a job for ICA and clustering to deal with...
    
    def __call__(self,window_length=10,overlap=1.0,reduce=np.max):

        segments, timestamps, trace_ids = self.__segment_stream(window_length,overlap)

        self.scattering_coefficients = self.__transform(segments,reduce)

        del segments

        self.timestamps = timestamps
        self.trace_ids = trace_ids

    
    def set_network(self,network):
        self.network = network

    def save_spectra(self,path):
        #use the chunk start/end times to make a file name

        #use xarray to save each layer, the frequencies and times will be the coordinates...
        #make a separate one for each channel, so they can easily be separated out when loading back in,.
        pass

    def load_spectra(self,path):
        #reverse process of above, just so we can then use flatten output to get back to the array form...
        pass

    def flatten_output(self,log=False,normalise=False,sqrt=False):

        spectra = {}

        for key, scattering in self.scattering_coefficients.items():
            scattering_coefficients = copy.deepcopy(scattering)
            N_layers = len(scattering_coefficients)

            top_layer = scattering_coefficients[0]
            deepest_layer = scattering_coefficients[-1]
            N_times, *N_filters = deepest_layer.shape


            #orders = []
            layers = []
            norm = 1 / np.sum(top_layer,axis=-1)

            for i in range(N_layers):
                #deal with each layer separately, then concatenate them...
                order_i = scattering_coefficients[i]

                if normalise:
                    if i == 0:
                        order_i *= norm[...,None]
                    else:
                        order_i *= (norm[...,None] / np.sum(order_i,axis=-1)[...,None])
                    norm = order_i

                    
                order_i_rs = order_i.reshape(N_times,-1) #collapse coefficients in frequencies
                #orders.append(order_i_rs)

                if sqrt:
                    order_i_rs = np.sqrt(order_i_rs)

                if log:
                    order_i_rs = np.log10(order_i_rs)

                layers.append(order_i_rs.reshape(N_times,order_i_rs.shape[-1]))
            
            spectra[key] = layers
            
        return spectra



    def __segment_stream(self,window_length,overlap):
        """
        take the attached stream from the seismic chunk and make the segments for processing for each trace.
        Just a helper function for the __call__ of the object.
        """
        timestamps = {}
        segments = {}
        trace_ids = {}

        for trace in self.stream: #loop over the stations:
            timestamps[trace.id] = []
            segments[trace.id] = []
            trace_ids[trace.id] = []
            for window in trace.slide(window_length, window_length * overlap):
                timestamps[trace.id].append(mdates.num2date(window.times(type="matplotlib")[0]))
                segments[trace.id].append(window.data[:-1])
                trace_ids[trace.id].append(trace.id)

        
            timestamps[trace.id] = np.array(timestamps[trace.id])
            trace_ids[trace.id] = np.array(trace_ids[trace.id])
        return segments, timestamps, trace_ids
    
    def __transform(self,segments,reduce):
        scattering_coefficients = {}
        for key, trace_segments in segments.items():
            scattering_coefficients[key] = self.network.transform(trace_segments,reduce_type=reduce)

        return scattering_coefficients



class SpectralClustering:
    def __init__(self,spectra,timestamps,trace_ids):
        self.spectra = spectra
        self.timestamps = timestamps
        self.trace_ids = trace_ids



    def train_ica(self,n_components,fun='logcosh',fun_args=None):
        # make this a training step so it lessens the need to track which stations / times things come from because can then
        #transform individual stations/times into the dimension-reduced space. Less efficient but works better with data structures.
        X, t, ids = self.__construct_matrix(self.spectra,self.timestamps,self.trace_ids)

        model = FastICA(n_components,fun=fun,fun_args=fun_args) #gives the option of testing exp function for approximating neg-entropy
        S = model.fit_transform(X)

        self.S = S
        self.H = model.components_ #mixing matrix (kind of, some stuff about whittening in the documentation)...
        self.t = t
        self.ids = ids

        self.trained_ica = model

    def transform_ica(self,spectra,timestamps,trace_ids):
        X, t, ids = self.__construct_matrix(spectra,timestamps,trace_ids)
        S = self.trained_ica.transform(X)
        return S
    
    def train_clustering(self,n_clusters=None,threshold=1,branching_factor=1000):

        #try the birch algorithm as it might deal with outliers a bit better??
        sub_model = AgglomerativeClustering(n_clusters=n_clusters,metric='euclidean',linkage='ward')
        birch = Birch(n_clusters=sub_model, threshold=threshold,branching_factor=branching_factor).fit(self.S) #! issue here - need higher threshold in some cases, otherwise too many subclusters for Agglomerative. Try using proportion of IQR to capture scale...?
        
        self.trained_cluster = birch
        self.labels = birch.labels_
        self.cluster_names = np.unique(self.labels)
        self.n_clusters = self.cluster_names.size

    def compute_centroids(self):
        #take the average point in feature space for each cluster, scattering spectrum of centroid can be computed from there using H.

        centroids = {}
        dist = {}
        centroid_specs = {}

        group_times = {}
        group_ids = {}


        for name in self.cluster_names:

            ind = np.argwhere(self.labels==name).flatten()
            locs = self.S[ind,:]

            group_times[name] = self.t[ind]
            group_ids[name] = self.ids[ind]

            centroids[name] = np.mean(locs,axis=0)
            dist[name] = np.sqrt(np.sum((centroids[name] - locs)**2,axis=1))

            centroid_specs[name] = centroids[name] @ self.H
        
        self.centroids = centroids
        self.centroid_specs = centroid_specs
        self.dist = dist

        self.group_times = group_times
        self.group_ids = group_ids

        return centroids, centroid_specs, dist




    def __construct_matrix(self,spectra,timestamps,trace_ids):

        X_lst = []
        t_lst = []
        id_lst = []

        for key, layers in spectra.items():
            layer = layers[-1] #just take the deepest layer, can make more general later...
            ind = ~np.isnan(layer).any(axis=1)
            X_lst.append(layer[ind,...]) #this is probably unneccessary because they should be flattened by now?

            t_lst.append(timestamps[key][ind])
            id_lst.append(trace_ids[key][ind])


        X = np.concatenate(X_lst,axis=0)
        t = np.concatenate(t_lst)
        ids = np.concatenate(id_lst)
        
        return X, t, ids