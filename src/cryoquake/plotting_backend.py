"""functions for the actual plotting styles of waveforms/spectrograms/psds"""

import numpy as np
import matplotlib.pyplot as plt
from obspy.core import Stream, UTCDateTime
from obspy.imaging.util import _set_xaxis_obspy_dates


class StreamPlotting:
    def __init__(self,fig,rel_time=None):
        self.fig = fig
        self.rel_time = rel_time


    def make_axes(self,inv,components='Z',stack='stations',spectra=None,text=None):

        fig = self.fig

        if isinstance(components,str):
            components = [components]

        tr_ax = {}
        spec_ax = {}
        text_ax = {}

        num_stations = len(inv.get_contents()['stations'])    
        num_components = len(components)
        
        
        if text == 'top':
            text_fig, plot_fig = fig.subfigures(nrows=2,height_ratios=(0.4,1))
            text_ax['text'] = text_fig.add_subplot(111)
        elif text == 'right':
            plot_fig, text_fig = fig.subfigures(ncols=2,width_ratios=(1,0.4))
            text_ax['text'] = text_fig.add_subplot(111)
        else:
            plot_fig = fig

        if stack == 'stations':
            nrows, ncols = num_stations,num_components
            subfigs = plot_fig.subfigures(nrows=nrows,ncols=ncols,wspace=0,frameon=False)
            if num_stations == 1:
                subfigs = np.expand_dims(subfigs,axis=0)
            if num_components == 1:
                subfigs = np.expand_dims(subfigs,axis=1)
        elif stack == 'components':
            nrows, ncols = num_components, num_stations
            subfigs = plot_fig.subfigures(nrows=nrows,ncols=ncols,wspace=0,frameon=False)
            if num_components == 1:
                subfigs = np.expand_dims(subfigs,axis=0)
            if num_stations == 1:
                subfigs = np.expand_dims(subfigs,axis=1)
        else:
            raise(Exception('Stack option must be one of stations or components'))
    
        
        
        i = 0 #station index
        for network in inv:
            for station in network:
                for j, comp in enumerate(components): #j is component index

                    if stack == 'components':
                        ii = j
                        jj = i
                    else:
                        ii = i
                        jj = j

                    if spectra == 'right':
                        ax_grid = subfigs[ii,jj].add_gridspec(nrows=1,ncols=2,wspace=0.1,bottom=0.1,top=0.9)
                        tr_ax[station.code + '.' + comp] = subfigs[ii,jj].add_subplot(ax_grid[0])
                        spec_ax[station.code + '.' + comp] = subfigs[ii,jj].add_subplot(ax_grid[1])
                    elif spectra == 'bottom':
                        ax_grid = subfigs[ii,jj].add_gridspec(nrows=2,ncols=1,wspace=0.1,bottom=0.1,top=0.9)
                        tr_ax[station.code + '.' + comp] = subfigs[ii,jj].add_subplot(ax_grid[0])
                        spec_ax[station.code + '.' + comp] = subfigs[ii,jj].add_subplot(ax_grid[1])
                    else:
                        tr_ax[station.code + '.' + comp] = subfigs[ii,jj].add_subplot(111)

                    if ii < (nrows - 1):
                        tr_ax[station.code + '.' + comp].set_xticks([])
                        if spectra is not None:
                            spec_ax[station.code + '.' + comp].set_xticks([])
                    else:
                        if isinstance(self.rel_time,UTCDateTime):
                            plt.setp(tr_ax[station.code + '.' + comp].get_xticklabels(), fontsize='small')#,horizontalalignment='right')
                            if spectra is not None:
                                plt.setp(spec_ax[station.code + '.' + comp].get_xticklabels(), fontsize='small')#,horizontalalignment='right')
                        else:
                            _set_xaxis_obspy_dates(tr_ax[station.code + '.' + comp])
                            plt.setp(tr_ax[station.code + '.' + comp].get_xticklabels(), fontsize='small',rotation=90)#,horizontalalignment='right')
                            if spectra is not None:
                                _set_xaxis_obspy_dates(spec_ax[station.code + '.' + comp])
                                plt.setp(spec_ax[station.code + '.' + comp].get_xticklabels(), fontsize='small',rotation=90)#,horizontalalignment='right')

                    
                    if spectra is not None:
                        if jj < (ncols - 1):
                            spec_ax[station.code + '.' + comp].set_yticks([])
                        else:
                            spec_ax[station.code + '.' + comp].yaxis.set_label_position("right")
                            spec_ax[station.code + '.' + comp].yaxis.tick_right()
                            plt.setp(spec_ax[station.code + '.' + comp].get_yticklabels(), fontsize='small')


                    if jj == 0:
                        plt.setp(tr_ax[station.code + '.' + comp].get_yticklabels(), fontsize='small')
                    else:
                        tr_ax[station.code + '.' + comp].set_yticks([])

                    subfigs[ii,jj].suptitle(network.code + '.' + station.code + ':' + comp,fontsize=9,ha='center',va='center')
                i += 1

        self.tr_ax = tr_ax
        self.spec_ax = spec_ax
        self.text_ax = text_ax

    def make_condensed_axes(self,inv,components='Z',stack='stations'):

        fig = self.fig

        if isinstance(components,str):
            components = [components]

        tr_ax = {}
    
        num_stations = len(inv.get_contents()['stations'])    
        num_components = len(components)

        if stack == 'stations':
            subplots = [fig.add_subplot(1,num_components,j+1) for j in range(num_components)]

        elif stack == 'components':
            subplots = [fig.add_subplot(1,num_stations,j+1) for j in range(num_stations)]

        else:
            raise(Exception('Stack option must be one of stations or components'))
        
        for ax in subplots:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.set_yticks([])
            ax.set_xticks([])
        
        
        i = 0 #station index
        for network in inv:
            for station in network:
                for j, comp in enumerate(components): #j is component index
                    
                    if stack == 'components':
                        ii = j
                        jj = i
                    else:
                        ii = i
                        jj = j
                    
                    offset = ii 
                    tr_ax[station.code + '.' + comp] = (subplots[jj],offset) #tuple with subplot and offset for this trace
                
                i += 1
        
        self.tr_ax = tr_ax

    def plot_stream(self,stream,components,detections=None):

        amp_stream = Stream()
        for comp in components:
            amp_stream += stream.select(component=comp)

        max_amp = 0.0
        for tr in amp_stream:
            max_trace = np.max(np.absolute(tr.data))
            if max_trace > max_amp:
                max_amp = max_trace
        
        self.max_amp = 1.2 * max_amp

        for key, ax in self.tr_ax.items():
            station, comp = key.split('.')

            if isinstance(detections,dict):
                detections.setdefault(station,False)
                if detections[station]:
                    color = 'black'
                else:
                    color = 'grey'
            else:
                color = 'black'

            selected = stream.select(station=station,component=comp)
            if len(selected) == 1:
                tr = stream.select(station=station,component=comp)[0] #failsafe in here for missing component - try and except
                self.tr_ax[key] = self.__plot_trace(ax,tr,color=color,max_amp=self.max_amp)
            elif len(selected) > 1:
                print('Nonuniquness in station/channel IDs')
            else:
                print('Missing data for ',key)

    def plot_arrival(self,arrival_dict,color='red',ls='-',lw=1,label=None):
        for key, ax in self.tr_ax.items():
            max_amp = np.abs(np.array(ax.get_ylim())).max()

            station, comp = key.split('.')
            arrival_dict.setdefault(station,{})
            station_dict = arrival_dict[station]

            for label, time in station_dict.items():
                if isinstance(self.rel_time,UTCDateTime):
                    t = time - self.rel_time
                else:
                    t = time.matplotlib_date
                ax.vlines(t,ymin=-max_amp,ymax=max_amp,color=color,label=label,ls=ls,lw=lw)


    
    def plot_condensed_stream(self,stream,components,detections=None):
        amp_stream = Stream()
        for comp in components:
            amp_stream += stream.select(component=comp)

        max_amp = 0.0
        for tr in amp_stream:
            max_trace = np.max(np.absolute(tr.data))
            if max_trace > max_amp:
                max_amp = max_trace
        
        self.max_amp = max_amp

        for key, (ax,offset) in self.tr_ax.items():
            station, comp = key.split('.')

            if isinstance(detections,dict):
                if detections[station]:
                    color = 'black'
                else:
                    color = 'grey'
            else:
                color = 'black'

            selected = stream.select(station=station,component=comp)
            if len(selected) == 1:
                tr = stream.select(station=station,component=comp)[0] #failsafe in here for missing component - try and except
                self.tr_ax[key] = (self.__plot_wavestack(ax,tr,offset=-0.5*max_amp*offset,color=color),offset)
            elif len(selected) > 1:
                print('Nonuniquness in station/channel IDs')
            else:
                print('Missing data for ',key)
        
        yrange = [0,0]
        for key, (ax,offset) in self.tr_ax.items():
            ylim = ax.get_ylim()
            if ylim[0] < yrange[0]:
                yrange[0] = ylim[0]
            if ylim[1] > yrange[1]:
                yrange[1] = ylim[1]
        for key, (ax,offset) in self.tr_ax.items():
            ax.set_ylim(yrange)
            ax.tick_params(labelsize=8)
    
    def plot_condensed_arrival(self,arrival_dict,color='red',label=None):
        for key, (ax,offset) in self.tr_ax.items():
            shift = -0.5 * offset * self.max_amp

            station, comp = key.split('.')
            arrival_dict.setdefault(station,{})
            station_dict = arrival_dict[station]

            for label, time in station_dict.items():
                m_time = time.matplotlib_date
                ax.vlines(m_time,ymin=shift-0.25*self.max_amp,ymax=shift+0.25*self.max_amp,color=color,label=label)

    def plot_spectrogram(self,specs,components,detections=None):
        
        max_power = 0.0
        for station, spec in specs.items():
            for comp in components:
                spec_max = np.max(getattr(spec,comp))

                if spec_max > max_power: max_power = spec_max

        self.max_power = max_power

        for key, ax in self.spec_ax.items():
            station, comp = key.split('.')

            if isinstance(detections,dict):
                if detections[station]:
                    cmap = 'Blues'
                else:
                    cmap = 'Greys'
            else:
                cmap = 'Blues'

            plot_spec = getattr(specs[station],comp)
            f = specs[station].f
            t = specs[station].t
            self.spec_ax[key] = self.__plot_spectrogram(ax,plot_spec,f,t,cmap=cmap,max_power=self.max_power)

    def plot_density(self,psds,components,detections=None):
        pass

    def set_title(self,title,fontsize=16):
        self.fig.supylabel(title,fontsize=fontsize,weight='bold')

    def __plot_trace(self,ax,tr,color='black',max_amp=None):
        
        if isinstance(self.rel_time,UTCDateTime):
            times = tr.times('utcdatetime') - self.rel_time #time in seconds after relative time
        else:
            times = tr.times('matplotlib')
        
        ax.plot(times,tr.data,color=color,lw=0.8)

        if max_amp is not None:
            ax.set_ylim((-max_amp,max_amp))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        return ax
    
    def __plot_spectrogram(self,ax,plot_spec,f,t,cmap='Blues',max_power=None):
        """
        """   
        if isinstance(self.rel_time,UTCDateTime):
            times = (t - self.rel_time).astype(np.float64) #time in seconds after relative time
        else:
            times = [time.matplotlib_date for time in t]
        ax.pcolormesh(times,f,plot_spec,cmap=cmap,vmin=0,vmax=max_power)
        return ax

    def __plot_wavestack(self,ax,tr,offset,color='black'):

        try:
            ax.plot(tr.times("matplotlib"),tr.data + offset,color=color,lw=0.8)
        except:
            print('Missing data for station')

        #now put a label on the trace
        ax.set_yticks(list(ax.get_yticks()) + [offset])
        labels = ax.get_yticklabels()
        labels[-1] = tr.id
        ax.set_yticklabels(labels)
        return ax

