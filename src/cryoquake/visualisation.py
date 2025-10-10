import os
from cryoquake.data_objects import SeismicEvent, SeismicChunk
from cryoquake.plotting_backend import StreamPlotting
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from obspy.core import UTCDateTime, Stream, Trace
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


from matplotlib import rc
import matplotlib.font_manager as fm


class EventPlot(SeismicEvent):
    """
    Takes a SeismicEvent object and adds a collection of plotting methods.
    """

    def attach_fig(self,fig,save_path):
        """
        Attach a matplotlib figure which will be used for plotting methods. Also assigns a save directory.

        Inputs:
            fig: matplotlib.pyplot Figure object of desired size and layout parameters.
            save_path: directory to save output figures into.
        """
        self.fig = fig
        self.save_path = save_path

    def plot_event(self,components=['Z'],stack='stations',spectrogram='right',style='composite',textbox=None):
        """
        Master function for plotting individual events. 

        Inputs:
            components: components to include in plot. These must match the final character in the channel id
            stack: whether to vertically stack stations or components. Other will be stacked horizontally
            spectrogram: whether to plot spectrogram to the right ('right'), underneath ('bottom'), as a power
                spectrum ('spectrum', always to right) or not at all (None). Only applicable for composite style.
            style: whether to plot as a detailed composite plot with all axis labels and trigger times, or
                as a simplified condensed plot showing only the waveforms with tight spacing and minimal labelling.
        """
        if isinstance(components,str):
            components = [components] #if given as single component string, make it a single element list.
        self.fig.clear()
        
        if style == 'composite':
            fig = self.__composite_plot(self.fig,components=components,stack=stack,spectrogram=spectrogram,textbox=textbox)
        
        if style == 'condensed':
            fig = self.__condensed_plot(self.fig,components,stack=stack)
        return fig 

    def new_plotting(self,fig,components,stack='stations',spectrogram='right',textbox=None):

        if self.length is None:
            detrigger = True #only plot the detrigger time if the time axis is defined using last detrigger time.
        else:
            detrigger = False

        stream = self.get_data_window()
        detections = {station.split('.')[1]:True for station in self.stations}

        plotter = StreamPlotting(fig,rel_time=self.window_start)
        plotter.make_axes(self.inv,components=components,stack=stack,spectra=spectrogram,text=textbox)
        plotter.plot_stream(stream,components,detections=detections)
        plotter.set_title(self.event_id)

        
        arrivals = {}
        for code, tr_row in self.trace_rows.iterrows():
            station = code.split('.')[1]
            arrivals[station] = {}
            arrivals[station]['trigger'] = UTCDateTime(tr_row['time'])
            if detrigger:
                arrivals[station]['detrigger'] = UTCDateTime(tr_row['time'])+tr_row['duration']

        plotter.plot_arrival(arrivals,color='red',ls='--',lw=0.7)

        if spectrogram == 'density':
            plotter.plot_density(self.psds,components,detections=detections)

        if spectrogram in ['right','bottom']:
            plotter.plot_spectrogram(self.specs,components,detections=detections)

        if textbox is not None:
            plotter.textbox() #TODO this needs to be transferred from the old plotting code.

        return plotter.fig
    
    def __composite_plot(self,fig,components,stack='stations',spectrogram='right',textbox=None):
        """
        Produces a 'composite' plot for the event. This is a plot that always contains the time-domain waveform
        for each of the attached stations, and has optional frequency-domain representations for each.
        
        Inputs:
            fig: matplotlib.pyplot figure to add plotting axes to. This will generallly be pulled from self
                after the 'attach_fig' method is called.
            components: components to include in plot. These must match the final character in the channel id
            stack: whether to vertically stack stations or components. Other will be stacked horizontally
            spectrogram: whether to plot spectrogram to the right ('right'), underneath ('bottom'), as a power
                spectrum ('spectrum', always to right) or not at all (None).
        """
        if self.length is None:
            detrigger = True #only plot the detrigger time if the time axis is defined using last detrigger time.
        else:
            detrigger = False
        
        stream = self.get_data_window()

        amp_stream = Stream()
        for comp in components:
            amp_stream += stream.select(component=comp)

        max_amp = 0.0
        for tr in amp_stream:
            max_trace = np.max(np.absolute(tr.data))
            if max_trace > max_amp:
                max_amp = max_trace
        
        self.max_amp = 1.2 * max_amp

        if spectrogram == 'density':
            max_power = 0.0
            for station, psd in self.psds.items():
                if psd.max_power > max_power: max_power = psd.max_power

            self.max_power = max_power

        if spectrogram in ['right','bottom']:
            max_power = 0.0
            for station, spec in self.specs.items():
                if spec.max_power > max_power: max_power = spec.max_power

            self.max_power = max_power
        

        num_stations = len(self.inv)
        num_components = len(components)
        
        detections = [station.split('.')[1] for station in self.stations]

        
        if textbox == 'top':
            text_fig, plot_fig = fig.subfigures(nrows=2,height_ratios=(0.4,1))
        elif textbox == 'right':
            plot_fig, text_fig = fig.subfigures(ncols=2,width_ratios=(1,0.4))
        else:
            plot_fig = fig

        if stack == 'stations':
            subfigs = plot_fig.subfigures(nrows=num_stations,ncols=num_components)
            if num_stations == 1:
                subfigs = np.expand_dims(subfigs,axis=0)
            if num_components == 1:
                subfigs = np.expand_dims(subfigs,axis=1)
        elif stack == 'components':
            subfigs = plot_fig.subfigures(nrows=num_components,ncols=num_stations)
            if num_components == 1:
                subfigs = np.expand_dims(subfigs,axis=0)
            if num_stations == 1:
                subfigs = np.expand_dims(subfigs,axis=1)
        else:
            raise(Exception('Stack option must be one of stations or components'))
        
        i = 0 #station index
        for network in self.inv:
            for station in network:

                if station.code in detections:
                    detect = True
                    fontcolor = 'black'
                else:
                    detect = False
                    fontcolor = 'grey'


                for j, comp in enumerate(components): #j is component index

                    trace = stream.select(station=station.code,component=comp)[0] #get the associated trace

                    if stack == 'components':
                        ii = j
                        jj = i
                    else:
                        ii = i
                        jj = j


                    if spectrogram == 'right':
                        ax_grid = subfigs[ii,jj].add_gridspec(nrows=1,ncols=2)
                        trace_ax = subfigs[ii,jj].add_subplot(ax_grid[0])
                        spec_ax = subfigs[ii,jj].add_subplot(ax_grid[1])
                    elif spectrogram == 'bottom':
                        ax_grid = subfigs[ii,jj].add_gridspec(nrows=2,ncols=1)
                        trace_ax = subfigs[ii,jj].add_subplot(ax_grid[0])
                        spec_ax = subfigs[ii,jj].add_subplot(ax_grid[1])
                    elif spectrogram == 'density':
                        ax_grid = subfigs[ii,jj].add_gridspec(nrows=1,ncols=2)
                        trace_ax = subfigs[ii,jj].add_subplot(ax_grid[0])
                        spec_ax = subfigs[ii,jj].add_subplot(ax_grid[1])
                    else:
                        trace_ax = subfigs[ii,jj].add_subplot(111)

                    trace_ax = self.__plot_waveform(trace_ax,trace,detect=detect,detrigger=detrigger)

                    if spectrogram in ['right','bottom']:
                        spec_ax = self.__plot_spectrogram(spec_ax,self.specs[station.code],detect=detect,component=comp)
                    elif spectrogram == 'density':
                        spec_ax = self.__plot_psd(spec_ax,self.psds[station.code],component=comp)
                
                    subfigs[ii,jj].supylabel(trace.id,fontsize=12,ha='center',va='center',color=fontcolor)

                i += 1
        
        if textbox in ['top','right']:
            text_fig = self.__add_text(text_fig,textbox)

        fig.suptitle(self.event_id,fontsize=18)
        return fig
    
    def __condensed_plot(self,fig,components,stack='stations'):
        """
        Produces a condensed plot of the event. This includes the waveforms in the time domain with minimal
        axis labels and spacing. This is useful when looking at a larger number of stations, where the
        composite plot becomes crowded. No frequency-domain representation is included for this plot type.

        Inputs:
            fig: matplotlib.pyplot figure to add plotting axes to. This will generallly be pulled from self
                after the 'attach_fig' method is called.
            components: components to include in plot. These must match the final character in the channel id
            stack: whether to vertically stack stations or components. Other will be stacked horizontally
        """
        stream = self.get_data_window()
        amp_stream = Stream()
        for comp in components:
            amp_stream += stream.select(component=comp)

        max_amp = 0
        for tr in amp_stream:
            max_trace = np.max(np.absolute(tr.data))
            if max_trace > max_amp:
                max_amp = max_trace
        
        self.max_amp = 1.2 * max_amp        
        
        num_stations = len(self.inv)
        num_components = len(components)

        if stack == 'stations':
            subplots = [fig.add_subplot(1,num_components,j+1) for j in range(num_components)]

        elif stack == 'components':
            subplots = [fig.add_subplot(1,num_stations,j+1) for j in range(num_stations)]

        else:
            raise(Exception('Stack option must be one of stations or components'))
        
        for ax in subplots:
            self.__clean_axis(ax,ticks=False)
            ax.set_xlim((-self.buffer,self.length))
        
        
        i = 0 #station index
        for network in self.inv:
            for station in network:
                for j, comp in enumerate(components): #j is component index

                    trace = stream.select(station=station.code,component=comp)[0] #get the associated trace
                    
                    if stack == 'components':
                        ii = j
                        jj = i
                    else:
                        ii = i
                        jj = j
                    
                    offset = - 0.5 * self.max_amp * ii #move down by half of max amplitude for each 'row'
                    subplots[jj] = self.__plot_wavestack(subplots[jj],trace,offset,self.buffer)
                i += 1
        
        fontprops = fm.FontProperties(size=18)
        scalebar = AnchoredSizeBar(subplots[0].transData,2, '2 s', 'lower left', pad=0,color='red',frameon=False,size_vertical=0.01*self.max_amp,fontproperties=fontprops)
        subplots[0].add_artist(scalebar)

        #now set the ylims to be the same for all subplots
        yrange = [0,0]
        for ax in subplots:
            ylim = ax.get_ylim()
            if ylim[0] < yrange[0]:
                yrange[0] = ylim[0]
            if ylim[1] > yrange[1]:
                yrange[1] = ylim[1]
        for ax in subplots:
            ax.set_ylim(yrange)
        return fig
    
    def __plot_waveform(self,ax,trace,detect=True,detrigger=True):
        """
        Plots single single trace on given axis in format for composite plot.

        Inputs:
            ax: matplotlib.pyplot axis object to plot trace on
            trace: obspy Trace object that is to be plotted
            detect: boolian specifying if this station triggered a detection. Stations
                that did not trigger a detection are greyed out and do not have a trigger
                time annotated.
            detrigger: boolian to specify if detrigger time should be plotted.
        """
        
        #choose the color based on the detection
        if detect:
            color='black'
        else:
            color='grey'
            
        ax.plot(trace.times()-self.buffer,trace.data,color=color,lw=1.5)
        ax.set_ylim((-self.max_amp,self.max_amp))
        self.__clean_axis(ax)
        
        #work out the relative trigger times for plotting
        station_id = trace.id.split('.')[0] + '.' + trace.id.split('.')[1] + '.' 
        
        if detect:
            start = UTCDateTime(self.trace_rows.loc[station_id]['time']) - self.window_start
            end = start + self.trace_rows.loc[station_id]['duration']

            if self.triggers:
                if detrigger:
                    ax.vlines([start,end],ymin=-self.max_amp,ymax=self.max_amp,color='red',ls='--')
                else:
                    ax.vlines([start],ymin=-self.max_amp,ymax=self.max_amp,color='red',ls='--')
        ax.set_ylabel('Amplitude [count]') #TODO change this if instrument response is removed
        ax.set_xlabel('Relative time [s]')
        return ax
    
    def __plot_spectrogram(self,ax,spec,detect=True,component='Z'):
        """
        Plots spectrogram on given axis. The spectrogram for the event must be calculated
        first separately using the 'spectral' context for the event. This is attached as
        a EventSpectrum object that is accessed here for the plotting.

        Inputs:
            ax: axis to plot spectrogram on
            spec: EventSpectrum object containing short time Fourier transform for the event.
            detect: whether the station registered a detection. Spectrum will be greyed out
                if not detected at station.
            component: the component to plot the spectrogram for. 
        """

        if detect:
            cmap='Blues'
        else:
            cmap='Greys'

        plot_spec = getattr(spec,component)
            
        ax.pcolor(spec.t-self.buffer,spec.f,plot_spec,cmap=cmap,vmin=0,vmax=self.max_power)
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Relative time [s]')

        return ax
    
    def __plot_wavestack(self,ax,trace,offset,buffer):
        ax.plot(trace.times()-buffer,trace.data + offset,color='black',lw=1.5)
        return ax
    
    def __plot_psd(self,ax,psds,component='Z'):

        plot_spec = getattr(psds,component)
        
        ax.set_xscale('log')
        
        plot_f = psds.f
        
        cumulative_psd = np.cumsum(plot_spec)
        cumulative_psd /= cumulative_psd[-1] #normalise to final value to get CDF def

        i25 = np.abs(cumulative_psd - 0.25).argmin()
        i50 = np.abs(cumulative_psd - 0.50).argmin()
        i75 = np.abs(cumulative_psd - 0.75).argmin()

        ax.fill_between(plot_f[i25:i75+1],plot_spec[i25:i75+1],color='red',alpha=0.25)

        ax.vlines(plot_f[i25],ymin=0,ymax=plot_spec[i25],color='red',ls='--')
        ax.vlines(plot_f[i50],ymin=0,ymax=plot_spec[i50],color='red',ls='-')
        ax.vlines(plot_f[i75],ymin=0,ymax=plot_spec[i75],color='red',ls='--')
        ax.plot(plot_f,plot_spec,color='black')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_yticks([])
        #ax.set_ylim(bottom=0)
        ax.set_ylim((0,1.1 * self.max_power))
        return ax
    
    def __clean_axis(self,ax,ticks=True):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        if not ticks:
            ax.set_yticks([])
            ax.set_xticks([])

    def __add_text(self,text_fig,position):
        """
        Adds event text to the composite plot. This details the computed attributes, and time and
        environmental information (still to be implemented).
        """
        text_ax = text_fig.add_subplot(111) #only one needed (maybe do multiple columns later?)
        if position == 'right':
            text_ax.spines['bottom'].set_visible(False)
            x, y = 0.05, 0.98
        if position == 'top':
            text_ax.spines['left'].set_visible(False)
            x, y = 0.02, 0.95

        text_ax.spines['top'].set_visible(False)
        text_ax.spines['right'].set_visible(False)

        textstr = r''
        for i, row in self.attributes.iterrows():
            #write out the attributes
            label = row['Label']
            value = row['Value']
            if isinstance(value,UTCDateTime):
                t_str = value.strftime("%H:%M:%S")
                textstr += (label + ': ' + t_str + '\n')
            else:
                textstr += (label + ': {:.2f}'.format(value) + '\n')
        text_ax.set_xticks([])
        text_ax.set_yticks([])

        #textstr = 'testing'

        text_ax.text(x, y, textstr, transform=text_ax.transAxes, fontsize=12,
        verticalalignment='top')

        text_ax.set_title('Event Attributes',fontsize=14)
        return text_fig
    

class ChunkPlot(SeismicChunk):
    """
    Methods for summarising all data within a chunk (for event analysis in chunk, see ChunkEventPlot). 
    Access methods using SeismicChunk.context('plot').
    """

    def attach_fig(self,fig,save_path):
        """
        Attach a matplotlib figure which will be used for plotting methods. Also assigns a save directory.

        Inputs:
            fig: matplotlib.pyplot Figure object of desired size and layout parameters.
            save_path: directory to save output figures into.
        """
        self.fig = fig
        self.save_path = save_path

    def plot_waveforms(self,components=['Z'],stack='stations',style='composite'):
        """
        Master function for plotting individual events. 

        Inputs:
            components: components to include in plot. These must match the final character in the channel id
            stack: whether to vertically stack stations or components. Other will be stacked horizontally
            spectrogram: whether to plot spectrogram to the right ('right'), underneath ('bottom'), as a power
                spectrum ('spectrum', always to right) or not at all (None). Only applicable for composite style.
            style: whether to plot as a detailed composite plot with all axis labels and trigger times, or
                as a simplified condensed plot showing only the waveforms with tight spacing and minimal labelling.
        """
        if isinstance(components,str):
            components = [components] #if given as single component string, make it a single element list.
        self.fig.clear()
        
        if style == 'composite':
            fig = self.__composite_plot(self.fig,components=components,stack=stack)
        
        if style == 'condensed':
            fig = self.__condensed_plot(self.fig,components,stack=stack)
        return fig       
    
    def __composite_plot(self,plot_fig,components,stack='stations'):
        """
        Produces a 'composite' plot for the event. This is a plot that always contains the time-domain waveform
        for each of the attached stations, and has optional frequency-domain representations for each.
        
        Inputs:
            fig: matplotlib.pyplot figure to add plotting axes to. This will generallly be pulled from self
                after the 'attach_fig' method is called.
            components: components to include in plot. These must match the final character in the channel id
            stack: whether to vertically stack stations or components. Other will be stacked horizontally
            spectrogram: whether to plot spectrogram to the right ('right'), underneath ('bottom'), as a power
                spectrum ('spectrum', always to right) or not at all (None).
        """
    
        
        stream = self.stream

        amp_stream = Stream()
        for comp in components:
            amp_stream += stream.select(component=comp)

        max_amp = 0.0
        for tr in amp_stream:
            max_trace = np.max(np.absolute(tr.data))
            if max_trace > max_amp:
                max_amp = max_trace
        
        self.max_amp = 1.2 * max_amp
        

        num_stations = len(self.inv)
        num_components = len(components)
    
        if stack == 'stations':
            subfigs = plot_fig.subfigures(nrows=num_stations,ncols=num_components)
            if num_stations == 1:
                subfigs = np.expand_dims(subfigs,axis=0)
            if num_components == 1:
                subfigs = np.expand_dims(subfigs,axis=1)
        elif stack == 'components':
            subfigs = plot_fig.subfigures(nrows=num_components,ncols=num_stations)
            if num_components == 1:
                subfigs = np.expand_dims(subfigs,axis=0)
            if num_stations == 1:
                subfigs = np.expand_dims(subfigs,axis=1)
        else:
            raise(Exception('Stack option must be one of stations or components'))
        
        i = 0 #station index
        for network in self.inv:
            for station in network:

                for j, comp in enumerate(components): #j is component index

                    trace = stream.select(station=station.code,component=comp)[0] #get the associated trace

                    if stack == 'components':
                        ii = j
                        jj = i
                    else:
                        ii = i
                        jj = j
           
                    trace_ax = subfigs[ii,jj].add_subplot(111)
                    trace_ax = self.__plot_waveform(trace_ax,trace) #TODO move this function into general plotting area
                    subfigs[ii,jj].supylabel(trace.id,fontsize=12,ha='center',va='center') #TODO embed this into the new function?

                i += 1
        
        return plot_fig
    
    def __condensed_plot(self,fig,components,stack='stations'):
        """
        Produces a condensed plot of the event. This includes the waveforms in the time domain with minimal
        axis labels and spacing. This is useful when looking at a larger number of stations, where the
        composite plot becomes crowded. No frequency-domain representation is included for this plot type.

        Inputs:
            fig: matplotlib.pyplot figure to add plotting axes to. This will generallly be pulled from self
                after the 'attach_fig' method is called.
            components: components to include in plot. These must match the final character in the channel id
            stack: whether to vertically stack stations or components. Other will be stacked horizontally
        """
        stream = self.stream
        amp_stream = Stream()
        for comp in components:
            amp_stream += stream.select(component=comp)

        max_amp = 0
        for tr in amp_stream:
            max_trace = np.max(np.absolute(tr.data))
            if max_trace > max_amp:
                max_amp = max_trace
        
        self.max_amp = 1.2 * max_amp        
        
        num_stations = len(self.inv)
        num_components = len(components)

        if stack == 'stations':
            subplots = [fig.add_subplot(1,num_components,j+1) for j in range(num_components)]

        elif stack == 'components':
            subplots = [fig.add_subplot(1,num_stations,j+1) for j in range(num_stations)]

        else:
            raise(Exception('Stack option must be one of stations or components'))
        
        for ax in subplots:
            self.__clean_axis(ax,ticks=False)
        
        
        i = 0 #station index
        for network in self.inv:
            for station in network:
                for j, comp in enumerate(components): #j is component index

                    trace = stream.select(station=station.code,component=comp)[0] #get the associated trace
                    
                    if stack == 'components':
                        ii = j
                        jj = i
                    else:
                        ii = i
                        jj = j
                    
                    offset = - 0.5 * self.max_amp * ii #move down by half of max amplitude for each 'row'
                    subplots[jj] = self.__plot_wavestack(subplots[jj],trace,offset)
                i += 1
        
        #fontprops = fm.FontProperties(size=18)
        #scalebar = AnchoredSizeBar(subplots[0].transData,2, '2 s', 'lower left', pad=0,color='red',frameon=False,size_vertical=0.01*self.max_amp,fontproperties=fontprops)
        #subplots[0].add_artist(scalebar)

        #now set the ylims to be the same for all subplots
        yrange = [0,0]
        for ax in subplots:
            ylim = ax.get_ylim()
            if ylim[0] < yrange[0]:
                yrange[0] = ylim[0]
            if ylim[1] > yrange[1]:
                yrange[1] = ylim[1]
        for ax in subplots:
            ax.set_ylim(yrange)
        return fig
    
    def __plot_waveform(self,ax,trace,color='black'):
        """
        Plots single single trace on given axis in format for composite plot.

        Inputs:
            ax: matplotlib.pyplot axis object to plot trace on
            trace: obspy Trace object that is to be plotted
            detect: boolian specifying if this station triggered a detection. Stations
                that did not trigger a detection are greyed out and do not have a trigger
                time annotated.
            detrigger: boolian to specify if detrigger time should be plotted.
        """
        
        m_times = np.array([(trace.stats.starttime + delta).matplotlib_date for delta in trace.times()])       
            
        ax.plot(m_times,trace.data,color=color,lw=1.5)
        ax.set_ylim((-self.max_amp,self.max_amp))
        self.__clean_axis(ax)
            
        ax.set_ylabel('Amplitude [count]') #TODO change this if instrument response is removed
        ax.set_xlabel('Relative time [s]')
        return ax
    
    def __plot_spectrogram(self,ax,spec,detect=True,component='Z'):
        """
        Plots spectrogram on given axis. The spectrogram for the event must be calculated
        first separately using the 'spectral' context for the event. This is attached as
        a EventSpectrum object that is accessed here for the plotting.

        Inputs:
            ax: axis to plot spectrogram on
            spec: EventSpectrum object containing short time Fourier transform for the event.
            detect: whether the station registered a detection. Spectrum will be greyed out
                if not detected at station.
            component: the component to plot the spectrogram for. 
        """

        if detect:
            cmap='Blues'
        else:
            cmap='Greys'

        plot_spec = getattr(spec,component)
            
        ax.pcolor(spec.t-self.buffer,spec.f,plot_spec,cmap=cmap,vmin=0,vmax=self.max_power)
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Relative time [s]')

        return ax
    
    def __plot_wavestack(self,ax,trace,offset):
        m_times = np.array([(trace.stats.starttime + delta).matplotlib_date for delta in trace.times()])
        ax.plot(m_times,trace.data + offset,color='black',lw=1)
        return ax
    
    def __plot_psd(self,ax,psds,component='Z'):

        plot_spec = getattr(psds,component)
        
        ax.set_xscale('log')
        
        plot_f = psds.f
        
        cumulative_psd = np.cumsum(plot_spec)
        cumulative_psd /= cumulative_psd[-1] #normalise to final value to get CDF def

        i25 = np.abs(cumulative_psd - 0.25).argmin()
        i50 = np.abs(cumulative_psd - 0.50).argmin()
        i75 = np.abs(cumulative_psd - 0.75).argmin()

        ax.fill_between(plot_f[i25:i75+1],plot_spec[i25:i75+1],color='red',alpha=0.25)

        ax.vlines(plot_f[i25],ymin=0,ymax=plot_spec[i25],color='red',ls='--')
        ax.vlines(plot_f[i50],ymin=0,ymax=plot_spec[i50],color='red',ls='-')
        ax.vlines(plot_f[i75],ymin=0,ymax=plot_spec[i75],color='red',ls='--')
        ax.plot(plot_f,plot_spec,color='black')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_yticks([])
        #ax.set_ylim(bottom=0)
        ax.set_ylim((0,1.1 * self.max_power))
        return ax
    
    def __clean_axis(self,ax,ticks=True):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        if not ticks:
            ax.set_yticks([])
            ax.set_xticks([])

    def __add_text(self,text_fig,position):
        """
        Adds event text to the composite plot. This details the computed attributes, and time and
        environmental information (still to be implemented).
        """
        text_ax = text_fig.add_subplot(111) #only one needed (maybe do multiple columns later?)
        if position == 'right':
            text_ax.spines['bottom'].set_visible(False)
            x, y = 0.05, 0.98
        if position == 'top':
            text_ax.spines['left'].set_visible(False)
            x, y = 0.02, 0.95

        text_ax.spines['top'].set_visible(False)
        text_ax.spines['right'].set_visible(False)

        textstr = r''
        for i, row in self.attributes.iterrows():
            #write out the attributes
            label = row['Label']
            value = row['Value']
            if isinstance(value,UTCDateTime):
                t_str = value.strftime("%H:%M:%S")
                textstr += (label + ': ' + t_str + '\n')
            else:
                textstr += (label + ': {:.2f}'.format(value) + '\n')
        text_ax.set_xticks([])
        text_ax.set_yticks([])

        #textstr = 'testing'

        text_ax.text(x, y, textstr, transform=text_ax.transAxes, fontsize=12,
        verticalalignment='top')

        text_ax.set_title('Event Attributes',fontsize=14)
        return text_fig

    
class ChunkEventPlot(SeismicChunk):
    #TODO these types of plots should be applied to the EventCatalogue object - make plotting context for this with these types of plots.
    """
    Contains methods for plotting trends and characteristics of *events* within a chunk (rather than all data within the chunk
    as is the case with ChunkPlot). This is mainly in the form of histograms of event attributes and distributions.
    """
    def attach_fig(self,fig,save_path):
        self.fig = fig
        self.save_path = save_path

    def attribute_hist(self,att_name,num_bins,groups=True,polar=False,percentile=[0,100]):
        #self.attach_attributes()
        att_cat = self.attribute_catalogue[['event_id','group',att_name]]
        if polar:
            att_cat[att_name] *= (np.pi/180)
        
        att_arr = att_cat[att_name].to_numpy()

        if groups:
            group_ind = att_cat.group.unique()
            num_groups = len(group_ind)
        else:
            num_groups = None

        if polar:
            att_min, att_max = np.min(att_arr), np.max(att_arr)
            if ((att_min < 0) | (att_max > 2*np.pi)):
                raise(Exception('Attribute data is not constrained to [0,360], so directional plot has non-unique bins'))
            bin_edges = np.linspace(0,2*np.pi,num_bins+1)
        else:
            min_range, max_range = np.percentile(att_arr,percentile[0]), np.percentile(att_arr,percentile[1])
            bin_edges = np.linspace(min_range,max_range,num_bins+1)

        width = bin_edges[1] - bin_edges[0]
        
        #binning of attribute of interest
        bin_ind = np.arange(num_bins)
        att_cat['bin'] = pd.cut(att_cat[att_name],bin_edges,labels=bin_ind) #bin the attribute as a new column in the dataframe

        main_ax, group_ax = self.__hist_axes(groups,num_groups,polar)

        #now the axes are set up and the data binned, so ready for plotting histograms.
        cumulative = np.zeros(num_bins)
        c = ['maroon','red','darkorange','hotpink','gold','purple','blue','darkgreen'] #need way to make this longer if there is more groups than this - equally spaced cuts of rainbow colormap???
        if groups:
            for i, group_num in enumerate(group_ind):
                group_att = att_cat[att_cat['group'] == group_num]
                counts = group_att['bin'].value_counts().to_dict()
                for key in bin_ind:
                    counts.setdefault(key,0)
                
                values = np.array([counts[i] for i in range(len(bin_ind))])
                main_ax.bar(bin_edges[:-1],values,width=width,bottom=cumulative,align='edge',color=c[i],alpha=0.75)

                cumulative += values

                group_ax[i].bar(bin_edges[:-1],values,width=width,align='edge',color=c[i],alpha=0.75)
                if not polar:
                    group_ax[i].set_xlim((bin_edges[0],bin_edges[-1]))
        else:
            #single plot when there are not groups
            counts = att_cat['bin'].value_counts().to_dict()
            for key in bin_ind:
                counts.setdefault(key,0)
            values = np.array([counts[i] for i in range(len(bin_ind))])
            main_ax.bar(bin_edges[:-1],values,width=width,align='edge',color='grey',alpha=0.75)

        if not polar:
            main_ax.set_xlim((bin_edges[0],bin_edges[-1]))
            main_ax.set_xlabel(att_name) #TODO somehow attach a full name with units at metadata for automatic labelling.
            #TODO will need to make some type of attributes object which is attached to the event and has info on units, etc.


    def attribute_hist2d(self,att1,att2,num_bins,groups=True,polar=False,percentile=[0,100]):
        """
        Produces a 2D histogram of a given pair of attributes. This can be a rectangular plot if both attributes
        are linear, or polar if one of the attributes is directional. If a directional attribute is included, it must
        be entred as att1.

        Inputs:
            att1: name of attribute 1 (str). This will be plotted on the x axis if polar==False or as the angular
                coordinate if polar==True.
            att2: name of attribute 2 (str). This will be plotted on the y axis if polar==False or as the radial
                coordinate if polar==True.
            num_bins: number of histogram bins to use for each attribute (int)
            groups: whether to split the histogram into a separate plot for each assigned group (bool)
            percentile: percentile of the data to use as the limit of the upper and lower bins.
                Defaults to the min and max values.
        """
        #self.attach_attributes()
        joint_cat = self.attribute_catalogue[['event_id','group',att1,att2]]

        if polar:
            joint_cat[att1] *= (np.pi/180)
        
        att1_arr = joint_cat[att1].to_numpy()
        att2_arr = joint_cat[att2].to_numpy()

        if groups:
            group_ind = joint_cat.group.unique()
            num_groups = len(group_ind)
        else:
            num_groups = None

        if polar:
            att1_min, att1_max = np.min(att1_arr), np.max(att1_arr)
            if ((att1_min < 0) | (att1_max > 2*np.pi)):
                raise(Exception('Attribute data is not constrained to [0,2pi], so directional plot has non-unique bins'))
            bin1_edges = np.linspace(0,2*np.pi,num_bins+1)
            min_range2, max_range2 = np.percentile(att2_arr,percentile[0]), np.percentile(att2_arr,percentile[1])
            bin2_edges = np.linspace(min_range2,max_range2,num_bins+1)
        else:
            min_range1, max_range1 = np.percentile(att1_arr,percentile[0]), np.percentile(att1_arr,percentile[1])
            bin1_edges = np.linspace(min_range1,max_range1,num_bins+1)

            min_range2, max_range2 = np.percentile(att2_arr,percentile[0]), np.percentile(att2_arr,percentile[1])
            bin2_edges = np.linspace(min_range2,max_range2,num_bins+1)

        width1 = bin1_edges[1] - bin1_edges[0]
        width2 = bin2_edges[1] - bin2_edges[0]
        
        #binning of attribute of interest
        bin_ind = np.arange(num_bins)
        keys = [(i,j) for i in bin_ind for j in bin_ind]
        joint_cat['bin1'] = pd.cut(joint_cat[att1],bin1_edges,labels=bin_ind) #bin the attribute as a new column in the dataframe
        joint_cat['bin2'] = pd.cut(joint_cat[att2],bin2_edges,labels=bin_ind) #bin the attribute as a new column in the dataframe
        joint_cat = joint_cat.assign(cartesian=pd.Categorical(joint_cat.filter(regex='bin').apply(tuple, 1))) #make a tuple of the two bin indices.

        main_ax, group_ax = self.__hist_axes(groups,num_groups,polar,clean=False)

        if polar:
            main_ax.set_theta_zero_location('N')
            main_ax.set_theta_direction(-1)

        counts = joint_cat['cartesian'].value_counts().to_dict()
        scale = max(counts.values())
        for key in keys:
            counts.setdefault(key,0)
            count = counts[key]
            main_ax.bar(bin1_edges[key[0]],width2,width=width1,bottom=bin2_edges[key[1]],align='edge',color='black',alpha=count/scale)
        
        main_ax.set_xlim((bin1_edges[0],bin1_edges[-1]))
        main_ax.set_ylim((bin2_edges[0],bin2_edges[-1]))

        #now the axes are set up and the data binned, so ready for plotting histograms.
        c = ['maroon','red','darkorange','hotpink','gold','purple','blue','darkgreen'] #need way to make this longer if there is more groups than this - equally spaced cuts of rainbow colormap???
        if groups:
            for i, group_num in enumerate(group_ind):
                if polar:
                    group_ax[i].set_theta_zero_location('N')
                    group_ax[i].set_theta_direction(-1)
                group_att = joint_cat[joint_cat['group'] == group_num]
                counts = group_att['cartesian'].value_counts().to_dict()
                scale = max(counts.values())
                for key in keys:
                    counts.setdefault(key,0)
                    count = counts[key]
                    
                    group_ax[i].bar(bin1_edges[key[0]],width2,width=width1,bottom=bin2_edges[key[1]],align='edge',color=c[i],alpha=count/scale)
                    group_ax[i].set_xlim((bin1_edges[0],bin1_edges[-1]))
                    group_ax[i].set_ylim((bin2_edges[0],bin2_edges[-1]))

                    group_ax[i].set_xticks([])
                    group_ax[i].set_yticks([])
        
        if not polar:
            main_ax.set_xlabel(att1)
            main_ax.set_ylabel(att2) #TODO change these to the formatted names with units.



    def __hist_axes(self,groups,num_groups,polar,clean=True):
        """
        Sets up the axes for plotting 1D and 2D histograms. Used internally by these plotting methods.

        Inputs:
            groups: whether axes for each group should be made (bool)
            num_groups: number of groups the data has been assigned into (int)
            polar: whether the histograms should be polar plots (bool)
            clean: whether to remove the spines and labels from the axes (bool)
        """
        if groups:
            group_ax = []
            num_rows = 2
            num_cols = num_groups // 2 + 1
            width_ratios = [3] + ((num_cols-1) * [1])
            grid_spec = self.fig.add_gridspec(ncols=num_cols,nrows=num_rows,width_ratios=width_ratios)
            main_ax = self.fig.add_subplot(grid_spec[:,0],polar=polar)
            for i in range(num_groups):
                ax = self.fig.add_subplot(grid_spec[i//(num_cols-1),i%(num_cols-1) + 1],polar=polar)
                if clean: self.__clean_axes(ax,polar)
                group_ax.append(ax)
            if clean: self.__clean_axes(main_ax,polar,labels=True)
        else:
            main_ax = self.fig.add_subplot(111,polar=polar)
            if clean: self.__clean_axes(main_ax,polar,labels=True)
            group_ax = None

        return main_ax, group_ax
    
    def __clean_axes(self,ax,polar,labels=False):
        if polar:
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.spines['polar'].set_visible(False)
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

        if not labels:
            ax.set_xticks([])
            ax.set_yticks([])
