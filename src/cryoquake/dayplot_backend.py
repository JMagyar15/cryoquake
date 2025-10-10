from obspy.core.util import create_empty_data_chunk
import numpy as np
import matplotlib.pyplot as plt


class StackPlot:
    def __init__(self,fig,mins_per_row,time_offset):

        self.mins_per_row = mins_per_row
        self.time_offset = time_offset
        self.dpi = fig.dpi
        self.width_p = int(np.round(fig.get_figwidth() * self.dpi)) #width in pixels
        self.fig = fig
        self.spi = int(self.mins_per_row * 60) #seconds per row (interval)
        self.x_values = np.repeat(np.arange(self.width_p), 2)

        self.min_max = {}
        self.events = {}


    def process_chunk(self,chunk):
        #chunk here defines time period for a page of the final document.
        for tr in chunk.stream:
            extreme_values = self.min_max_trace(tr)
            self.min_max.setdefault(tr.id,{})
            self.min_max[tr.id][chunk.str_name] = extreme_values
        

    def add_events(self,chunk,event_cat):
        event_streams = chunk.event_stream(event_cat)
        for group, event_stream in event_streams.items():
            self.events.setdefault(chunk.str_name,{})
            for tr in event_stream:
                extreme_values = 0.0 * self.min_max_trace(tr) + 0.5 #flatten out into centred highlighting line.
                self.events[chunk.str_name].setdefault(tr.id,{})
                self.events[chunk.str_name][tr.id][group] = extreme_values


    def dayplot_axes(self,tr_id,starttime,endtime):

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        #now set the axis labels, ticks, and title for the plot using information from the trace
        self.__dayplot_set_x_ticks(ax,self.spi,self.width_p)
        self.__dayplot_set_y_ticks(ax,starttime,endtime)
        self.__dayplot_set_title(ax,tr_id,starttime,endtime)

        #set up the axis with a light grid and no external box
        ax.grid(color='lightgrey',linewidth=0.5)
        ax.yaxis.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        return ax
    
    def min_max_trace(self,trace):
        """
        Take the given trace and compute the min-max values for the dayplot. This does not normalise them, which
        is done later for a set of traces to retain a consistant normalisation factor. 
        """
        tr_copy = trace.copy() #make a copy in here to avoid doing it outside - trace is altered so copy needed.
        # Helper variables for easier access.
        trace_length = len(tr_copy.data)

        # Samples per interval.
        samp_pi = int(self.mins_per_row * 60 * tr_copy.stats.sampling_rate)
        # Check the approximate number of samples per pixel and raise
        # error as fit.
        spp = float(samp_pi) / self.width_p #samples per pixel
        if spp < 1.0:
            msg = """
            Too few samples to use dayplot with the given arguments.
            Adjust your arguments or use a different plotting method.
            """
            msg = " ".join(msg.strip().split())
            raise ValueError(msg)
        # Number of intervals plotted.
        noi = float(trace_length) / samp_pi
        inoi = int(round(noi))
        # Plot an extra interval if at least 2 percent of the last interval
        # will actually contain data. Do it this way to lessen floating point
        # inaccuracies.
        if abs(noi - inoi) > 2E-2:
            noi = inoi + 1
        else:
            noi = inoi

        # Adjust data. Fill with masked values in case it is necessary.
        number_of_samples = noi * samp_pi
        delta = number_of_samples - trace_length
        if delta < 0:
            tr_copy.data = tr_copy.data[:number_of_samples]
        elif delta > 0:
            tr_copy.data = np.ma.concatenate([tr_copy.data, create_empty_data_chunk(delta, tr_copy.data.dtype)])

        # Create array for min/max values. Use masked arrays to handle gaps.
        extreme_values = np.ma.empty((noi, self.width_p, 2))
        tr_copy.data.shape = (noi, samp_pi)

        ispp = int(spp)
        fspp = spp % 1.0
        if fspp == 0.0:
            delta = None
        else:
            delta = samp_pi - ispp * self.width_p

        # Loop over each interval to avoid larger errors towards the end.
        for _i in range(noi):
            if delta:
                cur_interval = tr_copy.data[_i][:-delta]
                rest = tr_copy.data[_i][-delta:]
            else:
                cur_interval = tr_copy.data[_i]
            
            cur_interval.shape = (self.width_p, ispp) #reshape each interval (row) so be number of pixels in row, and number of samples in each pixel.
            extreme_values[_i, :, 0] = cur_interval.min(axis=1) #these then find the min and max within each pixel by min/max across each row.
            extreme_values[_i, :, 1] = cur_interval.max(axis=1)
            # Add the rest.
            if delta:
                extreme_values[_i, -1, 0] = min(extreme_values[_i, -1, 0],rest.min())
                extreme_values[_i, -1, 1] = max(extreme_values[_i, -1, 0],rest.max())

        extreme_values = extreme_values.astype(float) * trace.stats.calib
        extreme_values -= extreme_values.sum() / extreme_values.size #demean the min/max trace.

        return extreme_values
    
    def normalise_traces(self,percentile_delta=0.005):
        """
        Takes a list of traces that need to be normalised together using a percentile method.
        """
        all_min = []
        all_max = []
        for tr_id, tr_dict in self.min_max.items():
            for chunk_str, arr in tr_dict.items():
                all_min.append(arr[:,:,0].compressed())
                all_max.append(arr[:,:,1].compressed())


        min_values = np.concatenate(all_min).flatten()
        max_values = np.concatenate(all_max).flatten()

        #now find the percentiles with a sort.
        max_values.sort()
        min_values.sort()
        length = len(max_values)
        index = int((1.0 - percentile_delta) * length)
        max_val = max_values[index]
        index = int(percentile_delta * length)
        min_val = min_values[index]

        # Normalization factor
        self.normalization_factor = max(abs(max_val), abs(min_val)) * 2

        #now go through attached dictionary and replace the min-max traces with the normalised versions.
        for tr_id, tr_dict in self.min_max.items():
            for chunk_str, extreme_values in tr_dict.items():
                self.min_max[tr_id][chunk_str] = extreme_values * (1. / self.normalization_factor) + 0.5
    
        

    def make_page(self,tr_id,page_chunk):
        #set up the axes for this page
        ax = self.dayplot_axes(tr_id,page_chunk.starttime,page_chunk.endtime)

        #get the minmax and event streams for this page
        min_max = self.min_max[tr_id][page_chunk.str_name]
        
        self.__plot_minmax(ax,min_max)

        self.events.setdefault(page_chunk.str_name,{})
        self.events[page_chunk.str_name].setdefault(tr_id,{})
        events = self.events[page_chunk.str_name][tr_id]
        for group, group_events in events.items():
            self.__plot_events(ax,group_events,colour=self.__colourmapping(group))

        return self.fig, ax


    def __plot_minmax(self,ax,extreme_values):
        intervals = extreme_values.shape[0]
        for i in range(intervals): #loop over the rows of the extreme values array.
            y_values = np.ma.empty(self.width_p * 2)
            y_values.fill(intervals - (i + 1))
            # Add min and max values.
            y_values[0::2] += extreme_values[i, :, 0]
            y_values[1::2] += extreme_values[i, :, 1]
            ax.plot(self.x_values,y_values,color='black',lw=0.75)

        ax.set_xlim(0, self.width_p - 1)
        ax.set_ylim(-0.3, intervals + 0.3)

        return ax

    def __plot_events(self,ax,events,colour='red'):
        intervals = events.shape[0]
        for i in range(intervals): #loop over the rows of the extreme values array.
            y_values = np.ma.empty(self.width_p * 2)
            y_values.fill(intervals - (i + 1))
            # Add min and max values.
            y_values[0::2] += events[i, :, 0]
            y_values[1::2] += events[i, :, 1]
            ax.plot(self.x_values,y_values,color=colour,lw=10,alpha=0.25,solid_capstyle='butt')
            
        return ax

    def __dayplot_set_x_ticks(self,ax,interval,width_p,tick_rotation='horizontal',x_labels_size=12):
        """
        Sets the xticks for the dayplot.
        """

        max_value = width_p - 1
        # Check whether it is sec/mins/hours and convert to a universal unit.
        if interval < 240:
            time_type = 'seconds'
            time_value = interval
        elif interval < 24000:
            time_type = 'minutes'
            time_value = interval / 60
        else:
            time_type = 'hours'
            time_value = interval / 3600
        count = None
        # Hardcode some common values. The plus one is intentional. It had
        # hardly any performance impact and enhances readability.
        if interval == 15 * 60:
            count = 15 + 1
        elif interval == 20 * 60:
            count = 4 + 1
        elif interval == 30 * 60:
            count = 6 + 1
        elif interval == 60 * 60:
            count = 4 + 1
        elif interval == 90 * 60:
            count = 6 + 1
        elif interval == 120 * 60:
            count = 4 + 1
        elif interval == 180 * 60:
            count = 6 + 1
        elif interval == 240 * 60:
            count = 6 + 1
        elif interval == 300 * 60:
            count = 6 + 1
        elif interval == 360 * 60:
            count = 12 + 1
        elif interval == 720 * 60:
            count = 12 + 1
        # Otherwise run some kind of autodetection routine.
        if not count:
            # Up to 15 time units and if it's a full number, show every unit.
            if time_value <= 15 and time_value % 1 == 0:
                count = int(time_value)
            # Otherwise determine whether they are divisible for numbers up to
            # 15. If a number is not divisible just show 10 units.
            else:
                count = 10
                for _i in range(15, 1, -1):
                    if time_value % _i == 0:
                        count = _i
                        break
            # Show at least 5 ticks.
            if count < 5:
                count = 5

        # Calculate and set ticks.
        ticks = np.linspace(0.0, max_value, count)
        ticklabels = ['%i' % _i for _i in np.linspace(0.0, time_value, count)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, rotation=tick_rotation,
                                        size=x_labels_size)
        ax.set_xlabel('%s %s' % ('time in',
                                            time_type), size=x_labels_size)

    def __dayplot_set_y_ticks(self,ax,starttime,endtime,timezone='local time',y_labels_size=12,tick_format='%H:%M:%S'):
        """
        Sets the yticks for the dayplot.
        """

        intervals = int(round((endtime - starttime) / self.spi)) #number of rows needed for trace

        #TODO can this calculation of repeat be moved into the y_ticks code??
        if self.mins_per_row < 60 and 60 % self.mins_per_row == 0:
            repeat = 60 // self.mins_per_row
        elif self.mins_per_row < 1800 and 3600 % self.mins_per_row == 0:
            repeat = 3600 // self.mins_per_row
        # Otherwise use a maximum value of 10.
        else:
            if intervals >= 10:
                repeat = 10
            else:
                repeat = intervals

        if intervals <= 5:
            tick_steps = list(range(0, intervals))
            ticks = np.arange(intervals, 0, -1, dtype=float)
            ticks -= 0.5
        else:
            tick_steps = list(range(0, intervals, repeat))
            ticks = np.arange(intervals, 0, -1 * repeat, dtype=float)
            ticks -= 0.5

        sign = '%+i' % self.time_offset
        sign = sign[0]
        label = "UTC (%s = UTC %s %02i:%02i)" % (timezone.strip(), sign, abs(self.time_offset),(self.time_offset % 1 * 60))
        ticklabels = [(starttime + _i *self.spi).strftime(tick_format) for _i in tick_steps]
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels, size=y_labels_size)
        ax.set_ylabel(label)

    def __dayplot_set_title(self,ax,tr_id,starttime,endtime):
        title_str = tr_id + '__' + starttime.strftime("%Y%m%dT%H%M%SZ")+'__'+endtime.strftime("%Y%m%dT%H%M%SZ")
        ax.set_title(title_str)

    def __colourmapping(self,group):
        if group == 1:
            color = 'maroon' #simple down/up, short low freq impuslive
        elif group == 2:
            color = 'red' #multiple peaks, low freq impusive
        elif group == 3:
            color = 'orange' #short high freq impulsive
        elif group == 4:
            color = 'pink' #long high freq impulsive
        elif group == 5:
            color = 'yellow' #other impulsive signals
        elif group == 6:
            color = 'purple' #extended signals with dominant frequency, well structure
        elif group == 7:
            color = 'blue' #emergent low frequency signals, lack structure
        elif group == 8:
            color = 'green' #other emergent signals
        else:
            color = 'grey' #unknown signals.
        return color


class Dayplot:
    def __init__(self,starttime,endtime,mins_per_row,time_offset):
        self.starttime = starttime
        self.endtime = endtime
        self.mins_per_row = mins_per_row
        self.time_offset = time_offset

        

    def dayplot_axes(self,fig):
        """
        Want this to make the axes for the dayplot (including axis labels, etc.) and return the important information for plotting on these axes.
        """

        spi = int(self.mins_per_row * 60) #seconds in each interval.

        intervals = int(round((self.endtime - self.starttime) / spi))

        width_p = int(np.round(fig.get_figwidth() * fig.dpi))

        if self.mins_per_row < 60 and 60 % self.mins_per_row == 0:
            repeat = 60 // self.mins_per_row
        elif self.mins_per_row < 1800 and 3600 % self.mins_per_row == 0:
            repeat = 3600 // self.mins_per_row
        # Otherwise use a maximum value of 10.
        else:
            if intervals >= 10:
                repeat = 10
            else:
                repeat = intervals

        ax = fig.add_subplot(1,1,1) #this is the common axis for all the dayplotting to follow.

        x_values = np.repeat(np.arange(width_p), 2)
        ax.set_xlim(0, width_p - 1)
        ax.set_ylim(-0.3, intervals + 0.3)
        self.__dayplot_set_x_ticks(ax,spi,width_p)
        self.__dayplot_set_y_ticks(ax,self.starttime,spi,intervals,repeat)
        ax.grid(color='lightgrey',linewidth=0.5)
        ax.yaxis.grid(False)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        self.x_values = x_values
        self.intervals = intervals
        self.width_p = width_p
        return fig, ax
    
    def plot_trace(self,trace,ax,color='black'):
        extreme_values = self.__dayplot_get_min_max_values(trace,self.mins_per_row*60,self.width_p)
        extreme_values = self.__dayplot_normalize_values(trace,extreme_values)
        for i in range(self.intervals):
            y_values = np.ma.empty(self.width_p * 2)
            y_values.fill(self.intervals - (i + 1))
            # Add min and max values.
            y_values[0::2] += extreme_values[i, :, 0]
            y_values[1::2] += extreme_values[i, :, 1]
            ax.plot(self.x_values,y_values,color=color,lw=0.75)
    
    def plot_highlight(self,event_stream,ax,color='red',alpha=0.25,lw=10):
        full_stream = event_stream.trim(self.starttime,self.endtime,pad=True,fill_value=None)
        extreme_values = self.__dayplot_get_min_max_values(full_stream,self.mins_per_row*60,self.width_p)
        extreme_values *= 0 #set amplitude to zero everywhere
        extreme_values += 0.5
        for i in range(self.intervals):
            y_values = np.ma.empty(self.width_p * 2)
            y_values.fill(self.intervals - (i + 1))
            # Add min and max values.
            y_values[0::2] += extreme_values[i, :, 0]
            y_values[1::2] += extreme_values[i, :, 1]
            ax.plot(self.x_values,y_values,color=color,alpha=alpha,linewidth=lw)
        
    def colourmapping(self,group):
        if group == 1:
            color = 'maroon' #simple down/up, short low freq impuslive
        elif group == 2:
            color = 'red' #multiple peaks, low freq impusive
        elif group == 3:
            color = 'orange' #short high freq impulsive
        elif group == 4:
            color = 'pink' #long high freq impulsive
        elif group == 5:
            color = 'yellow' #other impulsive signals
        elif group == 6:
            color = 'purple' #extended signals with dominant frequency, well structure
        elif group == 7:
            color = 'blue' #emergent low frequency signals, lack structure
        elif group == 8:
            color = 'green' #other emergent signals
        else:
            color = 'grey' #unknown signals.
        return color


    def dayplot_get_min_max_values(self,trace,interval,width):
        """
        Takes a Stream object and calculates the min and max values for each
        pixel in the dayplot.

        Writes a three dimensional array. The first axis is the step, i.e
        number of trace, the second is the pixel in that step and the third
        contains the minimum and maximum value of the pixel.
        """
        # Helper variables for easier access.
        trace_length = len(trace.data)

        # Samples per interval.
        spi = int(interval * trace.stats.sampling_rate)
        # Check the approximate number of samples per pixel and raise
        # error as fit.
        spp = float(spi) / width
        if spp < 1.0:
            msg = """
            Too few samples to use dayplot with the given arguments.
            Adjust your arguments or use a different plotting method.
            """
            msg = " ".join(msg.strip().split())
            raise ValueError(msg)
        # Number of intervals plotted.
        noi = float(trace_length) / spi
        inoi = int(round(noi))
        # Plot an extra interval if at least 2 percent of the last interval
        # will actually contain data. Do it this way to lessen floating point
        # inaccuracies.
        if abs(noi - inoi) > 2E-2:
            noi = inoi + 1
        else:
            noi = inoi

        # Adjust data. Fill with masked values in case it is necessary.
        number_of_samples = noi * spi
        delta = number_of_samples - trace_length
        if delta < 0:
            trace.data = trace.data[:number_of_samples]
        elif delta > 0:
            trace.data = np.ma.concatenate(
                [trace.data, create_empty_data_chunk(delta, trace.data.dtype)])

        # Create array for min/max values. Use masked arrays to handle gaps.
        extreme_values = np.ma.empty((noi, width, 2))
        trace.data.shape = (noi, spi)

        ispp = int(spp)
        fspp = spp % 1.0
        if fspp == 0.0:
            delta = None
        else:
            delta = spi - ispp * width

        # Loop over each interval to avoid larger errors towards the end.
        for _i in range(noi):
            if delta:
                cur_interval = trace.data[_i][:-delta]
                rest = trace.data[_i][-delta:]
            else:
                cur_interval = trace.data[_i]
            cur_interval.shape = (width, ispp)
            extreme_values[_i, :, 0] = cur_interval.min(axis=1)
            extreme_values[_i, :, 1] = cur_interval.max(axis=1)
            # Add the rest.
            if delta:
                extreme_values[_i, -1, 0] = min(extreme_values[_i, -1, 0],
                                                rest.min())
                extreme_values[_i, -1, 1] = max(extreme_values[_i, -1, 0],
                                                rest.max())
        # Set class variable.
        return extreme_values



    def __dayplot_normalize_values(self,trace,extreme_values, percentile_delta=0.005):
        """
        Normalizes all values in the 3 dimensional array, so that the minimum
        value will be 0 and the maximum value will be 1.

        It will also convert all values to floats.
        """
        # Convert to native floats.
        extreme_values = extreme_values.astype(float) * \
            trace.stats.calib

        extreme_values -= extreme_values.sum() / \
            extreme_values.size

        # Scale so that 99.5 % of the data will fit the given range.
        max_values = extreme_values[:, :, 1].compressed()
        min_values = extreme_values[:, :, 0].compressed()
        # Remove masked values.
        max_values.sort()
        min_values.sort()
        length = len(max_values)
        index = int((1.0 - percentile_delta) * length)
        max_val = max_values[index]
        index = int(percentile_delta * length)
        min_val = min_values[index]


        # Normalization factor.
        normalization_factor = max(abs(max_val), abs(min_val)) * 2

        # Scale from 0 to 1.
        # raises underflow warning / error for numpy 1.9
        # even though normalization_factor is 2.5
        # self.extreme_values = self.extreme_values / \
        #     self._normalization_factor
        extreme_values = extreme_values * \
            (1. / normalization_factor)
        extreme_values += 0.5
        return extreme_values

    def __dayplot_set_x_ticks(self,ax,interval,width_p,tick_rotation='horizontal',x_labels_size=12):
        """
        Sets the xticks for the dayplot.
        """

        max_value = width_p - 1
        # Check whether it is sec/mins/hours and convert to a universal unit.
        if interval < 240:
            time_type = 'seconds'
            time_value = interval
        elif interval < 24000:
            time_type = 'minutes'
            time_value = interval / 60
        else:
            time_type = 'hours'
            time_value = interval / 3600
        count = None
        # Hardcode some common values. The plus one is intentional. It had
        # hardly any performance impact and enhances readability.
        if interval == 15 * 60:
            count = 15 + 1
        elif interval == 20 * 60:
            count = 4 + 1
        elif interval == 30 * 60:
            count = 6 + 1
        elif interval == 60 * 60:
            count = 4 + 1
        elif interval == 90 * 60:
            count = 6 + 1
        elif interval == 120 * 60:
            count = 4 + 1
        elif interval == 180 * 60:
            count = 6 + 1
        elif interval == 240 * 60:
            count = 6 + 1
        elif interval == 300 * 60:
            count = 6 + 1
        elif interval == 360 * 60:
            count = 12 + 1
        elif interval == 720 * 60:
            count = 12 + 1
        # Otherwise run some kind of autodetection routine.
        if not count:
            # Up to 15 time units and if it's a full number, show every unit.
            if time_value <= 15 and time_value % 1 == 0:
                count = int(time_value)
            # Otherwise determine whether they are divisible for numbers up to
            # 15. If a number is not divisible just show 10 units.
            else:
                count = 10
                for _i in range(15, 1, -1):
                    if time_value % _i == 0:
                        count = _i
                        break
            # Show at least 5 ticks.
            if count < 5:
                count = 5

        # Calculate and set ticks.
        ticks = np.linspace(0.0, max_value, count)
        ticklabels = ['%i' % _i for _i in np.linspace(0.0, time_value, count)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, rotation=tick_rotation,
                                        size=x_labels_size)
        ax.set_xlabel('%s %s' % ('time in',
                                            time_type), size=x_labels_size)

    def __dayplot_set_y_ticks(self,ax,starttime,interval,intervals,repeat,timezone='local time',y_labels_size=12,tick_format='%H:%M:%S',right_vertical_labels=False,show_y_UTC_label=True):
        """
        Sets the yticks for the dayplot.
        """

        if intervals <= 5:
            tick_steps = list(range(0, intervals))
            ticks = np.arange(intervals, 0, -1, dtype=float)
            ticks -= 0.5
        else:
            tick_steps = list(range(0, intervals, repeat))
            ticks = np.arange(intervals, 0, -1 * repeat, dtype=float)
            ticks -= 0.5

        sign = '%+i' % self.time_offset
        sign = sign[0]
        label = "UTC (%s = UTC %s %02i:%02i)" % (timezone.strip(), sign, abs(self.time_offset),(self.time_offset % 1 * 60))
        ticklabels = [(starttime + _i *interval).strftime(tick_format) for _i in tick_steps]
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels, size=y_labels_size)
        # Show time zone label if requested
        if show_y_UTC_label:
            ax.set_ylabel(label)
        if right_vertical_labels:
            yrange = ax.get_ylim()
            twin_x = ax.twinx()
            twin_x.set_ylim(yrange)
            twin_x.set_yticks(ticks)
            y_ticklabels_twin = [(starttime + (_i + 1) *interval).strftime(tick_format) for _i in tick_steps]
            twin_x.set_yticklabels(y_ticklabels_twin,size=y_labels_size)