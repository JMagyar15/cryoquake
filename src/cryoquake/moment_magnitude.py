import numpy as np
import scipy
import pandas as pd
from multitaper.mtspec import MTSpec
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import cos, log10, exp



def SourceSpectrum(stream,arrivals,freqmin=0,freqmax=np.inf):
    f_dict = {}
    Pxx_dict = {}
    ci_dict = {}
    err_dict = {}

    for sta, row in arrivals.iterrows():
        tr = stream.select(station=sta,component='Z')[0]
        #tr.stats.starttime = UTCDateTime(2019,1,10)
        #tr.remove_response(inv,output='DISP',pre_filt=[0.1,0.1,500,500])
        fs = tr.stats.sampling_rate
        N_window = int((row['P_end'] - row['P_start'])*fs)
        N_start = int(row['P_start']*fs)

        #now compute the PSD for the window
        #tr_window = tr.data[N_start:N_start+N_window] * window
        #f, Pxx = periodogram(tr_window,fs=fs,scaling='spectrum')

        psd = MTSpec(tr.data[N_start:N_start+N_window],dt=1/fs)
        f, Pxx = psd.rspec()

        f = f.flatten()
        Pxx = Pxx.flatten()

        jack = psd.jackspec()
        _, ci = psd.rspec(jack)


        Pxx = np.sqrt(Pxx)    
        ci = np.sqrt(ci)   

        f_trim = f[(f>=freqmin)&(f<=freqmax)]
        Pxx_trim = Pxx[(f>=freqmin)&(f<=freqmax)]
        ci_trim = ci[(f>=freqmin)&(f<=freqmax),:]

        sigma = np.abs(ci_trim[:,1] - ci_trim[:,0]) / 4

        f_dict[sta] = f_trim
        Pxx_dict[sta] = Pxx_trim
        ci_dict[sta] = ci_trim
        err_dict[sta] = sigma
    
    return f_dict, Pxx_dict, ci_dict, err_dict


def IncidenceAngle(stream,sta_xy,freqmin=0,freqmax=np.inf):
    filt_stream = stream.copy()
    filt_stream.filter('bandpass',freqmin=freqmin,freqmax=freqmax)
    theta_dict = {}
    theta_sd_dict = {}
    
    for sta, row in sta_xy.iterrows():
        Z = filt_stream.select(station=sta,component='Z')[0]
        N = filt_stream.select(station=sta,component='1')[0]
        E = filt_stream.select(station=sta,component='2')[0]

        fs = Z.stats.sampling_rate

        Zabs = np.abs(Z.data)
        Rabs = np.sqrt(N.data**2 + E.data**2)

        if np.isnan(Zabs).any():
            incidence = np.nan
            in_error = np.nan

        else:
            #now just slice out the P window of interest
            N_window = int((row['P_end'] - row['P_start'])*fs)
            N_start = int(row['P_start']*fs)

            Z_trim = Zabs[N_start:N_start+N_window]
            R_trim = Rabs[N_start:N_start+N_window]

            def fit_func(beta, x):
                return beta[0] * x

            data = scipy.odr.Data(R_trim,Z_trim)
            model = scipy.odr.Model(fit_func)
            odr = scipy.odr.ODR(data, model, beta0=[1.])
            out = odr.run()
            in_slope = out.beta[0]   
            in_error = out.sd_beta[0] 
            incidence = np.arctan2(1.0, in_slope)

            in_error = 1.0 / ((1.0 ** 2 + in_slope ** 2) * incidence) * in_error
       
        theta_dict[sta] = incidence
        theta_sd_dict[sta] = in_error
        
    return theta_dict, theta_sd_dict

def BruneModel(f,omega0,fc,t_star):
    omega = omega0 * np.exp(-np.pi*f*t_star) / (1 + (f/fc)**2)
    return omega

def LogBruneModel(f,log_omega0,fc,t_star):
    omega = np.exp(log_omega0-np.pi*f*t_star) / (1 + (f/fc)**2)
    return omega

def FullLogBruneModel(f,log_omega0,log_fc,t_star):
    fc = np.exp(log_fc)
    omega = np.exp(log_omega0-np.pi*f*t_star) / (1 + (f/fc)**2)
    return omega


def FitBruneModel(f_dict,Pxx_dict,err_dict,arrivals,freqmin,freqmax):
    results = {}
    #TODO try using the log of the corner frequency to force it to stay positive without bounds
    #TODO also is consistant with how we look at the spectra using log of omega0 and fc.
    for key in f_dict.keys():
        f = f_dict[key]
        Pxx = Pxx_dict[key]
        sigma = err_dict[key]

        if np.isnan(Pxx).any():
            results[key] = None
        else:
            t_star_est = (arrivals['R'][key] / 3.87) / 30
            #x0 = [np.log(np.max(Pxx)),2,t_star_est]
            x0 = [np.log(np.max(Pxx)),np.log(2),t_star_est]
            #bounds = (np.array([-np.inf,freqmin,0]),np.array([np.inf,freqmax,np.inf]))
            #bounds = [[0,np.inf]] + [[freqmin,freqmax],[10,200]] * N
            res = curve_fit(FullLogBruneModel,f,Pxx,p0=x0,full_output=True,nan_policy='omit')#,x_scale='jac')#,bounds=bounds)#,sigma=sigma)#,jac=BruneJacobian) #TODO turn bounds back on? Need to try using log(Omega) as input so same scale as other params...
            results[key] = res
    return results

def CurveFit(disp_stream,arrivals,freqmin=0,freqmax=np.inf):
    #first compute the P wave spectrum at each of the stations
    f_dict, Pxx_dict, ci_dict, err_dict = SourceSpectrum(disp_stream,arrivals,freqmin=freqmin,freqmax=freqmax)

    #now fit the spectra at each with the Brune model to get Omega0...
    results = FitBruneModel(f_dict,Pxx_dict,err_dict,arrivals,freqmin,freqmax)
    #unpack this list into the individual parameters
    rows = []
    labels = []
    for sta in f_dict.keys():
        res = results[sta]
        if res == None:
            row = {'logomega0':np.nan,'fc':np.nan,'t_star':np.nan,'domega0':np.nan,'dfc':np.nan,'dt_star':np.nan}
        else:
            x = res[0]
            pcov = res[1]
            dx = np.sqrt(np.diag(pcov))
            log_fc = ufloat(x[1],dx[1])
            fc = exp(log_fc)
            row = {'logomega0':x[0],'fc':fc.nominal_value,'t_star':x[2],'dlogomega0':dx[0],'dfc':fc.std_dev,'dt_star':dx[2]}
        rows.append(row)
        labels.append(sta)
    df = pd.DataFrame(data=rows,index=labels)
    return df


def MomentMagnitude(disp_stream,sta_xy,freqmin=0,freqmax=np.inf):
    output_df = CurveFit(disp_stream,sta_xy,freqmin=freqmin,freqmax=freqmax)
    theta_dict, theta_unc_dict = IncidenceAngle(disp_stream,sta_xy,freqmin=freqmin,freqmax=freqmax)

    rho = ufloat(912,10.0) #TODO find reasonable uncertainty from literature
    v = ufloat(3870,100.0) #TODO find reasonable uncertainty from literature
    A_rad = ufloat(0.44,0.2) #TODO think about best uncertainty estimate
    theta_unc = {}
    r_unc = {}
    logomega0_unc = {}
    M0_unc = {}
    for sta, row in sta_xy.iterrows():
        r_fit = row['r_fit']*1000
        r_unc = row['sR'] * 1000
        theta_unc[sta] = ufloat(theta_dict[sta],theta_unc_dict[sta])
        r_unc[sta] = ufloat(r_fit,r_unc)
        logomega0_unc[sta] = ufloat(output_df['logomega0'][sta],output_df['dlogomega0'][sta])

        M0_unc[sta] = 4*np.pi*rho*v**3*r_unc[sta]*exp(logomega0_unc[sta]) / (A_rad * 2*cos(theta_unc[sta]))

    M0_vals = []
    M0_sd = []
    for key in M0_unc.keys():
        M0_vals.append(M0_unc[key].nominal_value)
        M0_sd.append(M0_unc[key].std_dev)
    M0_vals = np.array(M0_vals)
    M0_sd = np.array(M0_sd)

    M0_ave = np.nansum(M0_vals/(M0_sd**2)) / np.nansum(1/M0_sd**2)
    M0_ave_sd = 1 / np.sqrt(np.nansum(1/M0_sd**2))

    M0 = ufloat(M0_ave,M0_ave_sd)
    Mw = 2/3 * log10(M0) - 6.0
    #? split into two functions - the second is just going from the output dataframe to the moment magnitude (IE the last couple of lines here with unc est from the mean values)
    #TODO return dataframe with full details rather than just the mean Mw, M0 - will be a row in the full dataframe output for all events...
    return Mw, M0


def StationMomentMagnitude(disp_stream,arrivals,freqmin=0,freqmax=np.inf):
    station_mags = CurveFit(disp_stream,arrivals,freqmin=freqmin,freqmax=freqmax)
    theta_dict, theta_unc_dict = IncidenceAngle(disp_stream,arrivals,freqmin=freqmin,freqmax=freqmax)

    rho = ufloat(912,10.0) #TODO find reasonable uncertainty from literature
    v = ufloat(3870,100.0) #TODO find reasonable uncertainty from literature
    A_rad = ufloat(0.44,0.2) #TODO think about best uncertainty estimate

    for sta, row in station_mags.iterrows():
        logomega0 = ufloat(row['logomega0'],row['dlogomega0'])
        r = ufloat(arrivals['R'][sta]*1000,arrivals['dR'][sta]*1000) #! need to work out best way of feeding R uncertainty in - probably through arrivals dataframe...
        theta = ufloat(theta_dict[sta],theta_unc_dict[sta])
        M0 = 4*np.pi*rho*v**3*r*exp(logomega0) / (A_rad * 2*cos(theta))
        Mw = 2/3 * log10(M0) - 6.0

        station_mags.at[sta,'M0'] = M0.nominal_value
        station_mags.at[sta,'dM0'] = M0.std_dev
        station_mags.at[sta,'Mw'] = Mw.nominal_value
        station_mags.at[sta,'dMw'] = Mw.std_dev

    return station_mags


def CombinedMomentMagnitude(station_mags):
    M0_vals = []
    M0_sd = []

    fc_vals = []
    fc_sd = []
    for sta, row in station_mags.iterrows():
        M0_vals.append(row['M0'])
        M0_sd.append(row['dM0'])

        fc_vals.append(row['fc'])
        fc_sd.append(row['dfc'])

    M0_vals = np.array(M0_vals)
    M0_sd = np.array(M0_sd)

    fc_vals = np.array(fc_vals)
    fc_sd = np.array(fc_sd)

    M0_ave = np.nansum(M0_vals/(M0_sd**2)) / np.nansum(1/M0_sd**2)
    M0_ave_sd = 1 / np.sqrt(np.nansum(1/M0_sd**2))

    fc_ave = np.nansum(fc_vals/(fc_sd**2)) / np.nansum(1/fc_sd**2)
    fc_ave_sd = 1 / np.sqrt(np.nansum(1/fc_sd**2))

    fc = ufloat(fc_ave,fc_ave_sd)
    M0 = ufloat(M0_ave,M0_ave_sd)
    Mw = 2/3 * log10(M0) - 6.0

    return M0, Mw, fc


