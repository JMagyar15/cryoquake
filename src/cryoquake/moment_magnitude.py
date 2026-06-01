import numpy as np
import scipy
import pandas as pd
from multitaper.mtspec import MTSpec
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import cos, log10, exp


def Misfit2Arrivals(gamma_xr,Vs=1900):
    #takes the misfit surface Dataset and converts it to arrival windows for each station for computing moment magnitude
    sta_lst = gamma_xr.attrs['stations'] #get the stations that were included in the inversion
    t0 = gamma_xr.attrs['t0']

    rows = []
    for sta_code in sta_lst:
        t1 = gamma_xr.attrs['t_'+sta_code] + t0
        t2 = (gamma_xr.attrs['D_'+sta_code] / Vs) + t0

        r = gamma_xr.attrs['D_'+sta_code]
        dr = gamma_xr.attrs['dD_'+sta_code]

        row = {'P_start':t1,'P_end':t2,'P_window':t2-t1,'R':r,'dR':dr}
        rows.append(row)
    arrivals = pd.DataFrame(data=rows,index=sta_lst)
    return arrivals



def SourceSpectrum(stream,arrivals,freqmin=0,freqmax=np.inf):
    f_dict = {}
    Pxx_dict = {}
    ci_dict = {}
    err_dict = {}

    for sta, row in arrivals.iterrows():
        tr = stream.select(station=sta,component='Z')[0]
        fs = tr.stats.sampling_rate
        N_window = int((row['P_end'] - row['P_start'])*fs)
        N_start = int(row['P_start']*fs)

        #now compute the PSD for the window

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

            #below code is based off obspy particle motion polarization code https://github.com/obspy/obspy/blob/master/obspy/signal/polarization.py

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


def FitBruneModel(f_dict,Pxx_dict,arrivals):
    results = {}
 
    for key in f_dict.keys():
        f = f_dict[key]
        Pxx = Pxx_dict[key]

        if np.isnan(Pxx).any():
            results[key] = None
        else:
            t_star_est = (arrivals['R'][key] / 3870) / 30
            x0 = [np.log(np.max(Pxx)),np.log(2),t_star_est]
            res = curve_fit(FullLogBruneModel,f,Pxx,p0=x0,full_output=True,nan_policy='omit')#,x_scale='jac')#,bounds=bounds)#,sigma=sigma)#,jac=BruneJacobian) 
            results[key] = res
    return results

def CurveFit(disp_stream,arrivals,freqmin=0,freqmax=np.inf):
    #first compute the P wave spectrum at each of the stations
    f_dict, Pxx_dict, _, _ = SourceSpectrum(disp_stream,arrivals,freqmin=freqmin,freqmax=freqmax)

    #now fit the spectra at each with the Brune model to get Omega0...
    results = FitBruneModel(f_dict,Pxx_dict,arrivals)
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



def StationMomentMagnitude(disp_stream,arrivals,freqmin=0,freqmax=np.inf,rho=912.0,drho=10.0,v=3870.0,dv=100,A_rad=0.44,dA_rad=0.2):
    station_mags = CurveFit(disp_stream,arrivals,freqmin=freqmin,freqmax=freqmax)
    theta_dict, theta_unc_dict = IncidenceAngle(disp_stream,arrivals,freqmin=freqmin,freqmax=freqmax)

    u_rho = ufloat(rho,drho) 
    u_v = ufloat(v,dv) 
    u_A_rad = ufloat(A_rad,dA_rad) 

    for sta, row in station_mags.iterrows():
        logomega0 = ufloat(row['logomega0'],row['dlogomega0'])
        r = ufloat(arrivals['R'][sta],arrivals['dR'][sta]) #! need to work out best way of feeding R uncertainty in - probably through arrivals dataframe...
        theta = ufloat(theta_dict[sta],theta_unc_dict[sta])
        M0 = 4*np.pi*u_rho*u_v**3*r*exp(logomega0) / (u_A_rad * 2*cos(theta))
        Mw = 2/3 * log10(M0) - 6.0

        station_mags.at[sta,'M0'] = M0.nominal_value
        station_mags.at[sta,'dM0'] = M0.std_dev
        station_mags.at[sta,'Mw'] = Mw.nominal_value
        station_mags.at[sta,'dMw'] = Mw.std_dev

    return station_mags


def CombinedMomentMagnitude(station_mags):
    M0_vals = []
    M0_sd = []

    for sta, row in station_mags.iterrows():
        M0_vals.append(row['M0'])
        M0_sd.append(row['dM0'])

    M0_vals = np.array(M0_vals)
    M0_sd = np.array(M0_sd)

    M0_ave = np.nansum(M0_vals/(M0_sd**2)) / np.nansum(1/M0_sd**2)
    M0_ave_sd = 1 / np.sqrt(np.nansum(1/M0_sd**2))

    M0 = ufloat(M0_ave,M0_ave_sd)
    Mw = 2/3 * log10(M0) - 6.0

    return M0, Mw


