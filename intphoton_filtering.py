import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pt3t3r_to_hdf5 import *
from pycorrelate import *
def intphoton_filter(t3rfile, intphotn_lim=(0, 3e-4), bins_intphoton=300,
                    filter_val=1e-5, bintime=1e-3, time_lim=(None, None),
                    plotting=True, fitting=False, figsize=(20,5)):
    '''
    Photons were filtered out based on interphoton delay
    '''
    t3rfile_name = os.path.basename(t3rfile);
    file_path_hdf5 = t3r_to_hdf5(filename=t3rfile);
    h5 = h5py.File(file_path_hdf5);
    unit = h5['photon_data']['timestamps_specs']['timestamps_unit'][...];
    timestamps = h5['photon_data']['timestamps'][...][h5['photon_data']['detectors'][...] == 1];
    timestamps = unit*timestamps;
    #interphoton calculation and asignment to photon
    intphoton = np.diff(timestamps);
    timestamps = timestamps[1:];
    #figure parameters
    fig = plt.figure(figsize = figsize);
    nrows=2;ncols=2;
    ax00=plt.subplot2grid((nrows, ncols), (0,0));
    ax01=plt.subplot2grid((nrows, ncols), (0,1));
    ax10 = plt.subplot2grid((nrows,ncols), (1,0))
    ax11=plt.subplot2grid((nrows, ncols), (1,1));
    # plot interphoton hist
    hist, bin_edges = np.histogram(intphoton, bins=bins_intphoton)
    x=bin_edges[:-1]; y = hist;
    ax00.plot(x, y, '.', label=t3rfile_name)#hist/max(hist);
    ax00.axvline(filter_val, color='r', lw=2, label=str(filter_val)+' s' )
    if fitting:
        from lmfit import  Model
        expmodel = Model(mono_exp);
        biexpmodel = Model(bi_exp)
        result = expmodel.fit(y, x=x, A1=1e4, t1=1e-4)
        ax00.plot(x, result.best_fit, 'r-', label='mono exp')
        result = biexpmodel.fit(y, x=x, A1=1e4, t1=1e-5, A2=1e3, t2=1e-4)
        # print(result.fit_report())
        ax00.plot(x, result.best_fit, 'b-',label='bi exp')
    ax00.set_yscale('log');
    ax00.set_xlim(intphotn_lim);
    ax00.set_xticks(np.linspace(min(x), intphotn_lim[1], 4))
    ax00.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax00.set_ylim(1e0, None)
    ax00.set_xlabel('interphoton time/s')
    ax00.legend()
    #filtered time trace
    mask = intphoton < filter_val;
    timestamps_filtered = timestamps[mask];
    bins = int((max(timestamps_filtered)-min(timestamps_filtered))/bintime)
    hist, trace = np.histogram(timestamps_filtered, bins=bins);#arrival times to binned trace
    trace = trace[50:-1]; hist = hist[50:]*1e-3/bintime;
    ax01.plot(trace, hist, 'b', label=t3rfile_name);
    ax01.axhline(np.mean(hist), color='y', lw=2, label=np.mean(hist))
    ax01.axhline(np.max(hist), color='r', lw=2, label=np.max(hist))
    ax01.set_xlim(min(trace), max(trace));
    if time_lim:
        ax01.set_xlim(time_lim)
    ax01.set_xlabel('time/s')
    ax01.set_ylabel('counts/kcps')
    ax01.legend();
    # histogram intensity
    intensity_hist = np.histogram(hist, bins=200);#binned trace to histogram
    ax11.plot(intensity_hist[1][:-1], intensity_hist[0], 'b*--');
    ax11.set_xlim(0, None)
    ax11.set_yscale('log')
    ax11.set_xlabel('counts/kcps')
    ax11.set_ylabel('#')
    # Autocorrelation
    bins = make_loglags(-6, 2, 10);
    Gn = normalize_G(timestamps_filtered, timestamps_filtered, bins)
    ax10.plot(bins, np.hstack((Gn[:1], Gn))-1, drawstyle='steps-pre')
    ax10.set_xlabel('Time (s)')
    ax10.set_ylabel('G(t)-1')
    ax10.grid(True); 
    ax10.grid(True, which='minor', lw=0.5)
    ax10.set_xlim(1e-6, 1e0)
    ax10.set_xscale('log')
    
    #save the figure
    folder_temp = '/home/biswajit/Downloads/temp/';
    save_folder = os.path.join(folder_temp, t3rfile_name, 'intphotonfiltering');
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    savename = t3rfile_name + str(filter_val) +'s.png';
    savename = os.path.join(save_folder, savename)
    fig.savefig(savename, dpi=300)
    plt.close()
    return