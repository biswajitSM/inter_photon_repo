import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pt3t3r_to_hdf5 import *
from pycorrelate import *
def nanotime_filter(t3rfile, nanotime_lim=(0, 10), nanotime_cor=10,
                    filter_range=[3, 6], bintime=1e-3, time_lim=(None, None),
                    plotting=True, figsize=(20,5)):
    '''
    Filter time traces based on lifetime
    '''
    t3rfile_name = os.path.basename(t3rfile);
    file_path_hdf5 = t3r_to_hdf5(filename=t3rfile);
    h5 = h5py.File(file_path_hdf5);
    unit = h5['photon_data']['timestamps_specs']['timestamps_unit'][...];
    tcspc_unit = h5['photon_data']['nanotimes_specs']['tcspc_unit'][...];
    tcspc_num_bins = h5['photon_data']['nanotimes_specs']['tcspc_num_bins'][...]
    t = h5['photon_data']['timestamps'][...][h5['photon_data']['detectors'][...] == 1];
    nanotimes = h5['photon_data']['nanotimes'][...][h5['photon_data']['detectors'][...] == 1];
    #=======lifetime plot=======
    nanotimes = 1e9*nanotimes*tcspc_unit;#convert to ns
    nanotimes = (max(nanotimes)-nanotimes)-nanotime_cor
    if plotting:
        fig = plt.figure(figsize = figsize);
        nrows=2;ncols=2;
        ax00=plt.subplot2grid((nrows, ncols), (0,0));
        ax01=plt.subplot2grid((nrows, ncols), (0,1));
        ax10 = plt.subplot2grid((nrows,ncols), (1,0))
        ax11=plt.subplot2grid((nrows, ncols), (1,1));
        #lifetime plot
        hist, bin_edges = np.histogram(nanotimes,bins=tcspc_num_bins)
        ax00.plot(bin_edges[4:-1], hist[4:], '.',label=t3rfile_name);
        ax00.axvline(filter_range[0], color='b', lw=2, label=filter_range[0])
        ax00.axvline(filter_range[1], color='r', lw=2, label=filter_range[1])
        ax00.set_yscale('log');
        ax00.set_xlim(nanotime_lim)
        ax00.set_xlabel('lifetime/ns')
        ax00.set_ylabel('#')
        ax00.legend()
        #time trace plot
        mask = np.logical_and(nanotimes>filter_range[0],
                              nanotimes<filter_range[1]);
        nanotime_filtered = nanotimes[mask];
        timestamps_filtered = unit*t[mask];
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
        save_folder = os.path.join(folder_temp, t3rfile_name, 'nanotimefiltering');
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        savename = t3rfile_name + str(filter_range[0])+'_'+str(filter_range[1]) +'.png';
        savename = os.path.join(save_folder, savename)
        fig.savefig(savename, dpi=300)
        # plt.close()
    return timestamps_filtered