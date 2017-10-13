import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pt3t3r_to_hdf5 import *
#=======================for plotting time trace of all the files i a list=============
def timetrace_hist_filelist(file_list, bintime, time_lim=(None, None), figsize=(16, 20)):
    '''
    Argument:
    file_list: list of t3r_files as an array
    bintime: bintime of time trace in sec
    '''
    fig = plt.figure(figsize = figsize);
    nrows=len(file_list); ncols=4;
    for i in range(len(file_list)):
        t3rfile = file_list[i];
        t3rfile_name = os.path.basename(t3rfile);
        file_path_hdf5 = t3r_to_hdf5(filename=t3rfile);
        h5 = h5py.File(file_path_hdf5);
        unit = h5['photon_data']['timestamps_specs']['timestamps_unit'][...];
        tcspc_unit = h5['photon_data']['nanotimes_specs']['tcspc_unit'][...];
        t = h5['photon_data']['timestamps'][...][h5['photon_data']['detectors'][...] == 1];
        bins = int((max(t*unit)-min(t*unit))/bintime)
        binned_trace = np.histogram(t*unit, bins=bins);#arrival times to binned trace
        intensity_hist = np.histogram((binned_trace[0][50:]*1e-3)/bintime, bins=200);#binned trace to histogram
        #timetrace plot
        ax00 = plt.subplot2grid((nrows, ncols), (i,0), colspan=3)
        ax00.plot(binned_trace[1][50:-1], binned_trace[0][50:]*1e-3/bintime, 'b', label=t3rfile_name)
        ax00.set_xlim(min(binned_trace[1]), max(binned_trace[1]));
        if time_lim:
            ax00.set_xlim(time_lim)
        ax00.set_xlabel('time/s')
        ax00.set_ylabel('counts/kcps')
        ax00.legend()
        #histogram of intensity
        ax02 = plt.subplot2grid((nrows, ncols),(i, 3))
        ax02.plot(intensity_hist[0], intensity_hist[1][:-1], 'b');
        ymin, ymax = ax00.get_ylim()
        ax02.set_ylim((ymin), (ymax))
        ax02.set_xscale('log')
        ax02.set_xticks([])
        ax02.set_yticks([])
    ax02.set_xlabel('PDF')
    return
def nanotime_intphoton_filelist(file_list, bins_nanotime, bins_intphoton,
                             nanotime_lim, intphotn_lim, figsize=(20, 5)):
    '''
    Arguments:

    Returns:
    '''
    fig = plt.figure(figsize = figsize);
    cmap = plt.get_cmap('hsv')#jet_r
    N=len(file_list)
    nrows=1;ncols=2;
    ax00=plt.subplot2grid((nrows, ncols), (0,0));
    ax01=plt.subplot2grid((nrows, ncols), (0,1));
    for i in range(len(file_list)):
        color = cmap(float(i)/N)
        t3rfile = file_list[i];
        t3rfile_name = os.path.basename(t3rfile);
        file_path_hdf5 = t3r_to_hdf5(filename=t3rfile);
        h5 = h5py.File(file_path_hdf5);
        unit = h5['photon_data']['timestamps_specs']['timestamps_unit'][...];
        tcspc_unit = h5['photon_data']['nanotimes_specs']['tcspc_unit'][...];
        t = h5['photon_data']['timestamps'][...][h5['photon_data']['detectors'][...] == 1];
        nanotimes = h5['photon_data']['nanotimes'][...][h5['photon_data']['detectors'][...] == 1];
        #=======lifetime plot=======
        nanotimes = 1e9*nanotimes*tcspc_unit;#convert to ns
        nanotimes = max(nanotimes)-nanotimes
        hist, bin_edges = np.histogram(nanotimes,bins=bins_nanotime)
        ax00.plot(bin_edges[4:-1], hist[4:]/max(hist), color=color, label=t3rfile_name)
        ax00.set_yscale('log');
        ax00.set_xlim(nanotime_lim)
        ax00.legend()
        #=======interphoton plot========
        t = t*unit#convert to sec
        int_photon = np.diff(t);
        hist, bin_edges = np.histogram(int_photon, bins=bins_intphoton)
        ax01.plot(bin_edges[:-1], hist, color=color, label=t3rfile_name)#hist/max(hist)
        ax01.set_yscale('log');
        ax01.set_xscale('log');
        ax01.set_xlim(intphotn_lim);
        ax01.set_xlabel
        ax01.legend()
    return