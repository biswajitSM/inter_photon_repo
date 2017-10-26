import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pt3t3r_to_hdf5 import *
from pycorrelate import *
#===================================================================================================
#==============================NANOTIME FILTERING================================================
#===================================================================================================
def nanotime_filter(hdf5file, nanotime_lim=(0, 10), nanotime_cor=10,
                    filter_range=[3, 6], bintime=1e-3, time_lim=(None, None),
                    plotting=True, figsize=(20,5), closefig=True):
    '''
    Filter time traces based on lifetime
    '''
    hdf5file_name = os.path.basename(hdf5file);
    hdf5file_dirpath = os.path.dirname(hdf5file);
    hdf5file_dirname = os.path.basename(hdf5file_dirpath);
    print(hdf5file_dirname)
    file_path_hdf5 = os.path.abspath(hdf5file);
    h5 = h5py.File(file_path_hdf5);
    unit = h5['photon_data']['timestamps_specs']['timestamps_unit'][...];
    tcspc_unit = h5['photon_data']['nanotimes_specs']['tcspc_unit'][...];
    tcspc_num_bins = h5['photon_data']['nanotimes_specs']['tcspc_num_bins'][...]
    timestamps = unit* h5['photon_data']['timestamps'][...][h5['photon_data']['detectors'][...] == 1];
    nanotimes = h5['photon_data']['nanotimes'][...][h5['photon_data']['detectors'][...] == 1];
    #=======filtering/selecting photons=======
    nanotimes = 1e9*nanotimes*tcspc_unit;#convert to ns
    nanotimes = (max(nanotimes)-nanotimes)-nanotime_cor
    mask = np.logical_and(nanotimes>filter_range[0],
                          nanotimes<filter_range[1]);
    timestamps_filtered = timestamps[mask];
    timestamps_outside = timestamps[~mask]
    frac_filtered = 100 * len(timestamps_filtered)/len(timestamps)#in percentage
    frac_filtered = np.round(frac_filtered);
    print('fraction of photon filtered %.f%%' %frac_filtered)
    if plotting:
        fig = plt.figure(figsize = figsize);
        nrows=2;ncols=2;
        ax00=plt.subplot2grid((nrows, ncols), (0,0));
        ax01=plt.subplot2grid((nrows, ncols), (0,1));
        ax10 = plt.subplot2grid((nrows,ncols), (1,0))
        ax11=plt.subplot2grid((nrows, ncols), (1,1));
        color_filt='r'; color_out='b'
        fig.suptitle(hdf5file_name + ', fration filtered:' + str(frac_filtered), 
                    fontsize=14, fontweight='bold')

        #==============lifetime plot==================
        hist_nano, trace_nano = np.histogram(nanotimes,bins=tcspc_num_bins)
        ax00.plot(trace_nano[4:-1], hist_nano[4:], '.');
        ax00.axvline(filter_range[0], color='b', lw=2);
        ax00.axvline(filter_range[1], color='r', lw=2);
        ax00.axvspan(filter_range[0], filter_range[1], color=color_filt, alpha=0.5, lw=0)
        ax00.axvspan(min(trace_nano), filter_range[0], color=color_out, alpha=0.3, lw=0)
        ax00.axvspan(filter_range[1], max(trace_nano), color=color_out, alpha=0.3, lw=0)
        ax00.set_yscale('log');
        ax00.set_xlim(nanotime_lim)
        ax00.set_xlabel('lifetime/ns')
        ax00.set_ylabel('#')
        ax00.legend()
        ax00.set_title('nanotime histogram, filter range'+str(filter_range))
        #===================time trace plot================
        bins_tt = int((max(timestamps)-min(timestamps))/bintime);
        bins_tt = np.linspace(min(timestamps), max(timestamps), bins_tt)
        hist_filt, trace_filt = np.histogram(timestamps_filtered, bins=bins_tt);#arrival times to binned trace
        trace_filt = trace_filt[50:-1]; hist_filt = hist_filt[50:]*1e-3/bintime;
        ax01.plot(trace_filt, hist_filt, color=color_filt, label='filtered');
        ax01.axhline(np.mean(hist_filt), color='y', lw=2, label=np.mean(hist_filt))
        ax01.axhline(np.max(hist_filt), color='r', lw=2, label=np.max(hist_filt))
        ax01.tick_params('y', colors=color_filt)
        ax01.set_ylim(0, max(hist_filt))

        ax01_r = ax01.twinx()
        hist_outs, trace_outs = np.histogram(timestamps_outside, bins=bins_tt);#outside the filtering range
        trace_outs = trace_outs[50:-1]; hist_outs = hist_outs[50:]*1e-3/bintime;
        ax01_r.plot(trace_outs, hist_outs, color=color_out, alpha=0.3, label='leftout')
        ax01_r.tick_params('y', colors=color_out)
        ax01_r.set_ylim(0, max(hist_filt))

        ax01.set_xlim(min(trace_filt), max(trace_filt));        
        if time_lim:
            ax01.set_xlim(time_lim)
        ax01.set_xlabel('time/s')
        ax01.set_ylabel('counts/kcps')
        ax01.legend(loc=1); ax01_r.legend(loc=2)
        ax01.set_title('time trace')
        # ===========histogram intensity=============
        if max(hist_filt) > max(hist_outs):
            int_max = max(hist_filt)
        else:
            int_max = max(hist_outs)
        bins_int = np.linspace(0, int_max, 200)
        int_hist_filt, int_trace = np.histogram(hist_filt, bins=bins_int);#binned trace to histogram
        ax11.plot(int_trace[:-1], int_hist_filt, '*--', color=color_filt, label='filtered');
        ax11.set_yscale('log')
        ax11.tick_params('y', colors=color_filt)

        ax11_r = ax11.twinx()
        int_hist_outs, int_trace = np.histogram(hist_outs, bins=bins_int);#binned trace to histogram
        ax11_r.plot(int_trace[:-1], int_hist_outs, '*--', color=color_out, label='leftout');
        ax11_r.tick_params('y', colors=color_out)
        ax11_r.set_yscale('log')
        ax11.set_xlim(0, None)
        # ax11.set_xscale('log')
        ax11.set_xlabel('counts/kcps')
        ax11.set_ylabel('#')
        ax11.legend(loc=1); ax11_r.legend(loc=7)
        # ============Autocorrelation=============
        bins_lag = make_loglags(-6, 2, 10);
        Gn_filt = normalize_G(timestamps_filtered, timestamps_filtered, bins_lag)
        Gn_filt = np.hstack((Gn_filt[:1], Gn_filt))-1        
        ax10.plot(bins_lag, Gn_filt, drawstyle='steps-pre', color=color_filt, label='filtered')
        ax10.set_xscale('log')
        ax10.tick_params('y', colors=color_filt)

        ax10_r = ax10.twinx();
        Gn_outs = normalize_G(timestamps_outside, timestamps_outside, bins_lag)
        Gn_outs = np.hstack((Gn_outs[:1], Gn_outs))-1        
        ax10_r.plot(bins_lag, Gn_outs, drawstyle='steps-pre', color=color_out, label='leftout')
        ax10_r.set_xscale('log')
        ax10_r.tick_params('y', colors=color_out)

        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('G(t)-1')
        ax10.grid(True); 
        ax10.grid(True, which='minor', lw=0.5)
        ax10.set_xlim(1e-6, 1e0)
        ax10.legend(loc=1); ax10_r.legend(loc=2)        
        #============save the figure===============
        import datetime
        savetofolder = hdf5file_dirpath;
        date = datetime.datetime.today().strftime('%Y%m%d%H%M%S');
        onlydate = datetime.datetime.today().strftime('%Y%m%d');
        save_folder = os.path.join(savetofolder, hdf5file_name[:-5], onlydate, 'nanotimefiltering');
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        savename = date +'_'+ hdf5file_name +'.png';
        savename = os.path.join(save_folder, savename)
        fig.savefig(savename, dpi=300)
        if closefig:
            plt.close(fig)
    return int_trace, int_hist_filt, int_hist_outs, bins_lag, Gn_filt, Gn_outs
#=======over many nano-filtering values========
def nanotime_filter_range(hdf5file, nanotime_cor=10, inums=1, jnums=1,
                         bintime=0.1e-3, time_lim=(0, 5)):
    ''''''
    hdf5file_name = os.path.basename(hdf5file);
    hdf5file_dirpath = os.path.dirname(hdf5file);
    hdf5file_dirname = os.path.basename(hdf5file_dirpath);
    file_path_hdf5 = os.path.abspath(hdf5file);
    import datetime
    savetofolder = hdf5file_dirpath;
    date = datetime.datetime.today().strftime('%Y%m%d%H');
    onlydate = datetime.datetime.today().strftime('%Y%m%d');
    save_folder = os.path.join(savetofolder, hdf5file_name[:-5], onlydate, 'nanotimefiltering');
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    hdf5_tosave = date +'_'+ hdf5file_name[:-5] + 'nanofilt' +'.hdf5';
    hdf5_tosave = os.path.join(save_folder, hdf5_tosave)
    f_saveHDF5 = h5py.File(hdf5_tosave, "w");#create or open hdf5 file in write mode
    # ===save identity data====
    f_saveHDF5['filename'] = hdf5file_name;
    f_saveHDF5['filedir'] = hdf5file_dirname;
    f_saveHDF5['filepath'] = file_path_hdf5;
    f_saveHDF5['bintime'] = bintime;
    #======determining range====
    h5 = h5py.File(file_path_hdf5);
    tcspc_unit = h5['photon_data']['nanotimes_specs']['tcspc_unit'][...];
    tcspc_num_bins = h5['photon_data']['nanotimes_specs']['tcspc_num_bins'][...]
    nanotimes = 1e9 * tcspc_unit * h5['photon_data']['nanotimes'][...][h5['photon_data']['detectors'][...] == 1];
    nanotimes = (max(nanotimes)-nanotimes) - nanotime_cor
    hist_nano, trace_nano = np.histogram(nanotimes,bins=tcspc_num_bins)
    trace_nano = trace_nano[:-1];
    nanomax = trace_nano[hist_nano == max(hist_nano)][...];
    irange = np.linspace(nanomax-1, nanomax+1, inums);
    jrange = np.linspace(nanomax, nanomax+5, jnums);
    for i in irange:
        grp_i = f_saveHDF5.create_group('low_nanosec_'+str(np.round(i, 1)));        
        for j in jrange:
            if i < j:
                grp_j = grp_i.create_group('high_nanosec_'+str(np.round(j, 1)));
                # print(i, j)
                output = nanotime_filter(hdf5file, nanotime_lim=(0, 10), nanotime_cor=nanotime_cor,
                        filter_range=[i, j], bintime=bintime, time_lim=time_lim,
                        plotting=True, figsize=(10,6));
                [int_trace, int_hist_filt, int_hist_outs, bins_lag, Gn_filt, Gn_outs] = output
                # save outputs to hdf5 file
                grp_j['nanolow'] = i;
                grp_j['nanohigh'] = j;
                grp_j.create_dataset('int_trace', data=int_trace);
                grp_j.create_dataset('int_hist_filt', data=int_hist_filt);
                grp_j.create_dataset('int_hist_outs', data=int_hist_outs);
                grp_j.create_dataset('bins_lag', data=bins_lag);
                grp_j.create_dataset('Gn_filt', data=Gn_filt);
                grp_j.create_dataset('Gn_outs', data=Gn_outs);
                f_saveHDF5.flush()
    return hdf5_tosave
#===================================================================================================
#==============================INTERPHOTON FILTERING================================================
#===================================================================================================
def intphoton_filter(hdf5file, intphotn_lim=(0, 3e-4), bins_intphoton=300,
                    filter_val=1e-5, bintime=1e-3, time_lim=(None, None),
                    plotting=True, fitting=False, figsize=(20,5), closefig=True):
    '''
    Photons were filtered out based on interphoton delay
    '''
    hdf5file_name = os.path.basename(hdf5file);
    hdf5file_dirpath = os.path.dirname(hdf5file);
    hdf5file_dirname = os.path.basename(hdf5file_dirpath);
    print(hdf5file_dirname)
    file_path_hdf5 = os.path.abspath(hdf5file);
    h5 = h5py.File(file_path_hdf5);
    unit = h5['photon_data']['timestamps_specs']['timestamps_unit'][...];
    timestamps = h5['photon_data']['timestamps'][...][h5['photon_data']['detectors'][...] == 1];
    timestamps = unit*timestamps;
    #interphoton calculation and asignment to photon
    intphoton = np.diff(timestamps);
    timestamps = timestamps[:-1];
    # intphoton = 0.5 * (intphoton[:-1] + intphoton[1:])#avergae between previous and later
    # timestamps = timestamps[1:-1];
    #filtering photons
    mask = intphoton < filter_val;
    timestamps_filtered = timestamps[mask];
    timestamps_outside = timestamps[~mask]
    frac_filtered = 100 * len(timestamps_filtered)/len(timestamps)#in percentage
    frac_filtered = np.round(frac_filtered);
    print('fraction of photon filtered %.f%%' %frac_filtered)
    if plotting:
        #figure parameters
        fig = plt.figure(figsize = figsize);
        nrows=2;ncols=2;
        ax00=plt.subplot2grid((nrows, ncols), (0,0));
        ax01=plt.subplot2grid((nrows, ncols), (0,1));
        ax10 = plt.subplot2grid((nrows,ncols), (1,0))
        ax11=plt.subplot2grid((nrows, ncols), (1,1));
        color_filt='r'; color_out='b'
        fig.suptitle(hdf5file_name + ', fration filtered:' + str(frac_filtered)+'%', 
                    fontsize=14, fontweight='bold')
        # plot interphoton hist
        hist_intph, trace_intph = np.histogram(intphoton, bins=bins_intphoton)
        x=trace_intph[:-1]; y = hist_intph;
        ax00.plot(x, y, '.', label=hdf5file_name)#hist/max(hist);
        ax00.axvline(filter_val, color='r', lw=2, label=str(filter_val)+' s' )
        ax00.axvspan(0, filter_val, color=color_filt, alpha=0.3, lw=0)
        ax00.axvspan(filter_val, max(trace_intph), color=color_out, alpha=0.3, lw=0)
        if fitting:
            from lmfit import  Model
            expmodel = Model(mono_exp);
            biexpmodel = Model(bi_exp)
            result = expmodel.fit(y, x=x, A1=1e4, t1=1e-4)
            ax00.plot(x, result.best_fit, 'r-', label='mono exp')
            result = biexpmodel.fit(y, x=x, A1=1e4, t1=1e-5, A2=1e3, t2=1e-4)
            # print(result.fit_report())
            ax00.plot(x, result.best_fit, 'b-',label='bi exp')
        ax00.set_xscale('log');
        ax00.set_yscale('log');
        ax00.set_xlim(intphotn_lim);
        ax00.set_xticks(np.linspace(min(x), intphotn_lim[1], 4))
        # ax00.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax00.set_ylim(1e0, None)
        ax00.set_xlabel('interphoton time/s')
        ax00.legend()
        ax00.set_title('interphoton histogram, filter range '+str([0, filter_val]))
        #============filtered time traces===========
        bins_tt = int((max(timestamps)-min(timestamps))/bintime);
        bins_tt = np.linspace(min(timestamps), max(timestamps), bins_tt)
        hist_filt, trace_filt = np.histogram(timestamps_filtered, bins=bins_tt);#arrival times to binned trace
        trace_filt = trace_filt[50:-1]; hist_filt = hist_filt[50:]*1e-3/bintime;
        ax01.plot(trace_filt, hist_filt, color=color_filt, label='filtered');
        ax01.axhline(np.mean(hist_filt), color='y', lw=2, label=np.mean(hist_filt))
        ax01.axhline(np.max(hist_filt), color='r', lw=2, label=np.max(hist_filt))
        ax01.tick_params('y', colors=color_filt)
        ax01.set_ylim(0, max(hist_filt))

        ax01_r = ax01.twinx()
        hist_outs, trace_outs = np.histogram(timestamps_outside, bins=bins_tt);#outside the filtering range
        trace_outs = trace_outs[50:-1]; hist_outs = hist_outs[50:]*1e-3/bintime;
        ax01_r.plot(trace_outs, hist_outs, color=color_out, alpha=0.3, label='leftout')
        ax01_r.tick_params('y', colors=color_out)
        # ax01_r.set_ylim(0, max(hist_filt))

        ax01.set_xlim(min(trace_filt), max(trace_filt));        
        if time_lim:
            ax01.set_xlim(time_lim)
        ax01.set_xlabel('time/s')
        ax01.set_ylabel('counts/kcps')
        ax01.legend(loc=1); ax01_r.legend(loc=2)
        ax01.set_title('time trace')
        # ===========histogram intensity=============
        if max(hist_filt) > max(hist_outs):
            int_max = max(hist_filt)
        else:
            int_max = max(hist_outs)
        bins_int = np.linspace(0, int_max, 200)
        int_hist_filt, int_trace = np.histogram(hist_filt, bins=bins_int);#binned trace to histogram
        ax11.plot(int_trace[:-1], int_hist_filt, '*--', color=color_filt, label='filtered');
        ax11.set_yscale('log')
        ax11.tick_params('y', colors=color_filt)

        ax11_r = ax11.twinx()
        int_hist_outs, int_trace = np.histogram(hist_outs, bins=bins_int);#binned trace to histogram
        ax11_r.plot(int_trace[:-1], int_hist_outs, '*--', color=color_out, label='leftout');
        ax11_r.tick_params('y', colors=color_out)
        ax11_r.set_yscale('log')
        ax11.set_xlim(0, None)
        # ax11.set_xscale('log')
        ax11.set_xlabel('counts/kcps')
        ax11.set_ylabel('#')
        ax11.legend(loc=1); ax11_r.legend(loc=7)
        # ============Autocorrelation=============
        bins_lag = make_loglags(-7, 2, 10);
        Gn_filt = normalize_G(timestamps_filtered, timestamps_filtered, bins_lag)
        Gn_filt = np.hstack((Gn_filt[:1], Gn_filt))-1
        ax10.plot(bins_lag, Gn_filt, drawstyle='steps-pre', color=color_filt, label='filtered')
        ax10.set_xscale('log')
        ax10.tick_params('y', colors=color_filt)

        ax10_r = ax10.twinx();
        Gn_outs = normalize_G(timestamps_outside, timestamps_outside, bins_lag)
        Gn_outs = np.hstack((Gn_outs[:1], Gn_outs))-1
        ax10_r.plot(bins_lag, Gn_outs, drawstyle='steps-pre', color=color_out, label='leftout')
        ax10_r.set_xscale('log')
        ax10_r.tick_params('y', colors=color_out)

        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('G(t)-1')
        ax10.grid(True); 
        ax10.grid(True, which='minor', lw=0.5)
        ax10.set_xlim(1e-6, 1e0)
        ax10.legend(loc=1); ax10_r.legend(loc=2)        
        #===========save the figure===============
        import datetime
        savetofolder = hdf5file_dirpath;
        date = datetime.datetime.today().strftime('%Y%m%d%H%M%S');
        onlydate = datetime.datetime.today().strftime('%Y%m%d');
        save_folder = os.path.join(savetofolder, hdf5file_name[:-5], onlydate, 'intphotonfiltering');
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        savename = date +'_'+ hdf5file_name +'.png';
        savename = os.path.join(save_folder, savename)
        fig.savefig(savename, dpi=300)
        if closefig:
            plt.close(fig)
    return int_trace, int_hist_filt, int_hist_outs, bins_lag, Gn_filt, Gn_outs
#=======over many interphoton-filtering values========
def intphoton_filter_range(hdf5file, intphrange=[-7, -4], inums=1, bintime=0.1e-3, time_lim=(0, 5)):
    ''''''
    hdf5file_name = os.path.basename(hdf5file);
    hdf5file_dirpath = os.path.dirname(hdf5file);
    hdf5file_dirname = os.path.basename(hdf5file_dirpath);
    file_path_hdf5 = os.path.abspath(hdf5file);
    import datetime
    savetofolder = hdf5file_dirpath;
    date = datetime.datetime.today().strftime('%Y%m%d%H');
    onlydate = datetime.datetime.today().strftime('%Y%m%d');
    save_folder = os.path.join(savetofolder, hdf5file_name[:-5], onlydate, 'intphotonfiltering');
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    hdf5_tosave = date +'_'+ hdf5file_name[:-5] + 'intphfilt' +'.hdf5';
    hdf5_tosave = os.path.join(save_folder, hdf5_tosave)
    f_saveHDF5 = h5py.File(hdf5_tosave, "w");#create or open hdf5 file in write mode
    # ===save identity data====
    f_saveHDF5['filename'] = hdf5file_name;
    f_saveHDF5['filedir'] = hdf5file_dirname;
    f_saveHDF5['filepath'] = file_path_hdf5;
    f_saveHDF5['bintime'] = bintime;
    #======determining range====
    filter_vals = np.logspace(intphrange[0], intphrange[1], inums);
    for i in filter_vals:
        grp_i = f_saveHDF5.create_group('intph_cutval_'+str(i));
        output = intphoton_filter(hdf5file, intphotn_lim=(0, 3e-4), bins_intphoton=1000,
                        filter_val=i, bintime=bintime, time_lim=time_lim,
                        plotting=True, fitting=False, figsize=(10,6))
        # save outputs to hdf5 file
        [int_trace, int_hist_filt, int_hist_outs, bins_lag, Gn_filt, Gn_outs] = output;
        grp_i['filter_val'] = i;
        grp_i.create_dataset('int_trace', data=int_trace);
        grp_i.create_dataset('int_hist_filt', data=int_hist_filt);
        grp_i.create_dataset('int_hist_outs', data=int_hist_outs);
        grp_i.create_dataset('bins_lag', data=bins_lag);
        grp_i.create_dataset('Gn_filt', data=Gn_filt);
        grp_i.create_dataset('Gn_outs', data=Gn_outs);
        f_saveHDF5.flush()
    return hdf5_tosave