'''

'''

import numpy as np
import xarray as xr
import multiprocessing as mp
import itertools
import math
import time#Just for speed testing
from warnings import warn
from scipy.stats import gamma as sp_gamma
default_colors =  ['cornflowerblue','indianred','orange','darkblue','darkgreen','gold', 'brown']


class hsmm:
    
    def __init__(self, data, sfreq, cpus=1, bump_width=50, shape=2, estimate_magnitudes=True, estimate_parameters=True, min_stage_duration=None):
        '''
        HSMM calculates the probability of data summing over all ways of 
        placing the n bumps to break the trial into n + 1 flats.

        Parameters
        ----------
        data : ndarray
            2D ndarray with n_samples * components 
        sfreq : int
            Sampling frequency of the signal (initially 100)
        bump_width : int
            width of bumps in milliseconds, originally 5 samples
        min_stage_duration : float
            Minimum stage duration in milliseconds. Note that this parameter doesn't apply to first and last stage
        '''
        self.sfreq = sfreq
        self.steps = 1000/self.sfreq
        self.bump_width = bump_width
        self.bump_width_samples = int(self.bump_width / self.steps)
        if min_stage_duration == None:
            # print(f'Setting minimum stage duration parameter to bump width ({self.bump_width} ms) as min_stage_duration is unspecified')
            self.min_stage_duration = 1
        else:
            self.min_stage_duration = min_stage_duration/self.steps

        durations = data.unstack().sel(component=0).swap_dims({'epochs':'trials'})\
            .stack(trial_x_participant=['participant','trials']).dropna(dim="trial_x_participant",\
            how="all").groupby('trial_x_participant').count(dim="samples").cumsum().squeeze()
        dur_dropped_na = durations.dropna("trial_x_participant")
        starts = np.roll(dur_dropped_na.data, 1)
        starts[0] = 0
        self.starts = starts
        self.ends = dur_dropped_na.data-1 
        self.durations =  self.ends-self.starts+1
        self.named_durations =  durations.dropna("trial_x_participant") - durations.dropna("trial_x_participant").shift(trial_x_participant=1, fill_value=0)
        self.mean_rt = self.durations.mean()
        self.n_trials = len(self.durations)
        self.cpus = cpus
        self.coords = durations.reset_index('trial_x_participant').coords
        self.n_samples, self.n_dims = np.shape(data.data.T)
        self.bumps,self.data_summed  = self.calc_bumps(data.data.T)#adds bump morphology
        # self.data = data.data.T#adds bump morphology
        self.max_d = self.durations.max()
        self.max_bumps = self.compute_max_bumps()
        self.shape = float(shape)
        self.estimate_magnitudes = estimate_magnitudes
        self.estimate_parameters = estimate_parameters
    
    def calc_bumps(self,data):
        '''
        This function puts on each sample the correlation of that sample and the next 
        x samples (depends on sampling frequency and bump size) with a half sine on time domain.  Will be used fot the likelihood 
        of the EEG data given that the bumps are centered at each time point
        Parameters
        ----------
        data : ndarray
            2D ndarray with n_samples * components
        Returns
        -------
        bumbs : ndarray
            a 2D ndarray with samples * PC components where cell values have
            been correlated with bump morphology
        '''
        bump_idx = np.arange(self.bump_width_samples)*self.steps+self.steps/2
        bump_frequency = 1000/(self.bump_width*2)#gives bump frequency given that bumps are defined as half-sines
        template = np.sin(2*np.pi*np.linspace(0,1,1000)*bump_frequency)[[int(x) for x in bump_idx]]#bump morph based on a half sine with given bump width and sampling frequency  
        template = template/np.sum(template**2)#Weight normalized
        bumps = np.zeros(data.shape)
        data_summed = np.zeros(data.shape)

        for j in np.arange(self.n_dims):#For each PC
            temp = np.zeros((self.n_samples,self.bump_width_samples))
            temp[:,0] = data[:,j]#first col = samples of PC
            for i in np.arange(1,self.bump_width_samples):
                temp[:,i] = np.concatenate((temp[1:, i-1], [0]), axis=0)
                # puts the component in a [n_samples X bump_width_samples] matrix shifted.
                # each column is a copy of the first one but shifted one sample upwards
            data_summed[:,j] = temp[:,i].copy()
            bumps[:,j] = temp @ template
            # for each PC we calculate its correlation with half-sine defined above
        for trial in range(self.n_trials):#avoids confusion of gains between trials
            bumps[self.ends[trial]-self.bump_width_samples+1:self.ends[trial]+1, :] = 0
            data_summed[self.ends[trial]-self.bump_width_samples+1:self.ends[trial]+1, :] = 0
        return bumps, data_summed

    def fit_single(self, n_bumps, magnitudes=None, parameters=None, threshold=1, verbose=True,
            starting_points=1, parameters_to_fix=None, magnitudes_to_fix=None, method='random'):
        '''
        Fit HsMM for a single n_bumps model
        Parameters
        ----------
        n_bumps : int
            how many bumps are estimated
        magnitudes : ndarray
            2D ndarray components * n_bumps, initial conditions for bumps magnitudes
        parameters : list
            list of initial conditions for Gamma distribution scale parameter. If parameters are estimated, the list provided is used as starting point,
            if parameters are fixed, parameters estimated will be the same as the one provided. When providing a list, stage need to be in the same order
            _n_th gamma parameter is  used for the _n_th stage
        threshold : float
            threshold for the HsMM algorithm, 0 skips HsMM
        '''
        import pandas as pd 
        if verbose:
            print(f'Estimating {n_bumps} bumps model with {starting_points} starting point(s)')
        
        if self.estimate_magnitudes == False:#Don't need to manually fix mags if not estimated
            magnitudes_to_fix = np.arange(n_bumps)
        if self.estimate_parameters == False:#Don't need to manually fix pars if not estimated
            parameters_to_fix = np.arange(n_bumps+1)            
        #Formatting parameters
        if isinstance(parameters, (xr.DataArray,xr.Dataset)):
            parameters = parameters.dropna(dim='stage').values
        if isinstance(magnitudes, (xr.DataArray,xr.Dataset)):
            magnitudes = magnitudes.dropna(dim='bump').values  
        if isinstance(magnitudes, np.ndarray):
            magnitudes = magnitudes.copy()
        if isinstance(parameters, np.ndarray):
            parameters = parameters.copy()          
        if parameters_to_fix is None: parameters_to_fix=[]
        if magnitudes_to_fix is None: magnitudes_to_fix=[]
        if starting_points > 0:#Initialize with equally spaced option
            if parameters is None:
                parameters = np.tile([self.shape, (np.mean(self.durations))/(n_bumps+1)/self.shape], (n_bumps+1,1))
            initial_p = parameters
            
            if magnitudes is None:
                magnitudes = np.zeros((self.n_dims,n_bumps))
            initial_m = magnitudes
        
        if starting_points > 1:
            import multiprocessing as mp
            parameters = [initial_p]
            magnitudes = [initial_m]
            if method == 'random':
                for sp in np.arange(starting_points):
                    proposal_p = self.gen_random_stages(n_bumps, np.mean(self.durations))
                    proposal_m = np.zeros((self.n_dims,n_bumps))#Mags are NOT random but always 0
                    proposal_p[parameters_to_fix] = initial_p[parameters_to_fix]
                    proposal_m[magnitudes_to_fix] = initial_m[magnitudes_to_fix]
                    parameters.append(proposal_p)
                    magnitudes.append(proposal_m)
            elif method == 'grid':
                parameters = self.grid_search(n_bumps+1, starting_points)
                magnitudes = np.zeros((len(parameters), self.n_dims,n_bumps))
            else:
                raise ValueError('Unknown starting point method requested, use "random" or "grid"')
            with mp.Pool(processes=self.cpus) as pool:
                estimates = pool.starmap(self.fit, 
                    zip(itertools.repeat(n_bumps), magnitudes, parameters, itertools.repeat(1),\
                        itertools.repeat(magnitudes_to_fix),itertools.repeat(parameters_to_fix),))   
            lkhs_sp = [x[0] for x in estimates]
            mags_sp = [x[1] for x in estimates]
            pars_sp = [x[2] for x in estimates]
            eventprobs_sp = [x[3] for x in estimates]
            max_lkhs = np.where(lkhs_sp == np.max(lkhs_sp))[0][0]
            lkh = lkhs_sp[max_lkhs]
            mags = mags_sp[max_lkhs]
            pars = pars_sp[max_lkhs]
            eventprobs = eventprobs_sp[max_lkhs]
            
        elif starting_points==1:#informed starting point
            lkh, mags, pars, eventprobs = self.fit(n_bumps, initial_m, initial_p,\
                                        threshold, magnitudes_to_fix, parameters_to_fix)

        else:#uninitialized    
            if np.any(parameters)== None:
                parameters = np.tile([self.shape, (self.mean_rt)/self.shape], (n_bumps+1,1))
            if np.any(magnitudes)== None:
                magnitudes = np.zeros((self.n_dims,n_bumps))
            lkh, mags, pars, eventprobs = self.fit(n_bumps, magnitudes, parameters,\
                                        threshold, magnitudes_to_fix, parameters_to_fix)
        n_bumps = len(pars)-1
        if n_bumps != self.max_bumps+1:#align all dimensions
            pars = np.concatenate((pars, np.tile(np.nan, (self.max_bumps+1-len(pars),2))))
            mags = np.concatenate((mags, np.tile(np.nan, (np.shape(mags)[0], \
                self.max_bumps-np.shape(mags)[1]))),axis=1)
            eventprobs = np.concatenate((eventprobs, np.tile(np.nan, (np.shape(eventprobs)[0],\
                                        np.shape(eventprobs)[1], self.max_bumps-np.shape(eventprobs)[2]))),axis=2)
        xrlikelihoods = xr.DataArray(lkh , name="likelihoods")
        xrparams = xr.DataArray(pars, dims=("stage",'params'), name="parameters")
        xrmags = xr.DataArray(mags, dims=("component","bump"), name="magnitudes")
        part, trial = self.coords['participant'].values, self.coords['trials'].values

        n_samples, n_participant_x_trials, bumps_n = np.shape(eventprobs)
        xreventprobs = xr.Dataset({'eventprobs': (('bump', 'trial_x_participant','samples'), 
                                         eventprobs.T)},
                         {'bump':np.arange(bumps_n),
                          'samples':np.arange(n_samples),
                        'trial_x_participant':  pd.MultiIndex.from_arrays([part,trial],
                                names=('participant','trials'))})
        xreventprobs = xreventprobs.transpose('trial_x_participant','samples','bump')
        estimated = xr.merge((xrlikelihoods, xrparams, xrmags, xreventprobs))

        if verbose:
            print(f"Parameters estimated for {n_bumps} bumps model")
        return estimated
    
    def fit(self, n_bumps, magnitudes, parameters,  threshold, magnitudes_to_fix=None, parameters_to_fix=None):
        '''
        Fitting function underlying single and iterative fit
        '''
        initial_parameters =  np.copy(parameters)
        initial_magnitudes = np.copy(magnitudes)
        lkh, eventprobs = self.likelihood(magnitudes, parameters, n_bumps)
        initial_lkh = lkh
        if threshold == 0:
            lkh_prev = initial_lkh
            magnitudes_prev = initial_magnitudes.copy()
            parameters_prev = initial_parameters.copy()
            eventprobs_prev = eventprobs.copy()
        else:
            lkh_prev = -np.inf
        means = np.zeros((self.max_d, self.n_trials, self.n_dims))
        for trial in range(self.n_trials):
            means[:self.durations[trial],trial,:] = self.bumps[self.starts[trial]:self.ends[trial]+1,:]
            #Reorganize samples morphed with bump shape on trial basis
        while (lkh - lkh_prev) > threshold:#Expectation-Maximization algorithm
            #As long as new run gives better likelihood, go on  
            lkh_prev = lkh.copy()
            magnitudes_prev = magnitudes.copy()
            parameters_prev = parameters.copy()
            eventprobs_prev = eventprobs.copy()
            #Magnitudes from Expectation
            for bump in range(n_bumps):
                for comp in range(self.n_dims):
                    magnitudes[comp,bump] = np.mean(np.sum( \
                        eventprobs[:,:,bump]*means[:,:,comp], axis=0))
                    # 1) scale the bump morph by likelihood of the bump
                    # 2) sum across all samples in trial
                    # 3) mean across trials of the sum of samples in a trial
                    # repeated for each component and for each bump
            magnitudes[magnitudes_to_fix,:] = initial_magnitudes[magnitudes_to_fix,:]
            #Parameters from Expectation
            parameters = self.gamma_parameters(eventprobs, n_bumps)
            parameters[parameters_to_fix, :] = initial_parameters[parameters_to_fix,:]
            null_stages = np.where(parameters[:,1]==0)[0]
            if len(null_stages) > 0:
                print(f'stage(s) {null_stages} removed as < {self.min_stage_duration} sample duration')
                parameters = np.delete(parameters, null_stages, axis=0)
                initial_parameters = np.delete(initial_parameters, null_stages, axis=0)
                magnitudes = np.delete(magnitudes, null_stages, axis=1)
                initial_magnitudes =  np.delete(initial_magnitudes, null_stages, axis=1)
                n_bumps -= len(null_stages)
                lkh_prev = -np.inf#Likelihood is biased when 0 length stages present
            lkh, eventprobs = self.likelihood(magnitudes, parameters, n_bumps)
        # lkh_prev -= initial_lkh#Corrects for baseline
        return lkh_prev, magnitudes_prev, parameters_prev, eventprobs_prev

    def likelihood(self, magnitudes, parameters, n_bumps, lkh_only=False):
        '''
        Defines the likelihood function to be maximized as described in Anderson, Zhang, Borst and Walsh, 2016
        
        Returns
        -------
        likelihood : float
            likelihood
        eventprobs : ndarray
            [samples(max_d)*n_trials*n_bumps] = [max_d*trials*nBumps]
        '''
        n_stages = n_bumps+1
        gains = np.zeros((self.n_samples, n_bumps))
        signal = np.zeros((self.n_samples, n_bumps))
        last_stage = np.zeros((self.n_samples, 1))
        for comp in range(self.n_dims):
            # computes the gains, i.e. how much the bumps reduce the variance at 
            # the location where they are placed for all samples, see Appendix Anderson,Zhang, 
            # Borst and Walsh, 2016, last equation, right hand side parenthesis 
            # (S^2 -(S -B)^2) (Sb- B2/2). And sum over all PCA
            # gains += self.bumps[:,i][np.newaxis].T * magnitudes[i,:] - (magnitudes[i,:]**2)/2
            for trial in range(self.n_trials):
                gains[self.starts[trial]:self.ends[trial]+1-self.bump_width_samples+1] += \
                    self.bumps[self.starts[trial]:self.ends[trial]+1-\
                    self.bump_width_samples+1,comp][np.newaxis].T *\
                    magnitudes[comp,:] - (magnitudes[comp,:]**2)/2
                # gains[self.starts[trial]:self.ends[trial]+1] += 
                # self.bumps[self.starts[trial]:self.ends[trial]+1,i][np.newaxis].T \
                # * magnitudes[i,:] - (magnitudes[i,:]**2)
                # signal[self.starts[trial]:self.ends[trial]+1] += 
                # (self.data_summed[self.starts[trial]:self.ends[trial]+1,i][np.newaxis].T* magnitudes[i,:] - (magnitudes[i,:]**2)
#                 last_stage[self.starts[trial]:self.ends[trial]+1] += self.data_summed[self.starts[trial]:self.ends[trial]+1,i][np.newaxis].T**2/self.bump_width_samples
                
            # bump*magnitudes-> gives [n_samples*nBumps] It scales bumps prob. by the
            # global magnitudes of the bumps topology 'magnitudes' of each bump. 
            # tile append vertically the (estimated bump-magnitudes)^2 of one PC 
            # for all samples divided by 2.
            # gain(n,sum(pca)) = gain(n,pca) + corrBump(n,pca) * 
            # estBumpsMorph(pca,bumps) - (estBumpMorph(pca,bumps)^2)/2
            # sum for all PCs of the 'normalized' correlation of P(having a sin)
            # and bump morphology
        # print(gains)
        gains = np.exp(gains)
        last_stage = np.exp(last_stage)

        probs = np.zeros([self.max_d,self.n_trials,n_bumps]) # prob per trial
        probs_b = np.zeros([self.max_d,self.n_trials,n_bumps])
        prob_last_stage = np.zeros([self.max_d,self.n_trials,1])
        for trial in np.arange(self.n_trials):
            # Following assigns gain per trial to variable probs 
            # in direct and reverse order
            probs[:self.durations[trial],trial,:] = \
                gains[self.starts[trial]:self.ends[trial]+1,:] 
            probs_b[:self.durations[trial],trial,:] = \
                gains[self.starts[trial]:self.ends[trial]+1,:][::-1,::-1]
            prob_last_stage[:self.durations[trial],trial] = last_stage[self.starts[trial]:self.ends[trial]+1]
        LP = np.zeros([self.max_d, n_stages]) # Gamma pdf for each stage parameters
        for stage in range(n_stages):
            LP[:,stage] = self.gamma_EEG(parameters[stage,0], parameters[stage,1])
            # Compute Gamma pdf from 0 to max_d with give scale parameters
        # likelihood =  np.sum(np.log(np.sum(np.transpose(np.tile(LP[:,:], (self.n_trials, 1,1)), axes=(1,0,2))*\
        #     np.concatenate([probs, prob_last_stage], axis=2), axis=(0,2))))#np.sum(np.log(eventprobs.sum(axis=(0,2))))#sum over max_d to avoid 0s in log

        likelihood =  np.sum(np.log(np.sum(np.transpose(np.tile(LP[:,:-1], (self.n_trials, 1,1)), axes=(1,0,2))*probs, axis=(0,2))))#n
        BLP = LP[:,::-1] # States reversed gamma pdf
        forward = np.zeros((self.max_d, self.n_trials, n_bumps))
        backward = np.zeros((self.max_d, self.n_trials, n_bumps))
        # eq1 in Appendix, first definition of likelihood
        # For each trial compute gamma pdf * gains
        # Start with first bump to loop across others after
        forward[:,:,0] = np.tile(LP[:,0][np.newaxis].T,\
            (1,self.n_trials))*probs[:,:,0]
        backward[:,:,0] = np.tile(BLP[:,0][np.newaxis].T,\
                    (1,self.n_trials)) # reversed Gamma pdf

        for bump in np.arange(1,n_bumps):#continue with other bumps
            add_b = backward[:,:,bump-1]*probs_b[:,:,bump-1]
            for trial in np.arange(self.n_trials):
                temp = np.convolve(forward[:,trial,bump-1], LP[:,bump])
                # convolution between gamma * gains at previous states and state i
                forward[:,trial,bump] = temp[:self.max_d]
                temp = np.convolve(add_b[:,trial], BLP[:, bump])
                # same but backwards
                backward[:,trial,bump] = temp[:self.max_d]
            forward[:,:,bump] = forward[:,:,bump]*probs[:,:,bump]
        for trial in np.arange(self.n_trials):#Undoes inversion
            backward[:self.durations[trial],trial,:] = \
                backward[:self.durations[trial],trial,:][::-1,::-1]
        eventprobs = forward * backward
        # likelihood = np.sum(np.log(eventprobs[:,:,0].sum(axis=0)))#sum over max_d to avoid 0s in log
        eventprobs = eventprobs / np.tile(eventprobs.sum(axis=0), [self.max_d, 1, 1])
        #normalization [-1, 1] divide each trial and state by the sum of the n points in a trial
        if lkh_only:
            return likelihood
        else:
            return [likelihood, eventprobs]

    def gamma_EEG(self, a, scale):
        '''
        Returns PDF of gamma dist with shape = a and scale, on a range from 0 to max_length 
        
        Parameters
        ----------
        a : float
            shape parameter
        scale : float
            scale parameter      
        Returns
        -------
        d : ndarray
            density for a gamma with given parameters, normalized to 1
        '''
        d = sp_gamma.pdf(np.arange(1, self.max_d+1),a,scale=scale)
        d = d/np.sum(d)#pdf is independent of stage length, otherwise longer are likelier
        return d
    
    def gamma_parameters(self, eventprobs, n_bumps):
        '''
        The likeliest location of the bump is computed from eventprobs. The flats are then taken as the 
        distance between the bumps
        Parameters
        ----------
        eventprobs : ndarray
            [samples(max_d)*n_trials*n_bumps] = [max_d*trials*nBumps]
        durations : ndarray
            1D array of trial length
        mags : ndarray
            2D ndarray components * nBumps, initial conditions for bumps magnitudes
        shape : float
            shape parameter for the gamma, defaults to 2  
        Returns
        -------
        params : ndarray
            shape and scale for the gamma distributions
        '''
        averagepos = np.arange(1,self.max_d+1)@eventprobs.mean(axis=1)
        diffs = np.diff(averagepos, prepend=0)
        #Following part makes sure that stages lower than min_stage gets removed by 0ing
        #corresponding parameters
        bump = 1
        while bump < n_bumps:
            if diffs[bump] <= self.min_stage_duration:
                first = bump
                while bump < n_bumps and diffs[bump] < self.min_stage_duration:
                    bump += 1
                last = bump-1
                averagepos[first-1] += (averagepos[last]-averagepos[first-1])/2 
                averagepos[first:last+1] = averagepos[first-1] 
            bump += 1
        params = np.zeros((n_bumps+1,2), dtype=float)
        params[:,0] = self.shape
        params[:-1,1] = np.diff(averagepos, prepend=.5) / self.shape
        params[-1,1] = (self.mean_rt-averagepos[-1]) / self.shape#Last parameter is given by RT
        return params
    
    def iterative_fit(self, likelihoods, parameters, magnitudes):
        n_bumps_max = len(likelihoods)#With NAs
        likelihoods = likelihoods[np.isfinite(likelihoods)].copy()
        parameters = np.array(parameters[np.isfinite(parameters[:,0]),:].copy())
        magnitudes = magnitudes.copy()
        n_bumps = len(likelihoods)
        pars_n_bumps = []
        mags_n_bumps = []
        for n_bump in range(1, n_bumps+1):
            temp_par = parameters.copy()
            bump_idx = np.sort(np.argsort(likelihoods)[::-1][:n_bump])#sort the index of highest likelihood bumps
            bump_mags = magnitudes[:,bump_idx].copy()
            bump_pars = np.tile(float(self.shape), (n_bump+1,2))
            bump_pars[:-1,1] = temp_par[bump_idx,1]
            bump_pars[-1,1] = temp_par[-1,1]#-1/self.shape
            bump_pars[:,1] = np.diff(bump_pars[:,1], prepend=.5)
            print(bump_pars)
            print(np.cumsum(bump_pars[:,1])*2)
            pars_n_bumps.append(bump_pars)
            mags_n_bumps.append(bump_mags)
        if self.cpus > 1:
            with mp.Pool(processes=self.cpus) as pool:
                bump_loo_results = pool.starmap(self.fit_single, 
                    zip(np.arange(1,n_bumps+1), mags_n_bumps, pars_n_bumps,
                        itertools.repeat(0),itertools.repeat(False)))
        else:
            bump_loo_results = []
            for bump_tmp, flat_tmp in zip(mags_n_bumps,pars_n_bumps):
                n_bump = len(bump_loo_results)+1
                bump_loo_results.append(self.fit_single(n_bump, bump_tmp, flat_tmp, 0, False))
        bests = xr.concat(bump_loo_results, dim="n_bumps")
        bests = bests.assign_coords({"n_bumps": np.arange(1,n_bumps+1)})
        return bests
    
    def backward_estimation(self, max_fit=None, max_starting_points=1, method="random"):
        '''
        First read or estimate max_bump solution then estimate max_bump - 1 solution by 
        iteratively removing one of the bump and pick the one with the highest 
        likelihood
        
        Parameters
        ----------
        max_fit : xarray
            To avoid re-estimating the model with maximum number of bumps it can be provided 
            with this arguments, defaults to None
        max_starting_points: int
            how many random starting points iteration to try for the model estimating the maximal number of bumps
        
        '''

        if not max_fit:
            if max_starting_points>0:
                print(f'Estimating the model with the maximal number of bumps ({self.max_bumps}) with 1 pre-defined starting point and {max_starting_points-1} {method} starting points')
            bump_loo_results = [self.fit_single(self.max_bumps, starting_points=max_starting_points, method=method, verbose=False)]
        else:
            bump_loo_results = [max_fit]
        i = 0
        n_bumps = bump_loo_results[0].dropna('bump').bump.max().values
        list_values_n_bumps = [n_bumps]
        print(bump_loo_results[0].parameters.values)
        while n_bumps  > 0:
            print(f'Estimating all solutions for {n_bumps} number of bumps')
            temp_best = bump_loo_results[i]#previous bump solution
            temp_best = temp_best.dropna('bump')
            temp_best = temp_best.dropna('stage')
            n_bumps_list = np.arange(n_bumps+1)#all bumps from previous solution
            flats = temp_best.parameters.values
            print(flats)
            bumps_temp, flats_temp = [], []
            for bump in np.arange(n_bumps+1):#creating all possible solutions
                bumps_temp.append(temp_best.magnitudes.sel(bump = np.array(list(set(n_bumps_list) - set([bump])))).values)
                flat = bump + 1 #one more flat than bumps
                temp = flats[:,1].copy()
                temp[flat-1] += temp[flat]
                temp = np.delete(temp, flat)
                flats_temp.append(np.reshape(np.concatenate([np.repeat(self.shape, len(temp)), temp]), (2, len(temp))).T)
            if self.cpus > 1:
                with mp.Pool(processes=self.cpus) as pool:
                    bump_loo_likelihood_temp = pool.starmap(self.fit_single, 
                        zip(itertools.repeat(n_bumps), bumps_temp, flats_temp,
                            itertools.repeat(1),itertools.repeat(False)))
            else:
                bump_loo_likelihood_temp = []
                for bump_tmp, flat_tmp in zip(bumps_temp,flats_temp):
                    bump_loo_likelihood_temp.append(self.fit_single(n_bumps, bump_tmp, flat_tmp, 1, False))
            models = xr.concat(bump_loo_likelihood_temp, dim="iteration")
            bump_loo_results.append(models.sel(iteration=[np.where(models.likelihoods == models.likelihoods.max())[0][0]]).squeeze('iteration'))
            n_bumps = bump_loo_results[-1].dropna('bump').bump.max().values
            list_values_n_bumps.append(n_bumps)
            i += 1
        bests = xr.concat(bump_loo_results, dim="n_bumps")
        bests = bests.assign_coords({"n_bumps": list_values_n_bumps})
        return bests
    
    def compute_max_bumps(self):
        '''
        Compute the maximum possible number of bumps given bump width and mean or minimum reaction time
        '''
        return int(np.min(self.durations)/self.bump_width_samples)

    def bump_times(self, eventprobs, mean=True):
        '''
        Compute bump onset times based on bump probabilities

        Parameters
        ----------
        a : float
            shape parameter
        b : float
            scale parameter
        max_length : int
            maximum length of the trials        

        Returns
        -------
        d : ndarray
            density for a gamma with given parameters
        '''
        warn('This method is deprecated and will be removed in future version, use compute_times() instead', DeprecationWarning, stacklevel=2)
        eventprobs = eventprobs.dropna('bump', how="all")
        eventprobs = eventprobs.dropna('trial_x_participant', how="all")
        onsets = np.empty((len(eventprobs.trial_x_participant),len(eventprobs.bump)+1))
        i = 0
        for trial in eventprobs.trial_x_participant.dropna('trial_x_participant', how="all").values:
            onsets[i, :len(eventprobs.bump)] = np.arange(self.max_d) @ eventprobs.sel(trial_x_participant=trial).data
            onsets[i, -1] = self.ends[i] - self.starts[i]
            i += 1
        if mean:
            return np.mean(onsets, axis=0)
        else:
            return onsets
    
    @staticmethod        
    def compute_times(init, estimates, duration=False, fill_value=None, mean=False, cumulative=False, add_rt=False):
        '''
        Compute the likeliest onset times for each bump

        Parameters
        ----------
        estimates :
            Estimated instance of an hsmm model
        init : 
            Initialized HsMM object  
        duration : bool
            Whether to compute onset times (False) or stage duration (True)
        fill_value : float | ndarray
            What value to fill for the first onset/duration

        Returns
        -------
        times : xr.DataArray
            Bump onset or stage duration with trial_x_participant*bump dimensions
        '''

        eventprobs = estimates.eventprobs
        times = xr.dot(eventprobs, eventprobs.samples+1, dims='samples')#Most likely bump location
        n = len(times[0,:].values[np.isfinite(times[0,:].values)])
        if duration:
            fill_value=0
        if fill_value != None:            
            added = xr.DataArray(np.repeat(fill_value,len(times.trial_x_participant))[np.newaxis,:],
                                 coords={'bump':[0], 
                                         'trial_x_participant':times.trial_x_participant})
            times = times.assign_coords(bump=times.bump+1)
            times = times.combine_first(added)
        if add_rt:             
            rts = init.named_durations
            rts = rts.assign_coords(bump=int(times.bump.max().values+1))
            rts = rts.expand_dims(dim="bump")
            times = xr.concat([times, rts], dim='bump')
        if duration:
            #adding reaction time and treating it as the last bump
            times = times.rename({'bump':'stage'})
            if not cumulative:
                times = times.diff(dim='stage')
        if mean:
            times = times.mean('trial_x_participant')
        return times
   
    @staticmethod
    def compute_topologies(electrodes, estimated, bump_width_samples, extra_dim=False):
        shifted_times = estimated.eventprobs.shift(samples=bump_width_samples//2+1, fill_value=0).copy()#Shifts to compute electrode topology at the peak of the bump
        if extra_dim:
            return xr.dot(electrodes.rename({'epochs':'trials'}).\
                      stack(trial_x_participant=['participant','trials']).data.fillna(0), \
                      shifted_times.fillna(0), dims=['samples']).mean('trial_x_participant').\
                      transpose(extra_dim,'bump','electrodes')
        else:
            return xr.dot(electrodes.rename({'epochs':'trials'}).\
                      stack(trial_x_participant=['participant','trials']).data.fillna(0), \
                      shifted_times.fillna(0), dims=['samples']).mean('trial_x_participant').\
                      transpose('bump','electrodes')
    
    def compute_topo(self, data, eventprobs, mean=True):
        '''
        DEPRECATED
        '''
        warn('This method is deprecated and will be removed in future version, use onset_times() instead', DeprecationWarning, stacklevel=2)
        if 'trial_x_participant' not in data:
            data = data.stack(trial_x_participant=['participant','epochs'])
        eventprobs = eventprobs.dropna('bump', how="all")
        eventprobs = eventprobs.dropna('trial_x_participant', how="all")
        topologies = np.empty((len(eventprobs.trial_x_participant.data), 
                               len(eventprobs.bump.data),
                               len(data.electrodes.data)))
        i = 0
        for trial_x_participant in eventprobs.trial_x_participant:
            z = 0
            for bump in eventprobs.bump:
                trial_samples = np.arange(self.named_durations.sel(trial_x_participant=trial_x_participant).data)
                topologies[i,z, :] = data.sel(trial_x_participant = trial_x_participant,
                                              samples=trial_samples).data @ \
                    eventprobs.sel(trial_x_participant = trial_x_participant,  bump=bump, samples=trial_samples)
                z += 1
            i += 1
        if mean:
            return np.mean(topologies, axis=0)
        else:
            return topologies
        
    def multiple_topologies(self, data, eventprobs, mean=True):
        '''
        DEPRECATED
        '''
        warn('This method is deprecated and will be removed in future version, use onset_times() instead', DeprecationWarning, stacklevel=2)        
        topo = np.tile(np.nan, (len(eventprobs.n_bumps), len(eventprobs.n_bumps), len(data.electrodes)))
        for n_bumps in eventprobs.n_bumps:
            topo[n_bumps-1, :n_bumps.values, :] = self.compute_topo(data, eventprobs.sel(n_bumps=n_bumps))
        return topo
    
    def gen_random_stages(self, n_bumps, mean_rt):
        '''
        Returns random stage duration between 0 and mean RT by iteratively drawind sample from a 
        uniform distribution between the last stage duration (equal to 0 for first iteration) and 1.
        Last stage is equal to 1-previous stage duration.
        The stages are then scaled to the mean RT
        Parameters
        ----------
        n_bumps : int
            how many bumps
        mean_rt : float
            scale parameter
        Returns
        -------
        random_stages : ndarray
            random partition between 0 and mean_rt
        '''
        random_stages = np.array([[self.shape,x*mean_rt/self.shape] for x in np.random.beta(2, 2, n_bumps+1)])
        return random_stages
    
    def grid_search(self, n_stages, iter_limit=np.inf, verbose=True):
        '''
        This function decomposes the mean RT into a grid with points. Ideal case is to have a grid with one sample = one search point but the number
        of possibilities badly scales with the length of the RT and the number of stages. Therefore the iter_limit is used to select an optimal number
        of points in the grid with a given spacing. After having defined the grid, the function then generates all possible combination of 
        bump placements within this grid. It is faster than using random points (both should converge) but depending on the mean RT and the number 
        of bumps to look for, the number of combination can be really large. 
        
        Parameters
        ----------
        n_stages : int
            how many bump to look for
        iter_limit : int
            How much is too much

        Returns
        -------
        parameters : ndarray
            3D array with numper of possibilities * n_stages * 2 (gamma parameters)
        '''
        from itertools import combinations_with_replacement, permutations  
        from math import comb as binomcoeff
        import more_itertools as mit
        mean_rt = self.mean_rt
        n_points = int(mean_rt)
        check_n_posibilities = binomcoeff(n_points-1, n_stages-1)
        while binomcoeff(n_points-1, n_stages-1) > iter_limit:
            n_points = n_points-1
        spacing = mean_rt//n_points#1 if no points removed in the previous step
        grid = (np.arange(1, n_points+1))*spacing#bump cannot be at sample 0
        mean_rt = spacing*n_points
        grid = grid[grid < mean_rt - ((n_stages-2)*spacing)]#In case of >2 stages avoid impossible durations, just to speed up
        comb = np.array([x for x in combinations_with_replacement(grid, n_stages) if np.sum(x) == mean_rt])#A bit bruteforce
        new_comb = []
        for c in comb:
            new_comb.append(np.array(list(mit.distinct_permutations(c))))
        comb = np.vstack(new_comb)
        parameters = np.zeros((len(comb),n_stages,2))
        for idx, y in enumerate(comb):
            parameters[idx, :, :] = [[self.shape, x/self.shape] for x in y]
        if check_n_posibilities > iter_limit:
            print(f'Initial number of possibilities is {check_n_posibilities}. Given a number of max iteration = {iter_limit}: fitting {len(parameters)} models based on all possibilities from grid search with a spacing of {int(spacing)} samples and {int(n_points)} points and durations of {grid}')
        elif verbose:
            print(f'Fitting {len(parameters)} models using grid search')
        return parameters
    
    def sliding_bump(self, n_bumps=None, colors=default_colors, figsize=(12,3), verbose=True, method='derivative', plot_deriv=True, magnitudes=None, bump_likelihood=False):
        '''
        This method outputs the likelihood and estimated parameters of a 1 bump model with each sample, from 0 to the mean 
        epoch duration. The parameters and likelihoods that are returned are 
        Take the highest likelihood, place a bump by excluding bump width space around it, follow with the next one
        '''
        import matplotlib.pyplot as plt
        from itertools import cycle
        
        mean_rt = self.mean_rt
        init_n_bumps = n_bumps
        if n_bumps == None:
            n_bumps = self.max_bumps
        parameters = self.grid_search(2, verbose=verbose)#Looking for all possibilities with one bump
        if magnitudes is None:
            magnitudes = np.zeros((len(parameters), self.n_dims,1))
        parameters = parameters[parameters[:,1,1] >= self.bump_width_samples-2]#impossible space as bump would be computed on less than bump_width
        lkhs_init, mags_init, pars_init, _ = \
            self.estimate_single_bump(magnitudes, parameters, [0,1], [], 0)
        sort_idx = np.argsort(pars_init[:,0,1])
        if verbose:
            _, ax = plt.subplots(figsize=figsize, dpi=300)
            ax.plot(pars_init[sort_idx,0,1]*self.shape, lkhs_init[sort_idx], '-', color='k')
        
        if method == 'derivative':
            bump_idx = np.where(np.diff(np.sign(np.gradient(lkhs_init[sort_idx]))) < 0)[0]
            pars = np.tile(np.nan, (len(bump_idx)+1, 2))
            pars[:len(bump_idx), :] = pars_init[sort_idx, 0, :][bump_idx, :]
            pars[:len(bump_idx), 1] += .25#deriv is half-point
            pars[len(bump_idx),:] = np.array([self.shape, mean_rt/self.shape])#last stage defined as rt
            mags = mags_init[sort_idx, :, :][bump_idx, :, :]
            lkhs = lkhs_init[sort_idx][bump_idx]
            if verbose:
                for bump in range(len(bump_idx)):
                    ax.plot(np.array(pars)[bump,1]*self.shape, lkhs[bump], 'o', color=colors[bump])
                
  
            if bump_likelihood:
                bump = 0
                for mag in mags:
                    lkhs_mag, mags_mag, pars_mag, _ = \
                        self.estimate_single_bump(np.tile(mag, (len(parameters), 1, 1)), parameters, [0,1], [0], 0)
                    ax.plot(pars_mag[sort_idx,0,1]*self.shape, lkhs_mag[sort_idx], '--', color=colors[bump])
                    bump += 1
            if verbose:
                plt.show()
                if plot_deriv:
                    _, ax = plt.subplots(figsize=figsize, dpi=300)
                    plt.plot(pars_init[sort_idx,0,1]*self.shape, np.gradient(lkhs_init[sort_idx]), '-', color='k')
                    plt.hlines(0, 0, mean_rt)
                    # plt.ylim(-1,1)#For simulations, corner case
                    plt.show()  
                
        elif method == 'estimation':
            lkhs_sp, mags_sp, pars_sp, eventprobs_sp = \
                self.estimate_single_bump(np.zeros((len(parameters), self.n_dims,1)), \
                parameters, [], [], 1)
            lkhs_sp_sorting = lkhs_sp.copy()
            mags_sp_sorting = mags_sp.copy()
            pars_sp_sorting = pars_sp.copy()
            group_color = np.empty(len(lkhs_sp),dtype=str)
            cycol = cycle(colors)
            colors = [next(cycol) for x in range(n_bumps)]
            max_lkhs = []
            for bump in range(n_bumps):
                if not np.isnan(lkhs_sp_sorting).all():#Avoids problem if n_bumps > actual bumps
                    max_lkh = np.where(lkhs_sp_sorting == np.nanmax(lkhs_sp_sorting))[0][0]
                    max_lkhs.append(max_lkh)
                    gamma_max = pars_sp[max_lkh,0,1] 
                    neighbors = np.where(abs(pars_sp[:,0,1]-gamma_max) <= (self.bump_width_samples/2)/self.shape)[0]
                    group = np.concatenate([[max_lkh], neighbors])
                    if verbose:
                        ax.plot(np.array(pars_sp)[group,0,1]*self.shape, lkhs_sp_sorting[group], '.', color=colors[bump])
                    lkhs_sp_sorting[group] = np.nan
                    pars_sp_sorting[group, :, :] = np.nan

            if np.isnan(lkhs_sp_sorting).all() and init_n_bumps != None and verbose:
                print(f'The number of requested bumps exceeds the number of convergence points found in the parameter space, bumps {bump} to {n_bumps} were not found') 

            if not np.isnan(lkhs_sp_sorting).all() and verbose:#plot remaining points if all where not handled before
                ax.plot(pars_sp_sorting[:,0,1]*self.shape, lkhs_sp_sorting, '.', color='gray', label='Not attributed')
                ax.legend()
            if verbose: 
                ax.set_xlabel('sample #')
                ax.set_ylabel('Loglikelihood')
                plt.show()
            pars = np.tile(np.nan, (n_bumps+1, 2))
            pars[:len(max_lkhs), :] = pars_sp[max_lkhs, 0, :]
            order = np.argsort(pars[:len(max_lkhs),1])#sorted index based on first stage duration
            pars[:len(max_lkhs), :] = pars[order, :]
            mags = mags_sp[max_lkhs][order]
            max_lkhs = np.array(max_lkhs)[order]
            pars[len(max_lkhs),:] = np.array([self.shape, mean_rt/self.shape])#last stage defined as rt
            lkhs = np.repeat(np.nan, n_bumps)
            lkhs[:len(max_lkhs)] = lkhs_sp[max_lkhs]
        
        return pars, mags[:,:,0].T, lkhs

    def estimate_single_bump(self, magnitudes, parameters, parameters_to_fix, magnitudes_to_fix, threshold):
        if self.cpus >1:
            if np.shape(magnitudes) == 2:
                magnitudes = np.tile(magnitudes, (len(parameters), 1, 1))
            with mp.Pool(processes=self.cpus) as pool:
                estimates = pool.starmap(self.fit, 
                    zip(itertools.repeat(1), magnitudes, parameters, 
                        itertools.repeat(threshold), itertools.repeat(magnitudes_to_fix), 
                        itertools.repeat(parameters_to_fix)))  
        else:
            estimates = []
            for pars, mags in zip(parameters, magnitudes):
                estimates.append(self.fit(1, mags, pars, threshold, magnitudes_to_fix, parameters_to_fix))
        lkhs_sp = np.array([x[0] for x in estimates])
        mags_sp = np.array([x[1] for x in estimates])
        pars_sp = np.array([x[2] for x in estimates])
        eventprobs_sp = [x[3] for x in estimates]
        return lkhs_sp, mags_sp, pars_sp, eventprobs_sp