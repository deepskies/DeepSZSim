import sys
import subprocess
import os
import time
import random
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from scipy import interpolate
from scipy import integrate
from scipy.stats import skewnorm

#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'colossus'])
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from colossus.halo import mass_adv

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'emcee'])
import emcee

from astropy.constants import M_sun
from astropy.constants import G
from astropy import units as u

class GenerateHalo():

    def __init__(self):
        self.data =[]

        
    def flatdist_halo(self,zmin,zmax,m500min,m500max,size):
        zdist=np.random.uniform(low=zmin, high=zmax, size=size)
        mdist=np.random.uniform(low=m500min, high=m500max, size=size)
        
        return zdist, mdist
    
    def vir_to_200_colossus(self,cosmo,sigma8,ns,Mvir,z):
        
        
        params = {'flat': True, 'H0': cosmo.H0.value, 'Om0': cosmo.Om0, 'Ob0': cosmo.Ob0, 'sigma8':sigma8, 'ns': ns}
        cosmology.addCosmology('myCosmo', **params)
        cosmo_colossus= cosmology.setCosmology('myCosmo')
    
        M200, R200, c200 = mass_adv.changeMassDefinitionCModel(Mvir/cosmo.h, z, 'vir', '200c', c_model = 'ishiyama21')
        
        M200 *= cosmo.h #From M_solar/h to M_solar
        R200 = R200*cosmo.h/1000 #From kpc/h to Mpc
        R200 = R200/cosmo.scale_factor(z) #From Mpc proper to Mpc comoving
        
        return (M200,R200,c200)
    
    
    
    
#FUNCTIONS BELOW HERE HAVE NOT BEEN TESTED OR USED RECENTLY     
    
    
    
        
    """# **Single Redshift**"""

    #extract power function
    def extract_power(mass_arr):
        """
        Function to extract the power of galaxy cluster mass array, switch from 10^n to n
        
        parameters:
        -----------
        mass_arr: 1d NumPy array of cluster mass in 10^n M⊙ unit
        
        mass_arr_p: 1d numpy array of mass of power = n
        """
        mass_arr_p = np.log10(mass_arr)
        return mass_arr_p

    def reverse_extract_power(prim_mass_sample):
        """
        switch from n to 10^n

        Parameters:
        ----------- 
        prim_mass_sample: a priliminary numpy array

        mass_sample: a numpy array of cluter mass after unit transformation
        """
        mass_sample = np.power(10, prim_mass_sample)
        return mass_sample

    #the likelihood function
    def lnpo(mass, min, max, test_fun):
        """
        likelihood function used by MCMC

        parameters:
        -----------
        mass: a float or a 1d numpy array of cluster mass in 10^n M⊙ unit
        """
        if (mass < min) or (mass > max):
            return -np.inf
        return math.log(test_fun(mass)) #log likelihood is required by emcee
    
    def interpolate_MCMC(mass_array_p, mfunc_n, mass_range, sample_num):
        """
        interpolate and normalize mfunc_n, use the result as a likelihood function and perform MCMC method to get the sample.
        
        parameters:
        -----------
        mass_arr_p: 1d NumPy array of cluster mass power (for example, 10^14 M⊙ represented as 14 in arr)
        mfunc_n: 1d NumPy array of halo number density * 10^5
        mass_range: a tuple of cluster masses, lower limit, and upper limit for sampling
        sample_num: an integer of the number of samples
        
        sample_chain.flatten(): an 1d numpy array of mass sampling, same unit as mass_arr_p
        """

        min, max = mass_range
        #interpolate
        from scipy import interpolate
        interpolate_mfunc = interpolate.interp1d(mass_array_p, mfunc_n)
        test_arr = np.linspace(min, max, 5000)
        f_test = interpolate_mfunc(test_arr)
        #normalize to likelihood function by divided by integration, looking for better method
        from scipy import integrate
        val, err = integrate.quad(interpolate_mfunc, min, max)
        test_fun = interpolate.interp1d(test_arr, (f_test/val))
        #nval, nerr = integrate.quad(test_fun, min, max), 5000 test points will lead to error < 10^-4

        #create random walkers
        import emcee
        import random
        randomlist = []
        for i in range(20):
            n = random.uniform(min, max)
            randomlist.append(n)
        random_arr = np.array(randomlist)

        ndim, nwalkers = 1, 20
        ###this means that the sample must be larger than 20, otherwise it'll return error###

        #backend setup (discarded)
        # name = "rs_" + str(redshift) + "_sn_" + str(sample_num)
        # file_name = "mcmc_save.h5"
        # backend = emcee.backends.HDFBackend(filename, name = name)
        # backend.reset(nwalkers, ndim)

        #run MCMC
        import time
        process_sample_num = sample_num // (ndim * nwalkers)
        p0 = random_arr.reshape((nwalkers, ndim))
        the_random = random.uniform(1.0, 100.0)
        locals()["sample" + str(the_random)] = emcee.EnsembleSampler(nwalkers, ndim, lnpo, args = [min, max, test_fun])
        t0=time.time()
        pos, prob, state = locals()["sample" + str(the_random)].run_mcmc(p0, 25000) #burn-in sequence
        locals()["sample" + str(the_random)].reset()
        t1=time.time()
        print( ' Done with burn-in: ', t1-t0)
        locals()["sample" + str(the_random)].run_mcmc(pos, process_sample_num)
        t2=time.time()
        print( ' Done with MCMC: ', t2-t1)

        mass_chain = locals()["sample" + str(the_random)].chain
        locals()["sample" + str(the_random)].reset() #prevent storing error
        return test_fun, mass_chain.flatten()

    def mass_sampling(mass_range, redshift = 0.0, mdef = '200c', model = 'bocquet16', sample_num = 100000):
        """
        the function to give back a sample of cluster mass distribution based on the halo mass function
        
        Parameters:
        -----------
        mass_range: a tuple of cluster masses, lower limit, and upper limit for sampling
        redshift: a float, 0.0 by default
        sample_num: an integer of the number of samples, 100000 by default
        mdef: The mass definition in which the halo mass M is given
        model: the halo mass function model used by colossus
        
        mass_chain: a NumPy array of length = sample_num.
        test_func: the likelihood function
        """
        from colossus.cosmology import cosmology
        from colossus.lss import mass_function
        import numpy as np
        min, max = mass_range
        mass_arr = np.logspace(min, max, num = 200, base = 10)
        cosmology.setCosmology('WMAP9')
        mfunc = mass_function.massFunction(mass_arr, redshift, mdef = mdef, model = model, q_out = 'dndlnM')
        mass_arr_p = extract_power(mass_arr)
        test_func, prim_mass_sample = interpolate_MCMC(mass_arr_p, mfunc, mass_range, sample_num)
        return test_func, prim_mass_sample

    """test single redshift"""

  

    """# **Multiple Redshift**

    multiple redshift sample
    , shape of redshift reference: 
    1. https://arxiv.org/pdf/2101.08373.pdf
    2. https://pole.uchicago.edu/public/data/sptsz-clusters/


    code reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html#scipy.stats.skewnorm
    """

    def skew_sample(size = 10000):
        """
        the function to give back a sample of redshift based on skew gaussian distribution imitating SPT cluster data: https://pole.uchicago.edu/public/data/sptsz-clusters/
        
        Parameters:
        ----------- 
        size: integer, sample number

        mass_chain: a NumPy array of length = sample_num
        rs_sample: an 1d NumPy array of clusters' redshift sample with length = size
        """
        from scipy.stats import skewnorm
        a = 4
        sample = skewnorm.rvs(a, loc = 2, scale = 2, size = size) 
        rs_sample = sample/6.6666
        return rs_sample

    def single_redshift_num(z_range, sample_num, z_dist_model):
        """
        the function to give back redshifts and sample_num per redshift for multi-redshift sampling

        Parameters:
        ----------- 
        z_dist_model: a string, represent the distribution of cluster redshift
        z_range: a tuple of redshift range, (0.0, 1.5) by default
        sample_num: an integer of number of sample, 100000 by default
        
        chop: a NumPy array of redshifts
        num_per_redshift: a NumPy array of cluster num within the corresponding redshift interval of same index number in chop
        """
        if z_dist_model == "skewnorm": #more option reserved for future improvement
            z_sample = skew_sample(size = sample_num)

        #calculate chop_num, 1 chop take 22-25s to sample, be careful
        chop_num = (z_range[1] - z_range[0]) * 90
        chop = np.linspace(z_range[0], z_range[1], int(chop_num)) # an array of redshift upper & lower limit, redshift in this range will be approximate to lower limit
        num_per_redshift = [] 
        for i, redshift in enumerate(chop):
            min_rs = redshift
            condition1 = (min_rs <= z_sample)
            if redshift != z_range[1]:
                max_z = chop[i + 1]
            else:
                max_z = float('inf')
            condition2 = (z_sample < max_z)
            condition = condition1 & condition2

            num = np.count_nonzero(condition) #num of cluster fulfill this range
            num_per_redshift.append(num) 

        num_per_redshift = np.array(num_per_redshift) 

        #fin_chop_num = np.count_nonzero(num_per_redshift >= 20)
        print(f"you divide redshift range {z_range} into {chop_num} chops, it will take about {round(chop_num*22/60, 3)} minutes to complete.")
        return chop, num_per_redshift




    def mul_redshift_mass_sampling(z_dist = "skewnorm", z_range = (0.0, 1.5), mass_range = (14.0, 16.0), mdef = '200c', model = 'bocquet16', sample_num = 100000, store = True):
        """
        the function to give back a sample of multi-redshift cluster mass distribution based on halo mass function
    
        Parameters:
        -----------
        z_dist: a string, representing the distribution of cluster redshift, "skewnorm" by default
        z_range: a tuple of redshift range, (0.0, 1.5) by default
        mass_range: a tuple of cluster masses, lower limit and upper limit for sampling, [min, max] in 10^min M⨀ unit
        mdef: The mass definition in which the halo mass M is given; see colossus doc for more info (https://bdiemer.bitbucket.io/colossus/lss_mass_function.html#lss.mass_function.massFunction)
        model: the halo mass function model used by colossus; see colossus doc for more info
        sample_num: an integer of the number of samples, 100000 by default
        store: a boolean, if True store mass array and redshift into a csv file and return a string of path to file if False returns None
        
        fin_cluster: a Pandas dataframe with 2 col of ["mass_arr", "redshift"], NumPy array of cluster mass corresponding to redshift stored in each row
        filepath: str of file path if store=True, else None
        tot_num: a integer of final sample number, there's a small difference between sample_num and tot_num caused by conditions
        """  
        t0 = time.time()
        sample_num = int(sample_num * 1.1)
        import pandas as pd
        filepath = None
        chop, num_per_redshift = single_redshift_num(z_range, sample_num, z_dist)

        final_mass_list = []
        tot_num = 0
        flag = 0 #cannot use i = 0 to initiate masses and redshifts.
        for i, redshift in enumerate(chop):
            redshift = round(redshift, 4)
            sample_n = num_per_redshift[i]
            if sample_n < 20: #->too small
                sample_n = 20
            sample_n = int(math.ceil(sample_n / 10.0)) * 10 #round up
            print(f"starting sampling with redshift: {redshift}, sample_num : {sample_n}")
            test_func, chain = mass_sampling(mass_range, redshift = redshift, mdef = mdef, model = model, sample_num = sample_n)
            chain = chain[chain < 15.25] #exclude cluster that is too massive, 1.7782794e+15
            tot_num += chain.shape[0]
            if (flag == 0):
                masses = pd.Series(reverse_extract_power(chain))
                rs_arr = np.full((chain.shape[0]), redshift)
                redshifts = pd.Series(rs_arr)
                flag = 1
            else:
                new_mass = pd.Series(reverse_extract_power(chain))
                masses = pd.concat([masses, new_mass])
                rs_arr = np.full((chain.shape[0]), redshift)
                new_redshift = pd.Series(rs_arr)
                redshifts = pd.concat([redshifts, new_redshift])
            #mass_sampling(mass_range, redshift = 0.0, mdef = '200c', model = 'bocquet16', sample_num = 100000):
            #final_mass_list.append(masses(test_func, redshift, chain))
        fin_clusters = pd.DataFrame(columns = ["mass", "redshift"])
        fin_clusters["mass"] = masses
        fin_clusters["redshift"] = redshifts

        #decrease to sample size
        row_num = fin_clusters.shape[0]
        if sample_num < row_num:
            dif = row_num - sample_num
            fin_clusters = fin_clusters.drop(random.sample(range(0, row_num - 1), dif))
            fin_clusters = fin_clusters.reset_index()
        
        f_sample_num = len(fin_clusters)
        digit = len(str(f_sample_num))
        id = np.arange(1, (f_sample_num + 1), 1)
        id = id + int(t0)*(10**digit)
        fin_clusters.insert(0, "id", id)


        if store:
            from pathlib import Path
            filepath = Path(os.getcwd() + "/mass_samples/" + "z(" + str(z_range[0]) + "~" + str(z_range[1]) + ")num_" + str(f_sample_num) + ".csv")
            filepath.parent.mkdir(parents = True, exist_ok = True)
            fin_clusters.to_csv(filepath, index = False)
        t1 = time.time()
        print(f"finish running, using {t1 - t0}s, obtain {f_sample_num} cluster samples")
        return tot_num, filepath, fin_clusters

        
