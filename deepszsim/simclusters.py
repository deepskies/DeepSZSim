import numpy as np
from deepszsim import simtools, noise, load_vars, dm_halo_dist, make_sz_cluster, filters
from tqdm import tqdm

from astropy import constants as c
from astropy import units as u
import os
import h5py
from datetime import datetime as dt

class simulate_clusters:
    """
    class for simulating a distribution of clusters
    
    Parameters
    ----------
    M200: float or array-like of float
        the mass contained within R200 in solar masses (same length as z_dist)
    redshift_z: float or array-like of float
        the redshift of the cluster (unitless) (same length as M200_dist)
    num_halos: None or int
        number of halos to simulate if none supplied
    halo_params_dict: None or dict
        parameters from which to sample halos if `num_halos` specified,
        must contain `zmin`, `zmax`, `m200min_SM`, `m200max_SM`
    R200_Mpc: None or float or np.ndarray(float)
        if None, will calculate the R200 values corresponding to a given set of
        M200 and redshift_z values for the specified cosmology
    profile: str
        Name of profile, currently only supports "Battaglia2012"
    image_size_pixels: None or int
        image size in pixels (should be odd; if even, will return images whose
        sides are `image_size_pixels+1` in length)
    image_size_arcmin: None or float
        image size in arcmin
    pixel_size_arcmin: None or float
        pixel size in arcmin
    alpha: float
        fixed to equal 1.0 in Battaglia 2012
    gamma: float
        fixed to equal -0.3 in Battaglia 2012
    load_vars_yaml: None or str
        path to yaml file with params; if `None`, must explicitly include image specifications
    seed: None or int
        random seed value to sample with
    tqverb: bool
        whether or not to display tqdm progress bar while making T maps
    """
    def __init__(self, M200 = None, redshift_z = None, num_halos = None, halo_params_dict = None,
                 R200_Mpc = None, profile = "Battaglia2012",
                 image_size_pixels = None, image_size_arcmin = None, pixel_size_arcmin = None,
                 alpha = 1.0, gamma = -0.3,
                 load_vars_yaml = os.path.join(os.path.dirname(__file__), 'Settings', 'config_simACTDR5.yaml'),
                 seed = None, tqverb = False
                 ):
        
        if (M200 is not None) and (redshift_z is not None):
            self.M200, self.redshift_z = M200, redshift_z
        else:
            if (num_halos is None):
                print("must specify `M200` AND `redshift_z` simultaneously,",
                      "OR a number of halos to generate with `num_halos`",
                      "along with the arguments for `deepszsim.dm_halo_dist.flatdist_halo` via `halo_params_dict`.")
                num_halos = 100
            if (halo_params_dict is None):
                halo_params_dict = {'zmin': 0.1, 'zmax': 1.1, 'm200min_SM': 4e13, 'm200max_SM': 1e15}
            print(f"making {num_halos} clusters uniformly sampled from"
                  f"{halo_params_dict['zmin']}<z<{halo_params_dict['zmax']}, and "
                  f"{halo_params_dict['m200min_SM']:1.1e}<M200<{halo_params_dict['m200max_SM']:1.1e}")
            self.redshift_z, self.M200 = dm_halo_dist.flatdist_halo(halo_params_dict['zmin'],
                                                                    halo_params_dict['zmax'],
                                                                    halo_params_dict['m200min_SM'],
                                                                    halo_params_dict['m200max_SM'],
                                                                    int(num_halos), seed = seed)
        
        try:
            self._size = len(self.M200)
        except TypeError:
            self.M200, self.redshift_z = np.array([self.M200]), np.array([self.redshift_z])
            self._size = 1
        self.clusters = {}
        
        if profile != "Battaglia2012":
            print("only `Battaglia2012` is implemented, using that for now")
        self.profile = "Battaglia2012"
        
        if load_vars_yaml is not None:
            self.vars = load_vars(load_vars_yaml)
        self.image_size_pixels = self.vars['image_size_pixels'] if (image_size_pixels is None) else image_size_pixels
        self.image_size_arcmin = self.vars['image_size_arcmin'] if (image_size_arcmin is None) else image_size_arcmin
        self.pixel_size_arcmin = self.vars['pixel_size_arcmin'] if (pixel_size_arcmin is None) else pixel_size_arcmin
        self.beam_size_arcmin = self.vars['beam_size_arcmin']
        self.cosmo = self.vars['cosmo']
        self.tqverb = tqverb
        
        self.alpha, self.gamma = alpha, gamma
        self.seed, self._rng = seed, np.random.default_rng(seed)
        
        if R200_Mpc is not None:
            self.R200_Mpc = R200_Mpc
        else:
            self.R200_Mpc, self.angsize500_arcmin = np.array(
                [make_sz_cluster.get_r200_angsize_and_c200(self.M200[i], self.redshift_z[i], self.vars,
                                                           angsize_density = '500c')[1:3]
                 for i in range(self._size)]).T
        
        self.id_list = [f"{int(self.M200[i]/1e9):07}_{int(self.redshift_z[i]*100):03}_{self._rng.integers(10**6):06}"
                        for i in range(self._size)]
        self.clusters.update(zip(self.id_list, [{"params": {'M200': self.M200[i],
                                                            'redshift_z': self.redshift_z[i],
                                                            'R200': self.R200_Mpc[i],
                                                            'angsize_arcmin': self.angsize500_arcmin[i],
                                                            'angsize500_arcmin': self.angsize500_arcmin[i],
                                                            'image_size_pixels': self.image_size_pixels}}
                                                for i in range(self._size)]))
    
    def get_y_maps(self):
        """
        Returns
        -------
        np.ndarray(float)
            self._size many maps of the Compton `y` value, each of which is image_size_pixels x image_size_pixels in size
        """
        try:
            return self.y_maps
        except AttributeError:
            if self.tqverb: print("making `y` maps")
            self.y_maps = np.array([make_sz_cluster.generate_y_submap(self.M200[i],
                                                      self.redshift_z[i],
                                                      R200_Mpc = self.R200_Mpc[i],
                                                      load_vars_dict = self.vars)
                                    for i in tqdm(range(self._size), disable = (not self.tqverb))])
            return self.y_maps
    
    def get_dT_maps(self):
        """
        Returns
        -------
        np.ndarray(float)
            self._size many maps of the dT values in units of uK, each of which is image_size_pixels x
            image_size_pixels in size
        """
        try:
            return self.dT_maps
        except AttributeError:
            ym = self.get_y_maps()
            fSZ = simtools.f_sz(self.vars['survey_freq'], self.cosmo.Tcmb0)
            self.dT_maps = (ym * fSZ * self.cosmo.Tcmb0).to(u.uK).value
            return self.dT_maps
    
    def get_T_maps(self, add_CMB = True, returnval = False):
        """
        Parameters
        ----------
        add_CMB: bool
            whether or not to include the CMB contribution to the final map
        returnval: bool
            whether or not to return the T maps themselves or simply update internal attribute

        Returns
        -------
        np.ndarray(float)
            self._size many maps of the sky in units of uK, each of which is image_size_pixels x image_size_pixels in
            size
        """
        dT_maps = self.get_dT_maps()
        if add_CMB: self.ps = simtools.get_cls(ns = self.vars['ns'], cosmo = self.vars['cosmo'])
        if self.tqverb: print("making convolved T maps" + (" with CMB" if add_CMB else ""))
        _centerpix = self.image_size_pixels // 2
        for i in tqdm(range(self._size), disable = (not self.tqverb)):
            dTm, name = dT_maps[i], self.id_list[i]
            curdic = self.clusters[name]
            curdic['params']['dT_central'] = dTm[_centerpix,_centerpix]
            curdic['maps'] = {}
            beamsig_map = simtools.convolve_map_with_gaussian_beam(self.pixel_size_arcmin,
                                                                   self.beam_size_arcmin, dTm)
            if add_CMB:
                conv_map, cmb_map = simtools.add_cmb_map_and_convolve(dTm, self.ps,
                                                                      self.pixel_size_arcmin,
                                                                      self.beam_size_arcmin)
            else:
                conv_map, cmb_map = beamsig_map, np.zeros_like(beamsig_map)
            if not self.vars['noise_level'] == 0:
                noise_map = noise.generate_noise_map(self.image_size_pixels, self.vars['noise_level'],
                                                     self.pixel_size_arcmin)
            else:
                noise_map = np.zeros_like(conv_map)
            final_map = conv_map + noise_map
            curdic['maps']['conv_map'] = conv_map
            curdic['maps']['CMB_map'] = cmb_map
            curdic['maps']['signal_map'] = dTm
            curdic['maps']['beamsig_map'] = beamsig_map
            curdic['maps']['final_map'] = final_map
            curdic['params']['ap'] = filters.get_tSZ_signal_aperture_photometry(final_map,
                                                                                curdic['params']['angsize_arcmin'],
                                                                                self.pixel_size_arcmin)[-1]
        
        if returnval:
            return self.clusters
    
    def ith_T_map(self, i, add_CMB = True):
        """
        Parameters
        ----------
        i: int
            the map you want to return
        add_CMB: bool
            whether or not to include the CMB contribution to the final map

        Returns
        -------
        np.ndarray(float)
            the ith map of the sky in units of uK, which is image_size_pixels x image_size_pixels in size
        """
        try:
            return self.clusters[self.id_list[i]]['maps']['final_map']
        except KeyError:
            self.get_T_maps(add_CMB = add_CMB)
            return self.clusters[self.id_list[i]]['maps']['final_map']
    
    def save_map(self, i = None, nest_h5 = True, nest_name = None,
                 savedir = os.path.join(os.path.dirname(__file__), "outfiles")):
        """
        Parameters
        ----------
        i: None or int
            the map you want to save, if you only want to save a single map
        nest_h5: bool
            whether or not to nest the clusters into a single h5 file, assuming that you are saving all of the
            clusters that you have calculated
        nest_name: None or str
            a name for the overall file if you are nesting them (otherwise, it will name it with the number of
            clusters plus the date plus a random string)
        savedir: str
            the path to the directory you want to save into
        """
        self.savedir = savedir
        if not os.path.exists(self.savedir):
            print(f"making local directory `{self.savedir}`")
            os.mkdir(self.savedir)
        try:
            self.clusters[self.id_list[0]]['maps']
        except KeyError:
            self.get_T_maps()
        if i is not None:
            with h5py.File(os.path.join(self.savedir, self.id_list[i] + ".h5"), 'w') as f:
                for k, v in self.clusters[self.id_list[i]]['params'].items():
                    f.create_dataset('params/' + k, data = float(v))
                for k, v in self.clusters[self.id_list[i]]['maps'].items():
                    f.create_dataset('maps/' + k, data = v)
        elif nest_h5:
            file_name = "clusters_" + "N"+ str(self._size) +"_" + dt.strftime(dt.now(),
                                                                    '%y%m%d%H%M%S%f') if nest_name is None else nest_name
            with h5py.File(os.path.join(self.savedir, file_name + ".h5"), 'w') as f:
                for j in range(self._size):
                    for k, v in self.clusters[self.id_list[j]]['params'].items():
                        f.create_dataset(self.id_list[j] + '/params/' + k, data = float(v))
                    for k, v in self.clusters[self.id_list[j]]['maps'].items():
                        f.create_dataset(self.id_list[j] + '/maps/' + k, data = v)
        else:
            for i in range(self._size):
                with h5py.File(os.path.join(self.savedir, self.id_list[i] + ".h5"), 'w') as f:
                    for k, v in self.clusters[self.id_list[i]]['params'].items():
                        f.create_dataset('params/' + k, data = float(v))
                    for k, v in self.clusters[self.id_list[i]]['maps'].items():
                        f.create_dataset('maps/' + k, data = v)