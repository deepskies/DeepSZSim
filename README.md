# `DeepSZSim`

Code for producing fast simulations of the SZ effect for galaxy halos of varying z, $M_{200}$, based on average thermal pressure profile fits from [Battaglia et al. 2012](https://ui.adsabs.harvard.edu/abs/2012ApJ...758...75B/abstract). Simulated submaps can include tSZ signal from these halos, simulated CMB, instrument beam convolution and white noise.

## Installation 

Sam to add more here on installation instructions.

The simulated CMB signal relies on `camb`. 

From the top-level directory, you can do `pip install .`

## Usage

The usage of this code is documented in `notebooks/demo_simulation.ipynb`. A detailed walkthrough of the functions available in this code is in `notebooks/demo_full_pipeline.ipynb`.

A full list of potential inputs is documented in `settings/config.yaml` and you can edit `settings/inputdata.yaml` to reflect your desired simulation settings.  

`dm_halo_dist.py` generates a z, $M_{200}$ array. The functions in `make_sz_cluster.py` create pressure profiles, Compton-y, and SZ signal maps from these halos of various z, $M_{200}$ and produce the final simulated submaps. These submaps contain simulated CMB and simple instrument beam convolution from `simtools.py` and white noise from `noise.py`. Plotting tools are provided in `visualization.py`.

## Citation

If you use this code in your research, please cite this GitHub repo. Please also make use of the citation instructions for `camb` provided [here](https://camb.info).

## Contributing

If you would like to contribute, please open a new [issue](https://github.com/deepskies/deepszsim/issues), and/or be in touch with the [authors](#contact)

## Contact

The code was developed by [Eve M. Vavagiakis](http://evevavagiakis.com), [Camille Avestruz](https://sites.google.com/view/camilleavestruz), Kush Banker, Ioana Cristescu, [Samuel D. McDermott](https://samueldmcdermott.github.io), [Brian Nord](http://briandnord.com/bio), Elaine Ran, Hanzhi Tan, and Brian Zhang, and is maintained by the [DeepSkies lab](https://deepskieslab.com)
