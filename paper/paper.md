---
title: 'DeepSZSim: A Python code for fast, tunable Sunyaev–Zeldovich effect submap simulations'
tags:
  - Python
  - cosmology
  - cosmic microwave background
  - galaxy clusters
  - astronomy
authors:
  - name: Eve M. Vavagiakis
    orcid: 0000-0002-2105-7589
    equal-contrib: true
    affiliation: "1" 
  - name: Samuel D. McDermott
    orcid: 0000-0001-5513-1938
    equal-contrib: true
    affiliation: "2"
  - name: Humna Awan
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Elaine Y. Ran
    orcid: 0009-0007-1681-4745
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Kush Banker
    orcid: 0009-0000-8099-2609
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Samantha Usman
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Camille Avestruz
    orcid: 0000-0001-8868-0810
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Brian Nord
    orcid: 0000-0001-6706-8972
    equal-contrib: true
    affiliation: "2" # (Multiple affiliations must be quoted)

affiliations:
 - name: Department of Physics, Duke University, Durham, NC 27708, USA 
   index: 1
 - name: Department of Physics, Cornell University, Ithaca, NY 14853, USA
   index: 2
 - name: Department of Astronomy and Astrophysics University of Chicago, Chicago, IL
   index: 2
  - name: Department of Physics University of Michigan, Ann Arbor, MI 48109
   index: 5
 - name: Kavli Institute for Cosmological Physics, University of Chicago, 5801 S Ellis Ave, Chicago, IL 60637
   index: 3
 - name: Department of Physics, University of Chicago, 5801 S Ellis Ave, Chicago, IL 60637
   index: 4
 - name: Leinweber Center for Theoretical Physics, University of Michigan, Ann Arbor, MI 48109
   index: 6
 - name: Fermi National Accelerator Laboratory, Batavia, IL 60510, USA
   index: 7
date: 12 July 2024
bibliography: paper.bib

Authors: E. M. Vavagiakis, S. McDermott, H. Awan, E. Ran, K. Banker, S. Usman, C. Avestruz, Brian Nord


---


# Summary

Current and upcoming measurements of the cosmic microwave background (CMB), the oldest observable light in the universe, elucidate the fundamental physics of our universe, including the development of cosmic large-scale structure. Galaxy clusters are the largest gravitationally bound structures in our universe and make up a significant portion of this large-scale structure. Through measurements of galaxy clusters, we can derive insights into the growth of structure and place powerful constraints on cosmology. Simulations of galaxy clusters that are well-matched to upcoming data sets are a key tool for addressing systematics (e.g., cluster mass inference) that limit these current and future cluster-based cosmology constraints. However, most state-of-the-art simulations are too computationally intensive to produce multiple versions of significant  systematic effects: from underlying gas physics to observational modeling uncertainties. 

We present DeepSZSim, a novel user-friendly Python framework for generating simulations of the CMB and the thermal Sunyaev–Zel’dovich (tSZ) effect in galaxy clusters, which is  based on average galaxy cluster thermal pressure profile models. DeepSZSim includes CMB power spectra generation using CAMB and simulated CMB temperature maps using `namaster` [@alonsoUnifiedPseudoC_2019], as well as tSZ signal modeling, instrument beam convolution, and noise. By tuning the input parameters based on a cosmology, distributions of halo mass and redshift, and experiment properties (e.g., map depth and observation frequency), users are able to generate a variety of simulated primary and secondary CMB anisotropy images. These simulations offer a fast and flexible method for generating large datasets to test mass inference methods like machine learning and simulation-based inference. 

# Statement of Need

DeepSZSim fits a unique niche within the plethora of existing CMB primary and secondary anisotropy simulations and software. These simulators and data sets range in size, detail, and accuracy, speed, and ease of use. Most simulators are computationally intensive, and most simulated datasets are not optimized for machine learning training sets. For example, N-body simulations provide the major setting for high-fidelity forward models of the universe [@sehgalSimulationsMicrowaveSky2010, @liSimulatedCatalogsMaps2022, @steinWebskyExtragalacticCMB2020a, @IllustrisTNG]. These simulations are uniquely capable of capturing both large-scale and small-scale spatial modes of the cosmic web and the CMB and at multiple time steps. To achieve this, N-body simulations have high computational costs and are inflexible with respect to the specific physics models used in the simulations. Mechanistic forward modeling can provide much faster and at least somewhat lower fidelity (less-detailed) simulations [@lesgourguesCosmicLinearAnisotropy2011, @bollietClass_szOverview2023]. Other methods deploy a combination of N-body and mechanistic models [@yamadaImagingSimulationsSunyaevZel2012]. Machine learning has been tested for producing simulations of the CMB with generative adversarial networks [@hanDeepLearningSimulations2021a] and with autoencoders [@rothschildEmulatingSunyaevZeldovichImages2022]: unfortunately, machine learning methods for generative modeling lack interpretability and uncertainty quantification.

Overall, most software and simulations are difficult to access, especially for researchers new to these subjects, including students: they are not publicly available and are difficult to install. There is a need for codebases that are accessible, inexpensive, and multi-fidelity. 

While lacking the fidelity of N-body simulations, DeepSZSim meets a need for a user-friendly, fast, and realistic simulation of CMB primary anisotropies via DeepCMBSim, alongside tSZ signal modeling with highly customizable observational inputs. This is valuable for building and testing a wide range of models for the classification and detection of CMB-related objects like SZ clusters. This is particularly useful for machine learning settings, which typically require a large amount of data, which is often not available from N-body simulations. DeepSZSim is currently being used by Deep Skies researchers to generate large catalogs of CMB+tSZ signal submaps to study galaxy cluster parameter inference via simulation-based inference. 

# Features

![Software workflow for the `DeepSZSim` package, including the elements of `DeepCMBsim`.\label{fig:workflow}](figures/DeepSZSim_Workflow.png)

## DeepCMBSim

The `DeepCMBsim` package combines physical processes and sources of noise in a software framework that enables fast and realistic simulation of the CMB in which key cosmological parameters can be varied. DeepCMBSim simulates correlations of temperatures and polarization signals from the CMB, including large-scale gravitational lensing and BB polarization caused by non-zero tensor-to-scalar ratios.

DeepCMBSim’s primary physics module is `camb_power_spectrum`, which defines the `CAMBPowerSpectrum` class. This calls `CAMB` `[@Lewis:1999bs; @Howlett:2012mh]`. The power spectrum of the noise follows the form in `[@Hu:2001kj]`, assuming statistical independence in the Stokes parameters `[@Knox:1995dq; @Zaldarriaga:1996xe]`.
This software allows the user to specify cosmological parameters (e.g., omega matter, omega baryon, the lensing scale, the tensor-to-scalar ratio, which are inputs to CAMB) and experiment parameters (e.g., white noise level, beam size) in a  `yaml` configuration file to permit a user-friendly interface to permit reproducible simulations. The default parameters reproduce the Planck 2018 cosmology `[@Planck:2018vyg]`. 

![Example output angular spectra for the `DeepCMBsim` package for a set of tensor-to-scalar ratios r and lens scaling factors A_lens.\label{fig:cmb}](figures/CMBSpectra_Examples.png)

The package workflow is demonstrated in Figure \autoref{fig:workflow}. 

We provide an example notebook in `notebooks/simcmb_example.ipynb` which demonstrates the software functionality.

## DeepSZSim

`DeepSZsim` includes code for producing fast simulations of the thermal Sunyaev-Zeldovich effect for galaxy halos of varying mass and redshift, based on average thermal pressure profile fits from Battaglia et al. 2012 `[@Battaglia:2012]`. The output is an array of simulated submaps of the tSZ effect associated with galaxy halos, which can include simulated CMB, instrument beam convolution, and/or white noise. 

The user provides inputs to generate an array of redshift and mass ($M_200$) for dark matter halos, the desired pixel and submap size for the output submaps, and inputs such as experiment properties (observation frequency, noise level, beam size) and a cosmological model. These inputs are easily customizable, or the user can run defaults based on the Atacama Cosmology Telescope `[@ACT:2021]` and Planck cosmology `[@Planck:2019]`. Cosmology computations depend on `colossus` `[@Colossus:2018]` and `astropy` `[@Astropy:2013]`.

From these inputs, pressure profiles [cite], Compton-y profiles [cite], and tSZ signal maps are generated for the dark matter halo array. Simulated CMB primary anisotropy maps can be generated through a dependency on `DeepCMBSim`. Final simulated submaps can include instrument beam convolution and white noise `[@actnotebooks:2015]`. Plotting functions for the simulations and an aperture photometry filter are included as tools. The submap handling functions rely on `pixell` `[@pixell:2024]`.

We present examples of the primary outputs from `DeepCMBSim` and `DeepSZSim` in, Figure \autoref{fig:cmb} and Figure \autoref{fig:sz}, respectively. 

![Example outputs for the `DeepSZsim` package for a set of masses, redshifts, and noise configurations.\label{fig:sz}](figures/SZCluster_Examples.png)


# Acknowledgements

*E. M. Vavagiakis*: methodology, software, writing, supervision.
*S. McDermott*: methodology, software, writing, supervision. 
*H. Awan*: methodology, software, writing. 
*E. Ran*: methodology, software
*K. Banker*: methodology, software
*S. Usman*: methodology, software
*C. Avestruz*: conceptualization, methodology, project administration, funding acquisition, supervision, writing 
*Brian Nord*: conceptualization, methodology, project administration, funding acquisition, supervision, writing.

We acknowledge contributions from Maggie Voetberg, Junhe Zhang, Helen Tan, Ioana Corescu, and Antwine Willis.

We acknowledge the Deep Skies Lab as a community of multi-domain experts and collaborators who’ve facilitated an environment of open discussion, idea generation, and collaboration. This community was important for the development of this project.

Work supported by the Fermi National Accelerator Laboratory, managed and operated by Fermi Research Alliance, LLC under Contract No. DE-AC02-07CH11359 with the U.S. Department of Energy. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for U.S. Government purposes. 

EMV acknowledges support from NSF award AST-2202237.

This material is based upon work supported by the National Science Foundation under Grant No. 2009944.

# References
