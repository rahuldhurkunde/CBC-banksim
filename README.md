
# Banksim for simulated signals from compact binary mergers

![](https://pandao.github.io/editor.md/images/logos/editormd-logo-180x180.png)


### Features

- Check the validity of a given template bank using a reference population of simulated CBC signals.
- Different approximants for templates and signals are supported
-- Only aligned-spin templates supported at the moment
--  Eccentric and precessing signals
- Handles large template bank and signal population with ease 
-- Parallelization using PyCBC workflows implemented via the Pegasus API
-- Support for launching multiple workflows in parallel

**Table of Contents**

[TOCM]

[TOC]

# Requirements
## PyCBC
For installation please visit the link https://pycbc.org/

## Input-files
### Template Bank
Generate a template bank using either a lattice based or stochastic placement algorithm. For more details visit http://pycbc.org/pycbc/latest/html/tmpltbank.html

The template bank should be an HDF file and must contain the following parameters 

Parameter | Key
------------- | -------------
primary mass | mass1
secondary mass | mass2
*z* component of spin-1  |  spin1z
*z* component of spin-2  |  spin2z
eccentricity at a reference freq | eccentricity
mean anomaly parameter | long_asc_nodes
chirp time | tau0
chirp time | tau3

**The template bank must be sorted in increasing values of tau0**

### Injections
To generate the injections please visit http://pycbc.org/pycbc/latest/html/inference/examples/bbh.html

Injection parameters for all signals must be stored in a single HDF file.
The injection file must contain the following parameters in addition to the template bank parameters

Parameter | Key
------------- | -------------
*x* component of spin-1  |  spin1x
*y* component of spin-1  |  spin1y
*x* component of spin-2  |  spin2x
*y* component of spin-2  |  spin2y
eccentricity at a reference freq | eccentricity
mean anomaly parameter | long_asc_nodes
inclination of the orbit | inclination
polarization |  polarization
distance to the source | distance
right ascention of the orbit | ra 
source declination | dec

**Make sure the injection files contain the tau0 and tau3 parameters**

### PSD - power spectral density of the detector

Example PSD files can be generated via http://pycbc.org/pycbc/latest/html/psd.html
Current support only for a PSD file and not ASD.

## Configuration file

Once the template bank, injections and PSD files are ready, lastly, we need to set up the configuration file. An *example_workflow.ini*  is provided with the code.

First change the path to the executable from [executables]  section.

There are three important sections in the configuration 

Sections | Purpose
------------- | -------------
[Files]  | Contains the **full** path to the input files
[Required]  | Control the number of processes or workflows to split the analysis into.
[FF] | Arguments that are passed to the fitting-factor computation

### [Required]
+ tau0_tolerance -- Reduce the number of templates for the match calculation by comparing the tau0s of the injection and templates; it can be expected that the best matching template will have a tau0 close to the injection's. To allow some mismatch, we implement a tolerance on this value. A higher tolerance will incorporate more templates in the analysis.

+ nsplits -- Split the match computation of one signal into these many procecesses.

+ nworkflow -- Split the injections into multiple workflows. A value of **1** is recommended unless the template bank and no. of injections are very large.

###[FF]
+ approximant_tb and approximant_sg -- Approximants for the templates and signals respectively. Note only aligned spin templates are supported.
+ HMs (Only for signals)   -- Switch on-off the higher-modes of gravitaional wave. Provide **0** for no HMs and **1** to include HMs
+ sampling_freq and sampling_rate -- Provide the values ensuring the Nyquist criterion requirements.
+ detector -- Choose a single detector from ['H1', 'L1', 'V1']
+ f_min -- Low frequency cutoff for the match computation



# Running the code

This primarily involves three steps -- generating the workflow, submitting the workflow and consolidating the results.

##### Generating the workflow

Navigate to the codebase directory and make subdirectory. This will be the top level directory for the analysis. 

+ Open the **gen.sh** file and change the current working dir. The path will be the codebase directory in your local machine.
+  Change **first** and **last** --  corresponding to the indices of the injections analyzed from [first .... last] from the injection file.
+  Change the workflow_config

+ Execute the workflow generation using

    
        ./gen.sh sub_directory/
    
	

#### Submitting the workflow
Navigate to the sub-directory and launch the **submit.sh** and provide the number of workflows (usually 1).
    
        ./submit.sh 1
    
	
#### Consolidating the results
After all the jobs are finished, launch the **combine_FFs.py** and provide number of injections and no. of splits for each injection.

    
        python combine_FFs.py 10 10
    
