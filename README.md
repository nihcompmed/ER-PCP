# ER_DCA Protein Structure Prediction with DCA-Expectation Reflection
=======================

This repository is supplementarty to the publication (PUBLICATION LINK). In the different jupyter notebooks you can run the different steps of our method including:
* Data acuisition and preprocessing 
* Parallel computation of Direct Information
* Method result plotting
* Comparison with other popular methods

Feel free to contact <evancresswell@gmail.com> or <vipulp@niddk.nih.gov > regarding questions on implementation and execution of repo

#### Package Requirements
- Anaconda/Miniconda - to run the proper enviornment for code. If using another python package manager see the following files for enviornment requirements 
    - Expectation Reflection Environment: DCA_ER_requirments.txt
    - PYDCA Enviornment: PDYCA_requirements.txt

## Table of Contents
- [Anaconda Environment Setup](#Anaconda-Environment-Setup)
	- Setting up conda environment for ER and PYDCA simulations
- [PDB to MSA Mapping](#PDB-to-MSA-Mapping)
	- Given a PDB structure we find the best matching MSA to infer connections
- [Expectation Reflection](#Expectation-Reflection)
- [Other Methods](#Other-Methods)
	- Introduction of interfacing with pydca (https://github.com/KIT-MBS/pydca)
	- Presentation/Plotting of MF-DCA and PLM-DCA methods
	- Comparison of methods (De
- [Results](#Results)
	- Result Drivers used to generate figures for papers.


# Anaconda Environment Setup
[Back to Top](#Table-of-Contents)
	- Assumes you have Anaconda or Miniconda: https://www.anaconda.com/
	- If not using Anaconda or Miniconda see the following files for the DCA_ER and PYDCA requiremetns
		- DCA_ER_requirments.txt
    		- Enviornment: PDYCA_requirements.txt
* We want to create two environments for our implementation of Expectaion Reflection (DCA_ER env) and for the PYDCA implementation of the Mean Field and Pseudoliklihood models (PYDCA env)
* To create these envirnoments in conda simply execute the following commands
```console
foo@bar:~$ conda create --name DCA_ER --file DCA_ER_requirements.txt 
foo@bar:~$ conda create --name PYDCA --file PDYCA_requirements.txt
```
See Anaconda documentation for more on environment maintenence and implementation: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

# PDB to MSA Mapping
[Back to Top](#Table-of-Contents)

* Given a PDB structure file we use ProDy to search all available Pfam alignments for the best matching Multiple Sequence Alignment (MSA)
* This takes the following steps
	1. Load poly-peptide sequences for all chains in PDB structure
	2. Loop through different chains of PDB strucutre (ie Chain A, B, etc)
		* ```ProDy.searchPFAM()``` with poly-peptide chain
		* Append search results to DataFrame PDB-Pfam matches
	3. Sort PDB-Pfam matches by 'bitscore'

* We outline this process with a jupyter notebook using PDB ID 1ZDR as an example
### Example in PDB2MSA.ipynb jupyter notebook (cells 2-4)

# Data Processing
* Once a PDB ID and Pfam ID have been matched we can acquire the Pfam's MSA. However before applying Expectation Reflection to the MSA we must preprocess the data MOTIVATION!!!
* Our Data processing is outlined by the following steps
	1. Looping through PDB-Pfam matches
		* Load Pfam MSA
		* Find reference sequence in MSA
		* Trim MSA by reference sequence
		* Remove duplicate rows
		* Remove bad sequences ($\geq 80\%$ gaps in sequence)
		* Remove bad columnds ($\geq 80\%$ gaps in column)
		* Find and Replace amino acid states ($Z\rightarrow \{Q,E\}, B\rightarrow \{N,D\}, X\rightarrow \{\textrm{All AA}\}$)
		* Remove conserved columns ($\geq 80\%$ identity in column)
	2. Because the PDB-Pfam matches are ordered before data processing this looping data processing results in a series of pre-processed MSAs ordered by the strength of their match with PDB structure.
	3. We simply take the first of the series and move forward with that pre-processed MSA.


# Expectiation Reflection
[Back to Top](#Table-of-Contents)
	- Assumes you have the following files in **/path/to/er_covid19/** 
		- subject genome file: **wuhan_ref.fasta** 
		- aligned genome file: **covid_genome_full_aligned.fasta** (created in [Full Genome Alignment][#Full-Genome-Alignment])
		- directory for output: **/path/to/er_covid19/cov_fasta_files/**

- once you have an aligned file you can get DI using ER using run_covGENOME_ER.py
- file generates .pickle files with DI
- the different clade and full sequence run are already defined in run_cov_GENOME_ER file.
	- for all clades see **submit_DI_clade_swarm.script**:
```console
foo@bar:~$ ./submit_DI_clade_swarm.script
```
	- for full genome see **run_covGENOME_ER.py**
		- hardcoded existing full aligned file
```console
foo@bar:~$  singularity exec -B /path/to/er_covid19/biowulf/,/path/to/er_covid19/covid_proteins /path/to/er_covid19/LADER.simg python run_covGENOME_ER.py 
```


# Other Methods
[Back to Top](#Table-of-Contents)

## Scripts and Drivers
* incidence_plotting.py - file for plotting incidence by **region** and **clade**
* gen_incidence_data.py - driver for generating paper incidence data 
* make_codon_swarm.py - script to create swarm (cluster) file to map sequences to to resulting amino acids using **codon_mapping.py**
* make_pair_aa_swarm.py - script to create swarm (cluster) file to map network interactions (from ER simulation) to resulting amino acids using **codon_mapping.py**
* codon_mapping.py - file which maps a given sequence/region of nucleotides to the resulting amino acid
* print_pair_counts.py - driver for creating lists of amino acid pair counts from ER-inferred position pairs (detailed description in file)
* covid_aa_bp_variance.py - a file which computes statistics of amino acid and basepair expression from given genome positions
* plot_covid_genome_CT.py - file for plotting co-evolving postions of genome by **region** and **clade**
* expectation_reflection.py - file to run inference of network interactions using Expectation Reflection

## Results
[Back to Top](#Table-of-Contents)
All the data generated by the above scripts are in the Results/ directory and are organized as follows.
## Result Data:
- **.pickle** files containing DI of full genome and clades both processed and unprocessed
- **.npy** files containing amino acid and basepair data for infered genome positions
- **.txt** files containing all DI pairs from clade and full genome simulations (for inspection without python run)

