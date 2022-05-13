# ER_DCA Protein Structure Prediction with DCA-Expectation Reflection
=======================

This repository is supplementarty to the publication (PUBLICATION LINK). In the different jupyter notebooks you can run the different steps of our method including:
* Data acuisition and preprocessing 
* Parallel computation of Direct Information
* Method result plotting
* Comparison with other popular methods

### Scripts and Drivers
* data_processing - file for generating a PDB-MSA connection (**PDB2MSA**) and and preprocessing data
* expectation_reflection.py - file to run inference of network interactions using Expectation Reflection
### Example Jupyter Notebooks
* [PDB2MSA.ipynb](https://github.com/nihcompmed/ER_DCA/blob/main/PDB2MSA.ipynb) - Notebook which shows the processing for finding a PDB-MSA connection, pre-processing the connected MSA and running Expectation Reflections to acquire Direct Information (DI) used to predict tertiary protein structure
* [pydca_demo.ipynb](https://github.com/nihcompmed/ER_DCA/blob/main/pydca_demo.ipynb) - Notebook which shows how the PYDCA implementaion to acquire the Direct Information (DI) used to predict tertiary protein structure
* [Method_Comparison.ipynb](https://github.com/nihcompmed/ER_DCA/blob/main/Method_Comparison.ipynb) - Notebook which shows how Expectation Reflection can be compared against other methods (PYDCA results) in order to analyse the resulting Protein Strcture prediction.

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
		* Remove bad sequences ($\geq$ 80% gaps in sequence)
		* Remove bad columnds ($\geq$ 80% gaps in column)
		* Find and Replace amino acid states (Z$\rightarrow$ {Q,E}, B$\rightarrow$ {N,D}, X$\rightarrow$ {All AA})
		* Remove conserved columns ($\geq$ 80% identity in column)
* Because the PDB-Pfam matches are ordered before data processing this looping data processing results in a series of pre-processed MSAs ordered by the strength of their match with PDB structure.
### Example in PDB2MSA.ipynb jupyter notebook (cell 5)


# Expectiation Reflection
[Back to Top](#Table-of-Contents)


# Other Methods
[Back to Top](#Table-of-Contents)

## Results
[Back to Top](#Table-of-Contents)
## Result Data:


