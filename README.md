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

# PDB to MSA Mapping
[Back to Top](#Table-of-Contents)

### Required Files
* **09062020-Cov19gisaid--SeqIdDesc200-aligned-FastaCikti.fa** -- previously aligned fasta file (for testing)
	* downloaded from http://www.dilekbalik.com/SARS-CoV-2_ODOTool/
* **wuhan_ref_fasta** -- covid reference genome: NC_045512.2
* **09062020-Cov19gisaid--SeqIdDesc200-aligned-FastaCikti.fa** -- Full un-aligned fasta file
	* downloaded Oct 20 20202 from https://www.epicov.org/epi3/entities/tmp/tmp_sd_2020_10_20_02_12_qigibl_4j510bafde3/sequences_2020-10-19_07-21.fasta.gz

## Alignment Process
[Back to Top](#Table-of-Contents)

*we want to generate cov_gen_aligned.fasta from our un-aligned fasta file using the Wuhan reference genome*

### FASTA Data:
All FASTA data both created and required for these simulations can be found in the fasta_files/ directory, organized as follows.
- **.fasta** files containing full genome and clades alignments
### Test run

```console
foo@bar:~$ mafft --auto --keeplength --addfragments 09062020-Cov19gisaid--SeqIdDesc200-aligned-FastaCikti.fa wuhan_ref.fasta > 
```

### Full Genome Alignment
- Break up un-aligned fasta files
	- Assumes you have the following files in **/path/to/er_covid19/** 
		- subject genome file: **wuhan_ref.fasta** 
		- aligned genome file: **cov_gen_aligned.fasta** 
		- directory for output: **/path/to/er_covid19/cov_fasta_files/**
```console
foo@bar:~$ singularity exec -B /path/to/er_covid19/biowulf,/path/to/er_covid19/covid_proteins /path/to/er_covid19/LADER.simg python break_up_fasta.py /path/to/er_covid19/ 
```			
	- This creates 
		- broken up fasta files for alignment in *path/to/er_covid19/cov_fasta_files/* 
		- script to align:  **cov_align.swarm** (for cluster computation)

- Run alignment pieces which aligns with mafft (this can be done on a cluster or sequentially).

```console
foo@bar:~$ ./submit_align_swarm.script 
```

	- Individual imulation requirements
		- batches of 15 (lines in **cov_align.swarm**)
		- 10 GB
		- 2 hours
- Finish by concatenating resulting mini-alignments in cov_fasta_files/ (directory created to house mini-alignments) to create full alignment: **covid_genome_full_aligned.fasta**
```console
foo@bar:~$ singularity exec -B /path/to/er_covid19/biowulf,/path/to/er_covid19/covid_proteins /path/to/er_covid19/LADER.simg python concat_fasta.py /path/to/er_covid19/ 
```

### Get Clade Alignments 
- once you have the full genome aligned file you can get clades using get_clades.py
- get_clades.py has subject file hard coded:

``` console
foo@bar:~$  singularity exec -B /path/to/er_covid19/biowulf/,/path/to/er_covid19/covid_proteins /path/to/er_covid19/LADER.simg python get_clades.py /path/to/er_covid19/
```
# Anaconda Environment Setup
[Back to Top](#Table-of-Contents)

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

