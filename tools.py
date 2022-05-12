from numba import prange as parallel_range
import numpy as np
import sys, os, errno
import pandas as pd
# Import Bio data processing features 
from Bio import SeqIO
import Bio.PDB, warnings
from Bio.PDB import *
from scipy.spatial import distance_matrix
from Bio import BiopythonWarning
from Bio import pairwise2
from Bio.SubsMat.MatrixInfo import blosum62
from matplotlib import colors as mpl_colors
import random
import xml.etree.ElementTree as et
from pathlib import Path
from data_processing import data_processing, find_and_replace, data_processing, load_msa
from sklearn.metrics import roc_curve as roc_scikit
from sklearn.metrics import auc, precision_recall_curve

# PAT: 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

warnings.filterwarnings("error")
warnings.simplefilter('ignore', BiopythonWarning)
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', FutureWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()

letter2number = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, \
                     'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '-': 20,
                     'U': 21}
    # ,'B':20, 'Z':21, 'X':22}
number2letter = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L', \
                     10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: '-',
                     21: 'U'}
STANDARD_RESIDUES = {
    'RNA' : ('A', 'C', 'G', 'U'),

    'PROTEIN':('ALA', 'ARG', 'ASN', 'ASP', 'CYS',
        'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO',
        'SER', 'THR', 'TRP', 'TYR', 'VAL')
}


# unzip given file to output path
import gzip, shutil
def gunzip(file_path, output_path):
    print('Unzipping %s to %s' % (file_path, output_path))
    with gzip.open(file_path,"rb") as f_in, open(output_path,"wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


# Return the Hamming distance between string1 and string2.
# string1 and string2 should be the same length.
def hamming_distance(string1, string2): 
    # Start with a distance of zero, and count up
    distance = 0
    # Loop over the indices of the string
    L = len(string1)
    for i in range(L):
        # Add 1 to the distance if these two characters are not equal
        if string1[i] != string2[i]:
            distance += 1
    # Return the final count of differences
    return distance

def window(fseq, window_size=5):
    for i in range(len(fseq) - window_size + 1):
        yield fseq[i:i+window_size]


def filter_residues(residues, biomolecule='PROTEIN'):
    """Filters the standared residues from others (e.g. hetatom residues).
    Parameters
    ----------
        residues : list
            A list of Biopython PDB structure residues objects.
    Returns
    -------
        standard_residues : list
            A lif of Biopython PDB structure standared residues (after hetro
            atom residues are filtered).
    """
    biomolecule = biomolecule.strip().upper()
    standard_residues = []
    for res in residues:
        if res.get_resname().strip() in STANDARD_RESIDUES[biomolecule]:
            if not res.id[0].strip(): standard_residues.append(res) # filter out hetro residues
    return standard_residues


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# print("Direct Information from Expectation reflection:\n",di)
def no_diag(mat, diag_l, s_index=None, make_big=False):
    rows, columns = mat.shape
    if make_big:
        new_mat = 100. * np.ones((rows,columns))
    else:
        new_mat = np.zeros((rows,columns))
    for row in range(rows):
        for col in range(columns):
            if s_index is None:
                if abs(row-col) > diag_l:
                    new_mat[row, col] = mat[row ,col]
            else:
                if abs(s_index[row]-s_index[col]) > diag_l:
                    new_mat[row, col] = mat[row ,col]    
    return new_mat

def align_pairs_local(ref_seq, other_seq, score_only = False):
    """Performs pairwise alignment give two sequences

    Parameters
    ----------
        ref_seq : str
            Reference sequence
        other_seq : str
            Sequence to be aligned to reference
        biomolecule : str
            Sequences type, protein or RNA

    Returns
    -------
        alignments : tuple
            A tuple of pairwise aligned sequences, alignment score and
            start and end indices of alignment
    """
    scoring_mat = blosum62
    GAP_OPEN_PEN = -10
    GAP_EXTEND_PEN = -1
    alignments = pairwise2.align.localds(
        ref_seq,
        other_seq,
        scoring_mat,
        GAP_OPEN_PEN,
        GAP_EXTEND_PEN,
        score_only = score_only,
    )

    return alignments


def find_matching_seqs_from_alignment(sequences, ref_sequence):
    """Finds the best matching sequences to the reference
    sequence in the alignment. If multiple matching sequences
    are found, the first one (according to the order in the MSA)
    is taken

    Parameters
    ----------
	sequence : 
		npy array of arraysof letters/numbers (multiple sequence alignment)
		DO NOT NEED TO BE ALIGNED
	ref_sequence : 
		npy array of letters/numbers (reference sequence)
    Returns
    -------
        best_matching_seqs : list
            A list of best matching sequences to reference
    """

    # if the first sequence (gaps removed) in MSA matches with reference,
    # return this sequence.
    first_seq_in_alignment = sequences[0] 
    #first_seq_in_alignment_gaps_removed = first_seq_in_alignment.replace('-','')
    first_seq_in_alignment_gaps_removed = find_and_replace(first_seq_in_alignment, '-','')
    if first_seq_in_alignment_gaps_removed == ref_sequence:
        print('\n\tFirst sequence in alignment (gaps removed) matches reference,'
            '\n\tSkipping regorous search for matching sequence'
        )
        first_seq = list()
        first_seq.append(first_seq_in_alignment)
        return first_seq
    pairwise_scores = []
    for seq_indx, seq in enumerate(sequences):
        #seq_gaps_removed = seq.replace('-','')
        seq_gaps_removed = find_and_replace(seq, '-', '')
        print(seqs_gaps_removed)

        score = align_pairs_local(
            ref_sequence,
            seq_gaps_removed,
            score_only = True,
            )
        score_at_indx = (seq_indx, score)
        pairwise_scores.append(score_at_indx)

    seq_indx, max_score = max(pairwise_scores, key=lambda x: x[1])
    matching_seqs_indx = [
        indx  for indx, score in pairwise_scores if score == max_score
    ]

    best_matching_seqs = [
        sequences[indx] for indx in matching_seqs_indx
    ]
    num_matching_seqs = len(best_matching_seqs)
    if num_matching_seqs > 1 :
        print('\n\tFound %d sequences in MSA that match the reference'
            '\n\tThe first sequence is taken as matching'% num_matching_seqs
        )
    return best_matching_seqs



def scores_matrix2dict(scores_matrix, s_index, curated_cols=None):
    """
    # This functions converts the matrix of ER dca scores to the pydca-format dictionary (with 2 int index tuple as keys) 
    #   incorporates the removed columns during pre-processing (cols_removed) resulting in a pydca di matrix with
    #   correct dimensions
    """
    scores = []
    if curated_cols is not None:
        s_index = np.delete(s_index, curated_cols)
    for i in range(len(s_index)):
        for j in range(i+1, len(s_index)):
            # print(s_index[i],s_index[j])
            scores.append([(s_index[i], s_index[j]), scores_matrix[i,j]])

    sorted_scores = sorted(scores, key = lambda k : k[1], reverse=True)
    return sorted_scores
   
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
def score_APC(scores_matrix, N, s_index):
    """
    Performs Average Product Correction for a set of scores.

    takes:
    N: sequence length
    s_index: index of scores_matrix columns and rows which helps create the correct score pairs

    """
 
    scores = scores_matrix2dict(scores_matrix, s_index)
    # compute the average score of each site
    print('scores for %d pairs ' % len(scores))
    av_score_sites = list()
    for i in range(len(s_index)):
        i_scores = [score for pair, score in scores if s_index[i] in pair]
        assert len(i_scores) == len(s_index) - 1
        i_scores_sum = sum(i_scores)
        i_scores_ave = i_scores_sum/float(len(s_index) - 1)
        av_score_sites.append(i_scores_ave)
    # compute average product corrected DI
    av_all_scores = sum(av_score_sites)/float(len(s_index))
    sorted_score_APC = list()
    for pair, score in scores:
        i, j = pair
        i = np.where(s_index==i)[0][0]
        j = np.where(s_index==j)[0][0]
        score_apc = score - av_score_sites[i] * (av_score_sites[j]/av_all_scores)
        sorted_score_APC.append((pair, score_apc))
    # sort the scores as doing APC may have disrupted the ordering
    sorted_score_APC = sorted(sorted_score_APC, key = lambda k : k[1], reverse=True)
    return sorted_score_APC

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


def di_dict2mat(pydca_score, s_index, curated_cols = None, full_contact=False, aa_index_correction=True, cols_removed=False):
    # This functions converts the dictionary (with 2 int index tuple as keys) of pydca scores to a di matrix which
    #   incorporates the removed columns during pre-processing (cols_removed) resulting in a pydca di matrix with
    #   correct dimensions
    #for pair, score in pydca_score:
    #    print(pair, score)

    if full_contact:
        column_count = len(s_index) + len(cols_removed)
    else:
        column_count = len(s_index)

    pydca_di = np.zeros((column_count, column_count))
    # ijs = []
    for [(i, j), score] in pydca_score:
        # ijs.append(i)
        # ijs.append(j)
        if aa_index_correction:
            if cols_removed:
                ii = np.where(s_index==i-1)[0][0]
                jj = np.where(s_index==j-1)[0][0]
                pydca_di[ii, jj] = score
                pydca_di[jj, ii] = score
            else:
                pydca_di[i-1, j-1] = score
                pydca_di[j-1, i-1] = score

        else:
            pydca_di[i, j] = score
            pydca_di[j, i] = score

    # print('max index: ', max(ijs))
    print('DI shape (full size)' , pydca_di.shape)
    if cols_removed is not None:
        # trim the columns removed during the pre-processing for ER
        pydca_di = np.delete(pydca_di, cols_removed, 0)
        pydca_di = np.delete(pydca_di, cols_removed, 1)
    print('DI shape (scores removed)', pydca_di.shape,'\n(should be same as ER di shape..)')

    return pydca_di

# -------------------------------------------------------------------------------------------------------------------- #



 #
def npy2fa(data_path, pdb2msa_row, pdb_data_dir, index_pdb=0, n_cpu=4, create_new=True, processed_data_dir='./'): # letter_format is True
    # creates output for pydca to use

    print('\n\nnpy2fa_pdb2msa pdb2msa_row: ', pdb2msa_row)
    print('\n\n')
    dp_result =  data_processing(data_path, pdb2msa_row,\
                                         gap_seqs=0.2, gap_cols=0.2, prob_low=0.004,
                                         conserved_cols=0.8, printing=True, \
                                         out_dir=processed_data_dir, pdb_dir=pdb_data_dir,\
                                         letter_format=True, remove_cols=False, create_new=True,\
                                         n_cpu=min(2, n_cpu))   


    [s0, removed_cols, s_index, tpdb, pdb_s_index] = dp_result
    # # -------------------------------- # #

    # we still want to remove bad sequences
    # - Removing bad sequences (>gap_seqs gaps) -------------------- #
    from data_processing import remove_bad_seqs
    # removes all sequences (rows) with >gap_seqs gap %
    s, tpdb = remove_bad_seqs(s0, tpdb, .2, trimmed_by_refseq=False)
    # -------------------------------------------------------------- #

    # # ------- Write to FASTA --------- # #
    # Next save MSA to FASTA file
    pfam_id = pdb2msa_row['Pfam']
    print(processed_data_dir)
    msa_outfile = Path(processed_data_dir, '%s_msa_raw.fa' % pfam_id)

    with open(str(msa_outfile), 'w') as fh:
        for seq_num, seq in enumerate(s):
            msa_list = seq.tolist()
            msa_str = ''
            msa_str = msa_str.join(msa_list)
            if seq_num == tpdb:
                fasta_header = pfam_id + ' | REFERENCE'
            else:
                fasta_header = pfam_id + 'seq %d' % seq_num
            fh.write('>{}\n{}\n'.format(fasta_header, msa_str))
            
            
    # # ---- Get Reference Seq --------- # #                                                                                            
    
    print('Reference sequence (tpdb, s_ipdb) is sequence # %d' % tpdb)                                                                  
    gap_pdb = s[tpdb] == '-'  # returns True/False for gaps/no gaps in reference sequence                                           
    s_gap = s[:, ~gap_pdb]  # removes gaps in reference sequence
    ref_s = s_gap[tpdb]
    print("shape of s: ", s.shape)                                                                        
    print("shape of ref_s: ", ref_s.shape)
    # print(ref_s)

    # # -------------------------------- # #                                                                                            
    ref_outfile = Path(processed_data_dir, '%s_ref.fa' % pfam_id)
    with open(str(ref_outfile), 'w') as fh:
        ref_str = ''
#         ref_list = ref_s.tolist()
        ref_list = [char for char in ref_s]
        ref_str = ref_str.join(ref_list)
        fh.write('>{}\n{}\n'.format(pfam_id + ' | REFERENCE', ref_str))
    print(len(ref_str))
        # Reference string now gotten from geting PDB match in ecc_tools.find_best_pdb()
#         fh.write('>{}\n{}\n'.format(pfam_id + ' | REFERENCE', ref_seq))
    # # -------------------------------- # #


    return msa_outfile, ref_outfile, s, tpdb, removed_cols, s_index, pdb_s_index


# -------------------------------------------------------------------------------------------------------------------- #
# read fasta file of msa and ref (if given) where ref is inserted in correcto row (according to pdb) 
def read_FASTA(msa_file, ref_index=None):
    print('\nLoading Fasta File\n%s\n' % msa_file)
    for i, seq_record in enumerate(SeqIO.parse(msa_file, "fasta")):
        if ref_index is not None and i==ref_index:
            print('Reference Sequence (index %d) : ' % ref_index, seq_record.id)
            print(np.array(seq_record.seq), '\n\n')
            # print(repr(seq_record.seq))
            # print(len(seq_record))
            

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# write basic fasta file for given list
def write_FASTA(ref_seq, msa, pfam_id, number_form=True, processed=True, path='./', nickname=''):
    # Processed MSA to file in FASTA format
    if len(nickname) > 0:
        msa_outfile = path + 'MSA_' + pfam_id + '_' + nickname + '.fa'
        # Reference sequence to file in FASTA format
        ref_outfile = path + 'PP_ref_' + pfam_id + '_' + nickname + '.fa'
    else:
        msa_outfile = path + 'MSA_' + pfam_id + '.fa'
        # Reference sequence to file in FASTA format
        ref_outfile = path + 'PP_ref_' + pfam_id + '.fa'

    # print("Reference Sequence (shape=",msa[ref_seq].shape,"):\n",msa[ref_seq])

    if number_form:
        msa_letters = convert_number2letter(msa)
        ref_array = convert_numer2letter(ref_seq)
        if not processed:
            gap_ref = ref_array == '-'  # remove gaps from reference array
            ref_letters = msa_letters[ref_seq][~gap_ref]
        else:
            ref_letters = msa_letters[ref_seq]
    else:
        msa_letters = msa
    if printing:
        print('\n\n#-----------------------#\nGenerating Contact Map\n#----------------------------#\n')

    pdb_file = pdb_list.retrieve_pdb_file(str(pdb_id), file_format='pdb', pdir=pdb_out_dir)
    # pdb_file = pdb_list.retrieve_pdb_file(pdb_id)

    pdb_start = pdb_range[0] - 2
    pdb_end = pdb_range[1] - 1

    pdb_model = pdb_parser.get_structure(str(pdb_id), pdb_file)[0]
    found_pp_match = False
    print(queried_seq)
    for chain in pdb_model.get_chains():
        ppb = PPBuilder().build_peptides(chain)

        # # PYDCA method for getting polypeptide sequence...
        poly_seq_new = [res.get_resname().strip() for res in filter_residues(pdb_model[chain.get_id()].get_list())]
        print('new poly seq list: ', ''.join(poly_seq_new))

        # Get full list of CA coords from poly_seq
        poly_seq = list() 
        pp_ca_coords_full = list()
        for i, pp in enumerate(ppb):
            for char in str(pp.get_sequence()):
                poly_seq.append(char)
            poly_seq_ca_atoms = pp.get_ca_list()
            pp_ca_coords_full.extend([a.get_coord() for a in poly_seq_ca_atoms])      

        print('\nChain ', chain, ':\n', ''.join(poly_seq))
        poly_seq_range = poly_seq[pdb_start:pdb_end]
        print( '\n',''.join(poly_seq_range), '\n') 


        #str_index = ''.join(poly_seq).find(queried_seq)
        #if str_index == -1:
        #    continue
        # # Search polyseq using haming distance..
        for str_index, aa_window in enumerate(window(''.join(poly_seq), len(queried_seq))):
            ham_dist = hamming_distance(queried_seq, aa_window)
            if ham_dist > mismatches:
                continue 
            else:
                found_pp_match = True
                pdb_start = str_index
                pdb_end = str_index + len(queried_seq)
                poly_seq_range = poly_seq[pdb_start:pdb_end]
                print('Found Match!\n')
                print(''.join(poly_seq_range))
                print(queried_seq)
                break

    if not found_pp_match:
        print('looking for %d mismatches' % mismatches)
        print(''.join(poly_seq_range))
        print(len(poly_seq_range))
        print(queried_seq)
        print(len(queried_seq))
        print('ERROR: downloaded pdb sequence does not match the returned sequence from the query')
        sys.exit()
    
    pdb_chain = chain.get_id()
    n_amino_full = len(pp_ca_coords_full)

    # Extract coordinates and sequence char in PDB-range\
    pp_ca_coords_full_range = pp_ca_coords_full[pdb_start:pdb_end]

    ct_full = distance_matrix(pp_ca_coords_full, pp_ca_coords_full)



    poly_seq_curated = np.delete(poly_seq_range, removed_cols)
    pp_ca_coords_curated = np.delete(pp_ca_coords_full_range, removed_cols, axis=0)
    ct = distance_matrix(pp_ca_coords_curated, pp_ca_coords_curated)

    return pdb_chain, ct, ct_full, n_amino_full, poly_seq_curated, poly_seq_range, poly_seq, pp_ca_coords_curated, pp_ca_coords_full_range

def di_dict2mat(pydca_score, s_index, curated_cols = None, full_contact=False, aa_index_correction=True, removing_cols=False):
    # This functions converts the dictionary (with 2 int index tuple as keys) of pydca scores to a di matrix which
    #   incorporates the removed columns during pre-processing (cols_removed) resulting in a pydca di matrix with
    #   correct dimensions
    #for pair, score in pydca_score:                                                                     
    #    print(pair, score)

    if full_contact:
        column_count = len(s_index) + len(cols_removed)                                                  
    else:
        column_count = len(s_index)                                                                      
    
    pydca_di = np.zeros((column_count, column_count))
    # ijs = []                                                                                           
    for [(i, j), score] in pydca_score:                                                                  
        # ijs.append(i)                                                                                  
        # ijs.append(j)
        if aa_index_correction:                                                                          
            if removing_cols:
                if i-1 in s_index and j-1 in s_index:
                    ii = np.where(s_index==i-1)[0][0]                                                        
                    jj = np.where(s_index==j-1)[0][0]  
                    pydca_di[ii, jj] = score                                                                 
                    pydca_di[jj, ii] = score                                                                 
            else:
                pydca_di[i-1, j-1] = score
                pydca_di[j-1, i-1] = score                                                               
                
        else:
            pydca_di[i, j] = score 
            pydca_di[j, i] = score 
                                                                                                         
    # print('max index: ', max(ijs))                                                                     
    print('DI shape (full size)' , pydca_di.shape)
    if curated_cols is not None and not removing_cols:   # have we curated (set to gap instead of removed) 
                                                        # but not removed columns?
        # trim the columns removed during the pre-processing for ER                                      
        pydca_di = np.delete(pydca_di, curated_cols, 0)                                                  
        pydca_di = np.delete(pydca_di, curated_cols, 1)                                                  
    print('DI shape (scores removed)', pydca_di.shape,'\n(should be same as ER di shape..)')             
        
    return pydca_di


# print("Direct Information from Expectation reflection:\n",di)
def no_diag(mat, diag_l, s_index=None, make_big=False):
    rows, columns = mat.shape
    if make_big:
        new_mat = 100. * np.ones((rows,columns))
    else:
        new_mat = np.zeros((rows,columns))
    for row in range(rows):
        for col in range(columns):
            if s_index is None:
                if abs(row-col) > diag_l:
                    new_mat[row, col] = mat[row ,col]
            else:
                if abs(s_index[row]-s_index[col]) > diag_l:
                    new_mat[row, col] = mat[row ,col]    
    return new_mat



def contact_map_pdb2msa(pdb_df, pdb_file, removed_cols, pdb_out_dir='./', printing=True):
    if printing:
        print('\n\n#-----------------------#\nGenerating Contact Map\n#----------------------------#\n')

    pdb_id = pdb_df['PDB ID']
    # pdb_file = pdb_list.retrieve_pdb_file(pdb_id)

    pdb_start = pdb_df['start'] - 1
    pdb_end = pdb_df['end'] 
    pdb_chain = pdb_df['Chain']
    pdb_pp_index = pdb_df['Polypeptide Index']

    pdb_model = pdb_parser.get_structure(str(pdb_id), pdb_file)[0]
    found_pp_match = False
    for chain in pdb_model.get_chains():
        if chain.get_id() == pdb_chain:
            pass
        else:
            continue
        ppb = PPBuilder().build_peptides(chain)

        # # PYDCA method for getting polypeptide sequence...
        poly_seq_new = [res.get_resname().strip() for res in filter_residues(pdb_model[chain.get_id()].get_list())]
        print('new poly seq list: ', ''.join(poly_seq_new))

        # Get full list of CA coords from poly_seq
        poly_seq = list()
        pp_ca_coords_full = list()
        for i, pp in enumerate(ppb):
            if i == pdb_pp_index:
                pass
            else:
                continue
            for char in str(pp.get_sequence()):
                poly_seq.append(char)
            poly_seq_ca_atoms = pp.get_ca_list()
            pp_ca_coords_full.extend([a.get_coord() for a in poly_seq_ca_atoms])

        print('\nChain ', chain, ':\n', ''.join(poly_seq))
        poly_seq_range = poly_seq[pdb_start:pdb_end]
        print( '\n',''.join(poly_seq_range), '\n')
        print('poly_seq_range (%d)' % len(poly_seq_range))
        print('pp seq coordinates (%d)' % len(pp_ca_coords_full))

    n_amino_full = len(pp_ca_coords_full)

    # Extract coordinates and sequence char in PDB-range\
    pp_ca_coords_full_range = pp_ca_coords_full[pdb_start:pdb_end]
    print('pp seq coordinates in range (%d)' % len(pp_ca_coords_full_range))

    ct_full = distance_matrix(pp_ca_coords_full, pp_ca_coords_full)
    removed_cols_range = np.array([col-pdb_start for col in removed_cols if col<=pdb_end and col>=pdb_start])
    print('curating removed columns in range..(%d -> %d)\n'% (len(removed_cols), len(removed_cols_range)))
    print('poly_seq_range: ', len(poly_seq_range))
    poly_seq_curated = np.delete(poly_seq_range, removed_cols_range)
    pp_ca_coords_curated = np.delete(pp_ca_coords_full_range, removed_cols_range, axis=0)
    ct = distance_matrix(pp_ca_coords_curated, pp_ca_coords_curated)

    return ct, ct_full, n_amino_full, poly_seq_curated, poly_seq_range, poly_seq, pp_ca_coords_curated, pp_ca_coords_full_range, removed_cols_range


def contact_map_pdb2msa_new(pdb_df, pdb_file, removed_cols, pdb_s_index, pdb_out_dir='./', printing=True):
    if printing:
        print('\n\n#-----------------------#\nGenerating Contact Map\n#----------------------------#\n')

    pdb_id = pdb_df['PDB ID']
    # pdb_file = pdb_list.retrieve_pdb_file(pdb_id)

    pdb_start = pdb_df['ali_start'] - 1
    pdb_end = pdb_df['ali_end'] 
    print(pdb_start, pdb_end)
    #shifted_pdb_s_index = [col+ pdb_start for col in pdb_s_index]
    #print('PDB sequence columns in di: ', shifted_pdb_s_index)
    pdb_chain = pdb_df['Chain']
    pdb_pp_index = pdb_df['Polypeptide Index']

    pdb_model = pdb_parser.get_structure(str(pdb_id), pdb_file)[0]
    found_pp_match = False
    for chain in pdb_model.get_chains():
        if chain.get_id() == pdb_chain:
            pass
        else:
            continue
        ppb = PPBuilder().build_peptides(chain)

        # # PYDCA method for getting polypeptide sequence...
        poly_seq_new = [res.get_resname().strip() for res in filter_residues(pdb_model[chain.get_id()].get_list())]
        print('new poly seq list: ', ''.join(poly_seq_new))

        # Get full list of CA coords from poly_seq
        poly_seq = list()
        pp_ca_coords_full = list()
        for i, pp in enumerate(ppb):
            if i == pdb_pp_index:
                pass
            else:
                continue
            for char in str(pp.get_sequence()):
                poly_seq.append(char)
            poly_seq_ca_atoms = pp.get_ca_list()
            pp_ca_coords_full.extend([a.get_coord() for a in poly_seq_ca_atoms])

        print('\nChain ', chain, ':\n', ''.join(poly_seq))
        poly_seq_range = poly_seq[pdb_start:pdb_end]
        print( '\n',''.join(poly_seq_range), '\n')
        print('poly_seq_range (%d)' % len(poly_seq_range))
        print('pp seq coordinates (%d)' % len(pp_ca_coords_full))

        # NEW getting aligned poly seq and coords
        aligned_poly_seq  = poly_seq[pdb_start:pdb_end]
        aligned_poly_seq = [aligned_poly_seq[i] for i in pdb_s_index]
        print('Aligned poly_seq: (len %d)' % len(aligned_poly_seq), aligned_poly_seq)
        aligned_ca_coords = pp_ca_coords_full[pdb_start:pdb_end]
        aligned_ca_coords = [aligned_ca_coords[i] for i in pdb_s_index]
        print('Aligned pp ca coords len %d' % len(aligned_ca_coords))

        
    n_amino_full = len(pp_ca_coords_full)

    # Extract coordinates and sequence char in PDB-range\

    ct_full = distance_matrix(pp_ca_coords_full, pp_ca_coords_full)
    
    # NEW create ct using Aligned coordinates
    ct = distance_matrix(aligned_ca_coords, aligned_ca_coords)

    return ct, ct_full



def contact_map(pdb, ipdb, pp_range, cols_removed, s_index, ref_seq=None, printing=True, pdb_out_dir='./', refseq=None):
    if printing:
        print('\n\n#-----------------------#\nGenerating Contact Map\n#----------------------------#\n')

    print('Checking %d PDB sequence for best match to reference sequence' % pdb.shape[0])
    print(pdb[ipdb, :])
    pdb_id = pdb[ipdb, 5]
    pdb_chain = pdb[ipdb, 6]
    # pdb_start,pdb_end = int(pdb[ipdb,6])-1,int(pdb[ipdb,8])-1 # -1 due to array-indexing
    pdb_start_index, pdb_end_index = int(pp_range[0] - 1), int(pp_range[1] - 1)
    pdb_start = pdb_start_index

    # TODO: Figure out the indexing
    pdb_end = pdb_end_index
    pdb_end = pdb_end_index + 1

    # print('pdb id, chain, start, end, length:',pdb_id,pdb_chain,pdb_start,pdb_end,pdb_end-pdb_start+1)

    # print('download pdb file')
    pdb_file = pdb_list.retrieve_pdb_file(str(pdb_id), file_format='pdb', pdir=pdb_out_dir)
    # pdb_file = pdb_list.retrieve_pdb_file(pdb_id)

    # ------------------------------------------------------------------------------------------------- #

    chain = pdb_parser.get_structure(str(pdb_id), pdb_file)[0][pdb_chain]
    good_coords = []
    coords_all = np.array([a.get_coord() for a in chain.get_atoms()])
    ca_residues = np.array([a.get_name() == 'CA' for a in chain.get_atoms()])
    ca_coords = coords_all[ca_residues]
    in_range_ca_coords = ca_coords[pdb_start:pdb_end+1] ## change 1/18/22 ecc
    n_amino = len(in_range_ca_coords)

    ppb = PPBuilder().build_peptides(chain)
    #    print(pp.get_sequence())
    if printing:
        print('peptide build of chain produced %d elements' % (len(ppb)))

    # Get full list of CA coords from poly_seq
    poly_seq = list()
    pp_ca_coords_full = list()
    for i, pp in enumerate(ppb):
        for char in str(pp.get_sequence()):
            poly_seq.append(char)
        poly_seq_ca_atoms = pp.get_ca_list()
        pp_ca_coords_full.extend([a.get_coord() for a in poly_seq_ca_atoms])

    n_amino_full = len(pp_ca_coords_full)
    if printing:
        print('original poly_seq: \n', poly_seq, '\n length: ', len(poly_seq))
        print('Polypeptide Ca Coord list: ', len(pp_ca_coords_full), '\n length: ', len(pp_ca_coords_full))

    # Extract coordinates and sequence char in PDB-range
    print('length of pp_ca_coords_full ', len(pp_ca_coords_full))
    print('pdb range %d, %d' % (pdb_start, pdb_end))
    pp_ca_coords_full_range = pp_ca_coords_full[pdb_start:pdb_end + 1]
    print('length of pp_ca_coords_full ', len(pp_ca_coords_full_range))
    poly_seq_range = poly_seq[pdb_start:pdb_end + 1]
    if printing:
        print('\n\nExtracting pdb-range from full list of polypeptide coordinates')
        print('PDB-range poly_seq: \n', poly_seq_range, '\n length: ', len(poly_seq_range))
        print('PDB-range Polypeptide Ca Coord list: ', len(pp_ca_coords_full_range), '\n length: ',
              len(pp_ca_coords_full_range))
        print('\n\n')

        # Get curated polypeptide sequence (using cols_removed) & Get coords of curated pp seq
    poly_seq_curated = np.delete(poly_seq_range, cols_removed)
    pp_ca_coords_curated = np.delete(pp_ca_coords_full_range, cols_removed, axis=0)

    
    if printing:
        print('Sequence inside given PDB range (poly_seq_range) with columns removed: (poly_seq_range_curated): \n',
              poly_seq_curated, '\n length: ', len(poly_seq_curated))
        print('Curated Polypeptide Ca Coord list: ', len(pp_ca_coords_curated), '\n length: ',
              len(pp_ca_coords_curated))
        print('\n\ns_index len: ', len(s_index))
        print(
            "\n\n#---------#\nNumber of good amino acid coords: %d should be same as sum of s_index and cols_removed" % n_amino)
        print("s_index and col removed len %d " % (len(s_index) + len(cols_removed)))
        print(
            'all coords %d\nall ca coords: %d\nprotein rangs ca coords len: %d\npp ca coords: %d\npp ca curated coords len: %d\n#----------#\n\n' % (
                len(coords_all), len(ca_coords), len(in_range_ca_coords), len(pp_ca_coords_full),
                len(pp_ca_coords_curated)))

    ct_full = distance_matrix(in_range_ca_coords, in_range_ca_coords)
    ct_full = distance_matrix(pp_ca_coords_full, pp_ca_coords_full)
    coords = np.array(in_range_ca_coords)

    # -------------------------------------------------------------------------------------------------#
    coords_remain = np.delete(coords, cols_removed, axis=0)
    # print(coords_remain.shape)

    ct = distance_matrix(coords_remain, coords_remain)
    ct = distance_matrix(pp_ca_coords_curated, pp_ca_coords_curated)
    if printing:
        print(coords_remain.shape)
        print('\n\n#-----------------------#\nContact Map Generated \n#----------------------------#\n')

    return ct, ct_full, n_amino_full, poly_seq_curated, poly_seq_range


def roc_curve_new(ct, di, ct_thres):
    """
    ct: n x n matrix of True/False for contact

    di: n x n matrix of coupling info where n is number of positions with contact predictions

    ct_trhes:  distance for contact threshold
    """
    ct1 = ct.copy()

    ct_pos = ct1 < ct_thres
    ct1[ct_pos] = 1
    ct1[~ct_pos] = 0

    mask = np.triu(np.ones(di.shape[0], dtype=bool), k=1)
    # argsort sorts from low to high. [::-1] reverses 
    order = di[mask].argsort()[::-1]
    ct_flat = ct1[mask][order]
    # print("ct_flat dimensions: ", np.shape(ct_flat))
    # print(di[mask][order][:50])
    # print(ct_flat[:50])
    fpr, tpr, thresholds = roc_scikit(ct_flat, di[mask][order])
    roc_auc= auc(fpr, tpr)
    print('ct thresh %f gives auc = %f' % (ct_thres, roc_auc))
    return fpr, tpr, thresholds, roc_auc

def precision_curve(ct, di, ct_thres):
    """
    ct: n x n matrix of True/False for contact

    di: n x n matrix of coupling info where n is number of positions with contact predictions

    ct_trhes:  distance for contact threshold
    """
    ct1 = ct.copy()

    ct_pos = ct1 < ct_thres
    ct1[ct_pos] = 1
    ct1[~ct_pos] = 0

    mask = np.triu(np.ones(di.shape[0], dtype=bool), k=1)
    # argsort sorts from low to high. [::-1] reverses 
    order = di[mask].argsort()[::-1]
    ct_flat = ct1[mask][order]
    # print("ct_flat dimensions: ", np.shape(ct_flat))
    # print(di[mask][order][:50])
    # print(ct_flat[:50])
    precision, recall, thresholds = precision_recall_curve(ct_flat, di[mask][order])
    return precision, recall, thresholds

 
def roc_curve(ct, di, ct_thres):
    """
    ct: n x n matrix of True/False for contact

    di: n x n matrix of coupling info where n is number of positions with contact predictions

    ct_trhes:  distance for contact threshold
    """
    ct1 = ct.copy()

    ct_pos = ct1 < ct_thres
    ct1[ct_pos] = 1
    ct1[~ct_pos] = 0

    mask = np.triu(np.ones(di.shape[0], dtype=bool), k=1)
    # argsort sorts from low to high. [::-1] reverses 
    order = di[mask].argsort()[::-1]
    print("order dimensions: ", np.shape(order))
    print(order[:50])
    ct_flat = ct1[mask][order]
    print("ct_flat dimensions: ", np.shape(ct_flat))
    print(di[mask][order][:50])
    print(ct_flat[:50])

    tp = np.cumsum(ct_flat, dtype=float)
    fp = np.cumsum(~ct_flat.astype(int), dtype=float)
    print('last tp and fp cumsum vals:')
    print(tp[-1])
    print(fp[-1])

    if tp[-1] != 0:
        tp /= tp[-1]
        fp /= fp[-1]

    # Binning (to reduce the size of tp,fp and make fp having the same values for every Pfam)
    nbin = 101
    pbin = np.linspace(0, 1, nbin, endpoint=True)

    # print(pbin)

    fp_size = fp.shape[0]

    fpbin = np.ones(nbin)
    tpbin = np.ones(nbin)
    for ibin in range(nbin - 1):
        # find value in a range
        t1 = [(fp[t] > pbin[ibin] and fp[t] <= pbin[ibin + 1]) for t in range(fp_size)]

        if len(t1) > 0:
            fpbin[ibin] = fp[t1].mean()
            tpbin[ibin] = tp[t1].mean()
            # try:
            #     fpbin[ibin] = fp[t1].mean()
            # except RuntimeWarning:
            #     # print("Empty mean slice")
            #     fpbin[ibin] = 0
            # try:
            #     tpbin[ibin] = tp[t1].mean()
            # except RuntimeWarning:
            #     # print("Empty mean slice")
            #     tpbin[ibin] = 0
        else:
            # print(i)
            tpbin[ibin] = tpbin[ibin - 1]
            # print(fp,tp)
    # return fp,tp,pbin,fpbin,tpbin
    return pbin, tpbin, fpbin


on_pc = False
if on_pc:
    from IPython.display import HTML


    def hide_toggle(for_next=False):
        this_cell = """$('div.cell.code_cell.rendered.selected')"""
        next_cell = this_cell + '.next()'

        toggle_text = 'Toggle show/hide'  # text shown on toggle link
        target_cell = this_cell  # target cell to control with toggle
        js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

        if for_next:
            target_cell = next_cell
            toggle_text += ' next cell'
            js_hide_current = this_cell + '.find("div.input").hide();'

        js_f_name = 'code_toggle_{}'.format(str(random.randint(1, 2 ** 64)))

        html = """
        <script>
            function {f_name}() {{
            {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
        """.format(
            f_name=js_f_name,
            cell_selector=target_cell,
            js_hide_current=js_hide_current,
            toggle_text=toggle_text
        )

        return HTML(html)


# =========================================================================================
def distance_restr_sortedDI(sorted_DI_in, s_index=None):
    # if s_index is passes the resulting tuple list will be properly indexed
    sorted_DI = sorted_DI_in.copy()
    count = 0
    for site_pair, score in sorted_DI_in:

        # if s_index exists re-index sorted pair
        if s_index is not None:
            pos_0 = s_index[site_pair[0]]
            pos_1 = s_index[site_pair[1]]
        else:
            pos_0 = site_pair[0]
            pos_1 = site_pair[1]
            print('MAKE SURE YOUR INDEXING IS CORRECT!!')
            print('         or pass s_index to distance_restr_sortedDI()')

        if abs(pos_0 - pos_1) < 5:
            sorted_DI[count] = (pos_0, pos_1), 0
        else:
            sorted_DI[count] = (pos_0, pos_1), score
        count += 1
    sorted_DI.sort(key=lambda x: x[1], reverse=True)
    return sorted_DI


# =========================================================================================
def distance_restr(di, s_index, make_large=False):
    # Hamstring DI matrix by setting all DI values st |i-j|<5 to 0
    if di.shape[0] != s_index.shape[0]:
        print("Distance restraint cannot be imposed, bad input")
        # IndexError: index 0 is out of bounds for axis 0 with size 0
        print("s_index: ", s_index.shape[0], "di shape: ", di.shape[0])
        # print('di:\n',di[0])
        raise IndexError("matrix input dimensions do not matchup with simulation n_var")
    di_distal = np.zeros(di.shape)
    for i in range(di.shape[0]):
        for j in range(di.shape[1]):
            if (abs(s_index[i] - s_index[j]) < 5):
                if make_large:
                    di_distal[i][j] = 35.
                else:
                    di_distal[i][j] = 0.
            else:
                di_distal[i][j] = di[i][j]

    return di_distal


# =========================================================================================
def distance_restr_ct(ct, s_index, make_large=False):
    # Hamstring DI matrix by setting all DI values st |i-j|<5 to 0
    if 0:
        if ct.shape[0] <= s_index[-1]:
            print("ERROR in distance_restr_ct\n\nDistance restraint cannot be imposed, bad input\n")
            # IndexError: index 0 is out of bounds for axis 0 with size 0
            print("s_index max index: ", s_index[-1], "ct shape: ", ct.shape[0])
            # print('di:\n',di[0])
            raise IndexError("matrix input dimensions do not matchup with simulation n_var")
    ct_distal = np.zeros(ct.shape)
    print('contact map shape: ', ct_distal.shape)
    try:
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                if (abs(s_index[i] - s_index[j] < 5)):
                    if make_large:
                        ct_distal[i][j] = 35.
                    else:
                        ct_distal[i][j] = 0.
                else:
                    ct_distal[i][j] = ct[i][j]
    except(IndexError):
        print('ct shape: ', ct.shape, '  s_index length: ', len(s_index))
        print('INDEX ERROR IN ECC_TOOLS:DISTANCE_RESTR_CT')
        raise IndexError("matrix input dimensions do not matchup with simulation n_var")
    return ct_distal


# ER coupling setup 6/20/2020
def compute_sequences_weight(alignment_data=None, seqid=None):
    """Computes weight of sequences. The weights are calculated by lumping
    together sequences whose identity is greater that a particular threshold.
    For example, if there are m similar sequences, each of them will be assigned
    a weight of 1/m. Note that the effective number of sequences is the sum of
    these weights.

    Parameters
    ----------
        alignmnet_data : np.array()
            Numpy 2d array of the alignment data, after the alignment is put in
            integer representation
        seqid : float
            Value at which beyond this sequences are considered similar. Typical
            values could be 0.7, 0.8, 0.9 and so on

    Returns
    -------
        seqs_weight : np.array()
            A 1d numpy array containing computed weights. This array has a size
            of the number of sequences in the alignment data.
    """
    alignment_shape = alignment_data.shape
    num_seqs = alignment_shape[0]
    seqs_len = alignment_shape[1]
    seqs_weight = np.zeros((num_seqs,), dtype=np.float64)
    # count similar sequences
    for i in parallel_range(num_seqs):
        seq_i = alignment_data[i]
        for j in range(num_seqs):
            seq_j = alignment_data[j]
            iid = np.sum(seq_i == seq_j)
            if np.float64(iid) / np.float64(seqs_len) > seqid:
                seqs_weight[i] += 1
    # compute the weight of each sequence in the alignment
    for i in range(num_seqs): seqs_weight[i] = 1.0 / float(seqs_weight[i])
    return seqs_weight


def compute_single_site_freqs(alignment_data=None, seqs_weight=None, mx=None):
    """Computes single site frequency counts for a particular aligmnet data.

    Parameters
    ----------
        alignment_data : np.array()
            A 2d numpy array of alignment data represented in integer form.

        num_site_states : int
            An integer value fo the number of states a sequence site can have
            including a gap state. Typical value is 5 for RNAs and 21 for
            proteins.

        seqs_weight : np.array()
            A 1d numpy array of sequences weight

    Returns
    -------
        single_site_freqs : np.array()
            A 2d numpy array of of data type float64. The shape of this array is
            (seqs_len, num_site_states) where seqs_len is the length of sequences
            in the alignment data.
    """
    alignment_shape = alignment_data.shape
    # num_seqs = alignment_shape[0]
    seqs_len = alignment_shape[1]
    if seqs_len != len(mx):
        print('sequence length = %d and mx length = %d' % (seqs_len, len(mx)))
    m_eff = np.sum(seqs_weight)
    # single_site_freqs = np.zeros(shape = (seqs_len, num_site_states),dtype = np.float64)
    single_site_freqs = []  # list form so its easier to handle varied num_site_states
    for i in range(seqs_len):
        # for a in range(1, num_site_states + 1):#we need gap states single site freqs too
        single_site_freqs.append([])
        num_site_states = mx[i]
        # print('seq position %d has %d states'%(i,num_site_states))
        column_i = alignment_data[:, i]
        for a in np.unique(column_i):  # we use varying site states (unique vals in col)
            # print('    a = ',a)
            # print(np.unique(column_i)) # what values are in column_i?
            freq_ia = np.sum((column_i == a) * seqs_weight)
            single_site_freqs[-1].append(freq_ia / m_eff)
    return single_site_freqs


def get_reg_single_site_freqs(single_site_freqs=None, seqs_len=None,
                              mx=None, pseudocount=None):
    """Regularizes single site frequencies.

    Parameters
    ----------
        single_site_freqs : np.array()
            A 2d numpy array of single site frequencies of shape
            (seqs_len, num_site_states). Note that gap state frequencies are
            included in this data.
        seqs_len : int
            The length of sequences in the alignment data
        num_site_states : int
            Total number of states that a site in a sequence can accommodate. It
            includes gap states.
        pseudocount : float
            This is the value of the relative pseudo count of type float.
            theta = lambda/(meff + lambda), where meff is the effective number of
            sequences and lambda is the real pseudo count.

    Returns
    -------
        reg_single_site_freqs : np.array()
            A 2d numpy array of shape (seqs_len, num_site_states) of single site
            frequencies after they are regularized.
    """
    reg_single_site_freqs = single_site_freqs
    for i in range(seqs_len):
        num_site_states = mx[i]
        theta_by_q = np.float64(pseudocount) / np.float64(num_site_states)
        for a in range(num_site_states):
            reg_single_site_freqs[i][a] = theta_by_q + (1.0 - pseudocount) * reg_single_site_freqs[i][a]
    return reg_single_site_freqs


# This function is replaced by the parallelized version below
def compute_pair_site_freqs_serial(alignment_data=None, mx=None,
                                   seqs_weight=None):
    """Computes pair site frequencies for an alignmnet data.

    Parameters
    ----------
        alignment_data : np.array()
            A 2d numpy array conatining alignment data. The residues in the
            alignment are in integer representation.
        num_site_states : int
            The number of possible states including gap state that sequence
            sites can accomodate. It must be an integer
        seqs_weight:
            A 1d numpy array of sequences weight

    Returns
    -------
        pair_site_freqs : np.array()
            A 3d numpy array of shape
            (num_pairs, num_site_states, num_site_states) where num_pairs is
            the number of unique pairs we can form from sequence sites. The
            pairs are assumed to in the order (0, 1), (0, 2) (0, 3), ...(0, L-1),
            ... (L-1, L). This ordering is critical and any change must be
            documented.
    """
    alignment_shape = alignment_data.shape
    num_seqs = alignment_shape[0]
    seqs_len = alignment_shape[1]
    num_site_pairs = (seqs_len - 1) * seqs_len / 2
    num_site_pairs = np.int64(num_site_pairs)
    m_eff = np.sum(seqs_weight)
    # pair_site_freqs = np.zeros(
    #    shape=(num_site_pairs, num_site_states - 1, num_site_states - 1),
    #    dtype = np.float64)
    pair_site_freqs = []  # list form so its easier to handle varied num_site_states
    pair_counter = 0
    for i in range(seqs_len - 1):
        column_i = alignment_data[:, i]
        i_site_states = mx[i]
        if len(np.unique(column_i)) != i_site_states:
            print('unique vals doesn\'match site states')
            sys.exit()

        for j in range(i + 1, seqs_len):
            column_j = alignment_data[:, j]
            j_site_states = mx[j]
            if len(np.unique(column_j)) != j_site_states:
                print('unique vals doesn\'match site states')
                sys.exit()
            pair_site_freqs.append([])

            for a in np.unique(column_i):
                pair_site_freqs[-1].append([])
                count_ai = column_i == a

                for b in np.unique(column_j):
                    count_bj = column_j == b
                    count_ai_bj = count_ai * count_bj
                    freq_ia_jb = np.sum(count_ai_bj * seqs_weight)
                    # pair_site_freqs[pair_counter, a-1, b-1] = freq_ia_jb/m_eff
                    pair_site_freqs[-1][-1].append(freq_ia_jb / m_eff)
            # move to the next site pair (i, j)
            pair_counter += 1
    if len(pair_site_freqs) != num_site_pairs:
        print('Not enough site pairs generated')
        sys.exit()
    return pair_site_freqs


# I think this is wahte msa_numerics uses to initialize weights..
# maybe we can use this to initialize our weights (w i think)
# what is w and what is its purpose!?!?!?!
def construct_corr_mat(reg_fi=None, reg_fij=None, seqs_len=None,
                       mx=None):
    """Constructs correlation matrix from regularized frequency counts.

    Parameters
    ----------
        reg_fi : np.array()
            A 2d numpy array of shape (seqs_len, num_site_states) of regularized
            single site frequncies. Note that only fi[:, 0:num_site_states-1] are
            used for construction of the correlation matrix, since values
            corresponding to fi[:, num_site_states]  are the frequncies of gap
            states.
        reg_fij : np.array()
            A 3d numpy array of shape (num_unique_pairs, num_site_states -1,
            num_site_states - 1), where num_unique_pairs is the total number of
            unique site pairs execluding self-pairings.
        seqs_len : int
            The length of sequences in the alignment
        num_site_states : int
            Total number of states a site in a sequence can accommodate.

    Returns
    -------
        corr_mat : np.array()
            A 2d numpy array of shape (N, N)
            where N = seqs_len * num_site_states -1
    """
    # corr_mat_len = seqs_len * (num_site_states - 1)
    corr_mat_len = mx.cumsum()[-1]
    print('Generating NxN correlation matrix with N=', corr_mat_len)
    corr_mat = np.zeros((corr_mat_len, corr_mat_len), dtype=np.float64)
    pair_counter = 0
    for i in range(seqs_len - 1):
        if i == 0:
            site_i = 0
        else:
            site_i = mx.cumsum()[i - 1]
        for j in range(i + 1, seqs_len):
            site_j = mx.cumsum()[j - 1]
            for a in range(mx[i]):
                row = site_i + a
                for b in range(mx[j]):
                    col = site_j + b
                    if i == j:
                        print('Iteration through non-symmetric reg_fij list is not working ')
                        sys.exit()
                    else:
                        try:
                            corr_ij_ab = reg_fij[pair_counter][a][b] - reg_fi[i][a] * reg_fi[j][b]
                        except IndexError:
                            print('pair %d: (%d,%d)' % (pair_counter, i, j))
                            print('Indices: ', mx.cumsum())
                            print('Site Counts: ', mx)
                            print('Index out of bound')
                            print('par ranges: a= [%d,%d],b= [%d,%d]' % (
                                site_i, site_i + range(mx[i])[-1], site_j, site_j + range(mx[j])[-1]))
                            print('pair_counter = %d of %d (%d)' % (pair_counter, len(reg_fij), len(reg_fij)))
                            print('i site state = %d of %d (%d)' % (a, mx[i], len(reg_fij[pair_counter])))
                            print(b)
                            sys.exit()
                    # print(corr_mat)
                    # print(corr_ij_ab)
                    try:
                        corr_mat[row, col] = corr_ij_ab
                        corr_mat[col, row] = corr_ij_ab
                    except IndexError:
                        print('ERROR: \n    row = %d of %d' % (row, mx.cumsum()[-1]))
                        print('       \n    col = %d of %d' % (col, mx.cumsum()[-1]))
                        sys.exit()

            if i != j: pair_counter += 1
    # fill in diagonal block
    for ii, site_block in enumerate(mx):
        if ii == 0:
            site_block_start = 0
        else:
            site_block_start = mx.cumsum()[ii - 1]
        for a in range(site_block):
            for b in range(a, site_block):
                row = site_block_start + a
                col = site_block_start + b
                # print('combo (%d,%d)'%(row,col))
                fia, fib = reg_fi[ii][a], reg_fi[ii][b]
                corr_ij_ab = fia * (1.0 - fia) if a == b else -1.0 * fia * fib
                corr_mat[row, col] = corr_ij_ab
                corr_mat[col, row] = corr_ij_ab

    return corr_mat


def compute_couplings(corr_mat=None):
    """Computes the couplings by inverting the correlation matrix

    Parameters
    ----------
        corr_mat : np.array()
            A numpy array of shape (N, N) where N = seqs_len *(num_site_states -1)
            where seqs_len  is the length of sequences in the alignment data and
            num_site_states is the total number of states a site in a sequence
            can accommodate, including gapped states.

    Returns
    -------
        couplings : np.array()
            A 2d numpy array of the same shape as the correlation matrix. Note
            that the couplings are the negative of the inverse of the
            correlation matrix.
    """
    couplings = np.linalg.inv(corr_mat)
    couplings = -1.0 * couplings
    return couplings


def slice_couplings(couplings=None, site_pair=None, mx=None):
    """Returns couplings corresponding to site pair (i, j). Note that the
    the couplings involving gaps are included, but they are set to zero.

    Parameters
    ----------
        couplings : np.array
            A 2d numpy array of couplings. It has a shape of (L(q-1), L(q-1))
            where L and q are the length of sequences in alignment data and total
            number of standard residues plus gap.
        site_pair : tuple
            A tuple of site pairs. Example (0, 1), (0, L-1), ..., (L-2, L-1).
        num_site_states : int
            The value of q.

    Returns
    -------
        couplings_ij : np.array
            A2d numpy array of shape (q, q) containing the couplings. Note that
            couplings_ij[q, :] and couplings[:, q] are set to zero.
    """
    qi = mx[site_pair[0]]
    qj = mx[site_pair[1]]
    couplings_ij = np.zeros((qi, qj), dtype=np.float64)
    row_begin = mx.cumsum()[site_pair[0] - 1]
    row_end = row_begin + qi
    column_begin = mx.cumsum()[site_pair[1] - 1]
    column_end = column_begin + qj
    couplings_ij[:qi - 1, :qj - 1] = couplings[row_begin:row_end, column_begin:column_end]
    return couplings_ij

def zero_diag(mat, diag_l):
    rows, columns = mat.shape
    new_mat = np.zeros((rows,columns))
    for row in range(rows):
        for col in range(columns):
            if abs(row-col) > diag_l:
                new_mat[row, col] = mat[row ,col]
    return new_mat

