import os, sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

from Bio import SeqIO
import Bio.PDB, warnings
from Bio.PDB import *

from Bio import BiopythonWarning
from Bio.pairwise2 import format_alignment, align
pdb_parser = Bio.PDB.PDBParser()

from urllib.error import HTTPError

from pypdb import Query
from prody import *
from prody import searchPfam, fetchPfamMSA
#from ProDy import *
#from ProDy.prody import searchPfam, fetchPfamMSA





# ============================================================================================================================================================================ #
# -------------------------------------------- PDB-MSA Link finding Functions ------------------------------------------------------------------------------------------------ #
# ============================================================================================================================================================================ #



def pdb2msa(pdb_file, pdb_dir, create_new=True):
    pdb_id = os.path.basename(pdb_file)[3:7]

    # if you just want to load it
    if os.path.exists('%s/%s_pdb_df.csv' % (pdb_dir, pdb_id)) and not create_new:
        prody_df = pd.read_csv('%s/%s_pdb_df.csv' % (pdb_dir, pdb_id))
        #print('Error during fetchPfamMSA: ',e)
        return [prody_df]

    print(pdb_id)
    chain_matches = {}
    #for record in SeqIO.parse(pdb_file, "pdb-seqres"):
    #    print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))
    #    print(record.dbxrefs)

    pdb_model = pdb_parser.get_structure(str(pdb_id), pdb_file)[0]
    prody_columns =  ['PDB ID' , 'Chain', 'Polypeptide Index', 'Pfam', 'accession', 'class', 'id', 'type', 'PDB Sequence']
    float_values = ['bitscore', 'ind_evalue', 'cond_evalue']
    int_values = ['ali_end', 'ali_start', 'end', 'hmm_end', 'hmm_start', 'start']
 
    prody_df = None
    prody_dict = {}
    for chain in pdb_model.get_chains():
        ppb = PPBuilder().build_peptides(chain)
        if len(ppb) == 0: 
            print('\nChain %s has no polypeptide sequence\n' % chain.get_id())
            continue
        for i, pp in enumerate(ppb):
            poly_seq = list()
            for char in str(pp.get_sequence()):
                poly_seq.append(char)
            print('\nChain %s polypeptide %d (length %d): ' % (chain.get_id(), i, len(''.join(poly_seq))),''.join(poly_seq))

            #try:
            prody_search = searchPfam(''.join(poly_seq), timeout=300)
            #print(prody_search)
            #except Exception as e:
            #    print('Error with prody.searchPfam: ', e, '\n')
            #    continue

             
            for pfam_key in prody_search.keys():
                ps = prody_search[pfam_key]
           
                # prody_alignment_values = list(ps['locations'].values())
                prody_alignment_values = []
                for l_key in ps['locations'].keys():
                    if l_key in float_values:
                        prody_alignment_values.append(float(ps['locations'][l_key]))
                    elif l_key in int_values:
                        prody_alignment_values.append(int(ps['locations'][l_key]))
                    else:
                        prody_alignment_values.append(ps['locations'][l_key])

                prody_lst = [pdb_id, chain.get_id(), i, pfam_key, ps['accession'], ps['class'], ps['id'], ps['type'], ''.join(poly_seq)] + prody_alignment_values

                if prody_df is None:
                    prody_alignment_columns = [key for key in ps['locations'].keys()]
                    prody_df = pd.DataFrame([prody_lst], columns = prody_columns + prody_alignment_columns)
                else:
                    prody_df.loc[len(prody_df.index)] = prody_lst
                #else:
                #    print('Pfam %s already found and the match is not an improvement' % pfam_key)

    # reorder df to get best matches on top.
    if prody_df is None:
        print('PDB2MSA ERROR: No pdb matches found using prody.searchPfam for any of the chains/polypeptide sequences. !!')
        sys.exit(23)
    #print(prody_df.head())
    print('sorting ProdyDataframe')
    
    prody_df = prody_df.sort_values(by='bitscore', ascending = False).sort_values(by='ind_evalue') # could be cond_evalue??
    prody_df = prody_df.reset_index(drop=True)
    #print(prody_df.head())

    prody_df.to_csv('%s/%s_pdb_df.csv' % (pdb_dir, pdb_id))

    #print('Error during fetchPfamMSA: ',e)
    return [prody_df]



# ============================================================================================================================================================================ #
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# ============================================================================================================================================================================ #





# ============================================================================================================================================================================ #
# -------------------------------------- MSA Data-Processing Functions ------------------------------------------------------------------------------------------------------- #
# ============================================================================================================================================================================ #



def get_tpdb(s, ali_start_indx, ali_end_indx, pfam_start_indx, pfam_end_indx, aligned_pdb_str):
    # Requires DataFrame from data_processing.pdb2msa() function as input
    from tools import hamming_distance
    alignment_len =  ali_end_indx - ali_start_indx + 1
    print('Lets find the reference sequence in this MSA\n...We want index of the MSA sequence which best matches our PDB-polypeptide sequence...?')
    # print('alignment length: ', alignment_len)
    print('PDB-polypeptids sequence (len %d):\n%s\n\n' % (len(aligned_pdb_str), aligned_pdb_str))
    #print(alignment_len, len(aligned_pdb_str), aligned_pdb_str)

    min_ham = alignment_len
    max_pair_score = 0
    min_indx = -1
    best_alignment = None
    print("Looping through MSA sequences to find best matching sequence for our MSA reference sequence!")
    for i, seq in enumerate(s):
        gap_seq = seq == '-'  # returns True/False for gaps/no gaps
        subject = seq[~gap_seq]
        seq_str = ''.join(subject).upper()
        aligned_seq_str = seq_str[pfam_start_indx : pfam_end_indx+1]
        #if len(aligned_seq_str) != alignment_len:
        #   print('length mismatch %d, %d' % (len(aligned_seq_str), alignment_len))
        #   continue
        #ham_dist = hamming_distance(aligned_pdb_str, aligned_seq_str)
        alignments = align.globalxx(aligned_pdb_str, aligned_seq_str)
        
        if len(alignments) == 0:
            continue
        try:
            pair_score = alignments[0].score
        except(AttributeError):
            pair_score = alignments[0][2]

        # if ham_dist < min_ham:
        if pair_score > max_pair_score:
            #min_ham = ham_dist
            max_pair_score = pair_score
            min_indx = i
            best_alignment = alignments[0]
            print('    ... found a match upgrade at index ' , i)
            #print('%d: hamm dist=%d, pairwise score=%f\n' % (i, ham_dist, pair_score))
            #print('Sequence Alignment: ', best_alignment, '\n\n')
            print(format_alignment(*alignments[0]))
 
    if best_alignment is None:
        print('ERROR could not find alignment with score better than 0..') 

    gap_seq = s[min_indx] == '-'
    print('Best match is sequence %d\n    Hamming distance of %d (length %d)\n\n' % (min_indx, min_ham, len(s[min_indx][~gap_seq])))
           
    # get matching seq for both sequences (no gaps in either)
    try:
        aligned_pdb_char_array = np.array([char for char in best_alignment.seqA])
        aligned_ref_char_array = np.array([char for char in best_alignment.seqB])
    except(AttributeError):
        aligned_pdb_char_array = np.array([char for char in best_alignment[0]])
        aligned_ref_char_array = np.array([char for char in best_alignment[1]])
       
    
    # get array of gaps for both sequences
    seqA_gaps = aligned_pdb_char_array == '-'
    seqB_gaps = aligned_ref_char_array == '-'
    aligned_gaps = np.logical_or(seqA_gaps, seqB_gaps)

    
    # create index array for reference sequence so we know which msa columns associated with aligned arrays
    pdb_count = 0
    ref_count = 0
    gap_ref_index = -1 * np.ones(len(aligned_ref_char_array), dtype=int)
    gap_pdb_index = -1 * np.ones(len(aligned_pdb_char_array), dtype=int)
    for i, char in enumerate(aligned_ref_char_array):
        if char !='-':
            gap_ref_index[i] = int(ref_count)
            ref_count += 1
        if aligned_pdb_char_array[i] !='-':
            gap_pdb_index[i] = int(pdb_count)
            pdb_count += 1            
    
    # get columns to remove (gap in PDB) in MSA
    pdb_gap_cols_in_ref = gap_ref_index[seqA_gaps]
    print(len(pdb_gap_cols_in_ref), pdb_gap_cols_in_ref)

    # get s_index for mapping msa to pdb sequence.
    pdb_s_index = gap_pdb_index[~aligned_gaps]
    # print('PDB index map: ', len(pdb_s_index), pdb_s_index)
    
    # Extract further infor for aligned seqs.
    aligned_pdb_nogap = aligned_pdb_char_array[~aligned_gaps]
    aligned_ref_nogap = aligned_ref_char_array[~aligned_gaps]
    print('\nAligned PDB and ref seq:')
    print('PDB polypeptide sequence (length %d): ' % len(aligned_pdb_nogap), ''.join(aligned_pdb_nogap))
    print('MSA-matched sequence     (length %d): ' % len(aligned_ref_nogap), ''.join(aligned_ref_nogap))
    
    # Trim By gaps in Ref seq (tbr). Then Trim By gaps in Pdb seq (tpb)
    s_tbr = s[:, ~gap_seq]
    s_tbr = s_tbr[:,pfam_start_indx : pfam_end_indx+1]
    print('\n\nTrimming MSA by reference sequence (s -> s_tpr)...')
    print('dimensions of s_tbr:', s_tbr.shape)
    print('s_tbr[tpdb=%d] = ' % min_indx, ''.join(s_tbr[min_indx]))
    s_tbr_tbp = np.delete(s_tbr, pdb_gap_cols_in_ref, axis=1)

    # printed ref seq should be the same as the fully alinged, gapless pdb and ref seqs above.
    return min_indx, best_alignment, s_tbr_tbp, pdb_s_index


def remove_bad_seqs(s, tpdb, fgs=0.3, trimmed_by_refseq=True):
    # if trimmed by reference sequence, create a temp matrix to find bad sequences
    if not trimmed_by_refseq:
        s_temp = s.copy()
        gap_pdb = s[tpdb] == '-'  # returns True/False for gaps/no gaps in reference sequence
        s_temp = s_temp[:, ~gap_pdb]  # removes gaps in reference sequence
    else:
        s_temp = s

    # remove bad sequences having a gap fraction of fgs  
    l, n = s_temp.shape

    frequency = [(s_temp[t, :] == '-').sum() / float(n) for t in range(l)]
    bad_seq = [t for t in range(l) if frequency[t] > fgs]
    print(len(bad_seq))
    new_s = np.delete(s, bad_seq, axis=0)
    # Find new sequence index of Reference sequence tpdb
    seq_index = np.arange(s.shape[0])
    seq_index = np.delete(seq_index, bad_seq)
    new_tpdb = np.where(seq_index == tpdb)
    print("After removing bad sequences, tpdb is now ", new_tpdb[0][0])

    return new_s, new_tpdb[0][0]



def remove_seqs_list(s, tpdb, seqs_to_remove):
    # remove sequence rows in list seqs_to_remove
    #     -- update tpdb
    l, n = s.shape

    new_s = np.delete(s, seqs_to_remove, axis=0)
    # Find new sequence index of Reference sequence tpdb
    seq_index = np.arange(s.shape[0])
    seq_index = np.delete(seq_index, seqs_to_remove)
    new_tpdb = np.where(seq_index == tpdb)
    print("After removing bad sequences, tpdb is now ", new_tpdb[0][0])

    return new_s, new_tpdb[0][0]



def remove_bad_cols(s, fg=0.3, fc=0.9):
    # remove positions having a fraction fc of converved residues or a fraction fg of gaps 

    l, n = s.shape
    # gap positions:
    frequency = [(s[:, i] == '-').sum() / float(l) for i in range(n)]
    cols_gap = [i for i in range(n) if frequency[i] > fg]

    # conserved positions:
    frequency = [max(np.unique(s[:, i], return_counts=True)[1]) for i in range(n)]
    cols_conserved = [i for i in range(n) if frequency[i] / float(l) > fc]

    cols_remove = cols_gap + cols_conserved

    return np.delete(s, cols_remove, axis=1), cols_remove



def find_bad_cols(s, fg=0.2):
    # remove positions having a fraction fg of gaps
    l, n = s.shape
    # gap positions:
    frequency = [(s[:, i] == '-').sum() / float(l) for i in range(n)]
    bad_cols = [i for i in range(n) if frequency[i] > fg]

    # return np.delete(s,gap_cols,axis=1),np.array(gap_cols)
    return np.array(bad_cols)



def find_conserved_cols(s, fc=0.8):
    # remove positions having a fraction fc of converved residues
    l, n = s.shape

    # conserved positions:
    frequency = [max(np.unique(s[:, i], return_counts=True)[1]) for i in range(n)]
    conserved_cols = [i for i in range(n) if frequency[i] / float(l) > fc]

    # return np.delete(s,conserved_cols,axis=1),np.array(conserved_cols)
    return np.array(conserved_cols)



def convert_letter2number(s):
    letter2number = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, \
                     'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '-': 20,
                     'U': 21}
    # ,'B':20, 'Z':21, 'X':22}
    try:
        l, n = s.shape
    except(ValueError):
        n = s.shape[0] # if s is only one row.
        return np.array([letter2number[s[i].upper()]  for i in range(n)])
    # making sure all amino acids are uppercase # # this is done in PYDCA as well. though there are references that say lowercase means no-good
    return np.array([letter2number[s[t, i].upper()] for t in range(l) for i in range(n)]).reshape(l, n)



def convert_number2letter(s):
    number2letter = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L', \
                     10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: '-',
                     21: 'U'}
    try:
        l, n = s.shape
        return np.array([number2letter[s[t, i]] for t in range(l) for i in range(n)]).reshape(l, n)
    except(ValueError):
        print(s)
        return np.array([number2letter[r] for r in s])



def value_with_prob(a, p1):
    """ generate a value (in a) with probability
    input: a = np.array(['A','B','C','D']) and p = np.array([0.4,0.5,0.05,0.05]) 
    output: B or A (likely), C or D (unlikely)
    """
    p = p1.copy()
    # if no-specific prob --> set as uniform distribution
    if p.sum() == 0:
        p[:] = 1. / a.shape[0]  # uniform
    else:
        p[:] /= p.sum()  # normalize

    ia = int((p.cumsum() < np.random.rand()).sum())  # cordinate

    return a[ia]



def find_and_replace(s, z, a):
    """ find positions of s having z and replace by a with a probality of elements in s column
    input: s = np.array([['A','Q','A'],['A','E','C'],['Z','Q','A'],['A','Z','-']])
           z = 'Z' , a = np.array(['Q','E'])    
    output: s = np.array([['A','Q','A'],['A','E','C'],['E','Q','A'],['A','Q','-']]           
    """
    xy = np.argwhere(s == z)

    for it in range(xy.shape[0]):
        t, i = xy[it, 0], xy[it, 1]

        na = a.shape[0]
        p = np.zeros(na)
        for ii in range(na):
            p[ii] = (s[:, i] == a[ii]).sum()

        s[t, i] = value_with_prob(a, p)
    return s




def replace_lower_by_higher_prob(s, p0=0.3):
    # input: s: 1D numpy array ; threshold p0
    # output: s in which element having p < p0 were placed by elements with p > p0, according to prob

    # f = itemfreq(s)  replaced by next line due to warning
    f = np.unique(s, return_counts=True)
    # element and number of occurence
    a, p = f[0], f[1].astype(float)

    # probabilities    
    p /= float(p.sum())

    # find elements having p > p0:
    iapmax = np.argwhere(p > p0).reshape((-1,))  # position

    apmax = a[iapmax].reshape((-1,))  # name of aminoacid
    pmax = p[iapmax].reshape((-1,))  # probability

    # find elements having p < p0
    apmin = a[np.argwhere(p < p0)].reshape((-1,))

    if apmin.shape[0] > 0:
        for a in apmin:
            ia = np.argwhere(s == a).reshape((-1,))
            for iia in ia:
                s[iia] = value_with_prob(apmax, pmax)
    return s



def load_msa(data_path, pfam_id):
    s = np.load('%s/%s/msa.npy' % (data_path, pfam_id)).T
    # print("shape of s (import from msa.npy):\n",s.shape)

    # convert bytes to str
    try:
        s = np.array([s[t, i].decode('UTF-8') for t in range(s.shape[0]) \
                      for i in range(s.shape[1])]).reshape(s.shape[0], s.shape[1])
    # print("shape of s (after UTF-8 decode):\n",s.shape)
    except:
        print("\n\nUTF not decoded, pfam_id: %s \n\n" % pfam_id, s.shape)
        print("Exception: ", sys.exc_info()[0])
        # Create list file for missing pdb structures
        if not os.path.exists('missing_MSA.txt'):
            file_missing_msa = open("missing_MSA.txt", 'w')
            file_missing_msa.write("%s\n" % pfam_id)
            file_missing_msa.close()
        else:
            file_missing_msa = open("missing_MSA.txt", 'a')
            file_missing_msa.write("%s\n" % pfam_id)
            file_missing_msa.close()
        return
    return s



def data_processing(data_path, pdb_df,gap_seqs=0.2, gap_cols=0.2, prob_low=0.004, 
                        conserved_cols=0.8, printing=True, out_dir='./', pdb_dir='./', letter_format=False, 
                        remove_cols=True, create_new=True, n_cpu=2):
    pfam_id = pdb_df['Pfam']
    pdb_seq = pdb_df['PDB Sequence']
    pdb_id = pdb_df['PDB ID']
    ali_start_indx = int(pdb_df['ali_start'])-1
    ali_end_indx = int(pdb_df['ali_end'])-1
    pfam_start_indx = int(pdb_df['hmm_start'])-1
    pfam_end_indx = int(pdb_df['hmm_end'])-1

    aligned_pdb_str  = pdb_df['PDB Sequence'][ali_start_indx:ali_end_indx+1]


    print('PDB ID: %s, Pfam ID: %s' % (pdb_id, pfam_id))
    if remove_cols:
        processing_type = "preproc"
    else:
        processing_type = "allCols"
    # if not create_new and os.path.exists("%s/%s_%s_%s_msa.npy" % (out_dir, pfam_id, pdb_id, processing_type)):
    if 0:
        print('Because create_new is False and files exist we will load preprocessed data:')
        s = np.load("%s/%s_%s_%s_msa.npy" % (out_dir, pfam_id, pdb_id, processing_type))
        s_index = np.load("%s/%s_%s_%s_sindex.npy" % (out_dir, pfam_id, pdb_id, processing_type))
        removed_cols = np.load("%s/%s_%s_removed_cols.npy" % (out_dir, pfam_id, pdb_id))
        ref_seq = np.load("%s/%s_%s_%s_refseq.npy" % (out_dir, pfam_id, pdb_id, processing_type))

        if not letter_format and isinstance(s[0][0], str):
            s = convert_letter2number(s)
      
    
    # Load MSA
    s = load_msa(data_path, pfam_id)
    orig_seq_len = s.shape[1]
    print('Original Sequence length: ', orig_seq_len)

    
    # Using given MSA find best matching PDB structure from all available MSA sequences.
    
    if printing:
        print("\n\n#--------------------- Find PDB Sequence in MSA ---------------#")
       

       
    # Find PDB seq in MSA current
    tpdb, alignment, s, pdb_s_index = get_tpdb(s, ali_start_indx, ali_end_indx, pfam_start_indx, pfam_end_indx, aligned_pdb_str) # requires prody.searchPfam DF from pdb2msa as input

    if printing:
        print('MSA index %d matches PDB' % tpdb)
        print("\n\n#--------------------------------------------------------------#")
        print("#---------- Begin Pre-Processing MSA --------------------------#")
        print("#--------------------------------------------------------------#\n\n")

    if printing:
        print("\n\n#-------------------------Remove Gaps--------------------------#")
        print('Shape of s is : ', s.shape)
    
    # remove gaps to allow alignment with PDB sequence..
    # remove columns not in alignment range
    in_range_indices = np.arange(ali_start_indx, ali_end_indx) 
    print('in range indices: ', in_range_indices)


    s_index = np.arange(s.shape[1])

    if printing:
        print("s[tpdb] shape is ", s[tpdb].shape)
        print("though s still has gaps, s[%d] does not:\n" % (tpdb), s[tpdb])
        print("s shape is ", s.shape)
        print("#--------------------------------------------------------------#\n\n")

    lower_cols = np.array([i for i in range(s.shape[1]) if s[tpdb, i].islower()])
    if printing:
        print("removing non aligned (lower case) columns in subject sequence:\n", lower_cols, '\n')
        

    # --- remove duplicates before processing (as done in pydca) --- #
    if printing:
        print("#--------------------- Removing Duplicate Sequences -----------#")
    dup_rows = []
    s_no_dup = []
    for i, row in enumerate(s):
        if [a for a in row] in s_no_dup:
            if i != tpdb:   # do not want to remove reference sequence
                dup_rows.append(i)
            else:           # we need to add the reference sequence back in even if its a duplicate row.    
                s_no_dup.append([a for a in row])
        else:
            s_no_dup.append([a for a in row])
    if printing:
        print('...found %d duplicates! (Removing...)' % len(dup_rows))
        print("#--------------------------------------------------------------#\n\n")
    s, tpdb = remove_seqs_list(s, tpdb, dup_rows)
    # -------------------------------------------------------------- #

    
    # - Removing bad sequences (>gap_seqs gaps) -------------------- #    
    s, tpdb = remove_bad_seqs(s, tpdb, gap_seqs)  # removes all sequences (rows) with >gap_seqs gap %
    
    if printing:
        print('\nAfter removing bad sequences...\ntpdb (s_ipdb) is : ', tpdb)
        print(s.shape)
    # -------------------------------------------------------------- #


    # - Finding bad columns (>gap_cols) -------------------- #
    bad_cols = find_bad_cols(s, gap_cols)
    if printing:
        print('Found bad columns :=', bad_cols)
    # ------------------------------------------------------ #


    # ------------ Replace Bad Amino Acid Letters if valid ones, two strategies --------- #
    # replace 'Z' by 'Q' or 'E' with prob
    s = find_and_replace(s, 'Z', np.array(['Q', 'E']))

    # replace 'B' by Asparagine (N) or Aspartic (D)
    s = find_and_replace(s, 'B', np.array(['N', 'D']))

    # replace 'X' with any amino acid with equal probability
    amino_acids = np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', \
                            'T', 'V', 'W', 'Y', 'U'])
    s = find_and_replace(s, 'X', amino_acids)
    # ----------------------------------------------------------------------------------- #

    
    # --------------------- Find Conserved Columns ------------------------------------ #
    conserved_cols = find_conserved_cols(s, conserved_cols)
    if printing:
        print("Found conserved columns (80% repetition):\n", conserved_cols)
    # --------------------------------------------------------------------------------- #

    
    # ----------------------- Remove all Bad Columns ---------------------------------- #

    # get list of column candidates for removal
    removed_cols = np.array(list(set(bad_cols) | set(conserved_cols)))
    removed_cols = np.array(list(set(removed_cols) | set(lower_cols)))
    if printing:
        print("We remove conserved and bad columns with, at the following indices (len %d):\n" % len(removed_cols), removed_cols)

    # info still pased through removed_cols but this way we can interact with full msa if remove_cols is False
    if remove_cols:
        s = np.delete(s, removed_cols, axis=1)
        s_index = np.delete(s_index, removed_cols)
        pdb_s_index = np.delete(pdb_s_index, removed_cols)

    if printing and remove_cols:
        print("Removed Columns...")
        print("Done Pre-Processing MSA")
        print("#--------------------------------------------------------------#\n\n\n")
        print("#--------------------------------------------------------------#")
        print("Final s (MSA) has shape: ", s.shape)
        print("s_index (length=%d) = \n" % s_index.shape[0], s_index)
        print("pdb_s_index (length=%d) = \n" % pdb_s_index.shape[0], pdb_s_index)
        print("Ref Seq (shape=", s[tpdb].shape, "): \n", s[tpdb])
        if not letter_format:
            print('Finishing by converting amino acid letters to equvalent number form')
        print("#--------------------------------------------------------------#")

    # Convert S to number format (number representation of amino acids)
    if not letter_format:
        # convert letter to number:
        s = convert_letter2number(s)

    # replace lower probs by higher probs 
    for i in range(s.shape[1]):
        s[:, i] = replace_lower_by_higher_prob(s[:, i], prob_low)


    np.save("%s/%s_%s_removed_cols.npy" % (out_dir, pfam_id, pdb_id), removed_cols)
    if remove_cols:
        np.save("%s/%s_%s_preproc_msa.npy" % (out_dir, pfam_id, pdb_id), s)
        np.save("%s/%s_%s_preproc_sindex.npy" % (out_dir, pfam_id, pdb_id), s_index)
        np.save("%s/%s_%s_preproc_pdb_sindex.npy" % (out_dir, pfam_id, pdb_id), pdb_s_index)
        np.save("%s/%s_%s_preproc_refseq.npy" % (out_dir, pfam_id, pdb_id), s[tpdb])
    else:
        np.save("%s/%s_%s_allCols_msa.npy" % (out_dir, pfam_id, pdb_id), s)
        np.save("%s/%s_%s_allCols_sindex.npy" % (out_dir, pfam_id, pdb_id), s_index)
        np.save("%s/%s_%s_allCols_refseq.npy" % (out_dir, pfam_id, pdb_id), s[tpdb])

    return [s, removed_cols, s_index, tpdb, pdb_s_index]
