# Import system packages
import os.path, sys
import timeit
from pathlib import Path
from joblib import Parallel, delayed
import warnings

# import scientific computing packages
import numpy as np
np.random.seed(1)
from scipy.spatial import distance
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# import biopython packages
from Bio.PDB import *
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

# import pydca modules
import pydca
from pydca.plmdca import plmdca
from pydca.meanfield_dca import meanfield_dca
from pydca import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.contact_visualizer import contact_visualizer
from pydca import dca_utilities
import os
import pandas as pd
import numpy as np

# # --- Import our Code ---# #
#import emachine as EM
from direct_info import direct_info

# import data processing and general DCA_ER tools
from data_processing import pdb2msa, data_processing
import tools 
from pathlib import Path

colors_hex = {"red": "#e41a1c", "blue": "#2258A5", "green": "#349C55", "purple": "#984ea3", "orange": "#FF8B00",
                      "yellow": "#ffff33", "grey": "#BBBBBB"}
colors_key = ["blue", "orange", "green"]


def plot_di_compare_methods(ax, ct_flat, di1_flat, di2_flat, ld_flat, labels):
    ec = 'b'
    f1 = True
    f2 = True
    f3 = True
    for j in range(len(ct_flat), -1, -1):
        i = j-1
        contact = ct_flat[i]
        if not ld_flat[i]:
            if f1:
                ax.scatter(di1_flat[i], di2_flat[i], marker='x',  c='k', alpha=.2, label='to close')
                f1 = False
            else:
                ax.scatter(di1_flat[i], di2_flat[i],  marker='x',  c='k', alpha=.2)
    
        elif contact==1.:
            if f2:
                ax.scatter(di1_flat[i], di2_flat[i], marker='o', facecolors='none', edgecolors='g', label='contact')
                f2=False
            else:
                ax.scatter(di1_flat[i], di2_flat[i],  marker='o', facecolors='none', edgecolors='g')
        else:
            if f3:
                ax.scatter(di1_flat[i], di2_flat[i], marker='_', c='r', label='no contact')
                f3 = False
            else:
                ax.scatter(di1_flat[i], di2_flat[i], marker='_', c='r')
    ax.set_xlabel('%s DI' % labels[0], fontsize=14)
    ax.set_ylabel('%s DI' % labels[1], fontsize=14)
    return ax


def pydca_tp_plot(method_visualizers, methods, pdb_id, ld=4, contact_dist=5.):

    fig = plt.figure(figsize=(5,5))
    ax1 = plt.subplot2grid((1,1), (0,0))

   
    # Plot ER results
    if len(method_visualizers) > 1:
        for i, mv in enumerate(method_visualizers):
            true_positive_rates_dict = mv.compute_true_positive_rates()
            tpr = true_positive_rates_dict['dca']
            pdb_tpr = true_positive_rates_dict['pdb']
            max_rank = len(tpr)
            ranks = [i + 1 for i in range(max_rank)]
            
                        
            ax1.plot(ranks, tpr, label=methods[i], color=colors_hex[colors_key[i]])
            if i == 0:
                ax1.plot(ranks, pdb_tpr,color='k')
            ax1.set_xscale('log')
            ax_title = '''
            True Positive Rate Per Rank
            PDB cut-off distance : {} Angstrom
            Residue chain distance : {}
            '''
            #ax.set_title(ax_title.format(self.__contact_dist, self.__linear_dist,))
            ax1.set_xlabel('Rank (log scalled)', fontsize=14)
            ax1.set_ylabel('True Positives/Rank', fontsize=14)
            plt.legend()
            plt.grid()
            plt.tight_layout()
        plt.savefig('%s_%s_pydca_tp_rate.pdf' % (pdb_id, pfam_id) )
        
    else:
        true_positive_rates_dict = method_visualizers[0].compute_true_positive_rates()
        tpr = true_positive_rates_dict['dca']
        pdb_tpr = true_positive_rates_dict['pdb']
        max_rank = len(tpr)
        ranks = [i + 1 for i in range(max_rank)]
        
                    
        ax1.plot(ranks, tpr)
        ax1.plot(ranks, pdb_tpr, color='k')
        ax1.set_xscale('log')
        ax_title = '''
        True Positive Rate Per Rank
        PDB cut-off distance : {} Angstrom
        Residue chain distance : {}
        '''
        #ax.set_title(ax_title.format(self.__contact_dist, self.__linear_dist,))
        ax1.set_xlabel('Rank (log scalled)', fontsize=14)
        ax1.set_ylabel('True Positives/Rank', fontsize=14)
        plt.grid()
        plt.tight_layout()
        plt.savefig('%s_%s_%s_pydca_tp_rate.pdf' % (pdb_id, pfam_id, methods[0]) )
    
def pydca_contact_plot(method_visualizer, method, ld=4, contact_dist=5. ):
    contact_categories_dict = method_visualizer.contact_categories()
    true_positives = contact_categories_dict['tp']
    false_positives = contact_categories_dict['fp']
    missing_pairs = contact_categories_dict['missing']
    pdb_contacts =  contact_categories_dict['pdb']
    
    filtered_pdb_contacts_list = [ 
       site_pair for site_pair, metadata in pdb_contacts.items() if abs(site_pair[1] - site_pair[0]) > ld  
    ]
    num_filtered_pdb_contacts = len(filtered_pdb_contacts_list)
    
    fig = plt.figure(figsize=(5,5))
    ax2 = plt.subplot2grid((1,1), (0,0))
    if missing_pairs:
        x_missing, y_missing = method_visualizer.split_and_shift_contact_pairs(missing_pairs)
        ax.scatter(x_missing, y_missing, s=6, color='blue')
    x_true_positives, y_true_positives = method_visualizer.split_and_shift_contact_pairs(
        true_positives,
    )
    x_false_positives, y_false_positives = method_visualizer.split_and_shift_contact_pairs(
        false_positives
    )
    x_pdb, y_pdb = method_visualizer.split_and_shift_contact_pairs(pdb_contacts)
    ax_title = '''
    Maximum PDB contact distance : {} Angstrom
    Minimum residue chain distance: {} residues
    Fraction of true positives : {:.3g}
    '''.format(contact_dist, ld,
        len(true_positives)/(len(true_positives) + len(false_positives)),
    )
    
    ax2.scatter(y_true_positives, x_true_positives, s=6, color='green')
    ax2.scatter(y_false_positives, x_false_positives, s=6, color='red')
    ax2.scatter(x_pdb, y_pdb, s=6, color='grey')
    ax2.set_xlabel('Residue Position', fontsize=14)
    ax2.set_ylabel('Residue Position', fontsize=14)
    #ax2.set_title(ax_title)
    plt.tight_layout()
    plt.savefig('%s_%s_%s_pydca_contact_map.pdf' % (pdb_id, pfam_id, method) )
    
    
def plot_di_vs_ct(ax, ct_flat, dist_flat, dis_flat, ld_flat, labels):
    colors = 'brg'
    for ii, di_flat in enumerate(dis_flat):
        fmethod = False # plot label for method 
        for i, contact in enumerate(ct_flat):
            if fmethod:
                ax.scatter(dist_flat[i], di_flat[i], marker='.',  c=colors_hex[colors_key[ii]], label=labels[ii])
                fmethod=False
            else:
                ax.scatter(dist_flat[i], di_flat[i], marker='.',  c=colors_hex[colors_key[ii]])

    ax.set_xlabel('Distance ($\AA$)', fontsize=14)
    ax.set_ylabel('DI', fontsize=14)
    return ax



if __name__ == '__main__':
    create_new = True
    printing = True
    removing_cols = True
    
    
    data_path = Path('/data/cresswellclayec/DCA_ER/Pfam-A.full')
    data_path = Path('/data/cresswellclayec/Pfam-A.full')
    
    # Define data directories
    DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
    biowulf_dir = '%s/biowulf_full' % DCA_ER_dir
    
    out_dir = '%s/protein_data/di/' % biowulf_dir
    processed_data_dir = "%s/protein_data/data_processing_output" % biowulf_dir
    pdb_dir = '%s/protein_data/pdb_data/' % biowulf_dir
    
    
    pfam_dir = "/fdb/fastadb/pfam"
    
    # PDB ID: 1zdr, Pfam ID: PF00186
    pdb_id = '1zdr'
    pfam_id = 'PF00186'
    ref_outfile = Path(processed_data_dir, '%s_ref.fa' % pfam_id)
    prody_df = pd.read_csv('%s/%s_pdb_df.csv' % (pdb_dir, pdb_id))
    pdb2msa_row  = prody_df.iloc[0]
    print('\n\nGetting msa with following pdb2msa entry:\n', pdb2msa_row)
    #try:
    print(pdb2msa_row)
    pfam_id = pdb2msa_row['Pfam']
    pdb_id = pdb2msa_row['PDB ID']
    pdb_chain = pdb2msa_row['Chain']
    
    # --- Load scores --- #
    plm_out_file = "%s/%s_%s_PLM_di.npy" % (out_dir, pdb_id, pfam_id)
    mf_out_file = "%s/%s_%s_PMF_di.npy" % (out_dir, pdb_id, pfam_id)
    er_out_file = "%s/%s_%s_ER_di.npy" % (out_dir, pdb_id, pfam_id)
    plmdca_scores = np.load(plm_out_file)
    mfdca_scores = np.load(mf_out_file)
    # ER scores need to be translated
    ER_di = np.load(er_out_file)
    s_index = np.load("%s/%s_%s_preproc_sindex.npy" % (processed_data_dir, pfam_id, pdb_id))
    from ecc_tools import scores_matrix2dict
    ER_scores = scores_matrix2dict(ER_di, s_index)
    # ------------------- #
    
    er_scores_ordered = {}
    for [pair, score] in ER_scores:
        er_scores_ordered[pair] = score
    ER_scores = er_scores_ordered
    ER_scores = sorted(ER_scores.items(), key =lambda k : k[1], reverse=True)
    for site_pair, score in ER_scores[:25]:
        print(site_pair, score)
    
    plm_scores_ordered = {}
    for [pair, score] in plmdca_scores:
        plm_scores_ordered[pair] = score
    plmdca_scores = plm_scores_ordered
    plmdca_scores = sorted(plmdca_scores.items(), key =lambda k : k[1], reverse=True)
    for site_pair, score in plmdca_scores[:5]:
        print(site_pair, score)
    
    mf_scores_ordered = {}
    for [pair, score] in mfdca_scores:
        mf_scores_ordered[pair] = score
    mfdca_scores = mf_scores_ordered
    mfdca_scores = sorted(mfdca_scores.items(), key =lambda k : k[1], reverse=True)
    for site_pair, score in mfdca_scores[:5]:
        print(site_pair, score)
       
    
    print('Ref Seq will not match pdb seq because of data_processing but thats ok.')
    ld = 4
    contact_dist = 5.
    plmdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
        refseq_file = str(ref_outfile),
        sorted_dca_scores = plmdca_scores,
        linear_dist = ld,
        contact_dist = contact_dist
    )
    
    mfdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
        refseq_file = str(ref_outfile),
        sorted_dca_scores = mfdca_scores,
        linear_dist = ld,
        contact_dist = contact_dist
    )
        
    er_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
            refseq_file = str(ref_outfile),
            sorted_dca_scores = ER_scores,
            linear_dist = ld,
            contact_dist = contact_dist,
    )
    
    
    
    pydca_contact_plot(plmdca_visualizer, 'PLM', ld=4, contact_dist=5.)
    pydca_contact_plot(er_visualizer, 'ER', ld=4, contact_dist=5.)
    pydca_contact_plot(mfdca_visualizer, 'MF', ld=4, contact_dist=5.)
    colors = ['b', 'r', 'g']
    pydca_tp_plot( [er_visualizer, mfdca_visualizer, plmdca_visualizer], methods = [ 'ER', 'MF','PLM'],ld=4, contact_dist=5. )
    pydca_tp_plot( [plmdca_visualizer ], methods = ['PLM'],ld=4, contact_dist=5. )
    pydca_tp_plot( [ mfdca_visualizer], methods = ['MF'],ld=4, contact_dist=5. )
    pydca_tp_plot( [er_visualizer], methods = [ 'ER'],ld=4, contact_dist=5. )
    
    # --- Plot DI vs Distance ER --- %
    # Define data directories
    DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
    biowulf_dir = '%s/biowulf_full' % DCA_ER_dir
    
    
    out_dir = '%s/protein_data/di/' % biowulf_dir
    out_metric_dir = '%s/protein_data/metrics/' % biowulf_dir
    
    processed_data_dir = "%s/protein_data/data_processing_output/" % biowulf_dir
    pdb_dir = '%s/protein_data/pdb_data/' % biowulf_dir
    
    ct_file = "%s%s_%s_ct.npy" % (pdb_dir, pdb_id, pfam_id)
    ct = np.load(ct_file)
    
    
    # load DI data
    ER_di = np.load("%s/%s_%s_ER_di.npy" % (out_dir, pdb_id, pfam_id))
    MF_di = np.load("%s/%s_%s_MF_di.npy" % (out_dir, pdb_id, pfam_id))
    PMF_di_data = np.load("%s/%s_%s_PMF_di.npy" % (out_dir, pdb_id, pfam_id),allow_pickle=True)
    PLM_di_data = np.load("%s/%s_%s_PLM_di.npy" % (out_dir, pdb_id, pfam_id),allow_pickle=True)
    
    
    file_end = ".npy"
    fp_file = "%s%s_%s_ER_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_file = "%s%s_%s_ER_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    er_fp = np.load(fp_file)
    er_tp = np.load(tp_file)
    file_end = ".npy"
    fp_file = "%s%s_%s_PMF_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_file = "%s%s_%s_PMF_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    pmf_fp = np.load(fp_file)
    pmf_tp = np.load(tp_file)
    file_end = ".npy"
    fp_file = "%s%s_%s_PLM_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_file = "%s%s_%s_PLM_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    plm_fp = np.load(fp_file)
    plm_tp = np.load(tp_file)
    
    
    
    
    fp_uni_file = "%s%s_%s_ER_fp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_uni_file = "%s%s_%s_ER_tp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    er_fp_uni = np.load(fp_uni_file)
    er_tp_uni = np.load(tp_uni_file)
    
    
    ct1 = ct.copy()
    ct_pos = ct < 6
    ct1[ct_pos] = 1
    ct1[~ct_pos] = 0
    mask = np.triu(np.ones(ER_di.shape[0], dtype=bool), k=1)
    # argsort sorts from low to high. [::-1] reverses 
    order = ER_di[mask].argsort()[::-1]
    
    ld_thresh = 0.
    linear_distance = np.zeros((len(s_index),len(s_index)))                                                                                                   
    for i, ii in enumerate(s_index):                                                                                                                          
        for j, jj in enumerate(s_index):                                                                                                                      
            linear_distance[i,j] = abs(ii - jj)   
    
    
    ld = linear_distance >= ld_thresh                                                                                                                         
    ld_flat = ld[mask][order]          
    
    ER_di_flat = ER_di[mask][order]
    ct_flat = ct1[mask][order]
    dist_flat = ct[mask][order]
    
    
    labels = ['ER']
    flat_dis =  [ER_di_flat]
    
    plt.figure(figsize=(5,5))
    ax = plt.subplot2grid((1,1),(0,0))
    ax = plot_di_vs_ct(ax, ct_flat, dist_flat, flat_dis, ld_flat, labels)
    plt.tight_layout()
    # ax.legend()
    plt.savefig('%s_%s_%s_di_dist.pdf' % (pdb_id, pfam_id, 'ER') )
    
    
    # --- Compare Methods --- #
    PMF_di_data = np.load("%s/%s_%s_PMF_di.npy" % (out_dir, pdb_id, pfam_id),allow_pickle=True)
    PLM_di_data = np.load("%s/%s_%s_PLM_di.npy" % (out_dir, pdb_id, pfam_id),allow_pickle=True)
    
    
    # transform pydca DI dictionary to DI matrices.
    PLM_di = np.zeros(ER_di.shape)
    PLM_di_dict = {}
    for score_set in PLM_di_data:
        PLM_di_dict[(score_set[0][0], score_set[0][1])] = score_set[1]
    for i, index_i in enumerate(s_index):
        for j, index_j in enumerate(s_index):
            if i==j:
                PLM_di[i,j] = 1.
                continue
            try:
                PLM_di[i,j] = PLM_di_dict[(index_i, index_j)]
                PLM_di[j,i] = PLM_di_dict[(index_i, index_j)] # symetric
            except(KeyError):
                continue
    
    PMF_di = np.zeros(ER_di.shape)
    PMF_di_dict = {}
    for score_set in PMF_di_data:
        PMF_di_dict[(score_set[0][0], score_set[0][1])] = score_set[1]
    for i, index_i in enumerate(s_index):
        for j, index_j in enumerate(s_index):
            if i==j:
                PMF_di[i,j] = 1.
                continue
            try:
                PMF_di[i,j] = PMF_di_dict[(index_i, index_j)]
                PMF_di[j,i] = PMF_di_dict[(index_i, index_j)] # symetric
            except(KeyError):
                continue
    
    
    fp_file = "%s%s_%s_PLM_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_file = "%s%s_%s_PLM_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    plm_fp = np.load(fp_file)
    plm_tp = np.load(tp_file)
    
    fp_uni_file = "%s%s_%s_PLM_fp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_uni_file = "%s%s_%s_PLM_tp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    plm_fp_uni = np.load(fp_uni_file)
    plm_tp_uni = np.load(tp_uni_file)
    
    ct_pos_file = "%s%s_%s_PLM_ct_flat%s" % (processed_data_dir, pdb_id, pfam_id, file_end)
    plm_ct_flat = np.load(ct_pos_file)
    
    PLM_di_flat = PLM_di[mask][order] # get array of plm di in the order of ER di to plot together
    
    fp_file = "%s%s_%s_PMF_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_file = "%s%s_%s_PMF_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    pmf_fp = np.load(fp_file)
    pmf_tp = np.load(tp_file)
    
    fp_uni_file = "%s%s_%s_PMF_fp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_uni_file = "%s%s_%s_PMF_tp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    pmf_fp_uni = np.load(fp_uni_file)
    pmf_tp_uni = np.load(tp_uni_file)
    
    ct_pos_file = "%s%s_%s_PMF_ct_flat%s" % (processed_data_dir, pdb_id, pfam_id, file_end)
    pmf_ct_flat = np.load(ct_pos_file)
    
    PMF_di_flat = PMF_di[mask][order] # get array of pmf di in the order of ER di to plot together
    
    
    
    
    flat_dis =  [ER_di_flat, PLM_di_flat]
    labels = ['ER', 'PLM']
    plt.figure(figsize=(5,5))
    ax = plt.subplot2grid((1,1),(0,0))
    ax = plot_di_compare_methods(ax, ct_flat, flat_dis[0], flat_dis[1], ld_flat, labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('%s_%s_di_contact_%sv%s.pdf' % (pdb_id, pfam_id, labels[0],labels[1]) )
    
    
    flat_dis =  [ER_di_flat, PMF_di_flat]
    labels = ['ER', 'MF']
    plt.figure(figsize=(5,5))
    ax2 = plt.subplot2grid((1,1),(0,0))
    ax2 = plot_di_compare_methods(ax2, ct_flat, flat_dis[0], flat_dis[1], ld_flat, labels)
    ax2.legend()
    plt.tight_layout()
    plt.savefig('%s_%s_di_contact_%sv%s.pdf' % (pdb_id, pfam_id, labels[0],labels[1]) )
    
    plt.figure(figsize=(5,5))
    ax = plt.subplot2grid((1,1),(0,0))
    ax.plot(er_fp, er_tp, label='ER', color=colors_hex[colors_key[0]])
    ax.plot(pmf_fp, pmf_tp, label='MF', color=colors_hex[colors_key[1]])
    ax.plot(plm_fp, plm_tp, label='PLM', color=colors_hex[colors_key[2]])
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig('%s_%s_roc_comparison.pdf' % (pdb_id, pfam_id) )
    
    
