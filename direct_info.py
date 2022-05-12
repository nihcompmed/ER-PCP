import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import OneHotEncoder

#=========================================================================================#
def frequency(s0,q,i1i2,theta=0.2,pseudo_weight=0.5):
    n = s0.shape[1]
    # hamming distance
    dst = distance.squareform(distance.pdist(s0, 'hamming'))
    ma_inv = 1/(1+(dst < theta).sum(axis=1).astype(float))
    meff = ma_inv.sum() 

    onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
    s = onehot_encoder.fit_transform(s0)

    # fi_true:
    fi_true = (ma_inv[:,np.newaxis]*s[:,:]).sum(axis=0)
    fi_true /= meff

    # fi, fij
    fi = np.zeros(q.sum())
    for i in range(n):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        fi[i1:i2] = (1 - pseudo_weight)*fi_true[i1:i2] + pseudo_weight/q[i]

    return fi
#=========================================================================================#


#=========================================================================================#
#NOTE: direct information from w, ONLY apply for our method, NOT DCA since w is converted to 2D
def direct_info_value(w2d,fi,q,i1i2):
# w2d[nm,nm], fi[l,n],q[n]
    n = q.shape[0]


    #ew_all = np.exp(w2d)

    # dealing with RuntimeWarning
    try:
        ew_all = np.exp(w2d)
    except(RuntimeWarning):
        max_w2d = max([max(w) for w in w2d])
        print('subtracting max w2d value: ',max_w2d)
        w2d = w2d - max_w2d
        ew_all = np.exp(w2d)
   
 
    di = np.zeros((n,n))
    tiny = 10**(-100.)
    diff_thres = 10**(-4.)

    for i in range(n-1):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        for j in range(i+1,n): 
            j1,j2 = i1i2[j,0],i1i2[j,1]
            #ew = ew_all[i,j,:q[i],:q[j]]
            ew = ew_all[i1:i2,j1:j2]
            #------------------------------------------------------
            # find h1 and h2:

            # initial value
            diff = diff_thres + 1.
            eh1 = np.full(q[i],1./q[i])
            eh2 = np.full(q[j],1./q[j])

            #fi0 = fi[i,0:q[i]]
            #fj0 = fi[j,0:q[j]]
            fi0 = fi[i1:i2]
            fj0 = fi[j1:j2]
            
            for iloop in range(100):
                eh_ew1 = eh2.dot(ew.T)
                eh_ew2 = eh1.dot(ew)

                eh1_new = fi0/(eh_ew1+tiny) # ecc added tiny
                eh1_new /= eh1_new.sum()

                eh2_new = fj0/(eh_ew2+tiny) # ecc added tiny
                eh2_new /= eh2_new.sum()

                diff = max(np.max(np.abs(eh1_new - eh1)),np.max(np.abs(eh2_new - eh2)))

                eh1,eh2 = eh1_new,eh2_new    
                if diff < diff_thres: break        

            # direct information
            eh1eh2 = eh1[:,np.newaxis]*eh2[np.newaxis,:]
            pdir = ew*(eh1eh2)
            pdir /= (pdir.sum()+tiny) # ecc added tiny

            fifj = fi0[:,np.newaxis]*fj0[np.newaxis,:]

            dijab = pdir*np.log((pdir+tiny)/(fifj+tiny))
            di[i,j] = dijab.sum()

    # symmetrize di
    di = di + di.T
    return di
#=========================================================================================#

#=========================================================================================#
def direct_info(s0,w):
    w = (w+w.T)/2

    l,n = s0.shape
    mx = np.array([len(np.unique(s0[:,i])) for i in range(n)])
    #mx = np.array([m for i in range(n)])
    mx_cumsum = np.insert(mx.cumsum(),0,0)
    i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T

    q = mx   # just for convenience
    fi = frequency(s0,q,i1i2)
    di = direct_info_value(w,fi,q,i1i2)
    
    return di

def sort_di(di):
	"""
	Returns array of sorted DI values
	"""
	ind = np.unravel_index(np.argsort(di,axis=None),di.shape)	
	tuple_list = [((indices[0],indices[1]),di[indices[0],indices[1]]) for i,indices in enumerate(np.transpose(ind))]	
	tuple_list = tuple_list[::-1]
	return tuple_list

def sindex_di(sorted_di,s_index):
	s_index_di = []
	for di_tuple in sorted_di:
		#print(di_tuple ,"-->",((s_index[di_tuple[0][0]], s_index[di_tuple[0][1]]),di_tuple[1]))
		s_index_di.append(((s_index[di_tuple[0][0]], s_index[di_tuple[0][1]]),di_tuple[1])) 	
	return s_index_di
#=========================================================================================#
