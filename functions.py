import numpy as np
from ncon import ncon

#####******************************************************************#####

def read_mps(mps):
    ln = len(mps)
    mps_cut = []
    for idx in range(ln):
        tem = mps.module_list[idx].tensor.detach().numpy()
        mps_cut.append(tem)

    return mps_cut


#####*********************************************************************************************#####

def normalization(mps):
    ln = len(mps)
    mps_1 = mps.copy()
    #mps_1 = mps
    for idx in range(ln):
        #print(mps[idx].shape)
        if idx == 0:
            temp = np.tensordot(mps[idx], mps_1[idx].conj(), axes=([0],[0]))   #temp[r*r]
        if idx > 0 and idx < ln-1:
            temp1 = np.tensordot(temp, mps_1[idx].conj(), axes=([1],[0]))
            temp = np.tensordot(mps[idx], temp1, axes=([0,1], [0,1]))
        if idx == ln-1:
            temp1 = np.tensordot(temp, mps_1[idx].conj(), axes=([1],[0]))
            res = np.tensordot(mps[idx], temp1, axes=([0,1], [0,1]))

    return res


#####********************************************************************************************#####

def site_normalization(mps):
    ln = len(mps.shape)
    shape = list(i for i in range(ln))
    #mps_1 = mps.copy()
    mps_1 = mps
    #print("shape", shape)
    temp = np.tensordot(mps, mps_1.conj(), axes=(shape,shape))   #temp[r*r]
    return temp

#####********************************************************************************************#####

def calculate_Sz(mps):
    sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    #print("sz.shape", sz.shape)
    ln = len(mps)
    totalSz = 0.0
    for idx in range(ln):
        if idx == 0:
            mps = orth_centre(mps, site=idx+1)
            mps_1 = mps.copy()
            #mps_1 = mps
            tem = np.tensordot(np.tensordot(mps[idx], sz, axes=([0],[0])), mps_1[idx].conj(), axes=([1,0],[0,1]))   #temp[r*r]
            totalSz += tem
            #print("site_{}_sz".format(idx), tem)

        if idx > 0 and idx < ln-1:
            mps = orth_centre(mps, site=idx+1)
            mps_1 = mps.copy()
            #mps_1 = mps
            tem = np.tensordot(np.tensordot(mps[idx], sz, axes=([1],[0])), mps_1[idx].conj(), axes=([0,2,1],[0,1,2]))
            totalSz += tem
            #print("site_{}_sz".format(idx), tem)

        if idx == ln-1:
            mps = orth_centre(mps, site=idx+1)
            mps_1 = mps.copy()
            #mps_1 = mps
            tem = np.tensordot(np.tensordot(mps[idx], sz, axes=([1],[0])), mps_1[idx].conj(), axes=([0,1],[0,1]))
            totalSz += tem
            #print("site_{}_sz".format(idx), tem)

    return totalSz

#####********************************************************************************************************#####

def calculate_ENE(mps, ham):
    ln = len(mps)
    totalEne = 0.0
    for idx in range(ln-1):
        if idx == 0:
            mps = orth_centre(mps, site=idx+1)
            mps_1 = mps.copy()
            #mps_1 = mps
            tem = ncon([mps[idx], mps[idx+1], ham, mps_1[idx].conj(), mps_1[idx+1].conj()],[[1,2], [2,3,4],[1,3,5,7],[5,6],[6,7,4]])
            totalEne += tem
            #print("Energy_{}".format(idx), tem)

        if idx > 0 and idx < ln-2:
            mps = orth_centre(mps, site=idx+1)
            mps_1 = mps.copy()
            #mps_1 = mps
            tem = ncon([mps[idx], mps[idx+1], ham, mps_1[idx].conj(), mps_1[idx+1].conj()],[[1,2,3],[3,4,5],[2,4,6,8],[1,6,7],[7,8,5]])
            totalEne += tem
            #print("Energy_{}".format(idx), tem)

        if idx == ln-2:
            mps = orth_centre(mps, site=idx+1)
            mps_1 = mps.copy()
            #mps_1 = mps
            tem = ncon([mps[idx], mps[idx+1], ham, mps_1[idx].conj(), mps_1[idx+1].conj()],[[1,2,3], [3,4],[2,4,5,6],[1,5,7],[7,6]])
            totalEne += tem
            #print("Energy_{}".format(idx), tem)

    return totalEne

#####********************************************************************************************************#####

def set_grad_zero(file_name, shape):
    fin = open(file_name)
    lines = len(fin.readlines())
    fin.close()

    for idx in range(lines):
        data = np.genfromtxt(file_name, dtype = str, skip_header=idx,  max_rows=1)
        s = int(data[0])-1
        for l in range(int(data[1])):
            for r in range(int(data[2]),int(shape[2])):
                for i in range(int(shape[3])):
                    param.grad[s,l,r,i] = 0.0
                    #print("slri:", s,l,r,i)

        s = int(data[0])-1
        for l in range(int(data[1]), int(shape[1])):
            for r in range(int(shape[2])):
                for i in range(int(shape[3])):
                    param.grad[s,l,r,i] = 0.0
                    #print("slri:", s,l,r,i)


#####******************************************************************************************************#####


def orth_centre(mps, site):

    ln = len(mps)
    if site == 1:
        #print("11111111111")
        orth_mps = []
        u, s, vh =  np.linalg.svd(mps[ln-1], full_matrices=False)
        orth_mps.insert(0,vh)
        for idx in range(ln-2, 0, -1):
            shap = list(mps[idx].shape)
            mps[idx] = np.tensordot(np.tensordot(mps[idx], u, axes=([2],[0])), np.diag(s), axes=([2],[0]))
            u, s, vh =  np.linalg.svd(np.reshape(mps[idx], (shap[0], shap[1]*shap[2])), full_matrices=False)
            orth_mps.insert(0, np.reshape(vh, (shap[0], shap[1], shap[2])))

        mps[0] = np.tensordot(np.tensordot(mps[0], u, axes=([1],[0])), np.diag(s), axes=([1],[0]))
        orth_mps.insert(0,mps[0])
        #print(len(orth_mps))

        norm = np.sqrt(normalization(orth_mps))
        orth_mps[0] = orth_mps[0]*(1/norm)

    elif site == ln:
        #print("lnnnnnnnnnnn")
        orth_mps = []
        u, s, vh =  np.linalg.svd(mps[0], full_matrices=False)
        orth_mps.append(u)
        for idx in range(1, ln-1, 1):
            shap = list(mps[idx].shape)
            mps[idx] = np.tensordot(np.tensordot(np.diag(s), vh, axes=([1],[0])), mps[idx], axes=([1],[0]))
            u, s, vh =  np.linalg.svd(np.reshape(mps[idx], (shap[0]*shap[1], shap[2])), full_matrices=False)
            orth_mps.append(np.reshape(u, (shap[0], shap[1], shap[2])))

        mps[ln-1] = np.tensordot(np.tensordot(np.diag(s), vh, axes=([1],[0])), mps[ln-1], axes=([1],[0]))
        orth_mps.append(mps[ln-1])
        #print(len(orth_mps))

        norm = np.sqrt(normalization(orth_mps))
        orth_mps[0] = orth_mps[0]*(1/norm)

    elif site > 1 or site < ln:
        #print("111111lnnnnnnnnn")
        orth_mps_l = []
        orth_mps_r = []

        ul, sl, vhl =  np.linalg.svd(mps[0], full_matrices=False)
        orth_mps_l.append(ul)
        for idx in range(1, site-1, 1):
            shap = list(mps[idx].shape)
            mps[idx] = np.tensordot(np.tensordot(np.diag(sl), vhl, axes=([1],[0])), mps[idx], axes=([1],[0]))
            ul, sl, vhl =  np.linalg.svd(np.reshape(mps[idx], (shap[0]*shap[1], shap[2])), full_matrices=False)
            orth_mps_l.append(np.reshape(ul, (shap[0], shap[1], shap[2])))

        ur, sr, vhr =  np.linalg.svd(mps[ln-1], full_matrices=False)
        orth_mps_r.insert(0, vhr)
        for idx in range(ln-2, site-1, -1):
            shap = list(mps[idx].shape)
            mps[idx] = np.tensordot(np.tensordot(mps[idx], ur, axes=([2],[0])), np.diag(sr), axes=([2],[0]))
            ur, sr, vhr =  np.linalg.svd(np.reshape(mps[idx], (shap[0], shap[1]*shap[2])), full_matrices=False)
            orth_mps_r.insert(0, np.reshape(vhr, (shap[0], shap[1], shap[2])))


        shap = list(mps[site-1].shape)
        mps[site-1] = np.tensordot(np.tensordot(np.diag(sl), vhl, axes=([1],[0])), mps[site-1], axes=([1],[0]))
        mps[site-1] = np.tensordot(np.tensordot(mps[site-1], ur, axes=([2],[0])), np.diag(sr), axes=([2],[0]))

        orth_mps_l.append(mps[site-1])

        orth_mps = orth_mps_l + orth_mps_r
        #print(len(orth_mps))

        norm = np.sqrt(normalization(orth_mps))
        orth_mps[0] = orth_mps[0]*(1/norm)


    return orth_mps


#####*****************************************************************************************************#####

def heisenberg_hamiltonian():
    Sz1 = np.array([[0.5, 0], [0, -0.5]], dtype='d')  # single-site S^z
    Sp1 = np.array([[0, 1], [0, 0]], dtype='d')  # single-site S^+
    J = 1.0
    ham = ((J / 2) * (np.kron(Sp1, Sp1.conjugate().transpose()) + np.kron(Sp1.conjugate().transpose(), Sp1)) + J * np.kron(Sz1, Sz1)).reshape(2, 2, 2, 2)

    return ham


