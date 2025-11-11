import numpy as np

#####******************************************************************#####

def read_mps(mps):
    """Extracts and detaches all tensors from an mps object."""
    return [t.tensor.detach().cpu().numpy() for t in mps.module_list]

#####*********************************************************************************************#####

def normalization(mps):
    """Compute the inner product ⟨mps|mps⟩"""
    ln = len(mps)
    mps_conj = [t.conj() for t in mps]  # conjugate mps

    # Initialize with the contraction of the first tensor pair
    temp = np.tensordot(mps[0], mps_conj[0], axes=([0], [0]))  # shape: (r, r)

    # Contract middle tensors
    for idx in range(1, ln - 1):
        temp = np.tensordot(temp, mps_conj[idx], axes=([1], [0]))
        temp = np.tensordot(mps[idx], temp, axes=([0, 1], [0, 1]))

    # Final contraction with last tensor
    temp = np.tensordot(temp, mps_conj[-1], axes=([1], [0]))
    res = np.tensordot(mps[-1], temp, axes=([0, 1], [0, 1]))

    return res

#####********************************************************************************************#####

def site_normalization(mps):
    axes = tuple(range(mps.ndim))  # all dimensions
    return np.tensordot(mps, mps.conj(), axes=(axes, axes))


#####********************************************************************************************#####

def calculate_Sz(mps):
    """Compute total Sz expectation value for an MPS."""
    sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    totalSz = 0.0
    ln = len(mps)

    for idx in range(ln):
        # Bring MPS into orthogonal center at site idx
        mps = orth_centre(mps, site=idx + 1)
        mps_conj = mps[idx].conj()

        # Determine axes for tensordot based on position
        if idx == 0:
            # left-end: shape [i, r]
            temp = np.tensordot(np.tensordot(mps[idx], sz, axes=([0], [0])), mps_conj, axes=([1, 0], [0, 1]))
        elif idx == ln - 1:
            # right-end: shape [l, i]
            temp = np.tensordot(np.tensordot(mps[idx], sz, axes=([1], [0])), mps_conj, axes=([0, 1], [0, 1]))
        else:
            # middle sites: shape [l, i, r]
            temp = np.tensordot(np.tensordot(mps[idx], sz, axes=([1], [0])), mps_conj, axes=([0, 2, 1], [0, 1, 2]))

        totalSz += temp

    return totalSz


#####********************************************************************************************************#####

def calculate_ENE(mps, ham):
    """Compute total energy of an MPS"""
    ln = len(mps)
    totalEne = 0.0

    for idx in range(ln - 1):
        mps = orth_centre(mps, site=idx + 1)

        A = mps[idx]
        B = mps[idx + 1]
        A_conj = A.conj()
        B_conj = B.conj()

        if idx == 0:
            temp = np.tensordot(A, B, axes=([1], [0]))          # bond contraction 
            temp = np.tensordot(temp, ham, axes=([0,1], [0,1])) # connect to Hamiltonian
            temp = np.tensordot(temp, A_conj, axes=([1], [0]))  # contract left conj
            temp = np.tensordot(temp, B_conj, axes=([0,1,2], [2,1,0])) # fully contract
            tem = np.squeeze(temp)

        elif idx == ln - 2:
            temp = np.tensordot(A, B, axes=([2], [0]))          # bond contraction
            temp = np.tensordot(temp, ham, axes=([1,2], [0,1])) # connect Hamiltonian
            temp = np.tensordot(temp, A_conj, axes=([0,1], [0,1]))
            temp = np.tensordot(temp, B_conj, axes=([0,1], [1,0]))
            tem = np.squeeze(temp)

        else:
            temp = np.tensordot(A, B, axes=([2], [0]))          # bond contraction
            temp = np.tensordot(temp, ham, axes=([1,2], [0,1])) # connect physical legs
            temp = np.tensordot(temp, A_conj, axes=([0,2], [0,1]))
            temp = np.tensordot(temp, B_conj, axes=([0,1,2], [2,1,0]))
            tem = np.squeeze(temp)

        totalEne += tem

    return totalEne

#####********************************************************************************************************#####

def orth_centre(mps, site):
    """Bring MPS to orthogonal center at the given site (1-indexed)."""
    ln = len(mps)
    orth_mps = []

    # ---- SITE = 1 ----
    if site == 1:
        u, s, vh = np.linalg.svd(mps[ln-1], full_matrices=False)
        orth_mps.insert(0, vh)
        for idx in range(ln-2, 0, -1):
            shap = mps[idx].shape
            mps[idx] = np.tensordot(np.tensordot(mps[idx], u, axes=([2],[0])), np.diag(s), axes=([2],[0]))
            u, s, vh = np.linalg.svd(mps[idx].reshape(shap[0], shap[1]*shap[2]), full_matrices=False)
            orth_mps.insert(0, vh.reshape(shap))
        mps[0] = np.tensordot(np.tensordot(mps[0], u, axes=([1],[0])), np.diag(s), axes=([1],[0]))
        orth_mps.insert(0, mps[0])

    # ---- SITE = ln ----
    elif site == ln:
        u, s, vh = np.linalg.svd(mps[0], full_matrices=False)
        orth_mps.append(u)
        for idx in range(1, ln-1):
            shap = mps[idx].shape
            mps[idx] = np.tensordot(np.tensordot(np.diag(s), vh, axes=([1],[0])), mps[idx], axes=([1],[0]))
            u, s, vh = np.linalg.svd(mps[idx].reshape(shap[0]*shap[1], shap[2]), full_matrices=False)
            orth_mps.append(u.reshape(shap))
        mps[ln-1] = np.tensordot(np.tensordot(np.diag(s), vh, axes=([1],[0])), mps[ln-1], axes=([1],[0]))
        orth_mps.append(mps[ln-1])

    # ---- 1 < SITE < ln ----
    else:
        # Left sweep
        ul, sl, vhl = np.linalg.svd(mps[0], full_matrices=False)
        orth_mps_l = [ul]
        for idx in range(1, site-1):
            shap = mps[idx].shape
            mps[idx] = np.tensordot(np.tensordot(np.diag(sl), vhl, axes=([1],[0])), mps[idx], axes=([1],[0]))
            ul, sl, vhl = np.linalg.svd(mps[idx].reshape(shap[0]*shap[1], shap[2]), full_matrices=False)
            orth_mps_l.append(ul.reshape(shap))

        # Right sweep
        ur, sr, vhr = np.linalg.svd(mps[ln-1], full_matrices=False)
        orth_mps_r = [vhr]
        for idx in range(ln-2, site-1, -1):
            shap = mps[idx].shape
            mps[idx] = np.tensordot(np.tensordot(mps[idx], ur, axes=([2],[0])), np.diag(sr), axes=([2],[0]))
            ur, sr, vhr = np.linalg.svd(mps[idx].reshape(shap[0], shap[1]*shap[2]), full_matrices=False)
            orth_mps_r.insert(0, vhr.reshape(shap))

        # Orthogonality center
        shap = mps[site-1].shape
        tensor = np.tensordot(np.tensordot(np.diag(sl), vhl, axes=([1],[0])), mps[site-1], axes=([1],[0]))
        tensor = np.tensordot(np.tensordot(tensor, ur, axes=([2],[0])), np.diag(sr), axes=([2],[0]))
        orth_mps_l.append(tensor)

        orth_mps = orth_mps_l + orth_mps_r

    # ---- Normalize ----
    norm = np.sqrt(normalization(orth_mps))
    orth_mps[0] /= norm

    return orth_mps


#####*****************************************************************************************************#####

def heisenberg_hamiltonian():
    Sz1 = np.array([[0.5, 0], [0, -0.5]], dtype='d')  # single-site S^z
    Sp1 = np.array([[0, 1], [0, 0]], dtype='d')  # single-site S^+
    J = 1.0
    ham = ((J / 2) * (np.kron(Sp1, Sp1.conjugate().transpose()) + np.kron(Sp1.conjugate().transpose(), Sp1)) + J * np.kron(Sz1, Sz1)).reshape(2, 2, 2, 2)

    return ham


