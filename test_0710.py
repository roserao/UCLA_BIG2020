from pysnptools.snpreader import Bed
import numpy as np
import pandas as pd

def select_causal_gene(geno, num_snps, num_causal, num_iso, tss):
    # pseudo normal distribution around TSS
    snp_pos = geno.col_property[:, 2]
    mean = (np.abs(snp_pos - tss)).argmin()
    var = num_snps * 0.1
    # select causal gene
    snp_causal_idx = np.zeros([num_causal, num_iso], dtype=int)
    for j in range(num_iso):
        idx = np.random.normal(size=int(num_causal*1.2), loc = mean, scale = var).astype(int) # duplicates
        idx = np.unique(idx)[0:num_causal]
        idx.sort()
        assert len(idx) >= num_causal
        assert (idx >= 0).all() and (idx <= num_snps).all() # out of range
        snp_causal_idx[:, j] = idx
    return snp_causal_idx

def read_ld_save(ldfile, snp_causal_idx, num_iso):
    ld_full = pd.read_table(ldfile, header=None, sep=' ')
    snp_selected = snp_causal_idx[:, 0]
    for j in range(num_iso):
        snp_selected = np.union1d(snp_selected, snp_causal_idx[:, j])
        ld = ld_full.filter(items=snp_causal_idx[:, j], axis=0).filter(items=snp_causal_idx[:, j], axis=1)
        ld.to_csv('ld{}.txt'.format(j), header = False, index=False)
    ld_selected = ld_full.filter(items=snp_selected, axis=0).filter(items=snp_selected, axis=1)
    ld_selected.to_csv('ld_selected.txt'.format(j), header=False, index=False)

def compute_beta(h2g, num_iso, num_causal, cov_iso, snp_causal_idx):
    # special case: only two transcript
    # special case: all beta value is the same within one transcript
    # special case: all beta value is the same among all transcripts
    ld = pd.read_csv('ld_selected.txt', sep=',', header=None).to_numpy()
    all_snps = np.concatenate((snp_causal_idx[:, 0], snp_causal_idx[:, 1]))
    snps, counts_snps = np.unique(all_snps, return_counts=True)
    assert len(snps) == ld.shape[1] == ld.shape[0]
    c = h2g * np.sum(cov_iso) / np.linalg.multi_dot([counts_snps, ld, counts_snps])
    betas = np.ones([num_causal, num_iso]) *np.sqrt(c)
    np.savetxt('betas.txt', betas, delimiter=',')
    return betas

def compute_iso_h2(num_iso):
    betas = pd.read_csv("betas.txt", sep=',', header=None).to_numpy()
    h2is = np.zeros(num_iso)
    for j in range(num_iso):
        ld = pd.read_csv('ld{}.txt'.format(1), sep=',', header=None).to_numpy()
        assert len(ld) == betas.shape[0]
        h2is[j] = np.linalg.multi_dot([betas[:, j], ld, betas[:, j]])
    return h2is

def compute_phe_g(geno, snp_causal_idx, num_indv):
    phe_g = np.zeros([num_indv, num_iso])
    betas = pd.read_csv("betas.txt", sep=',', header=None).to_numpy()
    for j in range(num_iso):
        geno_val = geno[:, snp_causal_idx[:, j]].read().val
        f = np.sum(geno_val, axis = 0) / (2 * num_indv)
        std_geno_val = (geno_val - 2 * f) / np.sqrt(2 * f * (1-f))
        phe_g[:, j] = np.dot(std_geno_val, betas[:, j])
    return phe_g

def compute_phe(geno, snp_causal_idx, num_indv, h2is):
    phe_g = compute_phe_g(geno, snp_causal_idx, num_indv)
    phe = np.zeros_like(phe_g)
    for j in range(num_iso):
        phe_e = np.random.normal(loc=0.0, scale=1-h2is[j], size=num_indv)
        phe[:, j] = phe_e + phe_g[:, j]
    np.savetxt('phe.txt', phe, delimiter=',')

if __name__ == '__main__':
    # config
    tss = 234263153
    np.random.seed(1234)
    bfile = "dgkd.sample.SNPs.clean"
    ldfile = "plink.ld"

    # read in genotype
    geno = Bed(bfile, count_A1=False)
    num_indv, num_snps = geno.shape
    num_causal = int(num_snps * 0.01)  # for each isoform

    # isoforms
    num_iso = 2
    h2g = 0.08
    cov_iso = np.array([[1, 0.03], [0.03, 1]])

    # select causal gene
    snp_causal_idx = select_causal_gene(geno, num_snps, num_causal, num_iso, tss)
    print("causal gene randomly selected")

    # save only relevant ld matrix
    # read_ld_save(ldfile, snp_causal_idx, num_iso)
    print("LD matrix extracted")

    # compute beta
    compute_beta(h2g, num_iso, num_causal, cov_iso, snp_causal_idx)
    print("betas for causal gene generated")

    # compute isoform heritability
    h2is = compute_iso_h2(num_iso)
    print("isoform heritablity computed: " + str(h2is))

    # compute phenotype
    compute_phe(geno, snp_causal_idx, num_indv, h2is)
    print("overall phenotype computed")

    print("finished")
