from pysnptools.snpreader import Bed
import numpy as np
import pandas as pd
import os

class BaseModel:

    def __init__(self):
        # basic
        np.random.seed(1234)
        self.bfile = "dgkd.AFR.SNPs.clean"
        self.ldfile = "dgkd.AFR.SNPs.clean.ld"
        self.num_sim = 100
        # genotype
        geno = Bed(self.bfile, count_A1=False)
        self.tss = 234263153
        self.num_indv_all, self.num_snps_all = geno.shape
        # sampling
        self.num_indv = 100
        self.num_snp = 10
        self.num_causal = 2
        # isoform
        self.num_iso = 2
        self.h2g = 0.08
        self.cov_iso = np.array([[1, 0.3], [0.3, 1]])

    def submit(self, task):
        assert (task in ['sample', 'ld', 'beta', 'phe_g', 'phe', 'hess'])
        if task == 'sample':
            self.sample()
        elif task == 'ld':
            self.ld_extract()
        elif task == 'beta':
            self.compute_beta()
        elif task == 'h2i':
            self.compute_iso_h2()
        elif task == 'phe_g':
            self.compute_phe_g()
        elif task == 'phe':
            self.compute_phe()
        elif task == 'hess':
            self.hess_h2g()
        else:
            raise NotImplementedError

    def sample(self):
        # randomly select individuals, snps, and causal genes
        assert self.num_indv_all >= self.num_indv
        assert self.num_snps_all >= self.num_snp
        indv_idx = np.zeros([self.num_indv, self.num_sim], dtype=int)
        snp_idx = np.zeros([self.num_snp, self.num_sim], dtype=int)
        snp_causal = np.zeros([self.num_causal, self.num_sim], dtype=int)
        for sim_i in range(self.num_sim):
            indv_idx[:, sim_i] = np.random.choice(self.num_indv_all, size = self.num_indv, replace = False)
            snp_idx[:, sim_i] = np.random.choice(self.num_snps_all, size = self.num_snp, replace = False)
            # TODO: normal distribution of causal gene selection
            snp_causal[:, sim_i] = np.random.choice(snp_idx[:, sim_i], size = self.num_causal, replace = False)
        if not os.path.exists("./sample"):
            os.makedirs("./sample")
        indv_idx = np.sort(indv_idx, axis = 0)
        snp_idx = np.sort(snp_idx, axis=0)
        snp_causal = np.sort(snp_causal, axis=0)
        np.savetxt("./sample/indv_idx.txt", indv_idx, fmt='%i', delimiter=",")
        np.savetxt("./sample/snp_idx.txt", snp_idx, fmt='%i', delimiter=",")
        np.savetxt("./sample/snp_causal.txt", snp_causal, fmt='%i', delimiter=",")

    def ld_extract(self):
        # save only relevant ld matrix in the folder ld
        ld_full = pd.read_table(self.ldfile, header=None, sep=' ')
        snp_causal = np.loadtxt("./sample/snp_causal.txt", delimiter=",")
        snp_idx = np.loadtxt("./sample/snp_idx.txt", delimiter=",")
        if not os.path.exists("./ld"):
            os.makedirs("./ld")
        for sim_i in range(self.num_sim):
            causal = snp_causal[:, sim_i]
            full = snp_idx[:, sim_i]
            ld_c = ld_full.filter(items=causal, axis=0).filter(items=causal, axis=1)
            ld_f = ld_full.filter(items=full, axis=0).filter(items=full, axis=1)
            ld_c.to_csv('./ld/ld_c{}.txt'.format(sim_i), header=False, index=False)
            ld_f.to_csv('./ld/ld_f{}.txt'.format(sim_i), header=False, index=False)

    def compute_beta(self):
        # special case: only two transcript
        # special case: all beta value is the same within one transcript
        # special case: all beta value is the same among all transcripts
        # special case: causal genes are the same for two transcripts
        betas = np.ones([self.num_causal, self.num_iso])
        if not os.path.exists("./beta"):
            os.makedirs("./beta")
        for sim_i in range(self.num_sim):
            ld = np.loadtxt('./ld/ld_c{}.txt'.format(sim_i), delimiter=",")
            c = self.h2g * np.sum(self.cov_iso) / (self.num_iso ** 2) / np.sum(ld)
            betas[:, :] = np.sqrt(c)
            np.savetxt("./beta/beta{}.txt".format(sim_i), betas, delimiter=",")

    def compute_iso_h2(self):
        h2is = np.zeros([self.num_iso, self.num_sim])
        for sim_i in range(self.num_sim):
            beta = np.loadtxt("./beta/beta{}.txt".format(sim_i), delimiter=",")
            # special case: h2 for both transcripts are the same
            ld = np.loadtxt('./ld/ld_c{}.txt'.format(sim_i), delimiter=",")
            h2is[:, sim_i] = np.linalg.multi_dot([beta[:, 0], ld, beta[:, 0]])
        np.savetxt("./h2is.txt", h2is, delimiter=",")

    def compute_phe_g(self):
        geno = Bed(self.bfile, count_A1=False)
        indv_idx = np.loadtxt("./sample/indv_idx.txt", delimiter=",", dtype=int) # 100 indv
        snp_causal = np.loadtxt("./sample/snp_causal.txt", delimiter=",",dtype=int) # 10 snp
        phe_g = np.zeros([self.num_indv, self.num_iso])
        phe_g_np = np.zeros([self.num_sim, self.num_indv, self.num_iso])
        if not os.path.exists("./phe"):
            os.makedirs("./phe")
        for sim_i in range(self.num_sim):
            beta = np.loadtxt("./beta/beta{}.txt".format(sim_i), delimiter=",")
            geno_val = geno[indv_idx[:, sim_i], snp_causal[:, sim_i]].read().val
            f = np.sum(geno_val, axis=0) / (2 * self.num_indv)
            std_geno_val = (geno_val - 2 * f) / np.sqrt(2 * f * (1 - f))
            # special case: all beta value is the same among all transcripts
            phe_g[:, 0] = np.dot(std_geno_val, beta[:, 0])
            phe_g[:, 1] = np.dot(std_geno_val, beta[:, 1])
            np.savetxt("./phe/phe_g{}.txt".format(sim_i), phe_g, delimiter=",")
            phe_g_np[sim_i, :, :] = phe_g
        np.save("./phe/phe_g_full.npy", phe_g_np)

    def compute_phe(self):
        h2is = np.loadtxt("./h2is.txt", delimiter=",")
        phe_full = np.zeros([self.num_sim, self.num_indv, self.num_iso])
        phe_gene_std =  np.zeros([self.num_sim, self.num_indv])
        for sim_i in range(self.num_sim):
            phe_g = np.loadtxt("./phe/phe_g{}.txt".format(sim_i), delimiter=",")
            phe = np.zeros_like(phe_g)
            for iso_j in range(self.num_iso):
                phe_e = np.random.normal(loc=0.0, scale=1-h2is[iso_j, sim_i], size=self.num_indv)
                phe[:, iso_j] = phe_e + phe_g[:, iso_j]
            phe_full[sim_i, :, :] = phe
            np.savetxt("./phe/phe{}.txt".format(sim_i), phe, delimiter=",")
            # standardize
            # special case: two transcripts
            phe_gene = phe[:, 0] + phe[:, 1]
            phe_gene = (phe_gene - np.mean(phe_gene))/np.std(phe_gene)
            phe_gene_std[sim_i, :] = phe_gene
        np.save("./phe/phe_full.npy", phe_full)
        np.save("./phe/phe_gene_std.npy", phe_gene_std)

    def hess_h2g(self):
        geno = Bed(self.bfile, count_A1=False)
        indv_idx = np.loadtxt("./sample/indv_idx.txt", delimiter=",", dtype=int)  # 100 indv
        snp_idx = np.loadtxt("./sample/snp_idx.txt", delimiter=",", dtype=int)  # 10 snp
        phe = np.load("./phe/phe_gene_std.npy")
        h2g_est = np.zeros(self.num_sim)
        for sim_i in range(self.num_sim):
            geno_val = geno[indv_idx[:, sim_i], snp_idx[:, sim_i]].read().val
            f = np.sum(geno_val, axis=0) / (2 * self.num_indv)
            X = (geno_val - 2 * f) / np.sqrt(2 * f * (1 - f))
            y = phe[sim_i, :]
            n = self.num_indv
            beta_est = np.matmul(np.transpose(X), y) / n
            V = np.loadtxt('./ld/ld_f{}.txt'.format(sim_i), delimiter=",")
            V_pinv = np.linalg.pinv(V)
            q = np.trace(np.matmul(V_pinv, V))
            h2g = (n * np.linalg.multi_dot([beta_est, V_pinv, beta_est]) - q) / (n - q)
            h2g_est[sim_i] = h2g
        np.savetxt("h2g_hess.txt", h2g_est, delimiter=",")

if __name__ == '__main__':
    model = BaseModel()
    model.sample()
    model.ld_extract()
    model.compute_beta()
    model.compute_iso_h2()
    model.compute_phe_g()
    model.compute_phe()
    model.hess_h2g() # change the function here

