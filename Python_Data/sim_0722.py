from pysnptools.snpreader import Bed
import numpy as np
import pandas as pd
import os
import argparse

class BaseModel:

    def __init__(self):
        # basic
        np.random.seed(1345)
        self.bfile = "dgkd.AFR.SNPs.clean"
        self.num_sim = 100
        # genotype
        geno = Bed(self.bfile, count_A1=False)
        self.tss = 234263153
        self.num_indv_all, self.num_snps_all = geno.shape
        # sampling
        self.num_indv = 200
        self.num_snp = 10
        self.num_causal = 5
        # isoform
        self.num_iso = 2
        self.h2g = 0.20
        self.cov_iso = np.array([[1, 0.5], [0.5, 1]])

    def sample(self, gcta):
        # randomly select individuals, snps, and causal genes
        assert self.num_indv_all >= self.num_indv
        assert self.num_snps_all >= self.num_snp
        indv_idx = np.zeros([self.num_indv, self.num_sim], dtype=int)
        snp_idx = np.zeros([self.num_snp, self.num_sim], dtype=int)
        snp_causal = np.zeros([self.num_causal, self.num_sim], dtype=int)
        for sim_i in range(self.num_sim):
            indv_idx[:, sim_i] = np.random.choice(self.num_indv_all, size = self.num_indv, replace = False)
            snp_idx[:, sim_i] = np.random.choice(self.num_snps_all, size = self.num_snp, replace = False)
            # causal SNPs of two transcripts different
            # default first 2 for transcript 1, last 2 for transcript 2
            snp_causal[:, sim_i] = np.random.choice(snp_idx[:, sim_i], size = self.num_causal, replace = False)
        if not os.path.exists("./sample"):
            os.makedirs("./sample")
        indv_idx = np.sort(indv_idx, axis = 0)
        snp_idx = np.sort(snp_idx, axis= 0)
        snp_causal = np.sort(snp_causal, axis=0)
        np.savetxt("./sample/indv_idx.txt", indv_idx, fmt='%i', delimiter=",")
        np.savetxt("./sample/snp_idx.txt", snp_idx, fmt='%i', delimiter=",")
        np.savetxt("./sample/snp_causal.txt", snp_causal, fmt='%i', delimiter=",")
        if gcta == True: # add feature for GCTA GREML
            indv_list = pd.read_table("AFR_list.txt", header=None)
            snp_list = pd.read_table("dgkd.AFR.SNPs.clean.bim", header=None, sep='\t')
            for sim_i in range(self.num_sim):
                indv = indv_list.iloc[indv_idx[:, sim_i], 0]
                snp = snp_list.iloc[snp_idx[:, sim_i], 1]
                indv.to_csv("./sample/indv_sample{}.txt".format(sim_i), header=False, index=False)
                snp.to_csv("./sample/snp_sample{}.txt".format(sim_i), header=False, index=False)

    def ld_extract(self):
        # save only relevant ld matrix in the folder ld
        geno = Bed(self.bfile, count_A1=False)
        snp_causal = np.loadtxt("./sample/snp_causal.txt", delimiter=",", dtype=int)
        snp_idx = np.loadtxt("./sample/snp_idx.txt", delimiter=",", dtype=int)
        indv_idx = np.loadtxt("./sample/indv_idx.txt", delimiter=",", dtype=int)
        if not os.path.exists("./ld"):
            os.makedirs("./ld")
        for sim_i in range(self.num_sim):
            geno_val_f = geno[indv_idx[:, sim_i], snp_idx[:, sim_i]].read().val
            f = np.sum(geno_val_f, axis=0) / (2 * self.num_indv)
            std_geno_val_f = (geno_val_f - 2 * f) / np.sqrt(2 * f * (1 - f))
            ld_f = np.cov(np.transpose(std_geno_val_f), ddof=0)
            idx = np.isin(snp_idx[:, sim_i], snp_causal[:, sim_i])
            ld_c = ld_f[idx, :][:, idx]
            np.savetxt('./ld/ld_c{}.txt'.format(sim_i), ld_c, delimiter=",")
            np.savetxt('./ld/ld_f{}.txt'.format(sim_i), ld_f, delimiter=",")

    def compute_beta(self, option_iso):
        # special case: only two transcript
        # special case: all beta value is the same within one transcript
        # special case: causal genes are the same for two transcripts
        betas = np.zeros([self.num_causal, self.num_iso])
        if not os.path.exists("./beta"):
            os.makedirs("./beta")
        for sim_i in range(self.num_sim):
            ld = np.loadtxt('./ld/ld_c{}.txt'.format(sim_i), delimiter=",")
            c = np.sqrt(self.h2g * np.sum(self.cov_iso) / np.sum(ld))
            betas[:, 0] = c * option_iso
            betas[:, 1] = c * (1 - option_iso)
            np.savetxt("./beta/beta{}.txt".format(sim_i), betas, delimiter=",")

    def compute_iso_h2(self):
        h2is = np.zeros([self.num_iso, self.num_sim])
        for sim_i in range(self.num_sim):
            beta = np.loadtxt("./beta/beta{}.txt".format(sim_i), delimiter=",")
            ld = np.loadtxt('./ld/ld_c{}.txt'.format(sim_i), delimiter=",")
            h2is[0, sim_i] = np.linalg.multi_dot([beta[:, 0], ld, beta[:, 0]])
            h2is[1, sim_i] = np.linalg.multi_dot([beta[:, 1], ld, beta[:, 1]])
            assert h2is[0, sim_i] <= 1, "Error: impossible to compute"
            assert h2is[1, sim_i] <= 1, "Error: impossible to compute"
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
            phe_g[:, 0] = np.dot(std_geno_val, beta[:, 0])
            phe_g[:, 1] = np.dot(std_geno_val, beta[:, 1])
            phe_g_np[sim_i, :, :] = phe_g
        np.save("./phe/phe_g_full.npy", phe_g_np)

    def compute_phe(self, gcta):
        h2is = np.loadtxt("./h2is.txt", delimiter=",")
        phe_full = np.zeros([self.num_sim, self.num_indv, self.num_iso])
        phe_gene_std =  np.zeros([self.num_sim, self.num_indv])
        phe_g_full = np.load("./phe/phe_g_full.npy")
        for sim_i in range(self.num_sim):
            phe_g = phe_g_full[sim_i, :, :]
            phe = np.zeros_like(phe_g)
            phe_g_cov = np.cov(np.transpose(phe_g), ddof=0)[0, 1]
            cov_iso = np.array([[1 - h2is[0, sim_i], self.cov_iso[0, 1] - phe_g_cov],
                                [self.cov_iso[0, 1] - phe_g_cov, 1 - h2is[1, sim_i]]])
            for indv_j in range(self.num_indv):
                phe_indv = np.random.multivariate_normal(mean=np.array([0, 0]), cov=cov_iso)
                phe[indv_j, :] = phe_indv + phe_g[indv_j, :]
            phe_full[sim_i, :, :] = phe
            # Overall gene phenotype standardization
            phe_gene = phe[:, 0] + phe[:, 1]
            phe_gene = (phe_gene - np.mean(phe_gene))/np.std(phe_gene)
            phe_gene_std[sim_i, :] = phe_gene
            # add feature for GCTA GREML
            if gcta == True:
                ID = pd.read_table("./sample/indv_sample{}.txt".format(sim_i), header=None)
                pheno = pd.concat([ID, ID], axis=1)
                temp = pd.DataFrame(data =  {'Gene': phe_gene, 'Phe1': phe[:, 0], 'Phe2': phe[:, 1]})
                pheno = pd.concat([pheno, temp], axis = 1)
                pheno.to_csv('./phe/phe{}.txt'.format(sim_i), header=False, index=False, sep = '\t')
        np.save("./phe/phe_full.npy", phe_full)
        np.save("./phe/phe_gene_std.npy", phe_gene_std)

    def hess_h2g(self):
        geno = Bed(self.bfile, count_A1=False)
        indv_idx = np.loadtxt("./sample/indv_idx.txt", delimiter=",", dtype=int)
        snp_idx = np.loadtxt("./sample/snp_idx.txt", delimiter=",", dtype=int)
        phe = np.load("./phe/phe_gene_std.npy")
        n = self.num_indv
        p = self.num_snp
        h2g_est = np.zeros(self.num_sim)
        for sim_i in range(self.num_sim):
            geno_val = geno[indv_idx[:, sim_i], snp_idx[:, sim_i]].read().val
            f = np.sum(geno_val, axis=0) / (2 * self.num_indv)
            X = (geno_val - 2 * f) / np.sqrt(2 * f * (1 - f))
            y = phe[sim_i, :]
            beta_est = np.matmul(np.transpose(X), y) / n
            V = np.loadtxt('./ld/ld_f{}.txt'.format(sim_i), delimiter=",")
            V_pinv = np.linalg.pinv(V)
            q = np.trace(np.matmul(V_pinv, V))
            h2g = (n * np.linalg.multi_dot([beta_est, V_pinv, beta_est]) - q) / (n - q)
            h2g_est[sim_i] = h2g
        np.savetxt("h2g_hess.txt", h2g_est, delimiter=",")
        print("HESS:")
        print(np.nanmean(h2g_est), np.nanvar(h2g_est), np.nanmedian(h2g_est))
        var = (n/(n-p))**2 * (2*p*((1-self.h2g)/n)+4*self.h2g) * ((1-self.h2g)/n)
        print(var)


    def h2g_formula(self):
        # import data
        n = self.num_indv
        geno = Bed(self.bfile, count_A1=False)
        indv_idx = np.loadtxt("./sample/indv_idx.txt", delimiter=",", dtype=int)  # 100 indv
        snp_idx = np.loadtxt("./sample/snp_idx.txt", delimiter=",", dtype=int)  # 10 snp
        snp_causal = np.loadtxt("./sample/snp_causal.txt", delimiter=",", dtype=int)
        phe_gene = np.load("./phe/phe_gene_std.npy")
        phe_iso = np.load("./phe/phe_full.npy")
        # calculate
        h2g_est_iso = np.zeros(self.num_sim)
        h2g_est_gene = np.zeros(self.num_sim)
        for sim_i in range(self.num_sim):
            # standardize genotype
            geno_val = geno[indv_idx[:, sim_i], snp_causal[:, sim_i]].read().val
            f = np.sum(geno_val, axis=0) / (2 * self.num_indv)
            X = (geno_val - 2 * f) / np.sqrt(2 * f * (1 - f))
            # phenotype and beta estimation
            y_gene = phe_gene[sim_i, :]
            beta_est_gene = np.matmul(np.transpose(X), y_gene) / n
            y_iso0 = phe_iso[sim_i, :, 0]
            y_iso1 = phe_iso[sim_i, :, 1]
            alpha0_est = np.matmul(np.transpose(X), y_iso0) / n
            alpha1_est = np.matmul(np.transpose(X), y_iso1) / n
            beta_est_iso = alpha0_est + alpha1_est
            # LD
            V = np.loadtxt('./ld/ld_c{}.txt'.format(sim_i), delimiter=",")
            # heritability
            #idx = np.isin(snp_idx[:, sim_i], snp_causal[:, sim_i])
            #beta_est_gene = beta_est_gene[idx]
            #beta_est_iso = beta_est_iso[idx]

            #h2g_gene = np.linalg.multi_dot([beta_est_gene, V, beta_est_gene])
            n = self.num_indv
            p = self.num_causal
            h2g_gene = (n*np.linalg.multi_dot([beta_est_gene, np.linalg.inv(V), beta_est_gene]) - p) / (n - p)

            cov_iso = np.cov(np.transpose(phe_iso[sim_i, :, :]), ddof=1)
            #h2g_iso = np.linalg.multi_dot([beta_est_iso, V, beta_est_iso])/np.sum(cov_iso)
            h2i_1 = (n*np.linalg.multi_dot([alpha0_est, np.linalg.inv(V), alpha0_est]) - p) / (n - p)
            h2i_2 = (n*np.linalg.multi_dot([alpha1_est, np.linalg.inv(V), alpha1_est]) - p) / (n - p)
            h2g_iso = (np.linalg.multi_dot([beta_est_iso, np.linalg.inv(V), beta_est_iso]) - ((2-h2i_1-h2i_2)*p/n) ) / np.sum(cov_iso)

            h2g_est_gene[sim_i] = h2g_gene
            h2g_est_iso[sim_i] = h2g_iso
        np.savetxt("h2g_formula_gene.txt", h2g_est_gene, delimiter=",")
        np.savetxt("h2g_formula_iso.txt", h2g_est_iso, delimiter=",")
        print("GENE")
        print(np.nanmean(h2g_est_gene), np.nanvar(h2g_est_gene), np.nanmedian(h2g_est_gene))
        print("ISO")
        print(np.nanmean(h2g_est_iso), np.nanvar(h2g_est_iso), np.nanmedian(h2g_est_iso))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eQTL Heritability Simulation: two transcripts')
    parser.add_argument('--isoform', type=float, default=0.5,
                        help="percentage of gene heritability attributable to one isoform")
    parser.add_argument('--gcta', dest='gcta', action='store_true',
                        help='generate extra files for GREML analysis')
    args = parser.parse_args()
    print(args.gcta)
    print(args.isoform)

    model = BaseModel()

    if True:
        model.sample(args.gcta)
        model.ld_extract()

    if True:
        model.compute_beta(args.isoform)
        model.compute_iso_h2()
        model.compute_phe_g()
        model.compute_phe(args.gcta)

    if True:
        model.hess_h2g()
        model.h2g_formula()
