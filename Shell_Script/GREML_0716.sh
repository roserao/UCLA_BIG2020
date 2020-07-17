#!/bin/bash
#$ -cwd							
#$ -j y
#$ -l h_data=4G,h_rt=4:00:00
#$ -o ./OutFiles
#$ -e ./OutFiles

# update paths
# gcta=/u/local/apps/gcta/0.93.9/gcc-4.4.7/gcta64
# plink=/u/local/apps/plink/1.90b3.45/plink
# vcftools=/u/local/apps/vcftools/0.1.14/gcc-4.4.7/bin/vcftools
. /u/local/Modules/default/init/modules.sh
module load vcftools
module load plink
module load gcta

# local directories, data directories, results directors
ldir=/u/home/r/roserao
ddir=$ldir/data/
rdir=$ldir/results/

# format of file looks like this...
#DGKD	chr2	234263153	234380750	-

for sim_i in `seq 0 99`
do
    
    tdir=$SCRATCH/$sim_i
    if [ -d $tdir ]; then
        rm -r $tdir
    fi

    mkdir $tdir

    vcftools --vcf $ddir/dgkd.AFR.SNPs.clean.vcf \
        --keep $ddir/sample/indv_sample${sim_i}.txt \
        --snps $ddir/sample/snp_sample${sim_i}.txt \
        --recode --recode-INFO-all --out $tdir/sample${sim_i}

    vcftools --vcf $tdir/sample${sim_i}.recode.vcf --plink --out $tdir/DGKD${sim_i}

    plink --file $tdir/DGKD${sim_i} --make-bed --out $tdir/DGKD${sim_i}

    gcta64 --bfile $tdir/DGKD${sim_i} --make-grm --out $tdir/DGKD${sim_i}.cis

    for idx in `seq 1 3`
    do
        # h2g analysis
        gcta64 --reml \
            --pheno $ddir/phe/phe${sim_i}.txt \
            --mpheno $idx \
            --grm $tdir/DGKD${sim_i}.cis \
            --out $rdir/res${sim_i}.${idx}.cis
    done

done


