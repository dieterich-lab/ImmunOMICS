library(Seurat)
set.seed(0)
cohort = snakemake@input
n = length(cohort)
print(n)
for (i in 1:n) {
  #     cohort_path= paste0(path,"/",chr,".rds")
  print(cohort[[i]])
  chrt = readRDS(cohort[[i]])
  chr_n = strsplit(cohort[[i]], "/")[[1]]
  chr_n = chr_n[length(chr_n)]
  chrt@meta.data['batch'] = paste0(chrt@meta.data$batch, "_trainset_", i)
  if (i==1) {
    cohort_merged = chrt
  }
  else {
    cohort_merged = merge(cohort_merged, chrt)
  }
}
saveRDS(cohort_merged, snakemake@output[[1]])