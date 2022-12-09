library(Seurat)
library(SingleR)
library(celldex)
library(SeuratDisk)

set.seed(0)
#  Load  QC data
cohort_training = snakemake@input[[1]]
out = snakemake@output[[1]]
celltypes = snakemake@output[[2]]

pseud_mtx_all = NULL
ch = readRDS(cohort_training)
# Devide into batches
ch.batches <- SplitObject(ch, split.by =  "batch")
# Normalize per batch
ch.batches <-
  lapply(
    X = ch.batches,
    FUN = NormalizeData,
    normalization.method = "LogNormalize",
    scale.factor = 10000,
    verbose = FALSE
  )
'%!in%' <- function(x, y)
  ! ('%in%'(x, y))
# Load monaco dataset for annotation
monaco.se <- MonacoImmuneData()
i = 0
# Annotate per batch than merge cell composition matrices
for (ch in ch.batches) {
  input <- GetAssayData(object = ch,
                        slot = "data",
                        assay = "RNA")
  monaco <- SingleR(
    test = input,
    method = "single",
    fine.tune = FALSE,
    ref = monaco.se,
    labels = monaco.se$label.fine
  )
  
  ch$monaco.labels <- monaco$labels
  if (i == 0) {
    ch_all = ch
  }
  else {
    ch_all = merge(ch_all,
                   y = ch,
                   add.cell.ids = c("", unique(ch$batch)[1]))
  }
  i = i + 1
}

ch_all$monaco.labels[ch_all$monaco.labels %in% c(
  "Naive B cells",
  'Non-switched memory B cells',
  'Switched memory B cells')] = 'B cells'
ch_all$monaco.labels[ch_all$monaco.labels %in% c(
  "Th1/Th17 cells" ,
  "Th17 cells" ,
  "Th2 cells",
  "Follicular helper T cells",
  "Th1 cells",
  "Terminal effector CD4 T cells",
  "Naive CD4 T cells"
)] = 'CD4 T cells'
ch_all$monaco.labels[ch_all$monaco.labels %in% c(
  "Terminal effector CD8 T cells",
  "Effector memory CD8 T cells",
  "Central memory CD8 T cells",
  "Naive CD8 T cells"
)] = "CD8 T cells"

ch_all$monaco.labels[ch_all$monaco.labels %in% c("Vd2 gd T cells", "Non-Vd2 gd T cells")] = 'gd T cells'


SaveH5Seurat(ch_all, filename = out, overwrite = TRUE)
write.csv(unique(ch_all$monaco.labels), celltypes, row.names = FALSE)
