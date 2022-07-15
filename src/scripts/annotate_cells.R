library(Seurat)
library(SeuratDisk)
library(future)
set.seed(0)
future.seed = TRUE
plan("multicore", workers = 4)
options(future.globals.maxSize = 8000 * 1024 ^ 2)

cohort_training = snakemake@input[[1]]
out = snakemake@output[[1]]
cohort_ref = LoadH5Seurat(snakemake@params[[1]])
pseud_mtx_all = NULL
#Training set
#Normalization
ch = readRDS(cohort_training)
ch.batches <- SplitObject(ch, split.by =  "batch")
ch.batches <-
  lapply(X = ch.batches, FUN = SCTransform, verbose = FALSE)
#mapping
anchors <- list()
for (i in 1:length(ch.batches)) {
  anchors[[i]] <- FindTransferAnchors(
    reference = cohort_ref,
    query = ch.batches[[i]],
    normalization.method = "SCT",
    reference.reduction = "spca",
    dims = 1:50
  )
}

for (i in 1:length(ch.batches)) {
  ch.batches[[i]] <- MapQuery(
    anchorset = anchors[[i]],
    query = ch.batches[[i]],
    reference = cohort_ref,
    refdata = "celltype.l2",
    reference.reduction = "spca",
    reduction.model = "wnn.umap"
  )
}
saveRDS(ch.batches, paste0(out, "mapped_cohort.rds"))
'%!in%' <- function(x, y)
  ! ('%in%'(x, y))

#Cell composition training set
for (ch in ch.batches) {
  ch$sampleID=factor(ch$sampleID)  
  batch = ch@meta.data$batch[[1]]
  abundances <- table(ch@meta.data$predicted.id, ch$sampleID)
  abundances <- unclass(abundances)
  extra.info <-
    ch@meta.data[match(colnames(abundances), ch$sampleID), ]
  rownames(extra.info) = colnames(abundances)
  seur_cell = CreateSeuratObject(
    abundances,
    project = "abundances.celltype.l2",
    assay = "RNA",
    min.cells = 0,
    min.features = 0,
    names.field = 1,
    names.delim = "__",
    meta.data = extra.info
  )
  
  pseud_mtx = as.data.frame(t(as.matrix(seur_cell@assays$RNA@counts)))
  rownames(pseud_mtx) = seur_cell$sampleID
  outer = unique(cohort_ref$celltype.l2)[unique(cohort_ref$celltype.l2) %!in% colnames(pseud_mtx)]
  pseud_mtx[outer] = 0
  pseud_mtx = pseud_mtx[, sort(names(pseud_mtx))]
  pseud_mtx['condition'] = seur_cell$condition
  pseud_mtx_all = rbind(pseud_mtx_all, pseud_mtx)
}


write.csv(pseud_mtx_all, paste0(out))