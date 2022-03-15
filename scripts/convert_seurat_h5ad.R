library(Seurat)
library(SeuratDisk)

cohort1 <- readRDS("/prj/NUM_CODEX_PLUS/Covid19_data/cohort1.annote.rds")
SaveH5Seurat(cohort1, filename = "cohort1.h5Seurat")
Convert("cohort1.h5Seurat", dest = "h5ad")