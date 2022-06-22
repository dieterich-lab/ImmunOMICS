library(Seurat)
library(SeuratDisk)
inp = snakemake@input[[1]]
out = snakemake@output[[1]]

ch = LoadH5Seurat(inp)
Idents(ch) = 'condition'
markers <-
  FindMarkers(ch,
              ident.1 = "Severe",
              ident.2 = "Mild",
              verbose = FALSE, 
              latent.vars = "batch")
write.csv(markers, out)