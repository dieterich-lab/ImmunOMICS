# library(Seurat)
library(SeuratDisk)
library(dplyr)

inp = snakemake@input[[1]]
fc_file = snakemake@input[[2]]
out = snakemake@output[[1]]

ch_avr = LoadH5Seurat(inp)
fc = read.csv(fc_file, row.names = 1)
top <- fc %>% top_n(n = 15, wt = abs(avg_log2FC))

'%!in%' <- function(x, y)
  ! ('%in%'(x, y))

outer= rownames(top)[rownames(top) %!in% rownames(ch_avr)]
inner= rownames(top)[rownames(top) %in% rownames(ch_avr)]
pseud_mtx = as.data.frame(t(as.matrix(ch_avr@assays$RNA@data[inner, ])))
pseud_mtx[outer]=0
pseud_mtx = pseud_mtx[, sort(names(pseud_mtx))]
rownames(pseud_mtx) = ch_avr$sampleID
pseud_mtx['condition'] = ch_avr$condition
pseud_mtx['who_score'] = ch_avr$who_score
write.csv(pseud_mtx, out)