library(Seurat)
library(SeuratDisk)
library(DESeq2)
library(tidyverse)
library(edgeR)

set.seed(0)
# Load aggregated counts to sample level
inp <- snakemake@input[[1]]
out1 <- snakemake@params[[1]]
cohort <- LoadH5Seurat(inp)
ct = snakemake@params[[2]]
cohort$celltype = gsub('/', '', cohort$celltype)
ch = subset(cohort, subset = celltype == ct)
cts <- ch$RNA@counts
# Run edgeR --------
y <- DGEList(cts, samples = ch@meta.data)
keep <- filterByExpr(y, group = ch@meta.data$condition)
y <- y[keep, ]
y <- calcNormFactors(y)
design <- model.matrix( ~ 0 + batch + factor(condition), y$samples)
y <- estimateDisp(y, design)
fit <- glmQLFit(y, design, robust = TRUE)
resedger <- glmQLFTest(fit, coef = ncol(design))
#Filter genes with |logFC|>2 and FDR<0.05
if (sum(abs(resedger$table$logFC) >= 2) > 0) {
  res3 = topTags(resedger, n = sum(abs(resedger$table$logFC) >= 2), sort.by =
                   "logFC")$table
  res3 = res3[res3$FDR < 0.05, ]
} else{
  res3 = topTags(resedger, n = 1)$table
  rownames(res3) = c('ee')
  
}


# perform DESeq2 --------
# Create DESeq2 object
dds <- DESeqDataSetFromMatrix(
  countData = cts,
  colData = ch@meta.data,
  design = ~ batch + condition
)
# filter
dds <- dds[keep, ]
# run DESeq2
dds <- DESeq(dds)
# Generate results object
markers <- results(dds, name = "condition_Severe_vs_Mild")
res2 = as.data.frame(markers)
#Filter genes with |logFC|>2 and FDR<0.05
res2 = res2[abs(res2$log2FoldChange) >= 2 & res2$padj < 0.05, ]
# select intersection between TOP 10 of filtrerd genes from DESeq and edgeR
top_desq =  res2
top_edgeR =  res3
selected = rownames(top_edgeR)[rownames(top_edgeR) %in% rownames(top_desq)]

#Save markers, counts and ...
dir.create(paste0(paste0(out1, "/"), gsub("/", "", ct)))
write.csv(as.data.frame(markers),
          paste0(paste0(paste0(out1, "/") , gsub("/", "", ct)), "/fold_change_DESeq.csv"))
write.csv(as.data.frame(resedger$table),
          paste0(paste0(paste0(out1, "/"), gsub("/", "", ct)), "/fold_change_edgeR.csv"))
write.csv(as.data.frame(selected),
          paste0(paste0(paste0(out1, "/") , gsub("/", "", ct)), "/selected_genes.csv"))