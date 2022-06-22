
library(SingleR)
library(Seurat)
library(SeuratDisk)

cohort_training= snakemake@input[[1]]
out1= snakemake@output[[1]]
# dir.create(out)

ch=readRDS(cohort_training)
ch <- NormalizeData(object = ch, normalization.method = "LogNormalize", scale.factor = 10000)
RowsNA= rownames(ch)[rowSums(is.na(ch@assays$RNA@data)) > 0]
'%!in%' <- function(x,y)!('%in%'(x,y))
RowsKEEP<-rownames(ch)[rownames(ch) %!in% RowsNA]
ch<-subset(ch,features=RowsKEEP)


hpca.se <- HumanPrimaryCellAtlasData()
blueprint.se <- BlueprintEncodeData()
monaco.se <- MonacoImmuneData()
immune.se <- ImmGenData()
# dmap.se <- DatabaseImmuneCellExpressionData()
# hemato.se <- NovershternHematopoieticData()


input <- GetAssayData(object = ch, slot = "data", assay = "RNA")

singleR.list <- list()

# perform singleR classification
singleR.list$hpca <- SingleR(test = input, 
                             method="single",
                             fine.tune=FALSE,
                             ref = hpca.se, 
                             labels = hpca.se$label.main)

singleR.list$blueprint <- SingleR(test = input, 
                                  method="single",
                                  fine.tune=FALSE,
                                  ref = blueprint.se, 
                                  labels = blueprint.se$label.main)

singleR.list$monaco <- SingleR(test = input, 
                               method="single",
                               fine.tune=FALSE,
                               ref = monaco.se, 
                               labels = monaco.se$label.main)

singleR.list$immune <- SingleR(test = input, 
                               method="single",
                               fine.tune=FALSE,
                               ref = immune.se, 
                               labels = immune.se$label.main)



ch$hpca.labels <- singleR.list$hpca$labels
ch$blueprint.labels <- singleR.list$blueprint$labels
ch$monaco.labels <- singleR.list$monaco$labels
ch$immune.labels <- singleR.list$immune$labels

ch$neutrophils= with(ch@meta.data, ifelse(hpca.labels =='T_cells' | blueprint.labels =='CD4+ T-cells'| blueprint.labels =='CD8+ T-cells'|monaco.labels =='T cells'|immune.labels =='T cells', 1,  0)) 


ch = subset(ch, subset= neutrophils==1)

    RowsNA= rownames(ch)[rowSums(is.na(ch@assays$RNA@data)) > 0]
    '%!in%' <- function(x,y)!('%in%'(x,y))
    RowsKEEP<-rownames(ch)[rownames(ch) %!in% RowsNA]
    ch<-subset(ch,features=RowsKEEP)
    ch$identif <- paste(ch@meta.data$sampleID, ch@meta.data$condition, ch@meta.data$who_score,ch@meta.data$batch, sep = "__")
    ch_avr = AverageExpression(ch, group.by = "identif",verbose = FALSE, return.seurat = TRUE, slot = 'data')
    ch_avr@meta.data[c('sampleID','condition','who_score','batch')] = t(data.frame(strsplit(colnames(ch_avr), "__")))
AB.WT.index <- grep(pattern = "^AB-", x = rownames(ch_avr), value = FALSE) # Select row indices and not ERCC names 
ch_avr =ch_avr[-AB.WT.index,]
    SaveH5Seurat(ch_avr, filename = out1, overwrite = TRUE)