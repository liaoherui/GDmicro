library("yaml")
library("tidyverse")
library("SIAMCAT")

args<-commandArgs(T)

#inmatrix <- args[1]
#omatrix <- args[2]
tag<-args[1]


parameters <- yaml.load_file('parameters.yaml')
# extract parameters
norm.method <- parameters$model.building$norm.method
n.p <- list(log.n0=ifelse(tag %in% c('species', 'genus'),
                          as.numeric(parameters$model.building$log.n0),
                                                    as.numeric(parameters$model.building$log.n0.func)),
                                                                sd.min.q=as.numeric(parameters$model.building$sd.min.q),
                                                                            n.p=as.numeric(parameters$model.building$n.p),
                                                                                        norm.margin=as.numeric(parameters$model.building$norm.margin))
num.folds <- as.numeric(parameters$model.building$num.folds)
num.resample <- as.numeric(parameters$model.building$num.resample)
ml.method <- parameters$model.building$ml.method
min.nonzero.coeff <- as.numeric(parameters$model.building$min.nonzero.coeff)
modsel.crit <- list(parameters$model.building$modsel.crit)
perform.fs <- FALSE
param.fs <- list()
if (!tag %in% c('species', 'genus')){
      perform.fs <- TRUE
        param.fs.ss <- 
            list(thres.fs = as.numeric(
                  parameters$model.building$feature.selection$cutoff),
                           method.fs = parameters$model.building$feature.selection$type)
              param.fs.loso <- 
                  list(thres.fs = 3200,
                        method.fs = parameters$model.building$feature.selection$type)
}

fn.feat <- args[2]
#fn.feat
#quit()
feat.all <- as.matrix(read.table(fn.feat, sep='\t', header=TRUE, stringsAsFactors = FALSE,check.names = FALSE, quote=''))
#feat.all
#quit()

meta <- read_tsv(args[3])
stopifnot(all(meta$subjectID %in% colnames(feat.all)))
meta <- data.frame(meta)
rownames(meta) <- meta$subjectID


siamcat <- siamcat(feat=feat.all, meta=meta,label = 'disease', case=args[4])
siamcat <- normalize.features(siamcat, norm.method = norm.method,norm.param = n.p, feature.type = 'original',verbose=3)

fn <-get.norm_feat.matrix(siamcat)
write.table(fn,file=args[5],quote=FALSE, sep='\t', row.names=TRUE, col.names=TRUE)

