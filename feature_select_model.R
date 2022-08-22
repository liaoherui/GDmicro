library("tidyverse")
library("SIAMCAT")
library("yaml")

parameters <- yaml.load_file('parameters.yaml')

args<-commandArgs(T)

tag<-args[3]

norm.method <- parameters$model.building$norm.method
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
             method.fs = parameters$model.building$feature.selection$type,direction='absolute')
        param.fs.loso <- 
            list(thres.fs = 3200,
             method.fs = parameters$model.building$feature.selection$type,direction='absolute')
}


#param.fs.ss
#quit()
# Load datasets
feat<-read.table(args[1],sep="\t",header=TRUE,quote='')
#feat[110:114,1:2]
#quit()
meta<-read.table(args[2],sep="\t",header=TRUE,quote='')
#meta
#quit()

label<-create.label(meta=meta,label="diagnosis",case=1,control=0,p.lab=args[4],n.lab='healthy')

#label$info
#####################################
#fs.cutoff<-c(20,100,250)

#auroc.all<-tibble(cutoff=character(0), type=character(0),study.test=character(0), AUC=double(0))

#####################################
sc.obj.t<-siamcat(feat=feat,label=label,meta=meta)
#sc.obj.t <- filter.features(sc.obj.t, filter.method = 'prevalence',cutoff=0.01)
#sc.obj.t <- create.data.split(sc.obj.t,num.folds = 10, num.resample = 10)

sc.obj.t <- normalize.features(sc.obj.t, norm.method = norm.method,norm.param=n.p,feature.type='original',verbose=3)
sc.obj.t <- create.data.split(sc.obj.t,num.folds = num.folds, num.resample = num.resample)


sc.obj.t.fs <- train.model(sc.obj.t, method = 'lasso', perform.fs = perform.fs,param.fs = param.fs.ss,modsel.crit=modsel.crit,min.nonzero.coeff=min.nonzero.coeff)

models <- models(sc.obj.t.fs)
models[[1]]
temp <- feature_weights(sc.obj.t.fs)
write.csv(temp,paste(args[3],"_feature_weight.csv",sep=""),row.names=TRUE)
sc.obj.t.fs <- make.predictions(sc.obj.t.fs)
sc.obj.t.fs <- evaluate.predictions(sc.obj.t.fs)
model.evaluation.plot(sc.obj.t.fs,fn.plot = paste(args[3],'_evaluation.pdf',sep=""))
   #model.interpretation.plot(sc.obj.t.fs,fn.plot = 'ko_interpretation.pdf',consens.thres = 0.5,limits = c(-3, 3),heatmap.type = 'zscore')
   #model.interpretation.plot(sc.obj.t.fs,fn.plot = 'interpretation.pdf',consens.thres = 0.5,limits = c(-3, 3),heatmap.type = 'zscore',)
