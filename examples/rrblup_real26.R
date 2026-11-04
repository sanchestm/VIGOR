# Native rrBLUP REML on the identical ordered masked real26 input and binary K.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=3L) stop('usage: rrblup_real26.R train.csv K.bin prediction.csv')
train <- read.csv(args[1],check.names=FALSE,colClasses=c(IID='character'))
n <- nrow(train)
connection <- file(args[2],'rb')
K <- matrix(readBin(connection,'double',n=n*n,size=8,endian='little'),n,n,byrow=TRUE)
close(connection)
if (any(!is.finite(K)) || max(abs(K-t(K)))>1e-8) stop('invalid K')
result <- matrix(NA_real_,n,ncol(train)-1)
for (j in seq_len(ncol(result))) {
    y <- train[[j+1]]; ok <- is.finite(y)
    fit <- rrBLUP::mixed.solve(y=y[ok],K=K[ok,ok,drop=FALSE],method='REML',SE=FALSE)
    lambda <- as.numeric(fit$Ve/fit$Vu); mu <- as.numeric(fit$beta[1])
    result[,j] <- mu+K[,ok,drop=FALSE] %*% solve(K[ok,ok,drop=FALSE]+diag(lambda,sum(ok)),y[ok]-mu)
}
colnames(result) <- names(train)[-1]
write.csv(data.frame(IID=train$IID,result,check.names=FALSE),args[3],row.names=FALSE)
