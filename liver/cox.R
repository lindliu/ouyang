#install.packages("survival")
#install.packages("survminer")
# install.packages("cmprsk")
# install.packages("rms")
# install.packages("meta")
# install.packages("readxl")

rm(list = ls())

###example from https://search.r-project.org/CRAN/refmans/rms/html/nomogram.html

library("survival")
library("survminer")
library("cmprsk")
library("rms")  ##nomogram
library("meta")
library("readxl")

Dataset <- read_excel('/home/dliu/project/py38/ouyang/data/胰腺癌术后肝转移191例.xlsx')
# getwd()

## change name of columns
colnames(Dataset) <- Dataset[1,]
variables <- c('tas', 'status', 'sex', 'sctr', 'stra', 'age0')
d <- Dataset[2:192, variables]
d <- data.frame(tas = as.numeric(unlist(d['tas'])),
                status = as.numeric(unlist(d['status'])),
                sex = as.numeric(unlist(d['sex'])),
                sctr = as.numeric(unlist(d['sctr'])),
                stra = as.numeric(unlist(d['stra'])),
                age0 = as.numeric(unlist(d['age0'])))
ddist <- datadist(d); options(datadist='ddist')


f <- cph(Surv(tas,status) ~ sex + age0, data=d, surv=T, dist='lognormal')
med  <- Quantile(f)
surv <- Survival(f)  # This would also work if f was from cph
nom <- nomogram(f, fun=list(function(x) surv(3, x),
                            function(x) surv(6, x)),
                funlabel=c("3-Month Survival Probability", 
                           "6-month Survival Probability"))
plot(nom, xfrac=.7)









rm(list = ls())

library("survival")
library("survminer")
library("cmprsk")
library("rms")  ##nomogram
library("meta")
library("readxl")

##https://www.youtube.com/watch?v=X2g93g5nmm0&ab_channel=JoVE%28JournalofVisualizedExperiments%29
Dataset <- read_excel('/home/dliu/project/py38/ouyang/data/胰腺癌术后肝转移191例.xlsx')
# getwd()

## change name of columns
colnames(Dataset) <- Dataset[1,]
Dataset <- Dataset[2:192,]

tas <- as.numeric(unlist(Dataset['tas']))
status <- as.numeric(unlist(Dataset['status']))
sex <- as.numeric(unlist(Dataset['sex']))
sctr <- as.numeric(unlist(Dataset['sctr']))
stra <- as.numeric(unlist(Dataset['stra']))
age65 <- as.numeric(unlist(Dataset['age65']))

# attach(Dataset)
# tas <- as.factor(tas)
# status <- as.factor(status)
# sex <- as.factor(sex)
# sctr <- as.factor(sctr)
# stra <- as.factor(stra)
# age65 <- as.factor(age65)

# variables <- c('tas', 'status', 'sex', 'sctr', 'stra', 'age65')
# data_train <- data[2:192, variables]

f0 <- cph(Surv(tas, status)~sex+sctr+stra+age65, x=T, y=T, surv=T)

surv <- Survival(f0)

nom <- nomogram(f0, fun=list(function(x) surv(12,x), function(x) surv(36,x), function(x) surv(48,x),
                             lp=F,
                             funlabel=c("1-year survival", "3-year survival", "4-year survival"),
                             maxscale=70,
                             fun.at=c(0.1,0.2,0.4,0.6,0.7,0.8,0.9,0.95)))


# 
# #load data
# time <- c(4,6,8,11,15,15,20,20,25,31)
# status <- c(1,1,0,1,1,1,1,0,1,0)
# sex <- c(1,1,1,2,1,1,2,2,2,2)
# df <- data.frame(time, status, sex)
# 
# #table and curves using log
# fit <- survfit(Surv(time, status)~1)
# summary(fit)
# ggsurvplot(fit, data=df)
# 
# #table and curves using log-log
# fit2 <- survfit(Surv(time, status)~1, conf.type="log-log")
# summary(fit2)
# ggsurvplot(fit2, data=df)
# 
# #table and curves using plain
# fit3 <- survfit(Surv(time, status)~1, conf.type="plain")
# summary(fit3)
# ggsurvplot(fit3, data=df)
# 
# #by sex
# fit4 <- survfit(Surv(time, status)~sex, data=df)
# ggsurvplot(fit4, data=df)
# 
# survdiff(Surv(time, status)~sex, data=df)
