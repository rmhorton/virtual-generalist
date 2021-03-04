---
title: "Cancer example with bnlearn"
author: "Bob Horton"
date: "3/4/2021"
output: 
  html_document:
    keep_md: true
---



Note: This document makes use of packages that must be installed from BioConductor, including some dependencies that must be compiled. Assuming you already have `RTools`, you can install RGraphviz and dependencies as follows:
```
if (!requireNamespace("BiocManager", quietly = TRUE))
install.packages("BiocManager")
BiocManager::install()
BiocManager::install(c("graph", "Rgraphviz"))

```

```r
library(dplyr)
library(tidyr)
library(bnlearn)
```

This examples follows Figure 2.1 in  [_Bayesian Artificial Intelligence_](https://bayesian-intelligence.com/publications/bai/), chapter 2.



Let's use the probabilities given in the figure to generate some simulated data, then see if `bnlearn` can determine the structure of the Bayes Net from the simulated data:


```r
set.seed(42)

sim_cancer <- function(N){
  
  cancer_prob <- function(P, S){
    case_when(
      P=='H' & S=='T'  ~ 0.05,
      P=='H' & S=='F'  ~ 0.02,
      P=='L' & S=='T'  ~ 0.03,
      P=='L' & S=='F'  ~ 0.001
    )
  }
  
  xray_prob <- function(C) ifelse(C=='T', 0.9, 0.2)
  
  dyspnoea_prob = function(C) ifelse(C=='T', 0.65, 0.30)
  
  data.frame(
    Pollution = ifelse(runif(N) < 0.9, 'L', 'H'),
    Smoking = ifelse(runif(N) < 0.3, 'T', 'F'),
    LeftHanded = ifelse(runif(N) < 0.1, 'T', 'F'),
    RedHead = ifelse(runif(N) < 0.02, 'T', 'F')
  ) %>% mutate(
    Cancer = ifelse(cancer_prob(Pollution, Smoking) > runif(N), 'T', 'F'),
    Xray = ifelse(xray_prob(Cancer) > runif(N), 'T', 'F'),
    Dyspnoea = ifelse(dyspnoea_prob(Cancer) > runif(N), 'T', 'F')
  ) %>% lapply(factor) %>% as.data.frame
}

simdata <- sim_cancer(20000)

head(simdata)
```

```
##   Pollution Smoking LeftHanded RedHead Cancer Xray Dyspnoea
## 1         H       F          F       F      F    F        F
## 2         H       F          F       F      F    F        T
## 3         L       F          F       F      F    T        F
## 4         L       F          F       F      F    F        F
## 5         L       F          F       F      F    F        T
## 6         L       F          T       F      F    F        F
```

Learn the structure:


```r
cancer_structure <- simdata %>% fast.iamb # this os one of veveral structure-learning algorithmsgs

modelstring(cancer_structure)
```

```
## [1] "[Pollution][Smoking][LeftHanded][RedHead][Cancer|Pollution:Smoking][Xray|Cancer][Dyspnoea|Cancer]"
```

```r
cancer_structure %>% graphviz.plot
```

![](bnlearn_example_files/figure-html/plot_gs-1.png)<!-- -->

```r
# Try different numbers of training examples; more data usually works better.
# sim_cancer(10000) %>% fast.iamb %>% plot
```


## Fitting the network parameters


```r
cancer_fit <- bn.fit(cancer_structure, data=simdata)

cancer_fit
```

```
## 
##   Bayesian network parameters
## 
##   Parameters of node Pollution (multinomial distribution)
## 
## Conditional probability table:
##        H       L 
## 0.10015 0.89985 
## 
##   Parameters of node Smoking (multinomial distribution)
## 
## Conditional probability table:
##      F     T 
## 0.701 0.299 
## 
##   Parameters of node LeftHanded (multinomial distribution)
## 
## Conditional probability table:
##       F      T 
## 0.9012 0.0988 
## 
##   Parameters of node RedHead (multinomial distribution)
## 
## Conditional probability table:
##       F      T 
## 0.9781 0.0219 
## 
##   Parameters of node Cancer (multinomial distribution)
## 
## Conditional probability table:
##  
## , , Smoking = F
## 
##       Pollution
## Cancer            H            L
##      F 0.9820531228 0.9992872416
##      T 0.0179468772 0.0007127584
## 
## , , Smoking = T
## 
##       Pollution
## Cancer            H            L
##      F 0.9590163934 0.9716945996
##      T 0.0409836066 0.0283054004
## 
## 
##   Parameters of node Xray (multinomial distribution)
## 
## Conditional probability table:
##  
##     Cancer
## Xray          F          T
##    F 0.80135429 0.08530806
##    T 0.19864571 0.91469194
## 
##   Parameters of node Dyspnoea (multinomial distribution)
## 
## Conditional probability table:
##  
##         Cancer
## Dyspnoea         F         T
##        F 0.7036232 0.3554502
##        T 0.2963768 0.6445498
```

## Making predictions


```r
cpquery(cancer_fit, (Cancer=='T'), (Dyspnoea=='T' & Xray=='T' & Pollution=='H' & Smoking=='T'))
```

```
## [1] 0.4166667
```

```r
cpquery(cancer_fit, (Cancer=='T'), (Dyspnoea=='F' & Xray=='F' & Pollution=='H' & Smoking=='T'))
```

```
## [1] 0
```

```r
cpquery(cancer_fit, (Cancer=='T'), (Dyspnoea=='T' & Xray=='T'))
```

```
## [1] 0.04938272
```

```r
cpquery(cancer_fit, (Cancer=='T'), (Dyspnoea=='F' & Xray=='F'))
```

```
## [1] 0.0007102273
```

```r
cpquery(cancer_fit, (Smoking=='T'), (Dyspnoea=='F' & Xray=='F'))
```

```
## [1] 0.2907904
```
# Extracting Conditional Probability Tables


```r
reformat_CPT <- function(bnet, node_name){
  p_table <- bnet[[node_name]]$prob
  if ( (p_table %>% dim %>% length) == 1){
    prob_df <- p_table %>% as.matrix %>% t %>% as.data.frame
    names(prob_df) <- paste(node_name, names(prob_df), sep='_')
  } else {
    prob_df <- p_table %>% 
      ftable %>% 
      as.data.frame %>% 
      pivot_wider(names_from=node_name, values_from=Freq)
    output_cols <- c(ncol(prob_df) - 1, ncol(prob_df))
    names(prob_df)[output_cols] <- paste(node_name, names(prob_df)[output_cols], sep='_')
  }
  return(prob_df)
}


cancer_fit %>% reformat_CPT('Pollution')
```

```
##   Pollution_H Pollution_L
## 1     0.10015     0.89985
```

```r
cancer_fit %>% reformat_CPT('Smoking')
```

```
##   Smoking_F Smoking_T
## 1     0.701     0.299
```

```r
cancer_fit %>% reformat_CPT('Cancer')
```

```
## # A tibble: 4 x 4
##   Pollution Smoking Cancer_F Cancer_T
##   <fct>     <fct>      <dbl>    <dbl>
## 1 H         F          0.982 0.0179  
## 2 L         F          0.999 0.000713
## 3 H         T          0.959 0.0410  
## 4 L         T          0.972 0.0283
```

```r
cancer_fit %>% reformat_CPT('Xray')
```

```
## # A tibble: 2 x 3
##   Cancer Xray_F Xray_T
##   <fct>   <dbl>  <dbl>
## 1 F      0.801   0.199
## 2 T      0.0853  0.915
```

```r
cancer_fit %>% reformat_CPT('Dyspnoea')
```

```
## # A tibble: 2 x 3
##   Cancer Dyspnoea_F Dyspnoea_T
##   <fct>       <dbl>      <dbl>
## 1 F           0.704      0.296
## 2 T           0.355      0.645
```
