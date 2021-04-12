library(dplyr)
library(tidyr)
library(bnlearn)

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
