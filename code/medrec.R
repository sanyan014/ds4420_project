library(tidyverse)
library(recommenderlab)

df <- read.csv("final_flat_dataset.csv", stringsAsFactors = FALSE)

df_expanded <- df %>%
  filter(!is.na(drug), drug != "", !is.na(diagnosis)) %>%
  distinct(diagnosis, drug)

binary_df <- df_expanded %>%
  mutate(value = 1) %>%
  pivot_wider(names_from = drug, values_from = value, values_fill = 0)

diagnosis_labels <- binary_df$diagnosis

diagnosis_drug_matrix <- binary_df %>%
  select(-diagnosis) %>%
  as.matrix()

rrm <- as(diagnosis_drug_matrix, "binaryRatingMatrix")
rownames(rrm) <- diagnosis_labels

set.seed(123)
eval_scheme <- evaluationScheme(rrm, method = "split", train = 0.8, given = 3)
results <- evaluate(eval_scheme, method = "IBCF", type = "topNList", n = c(1, 3, 5, 10))
conf_matrix <- getConfusionMatrix(results)
df_metrics <- map_dfr(conf_matrix, ~ as_tibble(.x), .id = "n") %>%
  mutate(n = as.numeric(n)) %>%
  mutate(
    precision = ifelse((TP + FP) == 0, 0, TP / (TP + FP)),
    recall = ifelse((TP + FN) == 0, 0, TP / (TP + FN)),
    F1 = ifelse((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall)),
    FPR = ifelse((FP + TN) == 0, 0, FP / (FP + TN)),
    TPR = recall) %>%
  select(n, TP, FP, FN, TN, precision, recall, F1, FPR, TPR)

write.csv(df_metrics, "ibcf_summary_metrics.csv", row.names = FALSE)

print(df_metrics)
















