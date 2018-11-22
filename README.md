# test
beginner's quest.


# Libraries
library(data.table)
library(dplyr)
library(h2o, lib.loc = '/data/user/nh81470e/.conda')
#library(h2o)
randport = round(runif(1,1000,5000))
h2o.init(port=randport, max_mem_size="50g", nthreads = 72)
library(readxl)

# Functions
utils_folder <- paste0('/data/user/', Sys.info()[["user"]], '/pl_utils/src/r/')
function_list <- c('error_tabs_pretty.R', 'weighted_gini.R', 'normalized_weighted_gini.R',
                   'weighted_rmse.R', 'plot_a_vs_e2_loss_ratio_dt.R', 'sorter_function.R', 'pretty_table.R',
                   'get_pdp.R', 'weighted_mae.R', 'lr_plm_chart.R', 'lr_percentile_chart.R',
                   'lr_percentile_comparison_chart.R', 'lr_spread_chart.R',
                   'plot_a_vs_e2_loss_ratio_dt_no_old.R')
for(func in function_list) source(paste0(utils_folder, func))

h2o.shutdown()

# Read in Merkle rules
merkle_rules <- as.data.table(read_excel("/data/lake/plds/datahub/pl_plm_model/data_rules/Proposed_PLM_Variables.xlsx"))
merkle_ind_vars <- as.data.table(read_excel("/data/lake/plds/datahub/pl_plm_model/data_rules/merkle_validation_rules_ind_vars.xlsx"))

# Define original PLM and all Merkle model variables
h2o_schema <- ifelse(substr(merkle_rules$`Model Variable Name`,1, 3) == 'cv_', 'real', 'enum')
names(h2o_schema) <- merkle_rules$`Model Variable Name`
final_h2o_schema <- c(h2o_schema)

# Read in data, using Merkle rules for data formats
model_data <- h2o.importFile('/data/lake/plds/datahub/pl_plm_model/Merkle_NB_Model_Data_Cleaned.csv',
                            col.types = final_h2o_schema, header = T)

non_model_vars <- c('POLICY_NUMBER', 'PLCY_STATUS_CD', 'BUSINESS_SEGMENT',
    'policy_hig_year_count', 'RATE_PLAN_CD', 'RISK_STATE_CD', 'PLM_decile',
    'PLM_Flag', 'PLM_score', 'inforce_ind', 'data_partition', 'POLICY_YEAR',
    'time_partition', 'time_partition_24mo', 'time_partition_30mo',
    'time_partition_36mo', 'time_partition_40mo', 'TOTAL_EP', 'TOTAL_EPAPR',
    'TOTAL_EPAPR_xBI', 'EARNED_PD_EXPOSURE_YR_CNT', 'TOT_CAP_ULT_LOSS_AMT',
    'TOT_ULT_LOSS_AMT', 'TOT_XCAT_CAP_ULT_LOSS_AMT', 'TOT_XCAT_ULT_LOSS_AMT',
    'Total_Pred_Loss_or2_4_fix', 'TOT_XCAT_XBI_CAP_ULT_LOSS_AMT',
    'TOT_CAP_XCAT_LOSS_RATIO', 'TOT_XCAT_XBI_LOSS_RATIO', 'TOT_PRED_LOSS_RATIO',
    'TOTAL_EARNED_PREM', 'ADJ_XCAT_CAP_LOSS_RATIO', 'ADJ_XCAT_CAP_LOSS',
    'ADJ_XCAT_XBI_LOSS_RATIO', 'ADJ_XCAT_XBI_LOSS', 'ADJ_PRED_LOSS_RATIO',
    'ADJ_PRED_LOSS')


# Define target and weight variables
model_data$target <- model_data$ADJ_XCAT_CAP_LOSS_RATIO
model_data$weight <- model_data$TOTAL_EPAPR

# Different splits for train/test
train_split <- model_data[model_data$data_partition == 'Train' & model_data$POLICY_YEAR >= 2015,]
test_split <- model_data[model_data$data_partition == 'Test' & model_data$POLICY_YEAR >= 2015,]
holdout_split <- model_data[model_data$data_partition == 'Holdout' & model_data$POLICY_YEAR >= 2015,]

## Type III

## Unregularized GLM

var_list <- merkle_rules$`Model Variable Name`

# Fit regular GLM, using non-zero coefficient variables from Lasso GLM
fit_tm <- proc.time()
output_metrics <- data.table()
unreg_tweedie_model <- h2o.glm(x = var_list, y = 'target',
                               training_frame = train_split,
                               seed = 100,
                               nfolds = 5, weights_column = 'weight', family = 'tweedie',
                               tweedie_variance_power = 1.7,
                               tweedie_link_power = 0,
                               #alpha = 1,
                               lambda = 0,
        #                      lambda_search = TRUE,
                               standardize = FALSE)

train_split$predictions <- h2o.predict(unreg_tweedie_model, train_split)$predict
test_split$predictions <- h2o.predict(unreg_tweedie_model, test_split)$predict
    
train_slim <- as.data.table(train_split[,c('target', 'weight', 'predictions')])
test_slim <- as.data.table(test_split[,c('target', 'weight', 'predictions')])
    
train_gini <- normalized_weighted_gini(solution = train_slim$target,
                                       weights = train_slim$weight,
                                       submission = train_slim$predictions)
test_gini <- normalized_weighted_gini(solution = test_slim$target,
                                       weights = test_slim$weight,
                                       submission = test_slim$predictions)
fit_tm_final <- (proc.time() - fit_tm)

type_III_df <- data.frame(variable='base_model', 
                          train_gini=train_gini,
                          test_gini=test_gini,
                          gini_train_chg = 0,
                          gini_test_chg = 0
                         )

type_III_df

# Loop through each variable, performing Type III gini comparison on test data
for(var in var_list){
    # keep track of time per loop
    iteration_tm <- proc.time()
    # exclude all features of current core variable
    included_vars <- var_list[var_list != var]
    type_III_vars <- unique(included_vars)
    # print current variable to console
    print(var)
    print('Excluded Features')
    print(setdiff(unique(var_list), type_III_vars))
    #print(included_vars)
    #print(type_III_vars)   
      
unreg_tweedie_model <- h2o.glm(x = type_III_vars, y = 'target',
                                training_frame = train_split,
                                seed = 100,
                                nfolds = 5, weights_column = 'weight', family = 'tweedie',
                                tweedie_variance_power = 1.7,
                                tweedie_link_power = 0,
                                lambda = 0,
                                standardize = TRUE)
    
train_split$predictions <- h2o.predict(unreg_tweedie_model, train_split)$predict
test_split$predictions <- h2o.predict(unreg_tweedie_model, test_split)$predict
    
train_slim <- as.data.table(train_split[,c('target', 'weight', 'predictions')])
test_slim <- as.data.table(test_split[,c('target', 'weight', 'predictions')])
    
train_gini <- normalized_weighted_gini(solution = train_slim$target,
                                       weights = train_slim$weight,
                                       submission = train_slim$predictions)
test_gini <- normalized_weighted_gini(solution = test_slim$target,
                                       weights = test_slim$weight,
                                       submission = test_slim$predictions)
   
   # Save results into temporary data frame
    temp_df <- data.frame(variable=var,
                          train_gini=train_gini,
                          test_gini=test_gini,
                          gini_train_chg = (type_III_df[1, 'train_gini'] - train_gini),
                          gini_test_chg = (type_III_df[1, 'test_gini'] - test_gini)
                         )
        
    #apphend new results to dataframe
    type_III_df <- rbind(type_III_df, temp_df)
   # print iteration time
   iteration_tm_final <- (proc.time() - iteration_tm)
     print(iteration_tm_final)
}
# Order output dataframe by change in test gini (but keep the base model at the top)
type_III_df[2:nrow(type_III_df),] <- type_III_df[2:nrow(type_III_df),][order(type_III_df[2:nrow(type_III_df), 'gini_test_chg'], decreasing = T),]

# Select variables which improve the test gini
final_vars <- as.character(type_III_df$variable[type_III_df$gini_test_chg > 0])

# Refit using Type III qualified variables
unreg_tweedie_model <- h2o.glm(x = final_vars, y = 'target', 
                               training_frame = train_split,
                         #validation_frame = test_split,
                         seed = 100,
                         nfolds = 5, weights_column = 'weight', family = 'tweedie',
                         tweedie_variance_power = 1.7,
                         tweedie_link_power = 0,
                         #alpha = 1,
                         lambda = 0,
#                         lambda_search = TRUE,
                         standardize = TRUE,
                         compute_p_values = TRUE,
                              remove_collinear_columns = TRUE)

train_split$predictions <- h2o.predict(unreg_tweedie_model, train_split)$predict
test_split$predictions <- h2o.predict(unreg_tweedie_model, test_split)$predict
    
train_slim <- as.data.table(train_split[,c('target', 'weight', 'predictions')])
test_slim <- as.data.table(test_split[,c('target', 'weight', 'predictions')])
    
train_gini <- normalized_weighted_gini(solution = train_slim$target,
                                       weights = train_slim$weight,
                                       submission = train_slim$predictions)
test_gini <- normalized_weighted_gini(solution = test_slim$target,
                                       weights = test_slim$weight,
                                       submission = test_slim$predictions)

print(train_gini)
print(test_gini)
ls()

# only select variables which are non-zero coefficients (dropped by h2o's correlation check)
nrow(unreg_tweedie_model@model$coefficients_table)
typeIII_variable_tbl <- unreg_tweedie_model@model$coefficients_table
typeIII_variable_tbl
coefficients <- unreg_tweedie_model@model$coefficients_table$coefficients
variables <- unreg_tweedie_model@model$coefficients_table$names[coefficients != 0]

typeIII_variable_tbl

# Pull out main effect variables (unique variables)
library(stringr)
main_effects <- vector()
for(var in variables){
    if(var == 'Intercept'){next}
    if(str_detect(var, '\\.')){
        var <- substr(var, 0, str_locate(var, '\\.') - 1)
    }
    print(var)
    main_effects <- c(main_effects, var)
}
main_effects <- unique(main_effects)

main_effects

## Final Variable Model

# select final variables
# only changed 'cf_Merkle_Marketing_Index_(2_Qtrs_Ago)_ind' -> 'cf_Merkle_Marketing_Index_(4_Qtrs_Ago)_ind'

final_variables <- c('cf_Property_Indicator', 'cf_Recording_Date_(Ranges)', 'cf_Gender',
    'cf_Collectors', 'cf_Grandparent_(Field_Type)', 'cf_Cat_Owner_(Field_Type)',
    'cf_Home_Decorating_(Field_Type)', 'cf_Homeowner', 'cf_Dog_Owner',
    'cf_Gardening', 'cf_Video_Games', 'cf_Veterans', 'cf_General_Hobbies',
    'cf_Food_And_Beverage', 'cf_Finance_Loan_Store', 'cf_Do_it_yourselfer',
    'cf_Special_Foods_Buyer', 'cf_Credit_Card_Premium', 'cf_LD_BlueChip',
    'cf_Merkle_Marketing_Index_(4_Qtrs_Ago)_ind',
    'cf_Equity_in_Home_(Actual)_ind', 'cf_Home_Improvement_Value_(Actual)_ind',
    'cv_Merkle_Adjusted_Wealth_Rating', 'cv_Age_Range_In_Household',
    'cv_Merkle_Marketing_Index_(4_Qtrs_Ago)', 'cv_Equity_in_Home_(Actual)',
    'cv_Home_Improvement_Value_(Actual)', 'cv_Prob_Auto_Loan_for_New_Car',
    'cv_Prob_Purchasing_New_Laptop/Desktop', 'cv_Prob_Prescriptions_Filled',
    'cv_Prob_Catalog_Shopping_By_Phone', 'cv_Prob_Green_Products_Buyer',
    'cv_Individual_Count', 'cv_Census_(CAPE)_Persons_per_Household',
    'cv_Household_Income', 'cv_Occupation_household',
    'cv_Number_of_Total_Rooms', 'cv_Number_of_Children', 'cv_New_Credit_Range')

final_model <- h2o.glm(x = final_variables, y = 'target',
                               training_frame = train_split,
                               seed = 100,
                               nfolds = 5, weights_column = 'weight', family = 'tweedie',
                               tweedie_variance_power = 1.7,
                               tweedie_link_power = 0,
                               #alpha = 1,
                               lambda = 0,
        #                      lambda_search = TRUE,
                               standardize = TRUE,
                               compute_p_values = TRUE,
                              remove_collinear_columns = TRUE)

train_split$predictions <- h2o.predict(unreg_tweedie_model, train_split)$predict
test_split$predictions <- h2o.predict(unreg_tweedie_model, test_split)$predict
    
train_slim <- as.data.table(train_split[,c('target', 'weight', 'predictions')])
test_slim <- as.data.table(test_split[,c('target', 'weight', 'predictions')])
    
train_gini <- normalized_weighted_gini(solution = train_slim$target,
                                       weights = train_slim$weight,
                                       submission = train_slim$predictions)
test_gini <- normalized_weighted_gini(solution = test_slim$target,
                                       weights = test_slim$weight,
                                       submission = test_slim$predictions)

train_gini
test_gini

options(repr.matrix.max.rows = 500)
options(repr.matrix.max.cols = 200)

final_model@model$coefficients_table

## Validation

model_data$predictions <- h2o.predict(unreg_tweedie_model, model_data)$predict
train_slim <- as.data.table(train_split[,c('target', 'weight', 'predictions', 'TOT_XCAT_CAP_ULT_LOSS_AMT', 'PLM_decile', 'POLICY_YEAR', 'PLM_score', 'time_partition', 'TOTAL_EARNED_PREM', 'TOT_XCAT_ULT_LOSS_AMT')])
test_slim <- as.data.table(test_split[,c('target', 'weight', 'predictions', 'TOT_XCAT_CAP_ULT_LOSS_AMT', 'PLM_decile', 'POLICY_YEAR', 'PLM_score', 'time_partition', 'TOTAL_EARNED_PREM', 'TOT_XCAT_ULT_LOSS_AMT')])
model_data_slim <- as.data.table(model_data[,c('target', 'weight', 'predictions', 'TOT_XCAT_CAP_ULT_LOSS_AMT', 'PLM_decile', 'POLICY_YEAR', 'PLM_score', 'time_partition', 'TOTAL_EARNED_PREM', 'TOT_XCAT_ULT_LOSS_AMT')])

model_data_dt$predictions <- as.data.table(h2o.predict(tweedie_model, model_data))

## Pull out current PLM decile splits, weighted by premium

total_premium <- sum(model_data_slim$weight[!model_data_slim$PLM_decile %in% c(90, NA)])
plm_decile_splits <- model_data_slim %>%
                        filter(!PLM_decile %in% c(90, NA)) %>%
                        group_by(PLM_decile) %>%
                        summarise(prem = sum(weight)/total_premium) %>%
                        mutate(cumulative_prem = cumsum(prem))


train_slim$PLM_decile[is.na(train_slim$PLM_decile)] <- 90
test_slim$PLM_decile[is.na(test_slim$PLM_decile)] <- 90

## Loss ratio spread charts (attractive vs. unnatractive average loss ratio)

lr_spread_chart(model_data_slim[POLICY_YEAR >= 2015,], 'target', 'predictions', 'weight', 'TOT_XCAT_CAP_ULT_LOSS_AMT',
                              nbuckets = 100, title1 = 'Unattractive', 'Attractive')

lr_spread_chart(train_slim, 'target', 'predictions', 'weight', 'TOT_XCAT_CAP_ULT_LOSS_AMT',
                              nbuckets = 100, title1 = 'Unattractive', 'Attractive')

lr_spread_chart(test_slim, 'target', 'predictions', 'weight', 'TOT_XCAT_ULT_LOSS_AMT',
                              nbuckets = 100, title1 = 'Unattractive', 'Attractive')

lr_spread_chart(test_slim[time_partition == 'Test',], 'target', 'predictions', 'weight', 'TOT_XCAT_CAP_ULT_LOSS_AMT',
                              nbuckets = 100, title1 = 'Unattractive', title2 = 'Attractive')

## Train vs. Test Loss ratio comparisons by decile

lr_percentile_comparison_chart(model_data_slim, test_slim, 'target', 'predictions', 'weight', 'TOT_XCAT_CAP_ULT_LOSS_AMT',
                              nbuckets = 10,
                              custom_cuts = FALSE, cut_tbl = plm_decile_splits, cut_tbl_bin = 'cumulative_prem',
                              cut_tbl_label = 'PLM_decile', title1 = 'Full Data', title2 = 'Test')

lr_percentile_comparison_chart(train_slim, test_slim, 'target', 'predictions', 'weight', 'TOT_XCAT_CAP_ULT_LOSS_AMT',
                               nbuckets = 10,
                              custom_cuts = FALSE, cut_tbl = plm_decile_splits, cut_tbl_bin = 'cumulative_prem',
                              cut_tbl_label = 'PLM_decile', title1 = 'Train', title2 = 'Test')

## Train

lr_plm_chart(train_slim, 'target', 'predictions', 'weight', 'TOT_XCAT_CAP_ULT_LOSS_AMT', pre_sort = TRUE,
                   pre_sort_col = 'PLM_decile')

## Test

lr_plm_chart(test_slim, 'target', 'predictions', 'weight', 'TOT_XCAT_CAP_ULT_LOSS_AMT', pre_sort = TRUE,
                   pre_sort_col = 'PLM_decile')

## AvE

library(stringr)
main_effects <- vector()
# model_data_dt_2015 <- model_data_dt[POLICY_YEAR >= 2015,]
# pdf('plm_full_data_ave.pdf')
for(var in coef_tbl$names){
    if(var == 'Intercept'){next}
    if(str_detect(var, '\\.')){
        var <- substr(var, 0, str_locate(var, '\\.') - 1)
    }
    print(var)
#     plot_a_vs_e2_no_old(model_data_dt_2015, 'ADJ_XCAT_CAP_LOSS_RATIO',
#                         'predictions', 'TOTAL_EPAPR', var,
#                         act_label = 'Actual Loss Ratio',
#                         pred_label = 'Predicted Loss Ratio')
    main_effects <- c(main_effects, var)
}
# dev.off()

unique_mf <- unique(main_effects)

unique_mf

h2o.predict(tweedie_model, train_split)$predict
