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




weighted_gini <- function(solution, weights, submission){
  df <- data.frame(solution = solution, weights = weights, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  df$random <- cumsum((df$weights/sum(df$weights)))
  totalPositive <- sum(df$solution * df$weights)
  df$cumPosFound <- cumsum(df$solution * df$weights)
  df$Lorentz <- df$cumPosFound / totalPositive
  n <- nrow(df)
  gini <- sum(df$Lorentz[-1]*df$random[-n]) - sum(df$Lorentz[-n]*df$random[-1])
return(gini)
}

normalized_weighted_gini <- function(solution, weights, submission) {
	weighted_gini(solution, weights, submission) / weighted_gini(solution, weights, solution)
}


lr_percentile_chart <- function(data, target, predictions, weight, actual_loss,
		                            pre_sort = FALSE, pre_sort_col = NULL, custom_cuts = FALSE,
		                            cut_tbl, cut_tbl_bin, cut_tbl_label){

require(plotly)
require(rlang)
    
weight_sum <- sum(data[[weight]])+1

if(pre_sort == TRUE){
    df_sort <- data
    df_sort$loss_ratio_band <- as.factor(data[[pre_sort_col]])
    
} else if(custom_cuts == TRUE){
    df_sort <- data %>%
                   arrange_at(.vars = c(predictions)) %>%
                   mutate_at(.vars = c(weight),
                             .funs = funs(cumulative_wgt = cumsum(.)/weight_sum)
                             ) %>%
                   mutate_at(.vars = c(predictions),
                             .funs = funs(loss_ratio_band = cut(cumulative_wgt,
                                                                breaks = c(0, cut_tbl[[cut_tbl_bin]]),
                                                                labels = cut_tbl[[cut_tbl_label]])
                                         )
                            )
} else{
        df_sort <- data %>%
                   arrange_at(.vars = c(predictions)) %>%
                   mutate_at(.vars = c(weight),
                             .funs = funs(loss_ratio_band = floor(cumsum(.) / weight_sum * 10)/10)
                            )
} 
    
df_temp <- df_sort %>%
            group_by(loss_ratio_band) %>%
            summarise(model_loss_ratio = sum(!!sym(predictions)*!!sym(weight))/sum(!!sym(weight)),
                      actual_loss_ratio = sum(!!sym(actual_loss))/sum(!!sym(weight))) %>%
            ungroup()
    
chart <- plot_ly() %>%
            add_trace(df_temp, x = df_temp$loss_ratio_band, y= df_temp$actual_loss_ratio, type = 'bar',
                      name = 'Actuals', text = round(df_temp$actual_loss_ratio, 4), textposition = 'auto') %>%
            add_trace(df_temp, x = df_temp$loss_ratio_band, y= df_temp$model_loss_ratio, type = 'bar',
                      name = 'Predictions', text = round(df_temp$model_loss_ratio, 4), textposition = 'auto') %>%
            layout(title = 'Actual vs. Predicted Loss Ratio',
                     barmode = 'group',
                     xaxis = list(title = "Percentile"),
                     yaxis = list(title = "Loss Ratio"))
return(chart)
}



lr_plm_chart <- function(data, target, predictions, weight, actual_loss,
                                pre_sort = FALSE, pre_sort_col = NULL){

require(plotly)
require(rlang)
    
prediction_sum <- sum(data[[predictions]])+1

if(pre_sort == FALSE){
    df_sort <- data %>%
                   arrange_at(.vars = c(predictions)) %>%
                   mutate_at(.vars = c(predictions),
                             .funs = funs(loss_ratio_band = floor(cumsum(.) / prediction_sum * 10)/10))
} else{
    df_sort <- data
    df_sort$loss_ratio_band <- as.factor(data[[pre_sort_col]])
}    
df_temp <- df_sort %>%
            group_by(loss_ratio_band) %>%
            summarise(model_loss_ratio = sum(!!sym(predictions)*!!sym(weight))/sum(!!sym(weight)),
                      actual_loss_ratio = sum(!!sym(actual_loss))/sum(!!sym(weight)))
    
chart <- plot_ly() %>%
            add_trace(df_temp, x = df_temp$loss_ratio_band, y= df_temp$actual_loss_ratio, type = 'bar',
                      name = 'Actuals', text = round(df_temp$actual_loss_ratio, 4), textposition = 'auto') %>%
           layout(title = 'Actual vs. Predicted Loss Ratio',
                     barmode = 'group',
                     xaxis = list(title = "Percentile"),
                     yaxis = list(title = "Loss Ratio"))
return(chart)
}




library(plotly)
library(rlang)
      
lr_spread_chart <- function(data, target, predictions, weight, actual_loss, nbuckets = 10,
                                       title1, title2){
    
weight_sum <- sum(data[[weight]])+1
loss_sum <- sum(data[[actual_loss]])

df_sort <- data %>%
             arrange_at(.vars = c(predictions)) %>%
             mutate_at(.vars = c(weight),
                       .funs = funs(loss_ratio_band = floor(cumsum(.) / weight_sum * nbuckets)/nbuckets)
                       )

df_temp <- df_sort %>%
            group_by(loss_ratio_band) %>%
            summarise(actual_loss = sum(!!sym(actual_loss)),
                      total_prem = sum(!!sym(weight)),
                      model_loss_ratio = sum(!!sym(predictions)*!!sym(weight))/sum(!!sym(weight)),
                      actual_loss_ratio = sum(!!sym(actual_loss))/sum(!!sym(weight))) %>%
            ungroup() %>%
            mutate(decile = 1:n(),
                   attractive_loss = cumsum(actual_loss),
                   attractive_prem = cumsum(total_prem)) %>%
            mutate(unattractive_loss = loss_sum - attractive_loss,
                   unattractive_prem = weight_sum - attractive_prem) %>%
            mutate(attractive_lr = attractive_loss/attractive_prem,
                   unattractive_lr = unattractive_loss/unattractive_prem) %>%
            filter(decile < max(decile))
    
#return(df_temp)  
chart <- plot_ly() %>%
            add_trace(df_temp, x = df_temp$decile, y= df_temp$unattractive_lr, type = 'scatter', mode = 'lines',
                      name = title1, text = round(df_temp$unattractive_lr, 4), textposition = 'auto') %>%
            add_trace(df_temp, x = df_temp$decile, y= df_temp$attractive_lr, type = 'scatter', mode = 'lines',
                      name = title2, text = round(df_temp$attractive_lr, 4), textposition = 'auto') %>%
            layout(title = 'Attractive/Unnatractive Loss Ratio Spread',
                     #barmode = 'group',
                     xaxis = list(title = "Decile Split"),
                     yaxis = list(title = "Loss Ratio"))
return(chart)
}



weighted_rmse <- function(actual, predicted, weight){
    "Calculates the RMSE value between actual and predicted with weights
    
    Arguments:
        actual: vector of actual values
        predicted: vecotr of predicted values
        weight: vector of weights
    
    Written by ???
    Documented by Tim Ivancic
    "
    return(sqrt(sum((predicted-actual)^2*weight/sum(weight), na.rm=TRUE)))
}







predictor_compare <- function(df1, df2, chi_vars, ks_vars, alpha, title1, title2){
	
	suppressWarnings({
	# Dependencies
	require(dplyr)
	require(plotly)
	
	# Track function run time
	run_tm <- proc.time()

	# K-S Test
	ks_tbl <- data.frame()
	for(var in ks_vars){
    ks_temp <- ks.test(x = df1[[var]], y = df2[[var]])
    temp_df <- cbind(var, ks_temp$p.value)
    ks_tbl <- rbind(ks_tbl, temp_df, stringsAsFactors = F)
   }
	names(ks_tbl) <- c('variable', 'p_value')

	# Chi-Square Test
	chi_tbl <- data.frame()
	for(var in chi_vars){
			all_levels <- union(levels(factor(df1[[var]])), levels(factor(df2[[var]])))
	    probs_df1 <- table(factor(df1[[var]], levels = all_levels))/length(df1[[var]])
	    tbl_df2 <- table(factor(df2[[var]], levels = all_levels))
	    chi_results <- chisq.test(x = tbl_df2, p = probs_df1)#, simulate.p.value=TRUE)
	    temp_df <- cbind(var, chi_results$p.value)
	    chi_tbl <- rbind(chi_tbl, temp_df, stringsAsFactors = F)
	   }
	})
	names(chi_tbl) <- c('variable', 'p_value')
	chi_tbl$p_value <- as.numeric(chi_tbl$p_value)
	
	# Significance Filter (alpha)
	ks_tbl <- dplyr::filter(ks_tbl, p_value < alpha)
	chi_tbl <- dplyr::filter(chi_tbl, p_value < alpha)
	
	# Chi-square plots
	chi_plots <- list()
	for(var in chi_tbl$variable){
		df1_temp <- df1 %>%
	    					group_by_at(.vars =  var) %>%
	    					select(var) %>%
	    					summarise('counts' = n()/nrow(.))
	    
		df2_temp <- df2 %>%
	    					group_by_at(.vars =  var) %>%
	    					select(var) %>%
	    					summarise('counts' = n()/nrow(.))
	    
		chi_plots[[var]] <- plot_ly() %>%
			    						    add_trace(df1, x = df1_temp[[var]], y= df1_temp[['counts']], type = 'bar', name = title1,
											              text = round(df1_temp[['counts']], 2), textposition = 'auto') %>%
											    add_trace(df2, x = df2_temp[[var]], y= df2_temp[['counts']], type = 'bar', name = title2,
											              text = round(df2_temp[['counts']], 2), textposition = 'auto') %>%
										      layout(title = var,
												         barmode = 'group',
												         xaxis = list(title = ""),
												         yaxis = list(title = "Frequency"))
	  }
	  
	# K-S plots
	ks_plots <- list()
	for(var in ks_tbl$variable){
	    
		fit1 <- density(df1[[var]], na.rm = T)
		fit2 <- density(df2[[var]], na.rm = T)
		    
		bin_num <- length(unique(df1[[var]]))
		start_bin <- min(df1[[var]], na.rm = T)
		end_bin <- max(df1[[var]], na.rm = T)
		bin_length <- (start_bin - end_bin)/bin_num
		bin_list <- list(start = start_bin, end = end_bin, size = bin_length)
		    
		ks_plots[[var]] <- plot_ly(alpha = 0.6, height = 800, width = 1500) %>%
		         add_trace(df1, x = df1[[var]], type = 'histogram', histnorm = "probability", name = title1, color = I('mediumturquoise'),
		                   autobinx = FALSE, xbins = bin_list) %>%
		         add_trace(df2, x = df2[[var]], type = 'histogram', histnorm = "probability", name = title2, color = I('olivedrab3'),
		                   autobinx = FALSE, xbins = bin_list) %>%
		         add_trace(x = fit1$x, y = fit1$y, type = 'scatter', mode = "lines", yaxis = "y1", name = paste0(title1, " Density"), color = I('springgreen4')) %>%
		         add_trace(x = fit2$x, y = fit2$y, type = 'scatter', mode = "lines", yaxis = "y1", name = paste0(title2, " Density"), color = I('olivedrab4')) %>%
		         layout(title = var,
		                barmode = 'overlay',
		                xaxis = list(title = ""),
		                yaxis = list(title = "Frequency"),
		                margin = list(l = 130, r = 50, b = 50, t = 50, pad = 4))
		}
	
	# Compile outputs into one list	
	outputs <- list(ks_tbl = ks_tbl,
									chi_tbl = chi_tbl,
									ks_plots = ks_plots,
									chi_plots = chi_plots
									)
	# Print final runtime
	run_tm_final <- (proc.time() - run_tm)
	print(run_tm_final)

return(outputs)
}

