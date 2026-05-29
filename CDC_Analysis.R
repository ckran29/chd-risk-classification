
# ============================================================
# CHD Classification Analysis — CDC BRFSS 2024
# Author: C. Krance | Florida Southern College
# Course: BUS 4440 Data Mining
# ============================================================



#---Libraries---

library(haven)
library(tidyverse)
library(dplyr)
library(caret)
library(rpart)
library(rpart.plot)
library(glmnet) #lasso
library(ggplot2)
library(pROC)
library(xgboost)
library(Ckmeans.1d.dp)
library(doParallel)



#---Parallel Processing---
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)


#---Import Relevant Data---

df_raw <- read_xpt("LLCP2024.XPT")%>% 
  dplyr::select(
    `_MICHD`,     # outcome — reported having CHD or MI
    `_SEX`,       # biological sex
    `_AGEG5YR`,   # age categories
    `_RACE`,      # race categories
    `_BMI5CAT`,   # BMI categories
    `_EDUCAG`,    # education categories
    `_INCOMG1`,   # income brackets
    SMOKE100,     # smoked 100 cigs in lifetime
    CVDSTRK3,     # ever had a stroke
    DIABETE4,     # diabetes (Yes/No)
    GENHLTH,      # self assessed general health categories
    MEDCOST1,     # can't see doctor due to cost
    CHECKUP1,     # time since last routine checkup
    `_TOTINDA`,   # physical activity last 30 days
    AVEDRNK4,     # avg drinks over last 30 days
    `_HLTHPL2`,   # health insurance coverage
    MENTHLTH      # mental health bad days
  )




#---Preprocess---

df_clean <- df_raw %>% 
  mutate(
    MICHD = case_when(
      `_MICHD` == 2 ~ 0,
      `_MICHD` == 1 ~ 1,
      TRUE ~ NA_real_
    ),
    SEX = factor(`_SEX`,
                 levels = 1:2,
                 labels = c("Male", "Female")
                 
    ),
    AGEG5YR = factor(`_AGEG5YR`,
                     levels = 1:13,
                     labels = c("18_24", "25_29", "30_34", "35_39", "40_44",
                                "45_49", "50_54", "55_59", "60_64", "65_69",
                                "70_74", "75_79", ">80")
    ),
    RACE = factor(`_RACE`,
                  levels = 1:8,
                  labels = c("White", "Black", "American Indian/Alaskan Native", 
                             "Asian", "Native Hawaiian_Pacific Islander", 
                             "Other", "Multiracial", "Hispanic")
    ),
    EDUCAG = factor(`_EDUCAG`,
                    levels = 1:4,
                    labels = c("DNF_High_School","High_School_Grad",
                               "Attended_College","College_Grad")
                    
    ),
    BMI5CAT = factor(`_BMI5CAT`,
                     levels = 1:4,
                     labels = c("Underweight", "Normal", 
                                "Overweight", "Obese")
                     
    ),
    INCOMG1 = case_when(
      `_INCOMG1` == 9 ~ NA_real_,
      TRUE ~ `_INCOMG1`
    ),
    SMOKE100 = case_when(
      SMOKE100 == 1 ~ 1,
      SMOKE100 == 2 ~ 0,
      TRUE ~ NA_real_
      
    ),
    CVDSTRK3 = case_when(
      CVDSTRK3 == 1 ~ 1,
      CVDSTRK3 == 2 ~ 0,
      TRUE ~ NA_real_
      
    ),
    DIABETE4 = case_when(
      DIABETE4 == 1 ~ 1,
      DIABETE4 %in% 2:4 ~ 0,
      TRUE ~ NA_real_
      
    ),
    GENHLTH = factor(GENHLTH,
                     levels = 1:5,
                     labels = c("Excellent", "Very Good", "Good", "Fair", "Poor")
                     
    ),
    MEDCOST1 = case_when(
      MEDCOST1 == 1 ~ 1,
      MEDCOST1 == 2 ~ 0,
      TRUE ~ NA_real_
      
    ),
    CHECKUP1 = factor(CHECKUP1,
                      levels = c(1,2,3,4,8),
                      labels = c("<1_year","1_2_years","2_5_years",">5_years","Never")
                      
    ),
    TOTINDA = case_when(
      `_TOTINDA` == 1 ~ 1,
      `_TOTINDA` == 2 ~ 0,
      TRUE ~ NA_real_
      
    ),
    AVEDRNK4 = case_when(
      AVEDRNK4 == 88 ~ 0,      ## coded as none
      AVEDRNK4 %in% 1:76 ~ AVEDRNK4,
      TRUE ~ NA_real_ 
      
    ),
    HLTHPL2 = case_when(
      `_HLTHPL2` == 1 ~ 1,
      `_HLTHPL2` == 2 ~ 0,
      TRUE~ NA_real_
      
    ),
    MENTHLTH = case_when(
      MENTHLTH == 88 ~ 0,
      MENTHLTH %in% 1:30 ~ MENTHLTH,
      TRUE ~ NA_real_
    ),
  ) %>%
  dplyr::select(MICHD, everything(), -`_MICHD`, -`_SEX`, -`_AGEG5YR`, -`_RACE`, 
                -`_EDUCAG`, -`_BMI5CAT`, -`_INCOMG1`, 
                -`_TOTINDA`, -`_HLTHPL2`)




#---Drop NA---

df_clean <- df_clean %>%
  drop_na()



#---Downsample---

set.seed(123)
df_clean_down <- downSample(
  x = df_clean %>% dplyr::select(-MICHD),     ## Addresses huge class imbalance
  y = as.factor(df_clean$MICHD),
  yname = "MICHD"
)


#---Create Training and Holdout Sets---

set.seed(123)

idx <- createDataPartition(df_clean_down$MICHD, p = 0.6, list = FALSE)
train.df <- df_clean_down[idx,]
holdout.df <- df_clean_down[-idx,]



#---LASSO Variable Selection---


# 1. Separate features and target
x_train <- train.df[, !names(train.df) %in% "MICHD"]
x_test  <- holdout.df[, !names(holdout.df) %in% "MICHD"]
y_train <- train.df$MICHD
y_test  <- holdout.df$MICHD

# 2. Dummy encode using model.matrix
x_train_matrix <- model.matrix(~ . - 1, data = x_train)
x_test_matrix  <- model.matrix(~ . - 1, data = x_test)

# 3. Fit Lasso
cv_lasso <- cv.glmnet(x_train_matrix, y_train, alpha = 1, family = "binomial", nfolds = 5)

# 4. Predict on holdout
predictions <- predict(cv_lasso, newx = x_test_matrix, s = "lambda.min", type = "class")





#---LASSO Confusion Matrix---

# 1. Confusion matrix
confusionMatrix(as.factor(predictions), as.factor(y_test))

# 2. Quick accuracy
mean(predictions == y_test)

# 3. See which variables Lasso kept (non-zero coefficients)
coef(cv_lasso, s = "lambda.min")

# 4. Plot CV error vs lambda
plot(cv_lasso)






#---Variable Selection---

lasso_coefs <- coef(cv_lasso, s = "lambda.min")
lasso_df <- data.frame(
  variable = rownames(lasso_coefs),
  coefficient = as.numeric(lasso_coefs)
)

# Sort by absolute value to see most important variables
lasso_df <- lasso_df[order(abs(lasso_df$coefficient), decreasing = TRUE), ] # top 9 were used
lasso_df <- lasso_df[lasso_df$coefficient != 0, ]  # drop zeroes
print(lasso_df)



#---Create Final Dataset---

df_final <- df_clean_down %>% 
  dplyr::select(MICHD, AGEG5YR, CVDSTRK3, GENHLTH, SEX, CHECKUP1, RACE, DIABETE4, MEDCOST1, SMOKE100)


#---Final Training and Validation Split---

set.seed(123)

idx <- createDataPartition(df_final$MICHD, p = 0.6, list = FALSE)
train.df <- df_final[idx,]
holdout.df <- df_final[-idx,]


#---Decision Tree---

tree_model_1 <- rpart(MICHD ~ .,
                      data = train.df,
                      method = "class",
                      parms = list(prior = c(0.5, 0.5)),
                      control = rpart.control(
                        maxdepth = 10,    
                        minsplit = 20,    # min observations to attempt a split
                        cp = 0.001        # lower = more splits, more complex tree
                      ))
g = rpart.plot(tree_model_1, extra = 1, fallen.leaves = FALSE)


#---Tree PNG---

png("cdc_chd_decision_tree.png", width = 4000, height = 2500, res = 200)
rpart.plot(tree_model_1, 
           extra = 1, 
           fallen.leaves = FALSE,
           tweak = 1.2)
dev.off()



#---Decision Tree Confusion Matrix---

tree.pred <- predict(tree_model_1, holdout.df, type = "class")
confusionMatrix(tree.pred, as.factor(holdout.df$MICHD), 
                positive = "1")



#---Logistic Regression---

train.df$MICHD <- as.factor(train.df$MICHD)
holdout.df$MICHD <- as.factor(holdout.df$MICHD)

trControl <- caret::trainControl(method = "cv", number = 5, allowParallel = TRUE)
logit.reg <- caret::train(MICHD ~ ., data = train.df, trControl = trControl, 
                          method = "glm", family = "binomial")

logit.reg
summary(logit.reg$finalModel)



#---Logistic Regression Confusion Matrix---

logit.pred <- predict(logit.reg, holdout.df)
confusionMatrix(logit.pred, as.factor(holdout.df$MICHD), positive = "1")






#---Encode for XGBoost---
x_train_xgb <- model.matrix(MICHD ~ . - 1, data = train.df)
x_hold_xgb  <- model.matrix(MICHD ~ . - 1, data = holdout.df)
y_train_xgb <- as.numeric(as.character(train.df$MICHD))
y_hold_xgb  <- as.numeric(as.character(holdout.df$MICHD))

dtrain <- xgb.DMatrix(data = x_train_xgb, label = y_train_xgb)
dhold  <- xgb.DMatrix(data = x_hold_xgb,  label = y_hold_xgb)

# Train — early stopping prevents overfitting and saves time
set.seed(123)
xgb_model <- xgb.train(
  params = list(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    max_depth        = 6,
    eta              = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    nthread          = parallel::detectCores() - 1
  ),
  data              = dtrain,
  nrounds           = 500,
  watchlist         = list(train = dtrain, eval = dhold),
  early_stopping_rounds = 20,   # stops if AUC doesn't improve for 20 rounds
  verbose           = 0
)

# Predict
xgb_prob  <- predict(xgb_model, dhold)
xgb_class <- as.factor(ifelse(xgb_prob > 0.5, 1, 0))

confusionMatrix(xgb_class, as.factor(holdout.df$MICHD), positive = "1")

# Variable importance plot
xgb_imp <- xgb.importance(model = xgb_model)
xgb.ggplot.importance(xgb_imp, top_n = 15) +
  labs(title = "XGBoost Feature Importance") +
  theme_minimal(base_size = 12)





#---ROC Probabilities---
tree_prob <- predict(tree_model_1, holdout.df, type = "prob")[, "1"]
logit_prob <- predict(logit.reg, holdout.df, type = "prob")[, "1"]


#---Build ROC objects---
roc_tree  <- roc(as.numeric(as.character(holdout.df$MICHD)), tree_prob,  quiet = TRUE)
roc_logit <- roc(as.numeric(as.character(holdout.df$MICHD)), logit_prob, quiet = TRUE)
roc_xgb   <- roc(as.numeric(as.character(holdout.df$MICHD)), xgb_prob,   quiet = TRUE)



#---Aggregated ROC Chart---
ggroc(list(
  "Decision Tree"       = roc_tree,
  "Logistic Regression" = roc_logit,
  "XGBoost"             = roc_xgb
), linewidth = 0.8) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed", color = "gray50") +
  scale_color_manual(values = c("#E63946", "#457B9D", "#2A9D8F")) +
  labs(
    title = "ROC Curves — CHD Classification Models",
    x = "Specificity", y = "Sensitivity", color = "Model"
  ) +
 annotate("text", x = 0.75, y = 0.05,
         label = paste0("AUC:  Tree: ",     sprintf("%.3f", auc(roc_tree)),
                        "   |   Logit: ",   sprintf("%.3f", auc(roc_logit)),
                        "   |   XGBoost: ", sprintf("%.3f", auc(roc_xgb))),
         size = 3.2, hjust = 0)



#---Model Comparison---

model_comparison <- data.frame(
  Model          = c("Decision Tree", "Logistic Regression", "XGBoost"),
  Accuracy       = c(0.7566, 0.7611, 0.7588),
  Sensitivity    = c(0.8134, 0.7811, 0.7893),
  Specificity    = c(0.6998, 0.7410, 0.7282),
  Pos_Pred_Value = c(0.7304, 0.7510, 0.7439),
  Neg_Pred_Value = c(0.7895, 0.7720, 0.7756),
  Kappa          = c(0.5132, 0.5221, 0.5176),
  AUC            = c(0.810,  0.839,  0.838)
)
model_comparison







#---Production Model Refit---

# Refit final model on full downsampled dataset for deployment
set.seed(123)
final_logit <- caret::train(MICHD ~ ., 
                            data = df_final, 
                            method = "glm", 
                            family = "binomial",
                            trControl = trainControl(method = "cv", 
                                                     number = 5, 
                                                     allowParallel = TRUE))



#---End parallel processing---
stopCluster(cl)












