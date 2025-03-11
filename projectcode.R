install.packages(c("flexdashboard", "ggplot2", "dplyr", "caret", "randomForest", "DT", "readr"))
install.packages("elasticnet")

library(flexdashboard)
library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)
library(DT)
library(readr)
library(plotly)

# Load datasets
heart_data <- read_csv(file.choose()) #heart.csv
o2_data <- read_csv(file.choose())  #o2Saturation.csv

# Split the heart data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(heart_data$output, p = .8, 
                                  list = FALSE, 
                                  times = 1)
heart_train <- heart_data[trainIndex, ]
heart_test <- heart_data[-trainIndex, ]


  
  
  #### Linear Regression Model


lm_model <- train(output ~ ., data = heart_train, method = "lm")
lm_pred <- predict(lm_model, heart_test)
summary(lm_model)

  #### Ridge Regression Model
  

ridge_model <- train(output ~ ., data = heart_train, method = "ridge")
ridge_pred <- predict(ridge_model, heart_test)
summary(ridge_model)

  #### Random Forest Model
 

rf_model <- randomForest(output ~ ., data = heart_train)
rf_pred <- predict(rf_model, heart_test)
print(rf_model)


  ### Model Comparison
 
results <- data.frame(
  Model = c("Linear Regression", "Ridge Regression", "Random Forest"),
  RMSE = c(
    RMSE(lm_pred, heart_test$output),
    RMSE(ridge_pred, heart_test$output),
    RMSE(rf_pred, heart_test$output)
  )
)
p <- ggplot(results, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Comparison: RMSE", y = "Root Mean Squared Error") +
  theme_minimal()
ggplotly(p)



  ### Feature Importance from Random Forest
 

importance <- varImp(rf_model)
plot(importance, main = "Feature Importance from Random Forest")


### Distribution of Predictions


p2 <- ggplot(heart_test, aes(x = output, y = rf_pred)) +
  geom_point(color = "blue", alpha = 0.6) +
  labs(title = "Predicted vs Actual Values", x = "Actual Output", y = "Predicted Output") +
  theme_light()
ggplotly(p2)


  ### Bar Chart: Frequency of Target Variable


ggplot(heart_data, aes(x = as.factor(output), fill = as.factor(output))) +
  geom_bar() +
  labs(title = "Bar Chart: Frequency of Target Variable", x = "Output", y = "Count") +
  theme_minimal()


### Histogram Age distribution


ggplot(heart_data, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(title = "Histogram: Age Distribution", x = "Age", y = "Frequency") +
  theme_light()


  ### Bar Plot: Gender Distribution
  
ggplot(heart_data, aes(x = as.factor(sex), fill = as.factor(sex))) +
  geom_bar() +
  labs(title = "Bar Plot: Gender Distribution", x = "Gender (0 = Female, 1 = Male)", y = "Count") +
  theme_minimal()


### Histogram: Cholesterol Levels

ggplot(heart_data, aes(x = chol)) +
  geom_histogram(binwidth = 10, fill = "lightgreen", color = "black", alpha = 0.7) +
  labs(title = "Histogram: Cholesterol Levels", x = "Cholesterol", y = "Frequency") +
  theme_light()






