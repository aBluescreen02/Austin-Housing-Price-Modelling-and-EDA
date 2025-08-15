library(readr)
library(tidyverse)
library(ggplot2)
library(scales)

raw_df <- read_csv("austinHousingData.csv")

# Exploratory Data Analysis (EDA) #####

head(raw_df)

colSums(is.na(raw_df))

# Data Cleaning #####
df <- raw_df %>%
  select(-zpid) %>% 
  select(where(~ !is.character(.))) %>%
  select(-latest_saledate) %>%
  select(-latest_salemonth) %>%
  mutate(across(where(is.logical), as.numeric)) %>%
  filter(latestPrice >= 100000)

dim(df)

print(paste("Number of missing entries:", sum(is.na(df))))

# Univariate Analysis

# Number of listings over $1,000,000
above_mil <- sum(df$latestPrice >= 1000000)

print(paste("Number of properties listed above $1,000,000:", above_mil))
print(paste0("Percentage of properties listed above $1,000,000: ", round(above_mil / nrow(df) * 100, digits = 2), "%"))

# Histogram of prices
ggplot(df, aes(x = latestPrice)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  scale_x_log10(labels = label_dollar(scale_cut = cut_short_scale())) +
  labs(
    title = "Distribution of Latest Prices",
    x = "Listing Price (log10)",
    y = "Count"
  ) +
  theme_minimal()

library(plotly)

# Create individual boxplots
p1 <- plot_ly(df, y = ~livingAreaSqFt, type = "box", name = "Living Area (SqFt)") %>%
  layout(yaxis = list(title = "SqFt"))

p2 <- plot_ly(df, y = ~numOfBedrooms, type = "box", name = "Bedrooms") %>%
  layout(yaxis = list(title = "Count"))

p3 <- plot_ly(df, y = ~numOfBathrooms, type = "box", name = "Bathrooms") %>%
  layout(yaxis = list(title = "Count"))

p4 <- plot_ly(df, y = ~yearBuilt, type = "box", name = "Year Built") %>%
  layout(yaxis = list(title = "Year"))

# Combine into one row of plots
subplot(p1, p2, p3, p4, nrows = 1, shareX = FALSE, titleX = TRUE, titleY = TRUE) %>%
  layout(title = "Boxplots of Selected Features")

avg_price_by_zipcode <- df %>% 
  group_by(zipcode) %>%
  summarize(avg_price = mean(latestPrice)) %>%
  arrange(avg_price)

plot_ly(
  data = avg_price_by_zipcode,
  x = ~reorder(zipcode, avg_price),
  y = ~avg_price,
  type = "bar",
  marker = list(color = "steelblue"),
  hovertemplate = paste(
    "ZIP code: %{x}<br>",
    "Avg price: $%{y:,.0f}<extra></extra>"
  )
) %>%
  layout(
    title = "Average House Price by Zipcode",
    xaxis = list(title = "Zipcode", tickangle = -90),
    yaxis = list(title = "Average Price ($)", tickformat = ".2s"))

# Geographical Map

library(zipcodeR)
library(leaflet)

# Make sure both ZIPs are character type for join
avg_price_by_zipcode <- avg_price_by_zipcode %>%
  mutate(zipcode = as.character(zipcode))

plot_data <- avg_price_by_zipcode %>%
  left_join(zip_code_db, by = c("zipcode" = "zipcode"))

# Define color scale
palette <- colorNumeric(
  palette = c("#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"),
  domain = plot_data$avg_price
)

# Create map
leaflet(plot_data) %>%
  addProviderTiles("CartoDB.DarkMatter") %>%
  addCircleMarkers(
    lng = ~lng,
    lat = ~lat,
    radius = 8,
    color = ~palette(avg_price),
    fillOpacity = 0.6,
    stroke = FALSE,
    label = ~paste0(
      "ZIP code: ", zipcode,
      " | Avg price: $", prettyNum(round(avg_price), big.mark = ",")
    ),
    labelOptions = labelOptions(direction = "auto")
  ) %>%
  addLegend(
    position = "topright",
    pal = palette,
    values = ~avg_price,
    title = "Average Price",
    labFormat = labelFormat(prefix = "$")
  )

# Bivariate Analysis #####

# Exclude categorical variables
numeric_df <- df %>% select(where(is.numeric))

# Calculate correlation matrix
corr_matrix <- abs(cor(numeric_df, use = "complete.obs"))

# Get correlation with latestPrice
corr_latestPrice <- corr_matrix[, "latestPrice"] |> sort(decreasing = TRUE)

# Convert to dataframe
corr_df <- data.frame(
  Feature = names(corr_latestPrice),
  Correlation = corr_latestPrice
)

# Horizontal bar chart
p <- plot_ly(
  data = corr_df,
  x = ~Correlation,
  y = ~reorder(Feature, Correlation),
  type = "bar",
  orientation = "h",
  marker = list(color = "steelblue")
) %>%
  layout(
    title = "Correlation with Latest Price",
    xaxis = list(title = "Correlation Coefficient"),
    yaxis = list(title = "Feature", tickfont = list(size = 6)),
    margin = list(l = 150)
  )

# MODELS #####

library(caret)

# Data Split #####

set.seed(1)
train_index <- createDataPartition(df$latestPrice, p = 0.9, list = FALSE)

train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# LASSO MODEL #####

library(glmnet)

set.seed(1)

lasso_fit <- train(latestPrice ~ ., data = train_data,
                   method = "glmnet",
                   preProcess = c("center", "scale"),
                   trControl = trainControl(method = "cv", number = 10),
                   tuneGrid = expand.grid(
                     .alpha = 1,
                     .lambda = seq(3000, 7000, by = 10)
                   ),
                   metric = "RMSE")

# Predict using model
test_prediction <- predict(lasso_fit, newdata = test_data)
train_prediction <- predict(lasso_fit, newdata = train_data)

# Evaluate Model
lasso_test_rmse <- RMSE(pred = test_prediction, obs = test_data$latestPrice)
lasso_train_rmse <- RMSE(pred = train_prediction, obs = train_data$latestPrice)

print(paste("LASSO Test RMSE:", lasso_test_rmse))
print(paste("LASSO Training RMSE:", lasso_train_rmse))

plot(lasso_fit)

# Variable Importance
varImp(lasso_fit)

# GLM MODEL #####

set.seed(1)

glm_fit <- train(latestPrice ~ ., data = train_data,
                   method = "glm",
                   preProcess = c("center", "scale"),
                   trControl = trainControl(method = "repeatedcv", 
                                            number = 10,
                                            repeats = 20),
                   metric = "RMSE")

# Predict using model
test_prediction <- predict(glm_fit, newdata = test_data)
train_prediction <- predict(glm_fit, newdata = train_data)

# Evaluate Model
glm_test_rmse <- RMSE(pred = test_prediction, obs = test_data$latestPrice)
glm_train_rmse <- RMSE(pred = train_prediction, obs = train_data$latestPrice)

print(paste("GLM Test RMSE:", glm_test_rmse))
print(paste("GLM Training RMSE:", glm_train_rmse))

# Variable Importance
varImp(glm_fit)

# BOOSTING #####

library(xgboost)

set.seed(1)

xgb_fit <- train(latestPrice ~ ., data = train_data,
                      method = "xgbTree",
                      preProcess = c("center", "scale"),
                      trControl = trainControl(method = "cv", 
                                               number = 10),
                      metric = "RMSE",
                      verbosity = 0)

# Predict using model
test_prediction <- predict(xgb_fit, newdata = test_data)
train_prediction <- predict(xgb_fit, newdata = train_data)

# Evaluate Model
xgb_test_rmse <- RMSE(pred = test_prediction, obs = test_data$latestPrice)
xgb_train_rmse <- RMSE(pred = train_prediction, obs = train_data$latestPrice)

print(paste("XGBoost Test RMSE:", xgb_test_rmse))
print(paste("XGBoost Training RMSE:", xgb_train_rmse))

# Variable Importance
varImp(xgb_fit)

# Plot hyperparameter comparisons
plot(xgb_fit)

# Final model parameters
xgb_fit$bestTune


# Discussion #####

# Comparing distribution of predicted vs. actual prices
combined_data <- data.frame(
  value = c(test_data$latestPrice, test_prediction),
  type = rep(c("Actual", "Predicted"), each = nrow(test_data))
)

ggplot(combined_data, aes(x = value, fill = type)) +
  geom_histogram(alpha = 0.5, position = "identity", bins = 30, color = "white") +
  scale_x_log10(labels = label_dollar(scale_cut = cut_short_scale())) +
  scale_fill_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal() +
  labs(title = "XGBoost Test Data Actual vs Predicted Price Distributions",
       x = "Listing Price (log10)",
       y = "Count",
       fill = "Legend")

# Plot Deal Quality
library(plotly)

# For test data
deal_data <- data.frame(
  latestPrice = test_data$latestPrice,
  prediction = test_prediction,
  numOfBedrooms = test_data$numOfBedrooms,
  numOfBathRooms = test_data$numOfBathrooms,
  livingAreaSqFt = test_data$livingAreaSqFt
)

# Filter out negative predictions
deal_data <- subset(deal_data, prediction > 0 & latestPrice > 0)

# Calculate the difference ratio
deal_data$diff_ratio <- (deal_data$latestPrice - deal_data$prediction) / deal_data$latestPrice

# Assign colors
deal_data$deal_quality <- cut(
  deal_data$diff_ratio,
  breaks = c(-Inf, -0.3, 0.3, Inf),
  labels = c("Good", "Fair", "Bad")
)

# Map deal quality to custom colors
deal_colors <- c("Good" = "green3", "Fair" = "gray50", "Bad" = "red3")

# Add custom tooltip text
deal_data$hover_text <- paste0(
  "Actual: $", round(deal_data$latestPrice, 0), "<br>",
  "Predicted: $", round(deal_data$prediction, 0), "<br>",
  "Beds: ", deal_data$numOfBedrooms, "<br>",
  "Bathrooms: ", deal_data$numOfBathRooms, "<br>",
  "Square Footage: ", deal_data$livingAreaSqFt, "<br>"
)

plot_ly(
  data = deal_data,
  x = ~prediction,
  y = ~latestPrice,
  type = 'scatter',
  mode = 'markers',
  color = ~deal_quality,
  colors = deal_colors,
  text = ~hover_text,
  hoverinfo = 'text',
  marker = list(size = 8, opacity = 0.7)
) %>%
  layout(
    title = "XGBoost Listing Deal Quality Prediction (Test Set)",
    xaxis = list(title = "Predicted Price ($)", type = "log"),
    yaxis = list(title = "Actual Listing Price ($)", type = "log"),
    legend = list(title = list(text = "Deal Quality")),
    shapes = list(  # Add reference line (y = x)
      list(
        type = "line",
        x0 = min(deal_data$prediction),
        y0 = min(deal_data$prediction),
        x1 = max(deal_data$prediction),
        y1 = max(deal_data$prediction),
        line = list(dash = "dash", color = "black")
      )
    )
  )

# Pie chart for number of good, fair, and bad deals
deal_counts <- deal_data %>%
  count(deal_quality) %>%
  mutate(
    pct = n / sum(n) * 100,
    label = paste0(round(pct, 1), "%")
  )

xgb_deal_dist <- ggplot(deal_counts, aes(x = "", y = n, fill = deal_quality)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar(theta = "y") +
  scale_fill_manual(values = deal_colors) +
  geom_label(
    aes(label = label),
    position = position_stack(vjust = 0.5),
    color = "white",
    size = 5
  ) +
  theme_void() +
  labs(
    title = "XGBoost Deal Quality Distribution",
    fill = "Deal Quality"
  ) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "top",
        legend.direction = "horizontal")

xgb_deal_dist

set.seed(1)

tuned_xgb_fit <- train(latestPrice ~ ., data = train_data,
                       method = "xgbTree",
                       preProcess = c("center", "scale"),
                       trControl = trainControl(method = "cv", 
                                                number = 10),
                       tuneGrid = expand.grid(nrounds = 50,
                                              max_depth = 5,
                                              eta = 0.1,
                                              gamma = 1,
                                              colsample_bytree = 0.7,
                                              min_child_weight = 1,
                                              subsample = 0.75),
                       metric = "RMSE",
                       verbosity = 0)

# Final model parameters
tuned_xgb_fit$bestTune

# Predict using model
tuned_xgb_test_prediction <- predict(tuned_xgb_fit, newdata = test_data)
tuned_xgb_train_prediction <- predict(tuned_xgb_fit, newdata = train_data)

# Evaluate Model
print(paste("Tuned XGBoost Test RMSE:", RMSE(pred = tuned_xgb_test_prediction, obs = test_data$latestPrice)))
print(paste("Tuned XGBoost Training RMSE:", RMSE(pred = tuned_xgb_train_prediction, obs = train_data$latestPrice)))

# Comparison
print(paste("Old XGBoost Test RMSE:", xgb_test_rmse))
print(paste("Old XGBoost Training RMSE:", xgb_train_rmse))

# Comparing distribution of predicted vs. actual prices
combined_data <- data.frame(
  value = c(test_data$latestPrice, tuned_xgb_test_prediction),
  type = rep(c("Actual", "Predicted"), each = nrow(test_data))
)

ggplot(combined_data, aes(x = value, fill = type)) +
  geom_histogram(alpha = 0.5, position = "identity", bins = 30, color = "white") +
  scale_x_log10(labels = label_dollar(scale_cut = cut_short_scale())) +
  scale_fill_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal() +
  labs(title = "Tuned XGBoost Test Data Actual vs Predicted Price Distributions",
       x = "Listing Price (log10)",
       y = "Count",
       fill = "Legend")

# For test data
deal_data <- data.frame(
  latestPrice = test_data$latestPrice,
  prediction = tuned_xgb_test_prediction,
  numOfBedrooms = test_data$numOfBedrooms,
  numOfBathRooms = test_data$numOfBathrooms,
  livingAreaSqFt = test_data$livingAreaSqFt
)

# Filter out negative predictions
deal_data <- subset(deal_data, prediction > 0 & latestPrice > 0)

# Calculate the difference ratio
deal_data$diff_ratio <- (deal_data$latestPrice - deal_data$prediction) / deal_data$latestPrice

# Assign colors
deal_data$deal_quality <- cut(
  deal_data$diff_ratio,
  breaks = c(-Inf, -0.3, 0.3, Inf),
  labels = c("Good", "Fair", "Bad")
)

# Map deal quality to custom colors
deal_colors <- c("Good" = "green3", "Fair" = "gray50", "Bad" = "red3")

# Add custom tooltip text
deal_data$hover_text <- paste0(
  "Actual: $", round(deal_data$latestPrice, 0), "<br>",
  "Predicted: $", round(deal_data$prediction, 0), "<br>",
  "Beds: ", deal_data$numOfBedrooms, "<br>",
  "Bathrooms: ", deal_data$numOfBathRooms, "<br>",
  "Square Footage: ", deal_data$livingAreaSqFt, "<br>"
)

plot_ly(
  data = deal_data,
  x = ~prediction,
  y = ~latestPrice,
  type = 'scatter',
  mode = 'markers',
  color = ~deal_quality,
  colors = deal_colors,
  text = ~hover_text,
  hoverinfo = 'text',
  marker = list(size = 8, opacity = 0.7)
) %>%
  layout(
    title = "XGBoost Listing Deal Quality Prediction (Test Set)",
    xaxis = list(title = "Predicted Price ($)", type = "log"),
    yaxis = list(title = "Actual Listing Price ($)", type = "log"),
    legend = list(title = list(text = "Deal Quality")),
    shapes = list(  # Add reference line (y = x)
      list(
        type = "line",
        x0 = min(deal_data$prediction),
        y0 = min(deal_data$prediction),
        x1 = max(deal_data$prediction),
        y1 = max(deal_data$prediction),
        line = list(dash = "dash", color = "black")
      )
    )
  )

# Restrict dataset to prices below $1,000,000 listings
below_mil_df <- df %>%
  filter(latestPrice <= 1000000)

set.seed(1)
train_index <- createDataPartition(below_mil_df$latestPrice, p = 0.9, list = FALSE)

below_mil_train_data <- below_mil_df[train_index, ]
below_mil_test_data <- below_mil_df[-train_index, ]

# Train on subset of data using GLM model
set.seed(1)

below_mil_glm_fit <- train(latestPrice ~ ., data = below_mil_train_data,
                           method = "glm",
                           preProcess = c("center", "scale"),
                           trControl = trainControl(method = "cv", 
                                                    number = 10),
                           metric = "RMSE")

# Predict using model
below_mil_test_prediction <- predict(below_mil_glm_fit, newdata = below_mil_test_data)
below_mil_train_prediction <- predict(below_mil_glm_fit, newdata = below_mil_train_data)

# Evaluate Below $1,000,000 Model
print(paste("Below $1,000,000 GLM Test RMSE:", RMSE(pred = below_mil_test_prediction, obs = below_mil_test_data$latestPrice)))
print(paste("Below $1,000,000 GLM Training RMSE:", RMSE(pred = below_mil_train_prediction, obs = below_mil_train_data$latestPrice)))

# Comparison
print(paste("GLM Test RMSE:", glm_test_rmse))
print(paste("GLM Training RMSE:", glm_train_rmse))

# Restrict dataset to top 25 correlated features
top_features <- corr_df %>%
  slice_max(Correlation, n = 26) %>%
  pull(Feature)

top25_df <- df %>%
  select(all_of(top_features))

set.seed(1)
train_index <- createDataPartition(top25_df$latestPrice, p = 0.9, list = FALSE)

top25_train_data <- top25_df[train_index, ]
top25_test_data <- top25_df[-train_index, ]

# Train on subset of data using GLM model
set.seed(1)

top25_glm_fit <- train(latestPrice ~ ., data = top25_train_data,
                       method = "glm",
                       preProcess = c("center", "scale"),
                       trControl = trainControl(method = "cv", 
                                                number = 10),
                       metric = "RMSE")

# Predict using model
top25_test_prediction <- predict(top25_glm_fit, newdata = top25_test_data)
top25_train_prediction <- predict(top25_glm_fit, newdata = top25_train_data)

# Evaluate top 25 features model
print(paste("Top 25 features GLM Test RMSE:", RMSE(pred = top25_test_prediction, obs = top25_test_data$latestPrice)))
print(paste("Top 25 features GLM Training RMSE:", RMSE(pred = top25_train_prediction, obs = top25_train_data$latestPrice)))

# Comparison
print(paste("GLM Test RMSE:", glm_test_rmse))
print(paste("GLM Training RMSE:", glm_train_rmse))
