################################################################################
# R tests for Part 2 feature engineering logic using testthat.
#
# These tests operate on local R data frames (no Teradata connection needed).
################################################################################

library(testthat)

# Helper: Create sample customer data
create_sample_customer <- function() {
  data.frame(
    cust_id = 1:10,
    income = c(50000, 75000, 30000, 120000, 45000, 90000, 60000, 85000, 55000, 70000),
    age = c(25, 35, 45, 55, 30, 40, 50, 28, 38, 48),
    years_with_bank = c(5, 10, 15, 20, 3, 8, 12, 2, 7, 18),
    nbr_children = c(0, 2, 1, 3, 0, 1, 2, 0, 1, 4),
    gender = c('M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'),
    marital_status = c(1, 2, 3, 1, 2, 3, 1, 2, 3, 1),
    postal_code = c('90210', '10001', '60601', '85001', '43001',
                    '75001', '94102', '11201', '77001', '33101'),
    state_code = c('CA', 'NY', 'IL', 'AZ', 'OH', 'TX', 'CA', 'NY', 'TX', 'FL'),
    stringsAsFactors = FALSE
  )
}

# Helper: Create sample accounts data
create_sample_accounts <- function() {
  acct_types <- c('CK', 'SV', 'CC')
  data <- data.frame()
  for (cust_id in 1:10) {
    for (i in 1:3) {
      row <- data.frame(
        acct_nbr = paste0('ACCT', sprintf('%04d', cust_id), i - 1),
        cust_id = cust_id,
        acct_type = acct_types[i],
        account_active = 'Y',
        acct_start_date = as.Date('2020-01-15'),
        starting_balance = 1000 + cust_id * 100 + (i - 1) * 50,
        ending_balance = 1200 + cust_id * 100 + (i - 1) * 50,
        stringsAsFactors = FALSE
      )
      data <- rbind(data, row)
    }
  }
  data
}

# Test customer feature creation
test_that("customer gender indicators are correct", {
  cust <- create_sample_customer()
  cust$female_ind <- ifelse(cust$gender == 'F', 1, 0)

  expect_equal(cust$female_ind[cust$gender == 'F'], rep(1, sum(cust$gender == 'F')))
  expect_equal(cust$female_ind[cust$gender == 'M'], rep(0, sum(cust$gender == 'M')))
})

test_that("customer marital status indicators are correct", {
  cust <- create_sample_customer()
  cust$single_ind <- ifelse(cust$marital_status == 1, 1, 0)
  cust$married_ind <- ifelse(cust$marital_status == 2, 1, 0)
  cust$separated_ind <- ifelse(cust$marital_status == 3, 1, 0)

  expect_equal(cust$single_ind[cust$marital_status == 1], rep(1, sum(cust$marital_status == 1)))
  expect_equal(cust$single_ind[cust$marital_status != 1], rep(0, sum(cust$marital_status != 1)))
  expect_equal(cust$married_ind[cust$marital_status == 2], rep(1, sum(cust$marital_status == 2)))
  expect_equal(cust$separated_ind[cust$marital_status == 3], rep(1, sum(cust$marital_status == 3)))
})

test_that("customer state indicators are correct", {
  cust <- create_sample_customer()
  cust$ca_resident_ind <- ifelse(cust$state_code == 'CA', 1, 0)
  cust$ny_resident_ind <- ifelse(cust$state_code == 'NY', 1, 0)
  cust$tx_resident_ind <- ifelse(cust$state_code == 'TX', 1, 0)
  cust$il_resident_ind <- ifelse(cust$state_code == 'IL', 1, 0)
  cust$az_resident_ind <- ifelse(cust$state_code == 'AZ', 1, 0)
  cust$oh_resident_ind <- ifelse(cust$state_code == 'OH', 1, 0)

  expect_equal(sum(cust$ca_resident_ind), 2)  # CA appears twice
  expect_equal(sum(cust$ny_resident_ind), 2)  # NY appears twice
  expect_equal(sum(cust$tx_resident_ind), 2)  # TX appears twice
  expect_equal(cust$il_resident_ind[3], 1)     # Index 3 is IL
  expect_equal(cust$az_resident_ind[4], 1)     # Index 4 is AZ
  expect_equal(cust$oh_resident_ind[5], 1)     # Index 5 is OH
})

# Test account feature creation
test_that("account type indicators are correct", {
  acct <- create_sample_accounts()
  acct$ck_acct_ind <- ifelse(acct$acct_type == 'CK', 1, 0)
  acct$sv_acct_ind <- ifelse(acct$acct_type == 'SV', 1, 0)
  acct$cc_acct_ind <- ifelse(acct$acct_type == 'CC', 1, 0)

  ck_rows <- acct$acct_type == 'CK'
  expect_equal(acct$ck_acct_ind[ck_rows], rep(1, sum(ck_rows)))
  expect_equal(acct$sv_acct_ind[ck_rows], rep(0, sum(ck_rows)))
  expect_equal(acct$cc_acct_ind[ck_rows], rep(0, sum(ck_rows)))
})

# Test ADS aggregation
test_that("ADS aggregation by cust_id produces one row per customer", {
  cust <- create_sample_customer()
  cust$female_ind <- ifelse(cust$gender == 'F', 1, 0)
  cust$single_ind <- ifelse(cust$marital_status == 1, 1, 0)
  cust$married_ind <- ifelse(cust$marital_status == 2, 1, 0)
  cust$separated_ind <- ifelse(cust$marital_status == 3, 1, 0)

  # Simple aggregation test
  agg <- aggregate(income ~ cust_id, data = cust, FUN = min)
  expect_equal(nrow(agg), 10)
  expect_true(all(!duplicated(agg$cust_id)))
})

# Test RF model training (local)
test_that("randomForest model can be trained on small data", {
  skip_if_not_installed("randomForest")
  library(randomForest)

  set.seed(42)
  n <- 50
  train_data <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    x3 = rnorm(n),
    y = as.factor(sample(0:1, n, replace = TRUE))
  )

  model <- randomForest(y ~ x1 + x2 + x3, data = train_data, ntree = 100)

  expect_true(!is.null(model))
  expect_equal(model$ntree, 100)

  predictions <- predict(model, newdata = train_data[1:5, ])
  expect_equal(length(predictions), 5)
})
