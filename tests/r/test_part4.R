################################################################################
# R tests for Part 4 SCRIPT/ExecR logic using testthat.
#
# Tests model save/load, and RFScore.r input parsing with simulated data.
################################################################################

library(testthat)

# Test RF model save/load round-trip
test_that("randomForest model saveRDS/readRDS round-trip works", {
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

  # Save and reload
  tmp_file <- tempfile(fileext = ".rds")
  saveRDS(model, tmp_file)
  loaded_model <- readRDS(tmp_file)

  # Verify predictions match
  test_data <- data.frame(x1 = c(0.5, -0.5), x2 = c(1.0, -1.0), x3 = c(0.0, 0.0))
  orig_pred <- predict(model, newdata = test_data, type = "vote")
  loaded_pred <- predict(loaded_model, newdata = test_data, type = "vote")

  expect_equal(orig_pred, loaded_pred)

  # Cleanup
  file.remove(tmp_file)
})

# Test RFScore.r input parsing logic
test_that("RFScore input parsing handles tab-delimited data correctly", {
  # Create sample tab-delimited data matching the 28-column schema
  cn <- c("cust_id", "tot_income", "tot_age", "tot_cust_years", "tot_children",
          "female_ind", "single_ind", "married_ind", "seperated_ind",
          "ca_resident_ind", "ny_resident_ind", "tx_resident_ind",
          "il_resident_ind", "az_resident_ind", "oh_resident_ind",
          "ck_acct_ind", "cc_acct_ind", "sv_acct_ind",
          "ck_avg_bal", "sv_avg_bal", "cc_avg_bal",
          "ck_avg_tran_amt", "sv_avg_tran_amt", "cc_avg_tran_amt",
          "q1_trans_cnt", "q2_trans_cnt", "q3_trans_cnt", "q4_trans_cnt")

  ct <- c("integer", "double", "integer", "integer", "integer",
          "integer", "integer", "integer", "integer",
          "integer", "integer", "integer", "integer", "integer", "integer",
          "integer", "factor", "integer",
          "double", "double", "double",
          "double", "double", "double",
          "integer", "integer", "integer", "integer")

  # Create a temporary file with sample data
  tmp_file <- tempfile()
  sample_data <- paste(
    "1\t50000\t30\t5\t0\t0\t1\t0\t0\t1\t0\t0\t0\t0\t0\t1\t0\t1\t1000\t2000\t500\t150\t100\t50\t10\t8\t12\t9",
    "2\t75000\t35\t10\t2\t1\t0\t1\t0\t0\t1\t0\t0\t0\t0\t1\t1\t1\t1500\t2500\t600\t200\t120\t60\t11\t9\t13\t10",
    sep = "\n"
  )
  writeLines(sample_data, tmp_file)

  # Parse the data as RFScore.r would
  inputDF <- read.table(tmp_file, sep = "\t", flush = TRUE, header = FALSE,
                        quote = "", na.strings = "", colClasses = ct, col.names = cn)

  expect_equal(nrow(inputDF), 2)
  expect_equal(ncol(inputDF), 28)
  expect_equal(inputDF$cust_id[1], 1)
  expect_equal(inputDF$tot_income[1], 50000)
  expect_true(is.factor(inputDF$cc_acct_ind))

  # Cleanup
  file.remove(tmp_file)
})

# Test scoring with parsed data
test_that("RFScore scoring produces correct output format", {
  skip_if_not_installed("randomForest")
  library(randomForest)

  # Create a small model
  set.seed(42)
  n <- 50
  train_data <- data.frame(
    tot_income = rnorm(n, 50000, 10000),
    tot_age = sample(25:60, n, replace = TRUE),
    tot_cust_years = sample(1:20, n, replace = TRUE),
    cc_acct_ind = as.factor(sample(0:1, n, replace = TRUE))
  )

  model <- randomForest(cc_acct_ind ~ tot_income + tot_age + tot_cust_years,
                        data = train_data, ntree = 100)

  # Score
  test_data <- data.frame(
    cust_id = 1:5,
    tot_income = rnorm(5, 50000, 10000),
    tot_age = sample(25:60, 5, replace = TRUE),
    tot_cust_years = sample(1:20, 5, replace = TRUE),
    cc_acct_ind = as.factor(sample(0:1, 5, replace = TRUE))
  )

  predicted <- predict(model, newdata = test_data, type = "vote")
  scores <- data.frame(test_data$cust_id, predicted, test_data$cc_acct_ind)

  expect_equal(nrow(scores), 5)
  expect_equal(ncol(scores), 4)  # cust_id, Pred0, Pred1, cc_acct_ind
})
