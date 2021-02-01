
library(torch)
library(tidyverse)
library(stars)
library(viridis)
library(ggthemes)

torch_manual_seed(777)


# Data --------------------------------------------------------------------


# Input -------------------------------------------------------------------

# Source: https://ds.data.jma.go.jp/tcc/tcc/index.html

# Data processing details: https://ds.data.jma.go.jp/tcc/tcc/products/elnino/cobesst_doc.html

# File URLs obtainable here: https://ds.data.jma.go.jp/tcc/tcc/products/elnino/cobesst/cobe-sst.html

# save URLs and execute this once:
# purrr::walk(readLines("files"), function(f) download.file(url = f, destfile = basename(f)))


grb_dir <- "grb"

# The east-west grid points run eastward from 0.5ºE to 0.5ºW,
# while the north-south grid points run northward from 89.5ºS to 89.5ºN.

grb <-
  read_stars(file.path(grb_dir, map(readLines("files", warn = FALSE), basename)), along = "time") %>%
  st_set_dimensions(3,
                    values = seq(as.Date("1891-01-01"), as.Date("2020-12-01"), by = "months"),
                    names = "time")

# ggplot() +
#   geom_stars(data = grb %>% filter(between(time, as.Date("2020-01-01"), as.Date("2020-12-01"))), alpha = 0.8) +
#   facet_wrap("time") +
#   scale_fill_viridis() +
#   coord_equal() +
#   theme_map() +
#   theme(legend.position = "none") 
# 

# Target ------------------------------------------------------------------

# https://bmcnoldy.rsmas.miami.edu/tropics/oni/
# https://bmcnoldy.rsmas.miami.edu/tropics/oni/ONI_NINO34_1854-2020.txt

nino <- read_table2("ONI_NINO34_1854-2020.txt", skip = 9) %>%
  mutate(month = as.Date(paste0(YEAR, "-", `MON/MMM`, "-01"))) %>%
  select(month, NINO34_MEAN, PHASE) %>%
  filter(between(month, as.Date("1891-01-01"), as.Date("2020-08-01"))) %>%
  mutate(phase_code = as.numeric(as.factor(PHASE)))

nrow(nino)



# Preprocessing (input) ---------------------------------------------------

# just because those are not yet present in the target
sst <- grb %>% filter(time <= as.Date("2020-08-01"))

# use 0°–360° E and 55° S–60° N only
sst <- sst %>% filter(between(y,-55, 60))
dim(sst)
# 360, 115, 1560

sst_train <- sst %>% filter(time < as.Date("1990-01-01"))
sst_valid <- sst %>% filter(time >= as.Date("1990-01-01"))

sst_train <- as.tbl_cube.stars(sst_train)$mets[[1]]
sst_valid <- as.tbl_cube.stars(sst_valid)$mets[[1]]

sst_train <- sst_train + 273.15
quantile(sst_train, na.rm = TRUE)

train_mean <- mean(sst_train, na.rm = TRUE)
train_sd <- sd(sst_train, na.rm = TRUE)
sst_train <- (sst_train - train_mean) / train_sd
quantile(sst_train, na.rm = TRUE)
sst_train[is.na(sst_train)] <- 0
quantile(sst_train)

sst_valid <- sst_valid + 273.15
quantile(sst_valid, na.rm = TRUE)

sst_valid <- (sst_valid - train_mean) / train_sd
quantile(sst_valid, na.rm = TRUE)
sst_valid[is.na(sst_valid)] <- 0


# Preprocessing - target --------------------------------------------------


nino_train <- nino %>% filter(month < as.Date("1990-01-01"))
nino_valid <- nino %>% filter(month >= as.Date("1990-01-01"))

nino_train %>% group_by(phase_code, PHASE) %>% summarise(count = n(), avg = mean(NINO34_MEAN))
nino_valid %>% group_by(phase_code, PHASE) %>% summarise(count = n(), avg = mean(NINO34_MEAN))

train_mean_nino <- mean(nino_train$NINO34_MEAN)
train_sd_nino <- sd(nino_train$NINO34_MEAN)
nino_train <- nino_train %>% mutate(NINO34_MEAN = scale(NINO34_MEAN, center = train_mean_nino, scale = train_sd_nino))
nino_valid <- nino_valid %>% mutate(NINO34_MEAN = scale(NINO34_MEAN, center = train_mean_nino, scale = train_sd_nino))

sst_valid %>% dim()
nino_valid %>% dim()


# Torch dataset -----------------------------------------------------------


n_timesteps <- 6

enso_dataset <- dataset(
  name = "enso_dataset",
  
  initialize = function(sst, nino, n_timesteps) {
    self$sst <- sst
    self$nino <- nino
    self$n_timesteps <- n_timesteps
  },
  
  .getitem = function(i) {
    x <-
      torch_tensor(self$sst[, , i:(n_timesteps + i - 1)]) # (360, 115, n_timesteps)
    x <-
      torch_split(x, 1, dim = 3) # list of length n_timesteps of tensors (360, 115, 1)
    x <- torch_stack(x) # (n_timesteps, 360, 115, 1)
    x <- x$view(c(n_timesteps, 1, 360, 115))
    
    y1 <-
      torch_tensor(self$sst[, , n_timesteps + i])$unsqueeze(1) # (1, 360, 115)
    y2 <- torch_tensor(self$nino$NINO34_MEAN[n_timesteps + i])
    y3 <-
      torch_tensor(self$nino$phase_code[n_timesteps + i])$squeeze()$to(torch_long())
    list(x = x,
         y1 = y1,
         y2 = y2,
         y3 = y3)
  },
  
  .length = function() {
    nrow(self$nino) - n_timesteps
  }
  
)

train_ds <- enso_dataset(sst_train, nino_train, n_timesteps)
length(train_ds)

valid_ds <- enso_dataset(sst_valid, nino_valid, n_timesteps)
length(valid_ds)

first <- valid_ds$.getitem(1)
first$x # 6,1,360,115
first$y1 # 1,360,115
first$y2 # 1
first$y3 # 1


# dataloader --------------------------------------------------------------

batch_size <- 8

train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)

valid_dl <- valid_ds %>% dataloader(batch_size = batch_size/4)

iter <- dataloader_make_iter(valid_dl)
b <- dataloader_next(iter)
b$y1


# Model -------------------------------------------------------------------

source("../convlstm/convlstm.R")

model <- nn_module(
  
  initialize = function(channels_in,
                        convlstm_hidden,
                        convlstm_kernel,
                        convlstm_layers) {
    
    self$n_layers <- convlstm_layers
    
    self$convlstm <- convlstm(
      input_dim = channels_in,
      hidden_dims = convlstm_hidden,
      kernel_sizes = convlstm_kernel,
      n_layers = convlstm_layers
    )
    
    self$conv1 <-
      nn_conv2d(
        in_channels = 32,
        out_channels = 1,
        kernel_size = 5,
        padding = 2
      )
    
    self$conv2 <-
      nn_conv2d(
        in_channels = 32,
        out_channels = 32,
        kernel_size = 5,
        stride = 2
      )
    
    self$conv3 <-
      nn_conv2d(
        in_channels = 32,
        out_channels = 32,
        kernel_size = 5,
        stride = 3
      )
    
    self$linear <- nn_linear(33408, 64)
    
    self$b1 <- nn_batch_norm1d(num_features = 64)
    
    self$cont <- nn_linear(64, 128)
    self$cat <- nn_linear(64, 128)
    
    self$cont_output <- nn_linear(128, 1)
    self$cat_output <- nn_linear(128, 3)
    
  },
  
  forward = function(x) {
    
    ret <- self$convlstm(x)
    layer_last_states <- ret[[2]]
    last_hidden <- layer_last_states[[self$n_layers]][[1]]
    
    next_sst <- self$conv1(last_hidden)

    c2 <- last_hidden %>% self$conv2() 
    c3 <- c2 %>% self$conv3() 

    flat <- torch_flatten(c3, start_dim = 2)
    common <- self$linear(flat) %>% self$b1() %>% nnf_relu()

    next_temp <- common %>% self$cont() %>% nnf_relu() %>% self$cont_output()
    next_nino <- common %>% self$cat() %>% nnf_relu() %>% self$cat_output()
    
    list(next_sst, next_temp, next_nino)
    
  }
  
)

net <- model(
  channels_in = 1,
  convlstm_hidden = c(16, 16, 32),
  convlstm_kernel = c(3, 3, 5),
  convlstm_layers = 3
)

device <- torch_device(if (cuda_is_available())
  "cuda"
  else
    "cpu")

net <- net$to(device = device)
net

# train -------------------------------------------------------------------

optimizer <- optim_adam(net$parameters, lr = 0.001)

num_epochs <- 100

lw_sst <- 0.2
lw_temp <- 0.4
lw_nino <- 0.4


train_batch <- function(b) {
  optimizer$zero_grad()
  output <- net(b$x$to(device = device))
  
  sst_output <- output[[1]]
  sst_target <- b$y1$to(device = device)
  
  sst_loss <- nnf_mse_loss(sst_output[sst_target != 0], sst_target[sst_target != 0])
  temp_loss <- nnf_mse_loss(output[[2]], b$y2$to(device = device))
  nino_loss <- nnf_cross_entropy(output[[3]], b$y3$to(device = device))
  
  
  if ((i %% 1000 == 0)) {
    cat("\n")
    print(round(
      as.numeric(output[[2]]$to(device = "cpu")) * train_sd_nino + train_mean_nino,
      2
    ))
    print(round(as.numeric(b$y2$to(device = "cpu")) * train_sd_nino + train_mean_nino, 2))
    cat("\n")
    print(as.matrix(output[[3]]$to(device = "cpu")))
    print(as.numeric(b$y3$to(device = "cpu")))
    cat("\n")
  }
  
  i <<- i + 1
  
  loss <- lw_sst * sst_loss + lw_temp * temp_loss + lw_nino * nino_loss
  loss$backward()
  optimizer$step()
  
  train_pred_temp <<- c(train_pred_temp, as.numeric(output[[2]]$to(device = "cpu")))
  train_pred_nino <<- rbind(train_pred_nino, as.matrix(output[[3]]$to(device = "cpu")))
  
  gc()
  
  list(sst_loss$item(), temp_loss$item(), nino_loss$item(), loss$item())
  
}

valid_batch <- function(b) {
  
  output <- net(b$x$to(device = device))
  
  sst_output <- output[[1]]
  sst_target <- b$y1$to(device = device)
  
  sst_loss <- nnf_mse_loss(sst_output[sst_target != 0], sst_target[sst_target != 0])
  temp_loss <- nnf_mse_loss(output[[2]], b$y2$to(device = device))
  nino_loss <- nnf_cross_entropy(output[[3]], b$y3$to(device = device))
  
  if ((j %% 1000 == 0)) {
    cat("\n")
    print(round(
      as.numeric(output[[2]]$to(device = "cpu")) * train_sd_nino + train_mean_nino,
      2
    ))
    print(round(as.numeric(b$y2$to(device = "cpu")) * train_sd_nino + train_mean_nino, 2))
    cat("\n")
    print(as.matrix(output[[3]]$to(device = "cpu")))
    print(as.numeric(b$y3$to(device = "cpu")))
    cat("\n")
  }
  
  j <<- j + 1
  
  loss <- lw_sst * sst_loss + lw_temp * temp_loss + lw_nino * nino_loss
  
  valid_pred_temp <<- c(valid_pred_temp, as.numeric(output[[2]]$to(device = "cpu")))
  valid_pred_nino <<- rbind(valid_pred_nino, as.matrix(output[[3]]$to(device = "cpu")))
  
  gc()
  
  list(sst_loss$item(), temp_loss$item(), nino_loss$item(), loss$item())
  
}

for (epoch in 1:50) {
  
  net$train()
  
  train_loss_sst <- c()
  train_loss_temp <- c()
  train_loss_nino <- c()
  train_loss <- c()
  
  train_pred_temp <- c()
  train_pred_nino <- c()
  
  i <<- 1
  
  coro::loop(for (b in train_dl) {
    losses <- train_batch(b)
    train_loss_sst <- c(train_loss_sst, losses[[1]])
    train_loss_temp <- c(train_loss_temp, losses[[2]])
    train_loss_nino <- c(train_loss_nino, losses[[3]])
    train_loss <- c(train_loss, losses[[4]])
    
  })
  
  cat(
    sprintf(
      "\nEpoch %d, training: loss: %3.3f sst: %3.3f temp: %3.3f nino: %3.3f \n",
      epoch,
      mean(train_loss),
      mean(train_loss_sst),
      mean(train_loss_temp),
      mean(train_loss_nino)
    )
  )
  
  net$eval()
  
  valid_loss_sst <- c()
  valid_loss_temp <- c()
  valid_loss_nino <- c()
  valid_loss <- c()
  
  valid_pred_temp <- c()
  valid_pred_nino <- c()
  
  j <<- 1
  
  coro::loop(for (b in valid_dl) {
    losses <- valid_batch(b)
    valid_loss_sst <- c(valid_loss_sst, losses[[1]])
    valid_loss_temp <- c(valid_loss_temp, losses[[2]])
    valid_loss_nino <- c(valid_loss_nino, losses[[3]])
    valid_loss <- c(valid_loss, losses[[4]])
    
  })
  
  cat(
    sprintf(
      "\nEpoch %d, validation: loss: %3.3f sst: %3.3f temp: %3.3f nino: %3.3f \n",
      epoch,
      mean(valid_loss),
      mean(valid_loss_sst),
      mean(valid_loss_temp),
      mean(valid_loss_nino)
    )
  )
  
  
  torch_save(net, paste0(
    "model_",
    epoch,
    "_",
    round(mean(train_loss), 3),
    "_",
    round(mean(valid_loss), 3),
    ".pt"
  ))
  
  saveRDS(train_pred_temp, paste0("train_pred_temp_", epoch, ".rds"))
  saveRDS(train_pred_nino, paste0("train_pred_nino_", epoch, ".rds"))
  saveRDS(valid_pred_temp, paste0("valid_pred_temp_", epoch, ".rds"))
  saveRDS(valid_pred_nino, paste0("valid_pred_nino_", epoch, ".rds"))
  
}


# Saved predictions -------------------------------------------------------

valid_pred_nino <- readRDS("valid_pred_nino_50.rds") 
valid_pred_temp <- readRDS("valid_pred_temp_50.rds")

valid_perf <- data.frame(
  actual_temp = nino_valid$NINO34_MEAN[(batch_size + 1):nrow(nino_valid)] * train_sd_nino + train_mean_nino,
  actual_nino = factor(nino_valid$phase_code[(batch_size + 1):nrow(nino_valid)]),
  pred_temp = valid_pred_temp * train_sd_nino + train_mean_nino,
  pred_nino = factor(valid_pred_nino %>% apply(1, which.max))
)

yardstick::conf_mat(valid_perf, actual_nino, pred_nino)

valid_perf <- valid_perf %>% 
  select(actual = actual_temp, predicted = pred_temp) %>% 
  add_column(month = seq(as.Date("1990-07-01"), as.Date("2020-08-01"), by = "months")) %>%
  pivot_longer(-month, names_to = "Index", values_to = "temperature")

ggplot(valid_perf, aes(x = month, y = temperature, color = Index)) +
  geom_line() +
  scale_color_manual(values = c("#006D6F", "#B2FFFF")) +
  theme_classic()



# Generate Predictions -------------------------------------------------------------

batch_size <- 2

train_dl <- train_ds %>% dataloader(batch_size = batch_size)
valid_dl <- valid_ds %>% dataloader(batch_size = batch_size)

#dl <- valid_dl
dl <- valid_dl

device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
#device <- "cpu"

net <- torch_load("model_50_0.061_0.727.pt")
net <- net$to(device = device)
net$eval()

pred_index <- c()
pred_phase <- c()

coro::loop(for (b in dl) {

  #gc()

  output <- net(b$x$to(device = device))

  pred_index <<- c(pred_index, output[[2]]$to(device = "cpu"))
  pred_phase <<- rbind(pred_phase, as.matrix(output[[3]]$to(device = "cpu")))

})




# Papers and posts --------------------------------------------------------


# https://arxiv.org/abs/1511.06432
# Delving Deeper into Convolutional Networks for Learning Video Representations

# https://arxiv.org/pdf/1506.04214.pdf
# Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting

# https://www.researchgate.net/publication/317558562_Deep_Learning_for_Precipitation_Nowcasting_A_Benchmark_and_A_New_Model
# Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model

# https://www.climatechange.ai/papers/neurips2019/40/paper.pdf
# Forecasting El Niño with Convolutional andRecurrent Neural Networks

# https://www.researchgate.net/publication/335896498_Deep_learning_for_multi-year_ENSO_forecasts
# https://sshep.snu.ac.kr/indico/event/107/session/2/contribution/38/material/slides/0.pdf#
# https://www.nature.com/articles/s41586-019-1559-7
# Deep learning for multi-year ENSO forecasts

# https://holmdk.github.io/2020/04/02/video_prediction.html


