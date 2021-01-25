# https://ds.data.jma.go.jp/tcc/tcc/products/elnino/cobesst/cobe-sst.html

# The east-west grid points run eastward from 0.5ºE to 0.5ºW,
# while the north-south grid points run northward from 89.5ºS to 89.5ºN.

# https://ds.data.jma.go.jp/tcc/tcc/library/MRCS_SV12/index_e.htm

# https://www.climatechange.ai/CameraReadySubmissions%202-119/119/CameraReadySubmission/Forecasting_El_Nino_with_Convolutional_and_Recurrent_Neural_Networks(1).pdf
# 
# https://sshep.snu.ac.kr/indico/event/107/session/2/contribution/38/material/slides/0.pdf
# 
# https://www.nature.com/articles/s41586-019-1559-7
# 
# https://meso.nju.edu.cn/njdx/DFS//file/2019/11/26/201911261121480744epw3l.pdf?iid=6240

library(torch)
library(purrr)
library(stars)
library(readr)
library(dplyr)
library(ggplot2)
library(viridis)
library(ggthemes)

torch_manual_seed(777)

#purrr::walk(readLines("files"), function(f) download.file(url = f, destfile = basename(f)))
#read_stars("grb/sst189101.grb")

grb_dir <- "grb"

grb <-
  read_stars(file.path(grb_dir, map(readLines("files", warn = FALSE), basename)), along = "time") %>%
  st_set_dimensions(
    3,
    values = seq(as.Date("1891-01-01"), as.Date("2020-08-01"), by = "months"),
    names = "time"
  ) 

# ggplot() +  
#   geom_stars(data = grb, alpha = 0.8) + 
#   facet_wrap("time") +
#   scale_fill_viridis() +
#   coord_equal() +
#   theme_map() +
#   theme(legend.position = "bottom") +
#   theme(legend.key.width = unit(2, "cm"))

# use 0°–360° E and 55° S–60° N only?
sst <- grb %>% filter(between(y, -55, 60))
dim(sst)
# 360, 115, 1556

sst_train <- sst %>% filter(time < as.Date("2000-01-01"))
sst_train <- as.tbl_cube.stars(sst_train)$mets[[1]] 
sst_train <- sst_train + 273.15
quantile(sst_train, na.rm = TRUE)
train_mean <- mean(sst_train, na.rm = TRUE)
train_sd <- sd(sst_train, na.rm = TRUE)
sst_train <- (sst_train - train_mean) / train_sd
quantile(sst_train, na.rm = TRUE)
sst_train[is.na(sst_train)] <- 0
quantile(sst_train)
  
sst_valid <- sst %>% filter(time >= as.Date("2000-01-01"))
sst_valid <- as.tbl_cube.stars(sst_valid)$mets[[1]] 
sst_valid <- sst_valid + 273.15
sst_valid <- (sst_valid - train_mean) / train_sd
quantile(sst_valid, na.rm = TRUE)
sst_valid[is.na(sst_valid)] <- 0

#sst_valid[,,1] %>% as.matrix() %>% image()

nino <- read_table2("ONI_NINO34_1854-2020.txt", skip = 9) %>%
  mutate(month = as.Date(paste0(YEAR, "-", `MON/MMM`, "-01"))) %>%
  select(month, NINO34_MEAN, PHASE) %>%
  filter(between(month, as.Date("1891-01-01"), as.Date("2020-08-01"))) %>%
  mutate(phase_code = as.numeric(as.factor(PHASE)))
  
nrow(nino)

nino_train <- nino %>% filter(month < as.Date("2000-01-01"))
nino_valid <- nino %>% filter(month >= as.Date("2000-01-01"))

nino_train %>% group_by(phase_code, PHASE) %>% summarise(count = n(), avg = mean(NINO34_MEAN))
nino_valid %>% group_by(phase_code, PHASE) %>% summarise(count = n(), avg = mean(NINO34_MEAN))


train_mean_nino <- mean(nino_train$NINO34_MEAN)
train_sd_nino <- sd(nino_train$NINO34_MEAN)
nino_train <- nino_train %>% mutate(
  NINO34_MEAN = scale(NINO34_MEAN, center = train_mean_nino, scale = train_sd_nino)
)
nino_valid <- nino_valid %>% mutate(
  NINO34_MEAN = scale(NINO34_MEAN, center = train_mean_nino, scale = train_sd_nino)
)


#ggplot(nino %>% filter(PHASE != "M"), aes(x = month, y = NINO34_MEAN, color = PHASE)) + geom_path(size = 0.5)

sst_valid %>% dim()
nino_valid %>% dim()

n_timesteps <- 6
batch_size <- 8

enso_dataset <- dataset(
  
  name = "enso_dataset",
  
  initialize = function(sst, nino, n_timesteps) {
   self$sst <- sst
   self$nino <- nino
   self$n_timesteps <- n_timesteps
  },
  
  .getitem = function(i) {
    x <- torch_tensor(self$sst[ , , i:(n_timesteps + i - 1)]) # (360, 115, n_timesteps)
    x <- torch_split(x, 1, dim = 3) # list of length n_timesteps of tensors (360, 115, 1)
    x <- torch_stack(x) # (n_timesteps, 360, 115, 1)
    x <- x$view(c(n_timesteps, 1, 360, 115))
    
    y1 <- torch_tensor(self$sst[ , , n_timesteps + i])$unsqueeze(1) # (1, 360, 115)
    y2 <- torch_tensor(self$nino$NINO34_MEAN[n_timesteps + i])
    y3 <- torch_tensor(self$nino$phase_code[n_timesteps + i])$squeeze()$to(torch_long())
    list(x = x, y1 = y1, y2 = y2, y3 = y3)
  },
  
  .length = function() {
    nrow(self$nino) - n_timesteps 
  }
  
)

valid_ds <- enso_dataset(sst_valid, nino_valid, n_timesteps)
length(valid_ds)
first <- valid_ds$.getitem(1)
first$x # 6,1,360,115
first$y1 # 1,360,115
first$y2 # 1
first$y3 # 1

valid_dl <- valid_ds %>% dataloader(batch_size = batch_size, drop_last = TRUE)
length(valid_dl)

iter <- valid_dl$.iter()
first_batch <- iter$.next()
first_batch$y1

train_ds <-enso_dataset(sst_train, nino_train, n_timesteps)
length(train_ds)
train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE, drop_last = TRUE)
length(train_dl)


# Model -------------------------------------------------------------------

source("../convlstm/convlstm.R")

model <- nn_module(
  
  initialize = function(channels_in,
                        convlstm_hidden,
                        convlstm_kernel,
                        convlstm_layers) {
    
    self$n_layers <- convlstm_layers
    self$convlstm <- convlstm(input_dim = channels_in,
                              hidden_dims = convlstm_hidden,
                              kernel_sizes = convlstm_kernel,
                              n_layers = convlstm_layers
                              )
    self$conv1 <- nn_conv2d(in_channels = 32, out_channels = 1, kernel_size = 5, padding = 2)
    
    self$conv2 <- nn_conv2d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 2)
    
    self$conv3 <- nn_conv2d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 3)
 
    self$linear <- nn_linear(33408, 64)
    self$cont <- nn_linear(64, 128)
    self$cat <- nn_linear(64, 128)
    self$cont_output <- nn_linear(128, 1)
    self$cat_output <- nn_linear(128, 3)
    
  },
  
  forward = function(x) {
    
    
    
    ret <- self$convlstm(x)
    layer_last_states <- ret[[2]]
    last_hidden <- layer_last_states[[self$n_layers]][[1]]
    
    gc()
    
    next_sst <- self$conv1(last_hidden)
    
    c2 <- self$conv2(last_hidden)
    c3 <- self$conv3(c2)
    
    flat <- torch_flatten(c3, start_dim = 2)
    
    gc()
    
    common <- self$linear(flat) %>% nnf_relu()
    
    next_temp <- common %>% self$cont() %>% nnf_relu() %>% self$cont_output()
    next_nino <- common %>% self$cat() %>% nnf_relu() %>% self$cat_output()
    
    gc()
    
    list(next_sst, next_temp, next_nino)
    
  }
    
)

device <- torch_device(if(cuda_is_available()) "cuda" else "cpu")

net <- model(channels_in = 1,
             convlstm_hidden = c(16, 16, 32),
             convlstm_kernel = c(3, 3, 5),
             convlstm_layers = 3)

net(first_batch$x)
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
  
  
  if ((i %% 50 == 0)) {
    
    cat("\n")
    print(round(as.numeric(output[[2]]$to(device = "cpu")) * train_sd_nino + train_mean_nino, 2))
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
  
  
  if ((j %% 30 == 0)) {
    
    cat("\n")
    print(round(as.numeric(output[[2]]$to(device = "cpu")) * train_sd_nino + train_mean_nino, 2))
    print(round(as.numeric(b$y2$to(device = "cpu")) * train_sd_nino + train_mean_nino, 2))
    cat("\n")
    print(as.matrix(output[[3]]$to(device = "cpu")))
    print(as.numeric(b$y3$to(device = "cpu")))
    cat("\n")
  }
  
  j <<- j + 1
  
  loss <- lw_sst * sst_loss + lw_temp * temp_loss + lw_nino * nino_loss

  gc()
  
  list(sst_loss$item(), temp_loss$item(), nino_loss$item(), loss$item())
  
}

for (epoch in 1:num_epochs) {
  
  net$train()
  
  train_loss_sst <- c()
  train_loss_temp <- c()
  train_loss_nino <- c()
  train_loss <- c()
  
  i <<- 1
  
  coro::loop(for(b in train_dl) {
    
    losses <- train_batch(b)
    train_loss_sst <- c(train_loss_sst, losses[[1]])
    train_loss_temp <- c(train_loss_temp, losses[[2]])
    train_loss_nino <- c(train_loss_nino, losses[[3]])
    train_loss <- c(train_loss, losses[[4]])
    
  })
  
  torch_save(net, paste0("model_", epoch, ".pt"))

  cat(sprintf("\nEpoch %d, training: loss: %3.3f sst: %3.3f temp: %3.3f nino: %3.3f \n",
              epoch, mean(train_loss), mean(train_loss_sst), mean(train_loss_temp), mean(train_loss_nino)))
  
  net$eval()

  valid_loss_sst <- c()
  valid_loss_temp <- c()
  valid_loss_nino <- c()
  valid_loss <- c()

  j <<- 1
  
  coro::loop(for(b in valid_dl) {
    
      losses <- valid_batch(b)
      valid_loss_sst <- c(valid_loss_sst, losses[[1]])
      valid_loss_temp <- c(valid_loss_temp, losses[[2]])
      valid_loss_nino <- c(valid_loss_nino, losses[[3]])
      valid_loss <- c(valid_loss, losses[[4]])
    
  })

  cat(sprintf("\nEpoch %d, validation: loss: %3.3f sst: %3.3f temp: %3.3f nino: %3.3f \n",
              epoch, mean(valid_loss), mean(valid_loss_sst), mean(valid_loss_temp), mean(valid_loss_nino)))
  
}



# get predictions ---------------------------------------------------------

# recursive_predict <- function(n_advance) {
#   
#   net$eval()
#   
#   for (b in enumerate(valid_dl)) {
#     
#     loss <- valid_batch(b)
#     valid_loss <- c(valid_loss, loss)
#     
#     input <- b$x
#     target <- b$y
#     
#     for (i in 1:n_advance) {
#       
#       preds <- net()
#       
#     }
#     
#   }
#   
# }




# https://holmdk.github.io/2020/04/02/video_prediction.html

# https://arxiv.org/abs/1511.06432
# Delving Deeper into Convolutional Networks for Learning Video Representations

# https://arxiv.org/pdf/1506.04214.pdf
# Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting

# https://www.researchgate.net/publication/317558562_Deep_Learning_for_Precipitation_Nowcasting_A_Benchmark_and_A_New_Model
# Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model



  