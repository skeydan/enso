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
#library(ggthemes)

#purrr::walk(readLines("files"), function(f) download.file(url = f, destfile = basename(f)))
#read_stars("grb/sst189101.grb")

grb_dir <- "grb"
grb <-
  read_stars(file.path(grb_dir, map(readLines("files", warn = FALSE), basename)), along = "time") %>%
  st_set_dimensions(
    3,
    values = seq(as.Date("1891-01-01"), as.Date("2020-09-01"), by = "months"),
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


sst_train <- grb %>% filter(time < as.Date("2000-01-01"))
sst_train <- as.tbl_cube.stars(sst_train)$mets[[1]] 
sst_train <- sst_train + 273.15
quantile(sst_train, na.rm = TRUE)
train_mean <- mean(sst_train, na.rm = TRUE)
train_sd <- sd(sst_train, na.rm = TRUE)
sst_train <- (sst_train - train_mean) / train_sd
sst_train[is.na(sst_train)] <- 0
quantile(sst_train, na.rm = TRUE)
  
sst_valid <- grb %>% filter(time >= as.Date("2000-01-01"))
sst_valid <- as.tbl_cube.stars(sst_valid)$mets[[1]] 
sst_valid <- sst_valid + 273.15
sst_valid <- (sst_valid - train_mean) / train_sd
quantile(sst_valid, na.rm = TRUE)
sst_valid[is.na(sst_valid)] <- 0


nino <- read_table2("ONI_NINO34_1854-2020.txt", skip = 9) %>%
  mutate(month = as.Date(paste0(YEAR, "-", `MON/MMM`, "-01"))) %>%
  select(month, NINO34_MEAN, PHASE) %>%
  filter(between(month, as.Date("1891-01-01"), as.Date("2020-09-01")))
nrow(nino)

nino_train <- nino %>% filter(month < as.Date("2000-01-01"))
nino_valid <- nino %>% filter(month >= as.Date("2000-01-01"))

train_mean_nino <- mean(nino_train$NINO34_MEAN)
train_sd_nino <- sd(nino_train$NINO34_MEAN)
# nino_train <- nino_train %>% mutate(
#   NINO34_MEAN = scale(NINO34_MEAN, center = train_mean_nino, scale = train_sd_nino) 
# )
# nino_valid <- nino_valid %>% mutate(
#   NINO34_MEAN = scale(NINO34_MEAN, center = train_mean_nino, scale = train_sd_nino) 
# )


#ggplot(nino %>% filter(PHASE != "M"), aes(x = month, y = NINO34_MEAN, color = PHASE)) + geom_path(size = 0.5)

sst_valid %>% dim()
nino_valid %>% dim()

n_timesteps <- 6
batch_size <- 16

enso_dataset <- dataset(
  
  name = "enso_dataset",
  
  initialize = function(sst, nino, n_timesteps) {
   self$sst <- sst
   self$nino <- nino
   self$n_timesteps <- n_timesteps
  },
  
  .getitem = function(i) {
    x <- torch_tensor(self$sst[ , , i:(n_timesteps + i - 1)]) # (360, 180, n_timesteps)
    x <- torch_split(x, 1, dim = 3) # list of length n_timesteps of tensors (360, 180, 1)
    x <- torch_stack(x) # (n_timesteps, 360, 180, 1)
    x <- x$view(c(n_timesteps, 1, 360, 180))
    
    y1 <- torch_tensor(self$sst[ , , n_timesteps + i])$unsqueeze(1) # (1, 360, 180)
    y2 <- torch_tensor(self$nino$NINO34_MEAN[n_timesteps + i])
    list(x = x, y1 = y1, y2 = y2)
  },
  
  .length = function() {
    nrow(self$nino) - n_timesteps 
  }
  
)

valid_ds <- enso_dataset(sst_valid, nino_valid, n_timesteps)
length(valid_ds)
first <- valid_ds$.getitem(1)
first$x # 6,1,360,180
first$y1 # 1,360,180
first$y2 # 1

valid_dl <- valid_ds %>% dataloader(batch_size = batch_size)
length(valid_dl)

iter <- valid_dl$.iter()
first_batch <- iter$.next()
#first_batch

train_ds <-enso_dataset(sst_train, nino_train, n_timesteps)
length(train_ds)
train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)
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
    self$linear <- nn_linear(360 * 180 * n_timesteps, 128)
    self$output <- nn_linear(128, 1)
    
  },
  
  forward = function(x) {
    
    ret <- self$convlstm(x)
    layer_outputs <- ret[[1]]
    layer_last_states <- ret[[2]]
    next_sst <- layer_last_states[[self$n_layers]][[1]]
   
    flat <- torch_flatten(layer_outputs[[self$n_layers]], start_dim = 2)
    next_nino <- self$linear(flat) %>% nnf_relu() %>% self$output()
    list(next_sst, next_nino)
    
  }
    
)

device <- torch_device(if(cuda_is_available()) "cuda" else "cpu")
device <- "cpu"

net <- model(channels_in = 1,
             convlstm_hidden = c(16, 1),
             convlstm_kernel = c(3, 1),
             convlstm_layers = 2)

net <- net$to(device = device)


# train -------------------------------------------------------------------

optimizer <- optim_adam(net$parameters, lr = 0.001)

num_epochs <- 1

train_batch <- function(b) {
  
  optimizer$zero_grad()
  output <- net(b$x$to(device = device))
  
  sst_loss <- nnf_mse_loss(output[[1]], b$y1$to(device = device))
  nino_loss <- nnf_mse_loss(output[[2]], b$y2$to(device = device))
  
  if (i %% 40 == 0) {
    print(i)
    
    print(sst_loss$item())
    print(nino_loss$item())
    
    print(as.numeric(output[[2]]))
    print(as.numeric(b$y2))
  }
  
  i <<- i + 1
  
  loss <- sst_loss + nino_loss
  loss$backward()
  optimizer$step()
  
  list(sst_loss$item(), nino_loss$item(), loss$item())
  
}

valid_batch <- function(b) {
  
  output <- net(b$x$to(device = device))
  
  sst_loss <- nnf_mse_loss(output[[1]], b$y1$to(device = device))
  nino_loss <- nnf_mse_loss(output[[2]], b$y2$to(device = device))
  
  loss <- sst_loss + nino_loss

  list(sst_loss$item(), nino_loss$item(), loss$item())
  
}

for (epoch in 1:num_epochs) {
  
  net$train()
  train_loss_sst <- c()
  train_loss_nino <- c()
  train_loss <- c()
  
  i <<- 1

  for (b in enumerate(train_dl)) {
    
    losses <- train_batch(b)
    train_loss_sst <- c(train_loss_sst, losses[[1]])
    train_loss_nino <- c(train_loss_nino, losses[[2]])
    train_loss <- c(train_loss, losses[[3]])
    
    gc(full = TRUE)
  }

  torch_save(net, paste0("model_", epoch, ".pt"))

  cat(sprintf("\nEpoch %d, training: loss: %3.3f sst: %3.3f nino: %3.3f \n",
              epoch, mean(train_loss), mean(train_loss_sst), mean(train_loss_nino)))
  
  print(train_loss)
  print(train_loss_sst)
  print(train_loss_nino)
  
  net$eval()
  valid_loss_sst <- c()
  valid_loss_nino <- c()
  valid_loss <- c()

  for (b in enumerate(valid_dl)) {
    
    losses <- valid_batch(b)
    valid_loss_sst <- c(valid_loss_sst, losses[[1]])
    valid_loss_nino <- c(valid_loss_nino, losses[[2]])
    valid_loss <- c(valid_loss, losses[[3]])
    
    gc(full = TRUE)
    
  }
  
  cat(sprintf("\nEpoch %d, validation: loss: %3.3f sst: %3.3f nino: %3.3f \n",
              epoch, mean(valid_loss), mean(valid_loss_sst), mean(valid_loss_nino)))
  
  print(valid_loss)
  print(valid_loss_sst)
  print(valid_loss_nino)
  
}



# get predictions ---------------------------------------------------------

recursive_predict <- function(n_advance) {
  
  net$eval()
  
  for (b in enumerate(valid_dl)) {
    
    loss <- valid_batch(b)
    valid_loss <- c(valid_loss, loss)
    
    input <- b$x
    target <- b$y
    
    for (i in 1:n_advance) {
      
      preds <- net()
      
    }
    
  }
  
}




# https://holmdk.github.io/2020/04/02/video_prediction.html

# https://arxiv.org/abs/1511.06432
# Delving Deeper into Convolutional Networks for Learning Video Representations

# https://arxiv.org/pdf/1506.04214.pdf
# Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting

# https://www.researchgate.net/publication/317558562_Deep_Learning_for_Precipitation_Nowcasting_A_Benchmark_and_A_New_Model
# Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model



  