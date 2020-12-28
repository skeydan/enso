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


grb_train <- grb %>% filter(time < as.Date("2000-01-01"))
grb_train <- as.tbl_cube.stars(grb_train)$mets[[1]] 
grb_train <- grb_train + 273.15
quantile(grb_train, na.rm = TRUE)
train_mean <- mean(grb_train, na.rm = TRUE)
train_sd <- sd(grb_train, na.rm = TRUE)
grb_train <- (grb_train - train_mean) / train_sd
grb_train[is.na(grb_train)] <- 0
quantile(grb_train, na.rm = TRUE)
  
grb_valid <- grb %>% filter(time >= as.Date("2000-01-01"))
grb_valid <- as.tbl_cube.stars(grb_valid)$mets[[1]] 
grb_valid <- grb_valid + 273.15
grb_valid <- (grb_valid - train_mean) / train_sd
quantile(grb_valid, na.rm = TRUE)
grb_valid[is.na(grb_valid)] <- 0


nino <- read_table2("ONI_NINO34_1854-2020.txt", skip = 9) %>%
  mutate(month = as.Date(paste0(YEAR, "-", `MON/MMM`, "-01"))) %>%
  select(month, NINO34_MEAN, PHASE) %>%
  filter(between(month, as.Date("1891-01-01"), as.Date("2020-09-01")))
nrow(nino)

nino_train <- nino %>% filter(month < as.Date("2000-01-01"))
nino_valid <- nino %>% filter(month >= as.Date("2000-01-01"))

#ggplot(nino %>% filter(PHASE != "M"), aes(x = month, y = NINO34_MEAN, color = PHASE)) + geom_path(size = 0.5)

grb_valid %>% dim()
nino_valid %>% dim()

n_timesteps <- 6
batch_size <- 16

enso_dataset <- dataset(
  
  name = "enso_dataset",
  
  initialize = function(sst, nino34, n_timesteps) {
   self$x <- sst
   self$y <- nino34
   self$n_timesteps <- n_timesteps
  },
  
  .getitem = function(i) {
    x <- torch_tensor(self$x[ , , i:(n_timesteps + i - 1)]) # (360, 180, n_timesteps)
    x <- torch_split(x, 1, dim = 3) # list of length n_timesteps of tensors (360, 180, 1)
    x <- torch_stack(x) # (n_timesteps, 360, 180, 1)
    x <- x$view(c(n_timesteps, 1, 360, 180))
    
    y <- torch_tensor(self$y$NINO34_MEAN[(i + 1):(n_timesteps + i)])
    list(x = x, y = y)
  },
  
  .length = function() {
    nrow(self$y) - n_timesteps 
  }
  
)

valid_ds <- enso_dataset(grb_valid, nino_valid, n_timesteps)
length(valid_ds)
first <- valid_ds$.getitem(1)
first$x
first$y

valid_dl <- valid_ds %>% dataloader(batch_size = batch_size)
length(valid_dl)

iter <- valid_dl$.iter()
first_batch <- iter$.next()
#first_batch

train_ds <-enso_dataset(grb_train, nino_train, n_timesteps)
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
    self$output <- nn_linear(360 * 180, 1)
    
  },
  
  forward = function(x) {
    
    c(layer_outputs, layer_last_states) %<-% self$convlstm(x)
    flat <- torch_flatten(layer_outputs[[self$n_layers]], start_dim = 3)$squeeze(2)
    self$output(flat)$squeeze(3)
    
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

num_epochs <- 20

train_batch <- function(b) {
  
  print(i)
  
  optimizer$zero_grad()
  output <- net(b$x$to(device = device))
  target <- b$y$to(device = device)
  
  loss <-  nnf_mse_loss(output, target)
  
  if (i %% 200 == 0) {
    print(as.matrix(output))
    print(as.matrix(target))
    print(loss$item())
  }
  
  i <<- i + 1
  
  loss$backward()
  optimizer$step()
  
  gc(full = TRUE)
  
  loss$item()
  
}

valid_batch <- function(b) {
  
  output <- net(b$x$to(device = device))
  target <- b$y$to(device = device)
  
  loss <-  nnf_mse_loss(output, target)
  
  gc(full = TRUE)
  
  loss$item()
  
}

for (epoch in 1:num_epochs) {
  
  net$train()
  train_loss <- c()
  
  i <- 1

  for (b in enumerate(train_dl)) {
    loss <- train_batch(b)
    train_loss <- c(train_loss, loss)
  }

  torch_save(net, paste0("model_", epoch, ".pt"))

  cat(sprintf("\nEpoch %d, training: loss:%3f\n", epoch, mean(train_loss)))
  
  net$eval()
  valid_loss <- c()

  for (b in enumerate(valid_dl)) {
    
    loss <- valid_batch(b)
    valid_loss <- c(valid_loss, loss)
    
  }
  
  cat(sprintf("\nEpoch %d, validation: loss:%3f\n", epoch, mean(valid_loss)))
}




# https://holmdk.github.io/2020/04/02/video_prediction.html

# https://arxiv.org/abs/1511.06432
# Delving Deeper into Convolutional Networks for Learning Video Representations

# https://arxiv.org/pdf/1506.04214.pdf
# Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting

# https://www.researchgate.net/publication/317558562_Deep_Learning_for_Precipitation_Nowcasting_A_Benchmark_and_A_New_Model
# Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model



  