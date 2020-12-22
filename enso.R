# https://ds.data.jma.go.jp/tcc/tcc/products/elnino/cobesst/cobe-sst.html

# The east-west grid points run eastward from 0.5ºE to 0.5ºW,
# while the north-south grid points run northward from 89.5ºS to 89.5ºN.

# use 0°–360° E and 55° S–60° N only?

# https://ds.data.jma.go.jp/tcc/tcc/library/MRCS_SV12/index_e.htm

# https://www.climatechange.ai/CameraReadySubmissions%202-119/119/CameraReadySubmission/Forecasting_El_Nino_with_Convolutional_and_Recurrent_Neural_Networks(1).pdf
# 
# https://sshep.snu.ac.kr/indico/event/107/session/2/contribution/38/material/slides/0.pdf
# 
# https://www.nature.com/articles/s41586-019-1559-7
# 
# https://meso.nju.edu.cn/njdx/DFS//file/2019/11/26/201911261121480744epw3l.pdf?iid=6240


library(purrr)
library(stars)
library(readr)
library(dplyr)
library(ggplot2)
library(viridis)
library(ggthemes)

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
grb_train[is.na(grb_train)] <- 0
  
grb_valid <- grb %>% filter(time >= as.Date("2000-01-01"))
grb_valid <- as.tbl_cube.stars(grb_valid)$mets[[1]] 
grb_valid <- grb_valid + 273.15
quantile(grb_valid, na.rm = TRUE)
grb_valid[is.na(grb_valid)] <- 0


nino <- read_table("ONI_NINO34_1854-2020.txt", skip = 9) %>%
  mutate(month = as.Date(paste0(YEAR, "-", `MON/MMM`, "-01"))) %>%
  select(month, NINO34_MEAN, PHASE) %>%
  filter(between(month, as.Date("1891-01-01"), as.Date("2020-09-01")))
nrow(nino)

nino_train <- nino %>% filter(month < as.Date("2000-01-01"))
nino_valid <- nino %>% filter(month >= as.Date("2000-01-01"))

#ggplot(nino %>% filter(PHASE != "M"), aes(x = month, y = NINO34_MEAN, color = PHASE)) + geom_path(size = 0.5)

tmp <- as.tbl_cube.stars(grb_valid)$mets[[1]][ , , 1]

library(torch)

n_timesteps <- 6

enso_dataset <- dataset(
  
  name = "enso_dataset",
  
  initialize = function(ssts, nino34, n_timesteps) {
   self$x <- as.tbl_cube.stars(ssts)$mets[[1]] 
   self$y <- nino34
   self$n_timesteps <- n_timesteps
  },
  
  .getitem = function(i) {
    x <- torch_tensor(self$x[ , , i:(n_timesteps + i - 1)])
    x <- torch_split(x, 1, dim = 3)
    x <- torch_stack(x)
    x <- x$view(c(n_timesteps, 1, 360, 180))
    
    y <- torch_tensor(self$y$NINO34_MEAN[i + n_timesteps - 1])
    list(x = x, y = y)
  },
  
  .length = function() {
    nrow(self$y) - n_timesteps + 1
  }
  
)

valid_ds <- enso_dataset(grb_valid, nino_valid, n_timesteps)
length(valid_ds)
first <- valid_ds$.getitem(1)
first$x
first$y

batch_size <- 8
valid_dl <- valid_ds %>% dataloader(batch_size = batch_size)
length(valid_dl)

iter <- valid_dl$.iter()
b <- iter$.next()
b

train_ds <-enso_dataset(grb_train, nino_train, n_timesteps)
train_ds$.length()
train_dl <- train_ds %>% dataloader(batch_size = 8, shuffle = TRUE)
train_dl$.length()


# Model -------------------------------------------------------------------

source("../convlstm/convlstm.R")

model <- convlstm(input_dim = 1, hidden_dims = c(5, 1), kernel_sizes = rep(3, 2), n_layers = 2)
c(layer_outputs, layer_last_states) %<-% model(b$x)



# https://holmdk.github.io/2020/04/02/video_prediction.html


# https://arxiv.org/abs/1511.06432
# Delving Deeper into Convolutional Networks for Learning Video Representations

# https://arxiv.org/pdf/1506.04214.pdf
# Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting

# https://www.researchgate.net/publication/317558562_Deep_Learning_for_Precipitation_Nowcasting_A_Benchmark_and_A_New_Model
# Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model



  