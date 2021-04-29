#Bulk download brimstone butterfly images ----

#We will use the R folder created by Roy to speed up the image downloading and renaming sections 

install.packages("rinat")
install.packages("sf")
install.packages("dplyr")

library(rinat)
library(sf)
library(dplyr)

# Both download_images.R and gb_simple.RDS available on Canvas
source("download_images.R") 
gb_ll <- readRDS("gb_simple.RDS")

#if you look at the help section it is quite a flexible code, we will restrict our code to just 600 for now though
#also specifying that they must be of research quality only

#the downloads must be split in a usual 80:10:10 and into these three groups

#training dataset: this is used by the machine learning model to develop its weights
#validation dataset: during the training process, the outputs are continually compared with the validation dataset,
#and the weights updated to improve the fit
#test dataset: completely independent, to check model accuracy

#we also need to set the images into two folders: 
#An images folder, with subfolders for each species. Images will be downloaded into this folder from iNaturalist automatically split into the training and validation sets from here
#A test folder, with subfolders for each species. You will have to move the last 100 images for each species into here manually.

brimstone_recs <-  get_inat_obs(taxon_name  = "Gonepteryx rhamni",
                                bounds = gb_ll,
                                quality = "research",
                                # month=6,   # Month can be set.
                                # year=2018, # Year can be set.
                                maxresults = 600)

#we can also further filter this to hide the larvae, which are active in May-June

?filter()

download_images(spp_recs = brimstone_recs, spp_folder = "brimstone")

#Bulk download holly blue and orange tip butterfly records and images ---- 

#Repeating the steps for the aforementioned species 

# Holly blue; Celastrina argiolus
hollyblue_recs <-  get_inat_obs(taxon_name  = "Celastrina argiolus",
                                bounds = gb_ll,
                                quality = "research",
                                maxresults = 600)


# Orange tip; Anthocharis cardamines
orangetip_recs <-  get_inat_obs(taxon_name  = "Anthocharis cardamines",
                                bounds = gb_ll,
                                quality = "research",
                                maxresults = 600)

download_images(spp_recs = hollyblue_recs, spp_folder = "hollyblue")
download_images(spp_recs = orangetip_recs, spp_folder = "orangetip")

#Now all of the images have downloaded we can put them into different folders (as required) ----

image_files_path <- "images" # path to folder with photos

# list of spp to model; these names must match folder names
spp_list <- dir(image_files_path) # Automatically pick up names
#spp_list <- c("brimstone", "hollyblue", "orangetip") # manual entry

# number of spp classes (i.e. 3 species in this example)
output_n <- length(spp_list)

# Create test, and species sub-folders
for(folder in 1:output_n){
  dir.create(paste("test", spp_list[folder], sep="/"), recursive=TRUE)
}

# Now copy over spp_501.jpg to spp_600.jpg using two loops, deleting the photos
# from the original images folder after the copy
for(folder in 1:output_n){
  for(image in 501:600){
    src_image  <- paste0("images/", spp_list[folder], "/spp_", image, ".jpg")
    dest_image <- paste0("test/"  , spp_list[folder], "/spp_", image, ".jpg")
    file.copy(src_image, dest_image)
    file.remove(src_image)
  }
}

#Train up your deep learning model ----

#Initial setup will be the same as the CNN model we created for the first practical 

#remember to load keras though 

library(keras)

# image size to scale down to (original images vary by about 400 x 500 px)
img_width <- 150
img_height <- 150
target_size <- c(img_width, img_height)

# Full-colour Red Green Blue = 3 channels
channels <- 3

#Rescaling images and also defining proportion (20%)
# Rescale from 255 to between zero and 1
train_data_gen = image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

#Reading all the images from a folder

# training images
train_image_array_gen <- flow_images_from_directory(image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = spp_list,
                                                    subset = "training",
                                                    seed = 42)

# validation images
valid_image_array_gen <- flow_images_from_directory(image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = spp_list,
                                                    subset = "validation",
                                                    seed = 42)

# Check that things seem to have been read in OK
cat("Number of images per class:")

table(factor(train_image_array_gen$classes))

cat("Class labels vs index mapping")

train_image_array_gen$class_indices

#to look at one of the images 
plot(as.raster(train_image_array_gen[[1]][[1]][8,,,])) #bad quality image for me here

#Define additional parameters and configure model

# number of training samples
train_samples <- train_image_array_gen$n
# number of validation samples
valid_samples <- valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 32 # Useful to define explicitly as we'll use it later
epochs <- 10     # How long to keep training going for

# initialise model
model <- keras_model_sequential()

# add layers
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), input_shape = c(img_width, img_height, channels), activation = "relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), activation = "relu") %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n, activation = "softmax") 

print(model) #checking it's all set up properly

# Compile the model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

# Train the model with fit_generator
history <- model %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2
)

library(ggplot2) #to view the output in better quality 

#Assessing the accuracy and loss ----

plot(history) 

#Saving your model for future use ----

# The imager package also has a save.image function, so unload it to
# avoid any confusion
detach("package:imager", unload = TRUE)

# The save.image function saves your whole R workspace
save.image("animals.RData")

# Saves only the model, with all its weights and configuration, in a special
# hdf5 file on its own. You can use load_model_hdf5 to get it back.
#model %>% save_model_hdf5("animals_simple.hdf5")

load_model_hdf5

model %>% save_model_hdf5("animals_simple.hdf5")

#Doing all of this means we don't need to retrain the model over and over again 

#Testing your model ----
#Now we can test our model on an independent dataset 
#One difference is removing shuffle as we don't want it to happen with this model 

path_test <- "test"

test_data_gen <- image_data_generator(rescale = 1/255)

test_image_array_gen <- flow_images_from_directory(path_test,
                                                   test_data_gen,
                                                   target_size = target_size,
                                                   class_mode = "categorical",
                                                   classes = spp_list,
                                                   shuffle = FALSE, # do not shuffle the images around
                                                   batch_size = 1,  # Only 1 image at a time
                                                   seed = 123)

# Takes about 3 minutes to run through all the images
model %>% evaluate_generator(test_image_array_gen, 
                             steps = test_image_array_gen$n)

#A word of warning about unbalanced data (optional) ----

predictions <- model %>% 
  predict_generator(
    generator = test_image_array_gen,
    steps = test_image_array_gen$n
  ) %>% as.data.frame
colnames(predictions) <- spp_list

print(predictions) # looking at the vAlues in the table 

# Create 3 x 3 table to store data
confusion <- data.frame(matrix(0, nrow=3, ncol=3), row.names=spp_list)
colnames(confusion) <- spp_list

obs_values <- factor(c(rep(spp_list[1],100),
                       rep(spp_list[2], 100),
                       rep(spp_list[3], 100)))
pred_values <- factor(colnames(predictions)[apply(predictions, 1, which.max)])

install.packages("caret")
library(caret)
conf_mat <- confusionMatrix(data = pred_values, reference = obs_values)
conf_mat

#Making a prediction for a single image (optional) ----

# Original image
test_image_plt <- imager::load.image("test/hollyblue/spp_508.jpg")
plot(test_image_plt)

# Need to import slightly differently resizing etc. for Keras
test_image <- image_load("test/hollyblue/spp_508.jpg",
                         target_size = target_size)

test_image <- image_to_array(test_image)
test_image <- array_reshape(test_image, c(1, dim(test_image)))
test_image <- test_image/255

# Now make the prediction, and print out nicely
pred <- model %>% predict(test_image)
pred <- data.frame("Species" = spp_list, "Probability" = t(pred))
pred <- pred[order(pred$Probability, decreasing=T),][1:3,]
pred$Probability <- paste(round(100*pred$Probability,2),"%")
pred