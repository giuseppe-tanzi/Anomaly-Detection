import os

import mxnet as mx

import convLSTMAE
import utils

train_directory = "./UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/*/*"
test_directory = "./UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/"
output_test_directory = "./output/"
batch_size = 8
num_epochs = 2
ctx = mx.gpu()

# Get model
model = convLSTMAE.ConvLSTMAE()
model.hybridize()
params_file = "./parameters/autoencoder_ucsd_convLSTMAE.params"

# convolutional autoencoder with stacked frames and convolutional LSTMs (batch_size, 10, 227, 227)
# model, params_file = convLSTMAE.train(model, batch_size, ctx, num_epochs, path=train_directory)

try:
    os.mkdir(output_test_directory)
except:
    pass

for item in os.scandir(test_directory):
    if item.is_dir():
        path = test_directory + item.name + "/*"
        if not item.name.endswith("gt"):
            print(item.name)
            utils.plot_images(path, model, params_file, ctx, output_path=output_test_directory + item.name)
