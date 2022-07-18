# meteorml
Code I used for my machine learning projecct based on GMN(Gloabal Meteor Network Data)

To make use of the code and neural network model I made using the GMN data, all necessary scripts are in the "use" folder. All other folders are simply a record of the work that I have done.
To use the neural network you simply need to install tensorflow v2.4.1 in the vRMS environment as well as the dependencies, can be done using pip. Then you can use the overall.py file in the use folder, give it the full file path to the FTPdetect file and the corresponding directory path to the fits files. Then it will return a list of tuples, (prediction value, file name), the prediction is a probability value between 0 and 1, with 0 being not a meteor and 1 being a meteor. This list can be dumped into a csv if you uncomment the code at the end of the file.
Overall the space needed to install all the additional python packages and this code is ~2GB

The crop_and_convert folder contains scripts I used to tidy up the dataset and generate the png images from the fits files for training.
The keras folder contains the many different scripts I used to train neural networks on the data along with the performance of these neural network models, and additional scripts I used.
