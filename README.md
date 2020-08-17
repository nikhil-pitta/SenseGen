# SenseGen
Implementation of [GitHub](https://arxiv.org/abs/1701.08886). Adapted from [this repo](https://github.com/nesl/sensegen). 

### Details
The repo consists of a data set folder along with 5 other python files: 
- data_utils.py is used to parse the given data
- model.py and model_utils.py are used to create the machine learning model
- train.py is used to train the model
- test.py is used to test the model

### Training
The model will be trained on the data in `/dataset`.
train.py has two key variables:
- num_epochs: The number of epochs the machine learning model will be trained on
- save_after: The model will be saved after every save_after epochs.

The models will be stored in `/models` under the root directory.  
- When a model is saved, 3 files will be generated
It is recommended that the model is trained for at least 1000 epochs to see viable results.  

### Testing
test.py has two key variables as well:
- import_path = the path to the .meta file in order to first load the model
- ckpt_path = the path to the actual model

These variables will have a number in them, for example:  
`./models/mdnmodel.ckpt-1.meta`  
Replace the numbers in both import_path and ckpt_path with the highest number (should be the same number) in `/models` to use the most recent save  

After the model is loaded, an example of real data vs the GAN generated data is displayed on screen with matplotlib
