# CANIDVisionDevelopment environment for the machine learning tools used in CANID. 
As of now, the running enviornment is slated to be just:
The models folder, forward.py, models.py, and workspace.py, 
Intentions are as follows:
/data is where all development source data goes. 
    /dev is general purpose datasets used for model validation e.g. UCF101 for action recog, etc.
    /test is where validation images/videos go, for testing models on specifc inputs
    /train is where all custom datasets go, including facial images, specific actions, etc.
/models is where all trained models get saved to 
dataHandler.py is how primarily training data is parsed and loaded into datasets for the NNs to use
forward.py is where most of the main running stuff happens, used for testing validation and eventually deployment
models.py is where the various configurations of NNs will be stored, and accessed when they need to be set up. Hyperparams can also be found here
train.py is where models get run through training
utils.py includes useful functions like for saving models, loading models, training analysis, etc.
workspace.py is the main running environment of this development space. Used for calling the various functions of the network. Eventually, the contents of workspace.py will be part of what gets run during each CANID system loop. 
