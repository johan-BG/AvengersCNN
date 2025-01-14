import yaml
from pathlib import Path
import sys
from PyQt5.QtWidgets import QMessageBox,QApplication
import torch
from torchvision import transforms

from PyQt5.QtWidgets import QApplication, QMessageBox, QInputDialog
import sys

def import_transform(t_class):
    try:
        print("Transform loaded")
        transform=getattr(transforms,t_class)
        return transform()
    except AttributeError:
        raise ValueError(f"Transform '{t_class}' is not a valid torchvision transform.")

def ask_question(message, input_mode=False, input_label=None):
    '''
    Widget to ask a binary question or optionally get user input.
    
    Arguments:
        message: String to be displayed as question.
        input_mode: Boolean, if True, enables input field.
        input_label: String label for the input field (used in input mode).
    
    Example:
        ask_question(message="Do you want to proceed?")
        ask_question(message="Enter your name:", input_mode=True, input_label="Name:")
    '''
    # Setting app
    app = QApplication(sys.argv)
    
    # If input_mode is True, display an input dialog
    if input_mode:
        input_dialog = QInputDialog()
        input_dialog.setWindowTitle("Input Required")
        input_dialog.setLabelText(input_label if input_label else message)
        input_dialog.setOkButtonText("OK")
        input_dialog.setCancelButtonText("Cancel")
        
        if input_dialog.exec_() == QInputDialog.Accepted:
            user_input = input_dialog.textValue()
            print(f"User input: {user_input}")
            app.exit()
            return user_input  # Return the input provided by the user
        else:
            print("User cancelled the input.")
            app.exit()
            return None  # Return None if canceled

    # If not in input mode, display a binary question dialog
    else:
        # Creating widget
        message_box = QMessageBox()
        message_box.setWindowTitle("Question")
        message_box.setText(message)
        message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        result = message_box.exec_()
        
        # Closing app
        if result is not None:
            app.exit()
        
        # Checking response
        if result == QMessageBox.Yes:
            print("You chose Yes!")
            return True
        else:
            print("You chose No!")
            return False
        
    
#Save as YAML
def saveModel(model,modelname,class_names):
    '''
            To save a model as YAML
    Arguments:
        model:model to be saved
        modelname:name under which the model is saved
        
    Example:
        saveModel(model=CNN_Classifier
                  modelname="CNNmodel0.0",
                  class_names=data_classes)
    '''
    #Checking the file path
    filepath=Path("model")
    
    #Creating if it doesn't exist
    filepath.mkdir(parents=True,exist_ok=True)
    
    #Extracting model meta data
    device=next(model.parameters()).is_cuda
    metadata={"args":model.args,
            "state_dict":{keys: value.cpu().tolist() for keys,value in model.state_dict().items()},
            "classes":class_names,
            "device":device,
            "name":modelname}
    
    #Setting Message
    if filepath.exists():
        message=f"Do you want to overwrite the file {modelname}.yaml"
    else:
        message="Do you want to create a new file and save"
    
    #Saving model as YAML
    if ask_question(message):
        #Setting File name
        filepath=filepath/(modelname+".yaml")
        with open(filepath,"w") as yamlfile:
            yaml.dump(metadata,yamlfile,Dumper=yaml.CDumper)
        print(f"Model saved to {filepath}")
    elif filepath.exists():
        message="Do you want to create a new file?"
        if ask_question(message):
            modelname=ask_question(message,True,"New Model Name:")
            #Setting File name
            filepath=filepath/(modelname+".yaml")
            if modelname!=None:
                with open(filepath,"w") as yamlfile:
                    yaml.dump(metadata,yamlfile,Dumper=yaml.CDumper)
                print(f"Model saved to {filepath}")
                
# Load YAML file 
def loadModel(modelclass,modelname=None):
    if modelname!=None:
        modelname=Path(f"model\{modelname}")
        with open(f"{modelname}.yaml", "r") as yaml_file:
            model_metadata = yaml.load(yaml_file,Loader=yaml.CLoader)

    # Reconstruct the model
        state_dict = {key: torch.tensor(value) for key, value in model_metadata["state_dict"].items()}
        model=modelclass(**model_metadata["args"])
    else:
        model=modelclass(inunits=3,outunits=5,kernel=7,size=[1,128,128])
    
    #Loading and Device Agnostics 
    model.load_state_dict(state_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("Model loaded successfully")
    return model,model_metadata["classes"]

#Accuracy function
def acc(pred,test):
    '''
            function to calculate accuarcy of a model's prediction
            
        Arguments:
                pred:-predicted values of model
                test:-actual label of inputs
                
        Example code:
                for (X,y) in dataloader:
                    y_pred=model(X)
                    accuracy=acc(pred=y_pred,test=y)
                
    '''
    #finding index of class having max probability 
    pred=pred.argmax(dim=1)
    
    #finding number of correct prediction
    correct=torch.eq(pred,test).sum().item()
    
    #converting number of correct prediction to percentage
    acc=correct/len(test)*100
    
    #returns the accuracy percentage 
    return acc