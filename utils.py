import yaml
from pathlib import Path
import sys
from PyQt5.QtWidgets import QMessageBox,QApplication

def ask_question(message):
    app=QApplication(sys.argv)
    message_box = QMessageBox()
    message_box.setWindowTitle("Question")
    message_box.setText(message)
    message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    result = message_box.exec_()
    if result!=None:
        app.exit()
    if result == QMessageBox.Yes:
        print("You chose Yes!")
        return True
    else:
        print("You chose No!")
        return False
def saveModel(model,modelname):
    filepath=Path("avengers/AvengersCNN/model")
    filepath.mkdir(parents=True,exist_ok=True)
    filepath=filepath/(modelname+".yaml")
    device=next(model.parameters()).is_cuda
    metadata={"args":model.args,
            "state_dict":{keys: value.cpu().tolist() for keys,value in model.state_dict().items()},
            "device":device,
            "name":modelname}
    if filepath.exists():
        message=f"Do you want to overwrite the file {modelname}.yaml"
    else:
        message="Do you want to create a new file and save"
    if ask_question(message):
        with open(filepath,"w") as yamlfile:
            yaml.dump(metadata,yamlfile)
        print(f"Model saved to {filepath}")