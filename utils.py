import yaml
import torch
from pathlib import Path
from os import mkdir
def saveModel(model,modelname):
    filepath=Path("avengers/model")
    filepath.mkdir(parents=True,exist_ok=True)
    filepath=filepath/(modelname+".yaml")
    device=next(model.parameters()).is_cuda
    metadata={"args":model.args,
            "state_dict":{keys: value.cpu().tolist() for keys,value in model.state_dict().items()},
            "device":device,
            "name":modelname}
    with open(filepath,"w") as yamlfile:
        yaml.dump(metadata,yamlfile)
    print(f"Model saved to {filepath}")
saveModel(model,"AvengersCNN")
