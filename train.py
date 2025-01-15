"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import torch
import engine, model_builder
from utils import loadModel,saveModel
from data_setup import ToCudaTransform,create_dataloaders
from torchvision import transforms

def main(args):

    # Setup hyperparameters
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001

    # Setup directories
    train_dir = "data"

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create transformation for image (if one doesn't exist)
    transform_list=args.transform
    if transform_list is not None:
        t_list=[transforms.Resize(size=(128,128))]
        t_list+=(args.transform+[transforms.ToTensor(),ToCudaTransform(device)])
        data_transform=transforms.Compose(t_list)
    else:
        data_transform =transforms.Compose([
    transforms.Resize(size=(128,128)),
    transforms.ToTensor(),
    ToCudaTransform(device=device)]
    )
        
    print("Transform loaded")

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names,class_dict = create_dataloaders(
        train_dir=train_dir,
        #test_dir=test_dir,
        device=device,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )
    
    if args!=None:
    # Create model with help from model_builder.py
        model,_ = loadModel(model_builder.CNNmodel,args.model)
    else:
        model,_=loadModel(model_builder.CNNmodel)
    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=LEARNING_RATE)
    
    NUM_EPOCHS=args.epoches
# Start training with help from engine.py
    engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)
# Save the model with help from utils.py
    saveModel(model=model,
                 modelname=args.model,
                 class_names=class_names)

if  __name__=="__main__":
    main()