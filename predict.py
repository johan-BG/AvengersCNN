"""
Utility functions to make predictions.

Main reference for code creation: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set 
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse 
from typing import List, Tuple
from PIL import Image
from utils import loadModel,import_transform
from model_builder import CNNmodel
from data_setup import ToCudaTransform

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

#Set argparse
parser=argparse.ArgumentParser()
parser.add_argument("--model",
                    default="AvengersCNN",
                    help="model name to use for prediction filepath")
parser.add_argument("--image",
                    help="target image filepath to predict on")
parser.add_argument("--transform",
                    type=import_transform,
                    nargs='*',
                    help="input transform")
parser.add_argument("--device",
                    required=False,
                    help="Target device for computation")
args=parser.parse_args()
# Predict on a target image with a target model
# Function created in: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set
def pred_and_plot_image(
    model: str,
    image_path: str,
    transform_list = None,
    device: torch.device = device,
):
    """Predicts on a target image with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        image_path (str): Filepath to target image to predict on.
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to device.
    """

    # Open image
    img = Image.open(image_path)
    
    #Image size
    image_size=img.size

    # Create transformation for image (if one doesn't exist)
    if transform_list is not None:
        t_list=[transforms.Resize(size=(128,128))]
        t_list+=(args.transform+[transforms.ToTensor(),ToCudaTransform(device)])
        image_transform=transforms.Compose(t_list)
    else:
        image_transform =transforms.Compose([
    transforms.Resize(size=(128,128)),
    transforms.ToTensor(),
    ToCudaTransform(device=device)]
)
    
    
    ### Predict on image ###

    # Make sure the model is on the target device
    model,class_list=loadModel(CNNmodel,model)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {class_list[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False)
    plt.show()
if __name__=="__main__":
    pred_and_plot_image(model=args.model,image_path=args.image,transform_list=args.transform)
