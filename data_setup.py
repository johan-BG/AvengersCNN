"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split
import torch

NUM_WORKERS = 0



import torch
import torchvision.transforms as transforms

class ToCudaTransform:
    """
    A class that transforms input data to a specified device (e.g., CUDA), 
    and applies an optional operation on the data before moving it to the device.

    Attributes:
    device (str or torch.device): The device to which the tensor should be moved. 
                                  This can be either 'cuda' or 'cpu'.
    op (callable, optional): An optional operation to apply to the input data 
                              before moving it to the specified device.
    """

    def __init__(self, device="cuda", op=None):
        """
        Initializes the ToCudaTransform class.

        Args:
        device (str or torch.device): The device to which the tensor should be moved. 
                                      This can be either 'cuda' or 'cpu'.
        op (callable, optional): An optional operation to apply to the input data before 
                                  moving it to the device. Default is None.

        Example:
        transform = ToCudaTransform(device='cuda', op=some_function)
        """
        self.device = device  # Device (e.g., 'cuda' or 'cpu') to move the tensor to
        self.op = op  # Optional operation to apply to the input before moving

    def __call__(self, x):
        """
        Applies the optional operation (if any) to the input and then moves the input 
        to the specified device (CUDA or CPU).

        Args:
        x (any): The input data to be transformed. Can be a list, numpy array, or tensor.

        Returns:
        torch.Tensor: The input data after applying the optional operation (if any) and 
                      moved to the specified device (e.g., 'cuda' or 'cpu').

        Example:
        tensor_on_device = transform(input_data)
        """
        # Apply the optional operation if it is provided (example: rotation)
        if self.op == 0:
            # Example operation (rotate image)
            x = transforms.rotate(img=x, angle=20)

        # If the input is not a tensor, convert it to one
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        # Move the tensor to the specified device (e.g., CUDA or CPU)
        return x.to(self.device)

def create_dataloaders(
    train_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    device: torch.device,
    num_workers: int = NUM_WORKERS,
    test_dir: str = None
):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir (str): Path to training directory.
        test_dir (str, optional): Path to testing directory.
        transform (transforms.Compose): torchvision transforms to perform on training and testing data.
        batch_size (int): Number of samples per batch in each of the DataLoaders.
        num_workers (int, optional): An integer for number of workers per DataLoader. Default is NUM_WORKERS.

    Returns:
        tuple: A tuple of (train_dataloader, test_dataloader, class_names, class_dict).
            Where class_names is a list of the target classes.
        
    Example usage:
        train_dataloader, test_dataloader, class_names = 
            create_dataloaders(train_dir=path/to/train_dir,
                               test_dir=path/to/test_dir,
                               transform=some_transform,
                               device="cuda",
                               batch_size=32,
                               num_workers=4)
    """
    
    # Use ImageFolder to create dataset(s)
    fulldata = datasets.ImageFolder(
        root=train_dir,
        transform=transform,
        target_transform=ToCudaTransform(device)
    )
    
    # Split data into train and test if test_dir is not provided
    if test_dir is None:
        part1 = int(len(fulldata) * 0.8)
        part2 = len(fulldata) - part1
        train_data, test_data = random_split(fulldata, [part1, part2])
    else:
        train_data = fulldata
        test_data = datasets.ImageFolder(
            root=test_dir, 
            transform=transform,
            target_transform=ToCudaTransform(device)
        )
    
    # Get class names as list and dictionary
    class_dict = fulldata.class_to_idx
    class_names = fulldata.classes

    # Setting pin memory for DataLoader
    pin_memory =not torch.cuda.is_available()

    # Create train DataLoader
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Create test DataLoader
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print("data loaded")

    return train_dataloader, test_dataloader, class_names, class_dict