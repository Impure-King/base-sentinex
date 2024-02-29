import jax
from jaxtyping import Array

# Handling exports:

__all__ = ["is_device_available", "list_devices", "set_devices", "device_put"]

# Various GPU Allocation and Detection Managers:

def is_device_available(device_type: str = "gpu") -> bool:
    """Returns a boolean value indicating whether JAX can detect current device. Defaults to "gpu" detection.
    Wraps around ``jax.devices``.
    
    Args:
        device_type (str, optional): A string indicating what type of device to search for. Defaults to "gpu".
    
    Device List:
        "cpu"
        "gpu"/"cuda"/"rocm"
        "tpu"
    
    Returns:
        bool: A boolean indicating whether the device exists.
    """
    
    if device_type == 'cuda':
        device_type = "gpu"
    try:
        jax.devices(device_type.lower())
        return True
    except:
        return False

def list_devices(device_type: str = "gpu") -> list:
    """Returns a list of physical devices that are currently on the device.
    Wraps around ``jax.devices``.
    
    Args:
        device_type (str, optional): A string indicating what type of device to search for. Defaults to "gpu".
    
    Device List:
        "cpu"
        "gpu"/"cuda"/"rocm"
        "tpu"
    
    Returns:
        list: A list of XLA compatible devices   
    """
    
    if device_type.lower() in ['cuda', 'rocm']:
        device_type = "gpu"
    try:
        devices = jax.devices(device_type.lower())
    except:
        devices = []
    
    return devices

def set_devices(device_type: str = "gpu") -> None | str:
    """Sets the global device and completes all the operations on it.
    Wraps around ``jax.config.update``.
    
    Args:
        device_type (str, optional): The string specifying the type of device to set. Defaults to "gpu".
        
    Device List:
        "cpu"
        "gpu"/"cuda"/"rocm"
        "tpu"
    
    Returns:
        str: Returns a string, if the device doesn't exist.
    """
    if device_type.lower() == 'cuda':
        device_type = "gpu"
    try:
        jax.config.update("jax_platform_name", device_type.lower())
    except:
        return f"The following device doesn't exist: {device_type}"

def device_put(tensor: Array, device_type:str = "gpu", device_no:int = 0) -> Array:
    """Transfers an tensor to the specified device and completes all the operations on it.
    Wraps around ``jax.device_put``.
    
    Args:
        tensor (Array): The tensor to transfer.
        device_type (str, optional): The string specifying the type of device to search for. Defaults to "gpu".
        device_no (int, optional): Specifies what device index to put on. Defaults to 0.
        
    Device List:
        "cpu"
        "gpu"/"cuda"/"rocm"
        "tpu" 
    """

    device = list_devices(device_type)[device_no]
    return jax.device_put(tensor, device)