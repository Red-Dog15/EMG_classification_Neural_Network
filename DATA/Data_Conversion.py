""" Example Data : https://github.com/LibEMG/ContractionIntensity/blob/main/Info.txt"""
# Convert CSV to DATAFRAMES
import os
import pandas as pd
import torch
from torch import tensor

DATA_DIR = os.path.dirname(__file__)
EXAMPLE_DATA_DIR = os.path.join(DATA_DIR, "Example_data")


def _example_csv(filename):
    """Build absolute path to example dataset CSV file."""
    return os.path.join(EXAMPLE_DATA_DIR, filename)

# --- Label Mappings (Module Level) ---
MOVEMENT_LABELS = {
    0: "No_Movement",
    1: "Wrist_Flexion", 
    2: "Wrist_Extension",
    3: "Wrist_Pronation",
    4: "Wrist_Supination",
    5: "Chuck_Grip",
    6: "Hand_Open"
}

SEVERITY_LABELS = {
    0: "Light",
    1: "Medium",
    2: "Hard"
}

# Reverse mappings for encoding
MOVEMENT_TO_IDX = {v: k for k, v in MOVEMENT_LABELS.items()}
SEVERITY_TO_IDX = {v: k for k, v in SEVERITY_LABELS.items()}

# Module-level cache for tensors (lazy loaded)
tensors_dict = None


def load_dataset(file_path):
    """
    Load a single CSV file and return as a PyTorch tensor.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        torch.Tensor: Tensor of shape (num_samples, 8) for 8 EMG channels
    """
    df = pd.read_csv(file_path)
    return tensor(df.values, dtype=torch.float32)


def load_all_datasets():
    """
    Load all example datasets and return as a dictionary of tensors.
    
    Returns:
        dict: Dictionary with structure:
            {
                "Light": [list of 7 tensors for each movement],
                "Medium": [list of 7 tensors for each movement],
                "Hard": [list of 7 tensors for each movement]
            }
    """
    global tensors_dict
    
    # Return cached version if already loaded
    if tensors_dict is not None:
        return tensors_dict
    
    # CSV path definitions
    CSV_Light_No_Movement = _example_csv("S1_Light_C1_R1.csv")
    CSV_Light_Wrist_Flexion = _example_csv("S1_Light_C2_R1.csv")
    CSV_Light_Wrist_Extension = _example_csv("S1_Light_C3_R1.csv")
    CSV_Light_Wrist_Pronation = _example_csv("S1_Light_C4_R1.csv")
    CSV_Light_Wrist_Supination = _example_csv("S1_Light_C5_R1.csv")
    CSV_Light_Chuck_Grip = _example_csv("S1_Light_C6_R1.csv")
    CSV_Light_Hand_Open = _example_csv("S1_Light_C7_R1.csv")

    # assign variables for CSV paths of example Medium movements
    CSV_Medium_No_Movement = _example_csv("S1_Medium_C1_R1.csv")
    CSV_Medium_Wrist_Flexion = _example_csv("S1_Medium_C2_R1.csv")
    CSV_Medium_Wrist_Extension = _example_csv("S1_Medium_C3_R1.csv")
    CSV_Medium_Wrist_Pronation = _example_csv("S1_Medium_C4_R1.csv")
    CSV_Medium_Wrist_Supination = _example_csv("S1_Medium_C5_R1.csv")
    CSV_Medium_Chuck_Grip = _example_csv("S1_Medium_C6_R1.csv")
    CSV_Medium_Hand_Open = _example_csv("S1_Medium_C7_R1.csv")

    # assign variables for CSV paths of example Hard movements
    CSV_Hard_No_Movement = _example_csv("S1_Hard_C1_R1.csv")
    CSV_Hard_Wrist_Flexion = _example_csv("S1_Hard_C2_R1.csv")
    CSV_Hard_Wrist_Extension = _example_csv("S1_Hard_C3_R1.csv")
    CSV_Hard_Wrist_Pronation = _example_csv("S1_Hard_C4_R1.csv")
    CSV_Hard_Wrist_Supination = _example_csv("S1_Hard_C5_R1.csv")
    CSV_Hard_Chuck_Grip = _example_csv("S1_Hard_C6_R1.csv")
    CSV_Hard_Hand_Open = _example_csv("S1_Hard_C7_R1.csv")


    """ --- DATAFRAME Initialization --- """


    # create panda dataframes equivelents
    print ("Light Intesensity dataframes Loading...")

    df_Light_No_movement = pd.read_csv(CSV_Light_No_Movement)
    df_Light_Wrist_Flexion = pd.read_csv(CSV_Light_Wrist_Flexion)
    df_Light_Wrist_Extension = pd.read_csv(CSV_Light_Wrist_Extension)
    df_Light_Wrist_Pronation = pd.read_csv(CSV_Light_Wrist_Pronation)
    df_Light_Wrist_Supination = pd.read_csv(CSV_Light_Wrist_Supination)
    df_Light_Chuck_Grip = pd.read_csv(CSV_Light_Chuck_Grip)
    df_Light_Hand_Open = pd.read_csv(CSV_Light_Hand_Open)
    print ("Light Intensity dataframes Loaded")

    print ("Medium Intesensity dataframes Loading...")

    df_Medium_No_movement = pd.read_csv(CSV_Medium_No_Movement)
    df_Medium_Wrist_Flexion = pd.read_csv(CSV_Medium_Wrist_Flexion)
    df_Medium_Wrist_Extension = pd.read_csv(CSV_Medium_Wrist_Extension)
    df_Medium_Wrist_Pronation = pd.read_csv(CSV_Medium_Wrist_Pronation)
    df_Medium_Wrist_Supination = pd.read_csv(CSV_Medium_Wrist_Supination)
    df_Medium_Chuck_Grip = pd.read_csv(CSV_Medium_Chuck_Grip)
    df_Medium_Hand_Open = pd.read_csv(CSV_Medium_Hand_Open)
    print ("Medium Intensity dataframes Loaded")

    print ("High Intesensity dataframes Loading...")

    df_Hard_No_movement = pd.read_csv(CSV_Hard_No_Movement)
    df_Hard_Wrist_Flexion = pd.read_csv(CSV_Hard_Wrist_Flexion)
    df_Hard_Wrist_Extension = pd.read_csv(CSV_Hard_Wrist_Extension)
    df_Hard_Wrist_Pronation = pd.read_csv(CSV_Hard_Wrist_Pronation)
    df_Hard_Wrist_Supination = pd.read_csv(CSV_Hard_Wrist_Supination)
    df_Hard_Chuck_Grip = pd.read_csv(CSV_Hard_Chuck_Grip)
    df_Hard_Hand_Open = pd.read_csv(CSV_Hard_Hand_Open)
    print ("High Intensity dataframes Loaded")


    # create Pytorch tensor datasets equivalents
    # create tensors for each intensity
    print ("Light IntensityTensors Loading...")
    tensor_Light_No_movement = tensor(df_Light_No_movement.values, dtype=torch.float32)
    tensor_Light_Wrist_Flexion = tensor(df_Light_Wrist_Flexion.values, dtype=torch.float32)
    tensor_Light_Wrist_Extension = tensor(df_Light_Wrist_Extension.values, dtype=torch.float32)
    tensor_Light_Wrist_Pronation = tensor(df_Light_Wrist_Pronation.values, dtype=torch.float32)
    tensor_Light_Wrist_Supination = tensor(df_Light_Wrist_Supination.values, dtype=torch.float32)
    tensor_Light_Chuck_Grip = tensor(df_Light_Chuck_Grip.values, dtype=torch.float32)
    tensor_Light_Hand_Open = tensor(df_Light_Hand_Open.values, dtype=torch.float32)
    print ("Light intensity Tensors Loaded")

    # create lists of tensors for each intensity
    tensors_Light_list = [tensor_Light_No_movement, tensor_Light_Wrist_Flexion, tensor_Light_Wrist_Extension,
                    tensor_Light_Wrist_Pronation, tensor_Light_Wrist_Supination, tensor_Light_Chuck_Grip,
                    tensor_Light_Hand_Open]   


    print ("Medium IntensityTensors Loading...")
    tensor_Medium_No_movement = tensor(df_Medium_No_movement.values, dtype=torch.float32)
    tensor_Medium_Wrist_Flexion = tensor(df_Medium_Wrist_Flexion.values, dtype=torch.float32)
    tensor_Medium_Wrist_Extension = tensor(df_Medium_Wrist_Extension.values, dtype=torch.float32)
    tensor_Medium_Wrist_Pronation = tensor(df_Medium_Wrist_Pronation.values, dtype=torch.float32)
    tensor_Medium_Wrist_Supination = tensor(df_Medium_Wrist_Supination.values, dtype=torch.float32)
    tensor_Medium_Chuck_Grip = tensor(df_Medium_Chuck_Grip.values, dtype=torch.float32)
    tensor_Medium_Hand_Open = tensor(df_Medium_Hand_Open.values, dtype=torch.float32)
    print ("Medium intensity Tensors Loaded")

    tensors_Medium_list = [tensor_Medium_No_movement, tensor_Medium_Wrist_Flexion, tensor_Medium_Wrist_Extension,
                    tensor_Medium_Wrist_Pronation, tensor_Medium_Wrist_Supination, tensor_Medium_Chuck_Grip,
                    tensor_Medium_Hand_Open]   

    print ("Hard Intensity Tensors Loading...")
    tensor_Hard_No_movement = tensor(df_Hard_No_movement.values, dtype=torch.float32)
    tensor_Hard_Wrist_Flexion = tensor(df_Hard_Wrist_Flexion.values, dtype=torch.float32)
    tensor_Hard_Wrist_Extension = tensor(df_Hard_Wrist_Extension.values, dtype=torch.float32)
    tensor_Hard_Wrist_Pronation = tensor(df_Hard_Wrist_Pronation.values, dtype=torch.float32)
    tensor_Hard_Wrist_Supination = tensor(df_Hard_Wrist_Supination.values, dtype=torch.float32)
    tensor_Hard_Chuck_Grip = tensor(df_Hard_Chuck_Grip.values, dtype=torch.float32)
    tensor_Hard_Hand_Open = tensor(df_Hard_Hand_Open.values, dtype=torch.float32)
    print ("Hard intensity Tensors Loaded")


    tensors_Hard_list = [tensor_Hard_No_movement, tensor_Hard_Wrist_Flexion, tensor_Hard_Wrist_Extension,
                    tensor_Hard_Wrist_Pronation, tensor_Hard_Wrist_Supination, tensor_Hard_Chuck_Grip,
                    tensor_Hard_Hand_Open]   

    # create a dictionary of all tensors lists
    tensors_dict = {
        "Light": tensors_Light_list,
        "Medium": tensors_Medium_list,
        "Hard": tensors_Hard_list
    }

    print("All datasets loaded successfully!")
    return tensors_dict


    """ DICTIONARY DIRECTORY:

    tensors_dict = {
        
        "Light": [
        
        [0] = tensor_Light_No_movement, 
        [1] = tensor_Light_Wrist_Flexion,
        [2] = tensor_Light_Wrist_Extension,
        [3] = tensor_Light_Wrist_Pronation, 
        [4] = tensor_Light_Wrist_Supination, 
        [5] = tensor_Light_Chuck_Grip,
        [6] = tensor_Light_Hand_Open,
        
        "Medium": [
        
        [0] = tensor_Medium_No_movement, 
        [1] = tensor_Medium_Wrist_Flexion,
        [2] = tensor_Medium_Wrist_Extension,
        [3] = tensor_Medium_Wrist_Pronation, 
        [4] = tensor_Medium_Wrist_Supination, 
        [5] = tensor_Medium_Chuck_Grip,
        [6] = tensor_Medium_Hand_Open],
        
        "Hard": [
        
        [0] = tensor_Hard_No_movement, 
        [1] = tensor_Hard_Wrist_Flexion,
        [2] = tensor_Hard_Wrist_Extension,
        [3] = tensor_Hard_Wrist_Pronation, 
        [4] = tensor_Hard_Wrist_Supination, 
        [5] = tensor_Hard_Chuck_Grip,
        [6] = tensor_Hard_Hand_Open],
    }
    """


def create_labeled_dataset():
    """
    Create a dataset with labels for training.
    Automatically loads all datasets if not already loaded.
    
    Returns:
        list of tuples: [(tensor_data, movement_label, severity_label), ...]
        Each tensor_data is shape (num_samples, 8) for 8 EMG channels
    """
    global tensors_dict
    
    # Load datasets if not already loaded
    if tensors_dict is None:
        load_all_datasets()
    
    labeled_data = []
    
    for severity_name, movement_list in tensors_dict.items():
        severity_idx = SEVERITY_TO_IDX[severity_name]
        
        for movement_idx, tensor_data in enumerate(movement_list):
            # Each tensor is (num_samples, 8 channels)
            # Add labels: (data, movement_class, severity_level)
            labeled_data.append((tensor_data, movement_idx, severity_idx))
    
    return labeled_data


def get_num_classes():
    """Return number of movement classes and severity levels."""
    return len(MOVEMENT_LABELS), len(SEVERITY_LABELS)


# Export new functions and constants
__all__ = [
    'load_dataset',
    'load_all_datasets',
    'tensors_dict',
    'MOVEMENT_LABELS', 'SEVERITY_LABELS',
    'MOVEMENT_TO_IDX', 'SEVERITY_TO_IDX',
    'create_labeled_dataset', 'get_num_classes'
]