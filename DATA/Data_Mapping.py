"""
Docstring for Scripts.DATA.Data_Mapping

this module provides mapping utilities for EMG data classification.
"""
import numpy as np
from numpy import array  # Needed for eval() to recognize array()
from Data_Conversion import MOVEMENT_LABELS

data_dir = "Output/NNO.txt"

# Dict for myosuite movmenet applications
def get_MyoSuite_Movement_LUT():
    """
    Returns a lookup table for MyoSuite movement patterns.
    
    :return: Dictionary mapping movement names to activation patterns
    """
    return {
        "Hand_Open": [0.0, 0.8, ...],  # Example pattern
        "Chuck_Grip": [0.9, 0.1, ...]   # Example pattern
    }

# Mapping utilities for EMG data
def data_parser(file):
    """
    Parses NN output file and converts string to dictionary.
    
    :param file: File path for data parsing
    :return: Parsed dictionary containing NN predictions
    """
    with open(file, "r") as f:
        data_str = f.read()
        f.close()
    
    # Convert string representation to actual dictionary
    # eval() needs access to numpy types to parse the arrays in the string
    data_dict = eval(data_str, {"array": np.array, "float32": np.float32})
    return data_dict

def Get_Probable_Movements(data):
    """
    Filters movements by probability threshold.
    
    :param data: Model output dictionary with 'movement_probs' key
    :return: List of probable movement indices and their probabilities
    """
    probable_movements = []
    movement_probs = data["movement_probs"]  # Correct key name
    
    for idx, prob in enumerate(movement_probs):
        if prob > 0.1:  # Threshold for probable movement (Above 10%)
            probable_movements.append((MOVEMENT_LABELS[idx], float(prob)))
    
    return probable_movements

print(Get_Probable_Movements(data_parser(data_dir)))

def Severity_Converter(severity_level, max_severity=5):
    """
    Converts Severity predictions into intesity multipliers for simulation/hardware.
    
    :param severity_level: Severity level (int)
    :param max_severity: Maximum severity level, change depending on application (int)
    :return: Scaled severity (float)
    """
    if 0 <= severity_level <= max_severity:
        return severity_level / max_severity
    else:
        raise ValueError(f"Expected severity level must be between 0 and {max_severity}.")

def activation_blender(probabilities, weights):
    """
    combines probability weightings into mutli-muscle movement pattern
    
    :param : probability: list of movement probabilities
    """
    pass

class Muscle_Mapping:
    # Movement Pattern Dictionary
    # for dynamic applications use 1 instance of this class
    
    def __init__(self):
        pass

    def Muscle_activation_Index(self, index):
        """
        Docstring for Muscle_activation_Index
        
        :param index: Description
        """
        pass
    
    def get_Activation_Pattern(self, movement_name):
        """
        Docstring for get_Activation_Pattern
        
        :param movement_name: Name of movement
        """
        pass
    
    def MyoSuiteFormatter(self, data):
        """
        Converts data to MyoSuite compatible format
        
        :param data: data to be formatted, Expects: (instert expected format)
        """
        pass