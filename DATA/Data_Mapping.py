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
    data_dict = eval(data_str, {"array": np.array, "float32": np.float32}) # add types as needed
    return data_dict

print(data_parser(data_dir))
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

print(f"Probable Movements: {Get_Probable_Movements(data_parser(data_dir))}")

def get_Movement_Severity(data, weight = 0.5):
    """
    Identifies the most probable severity level from model output and weights with confidence.
    
    :param data: Model output dictionary with 'severity_pred' and 'severity_confidence' keys
    :param weight: Weighting factor for confidence adjustment (default 0.5)
    :return: Weighted severity level (float)
    """
    max_severity = data["severity_pred"]  # 0=Light, 1=Medium, 2=Hard
    severity_confidence = data["severity_confidence"]
    
    # Return weighted severity (confidence-adjusted)
    return max_severity * (severity_confidence * (1/weight))


print(f"movement_severities: {get_Movement_Severity(data_parser(data_dir))}")
severities = get_Movement_Severity(data_parser(data_dir))

def Severity_Converter(severity_level, max_severity=5):
    """
    Converts Severity predictions into intensity multipliers for simulation/hardware.
    
    :param severity_level: Severity level (int or float, or iterable of values)
    :param max_severity: Maximum severity level, change depending on application (default=5)
    :return: Generator yielding scaled severity values (float between 0 and 1)
    """
    # Handle scalar input by converting to list
    if not hasattr(severity_level, '__iter__'):
        severity_level = [severity_level]
    
    for severity in severity_level:
        if 0 <= severity <= max_severity:
            yield severity / max_severity
        else:
            raise ValueError(f"Expected severity level must be between 0 and {max_severity}.")

print(f"Converted_Severity: {list(Severity_Converter(severities))}")

def activation_blender(probabilities: list[float], weights: list[float]):
    """
    Combines probability weightings into multi-muscle movement pattern.
    
    :param probabilities: List of movement probabilities (floats 0-1)
    :param weights: List of activation weights/patterns for each probability
    :return: Generator yielding blended activation values
    """
    if len(probabilities) != len(weights):
        raise ValueError(f"Probabilities ({len(probabilities)}) and weights ({len(weights)}) must have the same length")
    
    # Blend each probability with its corresponding weight
    for prob, weight in zip(probabilities, weights):
        blended_val = prob * weight
        yield blended_val

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