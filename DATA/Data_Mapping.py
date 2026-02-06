"""
Docstring for Scripts.DATA.Data_Mapping

this module provides mapping utilities for EMG data classification.
"""
import numpy as np
from numpy import array  # Needed for eval() to recognize array()
from Data_Conversion import MOVEMENT_LABELS

data_dir = "Output/NNO.txt"

# Dict for myosuite movmenet applications
def get_MyoSuite_Movement_LUT(movement_name):
    """
    Returns a lookup table for MyoSuite movement patterns.
    
    :return: Dictionary mapping movement names to activation patterns
    """
    muscle_map_dict = {
        "NO_Movement": [0.0] * 10,  # Default pattern for no movement
        "Hand_Open": [0.0, 0.8, ...],  # Example pattern
        "Chuck_Grip": [0.9, 0.1, ...],  # Example pattern
        "Pinch_Grip": []    # Example pattern
   
    }
    return muscle_map_dict.get(movement_name, [0.0] * 10)  # Default to zero activation if not found

print (f"get myosuite movement lut /n{get_MyoSuite_Movement_LUT('TEST')}")

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
    Filters movements by probability threshold and includes severity information.
    
    :param data: Model output dictionary with 'movement_probs' and 'severity_pred' keys
    :return: List of tuples: (movement_name, probability, severity_level)
    """
    probable_movements = []
    movement_probs = data["movement_probs"]
    severity = data["severity_pred"]  # Single severity value for all movements
    
    for idx, prob in enumerate(movement_probs):
        if prob > 0.1:  # Threshold for probable movement (Above 10%)
            probable_movements.append((MOVEMENT_LABELS[idx], float(prob), severity))
    return probable_movements

print(f"Probable Movements: {Get_Probable_Movements(data_parser(data_dir))}")


def Severity_Converter(probable_movements, max_severity=5):
    """
    Converts Severity predictions into intensity multipliers for simulation/hardware.
    Takes output from Get_Probable_Movements and replaces severity levels with scaled values.
    
    :param probable_movements: List of tuples from Get_Probable_Movements: [(name, prob, severity), ...]
    :param max_severity: Maximum severity level, change depending on application (default=5)
    :return: List of tuples with converted severities: [(name, prob, scaled_severity), ...]
    """
    converted_movements = []
    
    for movement_name, prob, severity in probable_movements:
        if 0 <= severity <= max_severity:
            scaled_severity = severity / max_severity
            converted_movements.append((movement_name, prob, scaled_severity))
        else:
            raise ValueError(f"Expected severity level must be between 0 and {max_severity}.")
    
    return converted_movements

# Test the converter with probable movements
probable_movements = Get_Probable_Movements(data_parser(data_dir))
converted_movements = Severity_Converter(probable_movements)
print(f"Converted Movements: {converted_movements}")

def activation_blender(converted_movements):
    """
    Combines probability weightings with severity into final activation values.
    Uses probability as a weight applied to the severity multiplier.
    
    :param converted_movements: List of tuples from Severity_Converter: [(name, prob, scaled_severity), ...]
    :return: List of tuples with blended activation: [(name, blended_activation), ...]
    """
    blended_results = []
    
    for movement_name, prob, scaled_severity in converted_movements:
        # Blend: probability weighted by severity
        blended_activation = prob * (1/scaled_severity)
        blended_results.append((movement_name, blended_activation))    
    return blended_results

print(activation_blender(converted_movements))
blended_activations = activation_blender(converted_movements)


# Test activation blender with converted movements
print(f"Blended Activations: {blended_activations}")

"""
# Example usage of activation_blender with probable movements
data = data_parser(data_dir)
probable_movements = Get_Probable_Movements(data)

if probable_movements:
    # Extract movement probabilities and convert their severities
    movement_probs = [prob for _, prob, _ in probable_movements]
    movement_severities = [severity for _, _, severity in probable_movements]
    converted_severities = list(Severity_Converter(movement_severities))
    
    print(f"\nBlending {len(probable_movements)} probable movements:")
    blended_activations = list(activation_blender(movement_probs, converted_severities))
    print(f"Blended_Activations: {blended_activations}")
    
    for i, (name, prob, sev) in enumerate(probable_movements):
        print(f"  {name}: prob={prob:.3f}, severity={sev}, blended={blended_activations[i]:.3f}")
"""
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
    
    def MyoSuiteFormatter(self, movement_name, blended_activation):
        get_MyoSuite_Movement_LUT()

        