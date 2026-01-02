"""
Docstring for Scripts.DATA.Data_Mapping

this module provides mapping utilities for EMG data classification.
"""
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
    Docstring for parse_data
    
    :param file: File for data parsing
    """
    with open(file, "r") as f:
        data = f.read()
        f.close()
    print(data)
    return data

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
    
def activation_blender(probability, weights):
    """
    combines probability weightings into mutli-muscle movement pattern
    
    :param : Descriptio
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
    def MyoSuiteFormatter(data):
        """
        Converts data to MyoSuite compatible format
        
        :param data: data to be formatted, Expects: (instert expected format)
        """
        pass