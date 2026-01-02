"""
Docstring for Scripts.DATA.Data_Mapping

this module provides mapping utilities for EMG data classification.
"""

# Mapping utilities for EMG data

def NN_data_parser(file):
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
    
def activation_blender():
    """
    combines probability weightings into mutli-muscle movement pattern
    
    :param : Description
    """
    pass

class Muscle_Mapping:
    def __init__(self):
        pass

    def Muscle_activation_Index(index):
        """
        Docstring for Muscle_activation_Index
        
        :param index: Description
        """
        pass
        
    def MyoSuiteFormatter(data):
        """
        Converts data to MyoSuite compatible format
        
        :param data: data to be formatted, Expects: (instert expected format)
        """
        pass