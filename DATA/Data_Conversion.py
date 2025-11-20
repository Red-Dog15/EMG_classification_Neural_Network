""" Example Data : https://github.com/LibEMG/ContractionIntensity/blob/main/Info.txt"""
# Convert CSV to DATAFRAMES
import pandas as pd

# assign variables for CSV paths of example movements
CSV_No_Movement = r"./data/example_data/S1_Hard_C1_R1.csv"
CSV_Wrist_Flexion = r"./data/example_data/S1_Hard_C2_R1.csv"
CSV_Wrist_Extension = r"./data/example_data/S1_Hard_C3_R1.csv"
CSV_Wrist_Pronation = r"./data/example_data/S1_Hard_C4_R1.csv"
CSV_Wrist_Supination = r"./data/example_data/S1_Hard_C5_R1.csv"
CSV_Chuck_Grip = r"./data/example_data/S1_Hard_C6_R1.csv"
CSV_Hand_Open = r"./data/example_data/S1_Hard_C7_R1.csv"

# create panda dataframes equivelents
print ("dataframes Loading...")

df_No_movement = pd.read_csv(CSV_No_Movement)
df_Wrist_Flexion = pd.read_csv(CSV_Wrist_Flexion)
df_Wrist_Extension = pd.read_csv(CSV_Wrist_Extension)
df_Wrist_Pronation = pd.read_csv(CSV_Wrist_Pronation)
df_Wrist_Supination = pd.read_csv(CSV_Wrist_Supination)
df_Chuck_Grip = pd.read_csv(CSV_Chuck_Grip)
df_Hand_Open = pd.read_csv(CSV_Hand_Open)

print ("dataframes Loaded")

# create Pytorch tensor datasets equivalents
from torch import tensor

print ("Tensors Loading...")
tensor_No_movement = tensor(df_No_movement.values)
tensor_Wrist_Flexion = tensor(df_Wrist_Flexion.values)
tensor_Wrist_Extension = tensor(df_Wrist_Extension.values)
tensor_Wrist_Pronation = tensor(df_Wrist_Pronation.values)
tensor_Wrist_Supination = tensor(df_Wrist_Supination.values)
tensor_Chuck_Grip = tensor(df_Chuck_Grip.values)
tensor_Hand_Open = tensor(df_Hand_Open.values)
print ("Tensors Loaded")

tensors_list = [tensor_No_movement, tensor_Wrist_Flexion, tensor_Wrist_Extension,
                tensor_Wrist_Pronation, tensor_Wrist_Supination, tensor_Chuck_Grip,
                tensor_Hand_Open]   