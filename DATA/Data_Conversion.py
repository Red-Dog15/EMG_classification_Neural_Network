""" Example Data : https://github.com/LibEMG/ContractionIntensity/blob/main/Info.txt"""
# Convert CSV to DATAFRAMES
import pandas as pd

""" ---  CSV PATH LOADING Initialization --- """

# assign variables for CSV paths of example Light movements
CSV_Light_No_Movement = r"./data/example_data/S1_Light_C1_R1.csv"
CSV_Light_Wrist_Flexion = r"./data/example_data/S1_Light_C2_R1.csv"
CSV_Light_Wrist_Extension = r"./data/example_data/S1_Light_C3_R1.csv"
CSV_Light_Wrist_Pronation = r"./data/example_data/S1_Light_C4_R1.csv"
CSV_Light_Wrist_Supination = r"./data/example_data/S1_Light_C5_R1.csv"
CSV_Light_Chuck_Grip = r"./data/example_data/S1_Light_C6_R1.csv"
CSV_Light_Hand_Open = r"./data/example_data/S1_Light_C7_R1.csv"

# assign variables for CSV paths of example Light movements
CSV_Medium_No_Movement = r"./data/example_data/S1_Medium_C1_R1.csv"
CSV_Medium_Wrist_Flexion = r"./data/example_data/S1_Medium_C2_R1.csv"
CSV_Medium_Wrist_Extension = r"./data/example_data/S1_Medium_C3_R1.csv"
CSV_Medium_Wrist_Pronation = r"./data/example_data/S1_Medium_C4_R1.csv"
CSV_Medium_Wrist_Supination = r"./data/example_data/S1_Medium_C5_R1.csv"
CSV_Medium_Chuck_Grip = r"./data/example_data/S1_Medium_C6_R1.csv"
CSV_Medium_Hand_Open = r"./data/example_data/S1_Medium_C7_R1.csv"

# assign variables for CSV paths of example Hard movements
CSV_Hard_No_Movement = r"./data/example_data/S1_Hard_C1_R1.csv"
CSV_Hard_Wrist_Flexion = r"./data/example_data/S1_Hard_C2_R1.csv"
CSV_Hard_Wrist_Extension = r"./data/example_data/S1_Hard_C3_R1.csv"
CSV_Hard_Wrist_Pronation = r"./data/example_data/S1_Hard_C4_R1.csv"
CSV_Hard_Wrist_Supination = r"./data/example_data/S1_Hard_C5_R1.csv"
CSV_Hard_Chuck_Grip = r"./data/example_data/S1_Hard_C6_R1.csv"
CSV_Hard_Hand_Open = r"./data/example_data/S1_Hard_C7_R1.csv"


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
from torch import tensor

# create tensors for each intensity
print ("Light IntensityTensors Loading...")
tensor_Light_No_movement = tensor(df_Light_No_movement.values)
tensor_Light_Wrist_Flexion = tensor(df_Light_Wrist_Flexion.values)
tensor_Light_Wrist_Extension = tensor(df_Light_Wrist_Extension.values)
tensor_Light_Wrist_Pronation = tensor(df_Light_Wrist_Pronation.values)
tensor_Light_Wrist_Supination = tensor(df_Light_Wrist_Supination.values)
tensor_Light_Chuck_Grip = tensor(df_Light_Chuck_Grip.values)
tensor_Light_Hand_Open = tensor(df_Light_Hand_Open.values)
print ("Light intensity Tensors Loaded")

tensors_Light_list = [tensor_Light_No_movement, tensor_Light_Wrist_Flexion, tensor_Light_Wrist_Extension,
                tensor_Light_Wrist_Pronation, tensor_Light_Wrist_Supination, tensor_Light_Chuck_Grip,
                tensor_Light_Hand_Open]   


print ("Medium IntensityTensors Loading...")
tensor_Medium_No_movement = tensor(df_Medium_No_movement.values)
tensor_Medium_Wrist_Flexion = tensor(df_Medium_Wrist_Flexion.values)
tensor_Medium_Wrist_Extension = tensor(df_Medium_Wrist_Extension.values)
tensor_Medium_Wrist_Pronation = tensor(df_Medium_Wrist_Pronation.values)
tensor_Medium_Wrist_Supination = tensor(df_Medium_Wrist_Supination.values)
tensor_Medium_Chuck_Grip = tensor(df_Medium_Chuck_Grip.values)
tensor_Medium_Hand_Open = tensor(df_Medium_Hand_Open.values)
print ("Medium intensity Tensors Loaded")

tensors_Medium_list = [tensor_Medium_No_movement, tensor_Medium_Wrist_Flexion, tensor_Medium_Wrist_Extension,
                tensor_Medium_Wrist_Pronation, tensor_Medium_Wrist_Supination, tensor_Medium_Chuck_Grip,
                tensor_Medium_Hand_Open]   

print ("Hard Intensity Tensors Loading...")
tensor_Hard_No_movement = tensor(df_Hard_No_movement.values)
tensor_Hard_Wrist_Flexion = tensor(df_Hard_Wrist_Flexion.values)
tensor_Hard_Wrist_Extension = tensor(df_Hard_Wrist_Extension.values)
tensor_Hard_Wrist_Pronation = tensor(df_Hard_Wrist_Pronation.values)
tensor_Hard_Wrist_Supination = tensor(df_Hard_Wrist_Supination.values)
tensor_Hard_Chuck_Grip = tensor(df_Hard_Chuck_Grip.values)
tensor_Hard_Hand_Open = tensor(df_Hard_Hand_Open.values)
print ("Hard intensity Tensors Loaded")


tensors_Hard_list = [tensor_Hard_No_movement, tensor_Hard_Wrist_Flexion, tensor_Hard_Wrist_Extension,
                tensor_Hard_Wrist_Pronation, tensor_Hard_Wrist_Supination, tensor_Hard_Chuck_Grip,
                tensor_Hard_Hand_Open]   