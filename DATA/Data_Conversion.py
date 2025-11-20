""" Example Data : https://github.com/LibEMG/ContractionIntensity/blob/main/Info.txt"""

# Convert CSV to DATAFRAMES
import pandas as pd

# assign variables for CSV paths of example movements
print ("CSV's Loading...")

CSV_No_Movement = "Example_data\S1_Hard_C1_R1.csv"
CSV_Wrist_Flexion = "Example_data\S2_Hard_C1_R1.csv"
CSV_Wrist_Extension = "Example_data\S3_Hard_C1_R1.csv"
CSV_Wrist_Pronation = "Example_data\S4_Hard_C1_R1.csv"
CSV_Wrist_Supination = "Example_data\S5_Hard_C1_R1.csv"
CSV_Chuck_Grip = "Example_data\S6_Hard_C1_R1.csv"
CSV_Hand_Open = "Example_data\S7_Hard_C1_R1.csv"

print ("CSV's Loaded")


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