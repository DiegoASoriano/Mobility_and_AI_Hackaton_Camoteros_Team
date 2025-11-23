import pandas as pd
import random
import numpy as np

# 1. TECHNICIAN INVENTORY CONFIGURATION

#  KEY CONTROL VARIABLES 
NUM_SHIFTS = 3              # Defines the number of shifts (MUST MATCH the variable in the fault file).
TECHNICIANS_PER_SHIFT = 2   # Number of technicians assigned to each shift.

# Automatic calculation of the total number of technicians
NUM_TECHNICIANS = NUM_SHIFTS * TECHNICIANS_PER_SHIFT

# Central Coordinates of CDMX (Simulated Operations Base)
# Reference point for the technicians' initial locations.
lat_center = 19.4326
lon_center = -99.1332

# 2. DYNAMIC DATAFRAME GENERATION

# Dynamic generation of shift names (Shift_1, Shift_2, etc.)
shift_names = [f'Shift_{i+1}' for i in range(NUM_SHIFTS)]

# np.repeat assigns the required number of technicians to each shift name
# Example for 3 shifts, 2 technicians: ['Shift_1', 'Shift_1', 'Shift_2', 'Shift_2', 'Shift_3', 'Shift_3']
shift_assignment = np.repeat(shift_names, TECHNICIANS_PER_SHIFT).tolist()


technician_data = {
    # 1. technician_id: Unique identifier for the technician.
    'technician_id': [100 + i for i in range(NUM_TECHNICIANS)],
    
    # 2. name: Technician alias.
    'name': [f'Technician_{i+1}' for i in range(NUM_TECHNICIANS)],
    
    # 3. active_shift: Connection column with the fault database.
    'active_shift': shift_assignment,
    
    # 4. initial_latitude: Y-coordinate of the simulated base or initial position.
    'initial_latitude': [round(lat_center + random.uniform(-0.02, 0.02), 6) for _ in range(NUM_TECHNICIANS)],
    
    # 5. initial_longitude: X-coordinate of the simulated base or initial position.
    'initial_longitude': [round(lon_center + random.uniform(-0.02, 0.02), 6) for _ in range(NUM_TECHNICIANS)],
    
    # 6. availability: Initial status (assumed Available).
    'availability': ['Available'] * NUM_TECHNICIANS 
}

df_technician_inventory = pd.DataFrame(technician_data)

# 3. EXPORT

file_name = 'data/technician_inventory_dynamic.csv'
df_technician_inventory.to_csv(file_name, index=False)

print(f"Technician database generated with {NUM_SHIFTS} shifts.")
print(f"Total technicians: {NUM_TECHNICIANS}")
print(f"Saved to: {file_name}")
