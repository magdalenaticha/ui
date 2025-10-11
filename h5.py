import pandas as pd
import numpy as np
import h5py

# Real genes 
samples = ["KU008", "SA001", "F6004", "U4024", "A1005"]
genes = [
    "NAT2", "ADA", "CDH2", "AKT3", "MED6",
    "NAALAD2", "NAALADL1", "ACOT8", "ABI1", "GNPDA1"
]

# Simulated data 
data = np.array([
    [1.451219, 4.054724, 4.525082, 4.034952, 3.046681, 1.607111, 2.249023, 4.176643, 7.580103, 3.102467],
    [1.287009, 4.355784, 4.018522, 2.102942, 3.880011, 2.313341, 2.629223, 4.725783, 5.940103, 2.535407],
    [1.345129, 4.655144, 4.362052, 2.510192, 3.930471, 2.095541, 2.916913, 4.774883, 5.964063, 3.233657],
    [2.058609, 3.955264, 5.711552, 2.820302, 4.190321, 1.581841, 2.726893, 4.896003, 6.099253, 2.040327],
    [2.381069, 4.451534, 4.534742, 3.015402, 3.374131, 1.980091, 2.517693, 4.142483, 6.750993, 1.428257]
])

expression_df = pd.DataFrame(data, index=samples, columns=genes)
print("DataFrame created:")
print(expression_df.head())

# saving toHDF5
output_path = "real_genes.h5"
with h5py.File(output_path, "w") as f:
    dset = f.create_dataset("gene_expression", data=expression_df.values)
    dset.attrs["columns"] = np.array(expression_df.columns.tolist(), dtype="S")
    dset.attrs["index"] = np.array(expression_df.index.tolist(), dtype="S")

print(f"HDF5 file saved as {output_path}")
