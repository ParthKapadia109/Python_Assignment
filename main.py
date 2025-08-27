import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from bokeh.plotting import figure, show, output_notebook
import math
import unittest

output_notebook()

class BaseDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
    
    def load(self):
        try:
            self.df = pd.read_csv(self.filepath)
            return self.df
        except Exception as e:
            raise FileNotFoundError(f"Error loading {self.filepath}: {e}")

class TrainDataLoader(BaseDataLoader):
    pass

class IdealDataLoader(BaseDataLoader):
    pass

class TestDataLoader(BaseDataLoader):
    pass

# Load training, ideal, and test datasets
train_df = TrainDataLoader("train.csv").load()
ideal_df = IdealDataLoader("ideal.csv").load()
test_df  = TestDataLoader("test.csv").load()

# print("Training Data:\n", train_df.head(), "\n")
# print("Ideal Functions:\n", ideal_df.head(), "\n")
# print("Test Data:\n", test_df.head(), "\n")

# Create SQLite DB and save DataFrames as tables
engine = create_engine("sqlite:///assignment.db")
train_df.to_sql("training_data", engine, if_exists="replace", index=False)
ideal_df.to_sql("ideal_functions", engine, if_exists="replace", index=False)
test_df.to_sql("test_data", engine, if_exists="replace", index=False)

#Best Fit Mapping (Least Squares)
best_funcs = {}
for col in train_df.columns[1:]: 
    min_error, best = float("inf"), None
    for ideal_col in ideal_df.columns[1:]:
        error = np.sum((train_df[col] - ideal_df[ideal_col])**2)
        if error < min_error:
            min_error, best = error, ideal_col
    best_funcs[col] = best

print("Best Fit Functions Mapping:", best_funcs)

# Calculate Deviation Limits
deviation_limits = {}
for train_col, ideal_col in best_funcs.items():
    dev = abs(train_df[train_col] - ideal_df[ideal_col])
    max_dev = dev.max()
    deviation_limits[ideal_col] = max_dev * math.sqrt(2) # (max_deviation * sqrt(2))

print("\nDeviation Limits:")
for k, v in deviation_limits.items():
    print(f"{k}: {v:.4f}")

mapped = []      # assigned test points
unassigned = []  # unassigned test points

for _, row in test_df.iterrows():
    x, y = row["x"], row["y"]
    assigned = False
    for ideal_col, limit in deviation_limits.items():
        ideal_y = ideal_df.loc[ideal_df["x"] == x, ideal_col].values[0]
        dev = abs(y - ideal_y)
        if dev <= limit:
            mapped.append([x, y, ideal_col, dev])
            assigned = True
            break
    
    if not assigned:
        unassigned.append([x, y, None, None])
        
mapped_df = pd.DataFrame(mapped, columns=["x", "y", "ideal_function", "deviation"])
unassigned_df = pd.DataFrame(unassigned, columns=["x", "y", "ideal_function", "deviation"])

# Save mapping to DB tables
mapped_df.to_sql("mapped_test_data", engine, if_exists="replace", index=False)
unassigned_df.to_sql("unassigned_test_data", engine, if_exists="replace", index=False)

print("\n✅ Test data mapped and stored")
print(f"Assigned points: {len(mapped_df)}, Unassigned points: {len(unassigned_df)}")

# Visualization
p1 = figure(title="Training Data with Best Fit Ideal Functions",
            x_axis_label="X", y_axis_label="Y", width=800, height=400)

colors = ["red", "blue", "green", "orange"]

for i, (train_col, ideal_col) in enumerate(best_funcs.items()):
    
    p1.circle(train_df["x"], train_df[train_col], size=4, color=colors[i], legend_label=f"Training {train_col}")
    p1.line(ideal_df["x"], ideal_df[ideal_col], line_width=2, color=colors[i], alpha=0.6, legend_label=f"Ideal {ideal_col}")

p1.legend.click_policy = "hide"
show(p1)

# Mapped Test Data (Assigned vs Unassigned)
p2 = figure(title="Mapped Test Data (Assigned vs Unassigned)",
            x_axis_label="X", y_axis_label="Y", width=800, height=400)

# Assigned test points: green, unassigned: red
p2.circle(mapped_df["x"], mapped_df["y"], size=6, color="green", legend_label="Assigned")
p2.circle(unassigned_df["x"], unassigned_df["y"], size=6, color="red", legend_label="Unassigned")

p2.legend.location = "top_left"
show(p2)

# Max Deviation per Training Function
p3 = figure(title="Max Deviation per Training Function",
            x_range=list(best_funcs.keys()), y_axis_label="Deviation",
            width=800, height=400)

train_cols = list(best_funcs.keys())
max_devs = [abs(train_df[tc] - ideal_df[best_funcs[tc]]).max() for tc in train_cols]

p3.vbar(x=train_cols, top=max_devs, width=0.5, color="purple")
show(p3)

# 4. Figure 4: Test Points with Chosen Ideal Functions
p4 = figure(title="Test Data Points with Chosen Ideal Functions",
            x_axis_label="X", y_axis_label="Y", width=800, height=400)

# Ideal function lines
for ideal_col in deviation_limits.keys():
    p4.line(ideal_df["x"], ideal_df[ideal_col], line_width=2, legend_label=f"Ideal {ideal_col}")

# Mapped and unmapped test data
p4.circle(mapped_df["x"], mapped_df["y"], size=6, color="green", legend_label="Assigned Test Data")
p4.circle(unassigned_df["x"], unassigned_df["y"], size=6, color="red", legend_label="Unassigned Test Data")

p4.legend.click_policy = "hide"
show(p4)

# (Optionally, you can insert Figure 5 here - such as a histogram of deviations)
