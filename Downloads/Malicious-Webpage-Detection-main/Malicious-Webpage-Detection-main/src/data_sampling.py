import pandas as pd

# 1. Load and clean
df = pd.read_csv('data/raw/malicious_urls.csv')  # adjust path if needed
df['type'] = df['type'].str.strip().str.lower()

print(df['type'].value_counts())

# 2. Choose how many per class
n_per_class = 20000

# 3. Sample a balanced dataset
df_balanced = (
    df.groupby('type', group_keys=False)
      .apply(lambda x: x.sample(
          n=min(len(x), n_per_class),
          random_state=42
      ))
)

print(df_balanced['type'].value_counts())

# 4. Save this subset for your project
df_balanced.to_csv('data/urls_balanced_20k_per_class.csv', index=False)
print("Saved balanced dataset to data/interim/urls_balanced_20k_per_class.csv")