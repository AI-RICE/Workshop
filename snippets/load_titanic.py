import seaborn as sns

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Select relevant features
features = ["pclass", "sex", "age", "fare"]
X = titanic[features]
y = titanic["survived"]
