import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler

class MNISTLoader:
    def __init__(self, train_file="mnist_train.csv", test_file="mnist_test.csv"):
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.train_file = train_file
        self.test_file = test_file

        self.num_inputs = None
        self.num_outputs = None

    def load_train(self):
        try:
            df_train = pd.read_csv(self.train_file)
            self.num_inputs = df_train.drop("label", axis=1).shape[1]
            self.num_outputs = len(df_train["label"].unique())

            train_data = np.array(df_train, dtype=float)
            np.random.shuffle(train_data)

            self.y_train = train_data[:, 0]
            self.x_train = train_data[:, 1 : self.num_inputs+1]
            for i in range(len(self.x_train)):
                    self.x_train[i] = self.x_train[i]/255.0

            return self.x_train, self.y_train

        except FileNotFoundError:
            print("TRAIN FILE NOT FOUND")

    def load_test(self):
        try:
            df_test = pd.read_csv(self.test_file)
            test_data = np.array(df_test, dtype=float)
            self.y_test = test_data[:, 0]
            self.x_test = test_data[:, 1 : self.num_inputs+1]
            for i in range(len(self.x_test)):
                self.x_test[i] = self.x_test[i]/255.0

            return self.x_test, self.y_test

        except FileNotFoundError:
            print("TEST FILE NOT FOUND")

    def get_inputs_outputs(self):
        return self.num_inputs, self.num_outputs

class PenguinLoader:
    def __init__(self, train_file="penguins.csv"):
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.num_inputs = None
        self.num_outputs = None
        #------------------------------#
        self.train_file = train_file
        self.df = self.prepare(pd.read_csv(self.train_file))
        self.split()

    def prepare(self, df):
        new_df = df.copy()
        new_df["gender"].fillna(df["gender"].mode()[0], inplace=True)

        num_cols = new_df.drop(["gender", "species"], axis=1)

        full_pipeline = ColumnTransformer([
            ("ord", OrdinalEncoder(), ["species"]),
            ("num", MinMaxScaler(), num_cols.columns),
            ("one_hot", OneHotEncoder(drop='first'), ["gender"])
        ])

        df_prepared = full_pipeline.fit_transform(new_df)
        df_prepared = pd.DataFrame(df_prepared,
                        columns=["species", "bill_length_mm", "bill_depth_mm",
                        "flipper_length_mm", "body_mass_g", "gender"])
        
        return df_prepared

    def split(self):
        self.num_inputs = self.df.drop("species", axis=1).shape[1]
        self.num_outputs = len(self.df["species"].unique())

        split = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=42)
        for train_index, test_index in split.split(self.df, self.df["species"]):
            strat_train_set = self.df.loc[train_index]
            strat_test_set = self.df.loc[test_index]

        self.x_train = strat_train_set.drop("species", axis=1).to_numpy()
        self.y_train = strat_train_set["species"].to_numpy()

        self.x_test = strat_test_set.drop("species", axis=1).to_numpy()
        self.y_test = strat_test_set["species"].to_numpy()

    def load_train(self):
        return self.x_train, self.y_train

    def load_test(self):
        return self.x_test, self.y_test

    def get_inputs_outputs(self):
        return self.num_inputs, self.num_outputs