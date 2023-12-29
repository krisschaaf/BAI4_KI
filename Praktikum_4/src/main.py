import torch
from pandas import DataFrame
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from ColumnNames import ColumnNames

# define size of train, test and validation ratios
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

input_size = 3
hidden_size = 2
output_size = 1

num_epochs = 100


if __name__ == '__main__':
    data = pd.read_csv('../Praktikum4_Datensatz.csv')

    x, y = (
        data[[
            ColumnNames.Grundstuecksgroesse.value,
            ColumnNames.Stadt.value,
            ColumnNames.Hausgroesse.value,
            ColumnNames.Kriminalitaetsindex.value,
            ColumnNames.Baujahr.value]
        ], data[ColumnNames.Klasse.value]
    )

    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
    numeric_features = [ColumnNames.Grundstuecksgroesse.value, ColumnNames.Hausgroesse.value, ColumnNames.Baujahr.value]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = [ColumnNames.Stadt.value, ColumnNames.Kriminalitaetsindex.value]
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ("selector", SelectPercentile(chi2, percentile=50)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = Pipeline(
        steps=[("preprocessor", preprocessor)]
    )

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio))

    clf.fit_transform(X_train, y_train)  # Fit the model according to the given training data.

    # TODO: Standardisierung der Daten funktioniert noch nicht -> dadurch können die Tensor Werte noch nicht
    #  transformiert werden

    # Überführen in Tensor Datensätze
    tensor_X_train = torch.Tensor(X_train.values.astype(float))
    tensor_y_train = torch.Tensor(y_train)

    tensor_X_test = torch.Tensor(X_test)
    tensor_y_test = torch.Tensor(y_test)

    tensor_X_val = torch.Tensor(X_val)
    tensor_y_val = torch.Tensor(y_val)

    train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    test_dataset = TensorDataset(tensor_X_test, tensor_y_test)
    val_dataset = TensorDataset(tensor_X_val, tensor_y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, output_size),
        nn.Sigmoid()
    )

    loss_function = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for n in range(num_epochs):
        for step, (X, y) in enumerate(train_dataloader):
            y_pred = model(X)
            loss = loss_function(torch.squeeze(y_pred), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



