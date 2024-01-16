import torch
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
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

input_size = 6
hidden_size = 4
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

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio))

    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
    numeric_features = [ColumnNames.Grundstuecksgroesse.value, ColumnNames.Hausgroesse.value, ColumnNames.Baujahr.value]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = [ColumnNames.Stadt.value, ColumnNames.Kriminalitaetsindex.value]
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder()),
            ("selector", SelectPercentile(chi2, percentile=50)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_test_transformed = preprocessor.fit_transform(X_test, y_test)
    X_val_transformed = preprocessor.fit_transform(X_val, y_val)

    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(y_train.values.reshape(-1, 1))
    y_train_transformed = one_hot_encoder.transform(y_train.values.reshape(-1, 1)).toarray()[:, 1]

    one_hot_encoder.fit(y_test.values.reshape(-1, 1))
    y_test_transformed = one_hot_encoder.transform(y_test.values.reshape(-1, 1)).toarray()[:, 1]

    one_hot_encoder.fit(y_val.values.reshape(-1, 1))
    y_val_transformed = one_hot_encoder.transform(y_val.values.reshape(-1, 1)).toarray()[:, 1]

    # Überführen in Tensor Datensätze
    tensor_X_train = torch.tensor(X_train_transformed, dtype=torch.float32)
    tensor_y_train = torch.tensor(y_train_transformed, dtype=torch.float32)

    tensor_X_test = torch.tensor(X_test_transformed, dtype=torch.float32)
    tensor_y_test = torch.tensor(y_test_transformed, dtype=torch.float32)

    tensor_X_val = torch.tensor(X_val_transformed, dtype=torch.float32)
    tensor_y_val = torch.tensor(y_val_transformed, dtype=torch.float32)

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

    model.train()

    for n in range(num_epochs):
        for step, (X, y) in enumerate(train_dataloader):
            y_pred = model(X)
            loss = loss_function(torch.squeeze(y_pred), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



