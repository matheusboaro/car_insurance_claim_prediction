
from pytorch_tabnet.tab_model import TabNetClassifier,TabNetRegressor

import torch
from sklearn.preprocessing import LabelEncoder

import numpy as np 
import pandas as pd
import os


def setup_clf(data,target, task='classification',n_target = 1):
    # 8
    np.random.seed(0)
    train = data.copy()
    target = target
    if "Set" not in train.columns:
        train["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))

    train_indices = train[train.Set=="train"].index
    valid_indices = train[train.Set=="valid"].index
    test_indices = train[train.Set=="test"].index

    nunique = train.nunique()
    types = train.dtypes

    categorical_columns = []
    categorical_dims =  {}
    for col in train.columns:
        #print(col, train[col].nunique())
        if types[col] == 'object' or nunique[col] < 200:
            #print(col, train[col].nunique())
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        #else:
            #train.fillna(train.loc[train_indices, col].mean(), inplace=True)

 
    for col in train.columns[train.dtypes == 'float64']:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)


    # check that pipeline accepts strings
    '''if n_target==1:
        train.loc[train[target]==0, target] = "not_likely"
        train.loc[train[target]==1, target] = "likely"'''

    unused_feat = ['Set','index']

    features = [ col for col in train.columns if col not in unused_feat+[target]] 

    cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    
    tabnet_params = {"n_d":8,
                    "n_a":8,
                    "cat_idxs":cat_idxs,
                    "cat_dims":cat_dims,
                    "cat_emb_dim":1,
                    "optimizer_fn":torch.optim.Adam,
                    "optimizer_params":dict(lr=2e-2),
                    "scheduler_params":{"step_size":50, # how to use learning rate scheduler
                                    "gamma":0.9},
                    "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                    "mask_type":'sparsemax' # "sparsemax",
                
                    }

    clf = TabNetClassifier(**tabnet_params
                        )

    X_train = train[features].values[train_indices]
    y_train = train[target].values[train_indices]

    X_valid = train[features].values[valid_indices]
    y_valid = train[target].values[valid_indices]

    X_test = train[features].values[test_indices]
    y_test = train[target].values[test_indices]


    return X_train, y_train, X_valid,y_valid, X_test, y_test , clf, features

def setup_reg(data,target, task='regression'):
    train = data.copy()
    target = target
    if "Set" not in train.columns:
        train["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))

    train_indices = train[train.Set=="train"].index
    valid_indices = train[train.Set=="valid"].index
    test_indices = train[train.Set=="test"].index

    categorical_columns = []
    categorical_dims =  {}
    for col in train.columns[train.dtypes == object]:
        print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = train[col].fillna("VV_likely")
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)

    for col in train.columns[train.dtypes == 'float64']:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)

    unused_feat = ['Set']

    features = [ col for col in train.columns if col not in unused_feat+[target]] 

    cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    cat_emb_dim = [64,64,64,64,64,64,64,64]

    clf = TabNetRegressor(cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs)

    X_train = train[features].values[train_indices]
    y_train = train[target].values[train_indices].reshape(-1, 1)

    X_valid = train[features].values[valid_indices]
    y_valid = train[target].values[valid_indices].reshape(-1, 1)

    X_test = train[features].values[test_indices]
    y_test = train[target].values[test_indices].reshape(-1, 1)

    return X_train, y_train, X_valid,y_valid, X_test, y_test , clf, features
# define your embedding sizes : here just a random choice

def train_tab_reg(X_train, y_train, X_valid,y_valid, clf):
    max_epochs = 300
    from pytorch_tabnet.augmentations import RegressionSMOTE
    aug = RegressionSMOTE(p=0.2)

    clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    eval_metric=['mae'],
    max_epochs=max_epochs,
    patience=20,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    augmentations=aug, #aug
    ) 

    return clf.history['valid_mae']


def train_tab_clf(X_train, y_train, X_valid,y_valid, clf, task='classification'):


    max_epochs = 200 if not os.getenv("CI", False) else 2
    from pytorch_tabnet.augmentations import ClassificationSMOTE
    aug = ClassificationSMOTE(p=0.2)
    # This illustrates the warm_start=False behaviour
    save_history = []
    for _ in range(1):
        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=['train', 'valid'],
            eval_metric=['auc'],
            max_epochs=max_epochs , patience=20,
            batch_size=1024, virtual_batch_size=128,
            num_workers=0,
            weights=1,
            drop_last=False,
            augmentations=aug, #aug, None
        )
        save_history.append(clf.history["valid_auc"])


    return save_history