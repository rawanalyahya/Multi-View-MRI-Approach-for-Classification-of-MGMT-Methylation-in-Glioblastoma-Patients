import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument("--n_folds", default=5, type=int)
args = parser.parse_args()

train = pd.read_csv("/home/rfyahya/Desktop/Rawan/brain/clean/visual studios/dataframes/testing_set_t2w.csv")

skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=518)
oof = []
targets = []
target = "MGMT_value"

for fold, (trn_idx, val_idx) in enumerate(
    skf.split(train, train[target])
):
    train.loc[val_idx, "fold"] = int(fold)


train.to_csv("../../input/testing.csv", index=False)
