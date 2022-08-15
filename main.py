from training.training_single_axis import *
from training.training_slice_by_slice import *
from training.training_three_views import *
from training.training_slice_by_slice import *
from training.training_3d import *
import argparse
import sys
sys.path.append("/home/rawan/rawan/visual studios/training/")

import os


parser = argparse.ArgumentParser()
 
parser.add_argument("-n", "--experiment_name")
parser.add_argument("-t", "--task")
parser.add_argument("-b", "--batch_size")
parser.add_argument("-lr", "--learning_rate")
parser.add_argument("-opt", "--optimizer")
parser.add_argument("-l1", "--layer_1")
parser.add_argument("-l2", "--layer_2")
parser.add_argument("-loss")


# Read arguments from command line
args = parser.parse_args()
experiment_name = vars(args)["experiment_name"]
task = vars(args)["task"]



if experiment_name == "multi_view":
     multi_view = train_three_view(int(vars(args)["batch_size"]), float(vars(args)["learning_rate"]), vars(args)["optimizer"], l1=int(vars(args)["layer_1"]), l2=int(vars(args)["layer_2"]), loss_function="cross_entropy")
     if task == "train":
          multi_view.train()
     multi_view.eval()

if experiment_name == "sagittal_view":
     sagittal_view = training_single_axis(int(vars(args)["batch_size"]), float(vars(args)["learning_rate"]), vars(args)["optimizer"], l1=int(vars(args)["layer_1"]), l2=int(vars(args)["layer_2"]),  axis=0)
     if task == "train":
          sagittal_view.train()
     sagittal_view.eval()
     

if experiment_name == "coronal_view":
     coronal_view = training_single_axis(int(vars(args)["batch_size"]), float(vars(args)["learning_rate"]), vars(args)["optimizer"], l1=int(vars(args)["layer_1"]), l2=int(vars(args)["layer_2"]),  axis=1)
     if task == "train":
          coronal_view.train()
     coronal_view.eval()

if experiment_name == "axial_view":
     axial_view = training_single_axis(int(vars(args)["batch_size"]), float(vars(args)["learning_rate"]), vars(args)["optimizer"], l1=int(vars(args)["layer_1"]), l2=int(vars(args)["layer_2"]),  axis=2)
     if task == "train":
          axial_view.train()
     axial_view.eval()

if experiment_name == "slice_by_slice":
     slice_by_slice = training_slice_by_slice()
     if task == "train":
          slice_by_slice.train()
     slice_by_slice.eval()

if experiment_name == "3d":
     three_d = training_3d(int(vars(args)["batch_size"]), float(vars(args)["learning_rate"]), vars(args)["optimizer"], l1=int(vars(args)["layer_1"]), l2=int(vars(args)["layer_2"]),  axis=None)
     if task == "train":
          three_d.train()
     three_d.eval()
