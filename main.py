from training.training_three_views import *
import argparse
from training.training_slice_by_slice import *
from training.training_single_axis import *



parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-b", "--batch_size")
parser.add_argument("-lr", "--learning_rate")
parser.add_argument("-opt", "--optimizer")
parser.add_argument("-l1", "--layer_1")
parser.add_argument("-l2", "--layer_2")
parser.add_argument("-loss")

 
# Read arguments from command line
args = parser.parse_args()


x_axis = training_single_axis(int(vars(args)["batch_size"]), float(vars(args)["learning_rate"]), vars(args)["optimizer"], l1=int(vars(args)["layer_1"]), l2=int(vars(args)["layer_2"]), axis=2)
x_axis.train_evaluate()
