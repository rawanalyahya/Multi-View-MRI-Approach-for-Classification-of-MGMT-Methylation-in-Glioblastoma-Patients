from training.training_three_views import *

batch_sizes = [8, 16, 32, 64]
lrs = [10e-7, 10e-8, 10e-6, 10e-5, 10e-4]
optimizers = ["Adam", "SGD", "Adagrad"]
l1 = [150528//(128), 150528//(256), 150528//(512), 150528//(1024)]
l2 = [150528//(128), 150528//(256), 150528//(512), 150528//(1024)]
for batch_size in batch_sizes:
     for lr in lrs:
          for optimizer in optimizers:
               for l_1 in l1:
                    for l_2 in l2:
                         if(batch_size == 8 and lr == 10e-7 and optimizer == "Adam" and (l_1 == 294 or l_1 == 588 or l_1 == 1176)):
                              continue
                         three_view = train_three_view(batch_size, lr, optimizer, l_1, l_2)
                         three_view.train_evaluate()
                         torch.cuda.empty_cache()
# three_view = train_three_view(16, 10e-7, "Adam", l1=150528//(128), l2=150528//(256))
# three_view.train_evaluate()