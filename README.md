# GOAL: A Generalist Combinatorial Optimization Agent Learning

We provide the code to learn to solve 16 standard combinatorial optimization problems.

Multi-task model is pretrained on eight problems:
* the Asymetric Traveling Salesman Problems (ATSP)
* the Capacitated Vehicle Routing Problem (CVRP)
* the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)
* the Orienteering Problem (OP)
* the Knapsack Problem (KP)
* the Minimum Vertex Covering (MVC)
* the Unrelated Parallel Machihe Scheduling (UPMS)
* the Job-Shop Schedulling Problem (JSSP)

Fine-tuning (supervised and unsupervised) is possible on eight new problems:
* the Traveling Repairman Problem (TRP)
* the Price Collecting Traveling Salesman Problem (PCTSP)
* the Split-Delivery Capacitated Vehicle Routing Problem (SDCVRP)
* the Open Capacitated Vehicle Routing Problem (OCVRP)
* the Sequential Ordering Problem (SOP)
* the Maximal Independent Set (MIS)
* the Maximum Covering Location Problem (MCLP)
* the Open-Shop Schedulling Problem (OSSP)


## Run training on set of problems

```
python3 train.py 
  --problems [list_of_problems] 
  --train_datasets [list_of_train_datasets] 
  --val_datasets [list_of_val_datasets] 
  --test_datasets [list_of_test_datasets]
```

For all trainable arguments type
```
python3 train.py -h
```

## Testing the trained model

```
python3 test.py 
  --problems [tsp|cvrp|cvrptw|op|kp|upms|jssp] 
  --test_datasets [list_of_test_datasets]
  --pretrained_model pretrained/multi.best
```

For all test options type
```
python3 test.py -h
```

## Finetuning multi-task model

### Supervised
```
python3 finetune.py
  --problems [trp|pctsp|ocvrp|sdcvrp|sop|mis|mclp|ossp]
  --train_datasets [train_dataset]
  --test_datasets [train_dataset]
  --pretrained_model pretrained/multi.best
```

### Unsupervised
```
python3 finetune.py
  --problems [trp|pctsp|ocvrp|sdcvrp|sop|mis|mclp|ossp]
  --test_datasets [train_dataset]
  --pretrained_model pretrained/multi.best
```

## Data
Test and fine-tuning data are provided in data/ directory.

Pretrained models are provided in pretrained/ directory. 

