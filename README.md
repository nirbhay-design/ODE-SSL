## **NODEL: Neural ODE based SSL framework**

- We model representation space as continuous dynamical system
- The baselines follow from the previous ![repository](https://github.com/nirbhay-design/RepresentationLearningAlgorithms) 

## **Results**

|Algorithm|CIFAR10 (R50)|CIFAR100 (R50)|CIFAR10 (R18)|CIFAR100 (R18)|
|---|---|---|---|---|
|SimCLR|87.5|57.7|85.9|55.0|
|SupCon|**94.0**|74.7|**93.5**|**70.4**|
|Triplet|83.4|**76.3**|86.0|64.5|
|Barlow Twins|81.2|47.7|80.3|45.8|
|BYOL|83.0|47.0|84.8|54.8|
|SimSiam|76.5|34.5|88.6|62.3|

## **Reproducing the results**

Run the following command

```
python train.py --config configs/nodel.c10.yaml --gpu 0 --model resnet50 --epochs 500 --epochs_lin 100 --save_path nodel.c10.r50.pth
```