## **Leveraging Generative Modelling for Rich Representations**

- We model representation space as continuous dynamical system (NODEL, CARL)
- We model representation space as distribution (DARe)
- We leverage EBMs for rich representations (LEMa)
- The baselines follow from [nirbhay-design/RepresentationLearningAlgorithms](https://github.com/nirbhay-design/RepresentationLearningAlgorithms) 

## **Results**

|Algorithm|CIFAR10 (R50)|CIFAR100 (R50)|CIFAR10 (R18)|CIFAR100 (R18)|
|---|---|---|---|---|
|SimCLR|87.5|57.7|85.9|55.0|
|Barlow Twins|81.2|47.7|80.3|45.8|
|BYOL|83.0|47.0|84.8|54.8|
|SimSiam|76.5|34.5|88.6|62.3|
|DARe|89.4|62.3|87.3|61.6|
|NODEL|86.3|52.2|86.0|50.9|
|CARL|87.4|54.2|84.1|53.4|
|LEMa|89.7|64.1|87.4|58.4|
|LEMa (U)|89.5|63.6|87.6|58.6|
|LEMa (e500)|90.1|64.2|88.0|59.1|
|DAiLEMa|89.0|60.9|86.5|56.4|
|DAiLEMa (e500)|89.8|62.3|87.6|57.6|
|ScAlRe (score)|89.9|63.3|87.6|57.5|
|ScAlRe (energy)|89.7|63.8|87.2|57.0|
|SupCon|**94.0**|74.7|**93.5**|**70.4**|
|Triplet|83.4|**76.3**|86.0|64.5|


## **Workflows**

**DARe (Distribution Alignment Regularizer)**

![dare](workflows/DARe.svg)

**NODEL (Neural ODE Based SSL)**

![nodel](workflows/NODEL.svg)

**CARL (Continuous Time Adaptive SSL)**

![odessl](workflows/ODESSL.svg)

**LEMa (Low Energy Manifolds for Representation Learning)**

![lema](workflows/LEMa.svg)

<!-- ## **Experiments**

|Algorithm|CIFAR10 (r18)|CIFAR (r50)|optimizer|lr|Additional|
|---|---|---|---|---|---|
|NODEL|83.3|76.4|SGD|0.5|proj mlp is CARL mlp, spectral norm only at last layer in NODE|
|CARL|82.4|87.4||SGD|0.4|spectral norm only at last layer of NODE| -->

## **Reproducing the results**

```
python train.py --config configs/nodel.c10.yaml --gpu 0 --model resnet50 --epochs 600 --epochs_lin 100 --save_path nodel.c10.r50.pth
```

```
python train.py --config configs/carl.c10.yaml --gpu 0 --model resnet50 --epochs 600 --epochs_lin 100 --save_path carl.c10.r50.pth
```
