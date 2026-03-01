
## **Leveraging Generative Modelling for Rich Representations**

- We model representation space as continuous dynamical system (NODEL, CARL)
- We model representation space as distribution (DARe)
- We leverage EBMs for rich representations (LEMa)
- We leverage Score for rich representations (ScAlRe)
<!-- - The baselines follow from [nirbhay-design/RepresentationLearningAlgorithms](https://github.com/nirbhay-design/RepresentationLearningAlgorithms)  -->

## **Workflows**

**ScAlRe (Score Alignment Regularization for Representation Learning)**

![scalre](workflows/ScAlReFlow.svg)

## **Results**



|Algorithm|CIFAR10 (R50)||CIFAR100 (R50)||CIFAR10 (R18)||CIFAR100 (R18)||Timg (R18)||
|---|---|---|---|---|---|---|---|---|---|---|
||LR|kNN|LR|kNN|LR|kNN|LR|kNN|LR|kNN|
|SimCLR|||||91.2|89.4|62.6|58.0|||
|SimCLR-ScAlRe-E|||||91.0|89.3|63.9|57.8|||
|SimCLR-ScAlRe-S|||||91.5|89.9|63.9|57.9|||
|Barlow Twins|||||90.1|87.2|67.7|59.0|||
|Barlow Twins-ScAlRe-E|||||90.1|87.6|66.8|58.6|||
|Barlow Twins-ScAlRe-S|||||90.5|87.4|65.9|56.3|||
|BYOL|||||||||||
|BYOL-ScAlRe-E|||||||||||
|BYOL-ScAlRe-S|||||||||||
|SimSiam|||||90.4|88.5|62.6|57.1|||
|SimSiam-ScAlRe-E|||||90.5|89.1|62.7|58.0|||
|SimSiam-ScAlRe-S|||||90.6|88.8|62.8|57.9|||
|VicReg|||||87.7|84.2|62.7|52.2|||
|VicReg-ScAlRe-E|||||87.8|84.1|62.4|52.0|||
|VicReg-ScAlRe-S|||||87.5|84.3|62.8|52.3|||

## **Clustering Metrics Results**

|Algorithm|CIFAR10 (R18)||||CIFAR100 (R18)||||
|---|---|---|---|---|---|---|---|---|
||ARI|NMI|Silhoutte|DBS|ARI|NMI|Silhoutte|DBS|
|SimCLR|0.589|**0.707**|**0.082**|**3.246**|**0.244**|**0.535**|**0.115**|2.493|
|SimCLR-ScAlRe-E|**0.605**|0.703|**0.082**|3.345|0.231|**0.535**|0.114|**2.451**|
|SimCLR-ScAlRe-S|0.557|0.677|0.075|3.477|0.235|0.531|0.113|2.515|
|Barlow Twins|0.407|0.541|0.034|4.428|**0.179**|**0.472**|0.052|**3.206**|
|Barlow Twins-ScAlRe-E|**0.471**|**0.582**|**0.038**|**4.289**|0.171|0.464|**0.053**|3.207|
|Barlow Twins-ScAlRe-S|0.376|0.514|0.032|4.475|0.157|0.437|0.045|3.264|
|SimSiam|0.576|0.674|0.059|3.879|**0.228**|0.514|0.076|2.935|
|SimSiam-ScAlRe-E|0.553|**0.678**|**0.060**|**3.720**|0.227|**0.516**|**0.079**|**2.877**|
|SimSiam-ScAlRe-S|**0.581**|0.669|0.059|3.905|0.223|0.513|0.077|2.921|
|VicReg|**0.435**|**0.520**|**0.051**|**3.595**|0.150|0.414|0.048|3.078|
|VicReg-ScAlRe-E|0.399|0.496|0.048|3.792|**0.159**|**0.423**|**0.050**|**3.053**|
|VicReg-ScAlRe-S|0.400|0.492|0.047|3.644|0.155|0.420|0.049|3.062|
|BYOL|||||||||
|BYOL-ScAlRe-E|||||||||
|BYOL-ScAlRe-S|||||||||




<!-- |Algorithm|CIFAR10 (R50)|CIFAR100 (R50)|CIFAR10 (R18)|CIFAR100 (R18)|
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
|Triplet|83.4|**76.3**|86.0|64.5| -->

<!-- **DARe (Distribution Alignment Regularizer)**

![dare](workflows/DARe.svg)

**NODEL (Neural ODE Based SSL)**

![nodel](workflows/NODEL.svg)

**CARL (Continuous Time Adaptive SSL)**

![odessl](workflows/ODESSL.svg)

**LEMa (Low Energy Manifolds for Representation Learning)**

![lema](workflows/LEMa.svg) -->

<!-- ## **Experiments**

|Algorithm|CIFAR10 (r18)|CIFAR (r50)|optimizer|lr|Additional|
|---|---|---|---|---|---|
|NODEL|83.3|76.4|SGD|0.5|proj mlp is CARL mlp, spectral norm only at last layer in NODE|
|CARL|82.4|87.4||SGD|0.4|spectral norm only at last layer of NODE| -->

## **Reproducing the results**

- lookout for more commands in `run.sh`

```
python train.py --config configs/simclr.yaml --dataset cifar10 --gpu 1 --model resnet18 --epochs 800 --epochs_lin 100 --save_path simclr.c10.r18.pth > logs/simclr.c10.r18.log
```

## **Test the pretrained model 

```
python test.py --dataset cifar10 --model resnet18 --saved_path saved_models/simclr.c10.r18.pth --cmet --knn --lreg --linprobe --tsne --gpu 0 --verbose
```
