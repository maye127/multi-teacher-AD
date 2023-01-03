# multi-teacher-AD
Multi-Teacher Knowledge Distillation based Anomaly Detection


## 1.Dataset
Download dataset from:  
MVTec dataset  
https://www.mvtec.com/company/research/datasets/mvtec-ad/  
Cifar10 dataset  
http://www.cs.toronto.edu/~kriz/cifar.html  


Put mvtec and cifar10 dataset in floder ./Data

>Data
>>mvtec
>>>bottle  
>>>>train  
>>>>test  
>>>>...

>>>cable  
>>>capsule  
>>>...  

>>cifar-10-batches-py
>>>batches.meta
>>>data_batch_1  
>>>data_batch_2  
>>>data_batch_3  
>>>data_batch_4  
>>>data_batch_5  
>>>test_batch  
>>>readme.html  

## 2.Training process
Use resnet152, vgg19, and densenet210 as teacher network.  
Use resnet18, vgg11, and densenet121 as student network.  

Train networks on MVTec dataset.   
python finTS.py --dataset mvtec --student resnet --epoch 500 --evalepoch 50 --filepath ./mvtec_res --cover  
python finTS.py --dataset mvtec --student vgg --epoch 500 --evalepoch 50 --filepath ./mvtec_vgg --cover  
python finTS.py --dataset mvtec --student densenet --epoch 500 --evalepoch 50 --filepath ./mvtec_dense --cover  

Train networks on Cifar10 dataset.   
python finTS.py --dataset cifar10 --student resnet --epoch 500 --evalepoch 50 --batch_size 32 --filepath ./cifar_res --cover  
python finTS.py --dataset cifar10 --student vgg --epoch 500 --evalepoch 50 --batch_size 32 --filepath ./cifar_vgg --cover  
python finTS.py --dataset cifar10 --student densenet --epoch 500 --evalepoch 50 --batch_size 32 --filepath ./cifar_dense --cover  

The result and model will save in --filepath.  
finTS.csv save the AUC for each evaluation.  
Eval.csv save the AUC with best model.

## 3.Inference process
When train each student, inference process is including, finTS.csv and Eval.csv are saved.  
For multi student anomaly detection. Run following：  
python finTS_eval.py --dataset mvtec --filepath ./  
python finTS_eval.py --dataset cifar10 --filepath ./   

eval_mvtec.csv and eval_cifar10.csv will be saved in ./

## 4.Results on dataset.
### AUC on MVTec dataset
![RUNOOB ](https://github.com/maye127/multi-teacher-AD/blob/main/mvtec.png?raw=true )
### AUC on Cifar10 dataset
![RUNOOB ](https://github.com/maye127/multi-teacher-AD/blob/main/cifar10.png?raw=true )
