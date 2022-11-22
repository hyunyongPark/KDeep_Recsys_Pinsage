# KDeepFashion Project - Pinsage 

### Reference
- https://arxiv.org/pdf/1806.01973.pdf
- https://github.com/dmlc/dgl/tree/master/examples/pytorch/pinsage


### Model Description 
<table>
    <thead>
        <tr>
            <td>Pinsage Model Architecture</td>
            <td>Pinsage Training Process</td>
            <td>Pinsage Minibatch Process</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/KDeep_Recommendation/blob/main/img/architecture.PNG"/></td>
            <td><img src="https://github.com/hyunyongPark/KDeep_Recommendation/blob/main/img/architecture2.PNG"/></td>
            <td><img src="https://github.com/hyunyongPark/KDeep_Recommendation/blob/main/img/architecture3.PNG"/></td>
        </tr>
    </tbody>
</table>

### Build Docker Image
```
# Build Docker Image
sudo docker build -t graphrec:v1 .
# Check generated Docker Image
sudo docker images
```

```
# Run generated docker image
sudo docker run graphrec:v1
```
- Result
<table>
    <thead>
        <tr>
            <td>result</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src=""/></td>
        </tr>
    </tbody>
</table>


### Requirements
```
# python version : 3.8.13
pip install -r requirements.txt 
```



### cmd running

The install cmd is:
```
conda create -n your_prjname python=3.8
conda activate your_prjname
cd {The virtual environment directory that you created}
pip install -r requirements.txt
```
- your_prjname : Name of the virtual environment to create


### Trained weight file Download 
Download the trained weight file through the link below.
This file is a trained file that learned the k-deep fashion dataset.
Ensure that the weight file is located at "model/".
- https://drive.google.com/drive/folders/1tTCoCYQBNi-4dfTqwfa8LU6UJeTE41-r?usp=sharing

The testing cmd is: 
```

python3 validation-kdeep.py 

```

If you want to proceed with the new learning, adjust the parameters and set the directory and proceed with the command below.

The Training cmd is:
```

python3 training-kdeep.py 

```


### Result
- Our Model Performance Table

|Embedding|Dataset|RMSE-Score|
|---|---|---|
|Graph|train(9,373)/valid(1,042)|*0.9711*|
|Graph + User + Item|train(9,373)/valid(1,042)|*0.8357*|
|Graph + User + Item|train(20,123)/valid(2,516)|**0.7813**|

<table>
    </thead>
    <tbody>
        <tr>
            <td><img src=""/></td>
        </tr>
    </tbody>
</table>





- Reference Result Tables in Paper
<table>
    <thead>
        <tr>
            <td>HR@K(=500) Score</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/KDeep_Recommendation/blob/main/img/performance_paper.PNG"/></td>
        </tr>
    </tbody>
</table>
