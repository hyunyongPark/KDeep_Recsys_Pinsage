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



### Requirements
- python V  # python version : 3.8.13
- dgl==0.9.1
- tqdm
- torch==1.9.1
- torchvision==0.10.1
- torchaudio==0.9.1
- torchtext==0.10.1
- dask
- partd
- pandas
- fsspec==0.3.3
- scipy
- sklearn


### Build Docker Image
The Docker image shows the result of calculating the Hitrate at k=500 as shown in the original paper. 
Based on the test data not used for training, the average of the subgraph embeddings of each user is obtained through the KNN algorithm, the distance from all items is obtained, the recommended list is obtained, and one of the labels not used for training is calculated as Hit.
```
# Build Docker Image
sudo docker build -t pinsage:v1 .
# Check generated Docker Image
sudo docker images
```

```
# Run generated docker image
sudo docker run pinsage:v1
```

<table>
    <thead>
        <tr>
            <td>result</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/KDeep_Recommendation/blob/main/img/performance_K500_whole.PNG"/></td>
        </tr>
    </tbody>
</table>



### cmd running

The install cmd is:
```
conda create -n your_prjname python=3.8
conda activate your_prjname
cd {Repo Directory}
pip install -r requirements.txt
```
- your_prjname : Name of the virtual environment to create


##### Trained weight file Download 
Download the trained weight file through the link below.
This file is a trained file that learned the k-deep fashion dataset.
Ensure that the weight file is located at "model/".
- https://drive.google.com/file/d/11bt3BocyaukuP0GNVkAeM5Aue81A1kdX/view?usp=share_link

The testing cmd is: 
```

python3 validation-kdeep.py 

```

If you want to proceed with the new learning, adjust the parameters and set the directory and proceed with the command below.

The Training cmd is:
```

python3 training-kdeep.py 

```

At the time of learning, recall@10 was set to metric to update the validation reference score. The example is as follows.

<table>
    <thead>
        <tr>
            <td>Training example</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/KDeep_Recommendation/blob/main/img/training_log.PNG"/></td>
        </tr>
    </tbody>
</table>


### Test Result
- Pinsage Reference Result Tables in Original Paper
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


- Our Pinsage Model Performance Table

|Embedding|Dataset|HR@K(=500)|HR@K(=50)|HR@K(=30)|
|---|---|---|---|---|
|Graph + Item meta|train(30,570)/valid(3,804)/test(3,910)|*74.5%*|*54.8%*|*38.2%*|
|Graph + Item meta|train(139,637)/valid(17,339)/test(17,936)|**92.6%**|**74.8%**|**49.6%**|
|Graph + Item meta|train(199,923)/valid(24,929)/test(25,645)|*80.7%*|*53.0%*|*40.2%*|

<table>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/KDeep_Recommendation/blob/main/img/performance_K500_whole.PNG"/></td>
        </tr>
    </tbody>
</table>

- Example recommendations for Pinsage model's query items
<table>
    <thead>
        <tr>
            <td>K-Neareast Neighbors Result</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="https://github.com/hyunyongPark/KDeep_Recommendation/blob/main/img/example1.png"/></td>
        </tr>
    </tbody>
</table>
