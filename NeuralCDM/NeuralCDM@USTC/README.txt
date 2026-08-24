Huang et al. Neural Cognitive Diagnosis for Intelligent Education Systems.
Thirty-Fourth AAAI Conference on Artificial Intelligence.
2020

BigData Lab @USTC 中科大大数据实验室
https://github.com/bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD

USAGE
-------
0. Run trans_logdata.py to transform the format of dataset which can be executed by NeuralCDM@USTC
>>> python trans_logdata.py

1. Run divide_data.py to divide the original data set data/log_data.json into train set, validation set and test set. 
The data/ folder has already contained divided data so this step can be skipped.
>>> python divide_data.py

2. Train the model:
>>> python train.py {device} {epoch}
For example:
/home/ysb/.conda/envs/ysbenv/bin/python /home/ysb/CDMs/NeuralCDM@USTC/train.py cuda:0 5 
or 
/home/ysb/.conda/envs/ysbenv/bin/python /home/ysb/CDMs/NeuralCDM@USTC/train.py cpu 5

NOTE: when training the model in data ASSIST2009-2010, we need to manually set the:
student_n = 4163
exer_n = 17746
knowledge_n = 123

3. creating the problem description for model testing:
>>> prob_desc.py

4. Test the trained the model on the test set:
>>> python predict.py {epoch}
For example:
python predict.py 5