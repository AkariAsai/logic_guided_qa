# Logic-Guided Data Augmentation and Regularization for Consistent Question Answering

This is the original implementation of the following paper.

Akari Asai and Hannaneh Hajishirzi. [Logic-Guided Data Augmentation and Regularization for Consistent Question Answering](https://arxiv.org/abs/2004.10157). In: Proceedings of ACL (short). 2020.

```
@inproceedings{asai2020logic,
  title={Logic-Guided Data Augmentation and Regularization for Consistent Question Answering},
  author={Asai, Akari and Hajishirzi, Hannaneh},
  booktitle={ACL},
  year={2020}
}
```
In the paper, we introduce logic-guided data augmentation and regularization to improve accuracy and consistency in a range of question answering datasets, namely [WIQA](https://allenai.org/data/wiqa), [QuaRel](https://allenai.org/data/quarel) and [HotpotQA (comparison)](https://hotpotqa.github.io/). This repository contains example codes for WIQA and HotpotQA. 
<p align="center"><img width="65%" src="img/qa_inconsistency.jpg" /></p>


*Acknowledgements*: To implement our RoBERTa-based baselines for WIQA and HotpotQA, we used the [Hugging Face's transformers](https://github.com/huggingface/transformers) library. Huge thanks to the contributors of Hugging Face library! [Niket](https://allenai.org/team/nikett) helps us to reproduce the original baseline results from [the WIQA paper](https://arxiv.org/abs/1909.04739). 

*Currently this repository contains example codes for WIQA. Codes for other datasets will be added soon.* 


## 0. Setup 
### Install python packages
Run the command below to install python packages.
```
pip -r requirements.txt
```

### Download data
Run `download.sh` to download the original WIQA. 

```
bash download.sh
```

## 1. Data augmentation
After download the data, run the command below.

### WIQA 

```
python wiqa_augmentation.py --data_dir PATH_TO_WIQA_DATA_DIR --output_dir PATH_TO_AUGMENTED_DATA
```

Optional argument: 
```
optional arguments:
  --store_dev_test_augmented_data
                        Augment eval data. By default we keep eval data as is.
  --sample_ratio SAMPLE_RATIO
                        the random sample rate for original data to be used.
  --sample_ratio_augmentation SAMPLE_RATIO_AUGMENTATION
                        the random sample rate for augmented data to be added.
  --eval_mode           with eval mode, we only add the question included in
                        the original data.
```


## 2. Training
You can train our models from scratch or download the checkpoints we trained. The pre-trained weights can be downloaded from [here](https://drive.google.com/drive/folders/1ORfqbYNtuGRMRmJ2PWpd7RwAWMyEcEjP?usp=sharing). 

### WIQA 
You can train our RoBERTa baseline, RoBERTa+DA, and RoBERTa + DA + Regularization models using the commands below.  

You can reduce the number of `gradient_accumulation_steps` if you use multiple GPUs. We set the `per_gpu_train_batch_size` to fit a single GPU with 11 GB GPU Memory, and you can increase the number.   

Please refer the additional details of optional arguments in [run_classification_consistency.py](run_classification_consistency.py) or run `python run_classification_consistency.py -h`.

#### RoBERTa (base) baseline model

```
python run_classification_consistency.py \
--model_type roberta \
--model_name_or_path roberta-base \
--task_name wiqa \
--do_train \
--data_dir PATH_TO_WIQA_DATA_DIR \
--max_seq_length 256 --per_gpu_eval_batch_size=8 \
--per_gpu_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--learning_rate 2e-5 \
--weight_decay 0.01 \
--output_dir PATH_TO_WIQA_MODEL_OUTPUT_DIR \
--seed 789 
```

#### RoBERTa + Data Augmentation model

```
python run_classification_consistency.py \
--model_type roberta \
--model_name_or_path roberta-base \
--task_name wiqa \
--do_train \
--data_dir PATH_TO_AUGMENTED_DATA \
--max_seq_length 256 --per_gpu_eval_batch_size=8 \
--per_gpu_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--learning_rate 2e-5 \
--weight_decay 0.01 \
--output_dir PATH_TO_WIQA_MODEL_OUTPUT_DIR \
--seed 789 
```

#### RoBERTa + Data Augmentation + Consistency model
To train consistency models, you first need to train a model without consistency loss (for the *annealing* discussed in [Section 3](https://arxiv.org/abs/2004.10157) in our paper). You can either set the `--lambda_a` and `--lambda_b` to 0 for the first 3 epochs, or start from the checkpoint of the RoBERTa baseline trained for 3 epochs.

```
python run_classification_consistency.py \
--model_type roberta_cons \
--model_name_or_path PATH_TO_BASELINE_MODEL_CHECKPOINTS_DIR \
--task_name wiqa --do_train \
--data_dir PATH_TO_AUGMENTED_DATA \
--max_seq_length 256 --per_gpu_eval_batch_size=4 \
--output_dir PATH_TO_WIQA_MODEL_OUTPUT_DIR \
--per_gpu_train_batch_size=4 --gradient_accumulation_steps=16 \
--learning_rate 2e-5 --num_train_epochs 3 \
--weight_decay 0.01 \
--lambda_a 0.5 --lambda_b 0.1 \
--seed 789 \
--use_consistency 
```

*Note:* We've noticed that models (BERT, RoBERTa) are sensitive to some hyperparameter on WIQA. We have conducted intensive hyperparameter search at the beginning of our project, and use the same hyperparameter performing best with our RoBERTa baseline model throughout our experiments. Several recent papers (e.g., [Dodge et al. (2020)](https://arxiv.org/abs/2002.06305), [Bisk et al. (2020)](https://arxiv.org/abs/1911.11641)) discuss those sensitivity. There might be performance variance with different number of the training batch size.

## 3. Evaluation

### WIQA 
Run the command below (same for the RoBERTa baseline, RoBERTa + DA andRoBERTa + Data Augmentation + Consistency model). To test the performance on eval data, you can simply replace the `do_eval` option with `do_test`. 

```
python run_classification_consistency.py \
--model_type roberta \
--model_name_or_path PATH_TO_WIQA_MODEL_OUTPUT_DIR \
--task_name wiqa \
--do_eval \
--data_dir PATH_TO_WIQA_DATA_DIR \
--max_seq_length 256 --per_gpu_eval_batch_size=8 \
--weight_decay 0.01 \
--output_dir PATH_TO_WIQA_MODEL_OUTPUT_DIR \
--seed 789 
```


## 4. Contact
Please contact Akari Asai (Twitter:[@AkariAsai](https://twitter.com/AkariAsai), Email: alari[at]cs.washington.edu) for questions and suggestions.