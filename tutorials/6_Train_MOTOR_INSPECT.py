#!/usr/bin/env python
# coding: utf-8

# # Train MOTOR
# 
# This tutorial walks through the various steps to train a MOTOR model.
# 
# Training MOTOR is a four step process:
# 
# - Training a tokenizer
# - Prefitting MOTOR
# - Preparing batches
# - Training the model

import shutil
import os

os.environ["HF_DATASETS_CACHE"] = '/share/pi/nigam/projects/zphuo/.cache'
os.environ["WANDB_DISABLED"] = "true"

TARGET_DIR = 'trash/tutorial_6_INSEPCT'

from_pretrained = True
num_proc = 20


if not from_pretrained:
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)

    os.mkdir(TARGET_DIR)
    os.mkdir(os.path.join(TARGET_DIR, 'motor_model'))


import datasets
import femr.index
import femr.splits

# First, we want to split our dataset into train, valid, and test
# We do this by calling our split functionality twice

# dataset = datasets.Dataset.from_parquet('input/meds/data/*')
parquet_folder = '/share/pi/nigam/projects/zphuo/data/PE/inspect/timelines_smallfiles_meds/data_subset/*'
dataset = datasets.Dataset.from_parquet(parquet_folder)


index = femr.index.PatientIndex(dataset, num_proc=num_proc)
main_split = femr.splits.generate_hash_split(index.get_patient_ids(), 97, frac_test=0.15)


# Note that we want to save this to the target directory since this is important information

main_split.save_to_csv(os.path.join(TARGET_DIR, "motor_model", "main_split.csv"))

import pandas as pd
label_csv_subset = '/share/pi/nigam/projects/zphuo/data/PE/inspect/timelines_smallfiles_meds/cohort_0.2.0_master_file_anon_subset.csv'
label_df = pd.read_csv(label_csv_subset)
label_df = label_df[['patient_id', 'split', ]]
inspect_split_csv = '/share/pi/nigam/projects/zphuo/repos/femr/tutorials/trash/tutorial_6_INSEPCT/motor_model/main_split.csv'
label_df.to_csv(inspect_split_csv, index=False)

train_split = femr.splits.generate_hash_split(main_split.train_patient_ids, 87, frac_test=0.15)

# print(train_split.train_patient_ids)
# print(train_split.test_patient_ids)

main_dataset = main_split.split_dataset(dataset, index)
train_dataset = train_split.split_dataset(main_dataset['train'], femr.index.PatientIndex(main_dataset['train'], num_proc=num_proc))

# print(train_dataset)


import femr.models.tokenizer
from femr.models.tokenizer import FEMRTokenizer
import pickle

# First, we need to train a tokenizer
# Note, we need to use a hierarchical tokenizer for MOTOR


with open('input/meds/ontology.pkl', 'rb') as f:
    ontology = pickle.load(f)


if not from_pretrained:
    tokenizer = femr.models.tokenizer.train_tokenizer(
        main_dataset['train'], vocab_size=128, is_hierarchical=True, num_proc=num_proc, ontology=ontology)

    # Save the tokenizer to the same directory as the model
    tokenizer.save_pretrained(os.path.join(TARGET_DIR, "motor_model"))

else:
    # load pretrained tokenizer
    tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(os.path.join(TARGET_DIR, "motor_model"), ontology=ontology)


import femr.models.tasks

if 'subset' in parquet_folder:
    num_tasks = 39
else:
    num_tasks = 64

# Second, we need to prefit the MOTOR model. This is necessary because piecewise exponential models are unstable without an initial fit

motor_task = femr.models.tasks.MOTORTask.fit_pretraining_task_info(
    main_dataset['train'], tokenizer, num_tasks=num_tasks, num_bins=4, final_layer_size=32, num_proc=num_proc)


# It's recommended to save this with pickle to avoid recomputing since it's an expensive operation


import femr.models.processor
import femr.models.tasks

# Third, we need to create batches. 

processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)

# We can do this one patient at a time
print("Convert a single patient")
example_batch = processor.collate([processor.convert_patient(train_dataset['train'][0], tensor_type='pt')])

print("Convert batches")
# But generally we want to convert entire datasets
train_batches = processor.convert_dataset(train_dataset, tokens_per_batch=32, num_proc=num_proc, min_samples_per_batch=1)

print("Convert batches to pytorch")
# Convert our batches to pytorch tensors
train_batches.set_format("pt")
print("Done")


patient_id = 646017672

index_train = femr.index.PatientIndex(train_dataset['train'], num_proc=num_proc)

example_batch = processor.collate([processor.convert_patient(train_dataset['train'][index_train.get_index(patient_id)], tensor_type='pt')])


import torch
torch.flatten(example_batch['batch']['task']['is_event']).shape


print(len(train_batches['train']['patient_ids']))
for id in train_batches['train']['patient_ids']:
    print(len(id), set(id))


for batch in train_batches['train']:
    print(batch)
    break


largest_vocab = 0
for n in range(1100):
    example_batch = processor.collate([processor.convert_patient(train_dataset['train'][n], tensor_type='pt')])
    print(example_batch['batch']['task']['is_event'].shape)
    if example_batch['batch']['task']['is_event'].shape[1] > largest_vocab:
        largest_vocab = example_batch['batch']['task']['is_event'].shape[1]
        print(largest_vocab)


example_batch['batch']


example_batch['batch']['task']['is_event'].shape


for key in example_batch['batch']['transformer']:
    try:
        print(key, example_batch['batch']['transformer'][key].shape)
        print(example_batch['batch']['transformer'][key])
    except:
        print(key, len(example_batch['batch']['transformer'][key]))
        print(example_batch['batch']['transformer'][key])


for key in example_batch['batch']:
    try:
        print(key, example_batch['batch'][key].shape)
        print(example_batch['batch'][key])
    except:
        print(key, len(example_batch['batch'][key]))
        print(example_batch['batch'][key])


example_batch['batch']


import transformers

# Finally, given the batches, we can train CLMBR.
# We can use huggingface's trainer to do this.

transformer_config = femr.models.transformer.FEMRTransformerConfig(
    vocab_size=tokenizer.vocab_size, 
    is_hierarchical=tokenizer.is_hierarchical, 
    n_layers=2,
    hidden_size=64, 
    intermediate_size=64*2,
    n_heads=8,
)

config = femr.models.transformer.FEMRModelConfig.from_transformer_task_configs(transformer_config, motor_task.get_task_config())

model = femr.models.transformer.FEMRModel(config)

collator = processor.collate

trainer_config = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    output_dir='tmp_trainer',
    remove_unused_columns=False,
    num_train_epochs=100,

    eval_steps=20,
    evaluation_strategy="steps",

    logging_steps=20,
    logging_strategy='steps',

    prediction_loss_only=True,
    
    report_to=None,
)


transformer_config.vocab_size


transformer_config.hidden_size


trainer = transformers.Trainer(
    model=model,
    # data_collator=processor.collate,
    train_dataset=train_batches['train'],
    eval_dataset=train_batches['test'],
    args=trainer_config,
)


trainer.train()

model.save_pretrained(os.path.join(TARGET_DIR, 'motor_model'))