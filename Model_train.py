import os
import numpy as np
import random
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, TrainingArguments, Trainer
from datasets import load_metric, load_from_disk
from scipy.special import softmax
import argparse


# Initialize the parser
parser = argparse.ArgumentParser(description="JEDIS Training")

# Add arguments with default values
parser.add_argument('-m', '--maxn', type=int, default=5, help='neg STMS ratio')
parser.add_argument('-r', '--ratio', type=int, default=2, help='neg non_STMS ratio')
parser.add_argument('-s', '--seed', type=int, default=123, help='random seed')
parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size')
parser.add_argument('-tbs', '--test_batch_size', type=int, default=2048, help='test batch size')
parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu number')
parser.add_argument('-a', '--alpha', type=float, default=0.9, help='alpha value')
parser.add_argument('-d', '--dropout', type=float, default=0.2, help='dropout probability')
parser.add_argument('-pt', '--pretrained', type=bool, default=True, help='use pretrained model')
parser.add_argument('-ar', '--range', type=str, default='[0.9, 0.99]', help='alpha learnable range') # in the form of [a,b]
parser.add_argument('-i', '--input', type=str, default=None, help='train dataloader')
parser.add_argument('-o', '--output', type=str, default=None, help='output path')
parser.add_argument('-t', '--train', type=bool, default=False, help='train mode')

# Parse the arguments
args = parser.parse_args()

# print all arguments with its name and value  
print('[Arguments is]')
for arg in vars(args):
    print(arg, getattr(args, arg))

data_dir = './data'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu}"

ratio, maxn, random_seed, dropout = args.ratio, args.maxn, args.seed, args.dropout
batch_size = args.batch_size
test_batch_size = args.test_batch_size
train_mode = args.train
torch.manual_seed(random_seed)
random.seed(random_seed)

if args.pretrained is True:
    bert_path = os.path.join(data_dir,'BERT_pretrained_reddit')
else:
    bert_path = 'bert-base-uncased'

if args.output is None:
    path = f'./output/alpha{args.range}_pretrained{args.pretrained}'
else:
    path = args.output

    
name = f'ratio{ratio}_max{maxn}'

print(f'[Output Path is: {path}]')
print(f'[{name}]')


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

class CustomDataCollator:
    def __init__(self, tokenizer):
        self.data_collator_with_padding = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        # Extract standard fields
        standard_features = [{key: feature[key] for key in ['input_ids', 'attention_mask']} for feature in features]
        # Handle padding using DataCollatorWithPadding
        batch = self.data_collator_with_padding(standard_features)

        # Add custom fields
        batch['emb'] = torch.tensor([feature['emb'] for feature in features])
        batch['labels'] = torch.tensor([feature['label'] for feature in features])
        return batch

data_collator = CustomDataCollator(tokenizer=tokenizer)

## Define compute metrics
def compute_metrics(eval_pred): # softmax
    metric_list = ['accuracy', 'precision', 'recall', 'f1']

    (logits1, logits2, alpha), labels = eval_pred

    alpha = alpha[0] ## original alpha is a from of [val] * eval_steps
    # Convert logits to probabilities
    probs_part1 = softmax(logits1, axis=-1)
    probs_part2 = softmax(logits2, axis=-1)

    # Combine the probabilities using the learnable alpha
    combined_probs = alpha * probs_part1 + (1 - alpha) * probs_part2

    # Make predictions
    predictions = np.argmax(combined_probs, axis=-1)

    result = dict()
    for m in metric_list:
        metric = load_metric(m)
        result.update(metric.compute(predictions=predictions, references=labels))
    return result



if args.input is None:
    tokenized_dataset = load_from_disk(os.path.join(data_dir, f'Emb_Dataset_ratio{ratio}_max{maxn}_{random_seed}'))
else:
    tokenized_dataset = load_from_disk(args.input)

print('[Dataset Loaded]')
print(tokenized_dataset)


ranges = [float(val) for val in args.range[1:-1].split(',')]
class Model(nn.Module):
    def __init__(self, hidden_dim=768):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path, return_dict=False)
        self.hidden_dim = hidden_dim
        self.activation = nn.Tanh() # Activation fn
        self.dropout = nn.Dropout(args.dropout)
        # Define alpha as a learnable parameter, initialized to alpha_value
        self.unconstrained_alpha = nn.Parameter(torch.tensor(0.0)) # Unconstrained variable
        # Feed forward network for processing word embedding 
        self.ffn_word_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Layer Normalization
            self.activation,  # Activation function 
            self.dropout,
        )
        # Linear layer for binary classification 
        self.fc_part1 = nn.Linear(hidden_dim, 2, bias=False)
        self.fc_part2 = nn.Linear(hidden_dim, 2, bias=False)

    def forward(self, input_ids, attention_mask, emb, labels):

        alpha = ranges[0] + (ranges[1]-ranges[0])*(torch.tanh(self.unconstrained_alpha)+1)/2 ## For making range of input range

        # Make word embeddings to tensor
        embeddings = torch.tensor(emb)

        # Part 1
        # Run BERT on input_ids and attention_masks
        output = self.bert(input_ids, attention_mask=attention_mask)
        part1_out = output[1]  # pooled output of [CLS] token embedding

        # Part 2
        # Embeddings are assumed to be a tensor of shape (batch_size, hidden_dim)
        part2_out = self.ffn_word_embedding(embeddings)

        logits_part1 = self.fc_part1(part1_out) # Logits for Part 1
        logits_part2 = self.fc_part2(part2_out) # Logits for Part 2

        return logits_part1, logits_part2, alpha


model = Model()

## Define Custom trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, loss_fn=nn.CrossEntropyLoss()):
        labels = inputs['labels']
        logits_part1, logits_part2, alpha = model(**inputs)  # Assuming the model returns two logits and alpha

        loss1 = loss_fn(logits_part1, labels)
        loss2 = loss_fn(logits_part2, labels)

        total_loss = alpha*loss1 + (1-alpha)*loss2

        out = {'logits': (logits_part1, logits_part2, alpha)} ## the last element of logits will be alpha

        return (total_loss, out) if return_outputs else total_loss

class CustomTrainer_eval(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, loss_fn=nn.CrossEntropyLoss()):
        labels = inputs['labels']
        logits_part1, logits_part2, alpha = model(**inputs)  # Assuming the model returns two logits and alpha

        alpha = alpha.expand(len(logits_part1)) #### only for eval? or not?

        loss1 = loss_fn(logits_part1, labels)
        loss2 = loss_fn(logits_part2, labels)

        total_loss = alpha*loss1 + (1-alpha)*loss2

        out = {'logits': (logits_part1, logits_part2, alpha)} ## the last element of logits will be alpha

        return (total_loss, out) if return_outputs else total_loss


training_args = TrainingArguments(
    output_dir = os.path.join(path,name),
    do_train=True,
    do_eval=True,
    learning_rate=1e-5, 
    warmup_ratio = 0.1,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=test_batch_size,
    num_train_epochs = 10,
    save_strategy = 'steps',
    load_best_model_at_end=True,
    seed=random_seed,
    logging_steps=2000,
    eval_steps = 2000,
    save_steps=2000,
    remove_unused_columns=False,
    report_to="none",
    # disable_tqdm=True,
    )

trainer = CustomTrainer(
    model=model, args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['valid'],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics) ## use softmax with compute metrics as default

save_dir = os.path.join('./BEST_model', name)
if train_mode:
    print('[Start training]')
    trainer.train()
    print('[Training finished]')
    trainer.save_model(save_dir)
    print(f'[Model saved to {save_dir}]')
else:
    print('[No training]')
model_path = os.path.join(save_dir, 'pytorch_model.bin')

# Load model
model_best = Model()
model_best.load_state_dict(torch.load(model_path))
model_best.eval()

# Load tokenized dataset
tokenized_dataset_eval = load_from_disk(os.path.join(data_dir, 'Emb_Dataset_eval'))

# Define a function to make the process more concise
def run_evaluation(metric_fn):
    print(f'[Start testing for {metric_fn.__name__}]')
    training_args = TrainingArguments(os.path.join('./test_runs', name), 
                                      do_eval=True, do_predict=True,
                                      per_device_eval_batch_size=test_batch_size,
                                      seed=random_seed,
                                      logging_steps=1)
    trainer = CustomTrainer_eval(model=model_best, args=training_args,
                            tokenizer=tokenizer,
                            data_collator=data_collator,
                            compute_metrics=metric_fn)
    return trainer.predict(test_dataset=tokenized_dataset_eval['evaluation'])

# Get results
eval_outs1 = run_evaluation(compute_metrics) 
print(eval_outs1)

print('[Test finished]')

# # Save results
# with open(os.path.join(save_dir, 'eval_output.txt'), 'w') as file:
#     print(f"Result:\n{eval_outs1}", file=file)
# print('[Result saved]')

