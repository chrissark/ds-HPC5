from datasets import load_dataset
import time
import deepspeed


datasets = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir='./wikitexts')


from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('./distgpt2-tokenizer/')
model = GPT2LMHeadModel.from_pretrained('./distgpt2')

def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    "test-clm",
    learning_rate=2e-5,
    num_train_epochs=1,
    deepspeed = "./ds_config.json"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
#   eval_dataset=lm_datasets["validation"],
)

start = time.time()
trainer.train()
end = time.time()


#import math
#eval_results = trainer.evaluate()
#print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

print ("Time per epoch: ", round((end - start), 2))
