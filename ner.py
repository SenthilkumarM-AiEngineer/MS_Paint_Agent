# ner.py
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import re, string

class NER:
    # Named Entity Recognition (NER) Training
    def train():
        import pandas as pd
        import transformers
        from transformers import TrainingArguments, Trainer
        import numpy as np
        from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForTokenClassification, pipeline
        from transformers import TrainingArguments, Trainer
        from transformers import AutoTokenizer, pipeline
        from datasets import Dataset, DatasetDict
        from evaluate import load

        # CSV DATA
        df = pd.read_excel("data/training/ner.xlsx")
        
        # DATA PREPROCESSING
        df["ner_tags"] = df["ner_tags"].astype(str)
        df["tokens"] = df["tokens"].astype(str)

        df["tokens"] = df["tokens"].apply(lambda x: [token.strip(" ") for token in x.strip('[]').split(',')])
        df["ner_tags"] = df["ner_tags"].apply(lambda x: [int(tag) for tag in x.strip('[]').split(',')])

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)


        # Convert DataFrames to Datasets
        train_dataset = Dataset.from_pandas(df.iloc[:int(0.80 * len(df))])
        validation_dataset = Dataset.from_pandas(df.iloc[int(0.80 * len(df)):])
        test_dataset = Dataset.from_pandas(df.iloc[int(0.80 * len(df)):])

        # Create DatasetDict
        dataset = DatasetDict({
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset
        })


        # MODEL
        task = "ner" # Should be one of "ner", "pos" or "chunk"
        model_checkpoint = "distilbert-base-uncased"

        # data tokenization
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

        label_all_tokens = True

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

            labels = []
            for i, label in enumerate(examples[f"{task}_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(label[word_idx] if label_all_tokens else -100)
                    previous_word_idx = word_idx

                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)


        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

            labels = []
            for i, label in enumerate(examples[f"{task}_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        label_ids.append(label[word_idx] if label_all_tokens else -100)
                    previous_word_idx = word_idx

                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        # labels
        label_list = ["O", "SHAPE", "SIZE", "B-POSITION", "I-POSITION"]

        id2label = {i:label for i, label in enumerate(label_list)}
        label2id = {label:i for i, label in enumerate(label_list)}

        # model
        model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, id2label=id2label, label2id=label2id)

        # training arguments
        batch_size = 8
        args = TrainingArguments(
            "models/ner/checkpoint",
            eval_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=10,
            weight_decay=0.01,
            push_to_hub=False,
        )

        # data collator
        data_collator = DataCollatorForTokenClassification(tokenizer)

        # compute metrics
        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        metric = load("seqeval")

        # MODEL TRAINER
        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()

        trainer.evaluate()

        # Directory where the model and tokenizer will be saved
        save_directory = f"models/ner"

        # Save the trained model
        trainer.model.save_pretrained(save_directory)

        # Save the tokenizer associated with your model
        tokenizer.save_pretrained(save_directory)

        return 'Success'
        
    
    @staticmethod
    def process_entities(classified_data, groups):
        entities = {}

        # Normalize group names
        groups = [g.lower() for g in groups]

        for group in groups:
            # Get all entities that match this group
            matched = [item for item in classified_data if item['entity_group'].lower() == group]

            if matched:
                if group == 'position':
                    # Collect all position words as a list (capitalize each)
                    pos_list = [m['word'].strip().capitalize() for m in matched]

                    # Handle case like "top left" as a single phrase
                    split_positions = []
                    for p in pos_list:
                        # Split if multi-word like "top left"
                        split_positions.extend([x.capitalize() for x in p.split()])

                    # Store the list directly
                    entities['position'] = split_positions

                else:
                    # Take the first (or highest score) entity for other groups
                    best = max(matched, key=lambda x: x['score'])
                    entities[group] = best['word'].strip().lower()

        return entities
        
    
    # Named Entity Recognition (NER) Classification
    def predict(user_input):
        def preprocess_text(text: str) -> str:
            if not isinstance(text, str):
                return ""
            
            # Convert to lowercase
            text = text.lower()
            # Replace punctuation with space
            text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
            # Normalize multiple spaces to single space
            text = re.sub(r"\s+", " ", text).strip()
            return text
        
        user_input = preprocess_text(user_input)
    
        # List of entities for the intent
        result = [{'entity_types': ["O", "SHAPE", "SIZE", "B-POSITION", "I-POSITION"]}]

        groups = list(dict.fromkeys(item[2:].lower() if item[:2].lower() in ('b-', 'i-', 'B-', 'I-') else item.lower() for item in result[0]['entity_types'][1:]))
        
        # If Entities for the intent
        if groups:
            # Load the model and tokenizer from your local directory
            model_path = "models/ner"
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            task = "token-classification"

            # Create a token classification pipeline
            token_classifier = pipeline(task, model=model, tokenizer=tokenizer, aggregation_strategy="simple")

            # Get predictions
            classified_data = token_classifier(user_input)

            processed_entities = NER.process_entities(classified_data, groups)
        
        # If no entities for the intent
        elif not groups:
            processed_entities = {}
        
        return processed_entities
    
# train = NER.train()
# print(train)
# while True:
#     user_input = input("Enter: ")
#     predict = NER.predict(user_input)
#     print(predict)