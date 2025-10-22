# action.py
import warnings
warnings.filterwarnings("ignore")
import torch
import pandas as pd
# transformers 4.22.1 - install this if you use tensorflow in this project
from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re, string
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


class Action:
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

    def train():

        # Get data
        df = pd.read_csv('data/training/action.csv', encoding='cp1252')
        df['Text'] = df['Text'].apply(Action.preprocess_text)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.rename(columns={'Text': 'text', 'Category': 'category'}, inplace=True)
        labels = df['category'].unique().tolist()
        labels = [s.strip() for s in labels]

        tokenizer = BertTokenizerFast.from_pretrained("distilbert-base-uncased", max_length=512)

        NUM_LABELS= len(labels)
        id2label={id:label for id,label in enumerate(labels)}
        label2id={label:id for id,label in enumerate(labels)}
        df["labels"]=df.category.map(lambda x: label2id[x.strip()])

        SIZE = df.shape[0]
        train_size = int(0.7 * SIZE)

        # Splitting the data
        train_texts = list(df.text[:train_size])
        train_labels = list(df.labels[:train_size])
        val_texts = list(df.text[train_size:])
        val_labels = list(df.labels[train_size:])

        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings  = tokenizer(val_texts, truncation=True, padding=True)

        class DataLoader(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                # Retrieve tokenized data for the given index
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                # Add the label for the given index to the item dictionary
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)
            

        train_dataloader = DataLoader(train_encodings, train_labels)
        val_dataloader = DataLoader(val_encodings, val_labels)

        model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)
        model.to('device')

        def compute_metrics(pred):
            # Extract true labels from the input object
            labels = pred.label_ids

            # Obtain predicted class labels by finding the column index with the maximum probability
            preds = pred.predictions.argmax(-1)

            # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

            # Calculate the accuracy score using sklearn's accuracy_score function
            acc = accuracy_score(labels, preds)

            # Return the computed metrics as a dictionary
            return {
                'Accuracy': acc,
                'F1': f1,
                'Precision': precision,
                'Recall': recall
            }

        training_args = TrainingArguments(
            # The output directory where the model predictions and checkpoints will be written
            output_dir='models/classification/checkpoint', 
            do_train=True,
            do_eval=True,
            #  The number of epochs, defaults to 3.0 
            num_train_epochs=35,
            per_device_train_batch_size=16,  
            per_device_eval_batch_size=32,
            # Number of steps used for a linear warmup
            warmup_steps=100,
            weight_decay=0.01,
            logging_strategy='steps',
            # TensorBoard log directory                 
            logging_dir='models/classification/multi-class-logs',
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps", 
            load_best_model_at_end=True
        )

        trainer = Trainer(
            # the pre-trained model that will be fine-tuned
            model=model,
            # training arguments that we defined above
            args=training_args,
            train_dataset=train_dataloader,
            eval_dataset=val_dataloader,
            compute_metrics= compute_metrics
        )

        trainer.train()

        metrics = trainer.evaluate()
        print("Metrics:", metrics)

        # Directory where the model and tokenizer will be saved
        save_directory = "models/classification/"

        # Save the trained model
        trainer.model.save_pretrained(save_directory)

        # Save the tokenizer associated with your model
        tokenizer.save_pretrained(save_directory)

        return 'Success'


    def predict(user_input):
        user_input = Action.preprocess_text(user_input)

        # Load the model and tokenizer from your local directory
        model_path = 'models/classification/'
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # tokenizer = BertTokenizerFast.from_pretrained(model_path)
        # model = BertForSequenceClassification.from_pretrained(model_path)
        task = "text-classification"

        # Create a token classification pipeline
        text_classifier = pipeline(task, model=model, tokenizer=tokenizer)

        # Get predictions
        action = text_classifier(user_input)
        action = action[0]['label']
        return action


# train = Action.train()
# print(train)
# while True:
#     user_input = input("Enter: ")
#     action = Action.predict(user_input)
#     print(action)