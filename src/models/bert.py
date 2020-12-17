import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW, \
    get_cosine_with_hard_restarts_schedule_with_warmup
from tqdm import tqdm

from src.evaluate import get_accuracy
from src.data_loading import seed_everything

MODEL_FOLDER = './../models/'
seed_everything()


def run_bert(train_tweets, val_tweets, save_model,
             learning_rate=5e-6,
             model_name='bert',
             epochs=1):
    """
    Train a neural network with bert tokens.

    :param train_tweets: np.array with the features
    :param val_tweets: np.array with the features
    :param save_model: bool
    :param learning_rate: float
    :param model_name: str.
    :param epochs: int
    """
    print("\n" + "-" * 100)
    print("MODEL TO RUN: Neural network with bert tokens")
    print("-" * 100 + "\n")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Creating batches...")
    train_dataloader = DataLoader(train_tweets, shuffle=True, batch_size=32)
    validation_dataloader = DataLoader(val_tweets, shuffle=False, batch_size=32)

    model = AutoModelForSequenceClassification.from_pretrained(
        'digitalepidemiologylab/covid-twitter-bert',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,  # default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # default is 1e-8.
                      )

    total_steps = len(train_dataloader) * epochs
    warmup = len(train_dataloader) * 0.01
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                   num_warmup_steps=warmup,
                                                                   num_training_steps=total_steps)

    for epoch in range(epochs):
        train_loss = 0

        model.train()  # training mode
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.zero_grad()  # clear any previously calculated gradients

            inputs = {'input_ids': batch[0].to(device),
                      'attention_mask': batch[1].to(device),
                      'labels': batch[2].to(device),
                      'token_type_ids': None}
            outputs = model(**inputs)  # forward pass

            loss = outputs[0]
            loss.backward()  # backward pass to calculate the gradients.
            train_loss += loss.item()

            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()  # Update parameters
            scheduler.step()  # Update the learning rate.

        if save_model:
            print("Saving model...")
            path = MODEL_FOLDER + model_name + "_" + str(epoch)
            model.save_pretrained(path)
            print(path)
            print("Done!")

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss, avg_val_accuracy = validate_model(model,
                                                        validation_dataloader, device)

        print("EPOCH: {epoch}. Losses: train = {train}, val = {val}. \
            Accuracy: {acc}".format(epoch=epoch, train=avg_train_loss,
                                    val=avg_val_loss, acc=avg_val_accuracy))


def validate_model(model, validation_dataloader, device):
    """
    Uses validation batch to calculate loss and accurcay per epoch
    :param model: pytorch model
    :param validation_dataloader: pytorch DataLoader class with validation tokens
    :param device: tells pytorch to run in cuda or cpu
    """
    val_loss = 0
    val_accuracy = 0
    num_val_batch = 0
    model.eval()  # training modevvvvv
    epoch_iterator = tqdm(validation_dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        with torch.no_grad():
            inputs = {'input_ids': batch[0].to(device),
                      'attention_mask': batch[1].to(device),
                      'labels': batch[2].to(device),
                      'token_type_ids': None}

            outputs = model(**inputs)  # forward pass

            tmp_eval_loss, logits = outputs[:2]

            val_loss += tmp_eval_loss.mean().item()

            binary_preds = logits.argmax(axis=1).detach().cpu().numpy()
            labels = batch[2].to('cpu').numpy()

            val_accuracy += get_accuracy(binary_preds, labels)
            num_val_batch += 1

    avg_val_loss = val_loss / len(validation_dataloader)
    avg_val_accuracy = val_accuracy / num_val_batch
    return avg_val_loss, avg_val_accuracy


def predict_bert(dataset, model_name='bert_0'):
    """
    Uses pre-trained model to make predictions on the test dataset
    :param dataset: pytorch DataLoader class
    :param model_name: str name of model
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_dataloader = DataLoader(dataset, shuffle=False, batch_size=32)

    final_model_name = MODEL_FOLDER + model_name
    model = AutoModelForSequenceClassification.from_pretrained(
        final_model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)
    model.eval()

    binary_preds_list = []
    test_ids_list = []

    epoch_iterator = tqdm(test_dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'token_type_ids': None}
        with torch.no_grad():
            outputs = model(**inputs)  # forward pass

            logits = outputs[0]

            binary_preds = logits.argmax(axis=1).detach().cpu().numpy()
            test_ids = batch[2].to('cpu').numpy()

            binary_preds_list.append(binary_preds)
            test_ids_list.append(test_ids)

    return test_ids_list, binary_preds_list
