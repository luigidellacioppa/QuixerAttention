import torch
import math
from define_qlm import evaluate, create_model, setup_dataset

device_name = 'cpu'  # Change this if you want to use a different device
device = torch.device(device_name)

# Define the hyperparameters for the models
lstm_hparams = {
    "layers": 2,
    "window": 32,
    "residuals": False,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.30,
    "lr": 0.002,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "print_iter": 50
}

cdimensions = [96, 128]
model_map = {
    "LSTM": (lstm_hparams, cdimensions),
}

# Load the dataset
vocab, (train_iter, val_iter, test_iter), PAD_TOK = setup_dataset(
    device,
    lstm_hparams["batch_size"],
    lstm_hparams["window"]
)

for model_name, meta in model_map.items():
    fix_hyperparams, dimensions = meta
    for dim in dimensions:
        for seed in torch.randint(high=1000000, size=(10,)).tolist():
            fix_hyperparams["model"] = model_name
            fix_hyperparams["dimension"] = dim
            fix_hyperparams["seed"] = seed

            # Create the model
            model = create_model(fix_hyperparams, device, len(vocab))

            # Load the saved model
            checkpoint_fpath = f"./trained_models/q_transformer_lm_{fix_hyperparams['model']}_{fix_hyperparams['seed']}_{int(time.time())}.pt"
            model.load_state_dict(torch.load(checkpoint_fpath))

            # Evaluate the model on the test set
            test_loss = evaluate(
                model,
                test_iter,
                torch.nn.CrossEntropyLoss(),
                fix_hyperparams["window"],
                PAD_TOK,
                device,
                fix_hyperparams["batch_size"]
            )

            print(f"Test Loss for model {model_name} with dimension {dim} and seed {seed}: {test_loss:.3f} | Test ppl: {math.exp(test_loss)}")