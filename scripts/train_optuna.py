import optuna
from ultralytics import YOLO


# Define the objective function for optimization
def objective(trial):
    # Suggest hyperparameters for tuning
    epochs = trial.suggest_int('epochs', 30, 100)
    batch_size = trial.suggest_int('batch_size', 8, 32)
    img_size = trial.suggest_categorical('img_size', [320, 416, 512, 640])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    momentum = trial.suggest_uniform('momentum', 0.85, 0.95)

    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Train the model with the current set of hyperparameters
    model.train(
        data=r"C:\Users\sierr\Documents\Uni\TFM\data\datasets\minimal_DATASET\cfg.yaml",
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        lr0=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum
    )

    # Evaluate the model on validation data
    val_results = model.val()

    # Use mean Average Precision (mAP) as the evaluation metric
    mAP = val_results.metrics['mAP_0.5']

    return mAP


# Create and optimize an Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters found
print('Best hyperparameters: ', study.best_params)
