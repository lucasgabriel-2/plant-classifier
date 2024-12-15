from keras import layers, Sequential, utils

import os
import json
import matplotlib.pyplot as plt

def save_training_results(model, history, test_loss, test_accuracy):
    # Create 'res' directory if it doesn't exist
    os.makedirs("res", exist_ok=True)

    # Save model architecture
    model_json = model.to_json()
    with open("res/model_architecture.json", "w") as json_file:
        json_file.write(model_json)

    # Save model weights
    model.save_weights("res/model_weights.h5")

    # Save the entire model (architecture + weights)
    model.save("res/model_full.h5")

    # Save training history
    history_dict = history.history
    with open("res/training_history.json", "w") as json_file:
        json.dump(history_dict, json_file)

    # Save evaluation results
    evaluation_results = {"loss": test_loss, "accuracy": test_accuracy}
    with open("res/evaluation_results.json", "w") as json_file:
        json.dump(evaluation_results, json_file)

    # Plot and save training and validation loss and accuracy
    def plot_history(history, metric, title, file_name):
        plt.figure(figsize=(8, 6))
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(f"res/{file_name}")
        plt.close()

    plot_history(history, "accuracy", "Training and Validation Accuracy", "accuracy_plot.png")
    plot_history(history, "loss", "Training and Validation Loss", "loss_plot.png")




image_size = (150, 150) 
batch_size = 32  

# Criar os datasets de treino e validação
train_ds = utils.image_dataset_from_directory(
    "Nonsegmented",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = utils.image_dataset_from_directory(
    "Nonsegmented",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Pré-processamento 
normalization_layer = layers.Rescaling(1./255)

# Criar o modelo
model = Sequential([
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(12, activation='softmax')  # 12 classes, uma para cada subdiretório
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    batch_size=batch_size,
)

# Avaliar o modelo no conjunto de validação
test_loss, test_accuracy = model.evaluate(val_ds, verbose='1')
print(f"Test accuracy: {test_accuracy:.2f}")


save_training_results(model, history, test_loss, test_accuracy)

