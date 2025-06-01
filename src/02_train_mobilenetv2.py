# train_mobilenetv2.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Opcional: desactiva advertencia OneDNN
import json
import pandas as pd
import time
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback

# Configuración
DATA_DIR = "data/processed"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
AUTOTUNE = tf.data.AUTOTUNE

# Callback personalizado para medir tiempo por época
class TimeHistory(Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_start_time)

# Cargar datasets desde directorio con validación
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Obtener nombres de clases antes del preprocesamiento
class_names = train_ds.class_names
with open("data/class_names.json", "w") as f:
    json.dump(class_names, f)
# Aplicar preprocesamiento y mejorar rendimiento
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(buffer_size=AUTOTUNE)

# Crear modelo basado en MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con callback para tiempo
time_callback = TimeHistory()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[time_callback]
)

# Guardar historial en DataFrame con tiempos
history_df = pd.DataFrame(history.history)
history_df['epoch_time_sec'] = time_callback.times
history_df['cumulative_time_sec'] = history_df['epoch_time_sec'].cumsum()

print(history_df)

# Guardar historial y modelo
os.makedirs("reports", exist_ok=True)
history_df.to_csv("reports/training_history.csv", index=False)

os.makedirs("models", exist_ok=True)
model.save("models/mobilenetv2_trained.keras")
print("Modelo MobileNetV2 entrenado guardado en models/mobilenetv2_trained.keras")
