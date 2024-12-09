#%% Importing Required Libraries
import pandas as pd
import os
import numpy as np
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.metrics import AUC
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import (
    AvgPool2D, GlobalAveragePooling2D, Dense, Dropout, Input,
    Conv2D, multiply, BatchNormalization, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report, roc_curve, auc

#%% Enable CUDA
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

#%% Reading the Dataset
all_xray_df = pd.read_csv("/home/ubuntu/new_NLP/CV_Project/data/Data_Entry_2017.csv")
all_image_paths = {
    os.path.basename(x): x
    for x in glob(os.path.join('/home/ubuntu/new_NLP/CV_Project/data/', 'images*', '*', '*.png'))
}
all_xray_df['Image_Path'] = all_xray_df['Image Index'].map(all_image_paths.get)

# Creating One-Hot Encoded Columns for Labels
unique_labels = set(
    label.strip()
    for labels in all_xray_df['Finding Labels'].dropna()
    for label in labels.split('|')
)
for label in unique_labels:
    all_xray_df[label] = all_xray_df['Finding Labels'].apply(lambda x: 1 if label in x.split('|') else 0)

# Filter for specific labels
selected_labels = ['No Finding', 'Infiltration', 'Effusion', 'Atelectasis']
filtered_df = all_xray_df[
    (all_xray_df['No Finding'] == 1) |
    (all_xray_df['Infiltration'] == 1) |
    (all_xray_df['Effusion'] == 1) |
    (all_xray_df['Atelectasis'] == 1)
].copy()

# Limit dataset for demonstration (remove `.head(50000)` for full dataset)
filtered_df = filtered_df.head(50000)

#%% Custom Weighted Binary Cross-Entropy Loss
def compute_class_weights(df, labels):
    total_samples = len(df)
    class_weights = {}
    for label in labels:
        positive_samples = df[label].sum()
        negative_samples = total_samples - positive_samples
        class_weights[label] = {
            0: positive_samples / total_samples,
            1: negative_samples / total_samples
        }
    return class_weights

class_weights = compute_class_weights(filtered_df, selected_labels)

def weighted_binary_crossentropy(class_weights):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        weights = (
            y_true * tf.constant([class_weights[label][1] for label in selected_labels], dtype=tf.float32) +
            (1 - y_true) * tf.constant([class_weights[label][0] for label in selected_labels], dtype=tf.float32)
        )
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean(weights * bce)
    return loss

#%% Splitting Dataset
train_df, val_df = train_test_split(filtered_df, test_size=0.2, random_state=42)

#%% Data Augmentation
IMG_SIZE = (512, 512)
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    height_shift_range=0.1,
    width_shift_range=0.1,
    brightness_range=[0.7, 1.5],
    rotation_range=3,
    zoom_range=0.125,
    shear_range=0.01,
    fill_mode='nearest'
)

train_gen = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='Image_Path',
    y_col=selected_labels,
    target_size=IMG_SIZE,
    batch_size=32,
    color_mode='rgb',
    class_mode='raw',
    shuffle=True
)

val_gen = datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='Image_Path',
    y_col=selected_labels,
    target_size=IMG_SIZE,
    batch_size=40,
    color_mode='rgb',
    class_mode='raw',
    shuffle=False
)

#%% Building the Combined Attention Model
base_pretrained_model = VGG16(input_shape=(512, 512, 3), include_top=False, weights='imagenet')
base_pretrained_model.trainable = False

input_tensor = base_pretrained_model.input
pt_features = base_pretrained_model.output
bn_features = BatchNormalization(name="Features_BN")(pt_features)

# Attention Mechanism
attn_layer = Conv2D(128, kernel_size=(1, 1), padding="same", activation="elu")(bn_features)
attn_layer = Conv2D(32, kernel_size=(1, 1), padding="same", activation="elu")(attn_layer)
attn_layer = Conv2D(16, kernel_size=(1, 1), padding="same", activation="elu")(attn_layer)
attn_layer = AvgPool2D(pool_size=(2, 2), strides=(1, 1), padding="same")(attn_layer)
attn_layer = Conv2D(1, kernel_size=(1, 1), padding="valid", activation="sigmoid", name="AttentionMap2D")(attn_layer)

up_c2 = Conv2D(512, kernel_size=(1, 1), padding='same', name='UpscaleAttention', activation='linear', use_bias=False)(attn_layer)
mask_features = multiply([up_c2, bn_features])
gap_features = GlobalAveragePooling2D()(mask_features)
gap_attention = GlobalAveragePooling2D()(up_c2)

rescaled_gap = Lambda(lambda x: x[0] / (x[1] + 1e-8), name="RescaleGAP")([gap_features, gap_attention])

gap_dr = Dropout(0.5)(rescaled_gap)
dense_layer = Dense(128, activation='elu')(gap_dr)
dr_steps = Dropout(0.5)(dense_layer)
out_layer = Dense(len(selected_labels), activation='sigmoid')(dr_steps)

combined_model = Model(inputs=input_tensor, outputs=out_layer, name='combined_attention_model')

# Compile the model
combined_model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss=weighted_binary_crossentropy(class_weights)
)

#%% Callbacks
model_checkpoint = ModelCheckpoint(
    filepath='best_model.keras', monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min', verbose=1
)
reduce_lr_on_plateau = ReduceLROnPlateau(
    monitor='val_loss', factor=0.8, patience=5, verbose=1, mode='min', min_lr=1e-6
)
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, mode='min', verbose=1, restore_best_weights=True
)
callbacks = [model_checkpoint, reduce_lr_on_plateau, early_stopping]

#%% Train the Model
history = combined_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    steps_per_epoch=len(train_df) // train_gen.batch_size,
    validation_steps=len(val_df) // val_gen.batch_size,
    verbose=1,
    callbacks=callbacks
)
#%% Evaluate the Model
import matplotlib.pyplot as plt
def evaluate_model_and_plot(model, val_gen, val_df):
    y_true = val_df[selected_labels].values
    steps = len(val_df) // val_gen.batch_size
    if len(val_df) % val_gen.batch_size != 0:
        steps += 1

    y_pred = model.predict(val_gen, steps=steps, verbose=1)

    # Convert predictions to binary values using a threshold
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculate metrics
    roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred_binary, average='macro')
    precision = precision_score(y_true, y_pred_binary, average='macro')
    recall = recall_score(y_true, y_pred_binary, average='macro')

    print(f"ROC AUC (Macro): {roc_auc:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_binary, target_names=selected_labels))

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(selected_labels):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.2f})")
    plt.title('ROC Curves for Each Label')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

evaluate_model_and_plot(combined_model, val_gen, val_df)
