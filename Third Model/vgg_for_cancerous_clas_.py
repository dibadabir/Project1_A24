# -*- coding: utf-8 -*-
"""Final VGG for cancerous clas..ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nvspANgX0wl3F30LWWGJRYYg3d6jTk5v
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize

# ---------------------------
# 1. Building CNN model using VGG16
# ---------------------------
def build_vgg16_model(input_shape=(224, 224, 3), num_classes=3):
    # Load the VGG16 model with weights pretrained on ImageNet and exclude the top fully-connected layers.
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze all layers in the base model to retain pretrained features during initial training.
    for layer in base_model.layers:
        layer.trainable = False

    # Add a custom classification head with L2 regularization and dropout to reduce overfitting.
    x = layers.GlobalAveragePooling2D()(base_model.output)
    # Fully connected layer with 1024 neurons, ReLU activation and L2 regularization.
    x = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    # Apply dropout with a rate of 0.6.
    x = layers.Dropout(0.6)(x)
    # Final classification layer.
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Assemble the complete model.
    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

# Build and compile the model with the specified input shape and number of classes.
model = build_vgg16_model(input_shape=(224, 224, 3), num_classes=3)

# Unfreeze the last 10 layers to allow fine-tuning.
for layer in model.layers[-10:]:
    layer.trainable = True

# Set learning rate (LR) to 1e-4.
LR = 1e-4
optimizer = Adam(learning_rate=LR)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------
# 2. Loading dataset and computing class weights
# ---------------------------
train_dir = '/content/drive/MyDrive/Discipline-specific/split_dataset/train'
val_dir   = '/content/drive/MyDrive/Discipline-specific/split_dataset/val'
test_dir  = '/content/drive/MyDrive/Discipline-specific/split_dataset/test'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_datagen   = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen  = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224,224), batch_size=32, class_mode='categorical')
val_generator   = val_datagen.flow_from_directory(val_dir, target_size=(224,224), batch_size=32, class_mode='categorical')
test_generator  = test_datagen.flow_from_directory(test_dir, target_size=(224,224), batch_size=32, class_mode='categorical', shuffle=False)

class_labels = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# ---------------------------
# 3. Training the model with callbacks (EarlyStopping & ReduceLROnPlateau)
# ---------------------------
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, reduce_lr]
)

# ---------------------------
# 4. Evaluating the model and plotting metrics
# ---------------------------
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")

y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

cm = metrics.confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
class_names = list(train_generator.class_indices.keys())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print(metrics.classification_report(y_true, y_pred_classes, target_names=class_names))

# ---------------------------
# 5. Plotting training and validation history
# ---------------------------
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

plt.show()

# ---------------------------
# 6. Plotting ROC curves for each class
# ---------------------------
n_classes = 3
y_true_bin = label_binarize(y_true, classes=range(n_classes))
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_true_bin[:, i], y_pred[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

plt.figure(figsize=(8,6))
for i, class_name in enumerate(class_names):
    plt.plot(fpr[i], tpr[i], label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Skin Cancer Diagnosis')
plt.legend(loc="lower right")
plt.show()

# Save the model to a file called "model.h5"
model.save('model.h5')
