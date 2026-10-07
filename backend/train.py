import json
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# تحديد المسارات المطلقة بدقة لمجلدات التدريب والتحقق الجديدة
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_DIR = os.path.abspath(
    os.path.join(BASE_DIR, '../Plant Diseases Dataset/train')
)
VAL_DIR = os.path.abspath(
    os.path.join(BASE_DIR, '../Plant Diseases Dataset/valid')
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("⏳ جاري تحميل وفحص البيانات من المجلدات المنظمة بنجاح...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_dataset.class_names
print(f"✅ تم رصد وربط {len(class_names)} صنفاً زراعياً بنجاح.")

# حفظ الفهارس وترتيبها في ملف JSON لضمان عدم حدوث أي خطأ في مطابقة الأسماء نهائياً
json_path = os.path.join(BASE_DIR, 'classes.json')
with open(json_path, 'w', encoding='utf-8') as f:
  json.dump(class_names, f, ensure_ascii=False, indent=4)
print("💾 تم حفظ مرجع الأصناف بدقة تامة في 'classes.json'.")

AUTOTUNE = tf.data.AUTOTUNE

# تحسين أداء وسرعة قراءة البيانات من الذاكرة
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# 1. طبقة تعزيز الصور عشوائياً لمكافحة حفظ الصور (Data Augmentation)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# 2. بناء وتجهيز نموذج MobileNetV2 مع الضبط الدقيق (Fine-Tuning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet'
)

# فك تجميد أحدث 40 طبقة ليتعلم النموذج ملامح ودقائق أوراق النباتات بدقة عالية
base_model.trainable = True
for layer in base_model.layers[:-40]:
  layer.trainable = False

# 3. دمج شبكة النموذج بالكامل
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)

# 🎯 استخدام طبقة Rescaling الرسمية بدلاً من preprocess_input لمنع خطأ 127.5 عند التحميل في app.py
x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)

x = base_model(x, training=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# 4. تجميع النموذج بمعدل تعلم مناسب للضبط الدقيق (0.0001)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model_save_path = os.path.join(BASE_DIR, 'plant_disease_model.h5')

# 5. إعداد أدوات المراقبة والحفظ الذكي (Callbacks)
checkpoint = ModelCheckpoint(
    model_save_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1,
)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
)

# أداة لتقليل سرعة التعلم تلقائياً عند ثبات مستوى الخطأ لضمان استقرار الدقة
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1
)

EPOCHS = 15
print(f"🚀 بدأ التدريب العميق المطور (Fine-Tuning) لـ {EPOCHS} دورة...")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping, reduce_lr],
)

print("💾 تم تدريب وحفظ النموذج الاحترافي بنجاح تام!")