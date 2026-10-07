import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'plant_disease_model.h5')
JSON_PATH = os.path.join(BASE_DIR, 'classes.json')

# تحميل النموذج والفهارس المحدثة
print("⏳ جاري تحميل النموذج وقاموس الأصناف...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    class_names = json.load(f)

# قاعدة بيانات التوصيات والعلاج لكل مرض (يمكنك توسيعها لإضافة تفاصيل أكثر دقة)
TREATMENT_DATABASE = {
    "Apple___Apple_scab": {
        "ar_name": "جرب التفاح (الفطري)",
        "recoverable": "نعم، قابل للعلاج في المراحل المبكرة",
        "advice": "قم بإزالة الأوراق المصابة المتساقطة فوراً لتجنب انتشار الفطر، ورش الشجرة بمبيد فطري وقائي معتمد."
    },
    "Apple___Black_rot": {
        "ar_name": "العفن الأسود في التفاح",
        "recoverable": "صعب في المراحل المتقدمة",
        "advice": "قص الأفرع المصابة والتقرحات بمسافة 15 سم تحت الإصابة، وتطهير الأدوات بمحلول مطهر بعد كل قص."
    },
    "Potato___Early_blight": {
        "ar_name": "اللفحة المبكرة في البطاطس",
        "recoverable": "نعم، يمكن السيطرة عليها",
        "advice": "تجنب الري العلوي الذي يبلل الأوراق، واستخدم مبيدات فطرية تحتوي على النحاس."
    },
    "Potato___Late_blight": {
        "ar_name": "اللفحة المتأخرة (خطيرة)",
        "recoverable": "صعب جداً في حال الإصابة الشاملة",
        "advice": "التخلص الفوري من النباتات الشديدة الإصابة بحرقها أو دفنها لمنع انتقال العدوى لباقي المحصول."
    },
    "Tomato___Bacterial_spot": {
        "ar_name": "البقع البكتيرية في الطماطم",
        "recoverable": "متوسط، يتطلب تدخلاً سريعاً",
        "advice": "استخدام بخاخات نحاسية مضادة للبكتيريا، وتقليل التزاحم بين النباتات لتحسين التهوية."
    }
}

def analyze_plant_image(img_path):
    # تجهيز الصورة للتحليل
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # التنبؤ
    predictions = model.predict(x)
    score = tf.nn.softmax(predictions[0]) if tf.reduce_max(predictions[0]) > 1 else predictions[0]
    
    predicted_idx = np.argmax(score)
    confidence = float(np.max(score)) * 100
    class_name = class_names[predicted_idx]

    # التحقق مما إذا كانت النبتة سليمة
    is_healthy = "healthy" in class_name.lower()

    if is_healthy:
        severity_stage = "سليمة تماماً (صحة ممتازة)"
        recoverable = "لا توجد إصابة"
        advice = "النبتة لا تعاني من أي أمراض ظاهرة. استمر في برنامج الري والتسميد المنتظم."
        ar_name = "نبتة سليمة وزحية"
    else:
        # تقييم مرحلة الإصابة بناءً على نسبة ثقة النموذج
        if confidence < 65:
            severity_stage = "بدايات الإصابة (مرحلة مبكرة جداً)"
            recoverable = "نعم، قابلة للإنقاذ وسهلة العلاج"
        elif confidence < 82:
            severity_stage = "إصابة متوسطة الانتشار"
            recoverable = "نعم، يمكن السيطرة عليها بالعلاج المناسب"
        else:
            severity_stage = "إصابة متقدمة / حرجة (تلف كبير)"
            recoverable = "صعب جداً، يُخشى عدم جدوى العلاج"

        # جلب معلومات العلاج أو وضع إرشادات افتراضية دقيقة
        disease_info = TREATMENT_DATABASE.get(class_name, {
            "ar_name": class_name,
            "recoverable": recoverable,
            "advice": "يُنصح بفصل النبتة المصابة، إزالة الأوراق المتضررة، واستشارة أخصائي زراعي لتحديد المبيد الكيميائي المناسب."
        })
        advice = disease_info["advice"]
        ar_name = disease_info.get("ar_name", class_name)
        recoverable = disease_info["recoverable"]

    # هيكل النتيجة الكامل للواجهة الخلفية (Backend Response)
    analysis_result = {
        "success": True,
        "class_name": class_name,
        "display_name": ar_name,
        "confidence_score": round(confidence, 2),
        "is_healthy": is_healthy,
        "severity_stage": severity_stage,
        "recoverability": recoverable,
        "treatment_advice": advice
    }

    return analysis_result