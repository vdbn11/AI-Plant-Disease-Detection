"""
=============================================================
 PlantDiseaseAI - الملف الرئيسي للموقع (Flask)
=============================================================
 هذا الملف يشغّل الموقع كامل، ومسؤول عن:
   1) الاتصال بقاعدة البيانات (SQLite)
   2) تحميل موديل الذكاء الاصطناعي
   3) صفحات الموقع (تسجيل دخول / فحص / سجل / دليل أمراض)
   4) استقبال صورة النبتة وإرجاع نتيجة التشخيص

 ملاحظة: كل دالة فيها تعليق عربي يوضح وظيفتها.
=============================================================
"""

import json
import os
import sqlite3
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from tensorflow.keras.preprocessing import image as kimage

# =============================================================
# 1) المسارات والإعدادات
# =============================================================
BASE = os.path.abspath(os.path.dirname(__file__))
TEMPLATES = os.path.abspath(os.path.join(BASE, "../frontend/templates"))
STATIC = os.path.abspath(os.path.join(BASE, "../frontend/static"))
UPLOADS = os.path.join(BASE, "uploads")
DB_PATH = os.path.join(BASE, "database", "plant_records.db")

# ملف أسماء الأصناف: نستخدم الأحدث المتوفر (v6 ثم v5 ثم v4 ثم القديم)
for name in ("classes_v6.json", "classes_v5.json", "classes_v4.json", "classes.json"):
    if os.path.exists(os.path.join(BASE, name)):
        CLASSES_PATH = os.path.join(BASE, name)
        break

# ملف الموديل: نستخدم الأحدث المتوفر
for name in ("plant_disease_model_v6.keras", "plant_disease_model_v5.keras",
             "plant_disease_model_v4.keras", "plant_disease_model.h5"):
    if os.path.exists(os.path.join(BASE, name)):
        MODEL_PATH = os.path.join(BASE, name)
        break

GUIDE_PATH = os.path.join(BASE, "guide_data.json")
CONFIDENCE_THRESHOLD = 50.0   # أقل من هذه النسبة = النتيجة غير موثوقة وتُرفض

os.makedirs(UPLOADS, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES, static_folder=STATIC)
app.secret_key = "plant_secret_key_123"
app.config["TEMPLATES_AUTO_RELOAD"] = True


# =============================================================
# 2) قاعدة البيانات
# =============================================================
def init_db():
    """إنشاء جدولي المستخدمين والتشخيصات إذا ما كانت موجودة."""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, email TEXT UNIQUE, password TEXT
        );
        CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER, image_name TEXT, disease_name TEXT,
            confidence TEXT, timestamp TEXT
        );
    """)
    conn.commit()
    conn.close()


def db_query(sql, params=(), fetch=False):
    """دالة موحدة لتنفيذ أي استعلام في قاعدة البيانات (تقلل تكرار الكود)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall() if fetch else None
    conn.commit()
    conn.close()
    return rows


os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
init_db()


# =============================================================
# 3) تحميل الموديل وأسماء الأصناف
# =============================================================
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print("Loading AI model, please wait...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

# تأكد أن عدد أسماء الأصناف يطابق مخرجات الموديل (حماية من عدم التطابق)
if len(class_names) != model.output_shape[-1]:
    for name in ("classes_v5.json", "classes_v4.json", "classes.json"):
        p = os.path.join(BASE, name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                test = json.load(f)
            if len(test) == model.output_shape[-1]:
                class_names = test
                print("Using", name, "to match model output")
                break


# =============================================================
# 4) قاموس الأسماء العربية (النبات + المرض + العلاج)
# =============================================================
PLANTS_AR = {
    "Apple": "تفاح", "Banana": "موز", "Blueberry": "توت أزرق", "Cherry": "كرز",
    "Cherry_(including_sour)": "كرز", "Corn": "ذرة", "Corn_(maize)": "ذرة",
    "Grape": "عنب", "Mango": "مانجو", "Orange": "برتقال", "Peach": "خوخ",
    "Pepper": "فلفل رومي", "Pepper,_bell": "فلفل رومي", "Potato": "بطاطس",
    "Raspberry": "توت العليق", "Rice": "أرز", "Soybean": "فول الصويا",
    "Squash": "قرع", "Strawberry": "فراولة", "Tomato": "طماطم",
}

DISEASES_AR = {
    "Early_blight": "اللفحة المبكرة", "Late_blight": "اللفحة المتأخرة",
    "Bacterial_spot": "التبقع البكتيري", "Leaf_Mold": "عفن الأوراق",
    "Septoria_leaf_spot": "تبقع أوراق سيبتوريا",
    "Spider_mites_Two_spotted_spider_mite": "العنكبوت الأحمر",
    "Target_Spot": "البقعة المستهدفة", "Northern_Leaf_Blight": "اللفحة الشمالية",
    "Cedar_apple_rust": "صدأ التفاح", "Apple_scab": "الجرب التفاحي",
    "Black_rot": "العفن الأسود", "Powdery_mildew": "البياض الدقيقي",
    "Cercospora_leaf_spot Gray_leaf_spot": "تبقع الأوراق الرمادي",
    "Common_rust_": "الصدأ الشائع", "Esca_(Black_Measles)": "الإسكا",
    "Leaf_blight_(Isariopsis_Leaf_Spot)": "لفحة أوراق العنب",
    "Haunglongbing_(Citrus_greening)": "التخضير الحمضي",
    "Leaf_scorch": "احتراق الأوراق", "Tomato_mosaic_virus": "فيروس الموزاييك",
    "Tomato_Yellow_Leaf_Curl_Virus": "فيروس تجعد الأوراق الأصفر",
    "Bacterialblight": "اللفحة البكتيرية", "Blast": "لفحة الأرز",
    "Brownspot": "التبقع البني", "Tungro": "فيروس تنجرو",
    "Sigatoka": "تبقع الأوراق الأصفر (سيكاتوكا)", "Xanthomonas": "ذبول زانثوموناس",
    "Downy_Mildew": "البياض الزغبي", "Bacterial_Leaf_Spot": "التبقع البكتيري",
    "Anthracnose": "الأنثراكنوز", "Bacterial_Canker": "التقرح البكتيري",
    "Cutting_Weevil": "سوسة قطع الأوراق", "Die_Back": "الموت الرجعي",
    "Gall_Midge": "ذبابة أوراق المانجو", "Sooty_Mould": "العفن الهبابي",
    "Rust": "صدأ فول الصويا",
}


def describe_class(folder_name):
    """تحويل اسم الصنف الإنجليزي (مثل Tomato___Late_blight) إلى معلومات عربية."""
    if "___" in folder_name:
        raw_plant, raw_disease = folder_name.split("___")[0], folder_name.split("___")[1]
    elif "__" in folder_name:
        raw_plant, raw_disease = folder_name.split("__")[0], folder_name.split("__")[1]
    else:
        raw_plant, raw_disease = folder_name.split("_")[0], "_".join(folder_name.split("_")[1:])

    plant = PLANTS_AR.get(raw_plant, raw_plant)
    healthy = "healthy" in folder_name.lower()
    disease = "سليمة وصحية 🌿" if healthy else DISEASES_AR.get(raw_disease, raw_disease)

    # نصيحة علاجية مبدئية حسب نوع المرض
    low = folder_name.lower()
    if healthy:
        advice = "النبتة سليمة، استمر في الري المنتظم والتسميد المتوازن."
    elif "blight" in low or "rust" in low:
        advice = "استخدم مبيداً فطرياً وقائياً وعلاجياً، والتزم بالري المنتظم."
    elif "spot" in low:
        advice = "استخدم مركبات النحاس الوقائية وتجنب بلل الأوراق أثناء الري."
    elif "virus" in low:
        advice = "كافح الحشرات الناقلة وعقّم الأدوات الزراعية بانتظام."
    else:
        advice = "اعزل النبتة المصابة واستشر مهندساً زراعياً لتحديد العلاج."

    return {
        "plant": plant,
        "name": disease,
        "status": "healthy" if healthy else "infected",
        "treatment": advice,
        "map_query": "" if healthy else "agricultural store",
    }


# =============================================================
# 5) أدوات مساعدة (تسجيل الدخول والفحص)
# =============================================================
def logged_in():
    """هل المستخدم مسجل دخول؟"""
    return "user_id" in session


def user_records():
    """جلب سجلات التشخيص الخاصة بالمستخدم الحالي (الأحدث أولاً)."""
    return db_query(
        "SELECT id, image_name, disease_name, confidence, timestamp "
        "FROM diagnoses WHERE user_id = ? ORDER BY id DESC",
        (session["user_id"],), fetch=True)


def check_image(img):
    """فحص جودة الصورة: ضبابية؟ فيها ورقة نبات فعلاً؟ ألوان صناعية؟"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 5:
        return "الصورة ضبابية جداً، يرجى رفع صورة أوضح لورقة النبات."

    h, w = img.shape[:2]
    crop = img[int(h * .02):int(h * .98), int(w * .02):int(w * .98)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # نسبة اللون النباتي (أخضر/أصفر/بني) في الصورة
    plant_mask = cv2.inRange(hsv, np.array([8, 15, 15]), np.array([95, 255, 255]))
    ratio = cv2.countNonZero(plant_mask) / (crop.shape[0] * crop.shape[1]) * 100

    # نسبة الألوان النيون (رسومات صناعية)
    neon = cv2.inRange(hsv, np.array([30, 150, 180]), np.array([85, 255, 255]))
    neon_ratio = cv2.countNonZero(neon) / (crop.shape[0] * crop.shape[1]) * 100

    if ratio < 2.0 or neon_ratio > 30.0:
        return "لم يتم التعرف على ورقة نبات حقيقية، ارفع صورة واضحة لورقة النبتة."
    return None


# =============================================================
# 6) صفحات الموقع
# =============================================================
@app.route("/")
def landing():
    """الصفحة الرئيسية."""
    return render_template("landing.html")


@app.route("/login_page")
def login_page():
    """صفحة تسجيل الدخول / إنشاء حساب."""
    return render_template("index.html")


@app.route("/home", methods=["POST"])
def login():
    """التحقق من البريد وكلمة المرور ثم فتح صفحة الفحص."""
    row = db_query("SELECT id, name FROM users WHERE email = ? AND password = ?",
                   (request.form.get("email"), request.form.get("password")), fetch=True)
    if not row:
        return "<script>alert('البريد أو كلمة المرور غير صحيحة!');location='/login_page';</script>"
    session["user_id"], session["user_name"] = row[0][0], row[0][1]
    return redirect(url_for("predict_page"))


@app.route("/register", methods=["POST"])
def register():
    """إنشاء حساب جديد."""
    try:
        db_query("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                 (request.form.get("name"), request.form.get("email"), request.form.get("password")))
        return "<script>alert('تم إنشاء الحساب بنجاح!');location='/login_page';</script>"
    except sqlite3.IntegrityError:
        return "<script>alert('البريد مستخدم مسبقاً!');location='/login_page';</script>"


@app.route("/forgot-password", methods=["POST"])
def forgot_password():
    """استعادة كلمة المرور بتحديثها بالبريد."""
    if db_query("SELECT id FROM users WHERE email = ?", (request.form.get("email"),), fetch=True):
        db_query("UPDATE users SET password = ? WHERE email = ?",
                 (request.form.get("new_password"), request.form.get("email")))
        return "<script>alert('تم تحديث كلمة المرور!');location='/login_page';</script>"
    return "<script>alert('البريد غير مسجل!');location='/login_page';</script>"


@app.route("/logout")
def logout():
    """تسجيل الخروج."""
    session.clear()
    return redirect(url_for("landing"))


@app.route("/predict_page")
def predict_page():
    """صفحة فحص النبتة (تحتاج تسجيل دخول)."""
    return render_template("predict.html") if logged_in() else redirect(url_for("login_page"))


@app.route("/history_page")
def history_page():
    """صفحة السجل: كل الفحوصات السابقة."""
    return render_template("history.html", records=user_records()) if logged_in() \
        else redirect(url_for("login_page"))


@app.route("/delete_record/<int:record_id>")
def delete_record(record_id):
    """حذف سجل واحد."""
    if logged_in():
        db_query("DELETE FROM diagnoses WHERE id = ? AND user_id = ?",
                 (record_id, session["user_id"]))
    return redirect(url_for("history_page"))


@app.route("/dashboard")
def dashboard():
    """لوحة التحكم: إحصاءات سريعة عن الفحوصات."""
    if not logged_in():
        return redirect(url_for("login_page"))
    records = user_records()
    healthy = sum(1 for r in records if "0.0%" in str(r[3]) or "سليم" in str(r[2]))
    return render_template("dashboard.html", records=records, total_scans=len(records),
                           healthy_count=healthy, infected_count=len(records) - healthy)


@app.route("/guide_page")
def guide_page():
    """دليل الأمراض: يعرض كل الأمراض مع صور قبل/بعد."""
    with open(GUIDE_PATH, "r", encoding="utf-8") as f:
        guide = json.load(f)
    return render_template("guide.html", data=json.dumps(guide, ensure_ascii=False),
                           total=len(guide), plants=len({g["plant_ar"] for g in guide}))


@app.route("/search_page")
def search_page():
    """صفحة البحث السريع في الأمراض."""
    return render_template("search.html") if logged_in() else redirect(url_for("login_page"))


@app.route("/faq_page")
def faq_page():
    """صفحة الأسئلة الشائعة."""
    return render_template("faq.html") if logged_in() else redirect(url_for("login_page"))


# =============================================================
# 7) التشخيص (المسار الأهم في المشروع)
# =============================================================
@app.route("/predict", methods=["POST"])
def predict():
    """
    يستقبل صورة → يفحص جودتها → يشغّل الموديل → يحفظ النتيجة في السجل → يرجع JSON.
    """
    if not logged_in():
        return jsonify({"error": "يرجى تسجيل الدخول أولاً"})

    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "لم يتم رفع أي ملف"})

    path = os.path.join(UPLOADS, file.filename)
    file.save(path)

    img = cv2.imread(path)
    if img is None:
        return jsonify({"error": "ملف الصورة غير صالح"})

    # (أ) فحص جودة الصورة
    problem = check_image(img)
    if problem:
        return jsonify({"error": problem})

    # (ب) قص الحواف لتقليل الخلفية
    h, w = img.shape[:2]
    crop = img[int(h * .02):int(h * .98), int(w * .02):int(w * .98)]
    crop_path = os.path.join(UPLOADS, "crop_" + file.filename)
    cv2.imwrite(crop_path, crop)

    # (ج) تحويل الصورة وتشغيل الموديل
    pic = kimage.load_img(crop_path, target_size=model.input_shape[1:3])
    arr = tf.expand_dims(kimage.img_to_array(pic), 0)
    probs = model.predict(arr, verbose=0)[0]
    confidence = round(float(np.max(probs)) * 100, 2)

    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({"error": "لم يتم التعرف على ورقة نبات بدقة كافية، ارفع صورة أوضح."})

    info = describe_class(class_names[int(np.argmax(probs))])
    display = f"نبتة {info['plant']} - {info['name']}"
    rate = "0.0%" if info["status"] == "healthy" else f"{confidence}%"

    # (د) حفظ النتيجة في سجل المستخدم
    db_query("INSERT INTO diagnoses (user_id, image_name, disease_name, confidence, timestamp) "
             "VALUES (?, ?, ?, ?, ?)",
             (session["user_id"], file.filename, display, rate,
              datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    return jsonify({
        "class": display,
        "confidence": rate,
        "severity": "سليمة تماماً 🌿" if info["status"] == "healthy" else "إصابة تحتاج متابعة",
        "recoverability": "لا توجد إصابة" if info["status"] == "healthy" else "نعم قابلة للإنقاذ",
        "treatment": info["treatment"],
        "map_url": info["map_query"],
    })


# =============================================================
# 8) البحث في دليل الأمراض (API)
# =============================================================
@app.route("/search_disease")
def search_disease():
    """بحث نصي في أسماء النباتات والأمراض وإرجاع العلاج."""
    q = request.args.get("q", "").strip().lower()
    out = []
    for folder in class_names:
        info = describe_class(folder)
        title = f"{info['plant']} - {info['name']}"
        if q and (q in title.lower() or q in info["plant"].lower() or q in info["name"].lower()):
            out.append({"name": title, "plant": info["plant"],
                        "treatment": info["treatment"], "map_url": info["map_query"]})
    return jsonify(out)


# =============================================================
# 9) تشغيل الموقع
# =============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
