from datetime import datetime
import json
import os
import sqlite3
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.abspath(os.path.join(BASE_DIR, '../frontend/templates'))
STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, '../frontend/static'))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DB_FOLDER = os.path.join(BASE_DIR, 'database')
DB_PATH = os.path.join(DB_FOLDER, 'plant_records.db')
JSON_PATH = os.path.join(BASE_DIR, 'classes.json')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = 'plant_secret_key_123'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# 🎯 حد نسبة الثقة (أي صورة سيارة أو شكل غريب ستكون ثقتها أقل من 50% وتُرفض تلقائياً)
CONFIDENCE_THRESHOLD = 50.0


def init_db():
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_name TEXT,
            disease_name TEXT,
            confidence TEXT,
            timestamp TEXT
        )
    """)
  conn.commit()
  conn.close()


init_db()

if os.path.exists(JSON_PATH):
  with open(JSON_PATH, 'r', encoding='utf-8') as f:
    class_names = json.load(f)
else:
  class_names = []


def parse_plant_and_disease(folder_name):
  if 'non_plant' in folder_name.lower() or 'background' in folder_name.lower():
    return {
        'plant': 'Non_Plant',
        'name': 'غير نباتية',
        'status': 'non_plant',
        'treatment': '',
        'map_query': '',
    }

  if '___' in folder_name:
    parts = folder_name.split('___')
    raw_plant = parts[0]
    raw_disease = parts[1] if len(parts) > 1 else 'Unknown'
  elif '__' in folder_name:
    parts = folder_name.split('__')
    raw_plant = parts[0]
    raw_disease = parts[1] if len(parts) > 1 else 'Unknown'
  else:
    parts = folder_name.split('_')
    raw_plant = parts[0]
    raw_disease = '_'.join(parts[1:]) if len(parts) > 1 else 'healthy'

  plant_mapping = {
      'Potato': 'بطاطس',
      'Tomato': 'طماطم',
      'Pepper': 'فلفل رومي',
      'Apple': 'تفاح',
      'Corn': 'ذرة',
      'Grape': 'عنب',
      'Peach': 'خوخ',
      'Strawberry': 'فراولة',
  }

  clean_plant = raw_plant.replace('_', ' ').strip()
  plant_name = plant_mapping.get(
      raw_plant, plant_mapping.get(clean_plant, clean_plant)
  )

  if 'healthy' in folder_name.lower():
    return {
        'plant': plant_name,
        'name': 'سليم وصحي 🌿',
        'status': 'healthy',
        'treatment': (
            'النبتة سليمة تماماً ولا توجد أي إصابات. استمر في برنامج الري'
            ' والتسميد المتوازن.'
        ),
        'map_query': '',
    }

  disease_mapping = {
      'Early_blight': 'اللفحة المبكرة',
      'Late_blight': 'اللفحة المتأخرة',
      'Bacterial_spot': 'التبقع البكتيري',
      'Leaf_Mold': 'عفن الأوراق',
      'Septoria_leaf_spot': 'تبقع أوراق سيبتوريا',
      'Spider_mites_Two_spotted_spider_mite': 'العناكب الحمراء',
      'Target_Spot': 'البقعة المستهدفة',
      'YellowLeaf_Curl_Virus': 'فيروس تجعد أوراق الطماطم الأصفر',
      'Northern_Leaf_Blight': 'اللفحة الشمالية',
      'Cedar_Apple_Rust': 'صدأ التفاح',
  }

  clean_disease = raw_disease.replace('_', ' ').strip()
  disease_name = disease_mapping.get(
      raw_disease, disease_mapping.get(clean_disease, clean_disease)
  )

  f_lower = folder_name.lower()
  if 'blight' in f_lower or 'rust' in f_lower:
    treatment_text = (
        'استخدام مبيدات فطرية وقائية وعلاجية متخصصة، الالتزام بالري المنتظم.'
    )
  elif 'spot' in f_lower:
    treatment_text = (
        'استخدام مركبات النحاس الوقائية، وتجنب البلل الورقي أثناء الري.'
    )
  elif 'virus' in f_lower:
    treatment_text = (
        'مكافحة الحشرات الناقلة للعدوى وتعقيم الأدوات الزراعية بانتظام.'
    )
  else:
    treatment_text = 'استشر مهندساً زراعياً مختصاً لتحديد نوع المبيد المناسب.'

  return {
      'plant': plant_name,
      'name': disease_name,
      'status': 'infected',
      'treatment': treatment_text,
      'map_query': 'agricultural store',
  }


print('Loading AI model, please wait...')
model_path = os.path.join(BASE_DIR, 'plant_disease_model.h5')


class TrueDivide(tf.keras.layers.Layer):

  def __init__(self, **kwargs):
    super(TrueDivide, self).__init__()

  def call(self, inputs, **kwargs):
    return inputs

  @classmethod
  def from_config(cls, config):
    return cls()


try:
  model = tf.keras.models.load_model(model_path, compile=False)
except Exception:
  with tf.keras.utils.custom_object_scope({'TrueDivide': TrueDivide}):
    model = tf.keras.models.load_model(model_path, compile=False)

print('Model loaded successfully!')


@app.route('/')
def landing_page():
  return render_template('landing.html')


@app.route('/login_page')
def login_page():
  return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
  email = request.form.get('email')
  password = request.form.get('password')

  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  cursor.execute(
      'SELECT id, name FROM users WHERE email = ? AND password = ?',
      (email, password),
  )
  user = cursor.fetchone()
  conn.close()

  if user:
    session['user_id'] = user[0]
    session['user_name'] = user[1]
    return render_template('predict.html')
  else:
    return (
        "<script>alert('البريد الإلكتروني أو كلمة المرور غير"
        " صحيحة!'); window.location.href='/login_page';</script>"
    )


@app.route('/register', methods=['POST'])
def register():
  name = request.form.get('name')
  email = request.form.get('email')
  password = request.form.get('password')

  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  try:
    cursor.execute(
        'INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
        (name, email, password),
    )
    conn.commit()
    conn.close()
    return (
        "<script>alert('تم إنشاء الحساب بنجاح! يمكنك تسجيل الدخول"
        " الآن.'); window.location.href='/login_page';</script>"
    )
  except sqlite3.IntegrityError:
    conn.close()
    return (
        "<script>alert('البريد الإلكتروني مستخدم مسبقاً!');"
        " window.location.href='/login_page';</script>"
    )


@app.route('/forgot-password', methods=['POST'])
def forgot_password():
  email = request.form.get('email')
  new_password = request.form.get('new_password')

  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
  user = cursor.fetchone()

  if user:
    cursor.execute(
        'UPDATE users SET password = ? WHERE email = ?', (new_password, email)
    )
    conn.commit()
    conn.close()
    return (
        "<script>alert('تم تحديث كلمة المرور بنجاح! يمكنك تسجيل الدخول"
        " الآن.'); window.location.href='/login_page';</script>"
    )
  else:
    conn.close()
    return (
        "<script>alert('البريد الإلكتروني غير مسجل في النظام!');"
        " window.location.href='/login_page';</script>"
    )


@app.route('/logout')
def logout():
  session.clear()
  return redirect(url_for('landing_page'))


@app.route('/predict_page')
def predict_page():
  if 'user_id' not in session:
    return redirect(url_for('login_page'))
  return render_template('predict.html')


@app.route('/search_page')
def search_page():
  if 'user_id' not in session:
    return redirect(url_for('login_page'))
  return render_template('search.html')


@app.route('/dashboard')
def dashboard():
  if 'user_id' not in session:
    return redirect(url_for('login_page'))

  current_user_id = session['user_id']
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  cursor.execute(
      'SELECT id, image_name, disease_name, confidence, timestamp FROM'
      ' diagnoses WHERE user_id = ?',
      (current_user_id,),
  )
  records = cursor.fetchall()
  conn.close()

  total_scans = len(records)
  healthy_count = sum(
      1 for r in records if '0.0%' in str(r[3]) or 'سليم' in str(r[2])
  )
  infected_count = total_scans - healthy_count

  return render_template(
      'dashboard.html',
      total_scans=total_scans,
      healthy_count=healthy_count,
      infected_count=infected_count,
      records=records,
  )


@app.route('/faq_page')
def faq_page():
  if 'user_id' not in session:
    return redirect(url_for('login_page'))
  return render_template('faq.html')


@app.route('/history_page')
def history_page():
  if 'user_id' not in session:
    return redirect(url_for('login_page'))

  current_user_id = session['user_id']
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  cursor.execute(
      'SELECT id, image_name, disease_name, confidence, timestamp FROM'
      ' diagnoses WHERE user_id = ? ORDER BY id DESC',
      (current_user_id,),
  )
  records = cursor.fetchall()
  conn.close()

  return render_template('history.html', records=records)


@app.route('/delete_record/<int:record_id>')
def delete_record(record_id):
  if 'user_id' not in session:
    return redirect(url_for('login_page'))

  current_user_id = session['user_id']
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  cursor.execute(
      'DELETE FROM diagnoses WHERE id = ? AND user_id = ?',
      (record_id, current_user_id),
  )
  conn.commit()
  conn.close()

  return redirect(url_for('history_page'))


@app.route('/predict', methods=['POST'])
def predict_disease():
  if 'user_id' not in session:
    return jsonify({'error': 'يرجى تسجيل الدخول أولاً'})

  if 'file' not in request.files:
    return jsonify({'error': 'لم يتم رفع أي ملف'})

  file = request.files['file']
  if file.filename == '':
    return jsonify({'error': 'اسم الملف فارغ'})

  if file:
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    image_cv = cv2.imread(filepath)
    if image_cv is None:
      return jsonify({'error': 'عذراً، ملف الصورة غير صالح أو تالف.'})

    # 1. فحص الصورة الضبابية
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if laplacian_var < 5:
      return jsonify({
          'error': (
              'الصورة ضبابية جداً أو تالفة، يرجى رفع صورة أوضح لورقة النبات.'
          )
      })

    # 2. قص بسيط للحواف
    h, w, _ = image_cv.shape
    ymin, ymax = int(h * 0.02), int(h * 0.98)
    xmin, xmax = int(w * 0.02), int(w * 0.98)
    center_crop = image_cv[ymin:ymax, xmin:xmax]

    hsv_center = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)

    # 3. 🌿 نطاق ألوان واسع ومرن (يشمل الأخضر، الأصفر، والبني للأوراق المصابة والتالفة)
    lower_plant = np.array([8, 15, 15])
    upper_plant = np.array([95, 255, 255])
    mask_plant = cv2.inRange(hsv_center, lower_plant, upper_plant)
    plant_color_ratio = (
        cv2.countNonZero(mask_plant)
        / (center_crop.shape[0] * center_crop.shape[1])
    ) * 100

    # 🎨 فحص ألوان النيون الرسومات الاصطناعية
    lower_neon = np.array([30, 150, 180])
    upper_neon = np.array([85, 255, 255])
    mask_neon = cv2.inRange(hsv_center, lower_neon, upper_neon)
    neon_ratio = (
        cv2.countNonZero(mask_neon)
        / (center_crop.shape[0] * center_crop.shape[1])
    ) * 100

    # فحص أولي خفيف جداً (يستبعد الفراغ التام أو النيون فقط)
    if plant_color_ratio < 2.0 or neon_ratio > 30.0:
      return jsonify({
          'error': (
              'عذراً، لم يتم التعرف على ورقة نبات حقيقية في الصورة. يرجى رفع'
              ' صورة واضحة لورقة النبتة فقط.'
          )
      })

    mean_saturation = np.mean(hsv_center[:, :, 1])
    mean_value = np.mean(hsv_center[:, :, 2])
    is_dead_leaf = mean_saturation < 30 and mean_value < 110

    center_crop_path = os.path.join(UPLOAD_FOLDER, 'crop_' + filename)
    cv2.imwrite(center_crop_path, center_crop)

    img = image.load_img(center_crop_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # 4. التنبؤ بالذكاء الاصطناعي
    predictions = model.predict(img_array)
    max_prob = np.max(predictions[0])
    confidence = round(100 * max_prob, 2)

    # 🚨 5. الفلتر الرئيسي للأجسام الغريبة والسيارات:
    # أي صورة غير نباتية أو سيارة ستتشتت الأحتمالات ولن تتجاوز نسبة ثقتها 50%
    if confidence < CONFIDENCE_THRESHOLD:
      return jsonify({
          'error': (
              'عذراً، لم يتم التعرف على ورقة نبات معروفة بدقة كافية. يرجى'
              ' رفع صورة مكبرة وواضحة لورقة النبتة فقط.'
          )
      })

    pred_index = np.argmax(predictions[0])
    predicted_folder = (
        class_names[pred_index]
        if pred_index < len(class_names)
        else class_names[0]
    )

    info = parse_plant_and_disease(predicted_folder)

    if info['status'] == 'non_plant':
      return jsonify({
          'error': (
              'عذراً، الصورة المرفوعة لا تحتوي على ورقة نبات. يرجى رفع صورة'
              ' واضحة لورقة النبتة.'
          )
      })

    if is_dead_leaf:
      infection_rate = f'{confidence}% (تالفة كلياً)'
      severity_stage = 'ميتة / تالفة كلياً ⚠️'
      recoverable = 'غير قابلة للإنقاذ (يجب التخلص منها)'
      treatment_text = (
          'الورقة تالفة وميتة تماماً ولا يمكن علاجها. يُنصح بإعدامها فوراً'
          ' وحرقها أو إبعادها عن باقي المحصول لمنع انتشار الأبواغ الفطرية.'
      )
    else:
      if info['status'] == 'healthy':
        infection_rate = '0.0%'
        severity_stage = 'سليمة تماماً (صحة ممتازة 🌿)'
        recoverable = 'لا توجد إصابة'
        treatment_text = info['treatment']
      else:
        infection_rate = f'{confidence}%'
        severity_stage = 'إصابة تحتاج لمتابعة'
        recoverable = 'نعم، قابلة للإنقاذ'
        treatment_text = info['treatment']

    display_name = f"نبتة {info['plant']} - {info['name']}"

    current_user_id = session['user_id']
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        '''
            INSERT INTO diagnoses (user_id, image_name, disease_name, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''',
        (
            current_user_id,
            filename,
            display_name,
            infection_rate,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        ),
    )
    conn.commit()
    conn.close()

    return jsonify({
        'class': display_name,
        'confidence': infection_rate,
        'severity': severity_stage,
        'recoverability': recoverable,
        'treatment': treatment_text,
        'map_url': (
            info['map_query']
            if (not is_dead_leaf and info['status'] == 'infected')
            else ''
        ),
    })


@app.route('/search_disease', methods=['GET'])
def search_disease():
  query = request.args.get('q', '').lower()
  results = []

  for folder in class_names:
    info = parse_plant_and_disease(folder)
    full_title = f"{info['plant']} - {info['name']}"

    if (
        query
        in full_title.lower()
        .replace(' ', '')
        .replace('ة', 'ه')
        .replace('أ', 'ا')
        or query in info['plant'].lower()
        or query in info['name'].lower()
    ):
      results.append({
          'name': full_title,
          'plant': info['plant'],
          'treatment': info['treatment'],
          'map_url': info['map_query'],
      })

  return jsonify(results)


if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True, port=5000)