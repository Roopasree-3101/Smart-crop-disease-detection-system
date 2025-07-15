import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import urllib.parse
from googletrans import Translator
import os


# --- Landing / Welcome Page ---
import streamlit as st

# Check if user has entered already
if "entered" not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    st.set_page_config(page_title="Welcome - Smart Crop Disease Detection", page_icon="🌾", layout="wide")

    # Full background image using your own image
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url(images/background.jpeg); /* 👈 replace this path with your image */
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    height: 100vh;
    }}

    [data-testid="stHeader"], [data-testid="stToolbar"] {{
    background: rgba(0,0,0,0);
    visibility: hidden;
    }}

    .centered {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
    }}

    .stButton>button {{
        background-color: #28a745;
        color: white;
        font-size: 28px;
        font-weight: bold;
        padding: 0.8em 2.5em;
        border: none;
        border-radius: 15px;
        cursor: pointer;
        box-shadow: 2px 2px 12px gray;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Centered title + button
    st.markdown(
        """
        <div class="top">
            <h1 style='font-size: 65px; color: black;'>🌿 Welcome to Smart Crop Disease Detection</h1>
            <h3 style='color: #ffde59;'>Protect your Crops with 1 Click!</h3><br>
        """, unsafe_allow_html=True)

    start_button = st.button("🚀 Detect Disease Now")
    st.markdown("</div>", unsafe_allow_html=True)

    if start_button:
        st.session_state.entered = True

    st.stop()



# Set page config
st.set_page_config(page_title="Smart Crop Detection", layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/styles.css")


# --- Load Model ---
model = load_model("crop_model.h5")

# --- Class Labels ---
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --- Disease Recommendations ---
disease_solutions = {
    'Apple___Apple_scab': 'Apply fungicides early. Use resistant apple varieties and ensure proper pruning.',
    'Apple___Black_rot': 'Remove infected leaves. Use fungicides during the growing season.',
    'Apple___Cedar_apple_rust': 'Avoid planting near cedar trees. Apply appropriate fungicides.',
    'Apple___healthy': 'Your plant is healthy! No treatment needed.',
    'Blueberry___healthy': 'Plant is healthy. Maintain regular watering and mulching.',
    'Cherry_(including_sour)___Powdery_mildew': 'Prune infected branches. Apply sulfur-based fungicides.',
    'Cherry_(including_sour)___healthy': 'No disease detected. Ensure proper airflow and fertilization.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Use crop rotation. Apply foliar fungicides.',
    'Corn_(maize)___Common_rust_': 'Use resistant hybrids. Apply fungicide if infestation is high.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Practice crop rotation. Remove debris and apply fungicide.',
    'Corn_(maize)___healthy': 'Corn is healthy. Monitor for pests and maintain watering.',
    'Grape___Black_rot': 'Remove mummified fruit. Apply fungicides during bloom.',
    'Grape___Esca_(Black_Measles)': 'Prune infected vines. Avoid over-irrigation.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Remove infected leaves. Use mancozeb-based sprays.',
    'Grape___healthy': 'Vine is healthy. Maintain pruning and soil nutrition.',
    'Orange___Haunglongbing_(Citrus_greening)': 'Use insecticides for psyllids. Remove infected trees.',
    'Peach___Bacterial_spot': 'Apply copper-based sprays. Remove and destroy affected fruit.',
    'Peach___healthy': 'Healthy tree! Keep monitoring and water regularly.',
    'Pepper,_bell___Bacterial_spot': 'Use disease-free seeds. Apply bactericides and avoid overhead watering.',
    'Pepper,_bell___healthy': 'Plant is healthy. Continue good agronomic practices.',
    'Potato___Early_blight': 'Use certified seeds. Apply chlorothalonil-based fungicides.',
    'Potato___Late_blight': 'Destroy infected plants. Apply phosphorous acid fungicides.',
    'Potato___healthy': 'No disease. Maintain healthy irrigation practices.',
    'Raspberry___healthy': 'All good! Prune regularly and keep area weed-free.',
    'Soybean___healthy': 'Healthy crop. Rotate with cereals to keep diseases away.',
    'Squash___Powdery_mildew': 'Improve ventilation. Use neem oil or sulfur sprays.',
    'Strawberry___Leaf_scorch': 'Avoid overhead irrigation. Use resistant varieties.',
    'Strawberry___healthy': 'No disease found. Monitor for pests and mulch well.',
    'Tomato___Bacterial_spot': 'Remove affected plants. Use copper-based sprays.',
    'Tomato___Early_blight': 'Use crop rotation. Spray with mancozeb or chlorothalonil.',
    'Tomato___Late_blight': 'Destroy infected plants. Use fungicides like cymoxanil.',
    'Tomato___Leaf_Mold': 'Increase airflow. Use protective fungicides.',
    'Tomato___Septoria_leaf_spot': 'Remove lower leaves. Spray with fungicides like Daconil.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Spray insecticidal soap. Introduce natural predators like ladybugs.',
    'Tomato___Target_Spot': 'Prune and space properly. Apply appropriate fungicide.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Remove infected plants. Control whiteflies using sticky traps.',
    'Tomato___Tomato_mosaic_virus': 'Avoid tobacco handling. Disinfect tools regularly.',
    'Tomato___healthy': 'Your tomato plant is thriving! Keep it up.'
}


videos = [
    {
        "title": "🧪 How to Use Fertilizers Effectively",
        "url": "https://youtu.be/WCD6iOQuetw",
        "thumbnail": "https://img.youtube.com/vi/WCD6iOQuetw/0.jpg"
    },
    {
        "title": "🦠 How to Identify and Control Plant Diseases in Tomato",
        "url": "https://youtu.be/gjMIh19zH7k",
        "thumbnail": "https://img.youtube.com/vi/gjMIh19zH7k/0.jpg"
    },
    {
        "title": "🫑 How to Identify and Control Plant Diseases in Bell Pepper",
        "url": "https://youtu.be/wATQm6NnPoo",
        "thumbnail": "https://img.youtube.com/vi/wATQm6NnPoo/0.jpg"
    },
    {
        "title": "🌱 Best Practices for Seeding and Planting",
        "url": "https://youtu.be/Awtw0GAKb2c",
        "thumbnail": "https://img.youtube.com/vi/Awtw0GAKb2c/0.jpg"
    },
    {
        "title": "🌿 Introduction to Organic Farming",
        "url": "https://youtu.be/lRyXlvIJFWI",
        "thumbnail": "https://img.youtube.com/vi/lRyXlvIJFWI/0.jpg"
    },
    {
        "title": "📢 Telugu: Natural Farming by Subash Palekar",
        "url": "https://youtu.be/80vDBZx-z5w",
        "thumbnail": "https://img.youtube.com/vi/80vDBZx-z5w/0.jpg"
    }
]




# --- Initialize Session State ---
for key in ['predicted_class', 'confidence', 'prediction_done', 'show_solution']:
    if key not in st.session_state:
        st.session_state[key] = None if key in ['predicted_class', 'confidence'] else False

# --- Sidebar Navigation ---
page = st.sidebar.radio("📂 Navigate", [
    "🖼️ Predict from Upload",
    "📸 Camera Capture",
    "🧾 Disease Info",
    #"📊 Dataset Insights",
    "🌿 Prevention Tips",
    "📺 Farming Tutorials"
])

# Initialize translator
translator = Translator()

# Add language selector in sidebar
languages = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Kannada": "kn",
    "Tamil": "ta",
    "Marathi": "mr"
}

selected_lang = st.sidebar.selectbox("🌐 Choose Language", list(languages.keys()))
selected_code = languages[selected_lang]

def translate_text(text, dest_lang):
    if dest_lang == "en":
        return text
    try:
        translated = translator.translate(text, dest=dest_lang)
        return translated.text
    except Exception as e:
        return text  # Fallback to original if translation fails

# --- Page 1: Upload and Predict ---
if page == "🖼️ Predict from Upload":
    st.title("🖼️ " + translate_text("Smart Crop Disease Detection", selected_code))
    st.subheader(translate_text("Upload a leaf image to predict the disease.", selected_code))

    # 👉 Paste your existing prediction code here
    uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption="Uploaded Image", use_container_width=True)

        image_pil = ImageOps.fit(image_pil, (224, 224), method=Image.Resampling.LANCZOS)
        img_array = np.array(image_pil) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)

        if st.button("🔍 Predict", key="predict_button"):
            prediction = model.predict(img_array)
            pred_index = np.argmax(prediction)
            pred_label = class_labels[pred_index]
            confidence = float(np.max(prediction))
            

            st.session_state.predicted_class = pred_label
            st.session_state.confidence = confidence
            st.session_state.prediction_done = True
            st.session_state.show_solution = False

    if st.session_state.prediction_done and not st.session_state.show_solution:
        translated_disease = translate_text(st.session_state.predicted_class, selected_code)
        st.success(f"✅ Predicted Disease: {translated_disease}")

        

        # Color-coded confidence
        conf = st.session_state.confidence
        if conf > 0.85:
            st.success(f"📊 High Confidence: {conf:.2f}")
        elif conf > 0.6:
            st.warning(f"📊 Medium Confidence: {conf:.2f}")
        else:
            st.error(f"📊 Low Confidence: {conf:.2f} — consider retaking the image.")

        if st.button("💡 Give Solution", key="solution_button"):
            st.session_state.show_solution = True
        

    if st.session_state.show_solution:
        st.markdown("---")
        st.subheader("🩺 " + translate_text("Suggested Treatment & Recommendation", selected_code))

        disease = st.session_state.predicted_class
        recommendation = disease_solutions.get(disease, "No recommendation available.")

        translated_disease = translate_text(disease, selected_code)
        translated_recommendation = translate_text(recommendation, selected_code)

        st.markdown(f"### 🪴 Disease: `{translated_disease}`")
        st.warning(f"💡 Recommendation: {translated_recommendation}")


        st.button("⬅ Back", on_click=lambda: st.session_state.update({'show_solution': False}), key="back_button")
        


# --- Page 2: Camera Capture ---
elif page == "📸 Camera Capture":
    # Translated heading and subheading
    st.title("📸 " + translate_text("Camera-Based Detection", selected_code))
    st.subheader(translate_text("Use your webcam to capture a live image of a plant leaf.", selected_code))


    # Capture photo from camera
    img_file_buffer = st.camera_input(translate_text("📷 Take a photo", selected_code))

    if img_file_buffer is not None:
        # Read and preprocess image
        image_pil = Image.open(img_file_buffer).convert("RGB")
        st.image(image_pil, caption=translate_text("Captured Image", selected_code), use_container_width=True)

        image_pil = ImageOps.fit(image_pil, (224, 224), method=Image.Resampling.LANCZOS)
        img_array = np.array(image_pil) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)

        # Prediction
        prediction = model.predict(img_array)
        pred_index = np.argmax(prediction)
        pred_label = class_labels[pred_index]
        confidence = float(np.max(prediction))

        # Save in session for other pages
        st.session_state.predicted_class = pred_label
        st.session_state.confidence = confidence

        # Translations
        translated_label = translate_text(pred_label, selected_code)

        # Show result
        st.success(f"✅ {translate_text('Predicted Disease:', selected_code)} {translated_label}")
        if confidence > 0.85:
            st.success(f"📊 {translate_text('High Confidence:', selected_code)} {confidence:.2f}")
        elif confidence > 0.6:
            st.warning(f"📊 {translate_text('Medium Confidence:', selected_code)} {confidence:.2f}")
        else:
            st.error(f"📊 {translate_text('Low Confidence:', selected_code)} {confidence:.2f}")

        # Optional: Show recommendation
        if st.button(translate_text("💡 Show Solution", selected_code), key="cam_solution_btn"):
            recommendation = disease_solutions.get(pred_label, "No recommendation available.")
            translated_reco = translate_text(recommendation, selected_code)

            st.markdown(f"### 🪴 {translate_text('Disease:', selected_code)} `{translated_label}`")
            st.warning(f"💡 {translate_text('Recommendation:', selected_code)} {translated_reco}")

# --- Page 3: Disease Info ---
elif page == "🧾 Disease Info":
    st.title("🧾 " + translate_text("Disease Information", selected_code))

    if 'predicted_class' in st.session_state and st.session_state.predicted_class:
        disease = st.session_state.predicted_class
        st.subheader("🔬 " + translate_text(f"Detailed Info on: {disease}", selected_code))

        # Description and Wikipedia URL Dictionary
        disease_details = {
    "Apple___Apple_scab": {
        "desc": "Apple scab is caused by the fungus Venturia inaequalis. It forms olive-green spots on leaves and fruits, which can darken and crack as the disease progresses. Scab thrives in moist, cool spring conditions. It leads to deformed fruits and premature leaf drop, impacting yield. Management includes pruning, resistant varieties, and fungicide application. Sanitation such as removing fallen leaves is crucial. It's one of the most common apple diseases globally.",
        "url": "https://en.wikipedia.org/wiki/Apple_scab"
    },
    "Apple___Black_rot": {
        "desc": "Black rot is caused by Botryosphaeria obtusa, affecting apple fruit, leaves, and bark. Symptoms include leaf spots, fruit decay, and cankers. Fruit lesions start as small, dark circles and eventually rot. The disease survives in infected plant debris. Management includes pruning, fungicide sprays, and orchard hygiene. Black rot can be highly destructive if not managed in warm, humid climates.",
        "url": "https://en.wikipedia.org/wiki/Black_rot"
    },
    "Apple___Cedar_apple_rust": {
        "desc": "Cedar apple rust is a fungal disease caused by Gymnosporangium juniperi-virginianae. It requires both apple and cedar trees to complete its life cycle. On apples, it causes bright orange leaf spots and defoliation. The fungus forms galls on cedar trees, which release spores. Control involves removing nearby junipers and applying fungicides in early spring. Resistant cultivars are also effective.",
        "url": "https://en.wikipedia.org/wiki/Cedar-apple_rust"
    },
    "Apple___healthy": {
        "desc": "Healthy apple leaves are free from any lesions, deformities, or discoloration. The foliage appears uniformly green and firm. Maintaining healthy trees involves proper irrigation, fertilization, pruning, and disease prevention. Regular inspection for pests and early signs of fungal or bacterial issues is key to sustaining tree health and yield.",
        "url": "https://en.wikipedia.org/wiki/Apple"
    },
    "Blueberry___healthy": {
        "desc": "Healthy blueberry plants exhibit lush green foliage and firm, plump berries. There are no signs of wilting, leaf spots, or discoloration. Blueberries require acidic soil, good drainage, and adequate sunlight. Disease prevention includes removing fallen debris and proper spacing. Healthy plants yield flavorful and antioxidant-rich fruit.",
        "url": "https://en.wikipedia.org/wiki/Blueberry"
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "desc": "Powdery mildew on cherry trees is caused by Podosphaera clandestina. It produces white powdery fungal growth on leaves, buds, and fruit. Infected leaves curl and drop early. The disease spreads in dry, warm conditions. Preventive fungicide use, pruning for air circulation, and resistant varieties help manage this disease.",
        "url": "https://en.wikipedia.org/wiki/Powdery_mildew"
    },
    "Cherry_(including_sour)___healthy": {
        "desc": "Healthy cherry trees have vibrant green leaves and strong flowering. No signs of mildew, gummosis, or insect damage should be present. Proper fertilization, soil pH, and pest control ensure optimal growth. Regular pruning supports good structure and air movement, promoting disease resistance.",
        "url": "https://en.wikipedia.org/wiki/Cherry"
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "desc": "Gray leaf spot is a fungal disease in corn caused by Cercospora zeae-maydis. It forms narrow, rectangular, grayish lesions on leaves, reducing photosynthesis. The disease thrives in warm, humid environments. Crop rotation, resistant hybrids, and timely fungicide application help manage outbreaks. Severe infection can drastically lower yield.",
        "url": "https://en.wikipedia.org/wiki/Gray_leaf_spot_(maize)"
    },
    "Corn_(maize)___Common_rust_": {
        "desc": "Common rust in corn is caused by Puccinia sorghi. It creates reddish-brown pustules on leaves, often surrounded by yellow halos. The disease spreads via windborne spores in humid conditions. Management includes resistant hybrids and monitoring for severity. While not always yield-limiting, it can be problematic in tropical areas.",
        "url": "https://en.wikipedia.org/wiki/Common_rust"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "desc": "Northern leaf blight is caused by Exserohilum turcicum. It forms large, cigar-shaped gray lesions on corn leaves. The disease spreads in prolonged wet conditions and can lead to reduced yield. Management strategies include using resistant hybrids, crop rotation, and fungicides. It is a significant foliar disease in maize production.",
        "url": "https://en.wikipedia.org/wiki/Northern_corn_leaf_blight"
    },
    "Corn_(maize)___healthy": {
        "desc": "A healthy maize plant shows broad, green, turgid leaves with no signs of rust, blight, or lesions. Corn plants grow optimally in warm conditions with sufficient sunlight and water. Maintaining proper spacing, fertilization, and pest control ensures good growth. Regular field monitoring helps in early detection of diseases.",
        "url": "https://en.wikipedia.org/wiki/Maize"
    },
    "Grape___Black_rot": {
        "desc": "Black rot is a grape disease caused by Guignardia bidwellii. It produces dark, sunken lesions on leaves, tendrils, and fruit. Infected berries shrivel into hard, black mummies. Warm, wet conditions promote rapid spread. Proper pruning, fungicide sprays, and removal of infected fruit are effective in managing black rot.",
        "url": "https://en.wikipedia.org/wiki/Black_rot_(grape_disease)"
    },
    "Grape___Esca_(Black_Measles)": {
        "desc": "Esca or Black Measles is a complex disease involving several fungi like Phaeomoniella chlamydospora. It leads to tiger-stripe patterns on leaves and blackened, shriveled berries. Internal trunk rot and sudden vine death may occur. Control involves pruning, avoiding large cuts, and removing infected wood.",
        "url": "https://en.wikipedia.org/wiki/Esca_(grape_disease)"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "desc": "Caused by Pseudocercospora vitis, this disease creates brown angular spots on grape leaves. It reduces photosynthesis and can cause premature leaf drop. Moist conditions and poor ventilation increase risk. Fungicide application, pruning, and improved airflow are recommended to manage Isariopsis leaf spot.",
        "url": "https://en.wikipedia.org/wiki/Isariopsis_leaf_spot"
    },
    "Grape___healthy": {
        "desc": "Healthy grapevines have vibrant, green leaves, firm canes, and blemish-free fruit. Consistent pruning, proper fertilization, and sunlight exposure support vine health. Absence of lesions or necrosis indicates disease-free status. Good vineyard management minimizes risks of mildew, rot, and pest infestation.",
        "url": "https://en.wikipedia.org/wiki/Grape"
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "desc": "Huanglongbing (HLB), or citrus greening, is a deadly citrus disease caused by a bacterium and spread by the Asian citrus psyllid. Infected trees show yellowing, fruit drop, and misshapen, bitter fruits. There is no cure. Control includes removal of infected trees and psyllid vector management.",
        "url": "https://en.wikipedia.org/wiki/Huanglongbing"
    },
    "Peach___Bacterial_spot": {
        "desc": "Bacterial spot in peaches is caused by Xanthomonas arboricola. Symptoms include water-soaked lesions on leaves, fruit, and twigs. It thrives in wet, humid environments and spreads rapidly. Control strategies include copper sprays, resistant cultivars, and pruning for improved air circulation.",
        "url": "https://en.wikipedia.org/wiki/Bacterial_spot_(stone_fruit)"
    },
    "Peach___healthy": {
        "desc": "Healthy peach trees produce lush green leaves, unblemished fruits, and show no signs of bacterial or fungal diseases. Proper irrigation, fertilization, and disease surveillance support their growth. Pruning and sunlight exposure also help maintain tree vigor and fruit quality.",
        "url": "https://en.wikipedia.org/wiki/Peach"
    },
    "Pepper,_bell___Bacterial_spot": {
        "desc": "Bacterial spot in bell peppers is caused by Xanthomonas campestris. It leads to water-soaked lesions on leaves and fruit, which turn dark and scabby. It spreads through rain splash and contaminated tools. Management includes copper-based sprays, resistant varieties, and strict field sanitation.",
        "url": "https://en.wikipedia.org/wiki/Bacterial_leaf_spot"
    },
    "Pepper,_bell___healthy": {
        "desc": "Healthy bell pepper plants have green, glossy leaves and firm, brightly colored fruit. There should be no leaf spotting or wrinkling. Healthy plants grow well in warm, sunny environments with rich, well-drained soil. Regular watering and fertilizing improve yields and disease resistance.",
        "url": "https://en.wikipedia.org/wiki/Bell_pepper"
    },
    "Potato___Early_blight": {
        "desc": "Early blight in potatoes is caused by Alternaria solani. It produces brown concentric-ring lesions on lower leaves, causing them to wither. High humidity and warm temperatures favor it. Management involves crop rotation, resistant cultivars, and fungicide sprays to minimize yield loss.",
        "url": "https://en.wikipedia.org/wiki/Early_blight"
    },
    "Potato___Late_blight": {
        "desc": "Late blight is caused by Phytophthora infestans, the same pathogen responsible for the Irish Potato Famine. It causes water-soaked lesions that rapidly turn brown and kill foliage. It spreads quickly under moist conditions. Control requires fungicides and removal of infected plants.",
        "url": "https://en.wikipedia.org/wiki/Late_blight"
    },
    "Potato___healthy": {
        "desc": "Healthy potato plants feature robust stems and deep green leaves without lesions or wilting. They grow best in well-drained soil with moderate temperatures. Crop rotation, early disease detection, and balanced fertilization contribute to healthy growth and tuber production.",
        "url": "https://en.wikipedia.org/wiki/Potato"
    },
    "Raspberry___healthy": {
        "desc": "Healthy raspberry plants exhibit vibrant leaves, upright canes, and plump berries. Adequate spacing, water, and pruning are crucial for airflow and light penetration. Pest monitoring and sanitation help avoid fungal infections. Disease-free plants yield high-quality fruits.",
        "url": "https://en.wikipedia.org/wiki/Raspberry"
    },
    "Soybean___healthy": {
        "desc": "A healthy soybean plant has dark green leaves, erect stems, and no signs of yellowing or necrotic spots. Optimal conditions include warm temperatures and well-drained soil. Proper fertilization and pest management promote robust growth and good seed yield.",
        "url": "https://en.wikipedia.org/wiki/Soybean"
    },
    "Squash___Powdery_mildew": {
        "desc": "Powdery mildew in squash appears as white fungal growth on leaf surfaces, often in patches. It is caused by Podosphaera xanthii and spreads easily in dry but humid conditions. Fungicides, resistant varieties, and good air circulation are important for control.",
        "url": "https://en.wikipedia.org/wiki/Powdery_mildew"
    },
    "Strawberry___Leaf_scorch": {
        "desc": "Leaf scorch in strawberries is caused by the fungus Diplocarpon earliana. Symptoms include purple spots on leaves that expand and merge, leading to scorched appearance. The disease thrives in wet environments. Management includes crop rotation, resistant varieties, and fungicide applications.",
        "url": "https://en.wikipedia.org/wiki/Leaf_scorch_(strawberry)"
    },
    "Strawberry___healthy": {
        "desc": "Healthy strawberry plants show green trifoliate leaves and bright red berries. There should be no signs of leaf spots, mildew, or pests. Proper spacing, mulching, and watering ensure healthy growth and higher yields. Regular inspection keeps the crop free from disease.",
        "url": "https://en.wikipedia.org/wiki/Strawberry"
    },
    "Tomato___Bacterial_spot": {
        "desc": "Bacterial spot in tomatoes is caused by Xanthomonas spp. It leads to dark, water-soaked lesions on leaves and fruit. Spread occurs via splashing water and infected seeds. Copper-based fungicides and crop rotation are common control methods. Severe infections reduce fruit marketability.",
        "url": "https://en.wikipedia.org/wiki/Bacterial_leaf_spot"
    },
    "Tomato___Early_blight": {
        "desc": "Early blight, caused by Alternaria solani, is a widespread tomato disease. It creates dark concentric lesions on older leaves and stems. Infected plants may experience leaf drop and reduced yield. Management involves crop rotation, resistant varieties, and fungicide sprays.",
        "url": "https://en.wikipedia.org/wiki/Early_blight"
    },
    "Tomato___Late_blight": {
        "desc": "Late blight in tomatoes is caused by Phytophthora infestans. It causes dark, greasy lesions on leaves, stems, and fruits. This disease spreads rapidly in cool, wet conditions and is highly destructive. Entire crops can be lost if not treated early. It can be managed by resistant cultivars, fungicides, and field sanitation.",
        "url": "https://en.wikipedia.org/wiki/Late_blight"
    },
    "Tomato___Leaf_Mold": {
        "desc": "Tomato leaf mold is caused by Passalora fulva and typically affects greenhouse tomatoes. It presents as yellow spots on upper leaf surfaces with moldy olive-green patches on the underside. The disease thrives in high humidity. Good ventilation, resistant varieties, and fungicide application help control it.",
        "url": "https://en.wikipedia.org/wiki/Tomato_leaf_mold"
    },
    "Tomato___Septoria_leaf_spot": {
        "desc": "Caused by Septoria lycopersici, this disease causes small, circular gray spots with dark borders on tomato leaves. It starts on lower leaves and moves upward, leading to defoliation. High humidity favors its spread. Control includes pruning, fungicides, and crop rotation.",
        "url": "https://en.wikipedia.org/wiki/Septoria_leaf_spot"
    },
    "Tomato___Spider_mites_Two_spotted_spider_mite": {
        "desc": "Two-spotted spider mites (Tetranychus urticae) are tiny pests that suck sap from tomato leaves. They cause speckled discoloration, leaf curling, and webbing. Infestation leads to stunted growth. Management includes miticides, insecticidal soaps, and natural predators like ladybugs.",
        "url": "https://en.wikipedia.org/wiki/Tetranychus_urticae"
    },
    "Tomato___Target_Spot": {
        "desc": "Target spot is a fungal disease caused by Corynespora cassiicola. It forms concentric lesions on tomato leaves and sometimes fruits. Symptoms resemble a target pattern. It thrives in high humidity and poor airflow. Control includes fungicides, crop rotation, and field sanitation.",
        "url": "https://en.wikipedia.org/wiki/Corynespora_cassiicola"
    },
    "Tomato___Tomato_YellowLeaf_Curl_Virus": {
        "desc": "Tomato Yellow Leaf Curl Virus (TYLCV) is a devastating viral disease spread by whiteflies. It causes yellowing, leaf curling, stunted growth, and reduced fruit yield. There's no cure, but management includes using resistant varieties and controlling whitefly populations with insecticides.",
        "url": "https://en.wikipedia.org/wiki/Tomato_yellow_leaf_curl_virus"
    },
    "Tomato___Tomato_mosaic_virus": {
        "desc": "Tomato mosaic virus (ToMV) is a highly contagious virus that causes mottled leaf patterns, stunted growth, and deformed fruit. It spreads via tools, hands, and infected seeds. There is no chemical cure, so hygiene, seed treatment, and resistant cultivars are key to management.",
        "url": "https://en.wikipedia.org/wiki/Tomato_mosaic_virus"
    },
    "Tomato___healthy": {
        "desc": "A healthy tomato plant has lush green leaves, thick stems, and blemish-free fruits. There are no signs of leaf spots, curling, or wilting. With proper spacing, watering, and pest control, plants can thrive and produce abundant, high-quality tomatoes throughout the season.",
        "url": "https://en.wikipedia.org/wiki/Tomato"
    }
            # 🔁 You can keep adding more diseases like this later!
        }

        # Get description and link
        if disease in disease_details:
            info = disease_details[disease]
            translated_disease = translate_text(disease, selected_code)
            translated_desc = translate_text(info['desc'], selected_code)

            
            st.markdown(f"**📝 Description:**\n\n{translated_desc}")    


            st.markdown(f"""
                <a href="{info['url']}" target="_blank">
                    <button style='margin-top: 20px; padding: 0.5em 1em; background-color:#4CAF50; color:white; border:none; border-radius:5px;'>
                        🌐 Learn More on Wikipedia
                    </button>
                </a>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No detailed description available. Try another class or click below to search.")
            search_url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(f"{disease} plant disease")
            st.markdown(f"[Search {disease} on Google]({search_url})", unsafe_allow_html=True)

    else:
        st.warning("⚠️ " + translate_text("No disease has been predicted yet. Please use 'Upload' or 'Camera' page first.", selected_code))



# --- Page 5: Prevention Tips ---
elif page == "🌿 Prevention Tips":
    st.title("🌿 " + translate_text("Prevention Tips", selected_code))
    st.subheader(translate_text("Tips to manage plant diseases", selected_code))


    # Check if disease is predicted
    if 'predicted_class' in st.session_state and st.session_state.predicted_class:
        disease = st.session_state.predicted_class
        st.subheader("🌱 " + translate_text(f"Disease: {disease}", selected_code))

        # Add disease solutions/tips for each disease here
        disease_solutions = {
    "Apple___Apple_scab": {
        "prevention": "Apple scab can be managed by using resistant apple cultivars, applying fungicides, and practicing good sanitation by removing fallen leaves and infected fruit. Ensure proper air circulation between trees to reduce moisture buildup and help prevent spore germination.",
        "fertilizer": "Use balanced fertilizers with a ratio of 10-10-10 NPK. Avoid over-fertilizing with nitrogen as it encourages fungal growth."
    },
    "Apple___Black_rot": {
        "prevention": "Prune infected branches, apply copper-based fungicides, and remove infected fruit. Ensure good airflow and avoid overcrowding trees. Disinfect tools between uses to prevent the spread of the disease.",
        "fertilizer": "Use a balanced fertilizer with moderate nitrogen content to support healthy growth without promoting excessive foliage."
    },
    "Apple___Cedar_apple_rust": {
        "prevention": "Prune and dispose of infected leaves, control nearby cedar trees, and apply fungicides during the growing season. Avoid overhead irrigation to reduce moisture on leaves.",
        "fertilizer": "Apply slow-release nitrogen fertilizers to avoid excessive plant growth, and use organic compost for healthier soil."
    },
    "Apple___healthy": {
        "prevention": "Ensure proper irrigation, disease monitoring, and crop rotation to maintain a healthy apple tree. Keep the area around the trees free of weeds and debris.",
        "fertilizer": "Apply a balanced 10-10-10 fertilizer to support healthy root and fruit development."
    },
    "Blueberry___healthy": {
        "prevention": "Maintain acidic soil (pH 4.5-5.5) for blueberries. Control pests and prevent waterlogging by using good drainage.",
        "fertilizer": "Use fertilizers specifically for acid-loving plants. A balanced mix with micronutrients like magnesium and iron is beneficial."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "prevention": "Prune infected leaves and shoots, avoid overhead irrigation, and apply sulfur or neem oil to reduce fungal spores.",
        "fertilizer": "Use low-nitrogen fertilizers to avoid promoting excessive foliage growth. Add organic compost to improve soil health."
    },
    "Cherry_(including_sour)___healthy": {
        "prevention": "Regularly inspect trees for pests and diseases, and prune to promote airflow.",
        "fertilizer": "Apply a balanced 10-10-10 fertilizer for overall plant health. Consider using compost for additional nutrients."
    },
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": {
        "prevention": "Avoid excessive irrigation and ensure proper spacing. Rotate crops to minimize disease buildup.",
        "fertilizer": "Apply nitrogen-rich fertilizers, but avoid over-fertilizing, as it may make plants more susceptible to disease."
    },
    "Corn_(maize)___Common_rust_": {
        "prevention": "Use resistant maize varieties, rotate crops, and apply fungicides during the growing season.",
        "fertilizer": "Use a balanced 10-10-10 NPK fertilizer to provide the essential nutrients for healthy growth."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "prevention": "Plant resistant varieties, rotate crops, and apply fungicides as a preventive measure. Remove infected plant debris from the field.",
        "fertilizer": "Use a balanced fertilizer with adequate nitrogen to support plant growth but avoid over-fertilizing."
    },
    "Corn_(maize)___healthy": {
        "prevention": "Maintain proper spacing between plants, rotate crops, and monitor for pests and diseases.",
        "fertilizer": "A balanced 10-10-10 NPK fertilizer should be applied at planting, followed by a nitrogen-based fertilizer during the growing season."
    },
    "Grape___Black_rot": {
        "prevention": "Prune out and destroy infected berries and canes promptly. Apply protective fungicide sprays at bloom and preharvest to prevent black rot. Ensure good air circulation by training vines and thinning the canopy.",
        "fertilizer": "Use a balanced 10-10-10 NPK fertilizer in early spring, then switch to a low-nitrogen, high-potassium feed (6-12-12) during fruit set to improve disease resistance."
    },
    "Grape___Esca_(Black_Measles)": {
        "prevention": "Remove and burn heavily infected wood during dormancy. Avoid large pruning wounds and disinfect tools between cuts. Improve drainage and avoid water stress to reduce fungal entry.",
        "fertilizer": "Apply a balanced 10-10-10 fertilizer in spring, plus micronutrients like iron and magnesium as a foliar spray to strengthen vine health."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "prevention": "Spray fungicides at regular intervals in wet weather. Remove fallen leaves and fruit to break the disease cycle. Prune for better airflow and reduce humidity within the canopy.",
        "fertilizer": "Use a balanced 10-10-10 fertilizer application in early season, followed by potassium-rich 5-10-15 at berry development."
    },
    "Grape___healthy": {
        "prevention": "Maintain proper canopy management, monitor for pests, and remove any debris. Rotate blocks with non–vine crops if possible.",
        "fertilizer": "Side-dress with 10-10-10 NPK at budbreak, then apply micronutrients as needed based on soil tests."
    },

    # Citrus
    "Orange___Haunglongbing_(Citrus_greening)": {
        "prevention": "Control the Asian citrus psyllid vector with insecticides and yellow sticky traps. Remove and destroy infected trees immediately to reduce sources of infection.",
        "fertilizer": "Use a citrus‑specific fertilizer (e.g., 8-10-8 NPK) applied in three split doses during spring, summer and fall, plus regular micronutrient sprays (Mg, Fe, Zn)."
    },

    # Stone Fruits
    "Peach___Bacterial_spot": {
        "prevention": "Apply copper sprays at bud break and post‑bloom. Prune out cankered wood and disinfect pruning tools between cuts.",
        "fertilizer": "Side‑dress with a balanced 10-10-10 fertilizer in early spring and again after harvest to promote healthy growth."
    },
    "Peach___healthy": {
        "prevention": "Ensure good air circulation, avoid overhead watering, and remove fallen fruit promptly.",
        "fertilizer": "Use 8-8-6 NPK fertilizer in early spring and mulch with compost for slow nutrient release."
    },

    # Peppers
    "Pepper,_bell___Bacterial_spot": {
        "prevention": "Use certified disease‑free seeds, apply copper‑based bactericides, and avoid overhead irrigation to minimize leaf wetness.",
        "fertilizer": "Fertilize with 5-10-10 NPK at planting, then side‑dress with calcium nitrate to strengthen cell walls."
    },
    "Pepper,_bell___healthy": {
        "prevention": "Practice crop rotation and maintain field hygiene to prevent disease build‑up.",
        "fertilizer": "Apply a balanced 10-10-10 fertilizer at planting and again mid‑season."
    },

    # Potatoes
    "Potato___Early_blight": {
        "prevention": "Rotate crops every 3 years and remove cull piles. Apply protectant fungicides (chlorothalonil) every 7–10 days in humid weather.",
        "fertilizer": "Side‑dress with 10-10-20 NPK at tuber initiation and use magnesium sulfate foliar spray to reduce stress."
    },
    "Potato___Late_blight": {
        "prevention": "Plant certified seed potatoes, space plants to improve air flow, and spray with Oomycete fungicides (e.g., mancozeb) at 7‑day intervals.",
        "fertilizer": "Use a balanced 10-10-20 NPK at planting and avoid excessive nitrogen to reduce lush growth."
    },
    "Potato___healthy": {
        "prevention": "Keep rows weed‑free, monitor for volunteer potatoes, and rotate with non–solanaceous crops.",
        "fertilizer": "Apply 10-10-20 NPK at planting and side‑dress with calcium nitrate at hilling."
    },

    # Raspberries & Soybeans
    "Raspberry___healthy": {
        "prevention": "Prune to remove old canes, ensure good drainage, and mulch to suppress weeds.",
        "fertilizer": "Use 10-10-10 NPK in early spring and a foliar micronutrient spray (Fe, B, Zn) at bud break."
    },
    "Soybean___healthy": {
        "prevention": "Rotate with cereal crops, inoculate seed with Rhizobium, and avoid fields with a history of soil‑borne diseases.",
        "fertilizer": "Soybeans fix their own nitrogen; apply 0-20-20 (P‑K) at planting based on soil test recommendations."
    },

    # Squash & Strawberries
    "Squash___Powdery_mildew": {
        "prevention": "Promote airflow, avoid overhead watering, and apply sulfur or potassium bicarbonate when first symptoms appear.",
        "fertilizer": "Side‑dress with a balanced 10-10-10 NPK at flowering and again at fruit set."
    },
    "Strawberry___Leaf_scorch": {
        "prevention": "Avoid overhead irrigation, remove infected leaves, and apply protective fungicides (captan) early in the season.",
        "fertilizer": "Use 10-10-10 NPK at renovation and mulched compost for sustained nutrition."
    },
    "Strawberry___healthy": {
        "prevention": "Maintain proper plant spacing, use drip irrigation, and rotate planting beds every 3 years.",
        "fertilizer": "Side‑dress with 10-10-10 NPK after first harvest and apply calcium nitrate foliar sprays monthly."
    },

    # Tomatoes
    "Tomato___Bacterial_spot": {
        "prevention": "Use disease‑free transplants, avoid wet foliage, and spray copper‑based products at first sign of infection.",
        "fertilizer": "Apply 5-10-10 NPK at transplanting, then side‑dress with calcium nitrate to prevent blossom end rot."
    },
    "Tomato___Early_blight": {
        "prevention": "Rotate crops, remove lower leaves, and apply chlorothalonil fungicides at 7‑day intervals.",
        "fertilizer": "Use 10-10-10 NPK at planting and avoid excess nitrogen to reduce susceptibility."
    },
    "Tomato___Late_blight": {
        "prevention": "Grow resistant varieties, apply phosphonates or mancozeb, and remove infected plants immediately.",
        "fertilizer": "Use a balanced 10-10-10 fertilizer at planting and side‑dress with 8-32-16 (high P‑K) at fruit set."
    },
    "Tomato___Leaf_Mold": {
        "prevention": "Improve greenhouse ventilation or space plants widely, and apply chlorothalonil or copper sprays when needed.",
        "fertilizer": "Apply 5-10-10 NPK early, then foliar magnesium sulfate to strengthen cell walls."
    },
    "Tomato___Septoria_leaf_spot": {
        "prevention": "Remove and destroy infected foliage, mulch to prevent soil splash, and apply chlorothalonil fungicides.",
        "fertilizer": "Use 10-10-10 NPK at planting and side‑dress with 0-0-60 (potassium sulfate) at flowering."
    },
    "Tomato___Spider_mites_Two_spotted_spider_mite": {
        "prevention": "Spray water to dislodge mites, introduce predatory mites, and use acaricides if populations are high.",
        "fertilizer": "Apply balanced 10-10-10 NPK and foliar calcium sprays to reduce plant stress."
    },
    "Tomato___Target_Spot": {
        "prevention": "Use crop rotation, avoid overhead watering, and apply strobilurin fungicides early in infection cycles.",
        "fertilizer": "Use 10-10-10 NPK at planting with extra potassium (0-0-60) at fruit fill for stronger skin."
    },
    "Tomato___Tomato_YellowLeaf_Curl_Virus": {
        "prevention": "Control whitefly vectors with yellow sticky traps and insecticides, and remove infected plants quickly.",
        "fertilizer": "Apply 10-10-10 NPK at planting and side‑dress with 20-20-20 (complete) every 3 weeks."
    },
    "Tomato___Tomato_mosaic_virus": {
        "prevention": "Disinfect tools, avoid handling when sweaty, and remove tobacco sources near tomato crops.",
        "fertilizer": "Use 10-10-10 NPK at transplanting and booster foliar sprays of calcium nitrate monthly."
    },
    "Tomato___healthy": {
        "prevention": "Rotate Solanaceous crops, mulch to suppress weeds, and inspect regularly for pests and diseases.",
        "fertilizer": "Use 10-10-10 NPK at planting and apply micronutrient foliar feed (Fe, B, Zn) during flowering."
    }
}

        if disease in disease_solutions:
            solution = disease_solutions[disease]["prevention"]
            fertilizer = disease_solutions[disease]["fertilizer"]
            
            translated_tip = translate_text(solution, selected_code)
            translated_fert = translate_text(fertilizer, selected_code)


            st.write(f"**🌱 Prevention Tip:** {translated_tip}")
            st.write(f"**💊 Recommended Fertilizer:** {translated_fert}")

        else:
            st.warning("⚠️ No prevention information available for this disease.")
        
        # Optional: Link to more detailed solutions or relevant resources
        st.markdown(f"[Learn more about this disease] https://www.google.com/search?q=" + urllib.parse.quote_plus(disease + " plant disease prevention"))

    else:
        st.warning("⚠️ " + translate_text("Please predict a disease first (Upload or Camera Capture).", selected_code))

elif page == "📺 Farming Tutorials":
    st.title(translate_text("📺 Farming Tutorials & Guides", selected_code))
    st.write(translate_text("Explore these videos for step-by-step help with fertilizer use, disease control, planting, and more.", selected_code))

    # 2 columns layout
    for i in range(0, len(videos), 2):
        cols = st.columns(2)

        for j in range(2):
            if i + j < len(videos):
                video = videos[i + j]
                with cols[j]:
                    st.image(video["thumbnail"], use_container_width=True)
                    st.markdown(f"**{translate_text(video['title'], selected_code)}**")
                    st.markdown(f"[▶ {translate_text('Watch Video', selected_code)}]({video['url']})", unsafe_allow_html=True)

