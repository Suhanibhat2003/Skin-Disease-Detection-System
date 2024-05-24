import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import model_from_json

SKIN_CLASSES = {
    0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular skin lesion'
}

def find_medicine(pred):
    if pred == 0:
        return "Fluorouracil"
    elif pred == 1:
        return "Aldara"
    elif pred == 2:
        return "Prescription Hydrogen Peroxide"
    elif pred == 3:
        return "Corticosteroid gels"
    elif pred == 4:
        return "Fluorouracil (5-FU):"
    elif pred == 5:
        return "Fluorouracil"
    elif pred == 6:
        return "Heparinoid creams"

def detect_skin_condition(image_path):
    j_file = open('model.json', 'r')
    loaded_json_model = j_file.read()
    j_file.close()
    model = model_from_json(loaded_json_model)
    model.load_weights('model.h5')
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    pred = np.argmax(prediction)
    disease = SKIN_CLASSES[pred]
    accuracy = prediction[0][pred] * 100  # Confidence in percentage
    medicine = find_medicine(pred)
    return disease, accuracy, medicine, prediction[0]

def visualize_prediction(prediction):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(prediction)), prediction, color='skyblue')
    plt.xticks(range(len(prediction)), SKIN_CLASSES.values(), rotation=45, ha="right")
    plt.xlabel('Skin Condition')
    plt.ylabel('Prediction Probability')
    plt.title('Probability Distribution of Skin Conditions')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "C:/Users/HP/OneDrive/Desktop/github/med_ai-main/data/1.jpg"  # Update with your image path
    disease, accuracy, medicine, prediction = detect_skin_condition(image_path)
    print("Detected Disease:", disease)
    print("Accuracy:", round(accuracy, 2), "%")
    print("Medicine:", medicine)
    visualize_prediction(prediction)
