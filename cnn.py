import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Define a more elaborate model architecture function
def create_skin_disease_model():
    input_layer = Input(shape=(224, 224, 3), name="input_1")
    
    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', name="conv1")(input_layer)
    conv1_bn = BatchNormalization(name="conv1_bn")(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name="conv2")(conv1_bn)
    conv2_bn = BatchNormalization(name="conv2_bn")(conv2)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), name="maxpool1")(conv2_bn)
    
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name="conv3")(maxpool1)
    conv3_bn = BatchNormalization(name="conv3_bn")(conv3)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name="conv4")(conv3_bn)
    conv4_bn = BatchNormalization(name="conv4_bn")(conv4)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), name="maxpool2")(conv4_bn)
    
    # Flatten layer
    flatten = Flatten(name="flatten")(maxpool2)
    
    # Fully connected layers- overfitting reduce
    dense1 = Dense(256, activation='relu', name="dense1")(flatten)
    dropout1 = Dropout(0.5, name="dropout1")(dense1)
    output_layer = Dense(7, activation='softmax', name="output")(dropout1)  # 7 classes
    
    # Define model
    model = Model(inputs=input_layer, outputs=output_layer, name="skin_disease_model")
    return model

# Create a more elaborate model
model = create_skin_disease_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define image data generator for data augmentation and loading
datagen = ImageDataGenerator(
    rescale=1./255,  # rescale pixel values to [0,1]
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Load and augment images from the 'img' folder
train_generator = datagen.flow_from_directory(
    'img',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')  # categorical mode for multiple classes

# Define callbacks
#checkpoint = ModelCheckpoint("model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)

# Train the model with callbacks
model.fit(train_generator, epochs=30, callbacks=[checkpoint, early_stopping, reduce_lr])

# Save model architecture to JSON file
# skin_disease_model = model.to_json()
# with open("skin_disease_model.json", "w") as json_file:
#     json_file.write(model.json)
