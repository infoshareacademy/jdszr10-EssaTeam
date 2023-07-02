import streamlit as st
import tensorflow as tf
from PIL import Image

# Load the trained CNN model
model = tf.keras.models.load_model('C:/Users/miro/Desktop/wercia/Bootcamp/DL project/saved_models/model5.hdf5')

# Define the classes of Lego brick parts
classes = ['2357', '2412b', '2420', '2429','2430']

#Function to preprocess the image
def preprocess_image(image):
    # Resize the image to the required input shape of the CNN model
    image = image.resize((64, 64))
    # Convert the image to a NumPy array and normalize the pixel values
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    # Expand the dimensions of the image to match the CNN model's input shape
    image = tf.expand_dims(image, axis=0)
    return image

# Streamlit app
def main():
    st.title("Check your brick!")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image of a Lego brick", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Perform inference using the CNN model
        predictions = model.predict(processed_image)
        predicted_class = classes[predictions.argmax()]

        # Display the predicted Lego brick part
        st.subheader("Your brick is: ")
        st.write("The Lego brick part is:", predicted_class)

# Run the app
if __name__ == '__main__':
    main()


