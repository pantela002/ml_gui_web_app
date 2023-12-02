import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from streamlit_drawable_canvas import st_canvas
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 6 input image channel, 16 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # an affine operation: y = Wx + b
        self.fc2 = nn.Linear(120, 84)
        # an affine operation: y = Wx + b
        self.fc3 = nn.Linear(84, 10)

    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # Flatten the tensor
        x = x.view(-1, self.num_flat_features(x))
        # Apply relu function
        x = F.relu(self.fc1(x))
        # Apply relu function
        x = F.relu(self.fc2(x))
        # Apply softmax function
        x = self.fc3(x)
        return x


model_code = """
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Define your model architecture
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load your trained weights
model.load_weights('path_to_your_trained_weights.h5')
"""



def predict(image_path):
    model = ConvNet()

    model.load_state_dict(torch.load("./cnn.pth"))
    model.eval() 

    transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
    ])

    input_image = Image.open(image_path)
    input_tensor = transform(input_image).unsqueeze(0) 

     

    print(input_tensor.shape)

    # Make a prediction
    with torch.no_grad():
        output = model(input_tensor)

    return output
    # Post-process the output as needed (e.g., applying softmax)
    probs = F.softmax(output, dim=1)

    # Print or use the predicted probabilities
    print(probs)

    return probs





    




def home_page():
    st.write("Welcome to the Home page!")
    img_path = st.file_uploader("Select an image from your PC", key="main_image", type=["png", "jpg"])

    if img_path is not None:
        st.image(img_path, caption="Selected Image (Main)", use_column_width=True)

def resize_and_save_image(drawn_image_np, save_path, target_size=(28, 28)):
    # Convert NumPy array to PIL Image
    drawn_image_pil = Image.fromarray(drawn_image_np)

    # Convert the image to grayscale
    drawn_image_gray = drawn_image_pil.convert("L")

    # Resize the grayscale image to the target size
    resized_image = drawn_image_gray.resize(target_size, Image.ANTIALIAS)

    # Save the resized image
    resized_image.save(save_path)

    
def drawing_page():
    st.title("Draw a number")

    canvas_result = st_canvas(
        fill_color="black",  # Fixed fill color with some opacity
        stroke_width=40,
        stroke_color="white",
        background_color="black",
        update_streamlit=True,
        height=300,
        width=300,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="full_app",
    )

    drawn_image_np = None

    if canvas_result.image_data is not None:    
        #st.image(canvas_result.image_data)

        # Get the state of the user's drawing as a NumPy array
        drawn_image_np = np.array(canvas_result.image_data)

        drawn_image_bw = Image.fromarray(drawn_image_np).convert("L")
        drawn_image_np = np.array(drawn_image_bw)
        print(drawn_image_np.shape)

        resize_and_save_image(drawn_image_np,"drawn_image.jpg", target_size=(28, 28))

        prediction = predict("drawn_image.jpg")

        #st.text(f"Prediction: {prediction}")

         # Display probability bar chart
        st.title("Probability Distribution")
        probabilities = np.zeros(10) if prediction is None else F.softmax(prediction, dim=1).numpy().flatten()
        df_probabilities = pd.DataFrame({
            'Number': range(10),
            'Probability': probabilities
        })

        # Create a bar chart
        st.bar_chart(df_probabilities.set_index('Number'))


    print(drawn_image_np)

    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        for col in objects.select_dtypes(include=["object"]).columns:
            objects[col] = objects[col].astype("str")
        st.dataframe(objects)


    # Display the machine learning model code

    st.title("Implementation")

    st.code(model_code, language='python')


def other_page():
    st.write("Welcome to the Other Page!")
    content_placeholder = st.empty()
    content_placeholder.file_uploader("Select an image from your PC (Other Page)", key="other_page_image", type=["png", "jpg"])

def main():

    selected_page = st.sidebar.selectbox("Navigation", ["Home", "Drawing", "Other Page"])

    if selected_page == "Home":
        home_page()
    elif selected_page == "Drawing":
        drawing_page()
    elif selected_page == "Other Page":
        other_page()

if __name__ == "__main__":
    main()
