import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

idx2label = {
    0: 'Cassava Bacterial Blight (CBB)',
    1: 'Cassava Brown Streak Disease (CBSD)',
    2: 'Cassava Green Mottle (CGM)',
    3: 'Cassava Mosaic Disease (CMD)',
    4: 'Healthy'
}

class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding='same')
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 35 * 35, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, X):
        X = F.relu(self.avgpool1(self.conv1(X)))
        X = F.relu(self.avgpool2(self.conv2(X)))
        X = self.flatten(X)
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        return X

@st.cache_resource
def load_model(model_path, num_classes=5):
    model = LeNetClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model('apps/cassava_app/checkpoints/lenet_model_cassava.pt')

def inference(image, model):
    img_size = 150
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    img_new = transform(image)
    img_new = torch.unsqueeze(img_new, 0)
    with torch.no_grad():
        output = model(img_new)
    output = nn.Softmax(dim=1)(output)
    p_max, y_hat = torch.max(output.data, 1)
    return p_max.item() * 100, y_hat.item()

def main():
    st.title('Cassava Leaf Disease Classification')
    st.subheader('Model: LeNet. Dataset: Cassava Leaf Disease')
    option = st.selectbox('Choose an option:', ['Upload Image', 'Use Example Image'])
    if option == 'Upload Image':
        uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'png', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            p, y = inference(image, model)
            st.image(image, caption='Uploaded Image')
            st.success(f'Prediction: {idx2label[y]}. Confidence: {p:.2f}%')
    
    elif option == 'Use Example Image':
        image_path = 'data/Cassava/samples/sample_cbsd.jpg'
        image = Image.open(image_path)
        p, y = inference(image, model)
        st.image(image, caption='Example Image')
        st.success(f'Prediction: {idx2label[y]}. Confidence: {p:.2f}%')
    
if __name__ == '__main__':
    main()