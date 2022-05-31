import requests
from models import YOLONetwork
from models import CRNNetwork
import utils
from PIL import Image
from io import BytesIO
import torch
import streamlit as st

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
yolo_input_img_size = (418, 418)
crnn_input_img_size = (32, 128)
anchors = torch.Tensor([[37, 27], [134, 89], [286, 116], [85, 53]])
alphanum2ind, ind2alphanum = utils.get_encodings()

@st.cache
def get_models():
    yolo_model = YOLONetwork().to(device)
    chkpt = torch.load('yolo_chkpt', map_location=torch.device(device))
    yolo_model.load_state_dict(chkpt['model_state'])

    crnn_model = CRNNetwork(alphanum2ind).to(device)
    chkpt = torch.load('crnn_chkpt', map_location=torch.device(device))
    crnn_model.load_state_dict(chkpt['model_state'])

    return yolo_model, crnn_model

st.title('Text Detection and Recognition')
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png'])
if uploaded_file is not None:
    img = Image.open(BytesIO(uploaded_file.getvalue()))
    img = img.resize(yolo_input_img_size)
    st.image(img, caption='Uploaded image')

    yolo_model, crnn_model = get_models()
    boxes = utils.get_bounding_boxes(yolo_model, img, yolo_input_img_size, anchors, device)
    
    if boxes == None or len(boxes) == 0:
        st.warning('No text detected')
        st.stop()
    else:
        st.success('Text found')
        st.subheader('Image with bounding box around text')
        utils.draw_bounding_box(boxes, img, yolo_input_img_size)
        cropped_images = utils.get_cropped_images(img, boxes, yolo_input_img_size)
        predicted_texts = utils.get_predicted_texts(cropped_images, crnn_input_img_size, crnn_model, ind2alphanum)
        st.subheader('Predicted text from the cropped image')
        columns = st.columns(len(cropped_images))
        for i, col in enumerate(columns):
            col.image(cropped_images[i], caption=predicted_texts[i])
