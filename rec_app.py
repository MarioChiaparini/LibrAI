import streamlit as st
import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import funcs as fnc
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('my_model.hdf5')
  return model
with st.spinner('O modelo esta sendo carregado..'):
  model=load_model()

st.write("""
         # Linguagem de sinais brasileira 
         """
         )

file = st.file_uploader("Coloque a imagem aqui:", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

if file is None:
    st.text("Please upload an image file")

else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = fnc.import_predict(image, model)
    class_names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 
    'L', 'M', 'N', 'O', 'P', 'Q', 
    'R', 'S', 'T', 'U', 'V', 'W', 'Y']

    print_str="O sinal classificada é: "+class_names[np.argmax(predictions)]
    st.success(print_str)
