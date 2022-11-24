import streamlit as st
import joblib
# from sklearn.externals import joblib
import time
from PIL import Image

import predict

gender_nv_model = open("models/naivemodel.pkl","rb")
gender_clf = joblib.load(gender_nv_model)

def predict_gender(a,b,c):
  result = predict.predict(a,b,c)
  return result

# def load_css(file_name):
#     with open(file_name) as f:
#         st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
#
# def load_icon(icon_name):
#     st.markdown('<i class="material-icons">{}</i>'.format(icon_name), unsafe_allow_html=True)

def load_images(file_name):
  img = Image.open(file_name)
  return st.image(img,width=300)

def main():
  """Gender Classifier App
    With Streamlit

  """

  st.title("Anomaly Detection")
  html_temp = """
  <div style="background-color:blue;padding:10px">
  <h2 style="color:grey;text-align:center;">Streamlit App </h2>
  </div>

  """
  st.markdown(html_temp,unsafe_allow_html=True)
  # load_css('icon.css')
  # load_icon('people')

  dor = st.text_input("Enter duration_","Pleas Type Here")
  src = st.text_input("Enter src_bytes", "Pleas Type Here")
  des = st.text_input("Enter dst_bytes", "Pleas Type Here")
  if st.button("Predict"):
    result = predict_gender(dor,src,des)
    if result == 0:
      prediction = "Not an anomalous point"
      img = 'good.png'
    else:
      result == 1
      prediction = "Anomalous point"
      img = 'bad.png'

    st.success('duration_: {} , src_bytes: {} , dst_bytes: {} was classified as {}'.format(dor.title(),src.title(),des.title(),prediction))
    load_images(img)

main()