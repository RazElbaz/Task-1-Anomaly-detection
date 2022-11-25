

FROM python:3.9.13




WORKDIR /app
RUN apt-get update && apt-get install -y apache2 && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install joblib
RUN pip install sklearn
RUN pip install streamlit
RUN pip install Image
RUN pip install -U scikit-learn
ADD . /app
EXPOSE 8501
COPY . /app

CMD streamlit run app.py