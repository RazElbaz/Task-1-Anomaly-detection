# Task-1-Anomaly-detection 
In the Docker app with UI folder (Part 5 in the assignment, found in PDF in the EX1 folder) there is a Deploying a Simple Streamlit app which shows a prediction for a new point that will enter the prediction model based on the IsolationForest algorithm.

To run the app: streamlit run app.py
### view my Streamlit app in your browser press on the Local URL:
![run](https://github.com/RazElbaz/Task-1-Anomaly-detection/blob/main/images/run.png)

### The app:

![app](https://github.com/RazElbaz/Task-1-Anomaly-detection/blob/main/images/app.png)

### Depending on the insertion of points we will get a prediction of an anomaly or no anomaly:

![anomaly](https://github.com/RazElbaz/Task-1-Anomaly-detection/blob/main/images/anomaly.png)

![not anomaly](https://github.com/RazElbaz/Task-1-Anomaly-detection/blob/main/images/not%20anomaly.png)

### To run with docker:
1) Building a Docker image:
docker build -t streamlitapp:latest .
2) Creating a container:
docker run -p 8501:8501 streamlitapp:latest
