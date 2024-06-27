# Tomato Leaf Prediction API

```
git clone https://github.com/Nocturnailed-Community/Tomato-Leaf-Prediction-API.git
```

## Install Library dependencies

```
flask
flask_cors
gunicorn
Werkzeug
numpy
tensorflow
pillow
```

```
pip install -r requirements.txt
```

## Deployment

Instructions for running the model in docker

```
docker build -t tomato-leaf-prediction .
docker run -d -p 5000:5000 tomato-leaf-prediction
```

The instruction looks at running containers

```
docker ps
```

Push Docker Hub

```
docker build . -t your_username_docker/tomato-leaf-prediction
docker login
docker push your_username_docker/tomato-leaf-prediction
```