import requests

url = 'http://localhost:5010/predict_api'
r = requests.post(url,json={'Name': "Chayan"})

print(r.json())