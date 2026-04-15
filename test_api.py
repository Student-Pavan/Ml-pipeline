import requests

def test():
    url = "http://127.0.0.1:8000/predict?a=5.1&b=3.5&c=1.4&d=0.2"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        print("API Test Passed:", response.json())
    else:
        raise Exception("API Test Failed")

test()