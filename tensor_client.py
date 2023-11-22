import requests


SERVER_URL = 'https://linear-model-service-nearr27.cloud.okteto.net/v1/models/linear-model:predict'

def main():
    
    predict_request = '{"instances" : [ [20], [40], [60], [80], [100] ]}'
    
    
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()  
    prediction = response.json()
    print("Predicciones de precio de venta para diferentes costos de producción:", prediction)

if __name__ == '__main__':
    main()
