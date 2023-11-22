import requests

# URL del servidor que aloja el modelo de predicción del precio de venta
SERVER_URL = 'http://172.17.0.3:8501/v1/models/linear-model:predict'

def main():
    # Datos de entrada simulando diferentes costos de producción
    predict_request = '{"instances" : [ [20], [40], [60], [80], [100] ]}'
    
    # Enviar la solicitud y obtener la predicción
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()  # Verificar si la respuesta contiene un error
    prediction = response.json()
    print("Predicciones de precio de venta para diferentes costos de producción:", prediction)

if __name__ == '__main__':
    main()