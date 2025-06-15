import cv2
import time
import winsound  # Som no Windows
from inference_sdk import InferenceHTTPClient

# Inicializa cliente da Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="x18RThrdUZ2ElVFJvHMo"
)

# Modelo da Roboflow
MODEL_ID = "knife_hand_holding-knife-dataset-mwf2d/1"

# Captura da webcam
cap = cv2.VideoCapture(0)

# Temporizador
last_sent_time = 0
delay_seconds = 1
predictions = []

# Função para emitir um bipe
def emitir_alarme():
    duration = 500  # milissegundos
    freq = 1000     # Hz
    winsound.Beep(freq, duration)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensiona
    resized_frame = cv2.resize(frame, (640, 640))

    # A cada 2 segundos, envia imagem
    current_time = time.time()
    if current_time - last_sent_time > delay_seconds:
        cv2.imwrite("temp.jpg", resized_frame)
        result = CLIENT.infer("temp.jpg", model_id=MODEL_ID)
        predictions = result.get("predictions", [])
        last_sent_time = current_time

        # Verifica se há uma faca entre as predições
        for pred in predictions:
            if pred["class"].lower() == "faca" or "knife" in pred["class"].lower():
                emitir_alarme()
                break  # Evita toques múltiplos

    # Desenha as caixas
    for pred in predictions:
        x = int(pred["x"])
        y = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        label = f'{pred["class"]} ({pred["confidence"]:.2f})'

        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(resized_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Exibe a imagem com detecção
    cv2.imshow("Detecção Roboflow", resized_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
