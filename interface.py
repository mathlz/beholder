import tkinter as tk
from tkinter import filedialog
import cv2
import time
import threading
import winsound
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageTk

# --- Configuração do cliente Roboflow ---
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="x18RThrdUZ2ElVFJvHMo"
)
MODEL_ID = "knife_hand_holding-knife-dataset-mwf2d/1"


# --- Alarme sonoro ---
def emitir_alarme():
    duration = 500  # milissegundos
    freq = 1000     # Hz
    winsound.Beep(freq, duration)


# --- Detectar em imagem ---
def detectar_em_imagem():
    file_path = filedialog.askopenfilename()
    if file_path:
        result = CLIENT.infer(file_path, model_id=MODEL_ID)
        predictions = result.get("predictions", [])

        img = cv2.imread(file_path)
        img = cv2.resize(img, (640, 640))

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
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if "knife" in pred["class"].lower() or "faca" in pred["class"].lower():
                emitir_alarme()

        cv2.imshow("Resultado da imagem", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# --- Webcam ao vivo ---
def rodar_webcam():
    cap = cv2.VideoCapture(0)
    last_sent_time = 0
    delay_seconds = 2
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (640, 640))
        current_time = time.time()

        if current_time - last_sent_time > delay_seconds:
            cv2.imwrite("temp.jpg", resized_frame)
            result = CLIENT.infer("temp.jpg", model_id=MODEL_ID)
            predictions = result.get("predictions", [])
            last_sent_time = current_time

            for pred in predictions:
                if pred["class"].lower() == "faca" or "knife" in pred["class"].lower():
                    emitir_alarme()
                    break

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

        cv2.imshow("Detecção ao vivo", resized_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


# --- Função para rodar webcam em thread separada ---
def iniciar_thread_webcam():
    t = threading.Thread(target=rodar_webcam)
    t.daemon = True
    t.start()


# --- Interface Gráfica (Tkinter) ---
root = tk.Tk()
root.title("Deteccao de Facas com Roboflow")
root.geometry("400x250")

titulo = tk.Label(root, text="Sistema de Detecção", font=("Arial", 16))
titulo.pack(pady=20)

btn_webcam = tk.Button(root, text="Rodar Detecção com Webcam", command=iniciar_thread_webcam, width=30, height=2)
btn_webcam.pack(pady=5)

btn_imagem = tk.Button(root, text="Analisar Imagem do Computador", command=detectar_em_imagem, width=30, height=2)
btn_imagem.pack(pady=5)

btn_alarme = tk.Button(root, text="Tocar Alarme Manualmente", command=emitir_alarme, width=30, height=2)
btn_alarme.pack(pady=5)

root.mainloop()
