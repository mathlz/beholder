import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import torch
import threading
import winsound
from ultralytics import YOLO
import time
import pygame

alarme_ativo = False  # Flag global para controle

# Mapeamento de nomes das classes para rótulos personalizados
CLASSES_PERSONALIZADAS = {
    "knife": "Objeto Perigoso",
    "slap": "Tapa",
    "person": "Pessoa",
    "weapon holding": "Empunhadura"
}

# Inicializa o mixer do pygame
pygame.mixer.init()

def tocar_sirene_mp3():
    global alarme_ativo
    pygame.mixer.music.load("sirene.mp3")
    pygame.mixer.music.play(-1)  # -1 = loop infinito
    while alarme_ativo:
        pygame.time.wait(100)  # Pequena pausa para não travar o loop

def iniciar_alarme():
    global alarme_ativo
    if not alarme_ativo:
        alarme_ativo = True
        threading.Thread(target=tocar_sirene_mp3, daemon=True).start()

def parar_alarme():
    global alarme_ativo
    alarme_ativo = False
    pygame.mixer.music.stop()

# Variável para evitar spam de alarme
ultimo_alarme_time = 0

# Carrega o modelo YOLOv8 treinado
model = YOLO("best.pt")

# Alarme sonoro
def emitir_alarme():
    global ultimo_alarme_time
    current_time = time.time()

    # Só emite se passou pelo menos 1 segundo desde o último alarme
    if current_time - ultimo_alarme_time >= 1:
        ultimo_alarme_time = current_time

        def beep():
            duration = 500  # milissegundos
            freq = 1000     # Hz
            winsound.Beep(freq, duration)

        threading.Thread(target=beep, daemon=True).start()

# Variável de controle para parar a webcam
executando = False

# Loop da webcam em thread
def rodar_webcam():
    global executando
    executando = True
    cap = cv2.VideoCapture(0)

    def atualizar_frame():
        if not executando:
            cap.release()
            return

        ret, frame = cap.read()
        if ret:
            results = model(frame)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                classe_traduzida = CLASSES_PERSONALIZADAS.get(class_name.lower(), class_name)
                label = f"{classe_traduzida} ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if "faca" in class_name.lower() or "knife" in class_name.lower():
                    emitir_alarme()

            # Converte imagem para exibir no Tkinter
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            video_label.config(image=img_tk)
            video_label.image = img_tk

        root.after(30, atualizar_frame)

    atualizar_frame()

# Parar a webcam
def parar_webcam():
    global executando
    executando = False
    video_label.config(image='')

# Interface Gráfica
root = tk.Tk()
root.title("Detecção de Objetos Perigosos")
root.geometry("700x600")

titulo = tk.Label(root, text="Detecção de Objetos Perigosos", font=("Arial", 18))
titulo.pack(pady=10)

btn_webcam = tk.Button(root, text="Abrir Câmera de Segurança", command=lambda: threading.Thread(target=rodar_webcam).start(), width=35, height=2)
btn_webcam.pack(pady=5)

btn_parar = tk.Button(root, text="Fechar Câmera de Segurança", command=parar_webcam, width=35, height=2)
btn_parar.pack(pady=5)

btn_alarme = tk.Button(root, text="Tocar Sirene", command=iniciar_alarme, width=30, height=2)
btn_alarme.pack(pady=5)

btn_parar = tk.Button(root, text="Parar Sirene", command=parar_alarme, width=30, height=2)
btn_parar.pack(pady=5)

video_label = tk.Label(root)
video_label.pack(pady=10)

root.mainloop()
