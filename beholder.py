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

alarme_ativo = False  # Variável para controlar o alarme

# Tradução dos nomes das classes
CLASSES_TRADUZIDAS = {
    "knife": "Objeto Perigoso",
    "slap": "Tapa",
    "person": "Pessoa",
    "weapon holding": "Empunhadura",
    "handgun": "Arma",
    "slap": "Tapa",
    "punch": "Soco",
    "violence": "Violência",
}

# Inicializa o mixer do pygame
pygame.mixer.init()

def tocar_sirene_mp3():
    global alarme_ativo
    pygame.mixer.music.load("sirene.mp3")
    pygame.mixer.music.play(-1)  # loop infinito
    while alarme_ativo:
        pygame.time.wait(100)

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

# Alarme sonoro simples
def emitir_alarme():
    global ultimo_alarme_time
    current_time = time.time()
    if current_time - ultimo_alarme_time >= 1:
        ultimo_alarme_time = current_time
        def beep():
            winsound.Beep(1000, 500)
        threading.Thread(target=beep, daemon=True).start()

# Variável de controle para parar a webcam
executando = False

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

                # Traduz classe
                classe = CLASSES_TRADUZIDAS.get(class_name.lower(), class_name)
                label = f"{classe} ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # Alarme para classes perigosas
                if class_name.lower() in ["knife", "slap", "handgun", "punch", "violence"]:
                    emitir_alarme()

            # Atualiza imagem no label
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            video_label.config(image=img_tk)
            video_label.image = img_tk

        root.after(30, atualizar_frame)

    atualizar_frame()


def parar_webcam():
    global executando
    executando = False
    video_label.config(image='')

# --- Layout Tkinter ---
root = tk.Tk()
root.title("Beholder - Detecção de Objetos Perigosos")
root.geometry("1000x600")  # mais largo para o vídeo

# Frame para botões
frame_botoes = tk.Frame(root, width=200, bg='#f0f0f0')
frame_botoes.pack(side=tk.LEFT, fill=tk.Y)
frame_botoes.pack_propagate(False)

# Frame direito para vídeo
frame_video = tk.Frame(root, bg='black')
frame_video.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Título no topo do frame de botões
titulo = tk.Label(frame_botoes, text="Controles", font=("Arial", 16), bg='#f0f0f0')
titulo.pack(pady=20)

# Botões empilhados no frame
btn_webcam = tk.Button(
    frame_botoes, text="Abrir Câmera", 
    command=lambda: threading.Thread(target=rodar_webcam, daemon=True).start(),
    width=20, height=2
)
btn_webcam.pack(pady=10)

btn_parar = tk.Button(
    frame_botoes, text="Fechar Câmera", 
    command=parar_webcam,
    width=20, height=2
)
btn_parar.pack(pady=10)

btn_alarme = tk.Button(
    frame_botoes, text="Tocar Sirene", 
    command=iniciar_alarme,
    width=20, height=2
)
btn_alarme.pack(pady=10)

btn_parar_sirene = tk.Button(
    frame_botoes, text="Parar Sirene", 
    command=parar_alarme,
    width=20, height=2
)
btn_parar_sirene.pack(pady=10)

# Label do vídeo
video_label = tk.Label(frame_video)
video_label.pack(fill=tk.BOTH, expand=True)

root.mainloop()
