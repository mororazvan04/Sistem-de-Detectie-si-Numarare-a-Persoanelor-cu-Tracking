import cv2
from ultralytics import YOLO
import numpy as np


def adjust_gamma(image, gamma=1.0):

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


# --- CONFIGURARE ---
print("Se încarcă...")
model = YOLO('yolov8m.pt')

cap = cv2.VideoCapture(0)

clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8, 8))

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)

    # --- ETAPA 1: PRE-PROCESARE DSP COMPLEXĂ ---

    # A. CORECȚIE GAMMA (NOU!)
    # Aplicăm o corecție gamma de 1.2 pentru a "deschide" tonurile pielii
    # înainte de orice altă procesare.
    frame_gamma = adjust_gamma(frame, gamma=1.2)

    # B. Convertim la LAB (Luminanță + Culori)
    lab = cv2.cvtColor(frame_gamma, cv2.COLOR_BGR2LAB)

    # C. Separăm canalele
    l, a, b = cv2.split(lab)

    # D. Aplicăm CLAHE doar pe canalul L (Luminanță)
    # Acum, fiind setat pe 1.1, va scoate detaliile subtil
    l2 = clahe.apply(l)

    # E. Recombinăm canalele
    lab = cv2.merge((l2, a, b))
    frame_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- ETAPA 2: AI DETECȚIE ---
    results = model(frame_processed, classes=[0], conf=0.75, device='mps', verbose=False)

    boxes = results[0].boxes
    numar_persoane = len(boxes)

    # --- ETAPA 3: VIZUALIZARE ---

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame_processed, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Panou informativ
    overlay = frame_processed.copy()
    cv2.rectangle(overlay, (0, 0), (420, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame_processed, 0.4, 0, frame_processed)

    if numar_persoane > 0:
        text_count = f"PERSOANE: {numar_persoane}"
        color_count = (0, 255, 0)
    else:
        text_count = "NIMENI"
        color_count = (0, 0, 255)

    cv2.putText(frame_processed, text_count, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_count, 3)

    # Lista de tehnici folosite
    cv2.putText(frame_processed, "1. DSP: Gamma Correction (Luminare)", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame_processed, "2. DSP: CLAHE (Contrast Adaptiv)", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame_processed, "3. AI: YOLOv8 Medium", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Proiect PS", frame_processed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()