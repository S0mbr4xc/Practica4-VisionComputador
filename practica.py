import cv2
from ultralytics import YOLO

# ——— CONFIG ———
VIDEO_IN  = 'test.mp4'  # tu vídeo original
VIDEO_OUT = 'test_yolo.mp4'
MODEL_PT  = 'best.pt'

# ——— Carga el modelo ———
model = YOLO(MODEL_PT)

# ——— Prepara lectura y escritura de vídeo ———
cap = cv2.VideoCapture(VIDEO_IN)
fps    = cap.get(cv2.CAP_PROP_FPS)
w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

# ——— Bucle de inferencia ———
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferencia: devuelve un objeto Results con .boxes
    results = model(frame)[0]

    # Dibuja cada caja con etiqueta y confianza
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        # rectángulo
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # texto arriba de la caja
        cv2.putText(
            frame,
            f'{label} {conf:.2f}',
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    # Muestra en pantalla
    cv2.imshow('YOLOv8 Russian Letters', frame)
    # Guarda en el vídeo de salida
    out.write(frame)

    # Pulsa ESC para salir antes
    if cv2.waitKey(1) == 27:
        break

# ——— Limpia y cierra ———
cap.release()
out.release()
cv2.destroyAllWindows()
print(f'Vídeo anotado guardado en {VIDEO_OUT}')
