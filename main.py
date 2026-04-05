import argparse
import cv2
import pandas as pd
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="video/video2.mp4")
parser.add_argument("--model", default="yolov8n.pt")
args = parser.parse_args()

video_path = args.video
model_path = args.model
conf_threshold = 0.25 #порог уверенности детекции
padding = 80 #расширение зоны стола, чтобы учитывать людей, стоящих рядом с ним
person_class_id = 0 #класс "человек" в модели YOLOv8 имеет id 0
free_confirmation_sec = 4.0 #количество секунд, в течение которых человек должен отсутствовать, чтобы считать стол свободным


def clamp_box(box, width, height): #ограничение координат рамки, чтобы она не выходила за пределы кадра
    x1, y1, x2, y2 = box
    return (
        max(0, min(int(x1), width - 1)),
        max(0, min(int(y1), height - 1)),
        max(0, min(int(x2), width - 1)),
        max(0, min(int(y2), height - 1)),
    )


def intersects(box_a, box_b): #проверка пересечения двух рамок (человека и зоны стола)
    return not (
        box_a[2] < box_b[0]
        or box_a[0] > box_b[2]
        or box_a[3] < box_b[1]
        or box_a[1] > box_b[3]
    )


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"failed to open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_index = 0

ok, first_frame = cap.read()
if not ok:
    raise RuntimeError("failed to read the first video frame")

frame_index += 1
frame_h, frame_w = first_frame.shape[:2]

# выбор области стола на первом кадре
roi = cv2.selectROI("select table", first_frame, showCrosshair=True)
cv2.destroyWindow("select table")

if roi[2] == 0 or roi[3] == 0:
    raise RuntimeError("roi was not selected. draw the table area and press enter or space.")

#расширение зоны стола на заданное количество пикселей
table_box = clamp_box(
    (
        roi[0] - padding,
        roi[1] - padding,
        roi[0] + roi[2] + padding,
        roi[1] + roi[3] + padding,
    ),
    frame_w,
    frame_h,
)

#загрузка модели YOLOv8 для детекции людей
model = YOLO(model_path)

events = []
wait_times = []
confirmed_occupied = None
last_free_time = None
possible_free_start = None

#основной цикл обработки видео
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_index += 1
    current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if current_time_sec <= 0 and fps > 0:
        current_time_sec = frame_index / fps

    occupied = False
    people_count = 0

    #получение детекций от модели для текущего кадра
    results = model(frame, conf=conf_threshold, verbose=False)

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if class_id != person_class_id:
            continue

        coords = box.xyxy[0].cpu().numpy()
        person_box = clamp_box(coords, frame.shape[1], frame.shape[0])
        people_count += 1

        #проверка пересечения рамки человека с зоной стола
        if intersects(person_box, table_box):
            occupied = True

    if occupied:
        possible_free_start = None

    if confirmed_occupied is None:
        confirmed_occupied = occupied
        events.append(
            {
                "event": "start",
                "time_sec": round(current_time_sec, 2),
                "table_state": "occupied" if confirmed_occupied else "free",
                "people_in_frame": people_count,
            }
        )
    elif confirmed_occupied and not occupied:
        if possible_free_start is None:
            possible_free_start = current_time_sec

        if current_time_sec - possible_free_start >= free_confirmation_sec:
            events.append(
                {
                    "event": "became_free",
                    "time_sec": round(possible_free_start, 2),
                    "table_state": "free",
                    "people_in_frame": people_count,
                }
            )
            confirmed_occupied = False
            last_free_time = possible_free_start
            possible_free_start = None
    elif not confirmed_occupied and occupied:
        confirmed_occupied = True
        possible_free_start = None
        events.append(
            {
                "event": "became_occupied",
                "time_sec": round(current_time_sec, 2),
                "table_state": "occupied",
                "people_in_frame": people_count,
            }
        )

        if last_free_time is not None:
            wait_times.append(
                {
                    "free_time_sec": round(last_free_time, 2),
                    "next_occupied_time_sec": round(current_time_sec, 2),
                    "wait_seconds": round(current_time_sec - last_free_time, 2),
                }
            )
            last_free_time = None

    #визуализация результатов на кадре
    table_color = (0, 0, 255) if confirmed_occupied else (0, 255, 0)
    status_text = "occupied" if confirmed_occupied else "free"

    cv2.rectangle(
        frame,
        (table_box[0], table_box[1]),
        (table_box[2], table_box[3]),
        table_color,
        3,
    )
    cv2.putText(
        frame,
        f"table: {status_text}",
        (table_box[0], max(30, table_box[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        table_color,
        2,
    )
    cv2.putText(
        frame,
        f"people in frame: {people_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"time: {current_time_sec:.2f}s",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    if possible_free_start is not None and confirmed_occupied:
        cv2.putText(
            frame,
            f"no detection: {current_time_sec - possible_free_start:.1f}s / {free_confirmation_sec:.1f}s",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    cv2.imshow("table status", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

#вывод результатов в виде таблиц
events_df = pd.DataFrame(events)
wait_times_df = pd.DataFrame(wait_times)

print("\nсобытия по столику:")
if events_df.empty:
    print("события не записаны")
else:
    print(events_df.to_string(index=False))

print("\nинтервалы после освобождения столика:")
if wait_times_df.empty:
    print("не найдено корректных пар 'became_free -> became_occupied'")
else:
    print(wait_times_df.to_string(index=False))
    mean_wait = wait_times_df["wait_seconds"].mean()
    print(f"\nсреднее время между уходом гостя и подходом следующего человека: {mean_wait:.2f} сек")
