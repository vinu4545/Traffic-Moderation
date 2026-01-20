
# import cv2
# from ultralytics import YOLO
# import easyocr
# import re

# # Load trained YOLO model
# model = YOLO("runs/detect/indian_number_plate2/weights/best.pt")

# # OCR reader
# reader = easyocr.Reader(['en'], gpu=False)

# # Read input image
# img_path = "test.jpg"   # change if needed
# img = cv2.imread(img_path)

# if img is None:
#     print("❌ Image not found. Check path.")
#     exit()

# # Run detection
# results = model(img, conf=0.15)

# print("\n===== DETECTION RESULTS =====")
# print(results)

# for r in results:
#     for box in r.boxes:
#         # Bounding box
#         x1, y1, x2, y2 = map(int, box.xyxy[0])

#         # Class label
#         cls_id = int(box.cls[0])
#         label = model.names[cls_id]

#         # Crop number plate region
#         crop = img[y1:y2, x1:x2]

#         # OCR
#         text = reader.readtext(crop, detail=0)
#         raw_plate = " ".join(text)

#         # Clean Indian number plate format
#         plate = re.sub(r'[^A-Z0-9]', '', raw_plate.upper())

#         # ---- PRINT DETAILS ----
#         print(f"Label        : {label}")
#         print(f"Raw OCR Text : {raw_plate}")
#         print(f"Number Plate : {plate}")
#         print("-----------------------------")

#         # Draw bounding box
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         # Draw text on image
#         cv2.putText(
#             img,
#             f"{label}: {plate}",
#             (x1, y1 - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 255, 0),
#             2
#         )

# # Save output image
# output_path = "output.jpg"
# cv2.imwrite(output_path, img)

# print(f"\n✅ Output image saved as: {output_path}")



import cv2
from ultralytics import YOLO
import easyocr
import re

# ---------------------------
# 1. LOAD MODEL & OCR
# ---------------------------
model = YOLO(
    # "runs/detect/indian_number_plate2/weights/last.pt"
    "yolov8n.pt"
)

reader = easyocr.Reader(['en'], gpu=False)


def detect_hsrp(img, output_path):

    if img is None:
        print("❌ Image not found. Check image path.")
        exit()

    # ---------------------------
    # 3. RUN YOLO (LOW CONF)
    # ---------------------------
    results = model(img, conf=0.10)

    print("\n===== DETECTION RESULTS =====")

    detected = False

    # ---------------------------
    # 4. PROCESS DETECTIONS
    # ---------------------------
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        for box in r.boxes:
            detected = True

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Class label
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Crop plate
            crop = img[y1:y2, x1:x2]

            # OCR
            text = reader.readtext(crop, detail=0)
            raw_text = " ".join(text)

            # Clean Indian plate
            plate = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

            # -----------------------
            # PRINT DETAILS
            # -----------------------
            print(f"Label        : {label}")
            print(f"Raw OCR Text : {raw_text}")
            print(f"Number Plate : {plate}")
            print("-----------------------------")

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2),
                        (0, 255, 0), 2)

            # Draw text
            cv2.putText(
                img,
                f"{label}: {plate}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # ---------------------------
    # 5. HANDLE NO DETECTION
    # ---------------------------
    if not detected:
        print("❌ No number plate detected in this image.")

    # ---------------------------
    # 6. SAVE OUTPUT IMAGE
    # ---------------------------
    cv2.imwrite(output_path, img)

    print(f"\n✅ Output image saved as: {output_path}")

# ---------------------------
# 2. READ IMAGE (USE OLD / TRAIN IMAGE)
# ---------------------------
if __name__ == "__main__":
    img_path = "frame1.jpg"
    img = cv2.imread(img_path)
    output_path = "output.jpg"

    detect_hsrp(img, output_path)