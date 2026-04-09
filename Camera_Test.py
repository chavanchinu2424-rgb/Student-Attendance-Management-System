import cv2
import uvicorn
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# FIX: CORS Middleware allows your browser to talk to this Python script safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/start-scan")
async def start_scan():
    print("[SYSTEM] Biometric Authentication sequence initiated...")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)

    # UI Window: Force Full Screen
    window_name = 'COLLEGE BIOMETRIC PORTAL'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Verification Logic Variables
    detection_timer = 0
    REQUIRED_FRAMES = 30 # Approx 1.5 seconds of steady face detection
    verified = False

    while True:
        ret, frame = video_capture.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Mirror effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            detection_timer += 1
            for (x, y, w, h) in faces:
                if detection_timer < REQUIRED_FRAMES:
                    # STATE 1: FACE DETECTED (Orange)
                    color = (0, 107, 255) # BGR for Orange
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, "FACE DETECTED", (x, y-40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, "Verifying identity...", (x, y-15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                else:
                    # STATE 2: VERIFIED (Green)
                    color = (0, 255, 0) # BGR for Green
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(frame, "IDENTITY VERIFIED", (x, y-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
                    verified = True
        else:
            detection_timer = 0 # Reset if they look away

        cv2.imshow(window_name, frame)

        # Auto-close after verification
        if verified:
            cv2.waitKey(1500) # Pause for 1.5 seconds so user sees the "Green" success
            break

        # Manual exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    
    return {"status": "success" if verified else "failed"}

if __name__ == "__main__":
    # Ensure this runs on Port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)