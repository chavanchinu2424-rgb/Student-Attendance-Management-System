from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # This lets our HTML talk to Python securely
import cv2

app = FastAPI()

# SECURITY SETTING: Allow our local HTML files to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# THE NEW ENDPOINT: When the dashboard calls this, the camera opens!
@app.get("/api/start-scan")
def start_scan():
    print("Dashboard requested a face scan. Turning on camera...")
    
    # Load AI model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)

    # Force Full Screen
    window_name = 'Biometric Scanner - Press Q to Exit'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1) # Mirror effect
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Scanning...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)

        # Wait for the user to press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Turn off camera
    video_capture.release()
    cv2.destroyAllWindows()

    # Tell the Dashboard that we finished successfully!
    return {"status": "success", "message": "Camera closed successfully."}