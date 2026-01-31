import wx
import cv2
import time
import re
import nltk
import speech_recognition as sr
import csv
from datetime import datetime
from ultralytics import YOLO

# ---------------- YOLO MODEL ---------------- #
model = YOLO("yolov8n.pt")

# ---------------- DRIVER MONITORING ---------------- #
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

driver_state = "normal"
closed_start_time = None

def detect_driver_state(frame):
    global closed_start_time, driver_state
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes_detected = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) > 0:
            eyes_detected = True

    if not eyes_detected:
        if closed_start_time is None:
            closed_start_time = time.time()
        elif time.time() - closed_start_time > 2.0:
            driver_state = "drowsy"
    else:
        closed_start_time = None
        driver_state = "normal"

    return driver_state

# ---------------- ADAS LOGIC ---------------- #
def evaluate_adas(speed_kmh, distance_m, obstacle, road, driver_state):
    speed_ms = speed_kmh / 3.6
    ttc = distance_m / speed_ms if speed_ms > 0 else 999

    road_factor = {"Dry": 1.0, "Wet": 1.5, "Poor": 2.0}.get(road, 1.0)
    risk = (speed_ms * road_factor) / ttc

    if obstacle == "Animal":
        risk *= 1.3
    elif obstacle == "Pothole":
        risk *= 0.8

    state_map = {"normal": 0, "drowsy": 7, "distracted": 4}
    risk += state_map.get(driver_state, 0)

    if risk > 25: decision = "EMERGENCY BRAKE"
    elif risk > 15: decision = "GRADUAL BRAKE + ALERT"
    elif risk > 8: decision = "PRE-BRAKE + WARNING"
    else: decision = "MONITOR"

    return round(ttc, 2), round(risk, 2), decision

# ---------------- CSV LOGGING ---------------- #
def log_scenario(speed, distance, obstacle, road, driver_state, ttc, risk, decision):
    with open("adas_log.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            speed, distance, obstacle, road, driver_state, ttc, risk, decision
        ])

# ---------------- VOICE ---------------- #
def listen_once():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print("Recognized:", text)
        return text.lower()
    except:
        return ""

# ---------------- GUI ---------------- #
class ADASApp(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Unified ADAS System", size=(600, 700))
        panel = wx.Panel(self)
        panel.SetBackgroundColour("Beige")

        title = wx.StaticText(panel, label="ADAS Risk Assessment Tool")
        title.SetFont(wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))

        console_label = wx.StaticText(panel, label="Smart Command Console")
        self.command_input = wx.TextCtrl(panel, style=wx.TE_PROCESS_ENTER)
        self.command_input.SetHint("Ex: 'Set speed at 80 and distance 20'")

        speed_label = wx.StaticText(panel, label="Vehicle Speed (km/h)")
        self.speed_input = wx.TextCtrl(panel)

        distance_label = wx.StaticText(panel, label="Distance to Obstacle (m)")
        self.distance_input = wx.TextCtrl(panel)

        obstacle_label = wx.StaticText(panel, label="Obstacle Type")
        self.obstacle_choice = wx.Choice(panel, choices=["Vehicle", "Animal", "Pothole"])
        self.obstacle_choice.SetSelection(0)

        road_label = wx.StaticText(panel, label="Road Condition")
        self.road_choice = wx.Choice(panel, choices=["Dry", "Wet", "Poor"])
        self.road_choice.SetSelection(0)

        evaluate_btn = wx.Button(panel, label="Evaluate ADAS")
        voice_btn = wx.Button(panel, label="Voice Command")

        self.ttc_output = wx.StaticText(panel, label="TTC: -- seconds")
        self.risk_output = wx.StaticText(panel, label="Risk Score: --")
        self.decision_output = wx.StaticText(panel, label="Decision: --")
        self.decision_output.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)
        sizer.Add(console_label, 0, wx.LEFT, 20)
        sizer.Add(self.command_input, 0, wx.EXPAND | wx.ALL, 10)
        sizer.Add(wx.StaticLine(panel), 0, wx.EXPAND | wx.ALL, 5)

        sizer.Add(speed_label, 0, wx.LEFT, 20)
        sizer.Add(self.speed_input, 0, wx.EXPAND | wx.ALL, 10)
        sizer.Add(distance_label, 0, wx.LEFT, 20)
        sizer.Add(self.distance_input, 0, wx.EXPAND | wx.ALL, 10)
        sizer.Add(obstacle_label, 0, wx.LEFT, 20)
        sizer.Add(self.obstacle_choice, 0, wx.EXPAND | wx.ALL, 10)
        sizer.Add(road_label, 0, wx.LEFT, 20)
        sizer.Add(self.road_choice, 0, wx.EXPAND | wx.ALL, 10)

        sizer.Add(evaluate_btn, 0, wx.ALL | wx.CENTER, 15)
        sizer.Add(self.ttc_output, 0, wx.ALL | wx.CENTER, 5)
        sizer.Add(self.risk_output, 0, wx.ALL | wx.CENTER, 5)
        sizer.Add(self.decision_output, 0, wx.ALL | wx.CENTER, 10)
        sizer.Add(voice_btn, 0, wx.ALL | wx.CENTER, 5)

        panel.SetSizer(sizer)

        self.command_input.Bind(wx.EVT_TEXT_ENTER, self.on_voice_command)
        evaluate_btn.Bind(wx.EVT_BUTTON, self.on_evaluate)
        voice_btn.Bind(wx.EVT_BUTTON, self.on_voice_input)

        self.Show()

    def on_voice_command(self, event):
        text = self.command_input.GetValue().lower()
        tokens = nltk.word_tokenize(text)
        numbers = re.findall(r'\d+', text)

        if "speed" in tokens and len(numbers) > 0:
            self.speed_input.SetValue(numbers[0])
        if "distance" in tokens and len(numbers) > 1:
            self.distance_input.SetValue(numbers[1])

        self.on_evaluate(None)

    def on_voice_input(self, event):
        text = listen_once()
        if text:
            self.command_input.SetValue(text)
            self.on_voice_command(None)

    def on_evaluate(self, event):
        try:
            sp = float(self.speed_input.GetValue())
            di = float(self.distance_input.GetValue())
            obstacle = self.obstacle_choice.GetStringSelection()
            road = self.road_choice.GetStringSelection()

            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()

            if ret:
                driver_state = detect_driver_state(frame)
                model(frame, show=True)

                ttc, risk, decision = evaluate_adas(sp, di, obstacle, road, driver_state)
                self.ttc_output.SetLabel(f"TTC: {ttc} seconds")
                self.risk_output.SetLabel(f"Risk Score: {risk}")
                self.decision_output.SetLabel(f"Decision: {decision}")

                log_scenario(sp, di, obstacle, road, driver_state, ttc, risk, decision)
        except ValueError:
            wx.MessageBox("Please enter valid numbers", "Input Error")

# ---------------- RUN APP ---------------- #
app = wx.App()
ADASApp()
app.MainLoop()



import cv2
import time
import winsound  # For beep alert (Windows only)

# Haar cascades for face & eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)
closed_start_time = None
driver_state = "normal"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes_detected = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) > 0:
            eyes_detected = True

    # Drowsiness logic
    if not eyes_detected:
        if closed_start_time is None:
            closed_start_time = time.time()
        elif time.time() - closed_start_time > 2.0:  # Eyes closed > 2 sec
            driver_state = "drowsy"
            # 🔔 Wake-up alert
            cv2.putText(frame, "WAKE UP! DROWSY DRIVER!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            winsound.Beep(1000, 500)  # Beep sound
    else:
        closed_start_time = None
        driver_state = "normal"

    # Display driver state
    cv2.putText(frame, f"Driver State: {driver_state}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if driver_state == "normal" else (0, 0, 255), 2)

    cv2.imshow("Driver Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2 
import time
import winsound  # For beep alert (Windows only)

# Haar cascades for face & eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)
closed_start_time = None
driver_state = "normal"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes_detected = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) > 0:
            eyes_detected = True

    # Drowsiness logic
    if not eyes_detected:
        if closed_start_time is None:
            closed_start_time = time.time()
        elif time.time() - closed_start_time > 2.0:  # Eyes closed > 2 sec
            driver_state = "drowsy"
            # 🔔 Wake-up alert
            cv2.putText(frame, "WAKE UP! DROWSY DRIVER!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            winsound.Beep(1000, 500)  # Beep sound
    else:
        closed_start_time = None
        driver_state = "normal"

    # Display driver state
    cv2.putText(frame, f"Driver State: {driver_state}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if driver_state == "normal" else (0, 0, 255), 2)

    cv2.imshow("Driver Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# ---------------- Risk Calculation ---------------- #
def calculate_risk(state, speed, rain, night):
    base_score = 0
    state_map = {"normal": 0, "distracted": 4, "drowsy": 7}
    base_score += state_map.get(state, 0)

    if speed >= 65:
        base_score += 2
    elif speed >= 40:
        base_score += 1

    multiplier = 1
    if rain:
        multiplier += 0.5
    if night:
        multiplier += 0.3

    final_score = int(base_score * multiplier)
    return round(final_score, 1)


def get_recommendation(risk_score):
    if risk_score >= 10:
        return "High Risk, Pull over and rest."
    elif risk_score >= 7:
        return "High risk, reduce speed and stay focused."
    elif risk_score >= 4:
        return "Caution! Pay attention to the road."
    else:
        return "Status: Normal"


# Example usage
current_driver_state = driver_state
speed = 55
is_raining = True
is_night = False

risk_val = calculate_risk(current_driver_state, speed, is_raining, is_night)
message = get_recommendation(risk_val)

print(f"Risk Score: {risk_val} | Instruction: {message}")

 
 



    