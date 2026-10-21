import csv
import os
import re
import math
import threading
import time
from datetime import datetime

import cv2
import mediapipe as mp
import speech_recognition as sr
import wx
from ultralytics import YOLO


LOG_FILE = "adas_log.csv"

# Load YOLO model
model = YOLO("yolov8n.pt")


# ================= DRIVER MONITORING ================= #

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

NOSE_TIP, LEFT_CHEEK, RIGHT_CHEEK = 1, 234, 454


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(pts):
    p1, p2, p3, p4, p5, p6 = pts
    return (
        _dist(p2, p6) + _dist(p3, p5)
    ) / (2.0 * _dist(p1, p4) + 1e-6)


class DriverMonitor:

    def __init__(
        self,
        camera_index=0,
        ear_threshold=0.21,
        drowsy_seconds=2.0,
        distract_seconds=2.0,
        yaw_limit=0.35
    ):
        self.camera_index = camera_index
        self.ear_threshold = ear_threshold
        self.drowsy_seconds = drowsy_seconds
        self.distract_seconds = distract_seconds
        self.yaw_limit = yaw_limit

        self.error = None
        self._state = "normal"
        self._frame = None
        self._annotated = None

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        self._closed_since = None
        self._away_since = None

    def start(self):
        self._thread = threading.Thread(
            target=self._run,
            daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()

        if self._thread:
            self._thread.join(timeout=3)

    @property
    def state(self):
        with self._lock:
            return self._state

    def get_frame(self, annotated=False):
        with self._lock:
            frame = self._annotated if annotated else self._frame
            return None if frame is None else frame.copy()

    def _run(self):

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            self.error = (
                f"Could not open camera {self.camera_index}. "
                "Check camera permissions or try camera_index=1."
            )
            return

        try:
            with mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as mesh:

                while not self._stop.is_set():

                    ok, frame = cap.read()

                    if not ok:
                        time.sleep(0.05)
                        continue

                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]

                    results = mesh.process(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )

                    ear = None
                    yaw = 0.0

                    if results.multi_face_landmarks:

                        lm = results.multi_face_landmarks[0].landmark

                        def pts(indices):
                            return [
                                (lm[i].x * w, lm[i].y * h)
                                for i in indices
                            ]

                        ear = (
                            eye_aspect_ratio(pts(RIGHT_EYE))
                            + eye_aspect_ratio(pts(LEFT_EYE))
                        ) / 2.0

                        nose = lm[NOSE_TIP].x
                        lc = lm[LEFT_CHEEK].x
                        rc = lm[RIGHT_CHEEK].x

                        left_d = abs(nose - lc)
                        right_d = abs(rc - nose)

                        yaw = (
                            (left_d - right_d)
                            / (left_d + right_d + 1e-6)
                        )

                        # Draw eye landmarks
                        for i in RIGHT_EYE + LEFT_EYE:
                            cv2.circle(
                                frame,
                                (
                                    int(lm[i].x * w),
                                    int(lm[i].y * h)
                                ),
                                3,
                                (0, 255, 0),
                                -1
                            )

                    state = self._update_state(
                        ear, yaw, time.time()
                    )

                    shown = frame.copy()

                    color = (
                        (0, 0, 255)
                        if state == "drowsy"
                        else (0, 165, 255)
                        if state == "distracted"
                        else (0, 255, 0)
                    )

                    cv2.putText(
                        shown,
                        f"Driver State: {state}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2
                    )

                    if ear is not None:
                        cv2.putText(
                            shown,
                            f"EAR: {ear:.2f}  Yaw: {yaw:+.2f}",
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
                    else:
                        cv2.putText(
                            shown,
                            "No face detected",
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 165, 255),
                            2
                        )

                    if state == "drowsy":
                        cv2.putText(
                            shown,
                            "WARNING: DROWSINESS DETECTED",
                            (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 0, 255),
                            2
                        )

                    with self._lock:
                        self._state = state
                        self._frame = frame
                        self._annotated = shown

        except Exception as e:
            self.error = str(e)

        finally:
            cap.release()

    def _update_state(self, ear, yaw, now):

        closed = ear is not None and ear < self.ear_threshold
        away = ear is None or abs(yaw) > self.yaw_limit

        if closed:
            if self._closed_since is None:
                self._closed_since = now
        else:
            self._closed_since = None

        if away:
            if self._away_since is None:
                self._away_since = now
        else:
            self._away_since = None

        if (
            self._closed_since is not None
            and now - self._closed_since >= self.drowsy_seconds
        ):
            return "drowsy"

        if (
            self._away_since is not None
            and now - self._away_since >= self.distract_seconds
        ):
            return "distracted"

        return "normal"


# ================= ADAS RISK CALCULATION ================= #

def evaluate_adas(
    speed_kmh, distance_m, obstacle, road, driver_state
):

    speed_ms = speed_kmh / 3.6
    ttc = distance_m / speed_ms if speed_ms > 0 else 999

    road_factor = {
        "Dry": 1.0,
        "Wet": 1.5,
        "Poor": 2.0
    }.get(road, 1.0)

    risk = (speed_ms * road_factor) / ttc

    if obstacle == "Animal":
        risk *= 1.3
    elif obstacle == "Pothole":
        risk *= 0.8

    risk += {
        "normal": 0,
        "drowsy": 7,
        "distracted": 4
    }.get(driver_state, 0)

    if risk > 25:
        decision = "EMERGENCY BRAKE"
    elif risk > 15:
        decision = "GRADUAL BRAKE + ALERT"
    elif risk > 8:
        decision = "PRE-BRAKE + WARNING"
    else:
        decision = "MONITOR"

    return round(ttc, 2), round(risk, 2), decision


# ================= CSV LOGGING ================= #

def log_scenario(
    speed, distance, obstacle, road,
    driver_state, ttc, risk, decision
):

    new_file = not os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if new_file:
            writer.writerow([
                "time", "speed_kmh", "distance_m", "obstacle",
                "road", "driver_state", "ttc_s", "risk", "decision"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            speed, distance, obstacle, road,
            driver_state, ttc, risk, decision
        ])


# ================= VOICE COMMANDS ================= #

def listen_once():

    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(
                source, duration=0.5
            )
            audio = recognizer.listen(
                source, timeout=5, phrase_time_limit=6
            )

        return recognizer.recognize_google(audio).lower()

    except Exception as e:
        print("Voice error:", e)
        return ""


def parse_command(text):

    out = {}

    m = re.search(r"speed\D{0,15}?(\d+(?:\.\d+)?)", text)
    if m:
        out["speed"] = m.group(1)

    m = re.search(r"distance\D{0,15}?(\d+(?:\.\d+)?)", text)
    if m:
        out["distance"] = m.group(1)

    for word in ("vehicle", "animal", "pothole"):
        if word in text:
            out["obstacle"] = word.capitalize()

    for word in ("dry", "wet", "poor"):
        if re.search(rf"\b{word}\b", text):
            out["road"] = word.capitalize()

    return out


# ================= GUI ================= #

class ADASApp(wx.Frame):

    def __init__(self):

        super().__init__(
            None,
            title="Unified ADAS System",
            size=(750, 700)
        )

        panel = wx.Panel(self)
        panel.SetBackgroundColour("Beige")

        self.monitor = DriverMonitor()
        self.monitor.start()

        title = wx.StaticText(
            panel, label="ADAS Risk Assessment Tool"
        )
        title.SetFont(
            wx.Font(
                16, wx.FONTFAMILY_DEFAULT,
                wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD
            )
        )

        self.driver_label = wx.StaticText(
            panel, label="Driver State: starting camera..."
        )

        self.driver_label.SetFont(
            wx.Font(
                11, wx.FONTFAMILY_DEFAULT,
                wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD
            )
        )

        console_label = wx.StaticText(
            panel, label="Smart Command Console"
        )

        self.command_input = wx.TextCtrl(
            panel, style=wx.TE_PROCESS_ENTER
        )
        self.command_input.SetHint(
            "Ex: Set speed 80 and distance 20"
        )

        speed_label = wx.StaticText(
            panel, label="Vehicle Speed (km/h)"
        )
        self.speed_input = wx.TextCtrl(panel)

        distance_label = wx.StaticText(
            panel, label="Distance to Obstacle (m)"
        )
        self.distance_input = wx.TextCtrl(panel)

        obstacle_label = wx.StaticText(
            panel, label="Obstacle Type"
        )
        self.obstacle_choice = wx.Choice(
            panel, choices=["Vehicle", "Animal", "Pothole"]
        )
        self.obstacle_choice.SetSelection(0)

        road_label = wx.StaticText(
            panel, label="Road Condition"
        )
        self.road_choice = wx.Choice(
            panel, choices=["Dry", "Wet", "Poor"]
        )
        self.road_choice.SetSelection(0)

        evaluate_btn = wx.Button(
            panel, label="Evaluate ADAS"
        )
        self.voice_btn = wx.Button(
            panel, label="Voice Command"
        )

        self.ttc_output = wx.StaticText(
            panel, label="TTC: -- seconds"
        )
        self.risk_output = wx.StaticText(
            panel, label="Risk Score: --"
        )
        self.decision_output = wx.StaticText(
            panel, label="Decision: --"
        )

        self.decision_output.SetFont(
            wx.Font(
                12, wx.FONTFAMILY_DEFAULT,
                wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD
            )
        )

        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(title, 0, wx.ALL | wx.CENTER, 10)
        sizer.Add(self.driver_label, 0, wx.ALL | wx.CENTER, 5)
        sizer.Add(console_label, 0, wx.LEFT, 20)
        sizer.Add(self.command_input, 0, wx.EXPAND | wx.ALL, 10)
        sizer.Add(wx.StaticLine(panel), 0, wx.EXPAND | wx.ALL, 5)

        for label, control in [
            (speed_label, self.speed_input),
            (distance_label, self.distance_input),
            (obstacle_label, self.obstacle_choice),
            (road_label, self.road_choice)
        ]:
            sizer.Add(label, 0, wx.LEFT, 20)
            sizer.Add(control, 0, wx.EXPAND | wx.ALL, 10)

        sizer.Add(evaluate_btn, 0, wx.ALL | wx.CENTER, 15)
        sizer.Add(self.ttc_output, 0, wx.ALL | wx.CENTER, 5)
        sizer.Add(self.risk_output, 0, wx.ALL | wx.CENTER, 5)
        sizer.Add(self.decision_output, 0, wx.ALL | wx.CENTER, 10)
        sizer.Add(self.voice_btn, 0, wx.ALL | wx.CENTER, 5)

        panel.SetSizer(sizer)

        self.command_input.Bind(
            wx.EVT_TEXT_ENTER, self.on_command
        )
        evaluate_btn.Bind(
            wx.EVT_BUTTON, self.on_evaluate
        )
        self.voice_btn.Bind(
            wx.EVT_BUTTON, self.on_voice_input
        )
        self.Bind(wx.EVT_CLOSE, self.on_close)

        # Refresh driver state and camera continuously
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        self.timer.Start(100)

        self.camera_warned = False
        self.Show()

    # -------- Live camera and status -------- #

    def on_timer(self, event):

        if self.monitor.error:
            self.driver_label.SetLabel(
                f"Driver State: unavailable ({self.monitor.error})"
            )
        else:
            self.driver_label.SetLabel(
                f"Driver State: {self.monitor.state}"
            )

        frame = self.monitor.get_frame(annotated=True)

        if frame is not None:
            cv2.imshow("Driver Monitoring - Drowsiness", frame)
            cv2.waitKey(1)

    # -------- Command console -------- #

    def on_command(self, event):

        parsed = parse_command(
            self.command_input.GetValue().lower()
        )

        if "speed" in parsed:
            self.speed_input.SetValue(parsed["speed"])

        if "distance" in parsed:
            self.distance_input.SetValue(parsed["distance"])

        if "obstacle" in parsed:
            self.obstacle_choice.SetStringSelection(
                parsed["obstacle"]
            )

        if "road" in parsed:
            self.road_choice.SetStringSelection(
                parsed["road"]
            )

        self.on_evaluate(None)

    # -------- Voice commands -------- #

    def on_voice_input(self, event):

        self.voice_btn.Disable()
        self.voice_btn.SetLabel("Listening...")

        threading.Thread(
            target=self._voice_worker,
            daemon=True
        ).start()

    def _voice_worker(self):

        text = listen_once()
        wx.CallAfter(self._voice_done, text)

    def _voice_done(self, text):

        self.voice_btn.Enable()
        self.voice_btn.SetLabel("Voice Command")

        if text:
            self.command_input.SetValue(text)
            self.on_command(None)
        else:
            wx.MessageBox(
                "Didn't catch that. Check microphone and internet.",
                "Voice"
            )

    # -------- ADAS evaluation -------- #

    def on_evaluate(self, event):

        try:
            sp = float(self.speed_input.GetValue())
            di = float(self.distance_input.GetValue())

            if sp < 0:
                raise ValueError

        except ValueError:
            wx.MessageBox(
                "Please enter a valid non-negative speed and distance.",
                "Input Error"
            )
            return

        if di <= 0:
            wx.MessageBox(
                "Distance must be greater than 0.",
                "Input Error"
            )
            return

        obstacle = self.obstacle_choice.GetStringSelection()
        road = self.road_choice.GetStringSelection()

        if self.monitor.error and not self.camera_warned:
            wx.MessageBox(
                f"{self.monitor.error}\nAssuming driver state = normal.",
                "Camera"
            )
            self.camera_warned = True

        driver_state = (
            "normal"
            if self.monitor.error
            else self.monitor.state
        )

        ttc, risk, decision = evaluate_adas(
            sp, di, obstacle, road, driver_state
        )

        self.ttc_output.SetLabel(
            f"TTC: {ttc} seconds"
        )
        self.risk_output.SetLabel(
            f"Risk Score: {risk} (driver: {driver_state})"
        )
        self.decision_output.SetLabel(
            f"Decision: {decision}"
        )

        log_scenario(
            sp, di, obstacle, road,
            driver_state, ttc, risk, decision
        )

    # -------- Close safely -------- #

    def on_close(self, event):

        self.timer.Stop()
        self.monitor.stop()
        cv2.destroyAllWindows()
        event.Skip()


if __name__ == "__main__":

    app = wx.App()
    ADASApp()
    app.MainLoop()