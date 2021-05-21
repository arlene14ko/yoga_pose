# importing the necessary libraries
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    send_from_directory,
    Response,
)
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import time
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///class_report.db"
# Initialize the database
db = SQLAlchemy(app)

# Create db model
class Reports(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    class_name = db.Column(db.String(100), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.now)
    poses_held = db.Column(db.String(500))
    purpose = db.Column(db.String(500))
    recommendation_class = db.Column(db.String(100))

    # Create a function to return a string when we add something
    def __repr__(self):
        return "<Name %r>" % self.id


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
model = pickle.load(open("yoga_poses_model.pkl", "rb"))


@app.route("/")
@app.route("/home")
@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    """
    Function that has both GET and POST method.
    This is the function where it will ask the user input and 
    then analyze that input and return it back to the html file
    """
    if request.method == "GET":
        return render_template("analyze.html")

    if request.method == "POST":

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files.get("file")
        print(f"File Input: {file}")

        if file == "":
            return redirect(request.url)

        elif file:
            print("Elif Uploaded Video file")
            print(file)
            return render_template("video_feed")
        else:
            return render_template("analyze.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


@app.route("/reports", methods=["POST", "GET"])
def reports():
    if request.method == "POST":
        form_name = request.form.get("name")
        form_class = request.form.get("class_name")
        name_form = Reports(name=form_name, class_name=form_class)

        try:
            db.session.add(name_form)
            db.session.commit()
            return redirect("/reports")
        except:
            return "There was an error adding this name"
    else:
        report = Reports.query.order_by(Reports.date_created)
        return render_template("report.html", reports=report)


@app.route("/live_feed")
def index():
    """
    Video streaming home page.
    """
    return render_template("live_feed.html")


def angles(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def gen():
    """
    Video streaming generator function.
    """
    count = 0
    pose_counts = {
        "downward_facing_dog": 0,
        "happy_baby_pose": 0,
        "low_lunge": 0,
        "half_split_pose": 0,
        "child_pose": 0,
        "cobra_pose": 0,
        "cow_pose": 0,
        "cat_pose": 0,
        "high_plank": 0,
        "easy_pose": 0,
        "upward_facing_dog": 0,
        "standing_forward_bend": 0,
    }
    poses = {
        "downward_facing_dog": 0,
        "happy_baby_pose": 0,
        "low_lunge": 0,
        "half_split_pose": 0,
        "child_pose": 0,
        "cobra_pose": 0,
        "cow_pose": 0,
        "cat_pose": 0,
        "high_plank": 0,
        "easy_pose": 0,
        "upward_facing_dog": 0,
        "standing_forward_bend": 0,
        "low": 0,
    }

    cap = cv2.VideoCapture("dataset/test2.mp4")

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)
                # print(results)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark
                    # print(len(landmarks))

                    left_shoulder = [
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                    ]
                    left_hip = [
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                    ]
                    left_knee = [
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
                    ]
                    left_elbow = [
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                    ]

                    right_shoulder = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                    ]
                    right_hip = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
                    ]
                    right_knee = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                    ]
                    right_elbow = [
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                    ]

                    # Get coordinates
                    row = list(
                        np.array(
                            [
                                [
                                    landmark.x,
                                    landmark.y,
                                    landmark.z,
                                    landmark.visibility,
                                ]
                                for landmark in landmarks
                            ]
                        ).flatten()
                    )
                    # print(len(row))
                    X = pd.DataFrame([row])
                    # print()
                    model_class = int(model.predict(X)[0])
                    # print(model_class)
                    if model_class == 0:
                        pose_class = "Adho Mukha Svanasana - Downward-Facing Dog"
                        poses["downward_facing_dog"] += 1
                        left_angle = round(
                            angles(left_shoulder, left_hip, left_knee), 2
                        )
                        right_angle = round(
                            angles(right_shoulder, right_hip, right_knee), 2
                        )
                        print(
                            f"Pose: {pose_class}, Left Angle: {left_angle}, Right Angle: {right_angle}"
                        )
                        if (
                            poses["downward_facing_dog"] == 10
                            and left_angle > 45
                            and left_angle < 90
                            and right_angle > 45
                            and right_angle < 90
                        ):
                            class_pose = "Downward-Facing Dog"
                            pose_counts["downward_facing_dog"] += 1
                            poses["downward_facing_dog"] = 0
                        elif poses["downward_facing_dog"] == 10:
                            class_pose = "Downward-Facing Dog"
                            pose_counts["downward_facing_dog"] += 1
                            poses["downward_facing_dog"] = 0

                    elif model_class == 1:
                        pose_class = "Ananda Balasana - Happy Baby's Pose"
                        poses["happy_baby_pose"] += 1
                        print(f"Pose: {pose_class}, Left Angle: , Right Angle: ")
                        if poses["happy_baby_pose"] == 10:
                            class_pose = "Ananda Balasana - Happy Baby's Pose"
                            pose_counts["happy_baby_pose"] += 1
                            poses["happy_baby_pose"] = 0

                    elif model_class == 2:
                        pose_class = "Anjaneyasana - Low Lunge"
                        poses["low"] += 1
                        poses["low_lunge"] += 1
                        left_angle = round(angles(left_knee, left_hip, right_knee), 2)
                        right_angle = round(angles(right_knee, right_hip, left_knee), 2)
                        print(
                            f"Pose: {pose_class}, Left Angle: {left_angle}, Right Angle: {right_angle}"
                        )
                        if poses["low_lunge"] == 10:
                            if left_angle > 100 or right_angle > 100:
                                class_pose = "Low Lunge"
                                pose_counts["low_lunge"] += 1
                                poses["low_lunge"] = 0
                        else:
                            if left_angle > 100 and right_angle > 100:
                                class_pose = "Low Lunge"
                                poses["low"] = 0

                    elif model_class == 3:
                        pose_class = "Ardha Hanumanasana- Half Splits Pose"
                        poses["half_split_pose"] += 1
                        left_angle = round(
                            angles(left_shoulder, left_hip, left_knee), 2
                        )
                        right_angle = round(
                            angles(right_shoulder, right_hip, right_knee), 2
                        )
                        print(
                            f"Pose: {pose_class}, Left Angle: {left_angle}, Right Angle: {right_angle}"
                        )
                        if (
                            poses["half_split_pose"] == 10
                            and left_angle > 25
                            and left_angle < 60
                            and right_angle > 25
                            and right_angle < 60
                        ):
                            print(pose_class)
                            class_pose = "Half Splits Pose"
                            pose_counts["half_split_pose"] += 1
                            poses["half_split_pose"] = 0

                    elif model_class == 4:
                        pose_class == "Balasana - Child's Pose"
                        poses["child_pose"] += 1
                        print(f"Pose: {pose_class}, Left Angle: , Right Angle: ")
                        if poses["child_pose"] == 10:
                            print(pose_class)
                            class_pose = "Child's Pose"
                            pose_counts["child_pose"] += 1
                            poses["child_pose"] = 0

                    elif model_class == 5:
                        pose_class = "Bhujangasana - Cobra Pose"
                        poses["cobra_pose"] += 1
                        left_angle = round(
                            angles(left_shoulder, left_hip, left_knee), 2
                        )
                        right_angle = round(
                            angles(right_shoulder, right_hip, right_knee), 2
                        )
                        print(
                            f"Pose: {pose_class}, Left Angle: {left_angle}, Right Angle: {right_angle}"
                        )
                        if (
                            poses["cobra_pose"] == 10
                            and left_angle > 90
                            and left_angle < 180
                            and right_angle > 90
                            and right_angle < 180
                        ):
                            print(pose_class)
                            class_pose = "Cobra Pose"
                            pose_counts["cobra_pose"] += 1
                            poses["cobra_pose"] = 0

                    elif model_class == 6:
                        pose_class = "Bitilasana - Cow Pose"
                        poses["cow_pose"] += 1
                        print(f"Pose: {pose_class}, Left Angle: , Right Angle: ")
                        if poses["cow_pose"] == 10:
                            print(pose_class)
                            class_pose = "Cow Pose"
                            pose_counts["cow_pose"] += 1
                            poses["cow_pose"] = 0

                    elif model_class == 7:
                        pose_class = "Marjariasana - Cat Pose"
                        poses["cat_pose"] += 1
                        print(f"Pose: {pose_class}, Left Angle: , Right Angle: ")
                        if poses["cat_pose"] == 10:
                            print(pose_class)
                            class_pose = "Cat Pose"
                            pose_counts["cat_pose"] += 1
                            poses["cat_pose"] = 0

                    elif model_class == 8:
                        pose_class = "Phalakasana - High Plank"
                        poses["high_plank"] += 1
                        left_angle = round(
                            angles(left_elbow, left_shoulder, left_knee), 2
                        )
                        right_angle = round(
                            angles(right_elbow, right_shoulder, right_knee), 2
                        )
                        print(
                            f"Pose: {pose_class}, Left Angle: {left_angle}, Right Angle: {right_angle}"
                        )
                        if (
                            poses["high_plank"] == 10
                            and left_angle > 45
                            and left_angle < 170
                            and right_angle > 45
                            and right_angle < 170
                        ):
                            print(pose_class)
                            class_pose = "High Plank"
                            pose_counts["high_plank"] += 1
                            poses["high_plank"] = 0

                    elif model_class == 9:
                        pose_class == "Sukhasana - Easy Pose"
                        poses["easy_pose"] += 1
                        print(f"Pose: {pose_class}, Left Angle: , Right Angle: ")
                        if poses["easy_pose"] == 10:
                            print(pose_class)
                            class_pose = "Easy Pose"
                            pose_counts["easy_pose"] += 1
                            poses["easy_pose"] = 0

                    elif model_class == 10:
                        pose_class = "Urdhva Mukha Svanasana - Upward-Facing Dog"
                        poses["upward_facing_dog"] += 1
                        print(f"Pose: {pose_class}, Left Angle: , Right Angle: ")
                        if poses["upward_facing_dog"] == 10:
                            print(pose_class)
                            class_pose = "Upward-Facing Dog"
                            pose_counts["upward_facing_dog"] += 1
                            poses["upward_facing_dog"] = 0

                    elif model_class == 11:
                        pose_class = "Uttanasana - Standing Forward Bend"
                        poses["standing_forward_bend"] += 1
                        print(f"Pose: {pose_class}, Left Angle: , Right Angle: ")
                        if poses["standing_forward_bend"] == 10:
                            print(pose_class)
                            class_pose = "Standing Forward Bend"
                            pose_counts["standing_forward_bend"] += 1
                            poses["standing_forward_bend"] = 0

                    print(f"Class Pose: {class_pose}")
                    # Setup status box
                    cv2.rectangle(image, (0, 0), (800, 45), (245, 117, 230), -1)

                    # Rep data
                    cv2.putText(
                        image,
                        "POSE: ",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 100, 0),
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        image,
                        class_pose,
                        (70, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                except:
                    pass

                # cv2.imshow('Mediapipe Feed', image)
                frame = cv2.imencode(".jpg", image)[1].tobytes()
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
                # time.sleep(0.1)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    break

            else:
                print(pose_counts)
                break


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)

