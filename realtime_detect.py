"""
视频实时跌倒检测演示

流程: 视频帧 → YOLO Pose(17关键点) → SequenceBuffer(30帧缓存) → 拼接[pos,vel,acc]=(30,102) → LSTM → risk/time/label
"""

import os
import cv2
import numpy as np
from pose.pose_extractor import PoseExtractor
from sequence.sequence_buffer import SequenceBufferV3
from classifier.lstm_classifier import LSTMClassifier
from features.feature_builder import FeatureBuilder

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
POSE_MODEL = os.path.join(MODEL_DIR, "yolo11n-pose.pt")
LSTM_MODEL = os.path.join(MODEL_DIR, "lstm_multitask.pt")
NORM_PARAMS = os.path.join(MODEL_DIR, "norm_params.npz")

WINDOW_SIZE = 30
RISK_THRESHOLD = 0.5
PLAYBACK_DELAY = 100     # ms, 越小播放越快


def build_102_feature(buffer, person_id):
    raw = buffer.get_sequence(person_id)
    if raw is None:
        return None
    vel = buffer.get_velocity(person_id)
    acc = buffer.get_acceleration(person_id)
    return np.concatenate([raw, vel, acc], axis=1)


def draw_result(frame, bbox, result, person_id):
    x1, y1, x2, y2 = map(int, bbox)
    risk = result["risk"]
    time_val = result["time"]
    label = result["label"]

    if risk >= 0.8:
        color = (0, 0, 255)
        status = "FALL"
    elif risk >= RISK_THRESHOLD:
        color = (0, 165, 255)
        status = "WARNING"
    else:
        color = (0, 255, 0)
        status = "SAFE"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"ID:{person_id} {status}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"risk:{risk:.3f} time:{time_val:.1f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def _check_motion(buffer, person_id, min_displacement=8.0, max_torso_tilt=30.0):
    """位移 < 阈值且躯干直立 → 静止站立；位移大或躯干倾斜 → 放行"""
    seq = buffer.get_sequence(person_id)
    if seq is None:
        return False
    hip_y_first = seq[0, 34]   # left_hip y (feature 索引 11*3+1=34)
    hip_y_last = seq[-1, 34]
    nose_x_first = seq[0, 0]   # nose x
    nose_x_last = seq[-1, 0]
    displacement = np.sqrt((nose_x_last - nose_x_first)**2 + (hip_y_last - hip_y_first)**2)
    if displacement >= min_displacement:
        return True  # 有运动 → 放行给 LSTM
    # 无运动时检查躯干姿态：倾斜 > 阈值 → 可能躺倒 → 放行
    torso_angle = abs(seq[-1, 52])  # feature 索引 52 = torso_angle
    return torso_angle > max_torso_tilt


def main(video_path=None):
    pose_ext = PoseExtractor(model_path=POSE_MODEL)
    buffer = SequenceBufferV3(seq_len=WINDOW_SIZE)
    classifier = LSTMClassifier(model_path=LSTM_MODEL, norm_path=NORM_PARAMS)

    if video_path is None:
        cap = cv2.VideoCapture(0)
        source_name = "Camera"
    else:
        cap = cv2.VideoCapture(video_path)
        source_name = os.path.basename(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open {source_name}")
        return

    print(f"Source: {source_name}")
    print(f"WINDOW_SIZE={WINDOW_SIZE}, RISK_THRESHOLD={RISK_THRESHOLD}")
    print("Press 'q' to quit\n")

    frame_count = 0
    skip_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        persons = pose_ext.extract(frame)
        n_detected = len(persons)
        n_kept = 0
        pred_info = ""

        for person in persons:
            kpts = person["keypoints"]
            confs = person.get("keypoints_conf")
            bbox = person["bbox"]
            score = person.get("score", 0)
            mean_conf = np.mean(confs) if confs is not None else -1

            # --- 低置信度过滤 ---
            if confs is not None and mean_conf < 0.5:
                skip_count += 1
                pred_info += f" | skip(conf={mean_conf:.2f})"
                continue
            if score < 0.5:
                skip_count += 1
                pred_info += f" | skip(score={score:.2f})"
                continue

            n_kept += 1

            # --- 始终看作一个人 ---
            person_id = 1
            feature = FeatureBuilder.build(kpts, confs=confs)
            buffer.update(person_id, feature)
            buf_len = len(buffer.buffers.get(person_id, []))

            # --- 推理 + 运动门控 ---
            if buffer.is_ready(person_id):
                if not _check_motion(buffer, person_id, min_displacement=8.0):
                    result = {"risk": 0.0, "time": 0.0, "label": 0}
                    pred_info += f" | buf={buf_len} STATIC"
                else:
                    seq_102 = build_102_feature(buffer, person_id)
                    if seq_102 is not None:
                        result = classifier.predict(seq_102)
                        pred_info += (f" | buf={buf_len} "
                                      f"risk={result['risk']:.3f} time={result['time']:.1f} label={result['label']}")
                    else:
                        result = {"risk": 0.0, "time": 0.0, "label": 0}
                        pred_info += f" | buf={buf_len} seq=None"

                frame = draw_result(frame, bbox, result, person_id)
            else:
                pred_info += f" | buf={buf_len}/{WINDOW_SIZE}"

        # --- 每帧输出 ---
        print(f"[F{frame_count:04d}] detect={n_detected} keep={n_kept}{pred_info}")

        cv2.putText(frame, f"Frame: {frame_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if frame_count == 1:
            cv2.namedWindow("Fall Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Fall Detection", frame)

        if cv2.waitKey(PLAYBACK_DELAY) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames, skipped {skip_count} low-conf frames")


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else None
    main(video)