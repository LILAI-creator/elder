from collections import deque


class RiskEngine:
    """
    工业级风险评估引擎

    输出：
        SAFE / WARNING / DANGER

    特点：
        - EMA平滑
        - 连续触发
        - 状态强制闭环（不会UNKNOWN）
    """

    def __init__(
            self,
            smooth_alpha=0.6,
            warning_thres=0.35,
            danger_thres=0.6,
            trigger_frames=5,
            history_size=10
    ):

        self.alpha = smooth_alpha
        self.warning_thres = warning_thres
        self.danger_thres = danger_thres
        self.trigger_frames = trigger_frames
        self.history_size = history_size

        # 每个ID的状态
        self.history = {}
        self.counter = {}
        self.smooth_score = {}

    # ==================================================
    # 初始化
    # ==================================================
    def _init_if_needed(self, person_id, danger_prob):

        if person_id not in self.smooth_score:
            self.smooth_score[person_id] = danger_prob
            self.history[person_id] = deque(maxlen=self.history_size)
            self.counter[person_id] = 0

    # ==================================================
    # 更新
    # ==================================================
    def update(self, person_id, danger_prob):

        # -----------------------------
        # 初始化保护
        # -----------------------------
        self._init_if_needed(person_id, danger_prob)

        # -----------------------------
        # EMA 平滑
        # -----------------------------
        prev = self.smooth_score[person_id]
        smooth = self.alpha * danger_prob + (1 - self.alpha) * prev
        self.smooth_score[person_id] = smooth

        # -----------------------------
        # 历史记录
        # -----------------------------
        self.history[person_id].append(smooth)

        # -----------------------------
        # 连续危险计数
        # -----------------------------
        if smooth >= self.danger_thres:
            self.counter[person_id] += 1
        else:
            self.counter[person_id] = 0

        # -----------------------------
        # 状态机（强制三态闭环）
        # -----------------------------
        if self.counter[person_id] >= self.trigger_frames:
            state = "DANGER"

        elif smooth >= self.warning_thres:
            state = "WARNING"

        else:
            state = "SAFE"

        # -----------------------------
        # 强制兜底（防止 UNKNOWN）
        # -----------------------------
        if state not in ["SAFE", "WARNING", "DANGER"]:
            state = "SAFE"

        return {
            "risk_score": float(smooth),
            "state": state
        }

    # ==================================================
    # 重置
    # ==================================================
    def reset(self, person_id=None):

        if person_id is None:
            self.history.clear()
            self.counter.clear()
            self.smooth_score.clear()

        else:
            self.history.pop(person_id, None)
            self.counter.pop(person_id, None)
            self.smooth_score.pop(person_id, None)