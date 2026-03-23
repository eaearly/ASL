import cv2
import time
import threading
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from collections import deque
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import os

# Resolve paths relative to this script's directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_SCRIPT_DIR, "..", "model", "best.pt")


# ================================================================== #
#  Color Palette & Theme                                              #
# ================================================================== #
class Theme:
    # Dark modern palette
    BG_DARK = "#1a1a2e"          # Main background
    BG_CARD = "#16213e"          # Card / panel background
    BG_INPUT = "#0f3460"         # Input / text area background
    ACCENT = "#e94560"           # Primary accent (coral red)
    ACCENT_HOVER = "#ff6b81"     # Hover state
    ACCENT_GREEN = "#00d2d3"     # Success / detection color
    ACCENT_BLUE = "#0984e3"      # Info / secondary accent
    TEXT_PRIMARY = "#ffffff"     # Main text
    TEXT_SECONDARY = "#a4b0be"   # Muted text
    TEXT_DIM = "#636e72"         # Very muted
    BORDER = "#2d3436"           # Subtle borders
    SUCCESS_BG = "#00b894"       # Confidence bar green
    WARNING_BG = "#fdcb6e"       # Confidence bar yellow
    DANGER_BG = "#d63031"        # Confidence bar red


# ================================================================== #
#  Rounded-feel Button                                                #
# ================================================================== #
class StyledButton(tk.Canvas):
    """A modern pill-shaped button drawn on a Canvas."""

    def __init__(self, master, text="Button", command=None, width=200, height=44,
                 bg=Theme.ACCENT, fg=Theme.TEXT_PRIMARY, hover_bg=Theme.ACCENT_HOVER,
                 font_size=12, **kwargs):
        super().__init__(master, width=width, height=height,
                         bg=master["bg"], highlightthickness=0, **kwargs)
        self.command = command
        self._bg = bg
        self._fg = fg
        self._hover_bg = hover_bg
        self._button_width = width
        self._button_height = height
        self._text = text
        self._font = tkFont.Font(family="Segoe UI", size=font_size, weight="bold")
        self._draw(bg)
        self.bind("<Enter>", lambda e: self._draw(hover_bg))
        self.bind("<Leave>", lambda e: self._draw(bg))
        self.bind("<ButtonRelease-1>", lambda e: self._on_click())
        self.config(cursor="hand2")

    def _draw(self, fill):
        self.delete("all")
        r = self._button_height // 2
        # Pill shape
        self.create_oval(0, 0, self._button_height, self._button_height, fill=fill, outline="")
        self.create_oval(self._button_width - self._button_height, 0, self._button_width, self._button_height, fill=fill, outline="")
        self.create_rectangle(r, 0, self._button_width - r, self._button_height, fill=fill, outline="")
        self.create_text(self._button_width // 2, self._button_height // 2, text=self._text,
                         fill=self._fg, font=self._font)

    def _on_click(self):
        if self.command:
            self.command()

    def set_text(self, text):
        self._text = text
        self._draw(self._bg)


# ================================================================== #
#  Main Application                                                   #
# ================================================================== #
class ASLDetectorGUI:
    # Best confidence thresholds per ASL sign type
    # - Static signs (A, B, C, …): high confidence works well
    # - Similar signs (M/N, U/V, R/U): lower threshold catches them
    # - Motion signs (J, Z): need lower threshold for mid-motion frames
    RECOMMENDED_CONF = 0.45  # Good balance for ASL overall

    def __init__(self, window, model_path, conf_threshold=None):
        """Initialize ASL detector with CUDA GPU acceleration & modern UI."""
        if conf_threshold is None:
            conf_threshold = self.RECOMMENDED_CONF

        self.window = window
        self.window.title("ASL Detector")
        self.window.geometry("1280x820")
        self.window.minsize(1000, 650)
        self.window.configure(bg=Theme.BG_DARK)

        # --- CUDA / GPU Setup ---
        if torch.cuda.is_available():
            self.device = "cuda"
            self.gpu_name = torch.cuda.get_device_name(0)
            print(f"[GPU] Using CUDA device: {self.gpu_name}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            self.device = "cpu"
            self.gpu_name = "CPU"
            print("[GPU] CUDA not available — falling back to CPU")

        # Initialize model and move to GPU
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Warm-up
        print("[GPU] Warming up model…")
        dummy = torch.zeros(1, 3, 640, 640, device=self.device)
        self.model.predict(dummy, verbose=False)
        print("[GPU] Warm-up complete")

        self.conf_threshold = conf_threshold
        self.cap = None
        self.is_running = False

        # FPS smoothing
        self.fps_history = deque(maxlen=30)
        self.prev_time = time.perf_counter()

        # Sentence builder — accumulate detected letters
        self.sentence = []
        self._last_letter = None
        self._letter_hold_count = 0
        self._HOLD_FRAMES = 8  # require N consecutive frames to accept a letter

        # Threaded camera
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._camera_thread = None
        self._after_id = None          # track scheduled after() so we can cancel it
        self._destroyed = False        # guard against widget access after close

        # Fonts
        self.font_title = tkFont.Font(family="Segoe UI", size=20, weight="bold")
        self.font_heading = tkFont.Font(family="Segoe UI", size=13, weight="bold")
        self.font_body = tkFont.Font(family="Segoe UI", size=11)
        self.font_big_letter = tkFont.Font(family="Segoe UI", size=72, weight="bold")
        self.font_sentence = tkFont.Font(family="Segoe UI", size=16)
        self.font_small = tkFont.Font(family="Segoe UI", size=9)

        self._build_ui()

    # ------------------------------------------------------------------ #
    #  UI Construction                                                     #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        # Top bar
        top_bar = tk.Frame(self.window, bg=Theme.BG_CARD, height=56)
        top_bar.pack(fill=tk.X)
        top_bar.pack_propagate(False)

        tk.Label(top_bar, text="ASL Detector", font=self.font_title,
                 bg=Theme.BG_CARD, fg=Theme.ACCENT).pack(side=tk.LEFT, padx=20)

        self.device_label = tk.Label(
            top_bar, text=f"{self.gpu_name}  |  {self.device.upper()}",
            font=self.font_small, bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY)
        self.device_label.pack(side=tk.RIGHT, padx=20)

        self.fps_label = tk.Label(top_bar, text="FPS: —", font=self.font_body,
                                  bg=Theme.BG_CARD, fg=Theme.ACCENT_GREEN)
        self.fps_label.pack(side=tk.RIGHT, padx=10)

        # Body
        body = tk.Frame(self.window, bg=Theme.BG_DARK)
        body.pack(expand=True, fill=tk.BOTH, padx=16, pady=(10, 16))

        # --- Left: Video ---
        video_card = tk.Frame(body, bg=Theme.BG_CARD, padx=3, pady=3)
        video_card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 10))

        self.video_label = tk.Label(video_card, bg="#000000")
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # --- Right: Sidebar ---
        sidebar = tk.Frame(body, bg=Theme.BG_DARK, width=320)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # -- Card 1: Controls --
        ctrl_card = self._make_card(sidebar, "Controls")

        self.start_btn = StyledButton(ctrl_card, text="Start Detection",
                                       command=self.toggle_detection, width=280, height=44)
        self.start_btn.pack(pady=(8, 14))

        # Confidence slider with live value label
        slider_row = tk.Frame(ctrl_card, bg=Theme.BG_CARD)
        slider_row.pack(fill=tk.X, pady=(0, 4))
        tk.Label(slider_row, text="Confidence", font=self.font_body,
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side=tk.LEFT)
        self.conf_value_label = tk.Label(slider_row, text=f"{self.conf_threshold:.0%}",
                                          font=self.font_body, bg=Theme.BG_CARD,
                                          fg=Theme.ACCENT_GREEN)
        self.conf_value_label.pack(side=tk.RIGHT)

        # Style the ttk Scale to fit the dark theme
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Custom.Horizontal.TScale",
                         background=Theme.BG_CARD,
                         troughcolor=Theme.BG_INPUT,
                         sliderthickness=18)

        self.confidence_slider = ttk.Scale(ctrl_card, from_=0.10, to=1.0,
                                            orient=tk.HORIZONTAL,
                                            value=self.conf_threshold,
                                            style="Custom.Horizontal.TScale",
                                            command=self._on_conf_change)
        self.confidence_slider.pack(fill=tk.X, pady=(0, 6))

        # Tip text
        tk.Label(ctrl_card,
                 text="Tip: 40-50% works best for ASL.\n"
                      "Lower for motion signs (J, Z).\n"
                      "Higher for static signs (A, B, C).",
                 font=self.font_small, bg=Theme.BG_CARD,
                 fg=Theme.TEXT_DIM, justify=tk.LEFT).pack(anchor="w", pady=(0, 4))

        # -- Card 2: Detection --
        det_card = self._make_card(sidebar, "Detection")

        # Big detected letter display
        self.big_letter_label = tk.Label(det_card, text="—",
                                          font=self.font_big_letter,
                                          bg=Theme.BG_CARD, fg=Theme.ACCENT)
        self.big_letter_label.pack(pady=(4, 2))

        # Confidence bar
        self.conf_bar_canvas = tk.Canvas(det_card, height=14, bg=Theme.BG_INPUT,
                                          highlightthickness=0)
        self.conf_bar_canvas.pack(fill=tk.X, padx=20, pady=(0, 4))

        self.det_conf_label = tk.Label(det_card, text="Confidence: —",
                                        font=self.font_small, bg=Theme.BG_CARD,
                                        fg=Theme.TEXT_SECONDARY)
        self.det_conf_label.pack(pady=(0, 8))

        # -- Card 3: Sentence Builder --
        sent_card = self._make_card(sidebar, "Sentence Builder")

        self.sentence_label = tk.Label(sent_card, text="", font=self.font_sentence,
                                        bg=Theme.BG_INPUT, fg=Theme.TEXT_PRIMARY,
                                        anchor="w", padx=10, pady=8, wraplength=260,
                                        justify=tk.LEFT)
        self.sentence_label.pack(fill=tk.X, pady=(4, 6))

        btn_row = tk.Frame(sent_card, bg=Theme.BG_CARD)
        btn_row.pack(fill=tk.X)

        self.space_btn = StyledButton(btn_row, text="Space", width=130, height=34,
                                       bg=Theme.ACCENT_BLUE, hover_bg="#74b9ff",
                                       font_size=10, command=self._add_space)
        self.space_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.clear_btn = StyledButton(btn_row, text="Clear", width=130, height=34,
                                       bg=Theme.BORDER, hover_bg=Theme.TEXT_DIM,
                                       font_size=10, command=self._clear_sentence)
        self.clear_btn.pack(side=tk.LEFT)

        # -- Card 4: Detection Log --
        log_card = self._make_card(sidebar, "Log")

        self.log_text = tk.Text(log_card, height=6, font=self.font_small,
                                 bg=Theme.BG_INPUT, fg=Theme.TEXT_SECONDARY,
                                 relief="flat", wrap=tk.WORD, padx=8, pady=6,
                                 insertbackground=Theme.TEXT_SECONDARY, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _make_card(self, parent, title):
        """Create a dark themed card with title."""
        card = tk.Frame(parent, bg=Theme.BG_CARD, padx=14, pady=10)
        card.pack(fill=tk.X, pady=(0, 10))
        tk.Label(card, text=title, font=self.font_heading,
                 bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        sep = tk.Frame(card, bg=Theme.ACCENT, height=2)
        sep.pack(fill=tk.X, pady=(4, 6))
        return card

    # ------------------------------------------------------------------ #
    #  Confidence slider callback                                         #
    # ------------------------------------------------------------------ #
    def _on_conf_change(self, value):
        v = float(value)
        self.conf_value_label.config(text=f"{v:.0%}")
        # Color code the label
        if v >= 0.60:
            self.conf_value_label.config(fg=Theme.SUCCESS_BG)
        elif v >= 0.35:
            self.conf_value_label.config(fg=Theme.WARNING_BG)
        else:
            self.conf_value_label.config(fg=Theme.DANGER_BG)

    # ------------------------------------------------------------------ #
    #  Sentence builder helpers                                           #
    # ------------------------------------------------------------------ #
    def _add_space(self):
        self.sentence.append(" ")
        self._update_sentence_display()

    def _clear_sentence(self):
        self.sentence.clear()
        self._last_letter = None
        self._letter_hold_count = 0
        self._update_sentence_display()

    def _update_sentence_display(self):
        self.sentence_label.config(text="".join(self.sentence) + "▌")

    # ------------------------------------------------------------------ #
    #  Confidence bar                                                     #
    # ------------------------------------------------------------------ #
    def _draw_conf_bar(self, confidence):
        c = self.conf_bar_canvas
        c.delete("all")
        w = c.winfo_width() or 260
        h = c.winfo_height() or 14
        fill_w = int(w * confidence)
        if confidence >= 0.70:
            color = Theme.SUCCESS_BG
        elif confidence >= 0.45:
            color = Theme.WARNING_BG
        else:
            color = Theme.DANGER_BG
        c.create_rectangle(0, 0, fill_w, h, fill=color, outline="")
        self.det_conf_label.config(text=f"Confidence: {confidence:.1%}")

    # ------------------------------------------------------------------ #
    #  Threaded camera capture                                            #
    # ------------------------------------------------------------------ #
    def _camera_reader(self):
        while self.is_running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                with self._frame_lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.001)

    # ------------------------------------------------------------------ #
    #  Main update loop                                                   #
    # ------------------------------------------------------------------ #
    def update_frame(self):
        if not self.is_running or self._destroyed:
            return

        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None

        if frame is not None:
            # FPS
            now = time.perf_counter()
            dt = now - self.prev_time
            self.prev_time = now
            if dt > 0:
                self.fps_history.append(1.0 / dt)
            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

            # --- Inference ---
            conf = self.confidence_slider.get()
            results = self.model.predict(
                frame, conf=conf, device=self.device,
                half=(self.device == "cuda"),
                imgsz=640, verbose=False,
            )
            result = results[0]

            # --- Process detections (use slider value — NO hardcoded filter) ---
            best_letter = None
            best_conf = 0.0
            detected_letters = []

            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                predicted_letter = self.model.names[class_id]
                detected_letters.append((predicted_letter, confidence))

                # Track the highest-confidence detection
                if confidence > best_conf:
                    best_conf = confidence
                    best_letter = predicted_letter

                # Draw bounding box with color based on confidence
                if confidence >= 0.70:
                    color = (0, 210, 148)   # green
                elif confidence >= 0.45:
                    color = (0, 210, 253)   # cyan
                else:
                    color = (96, 163, 188)  # muted blue

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{predicted_letter} {confidence:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # --- Update big letter display ---
            if best_letter:
                self.big_letter_label.config(text=best_letter, fg=Theme.ACCENT)
                self._draw_conf_bar(best_conf)

                # Sentence builder: require N consecutive identical detections
                if best_letter == self._last_letter:
                    self._letter_hold_count += 1
                else:
                    self._last_letter = best_letter
                    self._letter_hold_count = 1

                if self._letter_hold_count == self._HOLD_FRAMES:
                    self.sentence.append(best_letter)
                    self._update_sentence_display()
            else:
                self.big_letter_label.config(text="—", fg=Theme.TEXT_DIM)
                self._draw_conf_bar(0)
                self._last_letter = None
                self._letter_hold_count = 0

            # --- Update log ---
            if detected_letters:
                self.log_text.config(state=tk.NORMAL)
                self.log_text.delete(1.0, tk.END)
                for letter, c in detected_letters:
                    self.log_text.insert(tk.END, f"  {letter}  →  {c:.1%}\n")
                self.log_text.config(state=tk.DISABLED)

            # --- FPS label ---
            self.fps_label.config(text=f"FPS: {avg_fps:.1f}")

            # --- Render frame ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(image=frame_pil)
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk

        if not self._destroyed:
            self._after_id = self.window.after(1, self.update_frame)

    # ------------------------------------------------------------------ #
    #  Start / Stop                                                       #
    # ------------------------------------------------------------------ #
    def _cancel_pending_after(self):
        """Cancel any pending window.after callback."""
        if self._after_id is not None:
            try:
                self.window.after_cancel(self._after_id)
            except (tk.TclError, Exception):
                pass
            self._after_id = None

    def toggle_detection(self):
        if self.is_running:
            self.is_running = False
            self._cancel_pending_after()
            self.start_btn.set_text("Start Detection")
            self.start_btn._bg = Theme.ACCENT
            self.start_btn._draw(Theme.ACCENT)
            if self._camera_thread is not None:
                self._camera_thread.join(timeout=2)
                self._camera_thread = None
            self.release_camera()
        else:
            self.initialize_camera()
            self.is_running = True
            self.start_btn.set_text("Stop Detection")
            self.start_btn._bg = Theme.DANGER_BG
            self.start_btn._draw(Theme.DANGER_BG)
            self._camera_thread = threading.Thread(target=self._camera_reader, daemon=True)
            self._camera_thread.start()
            self.prev_time = time.perf_counter()
            self.fps_history.clear()
            self.update_frame()

    def initialize_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def release_camera(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None

    def shutdown(self):
        """Clean shutdown: stop loop, cancel callbacks, release camera."""
        self._destroyed = True
        self.is_running = False
        self._cancel_pending_after()
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=2)
            self._camera_thread = None
        self.release_camera()

    def __del__(self):
        # Only release camera — don't touch Tk widgets (they may already be gone)
        self.is_running = False
        try:
            self.release_camera()
        except Exception:
            pass


def main():
    root = tk.Tk()
    app = ASLDetectorGUI(root, _MODEL_PATH)

    def on_close():
        app.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()