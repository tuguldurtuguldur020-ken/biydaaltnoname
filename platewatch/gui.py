"""Pure Python / Tkinter desktop interface. All UI work stays on the main thread."""
from copy import deepcopy
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageTk
from . import __version__

from .plates import clean_text, display_plate, normalize_plate
from .recognition import PlateRecognizer, overlay_font
from .media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from .rules import DAY_NAMES, evaluate, rule_plate, restriction_days
from .settings import APP_DIR, APP_TITLE, DATA_DIR, UB_TIME, load_settings, now_ub, save_settings
from .storage import RecordStore
from .worker import DetectionWorker

BG = "#091419"
PANEL = "#10232b"
RAISED = "#19333e"
BORDER = "#284650"
TEXT = "#edf8fa"
MUTED = "#93afb7"
GREEN = "#4de0ba"
RED = "#fb7185"
AMBER = "#fbbf24"
ACCENTS = {"Цэнхэр": "#48b7ff", "Ногоон": "#33d5b3", "Нил ягаан": "#aa91ff", "Ягаан": "#f28dc0"}
MODES = {"Ажлын өдрийн хуваарь": "weekly", "Тэгш / сондгой": "odd_even", "Унтраалттай": "off"}
RECOGNITION_MODES = {"Хурдан": "fast", "Тэнцвэртэй": "balanced", "Нарийвчилсан": "detailed"}


class PlateWatchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        height = min(820, max(640, self.winfo_screenheight() - 90))
        width = min(1440, self.winfo_screenwidth() - 40)
        self.geometry(f"{width}x{height}")
        self.minsize(min(1060, width), min(640, height))
        if os.name == "nt":
            self.state("zoomed")
        self.configure(bg=BG)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=str(DATA_DIR / "app.log"), level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
        self.settings = load_settings()
        self.accent = ACCENTS.get(self.settings.get("accent"), ACCENTS["Цэнхэр"])
        self.store = RecordStore()
        self.worker = DetectionWorker(self.settings, self.store)
        self.preparing = False
        self.pending_source = None
        self.active_kind = None
        self.active_source = None
        self.camera_overlay = ([], 0)
        self.last_error = ""
        self.last_import_dir = str(APP_DIR)
        self.preview_displayed = False
        self.closing = False
        self.preview_photo = None
        self._preview_item = None
        self._preview_dirty = True
        self._preview_limit = 1920
        self._preview_retry_at = 0
        self._preview_failed = False
        self._records_dirty = False
        self.crop_photo = None
        self.last_rgb = None
        self.last_detection = None
        self.last_rule_result = None
        self.fullscreen = False
        self.preview_fill = False
        self.video_source = None
        self.playback_seconds = 0
        self.video_duration = 0
        self.video_fps = 25
        self.frame_check_busy = False
        self._frame_original = None
        self._frame_source = None
        self._frame_seconds = None
        self._requested_original = None
        self._frame_pinned = False
        self.session_records = {}
        self.video_time = tk.StringVar(value="Зураг, видео эсвэл камер нээнэ үү")
        self.analysis_time = tk.StringVar(value="")
        self.current_page = "monitor"
        self.status_text = tk.StringVar(value="Бэлэн • Зураг, видео эсвэл камер сонгоно уу")
        self.stat_vars = [tk.StringVar(value="0") for _ in range(4)]
        self.camera_var = tk.StringVar(value="Авто")
        self.import_path_var = tk.StringVar()
        self.query_plate = tk.StringVar()
        self.query_date = tk.StringVar(value=now_ub().date().isoformat())
        self.only_restricted = tk.BooleanVar(value=False)
        self.progress = tk.DoubleVar(value=0)
        self.configure_styles()
        self.build_ui()
        self.refresh_records()
        self.tick()
        self.after(50, self.poll)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", self.escape)
        self.bind("<F11>", lambda event: self.toggle_fullscreen())
        self.bind("<Control-space>", lambda event: self.pause())
        self.bind("<Control-o>", lambda event: self.choose_images())
        self.bind("<Control-s>", lambda event: self.snapshot())

    def report_callback_exception(self, exc, value, traceback):
        logging.error("GUI error", exc_info=(exc, value, traceback))
        if not self.closing:
            messagebox.showerror("Алдаа", str(value), parent=self)

    def configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                        rowheight=33, borderwidth=0, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background=RAISED, foreground=MUTED,
                        font=("Segoe UI", 9, "bold"), relief="flat", padding=8)
        style.map("Treeview", background=[("selected", "#21446c")], foreground=[("selected", "white")])
        style.configure("TCombobox", fieldbackground=RAISED, background=RAISED, foreground=TEXT,
                        arrowcolor=TEXT, padding=6, bordercolor=BORDER)
        style.map("TCombobox", fieldbackground=[("readonly", RAISED)], foreground=[("readonly", TEXT)])
        style.configure("Horizontal.TProgressbar", background=self.accent, troughcolor=RAISED, borderwidth=0)
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", PANEL)])
        style.configure("TScrollbar", background=BORDER, troughcolor=PANEL, arrowcolor=MUTED, borderwidth=0)
        self.option_add("*TCombobox*Listbox.background", RAISED)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*Font", "{Segoe UI} 10")

    def label(self, parent, text="", size=10, color=TEXT, bold=False, bg=None, **kwargs):
        return tk.Label(parent, text=text, bg=bg or parent.cget("bg"), fg=color,
                        font=("Segoe UI", size, "bold" if bold else "normal"), **kwargs)

    def button(self, parent, text, command, primary=False, danger=False, **kwargs):
        color = self.accent if primary else "#402537" if danger else RAISED
        button = tk.Button(parent, text=text, command=command, bg=color,
                           fg=BG if primary else RED if danger else TEXT,
                           activebackground="#304666", activeforeground=TEXT,
                           relief="flat", bd=0, cursor="hand2", padx=14, pady=10,
                           font=("Segoe UI", 10, "bold"), **kwargs)
        button.bind("<Enter>", lambda event: button.configure(bg="#28405e" if not primary else "#85d0ff"))
        button.bind("<Leave>", lambda event: button.configure(bg=color))
        return button

    def entry(self, parent, variable=None, width=22):
        return tk.Entry(parent, textvariable=variable, width=width, bg=RAISED, fg=TEXT,
                        insertbackground=TEXT, relief="flat", highlightthickness=1,
                        highlightbackground=BORDER, highlightcolor=self.accent,
                        font=("Segoe UI", 11))

    def card(self, parent, **kwargs):
        return tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER, **kwargs)

    def build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        sidebar = self.sidebar = tk.Frame(self, bg=PANEL, width=208, padx=14, pady=20)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.pack_propagate(False)
        self.label(sidebar, "◈  PLATEWATCH", 17, self.accent, True).pack(anchor="w", pady=(0, 4))
        self.label(sidebar, "ДУГААР ТАНИХ СИСТЕМ", 8, MUTED).pack(anchor="w", pady=(0, 22))
        self.label(sidebar, "АЖЛЫН ТАЛБАР", 8, MUTED, True).pack(anchor="w", pady=(0, 8))
        self.nav = {}
        for key, title in [("monitor", "◉   Шууд хяналт"), ("history", "▤   Илрэлтийн түүх"), ("rules", "◷   Хязгаарлалт"), ("settings", "⚙   Тохиргоо")]:
            button = self.button(sidebar, title, lambda name=key: self.show_page(name), anchor="w")
            button.pack(fill="x", pady=3)
            self.nav[key] = button
        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", pady=20)
        self.label(sidebar, "ЭХ ҮҮСВЭР НЭМЭХ", 8, MUTED, True).pack(anchor="w", pady=(0, 9))
        self.image_button = self.button(sidebar, "+  Зураг импортлох", self.choose_images)
        self.image_button.pack(fill="x", pady=4)
        self.video_button = self.button(sidebar, "+  Видео шалгах", self.choose_video, primary=True)
        self.video_button.pack(fill="x", pady=4)
        self.label(sidebar, "Камер • автоматаар олно", 9, MUTED).pack(anchor="w", pady=(14, 5))
        camera_row = tk.Frame(sidebar, bg=PANEL)
        camera_row.pack(fill="x")
        ttk.Combobox(camera_row, textvariable=self.camera_var, values=["Авто", "0", "1", "2", "3"], width=5).pack(side="left")
        self.camera_button = self.button(camera_row, "Камер", self.start_camera)
        self.camera_button.pack(side="right")
        self.label(sidebar, "Эсвэл файлын замыг оруулна уу", 8, MUTED).pack(anchor="w", pady=(10, 4))
        path_row = tk.Frame(sidebar, bg=PANEL)
        path_row.pack(fill="x")
        path_entry = self.entry(path_row, self.import_path_var, 14)
        path_entry.pack(side="left", fill="x", expand=True, ipady=6)
        path_entry.bind("<Return>", lambda event: self.open_import_path())
        self.button(path_row, "Нээх", self.open_import_path).pack(side="right", padx=(4, 0))
        bottom = tk.Frame(sidebar, bg=PANEL)
        bottom.pack(side="bottom", fill="x")
        self.button(bottom, "↗  Хадгалсан файлууд", lambda: self.open_path(DATA_DIR)).pack(fill="x", pady=(0, 12))
        self.label(bottom, "●  ТӨХӨӨРӨМЖ ДЭЭР АЖИЛЛАНА", 8, GREEN, True).pack(anchor="w")
        self.label(bottom, f"PlateWatch {__version__}\nF11 • Бүтэн дэлгэц", 9, MUTED, justify="left").pack(anchor="w", pady=(6, 0))

        shell = self.shell = tk.Frame(self, bg=BG, padx=20, pady=16)
        shell.grid(row=0, column=1, sticky="nsew")
        shell.grid_columnconfigure(0, weight=1)
        shell.grid_rowconfigure(1, weight=1)
        header = self.header = tk.Frame(shell, bg=BG)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        self.page_title = self.label(header, "Хөдөлгөөний ухаалаг хяналт", 21, TEXT, True)
        self.page_title.pack(side="left")
        self.clock_label = self.label(header, "", 10, MUTED, justify="right")
        self.clock_label.pack(side="right")
        self.page_area = tk.Frame(shell, bg=BG)
        self.page_area.grid(row=1, column=0, sticky="nsew")
        self.page_area.grid_rowconfigure(0, weight=1)
        self.page_area.grid_columnconfigure(0, weight=1)
        self.pages = {key: tk.Frame(self.page_area, bg=BG) for key in self.nav}
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")
        self.build_monitor(self.pages["monitor"])
        self.build_history(self.pages["history"])
        self.build_rules(self.pages["rules"])
        self.build_settings(self.pages["settings"])
        status = self.status_bar = tk.Frame(shell, bg=BG)
        status.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.label(status, "●", 10, GREEN).pack(side="left", padx=(0, 8))
        self.status_label = self.label(status, "", 9, MUTED, anchor="w")
        self.status_label.configure(textvariable=self.status_text, wraplength=1050)
        self.status_label.pack(side="left", fill="x", expand=True)
        self.show_page("monitor")

    def build_monitor(self, page):
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        stats = self.stats_panel = tk.Frame(page, bg=BG)
        stats.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        for i, (title, sub, color) in enumerate([
            ("ӨНӨӨДРИЙН БҮРТГЭЛ", "хадгалсан илрэлт", self.accent),
            ("ДАВХАРДААГҮЙ ДУГААР", "өнөөдрийн автомашин", GREEN),
            ("ХЯЗГААРЛАЛТТАЙ", "тохируулсан хуваариар", RED),
            ("AI БОЛОВСРУУЛАЛТ", "кадр / секунд", "#b8a1ff"),
        ]):
            stats.grid_columnconfigure(i, weight=1, uniform="stats")
            card = self.card(stats, padx=14, pady=8)
            card.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0 if i == 3 else 6))
            tk.Frame(card, bg=color, height=2).pack(fill="x", pady=(0, 7))
            self.label(card, title, 8, MUTED, True).pack(anchor="w")
            self.label(card, "", 22, color, True, textvariable=self.stat_vars[i]).pack(anchor="w", pady=(2, 0))
            self.label(card, sub, 8, MUTED).pack(anchor="w")

        middle = tk.Frame(page, bg=BG)
        middle.grid(row=1, column=0, sticky="nsew")
        middle.grid_columnconfigure(0, weight=1)
        middle.grid_rowconfigure(0, weight=1)
        preview_card = self.preview_card = self.card(middle)
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        preview_card.grid_columnconfigure(0, weight=1)
        preview_card.grid_rowconfigure(2, weight=1)
        bar = tk.Frame(preview_card, bg=PANEL, padx=14, pady=10)
        bar.grid(row=0, column=0, sticky="ew")
        self.label(bar, "●", 10, GREEN).pack(side="left")
        self.main_image_button = self.button(bar, "+ Зураг", self.choose_images)
        self.main_image_button.configure(padx=10, pady=4, font=("Segoe UI", 9, "bold"))
        self.main_image_button.pack(side="left", padx=(12, 5))
        for text, command in [("+ Видео", self.choose_video), ("Камер", self.start_camera)]:
            button = self.button(bar, text, command, primary=text == "+ Видео")
            button.configure(padx=10, pady=4, font=("Segoe UI", 9, "bold"))
            button.pack(side="left", padx=3)
        self.fullscreen_button = self.button(bar, "⛶  F11", self.toggle_fullscreen)
        self.fullscreen_button.configure(padx=8, pady=4, font=("Segoe UI", 9, "bold"))
        self.fullscreen_button.pack(side="right")
        self.fill_button = self.button(bar, "Бүтнээр", self.toggle_fill)
        self.fill_button.configure(padx=8, pady=4, font=("Segoe UI", 9))
        self.fill_button.pack(side="right", padx=5)
        self.rotate_button = self.button(bar, "↶ 90°", self.rotate_source)
        self.rotate_button.configure(padx=8, pady=4, font=("Segoe UI", 9))
        self.rotate_button.pack(side="right")
        self.activity_label = self.label(preview_card, "", 10, self.accent, anchor="w", wraplength=650, justify="left", padx=12, pady=5)
        self.activity_label.configure(textvariable=self.status_text)
        self.activity_label.grid(row=1, column=0, sticky="ew")
        self.preview = tk.Canvas(preview_card, bg="#061014", bd=0, highlightthickness=0)
        self.preview.grid(row=2, column=0, sticky="nsew")
        self.preview.bind("<Configure>", self.redraw_preview)
        self.preview.bind("<Button-1>", lambda event: self.choose_images() if self.last_rgb is None else None)
        self.preview.bind("<Double-Button-1>", lambda event: self.toggle_fullscreen() if self.last_rgb is not None else None)
        controls = tk.Frame(preview_card, bg=PANEL, padx=10, pady=9)
        controls.grid(row=3, column=0, sticky="ew")
        self.pause_button = self.button(controls, "Ⅱ  Дүрс зогсоох", self.pause)
        self.pause_button.configure(padx=9, pady=8)
        self.pause_button.pack(side="left", padx=(0, 7))
        self.button(controls, "■", self.stop, danger=True).pack(side="left")
        self.replay_button = self.button(controls, "↻", self.replay_video)
        self.replay_button.pack(side="left", padx=5)
        self.button(controls, "▣  Хадгалах", self.snapshot).pack(side="right")
        actions = tk.Frame(preview_card, bg=PANEL, padx=10, pady=3)
        actions.grid(row=4, column=0, sticky="ew")
        self.frame_button = self.button(actions, "Энэ кадрыг таних", self.check_current_frame, primary=True)
        self.frame_button.configure(pady=6, padx=10)
        self.frame_button.pack(side="left", padx=(0, 8))
        self.review_button = self.button(actions, "Видеог шалгах", self.review_video)
        self.review_button.configure(pady=6, padx=10)
        self.review_button.pack(side="left")
        timeline = tk.Frame(preview_card, bg=PANEL, padx=12, pady=6)
        timeline.grid(row=5, column=0, sticky="ew")
        self.source_label = self.label(timeline, "Эх үүсвэр сонгоогүй", 8, MUTED, anchor="w")
        self.source_label.pack(fill="x", pady=(0, 3))
        position = tk.Frame(timeline, bg=PANEL)
        position.pack(fill="x")
        self.label(position, "", 9, TEXT, textvariable=self.video_time).pack(side="left")
        self.label(position, "", 9, self.accent, textvariable=self.analysis_time).pack(side="right")
        ttk.Progressbar(preview_card, variable=self.progress, maximum=1).grid(row=6, column=0, sticky="ew")

        latest = self.latest_panel = self.card(middle, width=250, padx=14, pady=14)
        latest.grid(row=0, column=1, sticky="nsew")
        latest.grid_propagate(False)
        latest.pack_propagate(False)
        self.label(latest, "СҮҮЛИЙН ТАНИЛТ", 9, MUTED, True).pack(anchor="w", pady=(0, 15))
        plate_card = tk.Frame(latest, bg="#f0f4fc", padx=11, pady=12)
        plate_card.pack(fill="x")
        self.label(plate_card, "MNG  •  МОНГОЛ УЛС", 8, "#2375b7", True).pack(anchor="w")
        self.plate_label = self.label(plate_card, "— — — —", 25, "#111c31", True)
        self.plate_label.pack(pady=(5, 0))
        self.rule_label = self.label(latest, "Дугаар хүлээж байна", 10, MUTED, True, wraplength=232, justify="left")
        self.rule_label.pack(anchor="w", pady=(14, 6))
        self.confidence_label = self.label(latest, "OCR  —     YOLO  —", 9, MUTED)
        self.confidence_label.pack(anchor="w")
        self.crop_label = self.label(latest, "Дугаарын хэсэг", 10, MUTED)
        self.crop_label.pack(fill="both", expand=True, pady=10)
        self.label(latest, "ЭНЭ ШАЛГАЛТЫН ДУГААРУУД", 8, MUTED, True).pack(anchor="w", pady=(6, 4))
        self.session_list = tk.Listbox(latest, height=4, bg=RAISED, fg=TEXT, relief="flat", bd=0,
                                      highlightthickness=0, selectbackground=BORDER, font=("Segoe UI", 10))
        self.session_list.pack(fill="x", pady=(0, 8))
        self.last_time_label = self.label(latest, "", 9, MUTED)
        self.last_time_label.pack(anchor="w")
        self.label(latest, "Ногоон — танигдсан\nЯгаан — одоо хязгаарлалттай", 8, MUTED, justify="left").pack(anchor="w", pady=(6, 0))

    def make_table(self, parent, height=10):
        frame = tk.Frame(parent, bg=PANEL)
        frame.pack(fill="both", expand=True)
        columns = ("time", "plate", "source", "confidence", "rule")
        table = ttk.Treeview(frame, columns=columns, show="headings", height=height, selectmode="browse")
        for key, heading, width in [("time", "ОГНОО / ЦАГ", 170), ("plate", "УЛСЫН ДУГААР", 135),
                                     ("source", "ЭХ ҮҮСВЭР", 170), ("confidence", "OCR", 65), ("rule", "ХУВААРИЙН ТӨЛӨВ", 265)]:
            table.heading(key, text=heading)
            table.column(key, width=width, minwidth=55, stretch=True)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=table.yview)
        table.configure(yscrollcommand=scrollbar.set)
        table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        table.tag_configure("restricted", foreground=RED)
        table.tag_configure("allowed", foreground=GREEN)
        table.bind("<Double-1>", lambda event: self.show_record(table))
        table.bind("<Return>", lambda event: self.show_record(table))
        return table

    def build_history(self, page):
        self.label(page, "Танигдсан дугаар, хадгалсан зураг болон цагийн бүртгэл", 11, MUTED).pack(anchor="w", pady=(0, 18))
        filters = self.card(page, padx=15, pady=14)
        filters.pack(fill="x", pady=(0, 12))
        self.label(filters, "Дугаар", 9, MUTED).grid(row=0, column=0, sticky="w")
        self.entry(filters, self.query_plate, 17).grid(row=1, column=0, ipady=8, padx=(0, 12))
        self.label(filters, "Огноо (хоосон = бүгд)", 9, MUTED).grid(row=0, column=1, sticky="w")
        self.entry(filters, self.query_date, 14).grid(row=1, column=1, ipady=8, padx=(0, 12))
        self.button(filters, "Хайх", self.refresh_records, primary=True).grid(row=1, column=2, padx=(0, 8))
        self.button(filters, "Бүгд", self.clear_filters).grid(row=1, column=3)
        ttk.Checkbutton(filters, text="Хязгаарлалттай", variable=self.only_restricted, command=self.refresh_records).grid(row=1, column=4, padx=12)
        self.button(filters, "CSV экспорт", self.export_records).grid(row=1, column=5)
        self.history_info = self.label(page, "", 10, self.accent)
        self.history_info.pack(anchor="w", pady=(0, 10))
        self.history_table = self.make_table(page, height=13)
        self.label(page, "Мөрөн дээр давхар товшиж зураг, дугаарын хэсэг болон дэлгэрэнгүй мэдээллийг нээнэ.", 9, MUTED).pack(anchor="w", pady=10)
        self.label(page, "Камерын бүртгэлд илрээгүй нь тухайн автомашин огт яваагүй гэсэн баталгаа болохгүй.", 9, MUTED).pack(anchor="w")

    def build_rules(self, page):
        self.label(page, "Дугаарын 4 цифрээ оруулаад хязгаарлах болон зорчих өдрүүдээ харна.", 11, MUTED).pack(anchor="w", pady=(0, 18))
        checker = self.card(page, padx=20, pady=18)
        checker.pack(fill="x", pady=(0, 14))
        self.rule_plate = tk.StringVar(value="")
        self.rule_date = tk.StringVar(value=now_ub().strftime("%Y-%m-%d"))
        self.rule_time = tk.StringVar(value=now_ub().strftime("%H:%M"))
        for col, (title, variable, width) in enumerate([("Дугаар: 1113 эсвэл 1113УБА", self.rule_plate, 23), ("Шалгах огноо YYYY-MM-DD", self.rule_date, 18), ("Шалгах цаг HH:MM", self.rule_time, 12)]):
            self.label(checker, title, 9, MUTED).grid(row=0, column=col, sticky="w")
            self.entry(checker, variable, width).grid(row=1, column=col, ipady=9, padx=(0, 14), pady=(6, 0))
        self.button(checker, "Явах өдрийг харуулах", self.check_rule, primary=True).grid(row=1, column=3, pady=(6, 0))
        result = self.card(page, padx=20, pady=20)
        result.pack(fill="x", pady=(0, 14))
        self.rule_result_label = self.label(result, "Дугаарын 4 цифрээ оруулна уу", 17, self.accent, True, wraplength=930, justify="left")
        self.rule_result_label.pack(anchor="w")
        self.rule_days_label = self.label(result, "Жишээ: 1113 → сүүлийн цифр 3", 11, MUTED, wraplength=930, justify="left")
        self.rule_days_label.pack(anchor="w", pady=(8, 0))
        self.rule_date_result = self.label(result, "", 11, TEXT, wraplength=930, justify="left")
        self.rule_date_result.pack(anchor="w", pady=(12, 0))
        self.rule_detail_label = self.label(result, "", 10, MUTED, wraplength=900, justify="left")
        self.rule_detail_label.pack(anchor="w", pady=(10, 0))
        self.rule_seen_label = self.label(result, "", 10, TEXT)
        self.rule_seen_label.pack(anchor="w", pady=(8, 0))
        schedule = self.card(page, padx=20, pady=16)
        schedule.pack(fill="x")
        self.label(schedule, "7 ХОНОГИЙН ХУВААРЬ", 9, MUTED, True).pack(anchor="w", pady=(0, 10))
        self.schedule_labels = []
        weekdays = tk.Frame(schedule, bg=PANEL)
        weekdays.pack(fill="x")
        for index, day in enumerate(DAY_NAMES):
            cell = tk.Frame(weekdays, bg=RAISED, padx=8, pady=12)
            cell.pack(side="left", expand=True, fill="x", padx=3)
            self.label(cell, day, 9, MUTED).pack()
            label = self.label(cell, "", 15, self.accent, True)
            label.pack(pady=(5, 0))
            self.schedule_labels.append(label)
        self.schedule_note = self.label(schedule, "", 10, MUTED)
        self.schedule_note.pack(anchor="w", pady=(12, 0))
        self.label(page, "Төлөв нь сонгосон бүс, цаг, хуваариас хамаарна. Түр зохицуулалт, баярын өдөр, чөлөөлөх дугаарыг Тохиргооноос шинэчилнэ.\nИмпортолсон зураг/видеонд боловсруулах үеийн огноог ашиглана; видео дахь байрлалыг тусад нь секундээр хадгална.",
                   10, MUTED, wraplength=1000, justify="left").pack(anchor="w", pady=18)
        self.update_schedule()
        self.rule_plate.trace_add("write", lambda *_: self.update_restriction_preview())

    def scroll_page(self, page):
        canvas = tk.Canvas(page, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=BG)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        # Scroll only this page, without stealing mouse-wheel input elsewhere.
        self.bind("<MouseWheel>", lambda event: canvas.yview_scroll(-int(event.delta / 120), "units") if self.current_page == "settings" else None)
        return inner

    def build_settings(self, page):
        inner = self.scroll_page(page)
        self.label(inner, "Өөрчлөлтийг хадгалсны дараа дараагийн танилтад үйлчилнэ.", 10, MUTED).pack(anchor="w", pady=(0, 16))
        video = self.card(inner, padx=20, pady=18)
        video.pack(fill="x", pady=(0, 14))
        self.label(video, "ВИДЕО БА ТАНИХ ХУРД", 10, self.accent, True).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 16))
        self.recognition_mode_var = tk.StringVar(value=next((k for k, v in RECOGNITION_MODES.items()
                                                               if v == self.settings.get("recognition_mode")), "Тэнцвэртэй"))
        self.label(video, "Таних горим", 10, MUTED).grid(row=1, column=0, sticky="w", padx=(0, 20), pady=7)
        ttk.Combobox(video, textvariable=self.recognition_mode_var, values=list(RECOGNITION_MODES),
                     state="readonly", width=18).grid(row=1, column=1, sticky="w")
        self.video_stride_var = tk.StringVar(value=str(self.settings["frame_stride"]))
        self.label(video, "Хэдэн кадр тутамд шалгах (1–120)", 10, MUTED).grid(row=2, column=0, sticky="w", padx=(0, 20), pady=7)
        ttk.Spinbox(video, from_=1, to=120, textvariable=self.video_stride_var, width=8).grid(row=2, column=1, sticky="w")
        self.playback_speed_var = tk.StringVar(value=str(self.settings.get("playback_speed", 1.0)))
        self.label(video, "Дүрсний хурд", 10, MUTED).grid(row=3, column=0, sticky="w", padx=(0, 20), pady=7)
        ttk.Combobox(video, textvariable=self.playback_speed_var, values=["0.25", "0.5", "1.0", "1.5", "2.0", "4.0"],
                     state="readonly", width=8).grid(row=3, column=1, sticky="w")
        self.button(video, "Хэрэглэх", self.apply_video_controls).grid(row=4, column=1, sticky="w", pady=(10, 0))
        ai = self.card(inner, padx=20, pady=18)
        ai.pack(fill="x", pady=(0, 14))
        self.label(ai, "AI БА ДҮРС БОЛОВСРУУЛАЛТ", 10, self.accent, True).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 16))
        self.setting_vars = {"frame_stride": self.video_stride_var}
        values = [
            ("detection_confidence", "YOLO босго (0.05–0.95)", str(self.settings["detection_confidence"])),
            ("ocr_confidence", "OCR босго (0.05–0.95)", str(self.settings["ocr_confidence"])),
            ("cooldown_seconds", "Давтан хадгалах зай, сек (1–3600)", str(self.settings["cooldown_seconds"])),
            ("confirmation_frames", "Видео/камер батлах кадр (1–10)", str(self.settings["confirmation_frames"])),
            ("video_min_confidence", "Видео/камерын OCR доод босго (0.05–0.99)", str(self.settings.get("video_min_confidence", .65))),
            ("neighbor_frames", "Зогсоосон кадрын нэмэлт хайлт (0–10 кадр)", str(self.settings.get("neighbor_frames", 2))),
        ]
        for row, (key, label, value) in enumerate(values, start=1):
            var = tk.StringVar(value=value)
            self.setting_vars[key] = var
            self.label(ai, label, 10, MUTED).grid(row=row, column=0, sticky="w", pady=7, padx=(0, 20))
            self.entry(ai, var, 10).grid(row=row, column=1, sticky="w", ipady=6)
        self.model_var = tk.StringVar(value=self.settings.get("model_path", ""))
        self.label(ai, "Өөрийн YOLO .pt (хоосон = бэлэн загвар)", 10, MUTED).grid(row=8, column=0, sticky="w", pady=10, padx=(0, 15))
        self.entry(ai, self.model_var, 37).grid(row=8, column=1, columnspan=2, sticky="ew", ipady=6)
        self.button(ai, "Сонгох", self.choose_model).grid(row=8, column=3, padx=8)
        self.video_var = tk.BooleanVar(value=self.settings.get("save_video", False))
        ttk.Checkbutton(ai, text="Боловсруулсан видеог MP4 болгон хадгалах", variable=self.video_var).grid(row=9, column=0, columnspan=3, sticky="w", pady=10)
        self.device_var = tk.StringVar(value=self.settings.get("device", "cpu"))
        self.label(ai, "AI төхөөрөмж", 10, MUTED).grid(row=10, column=0, sticky="w", pady=6)
        ttk.Combobox(ai, textvariable=self.device_var, values=["cpu", "auto"], state="readonly", width=12).grid(row=10, column=1, sticky="w")
        self.accent_var = tk.StringVar(value=self.settings.get("accent", "Цэнхэр"))
        self.label(ai, "Товчны үндсэн өнгө", 10, MUTED).grid(row=11, column=0, sticky="w", pady=10)
        ttk.Combobox(ai, textvariable=self.accent_var, values=list(ACCENTS), state="readonly", width=16).grid(row=11, column=1, sticky="w")
        self.backend_var = tk.StringVar(value=self.settings.get("camera_backend", "auto"))
        self.label(ai, "Камер нээх горим", 10, MUTED).grid(row=12, column=0, sticky="w", pady=10)
        ttk.Combobox(ai, textvariable=self.backend_var, values=["auto", "dshow", "msmf"], state="readonly", width=16).grid(row=12, column=1, sticky="w")
        self.label(ai, "Дээрх видео тохиргооны Хэрэглэх товчоор таних горим, кадрын зай, дүрсний хурдыг ажиллаж байхад өөрчилнө.\n"
                       "Кадрын зай 1 = бүх кадр; 15 = 30 FPS видеонд 0.5 секунд тутам. Ихэсгэхэд хурдсаж, богино илрэлт алгасаж болно.\n"
                       "Батлах кадрыг нэмэхэд зөрүүтэй уншилт багасна. Нэмэлт хайлт 0 = зөвхөн сонгосон кадрыг шалгана.",
                   9, MUTED, justify="left", wraplength=800).grid(row=13, column=0, columnspan=4, sticky="w", pady=12)
        self.adaptive_var = tk.BooleanVar(value=self.settings.get("adaptive_sampling", True))
        ttk.Checkbutton(ai, text="Дугаар илэрвэл ойр кадруудаар давтан батлах (илүү их кадр шалгана)",
                        variable=self.adaptive_var).grid(row=14, column=0, columnspan=4, sticky="w", pady=8)

        rule_card = self.card(inner, padx=20, pady=18)
        rule_card.pack(fill="x", pady=(0, 14))
        self.label(rule_card, "ХӨДӨЛГӨӨНИЙ ХЯЗГААРЛАЛТ", 10, self.accent, True).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 16))
        rules = self.settings["rules"]
        self.mode_var = tk.StringVar(value=next((k for k, v in MODES.items() if v == rules["mode"]), list(MODES)[0]))
        self.label(rule_card, "Горим", 10, MUTED).grid(row=1, column=0, sticky="w", pady=8)
        ttk.Combobox(rule_card, textvariable=self.mode_var, values=list(MODES), state="readonly", width=28).grid(row=1, column=1, sticky="w")
        self.start_var = tk.StringVar(value=rules["start"])
        self.end_var = tk.StringVar(value=rules["end"])
        self.label(rule_card, "Эхлэх / дуусах цаг (HH:MM)", 10, MUTED).grid(row=2, column=0, sticky="w", pady=8, padx=(0, 20))
        time_row = tk.Frame(rule_card, bg=PANEL)
        time_row.grid(row=2, column=1, sticky="w")
        self.entry(time_row, self.start_var, 8).pack(side="left", ipady=6)
        self.label(time_row, "  —  ", 10, MUTED).pack(side="left")
        self.entry(time_row, self.end_var, 8).pack(side="left", ipady=6)
        self.zone_var = tk.BooleanVar(value=rules["in_zone"])
        self.parity_var = tk.BooleanVar(value=rules.get("even_on_even_date", True))
        ttk.Checkbutton(rule_card, text="Камер / эх үүсвэр хязгаарлалтын бүсэд байгаа", variable=self.zone_var).grid(row=3, column=0, columnspan=3, sticky="w", pady=8)
        ttk.Checkbutton(rule_card, text="Тэгш огноонд тэгш дугаар зорчино (тэгш/сондгой горим)", variable=self.parity_var).grid(row=4, column=0, columnspan=3, sticky="w", pady=8)
        self.weekday_vars = []
        for index, name in enumerate(DAY_NAMES):
            var = tk.StringVar(value=",".join(map(str, rules["weekdays"].get(str(index), []))))
            self.weekday_vars.append(var)
            self.label(rule_card, f"{name} • хязгаарлах сүүлийн цифр", 10, MUTED).grid(row=5 + index, column=0, sticky="w", pady=6)
            self.entry(rule_card, var, 18).grid(row=5 + index, column=1, sticky="w", ipady=6)
        self.holiday_var = tk.StringVar(value=", ".join(rules.get("holidays", [])))
        self.exempt_var = tk.StringVar(value=", ".join(rules.get("exempt_plates", [])))
        self.label(rule_card, "Чөлөөлөх огноо (таслалаар)", 10, MUTED).grid(row=12, column=0, sticky="w", pady=8)
        self.entry(rule_card, self.holiday_var, 45).grid(row=12, column=1, columnspan=3, sticky="ew", ipady=6)
        self.label(rule_card, "Чөлөөлөх дугаар (таслалаар)", 10, MUTED).grid(row=13, column=0, sticky="w", pady=8)
        self.entry(rule_card, self.exempt_var, 45).grid(row=13, column=1, columnspan=3, sticky="ew", ipady=6)
        footer = tk.Frame(inner, bg=BG)
        footer.pack(fill="x", pady=(0, 18))
        self.button(footer, "✓  Тохиргоо хадгалах", self.apply_settings, primary=True).pack(side="left", padx=(0, 12))
        self.button(footer, "AI загвар бэлтгэх / шалгах", self.prepare_ai).pack(side="left")
        self.button(footer, "Заавар нээх", lambda: self.open_path(APP_DIR / "README.md")).pack(side="right")

    def show_page(self, key):
        if self.fullscreen and key != "monitor":
            self.toggle_fullscreen()
        self.current_page = key
        self.pages[key].tkraise()
        self.status_bar.grid_remove() if key == "monitor" else self.status_bar.grid()
        titles = {"monitor": "Хөдөлгөөний ухаалаг хяналт", "history": "Илрэлтийн түүх", "rules": "Дугаарын хязгаарлалт", "settings": "Өөрийнхөөрөө тохируул"}
        self.page_title.configure(text=titles[key])
        for name, button in self.nav.items():
            button.configure(bg="#213651" if key == name else RAISED, fg=self.accent if key == name else TEXT)
        if key == "history":
            self.refresh_records()

    def escape(self, event=None):
        if self.fullscreen:
            self.toggle_fullscreen()
        else:
            self.stop()

    def toggle_fullscreen(self):
        if not self.fullscreen:
            self.show_page("monitor")
        self.fullscreen = not self.fullscreen
        self.attributes("-fullscreen", self.fullscreen)
        for widget in (self.sidebar, self.header, self.stats_panel, self.latest_panel):
            widget.grid_remove() if self.fullscreen else widget.grid()
        self.shell.configure(padx=8 if self.fullscreen else 20, pady=8 if self.fullscreen else 16)
        self.preview_card.grid_configure(padx=0 if self.fullscreen else (0, 14))
        self.fullscreen_button.configure(text="⛶  Буцах · Esc" if self.fullscreen else "⛶  F11")
        self._preview_dirty = True

    def toggle_fill(self):
        self.preview_fill = not self.preview_fill
        self.fill_button.configure(text="Дүүргэх" if self.preview_fill else "Бүтнээр")
        self.redraw_preview()

    @staticmethod
    def format_time(seconds):
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"

    def replay_video(self):
        if self.video_source:
            self.start_source("video", self.video_source)
        else:
            self.choose_video()

    def rotate_source(self):
        if self.active_source is None:
            return
        rotation = (int(self.worker.settings.get("_rotation", 0)) + 270) % 360
        self.start_source(self.active_kind, self.active_source, rotation=rotation)

    def review_video(self):
        if not self.video_source:
            self.status_text.set("Эхлээд + Видео товчоор бичлэгээ сонгоно уу.")
            return
        if self.apply_video_controls():
            self.start_source("video", self.video_source, review=True)

    def apply_video_controls(self):
        try:
            stride = int(self.video_stride_var.get())
            speed = float(self.playback_speed_var.get())
            mode = RECOGNITION_MODES[self.recognition_mode_var.get()]
            if not 1 <= stride <= 120 or not .25 <= speed <= 4:
                raise ValueError()
        except (ValueError, KeyError):
            self.status_text.set("Кадрын зай 1–120, дүрсний хурд 0.25–4 байна.")
            return False
        values = dict(frame_stride=stride, playback_speed=speed, recognition_mode=mode)
        self.settings.update(values)
        self.worker.settings.update(values)
        self.setting_vars["frame_stride"].set(str(stride))
        save_settings(self.settings)
        self.status_text.set(f"Хэрэглэлээ • {self.recognition_mode_var.get()} • {stride} кадр тутам • дүрс {speed:g}×. Дараагийн кадраас үйлчилнэ.")
        return True

    def check_current_frame(self):
        if self.last_rgb is None or self.frame_check_busy or self.pending_source is not None or self.preparing:
            return
        original = self._frame_original if self._frame_original is not None else self.last_rgb
        self.worker.pause_event.set()
        self.pause_button.configure(text="▶  Үргэлжлүүлэх")
        # Discard a queued playback frame so the exact visible picture stays put.
        while not self.worker.frames.empty():
            try:
                self.worker.frames.get_nowait()
            except queue.Empty:
                break
        while not self.worker.previews.empty():
            self.worker.previews.get_nowait()
        source = self.active_source
        if self.active_kind == "image" and isinstance(source, (list, tuple)):
            source = self._frame_source or (source[-1] if source else "Зураг")
        seconds = self._frame_seconds if self._frame_seconds is not None else self.playback_seconds
        self.frame_check_busy = self.worker.check_frame(original, self.active_kind or "image", str(source or "Кадр"),
                                                        seconds)
        if self.frame_check_busy:
            self.frame_button.configure(state="disabled", text="Кадр шалгаж байна...")
            self.status_text.set("Зогсоосон кадрыг томруулж, тодруулж уншиж байна...")

    def show_plate_crop(self, rgb):
        image = Image.fromarray(rgb)
        scale = min(230 / image.width, 95 / image.height)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
        self.crop_photo = ImageTk.PhotoImage(image)
        self.crop_label.configure(image=self.crop_photo, text="")

    def show_frame_result(self, result):
        self.last_rgb = result["rgb"]
        self._frame_original = result["original"]
        self._requested_original = result.get("requested_original", result["original"])
        self._frame_pinned = True
        self._preview_dirty = True
        self.playback_seconds = result["seconds"]
        self._frame_seconds = result["seconds"]
        self.video_time.set(("Ойролцоох баталгаажсан кадр • " if result.get("from_neighbor") else "Зогсоосон кадр • ") + self.format_time(result["seconds"]))
        found = result["detections"]
        readable = [d for d in found if d.plate]
        if found:
            best = max(found, key=lambda d: (bool(d.plate), d.ocr_confidence, d.detector_confidence))
            x1, y1, x2, y2 = best.bbox
            crop = self._frame_original[max(0, y1):y2, max(0, x1):x2]
            if crop.size:
                self.show_plate_crop(crop)
            self.last_detection = best
            self.plate_label.configure(text=display_plate(best.plate) if best.plate else "— — — —")
            self.confidence_label.configure(text=self.confidence_text(best))
            rule = evaluate(best.plate, self.settings["rules"]) if best.plate else None
            self.rule_label.configure(text=rule.label if rule else "Үсэг, тоо бүрэн уншигдсангүй", fg=GREEN if rule and not rule.restricted_now else RED if rule else AMBER)
        else:
            self.plate_label.configure(text="— — — —")
            self.last_detection = None
            self.confidence_label.configure(text="OCR  —     YOLO  —")
            self.rule_label.configure(text="Энэ кадрт дугаарын хэсэг илрээгүй", fg=AMBER)
            self.crop_label.configure(image="", text="Дугаар илрээгүй")
        pending = len(found) - len(readable)
        self.status_text.set((f"{len(readable)} дугаар баталгаажлаа • {pending} дугаар бүрэн тодроогүй." if pending else
                              f"Энэ кадраас {len(readable)} дугаар баталгаажлаа.") if readable else
                             "Дугаар бүрэн уншигдсангүй. Илүү ойр, тод харагдах кадрыг сонгоод дахин шалгана уу.")

    @staticmethod
    def confidence_text(detection):
        detector = f"{detection.detector_confidence:.0%}" if detection.detector_confidence else "—"
        return f"OCR  {detection.ocr_confidence:.0%}     YOLO  {detector}"

    def choose_images(self):
        logging.info("Image import button clicked")
        self.lift()
        paths = filedialog.askopenfilenames(parent=self, title="Нэг эсвэл олон зураг сонгох", initialdir=self.last_import_dir,
                                           filetypes=[("Зураг", tuple("*" + ext for ext in IMAGE_EXTENSIONS)), ("Бүх файл", "*.*")])
        if paths:
            if isinstance(paths, str):
                paths = self.tk.splitlist(paths)
            self.last_import_dir = str(Path(paths[0]).parent)
            logging.info("Image files selected: %s", paths)
            self.start_source("image", list(paths))
        else:
            logging.info("Image import cancelled")

    def choose_video(self):
        logging.info("Video import button clicked")
        self.lift()
        path = filedialog.askopenfilename(parent=self, title="Видео импортлох", initialdir=self.last_import_dir,
                                         filetypes=[("Видео", tuple("*" + ext for ext in VIDEO_EXTENSIONS)), ("Бүх файл", "*.*")])
        if path:
            self.last_import_dir = str(Path(path).parent)
            if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
                self.display_error("Энэ нь зургийн файл байна. «+ Зураг» товчоор оруулна уу; видео шалгахад MP4, AVI, MOV эсвэл MKV сонгоно уу.")
                return
            self.start_source("video", path)

    def open_import_path(self):
        path = Path(self.import_path_var.get().strip().strip('"'))
        if not path.is_file():
            self.display_error("Файл олдсонгүй. Файлын бүтэн замыг оруулах эсвэл Зураг импортлох товчоор сонгоно уу.")
            return
        self.start_source("video", str(path)) if path.suffix.lower() in VIDEO_EXTENSIONS else self.start_source("image", [str(path)])

    def start_camera(self):
        logging.info("Camera button clicked: %s", self.camera_var.get())
        try:
            index = "auto" if self.camera_var.get().lower() in ("auto", "авто") else int(self.camera_var.get())
            if index != "auto" and not 0 <= index <= 20:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Камер", "Камерын индекс 0–20 хооронд бүхэл тоо байна.", parent=self)
            return
        self.settings["camera_index"] = index
        save_settings(self.settings)
        self.start_source("camera", index)

    def start_source(self, kind, source, rotation=None, review=None):
        if self.worker.alive or self.preparing:
            self.pending_source = (kind, source, rotation, review)
            self.worker.stop()
            self.status_text.set("Эх үүсвэр сольж байна... Өмнөх AI үйлдэл дуусмагц сонгосон файл / камер нээгдэнэ.")
            self.show_page("monitor")
            return True
        self.pending_source = None
        if kind != self.active_kind or source != self.active_source:
            self.worker.settings["_rotation"] = 0
            self.worker.settings["_video_review"] = False
        if rotation is not None:
            self.worker.settings["_rotation"] = int(rotation) % 360
        if review is not None:
            self.worker.settings["_video_review"] = bool(review)
        self.active_kind = kind
        self.active_source = source
        if kind == "video":
            self.video_source = str(source)
        self.playback_seconds = self.video_duration = 0
        self._frame_original = None
        self._frame_source = self._frame_seconds = None
        self._requested_original = None
        self._frame_pinned = False
        self.frame_check_busy = False
        self.frame_button.configure(state="normal", text="Энэ кадрыг таних")
        self.session_records.clear()
        self.session_list.delete(0, "end")
        self.video_time.set("Видео нээж байна..." if kind == "video" else "Шууд камер" if kind == "camera" else "Зураг")
        self.analysis_time.set("AI бэлтгэж байна...")
        self.preview_displayed = False
        self.camera_overlay = ([], 0)
        self.last_error = ""
        self.activity_label.configure(fg=self.accent)
        self.last_rgb = None
        self.last_detection = None
        self.plate_label.configure(text="— — — —")
        self.rule_label.configure(text="Дугаар хүлээж байна", fg=MUTED)
        self.confidence_label.configure(text="OCR  —     YOLO  —")
        self.crop_label.configure(image="", text="Дугаарын хэсэг")
        self.last_time_label.configure(text="")
        self.show_page("monitor")
        self.progress.set(0)
        self.pause_button.configure(text="Ⅱ  Дүрс зогсоох")
        label = f"Камер {source}" if kind == "camera" else f"{len(source)} зураг" if kind == "image" else Path(source).name
        self.source_label.configure(text=label[:55])
        self.status_text.set("Камерын дүрс нээж байна..." if kind == "camera" else "Сонгосон файлыг нээж байна...")
        self.redraw_preview()
        return self.worker.start(kind, source)

    def pause(self):
        if not self.worker.alive and not self.worker.pause_event.is_set():
            return
        paused = self.worker.pause_event.is_set()
        self.worker.pause_event.clear() if paused else self.worker.pause_event.set()
        if paused:
            self._frame_pinned = False
        self.pause_button.configure(text="Ⅱ  Дүрс зогсоох" if paused else "▶  Үргэлжлүүлэх")
        self.status_text.set("Дүрс үргэлжилж байна" if paused else
                             "Дүрс зогссон • AI шалгалт үргэлжилнэ. Энэ кадрыг таних товчоор тусад нь шалгана." if self.active_kind == "video" else "Түр зогссон")

    def stop(self):
        self.pending_source = None
        if self.worker.alive or self.preparing:
            self.worker.stop()
            self.status_text.set("Зогсоож байна... Одоогийн AI үйлдэл дуусахыг хүлээнэ үү.")

    def snapshot(self):
        if self.last_rgb is None:
            self.status_text.set("Зураг хадгалахын өмнө зураг, видео эсвэл камерын дүрс нээнэ үү.")
            return
        folder = DATA_DIR / "snapshots"
        folder.mkdir(exist_ok=True)
        path = folder / (now_ub().strftime("%Y%m%d_%H%M%S_%f") + ".jpg")
        Image.fromarray(self.last_rgb).save(path, quality=95)
        self.status_text.set(f"Зураг хадгаллаа: {path.name}")

    def redraw_preview(self, event=None):
        self._preview_dirty = True
        if self.closing or time.monotonic() < self._preview_retry_at:
            return False
        try:
            self._render_preview()
        except (MemoryError, tk.TclError):
            # A failed allocation must not kill the only callback consuming frames.
            # Keep the last good image and retry at a smaller size, including when
            # an imported photo has no more frames queued by its worker.
            if not self._preview_failed:
                logging.exception("Preview allocation failed; retrying with a smaller image")
            self._preview_failed = True
            self._preview_limit = max(240, self._preview_limit // 2)
            self._preview_retry_at = time.monotonic() + .5
            return False
        self._preview_dirty = False
        self._preview_retry_at = 0
        if self._preview_failed:
            logging.info("GUI preview recovered at limit %s", self._preview_limit)
        self._preview_failed = False
        return True

    def _render_preview(self):
        width = max(100, self.preview.winfo_width())
        height = max(100, self.preview.winfo_height())
        if self.last_rgb is not None:
            import cv2
            # Resize the array before allocating a Pillow/Tk image. In particular,
            # never create a full-resolution Tk copy of a large imported photo.
            frame_height, frame_width = self.last_rgb.shape[:2]
            view = self.last_rgb
            offset_x = offset_y = 0
            if self.preview_fill:
                # Crop before resizing to avoid a huge intermediate allocation.
                ratio = width / height
                if frame_width / frame_height > ratio:
                    crop_width = max(1, round(frame_height * ratio))
                    offset_x = (frame_width - crop_width) // 2
                    view = view[:, offset_x:offset_x + crop_width]
                else:
                    crop_height = max(1, round(frame_width / ratio))
                    offset_y = (frame_height - crop_height) // 2
                    view = view[offset_y:offset_y + crop_height]
            view_height, view_width = view.shape[:2]
            scale = min(width / view_width, height / view_height,
                        self._preview_limit / max(view_width, view_height))
            size = (max(1, round(view_width * scale)), max(1, round(view_height * scale)))
            if self.preview_fill:
                display_scale = min(1, self._preview_limit / max(width, height))
                size = (max(1, round(width * display_scale)), max(1, round(height * display_scale)))
            rgb = cv2.resize(view, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR) if size != (view_width, view_height) else view
            image = Image.fromarray(rgb)
            detections, timestamp = self.camera_overlay
            if self.active_kind == "camera" and not self._frame_pinned and time.monotonic() - timestamp < 2.5:
                draw = ImageDraw.Draw(image)
                ratio = image.width / view_width
                font = overlay_font(15)
                for detection in detections:
                    x1, y1, x2, y2 = detection.bbox
                    box = (int((x1 - offset_x) * ratio), int((y1 - offset_y) * ratio),
                           int((x2 - offset_x) * ratio), int((y2 - offset_y) * ratio))
                    rule = evaluate(detection.plate, self.settings["rules"])
                    color = RED if rule.restricted_now else GREEN if detection.plate else AMBER
                    draw.rectangle(box, outline=color, width=2)
                    text = display_plate(detection.plate) if detection.plate else "Дугаар"
                    draw.text((box[0] + 2, max(0, box[1] - 20)), text, fill=color, font=font, stroke_width=1, stroke_fill=BG)
            if self.preview_photo is not None and (self.preview_photo.width(), self.preview_photo.height()) == image.size:
                self.preview_photo.paste(image)
            else:
                # Allocate successfully before replacing the last visible image.
                photo = ImageTk.PhotoImage(image, master=self)
                if self._preview_item is None:
                    self.preview.delete("placeholder")
                    self._preview_item = self.preview.create_image(width / 2, height / 2, image=photo)
                else:
                    self.preview.itemconfigure(self._preview_item, image=photo)
                self.preview_photo = photo
            self.preview.coords(self._preview_item, width / 2, height / 2)
            return
        self.preview.delete("all")
        self._preview_item = None
        self.preview_photo = None
        cx, cy = width / 2, height / 2
        for y in range(0, height, 8):
            mix = y / max(1, height)
            color = "#%02x%02x%02x" % (int(9 + 5 * mix), int(28 + 9 * mix), int(35 + 8 * mix))
            self.preview.create_rectangle(0, y, width, y + 8, fill=color, outline="", tags="placeholder")
        for x in range(0, width, 48):
            self.preview.create_line(x, 0, x, height, fill="#17303a", tags="placeholder")
        for y in range(0, height, 48):
            self.preview.create_line(0, y, width, y, fill="#17303a", tags="placeholder")
        self.preview.create_oval(cx - 45, cy - 72, cx + 45, cy + 18, outline=BORDER, width=2, tags="placeholder")
        self.preview.create_rectangle(cx - 24, cy - 45, cx + 24, cy - 12, outline=self.accent, width=2, tags="placeholder")
        self.preview.create_line(cx - 8, cy - 45, cx + 8, cy - 45, fill=self.accent, width=4, tags="placeholder")
        self.preview.create_text(cx, cy + 49, text="Дүрсээ нээгээд шалгаж эхлээрэй", fill=TEXT, font=("Segoe UI", 16, "bold"), tags="placeholder")
        self.preview.create_text(cx, cy + 79, text="+ Видео  ·  + Зураг  ·  Камер", fill=MUTED, font=("Segoe UI", 11), tags="placeholder")
        for x, y, dx, dy in [(22, 22, 1, 1), (width - 22, 22, -1, 1), (22, height - 22, 1, -1), (width - 22, height - 22, -1, -1)]:
            self.preview.create_line(x + dx * 20, y, x, y, x, y + dy * 20, fill="#294463", width=2, tags="placeholder")

    def fill_table(self, table, records):
        table.delete(*table.get_children())
        for record in records:
            timestamp = record["detected_at"].replace("T", " ")[:19]
            source = f"Камер {self.camera_var.get()}" if record["source_kind"] == "camera" else Path(record["source"]).name
            if record["source_kind"] == "camera":
                source = record["source"]
            if record.get("video_seconds"):
                source += f" @{float(record['video_seconds']):.1f}s"
            table.insert("", "end", iid=record["id"], values=(timestamp, display_plate(record["plate"]), source,
                         f"{float(record['ocr_confidence']):.0%}", record["rule_label"]), tags=(record["rule_code"],))

    def refresh_records(self):
        records = self.store.records()
        self.record_index = {r["id"]: r for r in records}
        today = [r for r in records if r["detected_at"].startswith(now_ub().date().isoformat())]
        self.stat_vars[0].set(str(len(today)))
        self.stat_vars[1].set(str(len({r["plate"] for r in today})))
        self.stat_vars[2].set(str(sum(r["rule_code"] == "restricted" for r in today)))
        day = self.query_date.get().strip()
        if day:
            try:
                day = datetime.strptime(day, "%Y-%m-%d").date().isoformat()
            except ValueError:
                self.history_info.configure(text="Огноог YYYY-MM-DD хэлбэрээр оруулна уу.", fg=RED)
                return
        query = clean_text(self.query_plate.get())
        if len(query) == 7:
            query = normalize_plate(query) or query
        self.filtered_records = self.store.query(query, day, self.only_restricted.get())
        self.fill_table(self.history_table, self.filtered_records[:5000])
        self.history_info.configure(text=f"{len(self.filtered_records)} бүртгэл  •  {len({r['plate'] for r in self.filtered_records})} ялгаатай дугаар  •  зураг нээхэд давхар товшино", fg=self.accent)

    def clear_filters(self):
        self.query_plate.set("")
        self.query_date.set("")
        self.only_restricted.set(False)
        self.refresh_records()

    def export_records(self):
        path = filedialog.asksaveasfilename(parent=self, title="Бүртгэл экспортлох", defaultextension=".csv", initialfile=f"platewatch_{now_ub():%Y%m%d}.csv", filetypes=[("Excel CSV", "*.csv")])
        if path:
            try:
                self.store.export(path, self.filtered_records)
            except (OSError, ValueError) as exc:
                self.last_error = str(exc)
                self.status_text.set("Экспорт хадгалж чадсангүй • " + str(exc))
                logging.error("Export failed: %s", exc)
                return False
            self.status_text.set(f"{len(self.filtered_records)} бүртгэл экспортоллоо: {Path(path).name}")
            return True

    def show_record(self, table):
        selected = table.selection()
        if not selected:
            return
        record = self.record_index.get(selected[0])
        if not record:
            return
        popup = tk.Toplevel(self)
        popup.title(display_plate(record["plate"]) + " • Бүртгэлийн дэлгэрэнгүй")
        popup.geometry("920x720")
        popup.configure(bg=BG)
        self.label(popup, display_plate(record["plate"]), 27, self.accent, True).pack(pady=(15, 4))
        self.label(popup, f"{record['detected_at']}  •  {record['rule_label']}", 10, TEXT).pack(pady=4)
        try:
            image_path = self.store.resolve(record["image_path"])
            if image_path.is_file():
                with Image.open(image_path) as original:
                    image = original.copy()
                image.thumbnail((865, 430))
                photo = ImageTk.PhotoImage(image)
                label = self.label(popup)
                label.configure(image=photo)
                label.image = photo
                label.pack(pady=10)
            else:
                self.label(popup, "Хадгалсан зураг олдсонгүй. Бүртгэлийн мэдээлэл доор байна.", 10, AMBER).pack(pady=12)
        except (OSError, ValueError):
            self.label(popup, "Зургийн файл нээгдэхгүй байна. Бүртгэлийн мэдээлэл доор байна.", 10, AMBER).pack(pady=12)
        detector = f"{float(record['detector_confidence']):.1%}" if float(record['detector_confidence']) else "—"
        details = f"Эх үүсвэр: {record['source']}\nYOLO: {detector}   OCR: {float(record['ocr_confidence']):.1%}"
        if record["source_kind"] == "video":
            details += f"   Видео дахь байрлал: {record['video_seconds']} сек"
        self.label(popup, details, 10, MUTED, wraplength=850, justify="left").pack(pady=5)
        buttons = tk.Frame(popup, bg=BG)
        buttons.pack(pady=12)
        for title, key in [("Бүтэн зураг", "image_path"), ("Дугаарын хэсэг", "crop_path"), ("JSON мэдээлэл", "metadata_path")]:
            self.button(buttons, title, lambda k=key: self.open_path(self.store.resolve(record[k]))).pack(side="left", padx=5)

    def update_restriction_preview(self):
        plate = rule_plate(self.rule_plate.get())
        self.last_rule_result = None
        self.rule_date_result.configure(text="")
        self.rule_detail_label.configure(text="")
        self.rule_seen_label.configure(text="")
        if not plate:
            self.rule_result_label.configure(text="Дугаарын 4 цифрээ оруулна уу", fg=self.accent)
            self.rule_days_label.configure(text="Жишээ: 1113 → сүүлийн цифр 3")
            return
        mode = self.settings["rules"].get("mode", "weekly")
        if mode == "weekly":
            days = restriction_days(plate, self.settings["rules"])
            names = ", ".join(DAY_NAMES[d] for d in days)
            self.rule_result_label.configure(text=f"{display_plate(plate)} • Хязгаарлах өдөр: {names or 'байхгүй'}", fg=self.accent)
            allowed = ", ".join(DAY_NAMES[d] for d in range(7) if d not in days)
            rules = self.settings["rules"]
            self.rule_days_label.configure(text=f"Зорчих өдрүүд: {allowed}\nСүүлийн цифр: {plate[3]} • Хязгаарлах цаг: {rules['start']}–{rules['end']} • тохируулсан хуваарь")
        else:
            label = "Тэгш / сондгой горим" if mode == "odd_even" else "Хязгаарлалтын шалгалт унтраалттай"
            self.rule_result_label.configure(text=f"{display_plate(plate)} • {label}", fg=self.accent)
            self.rule_days_label.configure(text="Энэ горимын төлөвийг доорх сонгосон огноо, цагаар шалгана.")

    def check_rule(self):
        plate = rule_plate(self.rule_plate.get())
        if not plate:
            messagebox.showerror("Дугаарын хэлбэр", "4 цифр эсвэл бүтэн дугаар оруулна уу. Жишээ: 1113, 1113УБА", parent=self)
            return
        try:
            at = datetime.strptime(self.rule_date.get().strip() + " " + self.rule_time.get().strip(), "%Y-%m-%d %H:%M").replace(tzinfo=UB_TIME)
        except ValueError:
            messagebox.showerror("Огноо / цаг", "YYYY-MM-DD болон HH:MM хэлбэрээр оруулна уу.", parent=self)
            return
        result = evaluate(plate, self.settings["rules"], at)
        self.update_restriction_preview()
        self.last_rule_result = result
        self.rule_date_result.configure(text=f"Шалгасан огноо: {at:%Y-%m-%d} • {DAY_NAMES[at.weekday()]} • {at:%H:%M}\nТухайн үеийн төлөв: {result.label}",
                                        fg=RED if result.restricted_now else AMBER if result.restricted_today else GREEN)
        self.rule_detail_label.configure(text=result.detail)
        records = self.store.query(plate, at.date().isoformat())
        subject = f"{plate} гэсэн 4 цифртэй дугаарууд" if len(plate) == 4 else display_plate(plate)
        self.rule_seen_label.configure(text=f"{subject}: сонгосон өдөр {len(records)} бүртгэл." +
                                       (f"  Анх: {records[-1]['detected_at'][11:19]}  Сүүлд: {records[0]['detected_at'][11:19]}" if records else ""))

    def update_schedule(self):
        rules = self.settings["rules"]
        for i, label in enumerate(self.schedule_labels):
            label.configure(text=", ".join(map(str, rules["weekdays"].get(str(i), []))) or "—")
        mode = next((k for k, v in MODES.items() if v == rules["mode"]), rules["mode"])
        self.schedule_note.configure(text=f"Идэвхтэй горим: {mode}  •  {rules['start']}–{rules['end']}  •  Улаанбаатар UTC+08:00")

    def choose_model(self):
        path = filedialog.askopenfilename(parent=self, title="Дугаар илрүүлэх YOLO загвар", filetypes=[("YOLO weights", "*.pt")])
        if path:
            self.model_var.set(path)

    def apply_settings(self):
        if self.worker.alive or self.preparing:
            self.status_text.set("Тохиргоо хадгалахын өмнө одоогийн танилтыг зогсооно уу.")
            return False
        settings = deepcopy(self.settings)
        try:
            settings["recognition_mode"] = RECOGNITION_MODES[self.recognition_mode_var.get()]
            settings["playback_speed"] = float(self.playback_speed_var.get())
            if not .25 <= settings["playback_speed"] <= 4:
                raise ValueError("Дүрсний хурд 0.25–4 хооронд байна.")
            for key, low, high, cast in [("detection_confidence", .05, .95, float), ("ocr_confidence", .05, .95, float),
                                          ("frame_stride", 1, 120, int), ("cooldown_seconds", 1, 3600, int), ("confirmation_frames", 1, 10, int),
                                          ("video_min_confidence", .05, .99, float), ("neighbor_frames", 0, 10, int)]:
                value = cast(self.setting_vars[key].get())
                if not low <= value <= high:
                    raise ValueError(f"{key}: {low}–{high} хооронд утга оруулна уу.")
                settings[key] = value
            for var in [self.start_var, self.end_var]:
                datetime.strptime(var.get().strip(), "%H:%M")
            if self.start_var.get().strip() == self.end_var.get().strip():
                raise ValueError("Эхлэх болон дуусах цаг ялгаатай байна.")
            weekdays = {}
            for index, var in enumerate(self.weekday_vars):
                digits = [int(v.strip()) for v in var.get().split(",") if v.strip()]
                if any(not 0 <= n <= 9 for n in digits):
                    raise ValueError("Хуваарийн цифр 0–9 хооронд байна.")
                weekdays[str(index)] = sorted(set(digits))
            holidays = [v.strip() for v in self.holiday_var.get().split(",") if v.strip()]
            for day in holidays:
                datetime.strptime(day, "%Y-%m-%d")
            exempt = [normalize_plate(v) for v in self.exempt_var.get().split(",") if v.strip()]
            if any(not v for v in exempt):
                raise ValueError("Чөлөөлөх дугаарт 1234УБА хэлбэр ашиглана уу.")
            model = self.model_var.get().strip()
            if model and not Path(model).is_file():
                raise ValueError("YOLO .pt файл олдсонгүй.")
            settings.update(model_path=model, accent=self.accent_var.get(), save_video=self.video_var.get(), device=self.device_var.get(), camera_backend=self.backend_var.get(),
                            adaptive_sampling=self.adaptive_var.get())
            settings["rules"].update(mode=MODES[self.mode_var.get()], start=self.start_var.get().strip(), end=self.end_var.get().strip(),
                                     in_zone=self.zone_var.get(), even_on_even_date=self.parity_var.get(), weekdays=weekdays,
                                     holidays=holidays, exempt_plates=exempt)
        except (ValueError, KeyError) as exc:
            messagebox.showerror("Тохиргооны утга", str(exc), parent=self)
            return False
        save_settings(settings)
        old_accent = self.settings.get("accent")
        recognizer = self.worker.recognizer
        runtime = {key: self.worker.settings[key] for key in ("_rotation", "_video_review") if key in self.worker.settings}
        reuse_model = all(settings.get(key) == self.settings.get(key) for key in ("model_path", "device"))
        self.settings = settings
        self.worker = DetectionWorker(settings, self.store)
        self.worker.settings.update(runtime)
        if reuse_model:
            self.worker.recognizer = recognizer
        self.video_stride_var.set(str(settings["frame_stride"]))
        self.update_schedule()
        if old_accent != settings["accent"]:
            current = self.current_page
            self.accent = ACCENTS[settings["accent"]]
            for child in self.winfo_children():
                child.destroy()
            self._preview_item = None
            self.preview_photo = None
            self.configure_styles()
            self.build_ui()
            self.refresh_records()
            self.show_page(current)
            self.redraw_preview()
        self.status_text.set("Тохиргоо хадгаллаа. Дараагийн танилтад шинэ тохиргоо үйлчилнэ.")
        return True

    def prepare_ai(self):
        if self.worker.alive or self.preparing:
            return
        if not self.apply_settings():
            return
        self.preparing = True
        worker = self.worker
        worker.stop_event.clear()
        def prepare():
            try:
                worker.recognizer = PlateRecognizer(worker.settings, lambda text: worker.emit("status", text), worker.stop_event)
                worker.emit("status", "YOLO + Монгол EasyOCR загварууд бэлэн. Одоо зураг, видео, камераа сонгоно уу.")
            except Exception as exc:
                logging.exception("AI setup failed")
                worker.emit("error", str(exc))
            finally:
                worker.emit("prepared", None)
        threading.Thread(target=prepare, daemon=True, name="platewatch-prepare").start()

    def poll(self):
        if self.closing:
            return
        delay = 33
        try:
            self._poll_once()
        except Exception:
            # Scheduling only at the end used to permanently freeze camera,
            # image import and queued source switches after one MemoryError.
            logging.exception("GUI update failed; continuing on the next tick")
            delay = 500
            self.status_text.set("Дэлгэцийн шинэчлэлтийг сэргээж байна...")
        finally:
            if not self.closing:
                self.after(delay, self.poll)

    def _poll_once(self):
        for _ in range(150):
            try:
                kind, value = self.worker.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                if self.pending_source is None:
                    self.status_text.set(value)
            elif kind == "error":
                if self.pending_source is None:
                    self.display_error(value)
            elif kind == "analysis_error":
                if self.pending_source is None:
                    self.analysis_time.set("AI шалгалт алдаатай")
                    self.display_error(value)
            elif kind == "frame_result":
                if self.pending_source is None:
                    self.show_frame_result(value)
            elif kind == "frame_status":
                if self.frame_check_busy and self.pending_source is None:
                    self.status_text.set(value)
            elif kind == "frame_error":
                if self.pending_source is None:
                    self.display_error(value)
            elif kind == "frame_check_done":
                self.frame_check_busy = False
                self.frame_button.configure(state="normal", text="Энэ кадрыг таних")
                if self.status_text.get().startswith("Зогсоож"):
                    self.status_text.set("Зогслоо. Өөр кадр эсвэл эх үүсвэр сонгож болно.")
            elif kind == "source_ready":
                logging.info("Source ready: %s", value)
                if value["kind"] == "camera":
                    self.source_label.configure(text=f"Камер {value['source']} • {value.get('backend', '')}")
                    self.camera_var.set(value["source"])
                    self.settings["camera_index"] = int(value["source"])
                    save_settings(self.settings)
                elif value["kind"] == "video":
                    self.video_duration = value.get("duration", 0)
                    self.video_fps = value.get("fps", 25)
                    self.source_label.configure(text=f"{Path(value['source']).name[:65]}  •  {value.get('fps', 0):.0f} FPS  •  {value.get('backend', '')}")
            elif kind == "playback":
                self.playback_seconds, duration = value
                self.video_time.set(f"▶  {self.format_time(self.playback_seconds)} / {self.format_time(duration)}")
            elif kind == "playback_done":
                state = "Дүрс тасарлаа" if value else "Дахин үзэх ↻"
                self.video_time.set(f"{self.format_time(self.playback_seconds)} / {self.format_time(self.video_duration)}  ·  {state}")
            elif kind == "analysis_position":
                seconds, duration = value
                self.analysis_time.set(f"AI  {self.format_time(seconds)} / {self.format_time(duration)}")
            elif kind == "overlay":
                self.camera_overlay = value
            elif kind == "prepared":
                self.preparing = False
            elif kind == "ready":
                if self.pending_source is None:
                    self.analysis_time.set("AI шалгаж байна...")
                    self.status_text.set(f"Видео шалгаж байна • {self.worker.settings['frame_stride']} кадр тутам • давтан уншилтыг баталгаажуулна" if value == "video" and self.worker.video_review else
                                         "Танилт ажиллаж байна • " + {"camera": "шууд камер", "image": "зураг", "video": "видео"}.get(value, value))
                    logging.info("GUI AI ready: %s", value)
            elif kind == "record":
                self._records_dirty = True
                if value["plate"] not in self.session_records:
                    self.session_records[value["plate"]] = value
                    at = self.format_time(float(value.get("video_seconds") or 0))
                    self.session_list.insert("end", f"{display_plate(value['plate'])}  ·  {at}  ·  {float(value['ocr_confidence']):.0%}")
                self.last_time_label.configure(text="Хадгалсан: " + value["detected_at"][11:19])
                if not self._frame_pinned:
                    import numpy as np
                    path = self.store.resolve(value["crop_path"])
                    with Image.open(path) as original:
                        self.show_plate_crop(np.array(original.convert("RGB")))
            elif kind == "detection":
                if self._frame_pinned:
                    continue
                detection, rule = value
                self.last_detection = detection
                self.plate_label.configure(text=display_plate(detection.plate))
                self.rule_label.configure(text=rule.label, fg=RED if rule.restricted_now else GREEN)
                self.confidence_label.configure(text=self.confidence_text(detection))
            elif kind == "metrics":
                count, milliseconds, fps = value
                self.stat_vars[3].set(str(fps))
            elif kind == "progress":
                self.progress.set(value)
            elif kind == "done":
                self.pause_button.configure(text="▶  Үргэлжлүүлэх" if self.worker.pause_event.is_set() else "Ⅱ  Дүрс зогсоох")
                if not self.last_error:
                    state = "Зогссон" if self.active_kind == "video" and self.progress.get() < 1 else "✓"
                    self.analysis_time.set(f"{state}  {self.worker.processed} кадр · {self.worker.saved_count} дугаар")
                if self.status_text.get().startswith("Зогсоож"):
                    self.status_text.set("Зогслоо. Өөр эх үүсвэр сонгож болно.")
        if self._records_dirty:
            self.refresh_records()
            self._records_dirty = False
        if not self.worker.pause_event.is_set():
            try:
                self.last_rgb, self._frame_source, self._frame_seconds = self.worker.previews.get_nowait()
                try:
                    self.worker.frames.get_nowait()
                except queue.Empty:
                    pass
                self._frame_original = None
                self._preview_dirty = True
            except queue.Empty:
                pass
            try:
                packet = self.worker.review_frames.get_nowait()
                self.last_rgb, self._frame_original, self.playback_seconds = packet[:3]
                self._frame_seconds = self.playback_seconds
                if len(packet) > 3:
                    self._frame_source = packet[3]
                if self.active_kind == "video":
                    self.video_time.set(f"Шалгасан кадр  {self.format_time(self.playback_seconds)} / {self.format_time(self.video_duration)}")
                self._preview_dirty = True
            except queue.Empty:
                pass
        if self._preview_dirty and self.redraw_preview() and self.last_rgb is not None:
            if not self.preview_displayed:
                logging.info("GUI preview displayed: %s %s", self.active_kind, self.last_rgb.shape)
                self.preview_displayed = True
        request_path = DATA_DIR / "open_request.json"
        if request_path.exists():
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
                request_path.unlink(missing_ok=True)
                if 0 <= now_ub().timestamp() - request.get("at", 0) < 30:
                    self.start_source(request["kind"], request["source"], request.get("rotation"))
            except (OSError, ValueError, KeyError):
                logging.exception("Could not process source request")
        if self.pending_source is not None and not self.worker.alive and not self.preparing:
            request = self.pending_source
            self.pending_source = None
            self.start_source(*request)

    def display_error(self, value):
        self.last_error = str(value)
        self.status_text.set("Алдаа • " + str(value))
        self.activity_label.configure(fg=RED)
        logging.error("User-visible error: %s", value)
        self.show_page("monitor")

    def tick(self):
        if self.closing:
            return
        now = now_ub()
        self.clock_label.configure(text=f"Өнөөдөр • {now:%Y.%m.%d} • {DAY_NAMES[now.weekday()]}\n{now:%H:%M:%S}  •  Улаанбаатар")
        self.after(1000, self.tick)

    def open_path(self, path):
        path = Path(path)
        if not path.exists():
            messagebox.showerror("Файл олдсонгүй", str(path), parent=self)
            return
        os.startfile(str(path.resolve()))

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.pending_source = None
        self.worker.stop()
        for timer in self.tk.call("after", "info"):
            try:
                self.after_cancel(timer)
            except tk.TclError:
                pass
        if self.worker.thread and self.worker.alive:
            self.worker.thread.join(timeout=.6)
        if self.worker.frame_worker and self.worker.frame_worker.thread:
            self.worker.frame_worker.thread.join(timeout=.6)
        self.destroy()
