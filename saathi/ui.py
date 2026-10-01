import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

from .audio import Speaker, record_audio
from .config import ROOT
from .conversation import CloudAssistant, greeting, parse_intent
from .recognition import Enrollment, IdentityGate, Match, Matcher
from .session import SceneGuard, token_is_current
from .storage import Store, clean_name
from .vision import CameraWorker

BG = "#0b1220"
CARD = "#142033"
TEXT = "#edf4ff"
MUTED = "#a4b5ca"
ACCENT = "#66e0c2"


class App:
    def __init__(self, root, config):
        self.root, self.config = root, config
        self.store = Store(ROOT / "data" / "profiles.sqlite3")
        self.matcher = Matcher(self.store.templates(), config.recognition_threshold, config.recognition_margin)
        self.gate = IdentityGate(config.stable_frames, config.greeting_cooldown_seconds)
        self.scene = SceneGuard()
        self.speaker = Speaker()
        self.camera = None
        self.camera_generation = 0
        self.snapshot = None
        self.enrollment = None
        self.events = queue.Queue()
        self.audio_cancel = threading.Event()
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.cloud_enabled = tk.BooleanVar(value=False)
        self.voice_enabled = tk.BooleanVar(value=True)
        self.busy = False
        self.job_id = 0
        self.history = []
        self.closed = False
        self.unknown_frames = 0
        self.last_unknown_prompt = float("-inf")
        self._build()
        self.refresh_profiles()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(40, self.tick)

    def _build(self):
        self.root.title("Saathi | See. Remember. Converse.")
        self.root.geometry("1220x840")
        self.root.minsize(1000, 720)
        self.root.configure(bg=BG)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8), background="#243750", foreground=TEXT)
        style.map("TButton", background=[("active", "#365371")])
        style.configure("Accent.TButton", background=ACCENT, foreground=BG)
        style.configure("TCheckbutton", background=BG, foreground=TEXT)
        style.map("TCheckbutton", background=[("active", BG)])
        style.configure("TEntry", fieldbackground=CARD, foreground=TEXT, padding=8)
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(15, 8))

        header = ttk.Frame(self.root, padding=(24, 18))
        header.pack(fill="x")
        ttk.Label(header, text="SAATHI", font=("Segoe UI", 23, "bold"), foreground=ACCENT).pack(side="left")
        ttk.Label(header, text="  /  A familiar face. A thoughtful conversation.", style="Muted.TLabel").pack(side="left")
        ttk.Button(header, text="Settings", command=self.settings).pack(side="right")
        body = ttk.Frame(self.root, padding=(24, 0, 24, 16))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        self.camera_label = tk.Label(left, text="Your camera is off\n\nStart the camera to meet Saathi.",
            bg=CARD, fg=MUTED, font=("Segoe UI", 16), width=52, height=17)
        self.camera_label.pack(fill="both", expand=True)
        self.identity = tk.StringVar(value="Ready when you are")
        ttk.Label(left, textvariable=self.identity, font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(16, 4))
        self.metrics = tk.StringVar(value="0 faces  /  0 FPS  /  Local face processing")
        ttk.Label(left, textvariable=self.metrics, style="Muted.TLabel").pack(anchor="w")
        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=16)
        self.start_button = ttk.Button(buttons, text="Start camera", style="Accent.TButton", command=self.start_camera)
        self.start_button.pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Stop camera", command=self.stop_camera).pack(side="left")
        ttk.Button(buttons, text="Enroll a face", command=self.enroll_dialog).pack(side="right")
        self.status = tk.StringVar(value="Faces stay on this device. Set up cloud conversation in Settings.")
        ttk.Label(left, textvariable=self.status, wraplength=590, style="Muted.TLabel").pack(anchor="w", fill="x")
        self.progress = ttk.Progressbar(left, maximum=self.config.enrollment_samples)
        self.progress.pack(fill="x", pady=10)
        ttk.Button(left, text="Cancel enrollment", command=self.cancel_enrollment).pack(anchor="w")

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        tabs = ttk.Notebook(right)
        tabs.pack(fill="both", expand=True)
        chat = ttk.Frame(tabs, padding=12)
        people = ttk.Frame(tabs, padding=12)
        tabs.add(chat, text="Conversation")
        tabs.add(people, text="Profiles & memory")
        self.transcript = tk.Text(chat, bg=CARD, fg=TEXT, wrap="word", font=("Segoe UI", 11),
            relief="flat", padx=14, pady=14, width=36, height=16, state="disabled")
        self.transcript.pack(fill="both", expand=True)
        self.transcript.tag_configure("assistant", foreground=ACCENT)
        self.transcript.tag_configure("system", foreground=MUTED)
        self.message = tk.StringVar()
        entry = ttk.Entry(chat, textvariable=self.message)
        entry.pack(fill="x", pady=(12, 8))
        entry.bind("<Return>", lambda _: self.send())
        row = ttk.Frame(chat)
        row.pack(fill="x")
        ttk.Button(row, text="Send", command=self.send).pack(side="left")
        self.listen_button = ttk.Button(row, text=f"Talk ({self.config.record_seconds}s)", command=self.listen)
        self.listen_button.pack(side="left", padx=8)
        ttk.Button(row, text="Stop voice", command=self.cancel_audio).pack(side="right")
        ttk.Checkbutton(chat, text="Speak replies with system voice", variable=self.voice_enabled,
                        command=self.voice_changed).pack(anchor="w", pady=10)
        ttk.Label(chat, text="Say or type: My name is Saiyam\nTo save a note: Remember that I study Python",
                  style="Muted.TLabel", wraplength=360).pack(anchor="w")
        self.people = tk.Listbox(people, bg=CARD, fg=TEXT, selectbackground="#345c68", relief="flat",
                                 font=("Segoe UI", 12), height=12, exportselection=False)
        self.people.pack(fill="both", expand=True)
        ttk.Label(people, text="Profiles contain face templates and explicitly saved notes.\nNo camera images or audio recordings are saved.",
                  style="Muted.TLabel", wraplength=360).pack(anchor="w", pady=12)
        ttk.Button(people, text="View / edit approved memory", command=self.memory_dialog).pack(fill="x", pady=4)
        ttk.Button(people, text="Delete selected profile", command=self.delete_profile).pack(fill="x", pady=4)
        self.log("system", "Welcome. Start the camera, then enroll your face. Recognition works without an API key.")

    def log(self, role, text):
        self.transcript.configure(state="normal")
        self.transcript.insert("end", f"{role.upper()}\n{text}\n\n", role)
        if int(self.transcript.index("end-1c").split(".")[0]) > 180:
            self.transcript.delete("1.0", "50.0")
        self.transcript.see("end")
        self.transcript.configure(state="disabled")

    def speak(self, text):
        self.log("assistant", text)
        if self.voice_enabled.get() and not self.busy:
            self.speaker.say(text)

    def refresh_profiles(self):
        self.profiles = {p.id: p for p in self.store.profiles()}
        self.profile_ids = list(self.profiles)
        self.people.delete(0, "end")
        for p in self.profiles.values():
            self.people.insert("end", p.name)
        self.matcher = Matcher(self.store.templates(), self.config.recognition_threshold, self.config.recognition_margin)

    def start_camera(self):
        if self.camera and self.camera.thread.is_alive():
            return
        self.stop_camera()
        self.camera = CameraWorker(self.config)
        self.camera.start()
        self.status.set("Starting camera...")
        self.start_button.configure(state="disabled")

    def stop_camera(self):
        if self.camera:
            self.camera.stop()
        self.camera_generation += 1
        self.snapshot = None
        self.cancel_enrollment()
        self.gate.reset()
        self.scene.reset()
        self.clear_session()
        self.identity.set("Camera off")
        self.camera_label.configure(image="", text="Your camera is off\n\nStart the camera to meet Saathi.")
        self.camera_label.image = None
        self.metrics.set("0 faces  /  0 FPS  /  Local face processing")
        self.status.set("Camera stopped. Microphone and conversation paused.")
        if not self.camera or not self.camera.thread.is_alive():
            self.start_button.configure(state="normal")

    def clear_session(self):
        self.job_id += 1
        self.audio_cancel.set()
        self.speaker.cancel()
        self.history.clear()
        self.busy = False
        self.listen_button.configure(text=f"Talk ({self.config.record_seconds}s)")
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")

    def token(self):
        return self.camera_generation, self.scene.epoch, self.gate.epoch

    def current(self, token):
        age = time.monotonic() - self.snapshot.captured_at if self.snapshot else float("inf")
        return token_is_current(token, self.token(), age, self.scene.valid)

    def ready_face(self):
        if not self.current(self.token()):
            raise ValueError("Keep one clearly visible face in the camera before continuing")
        if self.enrollment:
            raise ValueError("Finish or cancel face enrollment first")

    def handle_frame(self, shot):
        import cv2
        from PIL import Image, ImageTk
        if time.monotonic() - shot.captured_at >= 2:
            self.gate.reset()
            self.scene.reset()
            self.clear_session()
            self.status.set("Inference is too slow for a fresh match. Reduce resolution or disable object detection.")
            return
        self.snapshot = shot
        changed = self.scene.observe([f.embedding for f in shot.faces])
        matches = [self.matcher.match(f.embedding) if f.embedding is not None else Match(None) for f in shot.faces]
        old = self.gate.active
        active, greet = self.gate.observe(matches)
        if changed or old != active:
            self.clear_session()
        if changed:
            self.unknown_frames = 0
        if self.enrollment:
            self.enrollment_frame(shot)
        if len(shot.faces) > 1:
            self.identity.set("One person at a time, please")
        elif not shot.faces:
            self.identity.set("Looking for a face")
        elif shot.faces[0].quality:
            self.identity.set(shot.faces[0].quality)
        elif active and active in self.profiles:
            self.identity.set(f"Welcome, {self.profiles[active].name}")
            if greet and not self.enrollment:
                self.speak(greeting(self.profiles[active]))
        elif matches[0].person_id:
            self.identity.set("Checking your face...")
        else:
            self.identity.set("New face? Introduce yourself.")
            self.unknown_frames += 1
            now = time.monotonic()
            if (self.unknown_frames >= self.config.stable_frames and not self.enrollment and not self.busy
                    and now - self.last_unknown_prompt >= self.config.greeting_cooldown_seconds):
                self.last_unknown_prompt = now
                self.speak("Hello! I'm Saathi. Choose Talk and tell me your name, or choose Enroll a face.")
        for face, match in zip(shot.faces, matches):
            x, y, w, h = face.box
            known = match.person_id in self.profiles
            color = (194, 224, 102) if known else (75, 180, 255)
            label = self.profiles[match.person_id].name if known else (face.quality or "Unknown")
            cv2.rectangle(shot.frame, (x, y), (x+w, y+h), color, 2)
            # Similarity is not a calibrated probability; deliberately no percentage.
            cv2.putText(shot.frame, label.encode("ascii", "replace").decode(), (max(0, x), max(20, y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        im = Image.fromarray(cv2.cvtColor(shot.frame, cv2.COLOR_BGR2RGB))
        size = (max(320, self.camera_label.winfo_width()), max(240, self.camera_label.winfo_height()))
        im.thumbnail(size)
        photo = ImageTk.PhotoImage(im)
        self.camera_label.configure(image=photo, text="", width=0, height=0)
        self.camera_label.image = photo
        objects = " | " + ", ".join(f"{k}: {v}" for k, v in shot.object_counts.items()) if shot.object_counts else ""
        self.metrics.set(f"{len(shot.faces)} faces  /  {shot.fps:.1f} FPS  /  {shot.inference_ms:.0f} ms inference{objects}")

    def enroll_dialog(self, suggested=""):
        try:
            self.ready_face()
            if self.gate.active:
                raise ValueError("This face is already recognized. Delete the existing profile before re-enrolling.")
        except ValueError as exc:
            self.status.set(str(exc))
            return
        token = self.token()
        dialog = tk.Toplevel(self.root)
        dialog.title("Enroll your face")
        dialog.configure(bg=BG)
        dialog.transient(self.root)
        dialog.grab_set()
        content = ttk.Frame(dialog, padding=24)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text="What should I call you?", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        name = tk.StringVar(value=suggested)
        ttk.Entry(content, textvariable=name, width=38).pack(fill="x", pady=12)
        consent = tk.BooleanVar(value=False)
        ttk.Checkbutton(content, text="I agree to save my face templates and name on this device.", variable=consent).pack(anchor="w")
        ttk.Label(content, text="Only enroll yourself or a willing participant.\nFace the camera, then slowly turn a little left and right.\nNo photos are saved. You can delete your profile later.",
                  style="Muted.TLabel").pack(anchor="w", pady=12)
        def begin():
            try:
                if not self.current(token):
                    raise ValueError("The face changed while confirming. Close this dialog and try again.")
                value = clean_name(name.get())
                if any(p.name.casefold() == value.casefold() for p in self.profiles.values()):
                    raise ValueError("That name is already registered. Use a distinct full name.")
                self.enrollment = Enrollment(value, consent.get(), self.config.enrollment_samples,
                                             self.config.enrollment_timeout_seconds)
                self.clear_session()
                dialog.destroy()
                self.status.set("Capturing face samples. Keep one face in view and vary your angle slightly.")
            except ValueError as exc:
                messagebox.showerror("Cannot enroll", str(exc), parent=dialog)
        ttk.Button(content, text="Confirm name & start enrollment", command=begin, style="Accent.TButton").pack(fill="x")

    def enrollment_frame(self, shot):
        enrollment = self.enrollment
        try:
            if time.monotonic() > enrollment.deadline:
                raise ValueError("Enrollment timed out. Improve lighting and try again.")
            if len(shot.faces) != 1:
                raise ValueError("Enrollment cancelled because the face left or another person entered.")
            face = shot.faces[0]
            if face.embedding is None:
                self.status.set(f"Enrollment paused: {face.quality}")
                return
            if enrollment.add(face.embedding, self.matcher):
                self.store.enroll(enrollment.name, enrollment.samples, consent=True)
                self.enrollment = None
                self.refresh_profiles()
                self.gate.reset()
                self.status.set(f"Saved {enrollment.name}. Recognition is ready.")
                self.speak(f"Nice to meet you, {enrollment.name}. Your face profile is saved.")
            self.progress["value"] = len(enrollment.samples)
        except Exception as exc:
            self.enrollment = None
            self.progress["value"] = 0
            self.status.set(str(exc))

    def cancel_enrollment(self):
        if self.enrollment:
            self.enrollment = None
            self.status.set("Enrollment cancelled. No profile was saved.")
        self.progress["value"] = 0

    def settings(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Cloud conversation settings")
        dialog.configure(bg=BG)
        dialog.transient(self.root)
        frame = ttk.Frame(dialog, padding=24)
        frame.pack()
        ttk.Label(frame, text="Connect conversation", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(frame, text="OpenAI API key (kept in memory for this run)").pack(anchor="w", pady=(15, 4))
        key = tk.StringVar(value=self.api_key)
        ttk.Entry(frame, textvariable=key, show="*", width=58).pack(fill="x")
        allowed = tk.BooleanVar(value=self.cloud_enabled.get())
        ttk.Checkbutton(frame, text="Enable cloud conversation and microphone transcription", variable=allowed).pack(anchor="w", pady=12)
        ttk.Label(frame, text="Cloud requests send your messages, recent conversation, name and approved notes.\n"
                  "Talk sends the short microphone recording for transcription.\n"
                  "Face images and face templates stay local. System speech is generated locally.\n"
                  "API usage requires an API account and may incur charges.\n"
                  "Response storage is disabled; provider retention policies still apply.",
                  style="Muted.TLabel", wraplength=510).pack(anchor="w")
        ttk.Label(frame, text=f"Chat: {self.config.chat_model}\nSpeech recognition: {self.config.transcription_model}\n"
                  "Camera, microphone and thresholds: config.local.json (restart to apply).",
                  style="Muted.TLabel").pack(anchor="w", pady=12)
        def save():
            if allowed.get() and not key.get().strip():
                messagebox.showerror("API key needed", "Enter a key or leave cloud conversation disabled.", parent=dialog)
                return
            self.api_key = key.get().strip()
            self.cloud_enabled.set(allowed.get())
            self.clear_session()
            self.status.set("Cloud conversation enabled." if allowed.get() else "Cloud conversation disabled. Recognition is still available.")
            dialog.destroy()
        ttk.Button(frame, text="Apply settings", command=save, style="Accent.TButton").pack(fill="x")

    def selected_profile(self):
        selection = self.people.curselection()
        if not selection:
            self.status.set("Select a profile first.")
            return None
        return self.profiles.get(self.profile_ids[selection[0]])

    def memory_dialog(self):
        profile = self.selected_profile()
        if not profile:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Approved memory - {profile.name}")
        dialog.configure(bg=BG)
        frame = ttk.Frame(dialog, padding=20)
        frame.pack()
        ttk.Label(frame, text="Notes used to personalize future conversations.\nThese notes are sent to cloud AI when this profile chats.").pack(anchor="w")
        editor = tk.Text(frame, width=55, height=8, wrap="word")
        editor.pack(pady=12)
        editor.insert("1.0", profile.memory)
        def save():
            try:
                self.store.update_memory(profile.id, editor.get("1.0", "end-1c"))
                self.refresh_profiles()
                self.clear_session()
                dialog.destroy()
                self.status.set("Approved memory updated.")
            except ValueError as exc:
                messagebox.showerror("Cannot save", str(exc), parent=dialog)
        ttk.Button(frame, text="Approve & save notes", command=save).pack(fill="x")

    def delete_profile(self):
        profile = self.selected_profile()
        if profile and messagebox.askyesno("Delete profile", f"Delete {profile.name}'s face templates and approved memory?", parent=self.root):
            self.store.delete(profile.id)
            self.refresh_profiles()
            self.gate.reset()
            self.scene.reset()
            self.clear_session()
            self.status.set("Profile deleted from the local database.")

    def client(self):
        if not self.cloud_enabled.get():
            raise ValueError("Enable cloud conversation in Settings first")
        return CloudAssistant(self.api_key, self.config)

    def send(self, text=None):
        text = self.message.get().strip() if text is None else text.strip()
        try:
            if self.busy:
                raise ValueError("Wait for the current turn or press Stop voice")
            self.ready_face()
            intent = parse_intent(text)
            self.message.set("")
            if intent.kind == "introduce":
                self.enroll_dialog(intent.value)
                return
            profile = self.profiles.get(self.gate.active)
            if not profile:
                raise ValueError("Please enroll first, or wait for your face to be recognized")
            self.log("you", text)
            if intent.kind in {"remember", "forget_memory"}:
                token = self.token()
                value = (profile.memory + "\n" + intent.value).strip() if intent.kind == "remember" else ""
                if messagebox.askyesno("Approve memory change", f"Save this memory for {profile.name}?\n\n{value or '(clear all notes)'}\n\nSaved notes are included in future cloud conversations.", parent=self.root):
                    if not self.current(token):
                        raise ValueError("The face changed. The memory change was cancelled.")
                    self.store.update_memory(profile.id, value)
                    self.refresh_profiles()
                    self.speak("Your approved memory has been updated.")
                return
            client = self.client()
            token, history = self.token(), list(self.history)
            self.speaker.cancel()
            self.busy = True
            self.job_id += 1
            job = self.job_id
            self.status.set("Thinking...")
            def work():
                try:
                    answer = client.reply(profile, history, text)
                    self.events.put(("reply", job, token, (text, answer)))
                except Exception as exc:
                    self.events.put(("error", job, token, str(exc)))
                finally:
                    client.close()
            threading.Thread(target=work, daemon=True).start()
        except (ValueError, RuntimeError) as exc:
            self.status.set(str(exc))

    def listen(self):
        try:
            if self.busy:
                raise ValueError("Wait for the current turn or press Stop voice")
            self.ready_face()
            client = self.client()
            token = self.token()
            self.speaker.cancel()
            self.audio_cancel = threading.Event()
            cancel = self.audio_cancel
            self.busy = True
            self.job_id += 1
            job = self.job_id
            self.listen_button.configure(text="Listening...")
            self.status.set(f"Speak after the voice stops. Recording for {self.config.record_seconds} seconds.")
            def work():
                try:
                    deadline = time.monotonic() + 3
                    while self.speaker.busy.is_set():
                        if cancel.wait(0.05) or time.monotonic() > deadline:
                            raise ValueError("Speech output has not stopped. Try again.")
                    if cancel.wait(0.2):
                        return
                    audio = record_audio(self.config.record_seconds, self.config.microphone_index, cancel)
                    if cancel.is_set():
                        return
                    self.events.put(("status", job, token, "Transcribing your voice..."))
                    text = client.transcribe(audio)
                    self.events.put(("transcript", job, token, text))
                except Exception as exc:
                    # Cloud errors have already been sanitized in CloudAssistant.
                    message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "Microphone unavailable. Check permissions or microphone_index."
                    self.events.put(("error", job, token, message))
                finally:
                    client.close()
            threading.Thread(target=work, daemon=True).start()
        except (ValueError, RuntimeError) as exc:
            self.status.set(str(exc))

    def cancel_audio(self):
        self.job_id += 1
        self.audio_cancel.set()
        self.speaker.cancel()
        self.busy = False
        self.listen_button.configure(text=f"Talk ({self.config.record_seconds}s)")
        self.status.set("Voice stopped. Any in-flight cloud response will be ignored.")

    def voice_changed(self):
        if not self.voice_enabled.get():
            self.speaker.cancel()

    def tick(self):
        if self.closed:
            return
        try:
            if self.camera:
                while not self.camera.events.empty():
                    kind, text = self.camera.events.get_nowait()
                    if kind in {"error", "invalidate", "stopped"}:
                        self.snapshot = None
                        self.scene.reset()
                        self.gate.reset()
                        self.clear_session()
                        self.cancel_enrollment()
                    if kind == "error":
                        self.status.set(text)
                        self.identity.set("Camera unavailable")
                    elif kind == "status":
                        self.status.set(text)
                    elif kind == "stopped":
                        self.start_button.configure(state="normal")
                if not self.camera.stop_event.is_set() and self.camera.thread.is_alive():
                    try:
                        self.handle_frame(self.camera.frames.get_nowait())
                    except queue.Empty:
                        pass
            if self.snapshot and time.monotonic() - self.snapshot.captured_at >= 2:
                self.snapshot = None
                self.gate.reset()
                self.scene.reset()
                self.clear_session()
                self.identity.set("Waiting for fresh camera frames")
            while not self.events.empty():
                kind, job, token, payload = self.events.get_nowait()
                if job != self.job_id or not self.current(token):
                    continue
                if kind == "status":
                    self.status.set(payload)
                    continue
                self.busy = False
                self.listen_button.configure(text=f"Talk ({self.config.record_seconds}s)")
                if kind == "reply":
                    prompt, answer = payload
                    self.history.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}])
                    self.history = self.history[-10:]
                    self.status.set("Ready for your next message.")
                    self.speak(answer)
                elif kind == "transcript":
                    # Always review transcription before sending or enrolling.
                    self.message.set(payload)
                    self.status.set("Check the transcript, then press Send. You can correct names before enrollment.")
                elif kind == "error":
                    self.status.set(payload)
            while not self.speaker.events.empty():
                self.status.set(self.speaker.events.get_nowait())
        except Exception as exc:
            self.status.set(f"App error: {type(exc).__name__}. Stop and restart the camera; check your configuration.")
            self.gate.reset()
            self.scene.reset()
            self.clear_session()
        finally:
            self.root.after(40, self.tick)

    def close(self):
        self.closed = True
        if self.camera:
            self.camera.stop()
        self.audio_cancel.set()
        self.speaker.close()
        self.api_key = ""
        self.root.destroy()


def launch(config):
    root = tk.Tk()
    App(root, config)
    root.mainloop()
