import cv2
import csv
import hashlib
import numpy as np
import tkinter as tk

from tkinter import ttk, messagebox
from pathlib import Path


# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "dataset" / "indices_clean"

OUTPUT_DIR = BASE_DIR / "dataset" / "training"

IMAGES_DIR = OUTPUT_DIR / "labeled_indices"

LABELS_FILE = OUTPUT_DIR / "labels.csv"

WIDTH = 96
HEIGHT = 128

PREVIEW_SCALE = 3


# =====================================================
# VALORES Y PALOS
# =====================================================

VALUES = [
    "A", "2", "3", "4", "5", "6", "7",
    "8", "9", "10", "J", "Q", "K"
]

SUITS = {
    "Corazones ♥": "corazones",
    "Diamantes ♦": "diamantes",
    "Tréboles ♣": "treboles",
    "Picas ♠": "picas"
}


# =====================================================
# CREAR CARPETAS
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# NORMALIZAR IMÁGENES
# =====================================================

def normalize_image(image):

    h, w = image.shape[:2]

    scale = min(
        WIDTH / w,
        HEIGHT / h
    )

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC
    )

    canvas = np.full(
        (HEIGHT, WIDTH, 3),
        255,
        dtype=np.uint8
    )

    x = (WIDTH - new_w) // 2
    y = (HEIGHT - new_h) // 2

    canvas[
        y:y + new_h,
        x:x + new_w
    ] = resized

    return canvas


# =====================================================
# LEER ETIQUETAS EXISTENTES
# =====================================================

def load_existing_labels():

    existing = set()

    if LABELS_FILE.exists():

        with open(
            LABELS_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if row.get("source"):
                    existing.add(row["source"])

    return existing


# =====================================================
# GUARDAR ETIQUETA
# =====================================================

def save_label(source, filename, value, suit):

    file_exists = (
        LABELS_FILE.exists()
        and LABELS_FILE.stat().st_size > 0
    )

    with open(
        LABELS_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow([
                "source",
                "filename",
                "value",
                "suit"
            ])

        writer.writerow([
            source,
            filename,
            value,
            suit
        ])


# =====================================================
# INTERFAZ GRÁFICA
# =====================================================

class BlackjackDatasetApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "BLACKJACK - ENTRENAMIENTO DE CARTAS"
        )

        self.root.geometry("620x720")

        self.root.minsize(560, 650)

        self.root.configure(bg="#151515")

        self.existing = load_existing_labels()

        self.images = sorted(
            INPUT_DIR.rglob("*.png")
        )

        self.pending = [
            path
            for path in self.images
            if str(path.relative_to(BASE_DIR))
            not in self.existing
        ]

        self.index = 0

        self.skipped = []

        self.photo = None

        self.current_image = None

        self.create_interface()

        self.show_image()


    # =================================================
    # CREAR INTERFAZ
    # =================================================

    def create_interface(self):

        title = tk.Label(
            self.root,
            text="BLACKJACK",
            font=("Arial", 26, "bold"),
            fg="white",
            bg="#151515"
        )

        title.pack(pady=(20, 5))

        subtitle = tk.Label(
            self.root,
            text="PREPARACIÓN DEL DATASET",
            font=("Arial", 12),
            fg="#bbbbbb",
            bg="#151515"
        )

        subtitle.pack(pady=(0, 15))

        self.progress_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 12, "bold"),
            fg="#00dd88",
            bg="#151515"
        )

        self.progress_label.pack(pady=8)

        self.file_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 10),
            fg="white",
            bg="#151515",
            wraplength=560
        )

        self.file_label.pack(pady=5)

        # Área de imagen

        self.image_frame = tk.Frame(
            self.root,
            bg="#303030",
            width=330,
            height=400
        )

        self.image_frame.pack(pady=15)

        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(
            self.image_frame,
            bg="#303030",
            text="Cargando imagen...",
            fg="white"
        )

        self.image_label.pack(
            expand=True
        )

        # Selección de valor

        controls = tk.Frame(
            self.root,
            bg="#151515"
        )

        controls.pack(pady=10)

        tk.Label(
            controls,
            text="VALOR",
            font=("Arial", 11, "bold"),
            fg="white",
            bg="#151515"
        ).grid(
            row=0,
            column=0,
            padx=15,
            pady=5
        )

        tk.Label(
            controls,
            text="PALO",
            font=("Arial", 11, "bold"),
            fg="white",
            bg="#151515"
        ).grid(
            row=0,
            column=1,
            padx=15,
            pady=5
        )

        self.value_var = tk.StringVar()

        self.suit_var = tk.StringVar()

        self.value_selector = ttk.Combobox(
            controls,
            textvariable=self.value_var,
            values=VALUES,
            state="readonly",
            width=12
        )

        self.value_selector.grid(
            row=1,
            column=0,
            padx=15
        )

        self.suit_selector = ttk.Combobox(
            controls,
            textvariable=self.suit_var,
            values=list(SUITS.keys()),
            state="readonly",
            width=18
        )

        self.suit_selector.grid(
            row=1,
            column=1,
            padx=15
        )

        # Botones

        buttons = tk.Frame(
            self.root,
            bg="#151515"
        )

        buttons.pack(pady=20)

        self.save_button = tk.Button(
            buttons,
            text="GUARDAR Y SIGUIENTE",
            command=self.save_current,
            bg="#008855",
            fg="white",
            font=("Arial", 11, "bold"),
            width=22,
            height=2
        )

        self.save_button.grid(
            row=0,
            column=0,
            padx=8,
            pady=5
        )

        self.skip_button = tk.Button(
            buttons,
            text="OMITIR IMAGEN",
            command=self.skip_current,
            bg="#aa6600",
            fg="white",
            font=("Arial", 11, "bold"),
            width=18,
            height=2
        )

        self.skip_button.grid(
            row=0,
            column=1,
            padx=8,
            pady=5
        )

        self.retry_button = tk.Button(
            self.root,
            text="REVISAR IMÁGENES OMITIDAS",
            command=self.retry_skipped,
            bg="#333333",
            fg="white",
            font=("Arial", 10),
            width=32
        )

        self.retry_button.pack(pady=5)

        self.status_label = tk.Label(
            self.root,
            text="Sistema listo",
            font=("Arial", 10),
            fg="#aaaaaa",
            bg="#151515"
        )

        self.status_label.pack(pady=8)


    # =================================================
    # MOSTRAR IMAGEN
    # =================================================

    def show_image(self):

        total = len(self.pending)

        if self.index >= total:

            self.image_label.configure(
                image="",
                text="NO HAY MÁS IMÁGENES PENDIENTES",
                font=("Arial", 13, "bold"),
                fg="white"
            )

            self.photo = None

            self.file_label.configure(
                text="Proceso finalizado"
            )

            self.progress_label.configure(
                text=(
                    f"Etiquetadas: {len(self.existing)} | "
                    f"Omitidas: {len(self.skipped)}"
                )
            )

            self.save_button.configure(
                state="disabled"
            )

            self.skip_button.configure(
                state="disabled"
            )

            self.status_label.configure(
                text="Puedes cerrar la ventana."
            )

            return

        path = self.pending[self.index]

        image = cv2.imread(str(path))

        if image is None:

            self.skipped.append(path)

            self.index += 1

            self.show_image()

            return

        self.current_image = normalize_image(image)

        preview = cv2.resize(
            self.current_image,
            (
                WIDTH * PREVIEW_SCALE,
                HEIGHT * PREVIEW_SCALE
            ),
            interpolation=cv2.INTER_NEAREST
        )

        # Convertir BGR a RGB

        preview = cv2.cvtColor(
            preview,
            cv2.COLOR_BGR2RGB
        )

        # Convertir imagen a formato PPM
        # compatible con Tkinter

        h, w = preview.shape[:2]

        header = (
            f"P6 {w} {h} 255\n"
        ).encode("ascii")

        ppm_data = header + preview.tobytes()

        self.photo = tk.PhotoImage(
            data=ppm_data,
            format="PPM"
        )

        self.image_label.configure(
            image=self.photo,
            text=""
        )

        self.value_var.set("")

        self.suit_var.set("")

        self.file_label.configure(
            text=(
                f"Archivo: {path.name}\n"
                f"Origen: {path.parent.name}"
            )
        )

        self.progress_label.configure(
            text=(
                f"Imagen {self.index + 1} de {total} | "
                f"Etiquetadas: {len(self.existing)}"
            )
        )

        self.status_label.configure(
            text="Selecciona el valor y el palo."
        )


    # =================================================
    # GUARDAR IMAGEN
    # =================================================

    def save_current(self):

        value = self.value_var.get()

        suit_name = self.suit_var.get()

        if not value or not suit_name:

            messagebox.showwarning(
                "Datos incompletos",
                "Selecciona el valor y el palo."
            )

            return

        path = self.pending[self.index]

        source = str(
            path.relative_to(BASE_DIR)
        )

        suit = SUITS[suit_name]

        unique_id = hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()[:12]

        filename = (
            f"{value}_{suit}_{unique_id}.png"
        )

        destination = IMAGES_DIR / filename

        saved = cv2.imwrite(
            str(destination),
            self.current_image
        )

        if not saved:

            messagebox.showerror(
                "Error",
                "No se pudo guardar la imagen."
            )

            return

        try:

            save_label(
                source,
                filename,
                value,
                suit
            )

        except OSError as error:

            messagebox.showerror(
                "Error",
                f"No se pudo guardar la etiqueta:\n{error}"
            )

            return

        self.existing.add(source)

        print(
            f"[OK] {value} de {suit} guardado."
        )

        self.index += 1

        self.show_image()


    # =================================================
    # OMITIR IMAGEN
    # =================================================

    def skip_current(self):

        if self.index >= len(self.pending):
            return

        path = self.pending[self.index]

        self.skipped.append(path)

        print(
            f"[OMITIDA] {path.name}"
        )

        self.index += 1

        self.show_image()


    # =================================================
    # REVISAR IMÁGENES OMITIDAS
    # =================================================

    def retry_skipped(self):

        if not self.skipped:

            messagebox.showinfo(
                "Sin imágenes",
                "No hay imágenes omitidas."
            )

            return

        self.pending = [
            path
            for path in self.skipped
            if str(path.relative_to(BASE_DIR))
            not in self.existing
        ]

        self.skipped = []

        self.index = 0

        self.save_button.configure(
            state="normal"
        )

        self.skip_button.configure(
            state="normal"
        )

        self.show_image()


# =====================================================
# INICIAR PROGRAMA
# =====================================================

def main():

    print("\n====================================")
    print(" BLACKJACK - DATASET")
    print("====================================")

    root = tk.Tk()

    app = BlackjackDatasetApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()