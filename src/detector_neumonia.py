from tkinter import END, StringVar, Text, Tk, filedialog, font, ttk
from tkinter.messagebox import WARNING, askokcancel, showinfo

from PIL import Image, ImageTk

from src.integrator import predict
from src.load_model import model_fun
from src.read_img import read_image_file
from src.report import generate_pdf_report, save_result_csv


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")
        self.root.geometry("815x560")
        self.root.resizable(False, False)

        bold_font = font.Font(weight="bold")

        self.image_label = ttk.Label(
            self.root, text="Imagen Radiográfica", font=bold_font
        )
        self.heatmap_label = ttk.Label(
            self.root, text="Imagen con Heatmap", font=bold_font
        )
        self.result_label = ttk.Label(self.root, text="Resultado:", font=bold_font)
        self.id_label = ttk.Label(self.root, text="Cédula Paciente:", font=bold_font)
        self.title_label = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=bold_font,
        )
        self.probability_label = ttk.Label(
            self.root, text="Probabilidad:", font=bold_font
        )

        self.patient_id = StringVar()
        self.id_entry = ttk.Entry(self.root, textvariable=self.patient_id, width=10)

        self.image_panel = Text(self.root, width=31, height=15)
        self.heatmap_panel = Text(self.root, width=31, height=15)
        self.result_text = Text(self.root)
        self.probability_text = Text(self.root)

        self.predict_button = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.load_button = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.delete_button = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.pdf_button = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.save_button = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        self.image_label.place(x=110, y=65)
        self.heatmap_label.place(x=545, y=65)
        self.result_label.place(x=500, y=350)
        self.id_label.place(x=65, y=350)
        self.title_label.place(x=122, y=25)
        self.probability_label.place(x=500, y=400)
        self.predict_button.place(x=220, y=460)
        self.load_button.place(x=70, y=460)
        self.delete_button.place(x=670, y=460)
        self.pdf_button.place(x=520, y=460)
        self.save_button.place(x=370, y=460)
        self.id_entry.place(x=200, y=350)
        self.result_text.place(x=610, y=350, width=90, height=30)
        self.probability_text.place(x=610, y=400, width=90, height=30)
        self.image_panel.place(x=65, y=90)
        self.heatmap_panel.place(x=500, y=90)

        self.id_entry.focus_set()

        self.model = None
        self.array = None
        self.report_id = 0

    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
            ),
        )
        if filepath:
            self.array, preview = read_image_file(filepath)
            preview = preview.resize((250, 250), Image.Resampling.LANCZOS)
            self.input_photo = ImageTk.PhotoImage(preview)
            self.image_panel.delete("1.0", END)
            self.image_panel.image_create(END, image=self.input_photo)
            self.predict_button["state"] = "normal"

    def run_model(self):
        if self.model is None:
            self.model = model_fun()
        self.diagnosis, self.probability, heatmap = predict(self.array, self.model)
        overlay = Image.fromarray(heatmap)
        overlay = overlay.resize((250, 250), Image.Resampling.LANCZOS)
        self.heatmap_photo = ImageTk.PhotoImage(overlay)
        self.heatmap_panel.delete("1.0", END)
        self.heatmap_panel.image_create(END, image=self.heatmap_photo)
        self.result_text.delete("1.0", END)
        self.result_text.insert(END, self.diagnosis)
        self.probability_text.delete("1.0", END)
        self.probability_text.insert(END, f"{self.probability:.2f}%")

    def save_results_csv(self):
        save_result_csv(self.id_entry.get(), self.diagnosis, self.probability)
        showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        generate_pdf_report(self.root, self.report_id)
        self.report_id += 1
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            self.id_entry.delete(0, "end")
            self.result_text.delete("1.0", "end")
            self.probability_text.delete("1.0", "end")
            self.image_panel.delete("1.0", "end")
            self.heatmap_panel.delete("1.0", "end")
            self.array = None
            self.predict_button["state"] = "disabled"
            showinfo(title="Borrar", message="Los datos se borraron con éxito")


def main():
    app = App()
    app.root.mainloop()


if __name__ == "__main__":
    main()
