from tkinter import *
from PIL import Image, ImageTk

class MainMenuFrame(Frame):
    def __init__(self, parent):
        super().__init__(parent, bg='#042940')
        self.parent = parent

        # Add a label for the Main Menu
        canvas = Canvas(self,
                        width=1200,
                        height=700,
                        background='#042940',
                        highlightthickness=0)
        canvas.place(x=0, y=0)
        canvas.create_rectangle(0, 0, 1200, 135, fill='#9fc131')

        # Load the image
        logo_image = Image.open("Assets/Icons/Logo.png")
        logo_image = logo_image.resize((100, 100), Image.Resampling.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_image)
        logo_label = Label(self, image=logo_photo, bg='#9fc131')
        logo_label.image = logo_photo  # Keep a reference to avoid garbage collection
        logo_label.place(x=140, y=20)

        # Add the title label
        main_menu_label = Label(self,
                                text="Obesity Prediction System",
                                font=('instrument sans', 50, 'bold'),
                                fg='white',
                                bg='#9fc131')
        main_menu_label.place(relx=0.57, y=70, anchor="center")

        # List of buttons with their labels and corresponding frame names
        # TODO: Edit the corresponding frame name to your model
        buttons = [
            ("Enter Features", "EnterFeatures"),
            ("Display Models Accuracy", "DisplayModels"),
        ]

        # Create buttons dynamically using a loop
        for i, (label, frame_name) in enumerate(buttons):
            button = Button(self,
                            text=label,
                            font=('instrument sans', 14, 'bold'),
                            bg='#005c53',  # Default background color
                            fg='white',
                            activebackground='#9FC131',
                            activeforeground='white',
                            relief='flat',
                            cursor='hand2',
                            command=lambda frame_name=frame_name: self.parent.show_frame(frame_name))
            button.place(relx=0.5, y=350 + i * 70, anchor="center", width=400, height=50)

            # Add hover effect
            def on_enter(event, btn=button):
                btn.config(bg='#007a73')
            def on_leave(event, btn=button):
                btn.config(bg='#005c53')
            button.bind("<Enter>", on_enter)
            button.bind("<Leave>", on_leave)