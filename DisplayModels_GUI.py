from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Progressbar, Style

class DisplayModelsFrame(Frame):
    def __init__(self, parent):
        super().__init__(parent, bg='#042940')
        self.parent = parent

        # ---- Canvas ---- #
        canvas = Canvas(self,
                        width=1200,
                        height=700,
                        background='#042940',
                        highlightthickness=0)
        canvas.place(x=0, y=0)
        canvas.create_rectangle(0, 38, 400, 95, fill='#9fc131')  # vecBg2
        canvas.create_rectangle(0, 23, 410, 85, fill='#005c53')  # vecBg1

        # ---- Title ---- #
        title = Label(self,
                      text="Models Accuracy:",
                      font=('cal sans', 26, 'bold'),
                      fg='white',
                      bg='#005c53')
        title.place(x=40, y=25)

        # Create frame (VBox)
        VBox = Frame(self, bg='#042940')
        VBox.place(x=100, y=170, width=1000, height=400)

        # TODO: Replace with your model accuracy
        Models = [
            ('Logistic Regression', '97%'),
            ('SVM', '99%'),
            ('KNN', '88%'),
            ('Random Forest', '100%'),
            ('XG Boost', '100%'),
            ('Naive Bayes', '68%'),
            ('Decision tree', '93%'),
        ]
        self.display_models(Models, VBox)
        self.create_buttons()

    def display_models(self, Models, VBox):
        y_position = 0
        self.style = Style()  # Create a single Style instance for all progress bars

        # Configure the progress bar style
        progress_bar_style = "Custom.Horizontal.TProgressbar"
        self.style.configure(progress_bar_style,
                             troughcolor='#042940',
                             background='#9FC131')

        for model in Models:
            label_text, label_accuracy = model

            # Create a frame for each model inside VBox
            feature_frame = Frame(VBox, bg='#042940')
            feature_frame.place(x=0, y=y_position, width=1000, height=80)

            # Create the label inside the feature frame
            model_label = Label(feature_frame,
                                text=f"{label_text}: {label_accuracy}",
                                font=('instrument sans', 12, 'bold'),
                                fg='white',
                                bg='#042940',
                                anchor='w')
            model_label.place(x=10, y=10, width=400, height=30)

            # Create a progress bar inside the feature frame
            progress_value = int(label_accuracy.strip('%'))  # Convert accuracy to an integer
            progress_bar = Progressbar(feature_frame,
                                       orient="horizontal",
                                       length=500,
                                       mode="determinate",
                                       style=progress_bar_style)
            progress_bar.place(x=420, y=15, width=500, height=30)
            progress_bar["value"] = progress_value  # Set the progress bar value

            y_position += 50  # Add spacing between frames

    def create_buttons(self):
        # ---- Return to Main Menu Button ---- #
        return_to_main_menu_button = Button(self,
                                            text="Back",
                                            font=('instrument sans', 14, 'bold'),
                                            bg='#005c53',
                                            fg='white',
                                            activebackground='#9FC131',
                                            activeforeground='white',
                                            relief='flat',
                                            cursor="hand2",
                                            command=lambda: self.parent.show_frame("MainMenu"))
        return_to_main_menu_button.place(x=40, y=640, width=80, height=40)

        # Add hover effect for Return to Main Menu button
        add_hover_effect(return_to_main_menu_button, '#007a73', '#005c53')


def add_hover_effect(button, hover_bg, default_bg):
    def on_enter(event):
        button.config(bg=hover_bg)

    def on_leave(event):
        button.config(bg=default_bg)

    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)