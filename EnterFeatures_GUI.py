from tkinter import *
from tkinter import messagebox

class EnterFeaturesFrame(Frame):
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
        canvas.create_rectangle(0, 38, 500, 135, fill='#9fc131')  # vecBg2
        canvas.create_rectangle(0, 23, 510, 120, fill='#005c53')  # vecBg1
        
        # ---- Title ---- #
        title = Label(self,
                      text="Enter Features:",
                      font=('cal sans', 48, 'bold'),
                      fg='white',
                      bg='#005c53')
        title.place(x=40, y=25)

        # ---- HBox Setup ---- #
        features_frame1 = Frame(self, bg='#042940')
        features_frame1.place(x=0, y=170, width=1200, height=100)
        features_frame2 = Frame(self, bg='#042940')
        features_frame2.place(x=0, y=250, width=1200, height=100)
        # Center the third frame horizontally
        features_frame3 = Frame(self, bg='#042940')
        features_frame3.place(relx=0.5, y=380, anchor='center',  width=450, height=100)

        # List of features (each feature is a tuple of label text, widget type, and options)
        features1 = [
            ("Gender", OptionMenu, {"options": ["Male", "Female"], "default": "Male"}),
            ("Family History with Overweight", OptionMenu, {"options": ["Yes", "No"], "default": "Yes"}),
            ("FAVC", OptionMenu, {"options": ["Yes", "No"], "default": "Yes"})
        ]
        features2 = [
            ("Height", Entry, {"width": 8}),
            ("Weight", Entry, {"width": 8}),
            ("FCVC", Entry, {"width": 8}),
            ("SCC", OptionMenu, {"options": ["Yes", "No"], "default": "No"})
        ]
        features3 = [
            ('''Select a Model''', OptionMenu, {'options': ['Logistic Regression',
                                                        'SVM', 'KNN', 'Random Forest',
                                                        'XG Boost', 'Naive Bayes'],
                                                'default': 'Logistic Regression'})
        ]
        # Dictionary to store the values of all features
        self.feature_values = {}

        # Create the first HBox with listeners
        self.create_hbox_with_listeners(features1, features_frame1)
        self.create_hbox_with_listeners(features2, features_frame2)
        self.create_hbox_with_listeners(features3, features_frame3)
        
        # Create all other buttons
        self.create_buttons()

    def create_hbox_with_listeners(self, features, frame):
        default_feature_width = 100  # Default width for most features
        frame.update_idletasks()  # Ensure the frame's dimensions are updated

        spacing = 30
        x_position = spacing  # Start with the first gap

        for feature in features:
            label_text, widget_type, widget_options = feature

            # Handle special cases
            if label_text == "Family History with Overweight":
                feature_width = 260
            elif label_text == '''Select a Model''':
                feature_width = 400
            else:
                feature_width = default_feature_width

            # Create a frame for each feature
            feature_frame = Frame(frame, bg='#042940')
            feature_frame.place(x=x_position, y=0, width=feature_width, height=100)

            # Create the label
            feature_label = Label(feature_frame,
                                  text=label_text,
                                  font=('instrument sans', 12),
                                  fg='white',
                                  bg='#042940')
            feature_label.pack(pady=(0, 5))

            # Create the widget and add listeners
            if widget_type == OptionMenu:
                var = StringVar(self)
                var.set(widget_options["default"])  # Set default value
                self.feature_values[label_text] = var  # Store the variable in the dictionary
                dropdown = OptionMenu(feature_frame, var, *widget_options["options"])
                dropdown.config(font=('instrument sans', 12),
                                bg='#005c53',
                                fg='white',
                                width=200,
                                activebackground='#9FC131',
                                activeforeground='white',
                                highlightthickness=0,
                                bd=0)
                dropdown.pack()
            elif widget_type == Entry:
                # Validation function to allow only floating-point numbers (no empty values)
                def validate_numeric_input(P):
                    try:
                        if P == "":  # Disallow empty input
                            return False
                        float(P)  # Check if the input can be converted to a float
                        return True
                    except ValueError:
                        return False

                # Register the validation function
                validate_command = self.register(validate_numeric_input)
                var = StringVar(self)
                self.feature_values[label_text] = var  # Store the variable in the dictionary
                textbox = Entry(feature_frame,
                                textvariable=var,  # Bind the variable to the Entry widget
                                font=('instrument sans', 12),
                                bg='#005c53',
                                fg='white',
                                width=widget_options["width"],
                                insertbackground='white',
                                justify='center',
                                validate='key',  # Trigger validation on key press
                                validatecommand=(validate_command, '%P'))  # Pass the new value to the validation function
                textbox.pack()

            # Update the x_position for the next feature
            x_position += feature_width + spacing

    def validate_and_process_values(self):
        # Check if any feature value is empty
        for feature, value in self.feature_values.items():
            if value.get() == "":
                messagebox.showwarning("Incomplete Data", "Please fill in all the fields before proceeding.")
                return
        self.process_values()
    
    # TODO: Apply data processing + recommendation
    def process_values(self):
        for feature, value in self.feature_values.items():
            print(f"{feature}: {value.get()}")

    def create_buttons(self):
        # ---- Display Result Button ---- #
        display_result_button = Button(self,
                                       text="Display Result",
                                       font=('instrument sans', 14, 'bold'),
                                       bg='#9fc131',
                                       fg='white',
                                       activebackground='#005c53',
                                       activeforeground='white',
                                       relief='flat',
                                       cursor="hand2",
                                       command=self.validate_and_process_values)  # Call validate_and_process_values when clicked
        display_result_button.place(relx=0.5, y=450, anchor='center', width=180, height=40)

        # Add hover effect for Display Result button
        add_hover_effect(display_result_button, '#b4e197', '#9fc131')

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