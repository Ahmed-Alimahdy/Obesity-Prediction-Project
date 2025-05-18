import random
from tkinter import *
from tkinter import messagebox
import pandas as pd
import SVM_model as svm
import Xgboost_model as xg
import Logistic_regression as log_reg
import KNN_model as knn
import Decision_tree as dt
import randomforest_model as rf
from sklearn.discriminant_analysis import StandardScaler
class EnterFeaturesFrame(Frame):
    recommendation = {"Insufficient_Weight": ["Increase your caloric intake and consider a balanced diet.",
                                              "Include a good source of protein in every meal to support healthy muscle gain.",
                                              "Focus on strength training exercises to build lean muscle instead of just doing cardio.",
                                              "Consult a doctor to rule out any medical issues if you're struggling to gain weight despite eating more."],
                     "Normal_Weight": ["Maintain your current diet and exercise routine to keep your weight stable.",
                                      "Consider incorporating a variety of foods to ensure you're getting all the necessary nutrients.",
                                      "Stay active with a mix of cardio and strength training to support overall health.",
                                      "Regular check-ups with a healthcare provider can help monitor your health."],
                    "Obesity_Type_I": ["Focus on a balanced diet with controlled portion sizes to manage your weight.",
                                       "Incorporate regular physical activity, aiming for at least 150 minutes of moderate exercise per week.",
                                       "Consider working with a nutritionist to create a personalized meal plan.",
                                       "Regular check-ups with a healthcare provider can help monitor your health."],
                    "Obesity_Type_II": ["Adopt a balanced diet with controlled portion sizes to manage your weight.",
                                       "Incorporate regular physical activity, aiming for at least 150 minutes of moderate exercise per week.",
                                       "Consider working with a nutritionist to create a personalized meal plan.",
                                       "Regular check-ups with a healthcare provider can help monitor your health."],
                    "Obesity_Type_III": ["Focus on a balanced diet with controlled portion sizes to manage your weight.",
                                       "Incorporate regular physical activity, aiming for at least 150 minutes of moderate exercise per week.",
                                       "Consider working with a nutritionist to create a personalized meal plan.",
                                       "Regular check-ups with a healthcare provider can help monitor your health."],
                    "Overweight_Level_I": ["Focus on a balanced diet with controlled portion sizes to manage your weight.",
                                           "Incorporate regular physical activity, aiming for at least 150 minutes of moderate exercise per week.",
                                           "Consider working with a nutritionist to create a personalized meal plan.",
                                           "Regular check-ups with a healthcare provider can help monitor your health."],
                    "Overweight_Level_II": ["Focus on a balanced diet with controlled portion sizes to manage your weight.",
                                            "Incorporate regular physical activity, aiming for at least 150 minutes of moderate exercise per week.",
                                            "Consider working with a nutritionist to create a personalized meal plan.",
                                            "Regular check-ups with a healthcare provider can help monitor your health."],}
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
        canvas.create_rectangle(0, 23, 410, 80, fill='#005c53')  # vecBg1
        
        # ---- Title ---- #
        title = Label(self,
                      text="ObesiTrack AI",
                      font=('cal sans', 26, 'bold'),
                      fg='white',
                      bg='#005c53',
                      )
        title.place(x=70, y=25)

        # ---- HBox Setup ---- #
        features_frame1 = Frame(self, bg='#042940')
        features_frame1.place(x=0, y=170, width=1700, height=300)
        features_frame2 = Frame(self, bg='#042940')
        features_frame2.place(x=160, y=280, width=1700, height=100)
        # Center the third frame horizontally
        features_frame3 = Frame(self, bg='#042940')
        features_frame3.place(x= 550, y=400, anchor='center',  width=400, height=100)

        # List of features (each feature is a tuple of label text, widget type, and options)
        features1 = [
             ("Physical activity", OptionMenu, {"options": ["0", "1","2","3"], "default": "0"}),
            ("Daily water intake", OptionMenu, {"options": ["1","2","3"], "default": "1"}),
            ("vegetable consumption",OptionMenu, {"options": ["1","2","3"], "default": "1"}),
            ("Time spent using technology", OptionMenu, {"options": ["0","1","2","3"], "default": "0"}),   
            ("Number of meals", OptionMenu, {"options": ["1", "2","3",">3"], "default": "3"}),
        ]
        features2 = [
            ("Gender", OptionMenu, {"options": ["Male", "Female"], "default": "Male"}),
            ("Height", Entry, {"width": 8}),
            ("Weight", Entry, {"width": 8}),
            ("Age", Entry, {"width": 8}),
           
        ]
        features3 = [
            ('''Select a Model''', OptionMenu, {'options': ['Logistic Regression',
                                                        'SVM', 'KNN', 'Random Forest',
                                                        'XG Boost', 'Naive Bayes','Decision Tree',],
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

        spacing = 100
        x_position = spacing  # Start with the first gap

        for feature in features:
            label_text, widget_type, widget_options = feature

            # Handle special cases
            if label_text == "Number of meals per day":
                feature_width= 180
            elif label_text == "Physical activity frequency":
                feature_width = 180
            elif label_text == "Monitoring calorie intake":
                feature_width = 180
            elif label_text == "Daily water intake":
                feature_width = 100
            elif label_text == "Time spent using technology":
                feature_width = 180
            elif label_text == '''Select a Model''':
                feature_width = 300
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
                                  bg='#042940',  wraplength=feature_width, 
                      justify='center')
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
     self.result_label.config(text="processing...")
     self.result_label.update_idletasks()  # Show "processing..." immediately

     def do_model():
      new_input = {
            "FCVC": float(self.feature_values["Physical activity"].get()),
            "Meal_Regularity": 1 if self.feature_values["Number of meals"].get() == "3"
                            else 0.5 if self.feature_values["Number of meals"].get() == "2"
                            else 0,
            "Weight": float(self.feature_values["Weight"].get()),
            "Height": float(self.feature_values["Height"].get())/100,
            "Gender": 1 if self.feature_values["Gender"].get() == "Male" else 0,
            "CaloricIntake": float(self.feature_values["Daily water intake"].get()) +
                            float(self.feature_values["Physical activity"].get()) +
                            float(self.feature_values["vegetable consumption"].get()),
            "TUE": float(self.feature_values["Time spent using technology"].get()),
            "Age": float(self.feature_values["Age"].get())
        }
      if self.feature_values['Select a Model'].get() == 'SVM':
            result = svm.load_model(new_input)
      if self.feature_values['Select a Model'].get() == 'XG Boost':
            result = xg.load_model(new_input)
      if self.feature_values['Select a Model'].get() == 'Logistic Regression':
            result = log_reg.load_model(new_input)
      if self.feature_values['Select a Model'].get() == 'KNN':
            result = knn.load_model(new_input)
      if self.feature_values['Select a Model'].get() == 'Random Forest':
            result = rf.load_model(new_input)
      if self.feature_values['Select a Model'].get() == 'Naive Bayes':
            result = rf.load_model(new_input)
      if self.feature_values['Select a Model'].get() == 'Decision Tree':
            result = dt.load_model(new_input) 
        
      self.result_label.config(text=f"Result: {result}")

      self.recommendation_label.config(text=f"Recommendation: {random.choice(self.recommendation[result])}")
     # Wait 300ms before running the model, so "processing..." is visible
     self.after(300, do_model)
      


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

        # ---- Result Label ---- #
        self.result_label = Label(self,
                                  text="",
                                  font=('instrument sans', 14, 'bold'),
                                  bg='#042940',
                                  fg='white')
        self.result_label.place(relx=0.5, y=500, anchor='center')  # Below the button

         # ---- Result Label ---- #
        self.result_label = Label(self,
                                  text="",
                                  font=('instrument sans', 14, 'bold'),
                                  bg='#042940',
                                  fg='white')
        self.result_label.place(relx=0.5, y=500, anchor='center')  # Below the button

        # ---- Recommendation Label ---- #
        self.recommendation_label = Label(self,
                                          text="",
                                          font=('instrument sans', 12),
                                          bg='#042940',
                                          fg='#9fc131',
                                          wraplength=700,
                                          justify='center')
        self.recommendation_label.place(relx=0.5, y=550, anchor='center') 

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