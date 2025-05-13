from tkinter import *
from PIL import Image, ImageTk
from MainMenu import MainMenuFrame
from EnterFeatures_GUI import EnterFeaturesFrame
from DisplayModels_GUI import DisplayModelsFrame

class App(Tk):
    def __init__(self):
        super().__init__()
        self.geometry('1200x700')
        self.resizable(False, False)
        self.config(background='#042940')
        # Change window title and icon
        self.title("Obesity Prediction System")
        self.iconbitmap('Assets/Icons/icon.ico')
        # Dictionary to store frames
        self.frames = {}
        self.init_frames()

    def init_frames(self):
        # Create and store frames
        self.frames["MainMenu"] = MainMenuFrame(self)
        self.frames["EnterFeatures"] = EnterFeaturesFrame(self)
        self.frames["DisplayModels"] = DisplayModelsFrame(self)

        # Show the Main Menu frame initially
        self.show_frame("MainMenu")

    def show_frame(self, frame_name):
        # Hide all frames
        for frame in self.frames.values():
            frame.pack_forget()

        # Show the requested frame
        frame = self.frames[frame_name]
        frame.pack(fill="both", expand=True)


if __name__ == "__main__":
    app = App()
    app.mainloop()