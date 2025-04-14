# This script creates a GUI for training a voice cloning model using Tkinter.
# It allows users to input parameters such as character name, dataset path, output path, epochs, batch size, and learning rate.
# The GUI also provides a button to start the training process and displays the output in a text widget.
# Import necessary libraries
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter.ttk import Label, Button, Scale
import threading
import subprocess
import os
import logging

# Function to run the training command and display output in the GUI
def run_command_with_output(command, output_text_widget):
    """
    Run a shell command and display its output in the GUI.

    Args:
        command (str): The command to run.
        output_text_widget (tk.Text): The text widget to display the output.
    """
    def target():
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        for line in process.stdout:
            output_text_widget.insert(tk.END, line)
            output_text_widget.see(tk.END)
        for line in process.stderr:
            output_text_widget.insert(tk.END, line)
            output_text_widget.see(tk.END)

    thread = threading.Thread(target=target)
    thread.start()

# Function to start the training process
def start_training(character, action, epochs, batch_size, learning_rate, text=None, file=None):
    """
    Start the training or dataset management process with the provided parameters and display output in the GUI.

    Args:
        character (str): Character name.
        action (str): Action to perform.
        epochs (int): Number of training epochs (for training).
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for training.
        text (str, optional): Text for 'test' or 'use' actions.
        file (str, optional): File path for actions like 'provide', 'augment', 'trim', 'quality'.
    """
    try:
        command = (
            f"python voice_trainer_cli.py --character \"{character}\" --action \"{action}\" "
        )

        if action in ["train"]:
            command += f"--epochs {epochs} --batch_size {batch_size} --learning_rate {learning_rate} "
        if action in ["test", "use"] and text:
            command += f"--text \"{text}\" "
        if action in ["provide", "augment", "trim", "quality"] and file:
            command += f"--file \"{file}\" "

        run_command_with_output(command, output_text)
        messagebox.showinfo("Action", f"Action '{action}' for {character} started successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start action '{action}': {e}")

# Function to dynamically fetch character names from the voice_datasets directory
def fetch_character_list():
    """
    Fetches a list of character names based on subdirectories in the voice_datasets folder.

    Returns:
        list: A sorted list of character names.
    """
    base_dir = "voice_datasets"
    try:
        if not os.path.exists(base_dir):
            return []
        return sorted(
            [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
        )
    except Exception as e:
        logging.error(f"Error fetching character list: {e}")
        return []

# GUI Setup
root = tk.Tk()
root.title("Voice Trainer GUI")
root.geometry("500x400")

# Character Selection
Label(root, text="Select Character:").pack(pady=5)
character_list = fetch_character_list()
character_var = tk.StringVar(root)
if character_list:
    character_var.set(character_list[0])  # Default to the first character
else:
    character_var.set("No characters found")
character_dropdown = tk.OptionMenu(root, character_var, *character_list)
character_dropdown.pack(pady=5)

# Add a dropdown menu for selecting actions
Label(root, text="Select Action:").pack(pady=5)
actions = [
    "record",
    "provide",
    "validate",
    "stats",
    "augment",
    "trim",
    "quality",
    "train",
    "test",
    "use",
]
action_var = tk.StringVar(root)
action_var.set(actions[0])  # Default to the first action
action_dropdown = tk.OptionMenu(root, action_var, *actions)
action_dropdown.pack(pady=5)

# Dataset Path Display
Label(root, text="Dataset Path (Default):").pack(pady=5)
dataset_path_label = Label(root, text="voice_datasets/<character>", relief=tk.SUNKEN, anchor="w")
dataset_path_label.pack(pady=5, fill=tk.X)

# Output Path Display
Label(root, text="Output Path (Default):").pack(pady=5)
output_path_label = Label(root, text="trained_models/<character>", relief=tk.SUNKEN, anchor="w")
output_path_label.pack(pady=5, fill=tk.X)

# Training Parameters
Label(root, text="Training Parameters:").pack(pady=5)

Label(root, text="Epochs:").pack(pady=5)
epochs_scale = Scale(root, from_=1, to=1000, orient=tk.HORIZONTAL)
epochs_scale.set(500)
epochs_scale.pack(pady=5)

Label(root, text="Batch Size:").pack(pady=5)
batch_size_scale = Scale(root, from_=1, to=128, orient=tk.HORIZONTAL)
batch_size_scale.set(16)
batch_size_scale.pack(pady=5)

Label(root, text="Learning Rate:").pack(pady=5)
learning_rate_scale = tk.Scale(root, from_=0.0001, to=0.01, resolution=0.0001, orient=tk.HORIZONTAL)
learning_rate_scale.set(0.0002)
learning_rate_scale.pack(pady=5)

# Add a text widget to display CLI output
output_label = Label(root, text="CLI Output:")
output_label.pack(pady=5)

output_text = tk.Text(root, height=10, width=60)
output_text.pack(pady=5)

# Start Training Button
def on_start_training():
    character = character_var.get()
    action = action_var.get()
    epochs = epochs_scale.get()
    batch_size = batch_size_scale.get()
    learning_rate = learning_rate_scale.get()

    if not character:
        messagebox.showerror("Error", "Please select a character.")
        return

    if action in ["test", "use"]:
        text = simpledialog.askstring("Input", "Enter text for the action:")
        if not text:
            messagebox.showerror("Error", "Text is required for this action.")
            return
        start_training(character, action, epochs, batch_size, learning_rate, text=text)
    elif action in ["provide", "augment", "trim", "quality"]:
        file = filedialog.askopenfilename(title="Select a file for the action")
        if not file:
            messagebox.showerror("Error", "File is required for this action.")
            return
        start_training(character, action, epochs, batch_size, learning_rate, file=file)
    else:
        start_training(character, action, epochs, batch_size, learning_rate)

Button(root, text="Start Training", command=on_start_training).pack(pady=20)

# Run the GUI
root.mainloop()