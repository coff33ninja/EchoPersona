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
from tkinter import ttk  # Import ttk for modern widgets

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

# Use a Notebook widget to organize sections
notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Create frames for different sections
parameters_frame = ttk.Frame(notebook)
output_frame = ttk.Frame(notebook)
notebook.add(parameters_frame, text="Parameters")
notebook.add(output_frame, text="Output")

# Character Selection
character_list = fetch_character_list()
character_var = tk.StringVar(root)
if character_list:
    character_var.set(character_list[0])  # Default to the first character
else:
    character_var.set("No characters found")

ttk.Label(parameters_frame, text="Select Character:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
character_dropdown = ttk.Combobox(parameters_frame, textvariable=character_var, values=character_list, state="readonly")
character_dropdown.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

# Add a dropdown menu for selecting actions
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

ttk.Label(parameters_frame, text="Select Action:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
action_dropdown = ttk.Combobox(parameters_frame, textvariable=action_var, values=actions, state="readonly")
action_dropdown.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

# Dataset Path Display
ttk.Label(parameters_frame, text="Dataset Path (Default):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
dataset_path_label = ttk.Label(parameters_frame, text="voice_datasets/<character>", relief=tk.SUNKEN, anchor="w")
dataset_path_label.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

# Output Path Display
ttk.Label(parameters_frame, text="Output Path (Default):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
output_path_label = ttk.Label(parameters_frame, text="trained_models/<character>", relief=tk.SUNKEN, anchor="w")
output_path_label.grid(row=3, column=1, sticky="ew", padx=5, pady=5)

# Training Parameters
ttk.Label(parameters_frame, text="Epochs:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
epochs_scale = Scale(parameters_frame, from_=1, to=1000, orient=tk.HORIZONTAL)
epochs_scale.set(500)
epochs_scale.grid(row=4, column=1, sticky="ew", padx=5, pady=5)

# Add a label to display the current value of the epochs slider
epochs_value_label = ttk.Label(parameters_frame, text=f"Epochs: {epochs_scale.get()}")
epochs_value_label.grid(row=4, column=2, sticky="w", padx=5, pady=5)

# Update the label dynamically as the slider is moved
def update_epochs_label(value):
    epochs_value_label.config(text=f"Epochs: {value}")

# Configure the epochs slider to call the update function
epochs_scale.config(command=update_epochs_label)

ttk.Label(parameters_frame, text="Batch Size:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
batch_size_scale = Scale(parameters_frame, from_=1, to=128, orient=tk.HORIZONTAL)
batch_size_scale.set(16)
batch_size_scale.grid(row=5, column=1, sticky="ew", padx=5, pady=5)

# Add a label to display the current value of the batch size slider
batch_size_value_label = ttk.Label(parameters_frame, text=f"Batch Size: {batch_size_scale.get()}")
batch_size_value_label.grid(row=5, column=2, sticky="w", padx=5, pady=5)

# Update the label dynamically as the slider is moved
def update_batch_size_label(value):
    batch_size_value_label.config(text=f"Batch Size: {value}")

# Configure the batch size slider to call the update function
batch_size_scale.config(command=update_batch_size_label)

ttk.Label(parameters_frame, text="Learning Rate:").grid(row=6, column=0, sticky="w", padx=5, pady=5)
learning_rate_scale = tk.Scale(parameters_frame, from_=0.0001, to=0.01, resolution=0.0001, orient=tk.HORIZONTAL)
learning_rate_scale.set(0.0002)
learning_rate_scale.grid(row=6, column=1, sticky="ew", padx=5, pady=5)

# Add a label to display the current value of the learning rate slider
learning_rate_value_label = ttk.Label(parameters_frame, text=f"Learning Rate: {learning_rate_scale.get():.4f}")
learning_rate_value_label.grid(row=6, column=2, sticky="w", padx=5, pady=5)

# Update the label dynamically as the slider is moved
def update_learning_rate_label(value):
    learning_rate_value_label.config(text=f"Learning Rate: {float(value):.4f}")

# Configure the learning rate slider to call the update function
learning_rate_scale.config(command=update_learning_rate_label)

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

    # Switch to the CLI output tab
    notebook.select(output_frame)

start_button = ttk.Button(parameters_frame, text="Start Training", command=on_start_training)
start_button.grid(row=7, column=0, columnspan=3, pady=10)

# Add a stop button and functionality to jump to CLI output
stop_button = ttk.Button(parameters_frame, text="Stop Training", command=None)  # Placeholder for stop functionality
stop_button.grid(row=8, column=0, columnspan=3, pady=10)

# CLI Output Section
ttk.Label(output_frame, text="CLI Output:").pack(anchor="w", padx=5, pady=5)
output_text = tk.Text(output_frame, height=10, width=60)
output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Adjust column weights for better resizing
parameters_frame.columnconfigure(1, weight=1)
output_frame.columnconfigure(0, weight=1)

# Run the GUI
root.mainloop()