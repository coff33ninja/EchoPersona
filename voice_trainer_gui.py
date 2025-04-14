# This script creates a GUI for training a voice cloning model using Tkinter.
# It allows users to input parameters such as character name, dataset path, output path, epochs, batch size, and learning rate.
# The GUI also provides a button to start the training process and displays the output in a text widget.
# Import necessary libraries
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Label, Button, Scale, Entry
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
def start_training(character, dataset_path, output_path, epochs, batch_size, learning_rate):
    """
    Start the training process with the provided parameters and display output in the GUI.

    Args:
        character (str): Character name.
        dataset_path (str): Path to the dataset.
        output_path (str): Path to save the trained model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for training.
    """
    try:
        command = (
            f"python voice_clone_train.py --dataset_path \"{dataset_path}\" "
            f"--output_path \"{output_path}\" --epochs {epochs} "
            f"--batch_size {batch_size} --learning_rate {learning_rate}"
        )
        run_command_with_output(command, output_text)
        messagebox.showinfo("Training", f"Training for {character} started successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start training: {e}")

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

# Dataset Path Selection
Label(root, text="Dataset Path:").pack(pady=5)

def select_dataset():
    path = filedialog.askdirectory()
    dataset_path_entry.delete(0, tk.END)
    dataset_path_entry.insert(0, path)

dataset_path_entry = Entry(root, width=50)
dataset_path_entry.pack(pady=5)
Button(root, text="Browse", command=select_dataset).pack(pady=5)

# Output Path Selection
Label(root, text="Output Path:").pack(pady=5)

def select_output():
    path = filedialog.askdirectory()
    output_path_entry.delete(0, tk.END)
    output_path_entry.insert(0, path)

output_path_entry = Entry(root, width=50)
output_path_entry.pack(pady=5)
Button(root, text="Browse", command=select_output).pack(pady=5)

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
    dataset_path = dataset_path_entry.get()
    output_path = output_path_entry.get()
    epochs = epochs_scale.get()
    batch_size = batch_size_scale.get()
    learning_rate = learning_rate_scale.get()

    if not character or not dataset_path or not output_path:
        messagebox.showerror("Error", "Please fill in all fields.")
        return

    start_training(character, dataset_path, output_path, epochs, batch_size, learning_rate)

Button(root, text="Start Training", command=on_start_training).pack(pady=20)

# Run the GUI
root.mainloop()