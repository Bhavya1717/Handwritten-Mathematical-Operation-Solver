import numpy as np
import cv2
import os
from tensorflow.keras.models import model_from_json  # type: ignore
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageOps
import PIL.Image
import PIL.ImageTk

# Load the model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
json_file = open('model_final.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_final.weights.h5")

# Global variables for painting
last_x, last_y = None, None
canvas_image = None
draw = None
photo_img = None  # To hold a reference to the image so it doesn't get garbage-collected


# Preprocessing function for the image
def preprocess_image(img):
    img = ~img
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    rects = [cv2.boundingRect(c) for c in cnt]
    bool_rect = [[0 for _ in range(len(rects))] for _ in range(len(rects))]

    # Removing overlapping rectangles
    for i in range(len(rects)):
        for j in range(len(rects)):
            if i != j:
                if (rects[i][0] < rects[j][0] + rects[j][2] + 10 and
                        rects[j][0] < rects[i][0] + rects[i][2] + 10 and
                        rects[i][1] < rects[j][1] + rects[j][3] + 10 and
                        rects[j][1] < rects[i][1] + rects[i][3] + 10):
                    bool_rect[i][j] = 1

    dump_rect = []
    for i in range(len(rects)):
        for j in range(len(rects)):
            if bool_rect[i][j] == 1:
                area1 = rects[i][2] * rects[i][3]
                area2 = rects[j][2] * rects[j][3]
                if area1 == min(area1, area2):
                    dump_rect.append(rects[i])

    final_rect = [r for r in rects if r not in dump_rect]
    train_data = []

    for r in final_rect:
        x, y, w, h = r
        im_crop = thresh[y:y + h + 10, x:x + w + 10]
        im_resize = cv2.resize(im_crop, (28, 28))
        im_resize = np.reshape(im_resize, (28, 28, 1))
        train_data.append(im_resize)

    return train_data


# Function to predict the math operation
def predict_operation(train_data):
    s = ''
    for img in train_data:
        img = np.array(img).reshape(1, 28, 28, 1)
        result = np.argmax(loaded_model.predict(img), axis=-1)[0]
        if result == 10:
            s += '-'
        elif result == 11:
            s += '+'
        elif result == 12:
            s += '*'
        else:
            s += str(result)

    try:
        result_str = f"Predicted: {s}\nThe Output is: {eval(s)}"
    except Exception:
        result_str = "Error in evaluating expression!"
    result_label.config(text=result_str)


# Function to handle file selection and display the image on the canvas
def choose_file():
    global photo_img
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            train_data = preprocess_image(img)
            predict_operation(train_data)

            # Display the image on the canvas
            loaded_img = PIL.Image.open(file_path)
            loaded_img = loaded_img.resize((canvas.winfo_width(), canvas.winfo_height()), PIL.Image.LANCZOS)  # Updated line
            photo_img = PIL.ImageTk.PhotoImage(loaded_img)  # Keep a reference of the image
            canvas.create_image(0, 0, image=photo_img, anchor="nw")
            canvas.image = photo_img  # Store a reference to the image
        else:
            result_label.config(text="Error: Could not load image!")

# Function to clear the canvas
def clear_canvas():
    global canvas_image, draw
    canvas.delete("all")
    canvas_image = PIL.Image.new("RGB", (canvas.winfo_width(), canvas.winfo_height()), "white")
    draw = ImageDraw.Draw(canvas_image)


# Function to save canvas drawing and predict
def save_and_predict():
    canvas_image.save("canvas_image.jpg")
    img = cv2.imread("canvas_image.jpg", cv2.IMREAD_GRAYSCALE)
    train_data = preprocess_image(img)
    predict_operation(train_data)


# Function to handle drawing on the canvas
def paint(event):
    global last_x, last_y
    x1, y1 = (event.x - 2), (event.y - 2)
    x2, y2 = (event.x + 2), (event.y + 2)

    if last_x is None and last_y is None:
        last_x, last_y = event.x, event.y  # store the initial position

    # Draw a line from the last point to the current point
    canvas.create_line(last_x, last_y, event.x, event.y, fill="black", width=5)
    draw.line([last_x, last_y, event.x, event.y], fill="black", width=5)

    # Update the last positions
    last_x, last_y = event.x, event.y


# Function to reset the last_x, last_y on mouse release
def reset(event):
    global last_x, last_y
    last_x, last_y = None, None  # reset to None when the user releases the mouse button


# Function to resize canvas dynamically when the window is resized
def resize_canvas(event):
    global canvas_image, draw
    new_width = event.width
    new_height = event.height

    # Resize the canvas image
    resized_image = canvas_image.resize((new_width, new_height))
    canvas_image = resized_image
    draw = ImageDraw.Draw(canvas_image)
    canvas.config(width=new_width, height=new_height)


# Create main window
root = tk.Tk()
root.title("Math Operation Predictor")

# Make the window responsive
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Create canvas for drawing and displaying images
canvas = tk.Canvas(root, bg="white")
canvas.grid(row=0, column=0, pady=20, padx=20, sticky="nsew")
canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", reset)

# Create PIL image to draw on the canvas
canvas_image = PIL.Image.new("RGB", (400, 400), "white")
draw = ImageDraw.Draw(canvas_image)

# Bind canvas resizing
canvas.bind("<Configure>", resize_canvas)

# Buttons for actions
btn_predict = tk.Button(root, text="Predict from Drawing", command=save_and_predict)
btn_predict.grid(row=1, column=0, pady=10, sticky="ew")

btn_clear = tk.Button(root, text="Clear Whiteboard", command=clear_canvas)
btn_clear.grid(row=2, column=0, pady=10, sticky="ew")

btn_choose_file = tk.Button(root, text="Choose Image File", command=choose_file)
btn_choose_file.grid(row=3, column=0, pady=10, sticky="ew")

# Label to display prediction result
result_label = tk.Label(root, text="Draw or Choose an Image and Predict!", font=("Arial", 16))
result_label.grid(row=4, column=0, pady=10, sticky="ew")

# Make the window resizable
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Start the GUI event loop
root.mainloop()
