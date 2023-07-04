from tkinter import*
from random import*
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageGrab
import numpy as np
from numpy import asarray
from keras.models import load_model
import win32gui as wn

model = load_model("learn.h5")

def predict_digit(img):
    # изменение рзмера изобржений на 28x28
    img = img.resize((28,28))
    # конвертируем rgb в grayscale
    img = img.convert('L')
    img = np.array(img)
    # изменение размерности для поддержки модели ввода и нормализации
    img = img.reshape(1,28,28,1)
    img = img/255.0
    # предстказание цифры
    res = model.predict([img])[0]
    return np.argmax(res)

def draw(event):
    x1, y1 = (event.x - brush_size), (event.y - brush_size)
    x2, y2 = (event.x + brush_size), (event.y + brush_size)
    canv.create_oval(x1, y1, x2, y2, fill=color, width=0)

    
otv = True
def delete():
    global otv
    if otv: 
        canv.delete("all")

def hun():
    hwd = canv.winfo_id()
    rect = wn.GetWindowRect(hwd)
    im = ImageGrab.grab(rect)
    setc = predict_digit(im)
    txt = str(setc)
    tm["text"] = txt
    tm.update()
            
root = Tk()
root.geometry("1000x500")
root.resizable(False, False)

canv = Canvas(root, bg="white", width=500, height=500)
canv.pack(side=LEFT)

tm = Label(root, text="", font=("Arial", 48))
tm.pack(side=RIGHT)

brush_size = 20
color = "black"

bt_res = Button(root, text="Delete", padx="5", pady="3", command=delete)
bt_res.pack(side=BOTTOM)

bt_sev = Button(root, text="Recognize", padx="5", pady="3", command=hun)
bt_sev.pack(side=BOTTOM)

canv.bind("<B1-Motion>", draw)

root.mainloop()
