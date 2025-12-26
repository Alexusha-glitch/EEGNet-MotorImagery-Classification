import time
import tkinter as tk
import winsound
import random

root = tk.Tk()
root.attributes('-fullscreen', True)
root.update_idletasks()

width = root.winfo_screenwidth()
height = root.winfo_screenheight()

canvas = tk.Canvas(root, bg="black", highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)

x1 = None
x2 = None
arrow = None

time.sleep(1)

def show_cross():
    global x1, x2
    size = 100

    x = width/2
    y = height/2

    x1 = canvas.create_line(x - size, y - size, x + size, y + size, width=3, fill="white")
    x2 = canvas.create_line(x - size, y + size, x + size, y - size, width=3, fill="white")

    winsound.Beep(400, 400)

    root.after(2000, show_arrow)


def show_arrow():
    global arrow
    size = 150

    x = width/2
    y = height/2

    i = random.randint(1, 2)

    if i == 1:
        arrow = canvas.create_line(x + 200, y, x + 200 + size, y, arrow=tk.LAST, fill="white", width=5)
    else:
        arrow = canvas.create_line(x - 200, y, x - 200 - size, y, arrow=tk.LAST, fill="white", width=5)

    root.after(1250, hide_arrow)


def hide_cross():
    global x1, x2

    canvas.delete(x1)
    canvas.delete(x2)
    x1 = x2 = None

    root.after(2250, show_cross)


def hide_arrow():
    global arrow

    canvas.delete(arrow)
    arrow = None

    root.after(2750, hide_cross)

show_cross()
root.mainloop()