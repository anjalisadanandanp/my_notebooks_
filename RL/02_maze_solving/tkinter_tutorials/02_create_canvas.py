from tkinter import *
master = Tk()
canvas_height=400
canvas_width=400

w = Canvas(master, width=canvas_width, height=canvas_height)
w.pack()

y = int(canvas_height / 4)
w.create_line(0, y, canvas_width, y )

y = int(canvas_height / 2)
w.create_line(0, y, canvas_width, y )

y = int(canvas_height* 3/ 4)
w.create_line(0, y, canvas_width, y )

x = int(canvas_width / 4)
w.create_line(x, 0, x, canvas_height )

x = int(canvas_width / 2)
w.create_line(x, 0, x, canvas_height )

x = int(canvas_width* 3/ 4)
w.create_line(x, 0, x, canvas_height )

block1 = w.create_rectangle( int(canvas_width/4), int(canvas_height/4), int(canvas_width/2), int(canvas_height/2), fill='black')
block2 = w.create_rectangle( int(canvas_width*3/4), int(canvas_height/2), int(canvas_width), int(canvas_height*3/4), fill='red')

master.title('Grid')

mainloop()