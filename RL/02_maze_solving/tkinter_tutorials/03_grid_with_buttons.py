from tkinter import *
import time 
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

block1 = w.create_rectangle( 0, 0, int(canvas_width/4), int(canvas_height/4), fill='yellow')
block2 = w.create_rectangle( int(canvas_width/4), int(canvas_height/4), int(canvas_width/2), int(canvas_height/2), fill='red')
block3 = w.create_rectangle( 0, int(canvas_height/4), int(canvas_width/4), int(canvas_height/2), fill='green')
block4 = w.create_rectangle( int(canvas_width*3/4), int(canvas_height/2), int(canvas_width), int(canvas_height*3/4), fill='blue')

master.title('Grid')

def start():
    #start a timer untill stop function is called
    global start_time
    block4 = w.create_rectangle( int(canvas_width*3/4), int(canvas_height/2), int(canvas_width), int(canvas_height*3/4), fill='white')
    start_time = time.time()  
    return 

def stop():
    #stop a timer 
    global stop_time
    stop_time = time.time()  
    master.destroy()
    return 

button1 = Button(master, text='Start', width=25, height=25, command=start, activebackground="lightgreen")
button2 = Button(master, text='Stop', width=25, height=25, command=stop, activebackground="coral")
button1.pack()
button2.pack()

mainloop()

print("The execution time: {} (seconds)".format(round((stop_time - start_time), 5)))