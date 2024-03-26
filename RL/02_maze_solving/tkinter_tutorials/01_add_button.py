import tkinter as tk
import time

r = tk.Tk()
r.title('Button')


def start():
    #start a timer untill stop function is called
    global start_time
    start_time = time.time()  
    return 

def stop():
    #stop a timer 
    global stop_time
    stop_time = time.time()  

    r.destroy()

    return 

button1 = tk.Button(r, text='Start', width=25, height=25, command=start, activebackground="lightgreen")
button2 = tk.Button(r, text='Stop', width=25, height=25, command=stop, activebackground="coral")
button1.grid()
button2.grid()

r.mainloop()

print("The execution time: {} (seconds)".format(round((stop_time - start_time), 5)))