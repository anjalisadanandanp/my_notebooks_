# Python GUI – tkinter

Python offers multiple options for developing GUI (Graphical User Interface).   \
Out of all the GUI methods, tkinter is the most commonly used method.   \
It is a standard Python interface to the Tk GUI toolkit shipped with Python.    \
Python with tkinter is the fastest and easiest way to create the GUI applications.  \
Creating a GUI using tkinter is an easy task.   

<hr/>

Importing tkinter is same as importing any other module in the Python code.     \
Note that the name of the module in Python 2.x is ‘Tkinter’ and in Python 3.x it is ‘tkinter’.  

        import tkinter

There are two main methods used which the user needs to remember while creating the Python application with GUI.

1. **Tk(screenName=None,  baseName=None,  className=’Tk’,  useTk=1)** \
    To create a main window, tkinter offers a method ‘Tk(screenName=None,  baseName=None,  className=’Tk’,  useTk=1)’. To change the name of the window, you can change the className to the desired one. The basic code used to create the main window of the application is:

        m=tkinter.Tk()           where m is the name of the main window object

2. **mainloop()**  \
    There is a method known by the name mainloop() is used when your application is ready to run. mainloop() is an infinite loop used to run the application, wait for an event to occur and process the event as long as the window is not closed.

        m.mainloop()


To summarize:

        import tkinter
        m = tkinter.Tk()
        '''
        widgets are added here
        '''
        m.mainloop()

<hr/>

tkinter also offers access to the geometric configuration of the widgets which can organize the widgets in the parent windows. \
There are mainly three geometry manager classes class.

1. pack() method: It organizes the widgets in blocks before placing in the parent widget.

2. grid() method:It organizes the widgets in grid (table-like structure) before placing in the parent widget.

3. place() method:It organizes the widgets by placing them on specific positions directed by the programmer.

<hr/>

There are a number of widgets which you can put in your tkinter application. Some of the major widgets are explained below:

1. **Button** \
    To add a button in your application, this widget is used. The general syntax is:
        
        w=Button(master, option=value)

    master is the parameter used to represent the parent window.

    There are number of options which are used to change the format of the Buttons. 

    Number of options can be passed as parameters separated by commas. Some of them are listed below.

    *activebackground*: to set the background color when button is under the cursor. \
    *activeforeground*: to set the foreground color when button is under the cursor. \
    *bg*: to set the normal background color. \
    *command*: to call a function. \
    *font*: to set the font on the button label. \
    *image*: to set the image on the button. \
    *width*: to set the width of the button. \
    *height*: to set the height of the button.


        import tkinter as tk
        r = tk.Tk()
        r.title('Counting Seconds')
        button = tk.Button(r, text='Stop', width=25, command=r.destroy)
        button.pack()
        r.mainloop()


2. **Canvas** \
    It is used to draw pictures and other complex layout like graphics, text and widgets. \

    The general syntax is:

        w = Canvas(master, option=value)          master is the parameter used to represent the parent window.


    There are number of options which are used to change the format of the widget. \ Number of options can be passed as parameters separated by commas. 
    
    Some of them are listed below.

    *bd*: to set the border width in pixels. \
    *bg*: to set the normal background color. \
    *cursor*: to set the cursor used in the canvas. \
    *highlightcolor*: to set the color shown in the focus highlight. \
    *width*: to set the width of the widget. \
    *height*: to set the height of the widget.


        from tkinter import *
        master = Tk()
        w = Canvas(master, width=40, height=60)
        w.pack()
        canvas_height=20
        canvas_width=200
        y = int(canvas_height / 2)
        w.create_line(0, y, canvas_width, y )
        mainloop()