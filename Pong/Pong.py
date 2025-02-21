import tkinter as tk
import random

root = tk.Tk()
root.title("Pong")
root.resizable(0, 0)
canvas = tk.Canvas(root, width=800, height=600,bg='black')
canvas.pack()

paddle = canvas.create_rectangle(40, 250, 60, 350, fill= 'white')
ball = canvas.create_oval(390, 290, 410, 310, fill= 'white')


direction = 0
x = 9
y = 9

def move_p(event):
    px0, py0, px1, py1 = canvas.coords(paddle)
    if (event.keysym == 'Up' and py0 > 0):
        canvas.move(paddle, 0, -40) 
    elif (event.keysym == 'Down' and py1 < 600):
        canvas.move(paddle, 0, 40)

    update_poz()

def collision_top():
    global direction
    if(direction == 3):
        direction = 2
    else: direction = 1

def collision_bottom():
    global direction
    if(direction == 1):
        direction = 0
    else: direction = 3

def collision_right():
    global direction
    if(direction == 0):
        direction = 3
    else: direction = 2

def collision_left():
    global direction
    if(direction == 3):
        direction = 0
    else: direction = 1


def paddle_collision():
    global x, y
    bx0, by0, bx1, by1 = canvas.coords(ball)   
    px0, py0, px1, py1 = canvas.coords(paddle)
    
    collided = False

    if( (bx0 < 60 and bx0 > 40) and 
       ( (by0 > py0 and by0 < py1) or (by1 > py0 and by1 < py1) )):
        collided = True
        collision_left()
    return collided
   

def move_ball():
    global x, y
    x = abs(x)
    y = abs(y)
    bx0, by0, bx1, by1 = canvas.coords(ball)
    collided = paddle_collision()
    if(collided == False):
        if(bx0 - x < 0):
            reset_ball()
        elif(bx1 + x > 800):
            collision_right()
        elif(by0 - y < 0):
            collision_top()
        elif(by1 + y > 600):
            collision_bottom()

    if(direction == 0):
        y = -y
    elif(direction == 2):
        x = -x
    elif(direction == 3):
        x= -x
        y= -y

    canvas.move(ball, x, y)
    root.after(30 , move_ball)
    update_ball()

def update_poz():
    px0, py0, px1, py1 = canvas.coords(paddle)  
    bx0, by0, bx1, by1 = canvas.coords(ball)

    print(f"Paddle Poz: \n(b0: {px0}, {py0})  \n(b1: {px1}, {py1})")
    print(f"Ball Position: \n(b0: {bx0}, {by0})  \n(b1: {bx1}, {by1})")

def update_ball():
    bx0, by0, bx1, by1 = canvas.coords(ball)
    #print(f"Ball Position: \n(b0: {bx0}, {by0})  \n(b1: {bx1}, {by1})")

def reset_ball():
    rand = random.randint(0, 1)
    global direction
    direction = rand
    canvas.moveto(ball, 400, 300)

root.bind_all("<Up>", move_p)
root.bind_all("<Down>", move_p)
move_ball()

root.mainloop()
