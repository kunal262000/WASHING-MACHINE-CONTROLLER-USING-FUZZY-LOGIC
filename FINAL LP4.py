import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image

##############

top = tk.Tk()
top.geometry('1366x768')
top.title('Washing Machine Using fuzzy Logic')
load = Image.open('images/back2.png')
render = ImageTk.PhotoImage(load)
img = Label(top,image=render)
img.place(x=0,y=0)
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 35, 'bold'))
# label.place(row=0,column=1)
sign_image = Label(top)
plate_image_ocv = Label(top)
plate_image_cnn = Label(top)




########png########
#bg = PhotoImage(file="C:\\images\\t1 .png")
#bg_lable = Label(top,image=bg)
#bg_lable.place(x=0,y=0)
###################################

top.resizable(True,True)
DP = Label(top,text = 'Dirt %      ',font=("Arial",25))
DP.place(x =530,y = 70)
name=Entry(top,font=("Arial",25),width=5)
name.place(x=730,y=70)
GP = Label(top,text = 'Grease %',font=("Arial",25))
GP.place(x =530,y = 150,)
number = Entry(top,font=("Arial",25),width=5)
number.place(x = 730,y = 150)

# lblb.place(x=200, y=600)


detra1 = Label(top,font=("Arial",25))
detra1.place(x=660,y=260)


###################################################3
'''
Fuzzy controller for washing machine


step 1 : identify i/p and o/p descriptors
dirt            |   grease
sd->small dirt  |   ng->no grease
md->medium dirt |   mg->medium grease
ld->large dirt  |   lg->large grease

vs->very short time
s->short time
m->medium time
l->long time
vl->very long time

---------------------------------------------------------
step2 : membership function for i/p and o/p

mf for dirt :- 

mf(sd)(x)=  (50-x)/50
mf(md)(x)=  x/50        ,0<=x<=50
            (100-x)/50  ,50<=x<=100
mf(ld)(x)=  (x-50)/50   ,50<=x<=100

mf for grease :-

mf(ng)(y)=  (50-y)/50   ,0<=y<=50
mf(mg)(y)=  y/50        ,0<=y<=50
            (100-y)/50  ,50<=y<=100
mf(lg)(y)=  (y-50)/50   ,50<=y<=100

mf for wash time :-

mf(vs)(z)=  (10-z)/10   ,0<=z<=10
mf(s)(z)=   z/10        ,0<=z<=10
            (25-z)/15   ,10<=z<=25
mf(m)(z)=   (z-10)/15   ,10<=z<=25
            (40-z)/15   ,25<=z<=40
mf(l)(z)=   (z-25)/15   ,25<=z<=40
            (60-z)/15   ,40<=z<=60
mf(vl)(z)=  (z-40)/20   ,40<=z<=60
---------------------------------------------------------
'''
def membership_functions(value,variable,descriptor):
    if(variable=='dirt'):
        if(descriptor=='sd' and value>=0 and value<=50):
            return (50-value)/50
        elif(descriptor=='md' and value>=0 and value<=50):
            return value/50
        elif(descriptor=='md' and value>50 and value<=100):
            return (100-value)/50
        elif(descriptor=='ld' and value>=50 and value<=100):
            return (value-50)/50
    elif(variable=='grease'):
        if(descriptor=='ng' and value>=0 and value<=50):
            return (50-value)/50
        elif(descriptor=='mg' and value>=0 and value<=50):
            return value/50
        elif(descriptor=='mg' and value>50 and value<=100):
            return (100-value)/50
        elif(descriptor=='lg' and value>=50 and value<=100):
            return (value-50)/50
'''
step 3 : rule base
        ng  mg  lg
        0   1   2
0   sd  vs  m   l
1   md  s   m   l
2   ld  m   l   vs
'''
def rule_base(dirt,grease):
    if(dirt=='sd'):
        dirt = 0
    elif(dirt=='md'):
        dirt = 1
    elif(dirt=='ld'):
        dirt = 2
    if(grease=='ng'):
        grease = 0
    elif(grease=='mg'):
        grease = 1
    elif(grease=='lg'):
        grease = 2
    rules = [['vs','m','l'],['s','m','l'],['m','l','vl']]
    return rules[dirt][grease]
'''
step 4 : rule evalutaion
'''
def mapping_dirt(percentage):
    temp,temp1 = [],[]
    if(percentage>=0 and percentage<=50):
        # temp.append('sd')
        temp.append(membership_functions(percentage,'dirt','sd'))
        temp1.append('sd')
    if(percentage>=0 and percentage<=100):
        # temp.append('md')
        temp.append(membership_functions(percentage,'dirt','md'))
        temp1.append('md')
    if(percentage>=50 and percentage<=100):
        # temp.append('ld')
        temp.append(membership_functions(percentage,'dirt','ld'))
        temp1.append('ld')
    return temp,temp1

def mapping_grease(percentage):
    temp,temp1 = [],[]
    if(percentage>=0 and percentage<=50):
        temp.append(membership_functions(percentage,'grease','ng'))
        temp1.append('ng')
    if(percentage>=0 and percentage<=100):
        temp.append(membership_functions(percentage,'grease','mg'))
        temp1.append('mg')
    if(percentage>=50 and percentage<=100):
        temp.append(membership_functions(percentage,'grease','lg'))
        temp1.append('lg')
    return temp,temp1
def mapping_washing_time(variable,value):
    temp = []
    if(variable=='vs'):
        temp.append((-1)*((value*10)-10))
    elif(variable=='s'):
        temp.append(value*10)
        temp.append((-1)*((value*15)-25))
    elif(variable=='m'):
        temp.append((value*15)+10)
        temp.append((-1)*((value*15)-40))
    elif(variable=='l'):
        temp.append((value*15)+25)
        temp.append((-1)*((value*20)-60))
    elif(variable=='vl'):
        temp.append((value*20)+40)
    return sum(temp)/len(temp)









###################################################







def main():


    from tkinter import messagebox
    try:
        searchTerm = name.get()
        searchTerm = int(searchTerm)

        if (searchTerm < 0 or searchTerm > 100):
            messagebox.showerror("Incomplete", " Please Enter Dirt Percentage Value Between 0-100")
            name.focus()
            number.delete(0, END)
            return
        NoOfTerms = number.get()
        NoOfTerms = int(NoOfTerms)



        if (NoOfTerms < 0 or NoOfTerms > 100):
            messagebox.showerror("Incomplete", "Please Enter the Grease Percentage Value Between 0-100")
            number.delete(0, END)
            number.focus()
            return

        NoOfTerms = int(NoOfTerms)
        searchTerm = int(searchTerm)
    except:
        None
    import itertools as it
    # m,n = int(input("Enter dirt percentage : ")),int(input("Enter grease percentage : "))
    l1,l1_rule = mapping_dirt(searchTerm)
    l2,l2_rule = mapping_grease(NoOfTerms)
    a = list(map(min,list(it.product(l1,l2))))
    b = list(it.product(l1_rule,l2_rule))
    temp1,temp2 = list(map(str,b[a.index(max(a))]))



    a1 = (mapping_washing_time(rule_base(temp1,temp2),max(a)))
    a1 = "{:.2f}".format(a1)
    detra1.configure(text=a1)




classify_b = Button(top, text="Calculate Time", command=lambda:main(), padx=10, pady=5)
classify_b.configure(background='#364156', foreground='white', font=('arial', 15, 'bold'))
classify_b.place(x=600, y=320)





top.mainloop()









