from tkinter import *
import tkinter.filedialog


def xz():
    filename = tkinter.filedialog.askopenfilename()
    if filename != '':
        lb.config(text="您选择的文件是："+filename)
        show(filename)
    else:
        lb.config(text="您没有选择任何文件")


def show(string):
    print(string)


if __name__ == '__main__':
    root = Tk()
    root.title('鞋印图像开集分类')
    root.geometry('500x80')
    lb = Label(root, text='请选择待测试图像')
    lb.pack()

    btn = Button(root, text="选择并分类", command=xz)
    btn.pack()
    root.mainloop()
