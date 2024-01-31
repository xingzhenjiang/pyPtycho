# This is a  ptychographic data processing program.
# -----------------------------------------------------------------------------#
# Copyright (c) 2022 Xing Zhenjiang
# -----------------------------------------------------------------------------#

__authors__ = "Xing Zhenjiang"
__date__ = "Date : 30.03.2022"
__version__ = "1.0.1"

# import sys
from tkinter import *
import tkinter.filedialog
import time
from PIL import Image, ImageTk
import numpy as np
# import skimage
# from libtiff import TIFF
import tifffile
import matplotlib.pyplot as plt

after_id = None
position_file_type = 0

def gettime():
    global after_id
    # timestr = time.strftime("%H:%M:%S")  # 获取当前的时间并转化为字符串
    # pic1.configure(text=timestr)  # 重新设置标签文本
    # pic2.configure(text=timestr)  # 重新设置标签文本
    # s = 'Time: ' + pic1.cget('text') + '\n'
    # txt.insert(1.0, s)  # 追加显示运算结果
    # txt.delete(101.0, END)
    # time.sleep(5)
    # after_id = root.after(10000, gettime)  # 每隔1s调用函数 gettime 自身获取时间
    #


def stop_gettime():
    global after_id
    root.after_cancel(after_id)  # 中止after
    after_id = None


def run1():
    if Para4_btn1['bg'] == 'green':
        Para4_btn1['bg'] = 'red'
        Para4_btn1['text'] = 'Stop'
        root.update()
        Reconstruction()
    elif Para4_btn1['bg'] == 'red':
        Para4_btn1['bg'] = 'green'
        Para4_btn1['text'] = 'Run'
        root.update()
        stop_gettime()

def Mysel():
    if rd_var.get()==0:
        position_file_type = 0
    else:
        position_file_type = 1

def select_path():
    path_ = tkinter.filedialog.askopenfilename()
    path_ = path_.replace('/', '\\')
    Para3_inp1_path.set(path_)

def select_path_position():
    path_ = tkinter.filedialog.askopenfilename(filetypes=[('Text files', '*.txt;*.dat'), ('All files', '*.*')])
    path_ = path_.replace('/', '\\')
    Para1_inp17_path.set(path_)

def save_path():
    path_ = tkinter.filedialog.asksaveasfilename()
    path_ = path_.replace('/', '\\')
    Para3_inp2_path.set(path_)
    
def position_generator_para(n_grid_x, n_grid_y, step_pixel, N_matrix, scan_type):
    X, Y = np.meshgrid(np.array(range(0,n_grid_x)),np.array(range(0,n_grid_x)))
    X[1::2] = np.fliplr(X[1::2])*(scan_type=='s' or scan_type=='S')+X[1::2]*(scan_type=='z' or scan_type=='Z')
    n_grid = n_grid_x * n_grid_y
    pos0=np.round(N_matrix/2-step_pixel*np.array([n_grid_x,n_grid_y])/2)
    pos_relative = (np.array([X,Y]).reshape(2,n_grid)*step_pixel+np.tile(pos0,(n_grid,1)).T).T
    pos_relative = (np.array([X,Y]).reshape(2,n_grid)*step_pixel+np.tile(pos0,(n_grid,1)).T).T
    return pos_relative

def position_generator_file(position_file_path):
    a=1

def Reconstruction():
# This is a ptychograhic phase retrial program.
    time_start = time.time()
    txt.insert(1.0, 'Start...')  # 追加显示运算结果
    txt.delete(101.0, END)
    root.update()

    # h=6.62607015e-34
    # c=2.99792458e8
    # e=1.60217733e-19
    E_xray = float(Para1_inp1.get()) # eV   
    ccd_pixel = float(Para1_inp2.get()) #um
    d_sd = float(Para1_inp3.get())   # mm
    d_probe = float(Para1_inp4.get()) #um    
    d_zp = float(Para1_inp7.get())   #um
    delta_r = float(Para1_inp8.get()) #nm
    d_zp_in = float(Para1_inp9.get()) #um
    scan_type = Para1_inp13.get()
    scan_step = float(Para1_inp14.get()) #um
    n_grid_x = int(Para1_inp15.get())
    n_grid_y = int(Para1_inp16.get())
    
    alpha = float(Para2_inp1.get())
    beta = float(Para2_inp2.get())    
    n_iteration_max = int(Para2_inp3.get()) 
    n_show = int(Para2_inp4.get()) 
    n_save = int(Para2_inp5.get())
    n_shrink = int(Para2_inp6.get()) 
    delta_shrink = 0.1
    # lamda=1e9*h*c/(E_xray*e)
    Iccd = tifffile.imread(Para3_inp1.get())
    lamda = 1239.84/E_xray
    d_sd = d_sd*1e6
    ccd_pixel = ccd_pixel*1e3
    d_zp = d_zp*1e3
    d_zp_in = d_zp_in*1e3
    d_probe = d_probe*1e3
    r_probe = d_probe/2
    n_ccd=Iccd.shape[1]

    f_zp = d_zp*delta_r/lamda
    d_fs = d_probe*f_zp/d_zp
    d_fd = d_fs + d_sd
    obj_pixel = lamda*d_sd/(n_ccd*ccd_pixel)
    scan_step = scan_step*1e3
    step_pixel = scan_step/obj_pixel
    r_probe_pixel = round(r_probe/obj_pixel)

    area_scan = np.round((np.transpose([n_grid_x,n_grid_y])-1)*step_pixel)
    area_scan = 2*(area_scan/2)
    N_matrix = np.round(area_scan+3*r_probe_pixel)
    N_matrix = 2*np.round(N_matrix/2)*(N_matrix>=n_ccd)+np.transpose([n_ccd,n_ccd])*(N_matrix<=n_ccd)
    if rd_var.get():
        pos_relative = position_generator_para(n_grid_x, n_grid_y, step_pixel, N_matrix, scan_type)
    else:
        pos_relative = position_generator_para(n_grid_x, n_grid_y, step_pixel, N_matrix, scan_type)

    Iprobe = np.mean(Iccd,0)
    I_noise = 0.01*np.max(Iprobe)*np.ones([n_ccd,n_ccd])
    X, Y = np.meshgrid(np.array(range(0,n_ccd)),np.array(range(0,n_ccd)))
    xpos0 = np.round(np.sum(X*Iprobe*(Iprobe>=0.1*np.max(Iprobe)))/np.sum(Iprobe*(Iprobe>=0.1*np.max(Iprobe))))
    ypos0 = np.round(np.sum(Y*Iprobe*(Iprobe>=0.1*np.max(Iprobe)))/np.sum(Iprobe*(Iprobe>=0.1*np.max(Iprobe))))
    # print(type(Iccd[0][0][0]))
    Iprobe = np.roll(np.roll(Iprobe,int(np.round(n_ccd/2)-xpos0)).T,int(np.round(n_ccd/2)-ypos0)).T
    Iprobe = Iprobe*(Iprobe>=0.1*np.max(Iprobe))
    R_grid = np.sqrt((X-np.round(n_ccd/2))**2+(Y-np.round(n_ccd/2))**2)
    probe_ccd_guess = np.sqrt(Iprobe)*np.exp(1j*np.pi*(R_grid*ccd_pixel)**2/(lamda*d_fd))
    pro_samp0 = (lamda*d_sd)*np.exp(-1j*np.pi*R_grid*obj_pixel**2/(lamda*d_sd))*(np.fft.ifft2(np.fft.ifftshift(probe_ccd_guess*np.exp(-1j*np.pi*(R_grid*ccd_pixel)**2/(lamda*d_sd)))))
    pro_samp1 = np.fft.fftshift(pro_samp0)
    if np.sum(abs(pro_samp1[int(np.round(n_ccd/4)):int(np.round(n_ccd*3/4))][int(np.round(n_ccd/4)):int(np.round(n_ccd*3/4))]))>np.sum(abs(pro_samp0[int(np.round(n_ccd/4)):int(np.round(n_ccd*3/4))][int(np.round(n_ccd/4)):int(np.round(n_ccd*3/4))])) :
        pro_samp0 = pro_samp1
        r_ccd = round(n_ccd/2)

    Robj1 = np.floor(N_matrix/2-n_ccd/2)
    Robj2 = np.floor(N_matrix/2+n_ccd/2-1)

    f_probe_guess = 1/(lamda*d_sd)*pro_samp0
    f_obj_guess = np.ones([int(N_matrix[0]),int(N_matrix[1])])*np.exp(1j*0.1*np.random.rand(int(N_matrix[0]),int(N_matrix[1])))

    for i in range(n_grid_x*n_grid_y):
        Iccd[i] =np.sqrt(np.roll(np.roll(Iccd[i],int(np.round(n_ccd/2)-xpos0)).T,int(np.round(n_ccd/2)-ypos0)).T)

    plt.ion()  # 开启一个画图的窗口进入交互模式，用于实时更新数据
    for i in range(n_iteration_max):
        txt.insert(1.0, 'Iteration:  ' + str(i+1) + '\n')  # 追加显示迭代次数
        txt.insert(1.0, 'Time :' + str(int(time.time()-time_start)) + 's' + '\n')  # 追加显示用时
        txt.delete(101.0, END)
        root.update()

        np.random.seed(0)  
        s = np.random.permutation(range(n_grid_x*n_grid_y))
        for j in range(n_grid_x*n_grid_y):
            k=s[j]
            ddxy = pos_relative[k]-N_matrix/2
            f_obj_guess_cf = np.roll(np.roll(f_obj_guess,-int(ddxy[0])).T,-int(ddxy[1])).T
            f_obj_guess1 = f_obj_guess_cf[int(Robj1[0]):int(Robj2[0]+1),int(Robj1[1]):int(Robj2[1]+1)]
            ESW_guess = f_probe_guess*f_obj_guess1
            A_guess = np.fft.fftshift(np.fft.fft2(ESW_guess))
            A_ccd = Iccd[k]
    #         A_ccd = np.sqrt(np.roll(np.roll(np.abs(Iccd[k]*(Iccd[k]>=0.005*np.max(Iccd[k]))),int(np.round(N_matrix[0]/2)-xpos0)).T,int(np.round(N_matrix[1]/2)-ypos0)).T)
            I_guess = I_noise + A_guess * A_guess.conj()
            I_noise = I_noise*A_ccd**2/(I_guess)
            A_guess = A_guess*A_ccd/np.sqrt(I_guess)
            E_guess_new = np.fft.ifft2(np.fft.ifftshift(A_guess))
            delta_ESW = E_guess_new - ESW_guess

            f_probe_guess = f_probe_guess+beta*f_obj_guess1.conj()/np.max(np.abs(f_obj_guess1))**2*delta_ESW
            f_obj_guess1 = f_obj_guess1+alpha*f_probe_guess.conj()/np.max(np.abs(f_probe_guess))**2*delta_ESW
            # f_probe_guess = f_probe_guess+beta*f_obj_guess1*(delta_ESW*ESW_guess)
            # f_obj_guess1 = f_obj_guess1+alpha*f_probe_guess*(delta_ESW*ESW_guess)


            f_obj_guess_cf[int(Robj1[0]):int(Robj2[0]+1),int(Robj1[1]):int(Robj2[1]+1)] = f_obj_guess1
            f_obj_guess = np.roll(np.roll(f_obj_guess_cf,int(ddxy[0])).T,int(ddxy[1])).T
        f_obj_guess_abs = np.abs(f_obj_guess)
        f_obj_guess_max = np.max(f_obj_guess_abs)
        win_size_min = min(root.winfo_width(), root.winfo_height())
        pic1.width = 0.45 * win_size_min
        pic1.height = 0.45 * win_size_min
        img_obj_abs = ImageTk.PhotoImage(Image.fromarray(np.uint8(256*f_obj_guess_abs/f_obj_guess_max)).resize((int(pic1.width), int(pic1.height)), Image.Resampling.LANCZOS))
        pic1.configure(image=img_obj_abs)
        pic1.image = img_obj_abs

        pic2.width = 0.45 * win_size_min
        pic2.height = 0.45 * win_size_min
        f_obj_guess_angle = np.angle(f_obj_guess)
        img_obj_angle = ImageTk.PhotoImage(Image.fromarray(np.uint8(256*(f_obj_guess_angle+np.pi)/2+np.pi)).resize((int(pic2.width), int(pic2.height)), Image.Resampling.LANCZOS))
        pic2.configure(image=img_obj_angle)
        pic2.image = img_obj_angle
        root.update()

        if np.mod(i,n_shrink)==0 and i<=50:
            if f_obj_guess_max*(1-delta_shrink) > 1:
                f_obj_guess=f_obj_guess*(1-delta_shrink)
            elif f_obj_guess_max*(1+delta_shrink)<1:
                f_obj_guess=f_obj_guess*(1+delta_shrink)
            else:
                f_obj_guess=f_obj_guess/f_obj_guess_max

# Press the green button to run the script.
if __name__ == '__main__':
    root = Tk()
    root.title('pyPtycho 1.0.1')
    # root.configure(bg='WhiteSmoke')
    root.geometry('1000x600')  # 主窗口
    root.iconbitmap(r'.\ptycho.ico')
    canvas = Canvas(root)
    canvas.pack(fill=BOTH, expand=Y)

    menubar = Menu(root)
    # 创建子菜单
    menu1 = Menu(menubar, tearoff=0)
    menu5 = Menu(menubar, tearoff=0)
    for i in ['New...', 'Open...', 'Save...', 'Save as...']:
        menu1.add_command(label=i)
    for i in ['Information']:
        menu5.add_command(label=i)
    # 为顶级菜单实例添加菜单，并级联相应的子菜单实例
    menubar.add_cascade(label='File', menu=menu1)
    menubar.add_cascade(label='Edit')  # 这里省略了menu属性，没有将后面三个选项与子菜单级联
    menubar.add_cascade(label='Run')
    menubar.add_cascade(label='Tools')
    menubar.add_cascade(label='About', menu=menu5)
    root['menu'] = menubar

    # pic1 display
    win_width = root.winfo_width()
    win_height = root.winfo_height()
    win_size_min  = min(win_width,win_height)
    pic1 = Label(root, text='', relief=SUNKEN)
                 # fg='blue',
                 # font=("Times", 25),
                 # bg='white')
    # pic1.pack()
    pic1.place(relx=0.05, rely=0.02, width=0.45*win_size_min, height=0.45*win_size_min)
    # gettime()

    # pic2 display
    pic2 = Label(root, text='', relief=SUNKEN)
    pic2.place(relx=0.05, rely=0.5, width=0.45*win_size_min, height=0.45*win_size_min)

    # Para1 display
    lbx0 = 0.5
    lby0 = 0.08
    Len_lbx = 0.12
    Len_lby = 0.05
    Len_inpx = 0.05
    Len_inpy = Len_lby - 0.01

    Para1 = Label(root, text='',
                  # bg='gray',
                  relief=SUNKEN)
    Para1.place(relx=lbx0, rely=lby0, relwidth=(Len_lbx + Len_inpx + 0.02) * 2, relheight=(Len_lby + 0.01) * 8)

    Para1_lb = Label(root, text='Experimental parameters',
                     font=("Times", 15))
    Para1_lb.place(relx=lbx0, rely=lby0 - 0.05, relwidth=(Len_lbx + Len_inpx + 0.02) * 2, relheight=0.04)

    Para1_lb1 = Label(root, text='Energy (eV) :',
                      font=("times new roman", 10))
    Para1_lb1.place(relx=lbx0 + 0.01, rely=lby0 + 0.02, relwidth=Len_lbx, relheight=Len_lby)
    Para1_inp1 = Entry(root)
    Para1_inp1.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 1 - Len_inpx, rely=lby0 + 0.02, relwidth=Len_inpx,
                     relheight=Len_inpy)
    Para1_inp1.insert(0, '710')

    Para1_lb2 = Label(root, text='CCD pixel size (um) :',
                      font=("times new roman", 10))
    Para1_lb2.place(relx=lbx0 + 0.01 + (Len_lbx + 0.02 + Len_inpx) * (2 - 1), rely=lby0 + 0.02, relwidth=Len_lbx,
                    relheight=Len_lby)
    Para1_inp2 = Entry(root)
    Para1_inp2.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 2 - Len_inpx, rely=lby0 + 0.02, relwidth=Len_inpx,
                     relheight=Len_inpy)
    Para1_inp2.insert(0, '27')

    Para1_lb3 = Label(root, text='L_sd (mm) :',
                      font=("times new roman", 10))
    Para1_lb3.place(relx=lbx0 + 0.01, rely=lby0 + Len_lby * (2 - 1) + 0.02, relwidth=Len_lbx, relheight=Len_lby)
    Para1_inp3 = Entry(root)
    Para1_inp3.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 1 - Len_inpx,
                     rely=lby0 + Len_lby * (2 - 1) + 0.02, relwidth=Len_inpx, relheight=Len_inpy)
    Para1_inp3.insert(0, '72')

    Para1_lb4 = Label(root, text='Probe size (um) :',
                      font=("times new roman", 10))
    Para1_lb4.place(relx=lbx0 + 0.01 + (Len_lbx + 0.02 + Len_inpx) * (2 - 1), rely=lby0 + Len_lby * (2 - 1) + 0.02,
                    relwidth=Len_lbx, relheight=Len_lby)
    Para1_inp4 = Entry(root)
    Para1_inp4.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 2 - Len_inpx,
                     rely=lby0 + Len_lby * (2 - 1) + 0.02, relwidth=Len_inpx, relheight=Len_inpy)
    Para1_inp4.insert(0, '2')

    Para1_lb5 = Label(root, text='Zone Plate :',
                      font=("times new roman", 12, "bold")
                      )
    Para1_lb5.place(relx=lbx0 + 0.01, rely=lby0 + Len_lby * (3 - 1) + 0.02, relwidth=Len_lbx - 0.02, relheight=Len_lby)

    Para1_lb7 = Label(root, text='Diameter (um) :',
                      font=("times new roman", 10))
    Para1_lb7.place(relx=lbx0 + 0.01, rely=lby0 + Len_lby * (4 - 1) + 0.02, relwidth=Len_lbx, relheight=Len_lby)
    Para1_inp7 = Entry(root)
    Para1_inp7.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 1 - Len_inpx,
                     rely=lby0 + Len_lby * (4 - 1) + 0.02, relwidth=Len_inpx, relheight=Len_inpy)
    Para1_inp7.insert(0, '200')

    Para1_lb8 = Label(root, text='Outermost zone (nm) :',
                       font=("times new roman", 10))
    Para1_lb8.place(relx=lbx0 + 0.01 + (Len_lbx + 0.02 + Len_inpx) * (2 - 1), rely=lby0 + Len_lby * (4 - 1) + 0.02,
                     relwidth=Len_lbx, relheight=Len_lby)
    Para1_inp8 = Entry(root)
    Para1_inp8.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 2 - Len_inpx,
                      rely=lby0 + Len_lby * (4 - 1) + 0.02, relwidth=Len_inpx, relheight=Len_inpy)
    Para1_inp8.insert(0, '30')

    Para1_lb9 = Label(root, text='Beamstop (um) :',
                       font=("times new roman", 10))
    Para1_lb9.place(relx=lbx0 + 0.01, rely=lby0 + Len_lby * (5 - 1) + 0.02, relwidth=Len_lbx, relheight=Len_lby)
    Para1_inp9 = Entry(root)
    Para1_inp9.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 1 - Len_inpx,
                      rely=lby0 + Len_lby * (5 - 1) + 0.02, relwidth=Len_inpx, relheight=Len_inpy)
    Para1_inp9.insert(0, '80')

    Para1_lb11 = Label(root, text='Scanning :',
                       font=("times new roman", 12, "bold")
                       )
    Para1_lb11.place(relx=lbx0 + 0.01, rely=lby0 + Len_lby * (6 - 1) + 0.02, relwidth=Len_lbx - 0.02, relheight=Len_lby)

    rd_var = IntVar()
    Para1_rd1 = Radiobutton(root, text="Parameter", font=("times new roman", 12), variable=rd_var, value=0, command=Mysel)
    Para1_rd1.place(relx=lbx0 + 0.01+0.15, rely=lby0 + Len_lby * (6 - 1) + 0.02, relwidth=Len_lbx - 0.02, relheight=Len_lby)

    Para1_rd2 = Radiobutton(root, text="File",  font=("times new roman", 12), variable=rd_var, value=1, command=Mysel)
    Para1_rd2.place(relx=lbx0 + 0.01+0.25, rely=lby0 + Len_lby * (6 - 1) + 0.02, relwidth=Len_lbx - 0.02, relheight=Len_lby)

    Para1_lb13 = Label(root, text='Scanning mode :',
                       font=("times new roman", 10))
    Para1_lb13.place(relx=lbx0 + 0.01, rely=lby0 + Len_lby * (7 - 1) + 0.02, relwidth=Len_lbx, relheight=Len_lby)
    Para1_inp13 = Entry(root)
    Para1_inp13.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 1 - Len_inpx,
                      rely=lby0 + Len_lby * (7 - 1) + 0.02, relwidth=Len_inpx, relheight=Len_inpy)
    Para1_inp13.insert(0, 'z')

    Para1_lb14 = Label(root, text='Step length (um) :',
                       font=("times new roman", 10))
    Para1_lb14.place(relx=lbx0 + 0.01 + (Len_lbx + 0.02 + Len_inpx) * (2 - 1), rely=lby0 + Len_lby * (7 - 1) + 0.02,
                     relwidth=Len_lbx, relheight=Len_lby)
    Para1_inp14 = Entry(root)
    Para1_inp14.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 2 - Len_inpx,
                      rely=lby0 + Len_lby * (7 - 1) + 0.02, relwidth=Len_inpx, relheight=Len_inpy)
    Para1_inp14.insert(0, '0.25')

    Para1_lb15 = Label(root, text='X scanning number :',
                       font=("times new roman", 10))
    Para1_lb15.place(relx=lbx0 + 0.01, rely=lby0 + Len_lby * (8 - 1) + 0.02, relwidth=Len_lbx, relheight=Len_lby)
    Para1_inp15 = Entry(root)
    Para1_inp15.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 1 - Len_inpx,
                      rely=lby0 + Len_lby * (8 - 1) + 0.02, relwidth=Len_inpx, relheight=Len_inpy)
    Para1_inp15.insert(0, '12')

    Para1_lb16 = Label(root, text='Y scanning number :',
                       font=("times new roman", 10))
    Para1_lb16.place(relx=lbx0 + 0.01 + (Len_lbx + 0.02 + Len_inpx) * (2 - 1), rely=lby0 + Len_lby * (8 - 1) + 0.02,
                     relwidth=Len_lbx, relheight=Len_lby)
    Para1_inp16 = Entry(root)
    Para1_inp16.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 2 - Len_inpx,
                      rely=lby0 + Len_lby * (8 - 1) + 0.02, relwidth=Len_inpx, relheight=Len_inpy)
    Para1_inp16.insert(0, '12')

    Para1_lb17 = Label(root, text='Load File :',
                       font=("Times", 10, "bold"))
    Para1_lb17.place(relx=0.53, rely=lby0 + Len_lby * (9 - 1) + 0.02, relwidth=0.06, relheight=0.04)
    Para1_inp17_path = StringVar()
    Para1_inp17 = Entry(root, textvariable=Para1_inp17_path, font=("Times", 10))
    Para1_inp17.place(relx=0.59, rely=lby0 + Len_lby * (9 - 1) + 0.02, relwidth=0.23, relheight=0.04)
    Para1_btn17 = Button(root, text='Load', bg='gray', command=select_path_position)
    Para1_btn17.place(relx=0.83, rely=lby0 + Len_lby * (9 - 1) + 0.02, relwidth=0.04, relheight=0.04)

    # Para2 display
    lbx0 = 0.5
    lby0 = 0.6
    Len_lbx = 0.12
    Len_lby = 0.05
    Len_inpx = 0.05
    Len_inpy = Len_lby - 0.01

    Para2 = Label(root, text='',
                  # bg='gray', \
                  fg='black',
                  font=('参数', 10),
                  relief=SUNKEN)
    Para2.pack()
    Para2.place(relx=lbx0, rely=lby0, relwidth=(Len_lbx + Len_inpx + 0.02) * 2, relheight=(Len_lby + 0.01) * 3)

    Para2_lb = Label(root, text='Iterative parameters',
                     font=("Times", 15))
    Para2_lb.place(relx=lbx0, rely=lby0 - 0.05, relwidth=(Len_lbx + Len_inpx + 0.02) * 2, relheight=0.04)

    Para2_lb1 = Label(root, text='Alpha :',
                      font=("Times", 10))
    Para2_lb1.place(relx=lbx0 + 0.01, rely=lby0 + 0.02, relwidth=Len_lbx, relheight=Len_lby)
    Para2_inp1 = Entry(root)
    Para2_inp1.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 1 - Len_inpx, rely=lby0 + 0.02, relwidth=Len_inpx,
                     relheight=Len_inpy)
    Para2_inp1.insert(0, '0.5')

    Para2_lb2 = Label(root, text='Beta :',
                      font=("Times", 10))
    Para2_lb2.place(relx=lbx0 + 0.01 + (Len_lbx + 0.02 + Len_inpx) * (2 - 1), rely=lby0 + 0.02, relwidth=Len_lbx,
                    relheight=Len_lby)
    Para2_inp2 = Entry(root)
    Para2_inp2.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 2 - Len_inpx, rely=lby0 + 0.02, relwidth=Len_inpx,
                     relheight=Len_inpy)
    Para2_inp2.insert(0, '1')

    Para2_lb3 = Label(root, text='N_iteration :',
                      font=("Times", 10))
    Para2_lb3.place(relx=lbx0 + 0.01, rely=lby0 + Len_lby * (2 - 1) + 0.02, relwidth=Len_lbx, relheight=Len_lby)
    Para2_inp3 = Entry(root)
    Para2_inp3.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 1 - Len_inpx,
                     rely=lby0 + Len_lby * (2 - 1) + 0.02,
                     relwidth=Len_inpx, relheight=Len_inpy)
    Para2_inp3.insert(0, '200')

    Para2_lb4 = Label(root, text='N_show :',
                      font=("Times", 10))
    Para2_lb4.place(relx=lbx0 + 0.01 + (Len_lbx + 0.02 + Len_inpx) * (2 - 1), rely=lby0 + Len_lby * (2 - 1) + 0.02,
                    relwidth=Len_lbx, relheight=Len_lby)
    Para2_inp4 = Entry(root)
    Para2_inp4.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 2 - Len_inpx,
                     rely=lby0 + Len_lby * (2 - 1) + 0.02,
                     relwidth=Len_inpx, relheight=Len_inpy)
    Para2_inp4.insert(0, '5')

    Para2_lb5 = Label(root, text='N_save :',
                      font=("Times", 10))
    Para2_lb5.place(relx=lbx0 + 0.01, rely=lby0 + Len_lby * (3 - 1) + 0.02, relwidth=Len_lbx, relheight=Len_lby)
    Para2_inp5 = Entry(root)
    Para2_inp5.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 1 - Len_inpx,
                     rely=lby0 + Len_lby * (3 - 1) + 0.02,
                     relwidth=Len_inpx, relheight=Len_inpy)
    Para2_inp5.insert(0, '10')

    Para2_lb6 = Label(root, text='N_shrink :',
                      font=("Times", 10))
    Para2_lb6.place(relx=lbx0 + 0.01 + (Len_lbx + 0.02 + Len_inpx) * (2 - 1), rely=lby0 + Len_lby * (3 - 1) + 0.02,
                    relwidth=Len_lbx, relheight=Len_lby)
    Para2_inp6 = Entry(root)
    Para2_inp6.place(relx=lbx0 + 0.01 + (Len_lbx + 0.01 + Len_inpx) * 2 - Len_inpx,
                     rely=lby0 + Len_lby * (3 - 1) + 0.02,
                     relwidth=Len_inpx, relheight=Len_inpy)
    Para2_inp6.insert(0, '5')

    Para3_lb1 = Label(root, text='Load File :',
                      font=("Times", 10, "bold"))
    Para3_lb1.place(relx=0.51, rely=0.79, relwidth=0.06, relheight=0.04)
    Para3_inp1_path = StringVar()
    Para3_inp1 = Entry(root, textvariable=Para3_inp1_path, font=("Times", 10))
    Para3_inp1.place(relx=0.57, rely=0.79, relwidth=0.25, relheight=0.04)
    Para3_inp1.insert(0, './709eV_280ms_2ump_250nstep_rand12x12.TIF')    
    Para3_btn1 = Button(root, text='Load', bg='gray', command=select_path)
    Para3_btn1.place(relx=0.83, rely=0.79, relwidth=0.04, relheight=0.04)

    Para3_lb2 = Label(root, text='Save File :',
                      font=("Times", 10, "bold"))
    Para3_lb2.place(relx=0.51, rely=0.84, relwidth=0.06, relheight=0.04)
    Para3_inp2_path = StringVar()
    Para3_inp2 = Entry(root, textvariable=Para3_inp2_path, font=("Times", 10))
    Para3_inp2.place(relx=0.57, rely=0.84, relwidth=0.25, relheight=0.04)
    Para3_btn2 = Button(root, text='Save', bg='Silver', command=save_path)
    Para3_btn2.place(relx=0.83, rely=0.84, relwidth=0.04, relheight=0.04)

    img_iasf = PhotoImage(file=r'.\iasf.png')
    lb_img_iasf = Label(root, image=img_iasf)
    lb_img_iasf.place(relx=0.75, rely=0.89, relwidth=0.2, relheight=0.07)
    lb_info = Label(root, text='pyPtycho 1.0.1 by Xing Zhenjiang',
                      font=("Times", 8))
    lb_info.place(relx=0.76, rely=0.96, relwidth=0.2, relheight=0.03)

    Para4_btn1 = Button(root, text='Run', font=("Times", 15, "bold"), bg='green', command=run1)
    Para4_btn1.place(relx=0.38, rely=0.5, relwidth=0.06, relheight=0.06)

    txt = Text(root, font=('Times', 10), wrap=None)
    scrollbar_v = Scrollbar(txt)
    scrollbar_v.pack(side=RIGHT, fill=Y)
    scrollbar_v.config(command=txt.yview)
    txt.config(yscrollcommand=scrollbar_v.set)
    txt.place(relx=0.33, rely=0.7, relwidth=0.15, relheight=0.25)
    # print(int(Para1_inp1.get()))

    root.mainloop()
