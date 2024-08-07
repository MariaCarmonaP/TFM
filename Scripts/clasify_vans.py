import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import os
import cv2

class Display(tk.Tk):
    def __init__(self, image_folder, label_folder, new_label_folder, unlabeled_images_folder):
        super().__init__()
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.new_label_folder = new_label_folder
        self.unlabeled_images_folder = unlabeled_images_folder
        
        self.geometry("1200x800")
        self.new_img_h = 800
        self.id = 0
        
        self.file_names = [f[:-4] for f in os.listdir(label_folder) if f.endswith(('.txt',))]
        self.n_pics = len(self.file_names)
        self.imgs_info = [None]*self.n_pics
        
        self.frame = ttk.Frame(master=self)
        self.chkbuttons_frame = ttk.Frame(self.frame)
        self.idpp_entry_var = tk.StringVar(value=self.id+1)
        self.idpp_entry = ttk.Entry(master=self.frame, width=5,textvariable=self.idpp_entry_var)

        self.update_disp()
        
        self.frame.pack(expand=True)
        self.bind('<Left>', func=self.left)
        self.bind('<Right>', func=self.right)
        self.bind('<Button-3>', func=self.left)
    
    def read_yolo_labels(self, label_path):
        with open(label_path, 'r') as file:
            lines = file.readlines()
        classes = []
        for line in lines:
            values = line.strip().split()
            classes.append(int(values[0]))
        return classes
    
    def get_image_info(self, image_path:str):
        height, width, _ = cv2.imread(image_path).shape
        return height, width

    def save_classes(self):
        for index, var in enumerate(self.info['chk_vars']):
            if str(self.info['chk'][index]['state'])!='disabled':
                self.info['classes'][index]=var.get()+2
        
        original_txt = os.path.join(self.label_folder, self.info['label_file'])
        new_txt = os.path.join(self.new_label_folder, self.info['label_file'])
        
        with open(original_txt, 'r') as file:
            lines = file.readlines()
        with open(new_txt, 'w+') as file:
            for n_line, line in enumerate(lines):
                if len(self.info['classes'])>n_line:
                    clase = str(self.info['classes'][n_line])
                    line = clase + line[1:]
                    file.write(line)
        self.imgs_info[self.id] = self.info
    
    def right(self, *ignore):
        self.save_classes()
        self.id += 1
        if self.id>=self.n_pics:
            self.id = 0
        self.update_disp()
        
    def left(self, *ignore):
        self.save_classes()
        self.id -= 1
        if self.id<0:
            self.id = self.n_pics - 1
        self.update_disp()
    
    def new_n_image(self, *ignore):
        self.save_classes()
        self.id = int(self.idpp_entry.get())-1
        if self.id >= self.n_pics:
            self.id = self.n_pics - 1
        elif self.id < 0:
            self.id = 0
        self.update_disp()

    def update_disp(self):
        for w in self.frame.winfo_children():
            w.grid_forget()
            w.pack_forget()
        file = self.file_names[self.id]
        image_file = file+'.jpg'
        image_path = os.path.join(self.image_folder, image_file)
        label_file = file+'.txt'
        img_h, img_w = self.get_image_info(image_path)
        new_img_w = int(img_w*self.new_img_h/img_h)
        img = ImageTk.PhotoImage(Image.open(image_path).resize((new_img_w, self.new_img_h)))
        if os.path.isfile(os.path.join(self.new_label_folder, label_file)):
            classes = self.read_yolo_labels(os.path.join(self.new_label_folder, label_file))
        else:
            classes = self.read_yolo_labels(os.path.join(self.label_folder, label_file))
        chkbuttons_vars = []
        chkbuttons = []
        for index,clase in enumerate(classes):
            if clase in [2,3]:
                value = clase - 2
                state = tk.NORMAL
            else:
                value = 0
                state = tk.DISABLED
            var = tk.IntVar(value=value)
            chkbutton = ttk.Checkbutton(
                master=self.chkbuttons_frame,
                text=index, 
                variable=var,
                state=state)
            chkbuttons_vars.append(var)
            chkbuttons.append(chkbutton)
        
        self.imgs_info[self.id] = {
            'img_file': image_file,
            'img_path': image_path,
            'label_file': label_file,
            'img': img,
            'classes': classes,
            'chk_vars': chkbuttons_vars,
            'chk': chkbuttons
        }
        
        self.info = self.imgs_info[self.id]
        
        
        ttk.Label(master=self.frame, text='Imagen número: ').grid(row=0, column=0,sticky='e', pady=10)

        self.idpp_entry_var.set(self.id+1)
        self.idpp_entry.grid(row=0, column=1, pady=10, sticky='w')
        self.idpp_entry.bind("<Return>", self.new_n_image)
        ttk.Label(master=self.frame, text=self.info['label_file']).grid(row=1, column=0,columnspan=2,sticky='n')
        
        
        label = ttk.Label(self.frame, image=self.info['img'])
        label.image = self.info['img']
        label.grid(sticky='n', column=0, row=2)
        
        unlabelled_img = ImageTk.PhotoImage(Image.open(os.path.join(self.unlabeled_images_folder, image_file)).resize((new_img_w, self.new_img_h)))
        label = ttk.Label(self.frame, image=unlabelled_img)
        label.image = unlabelled_img
        label.grid(sticky='n', column=1, row=2)
        
        ttk.Label(master=self.frame, text='¿Es pesada?').grid(sticky='n', pady=10, columnspan=2)
        for w in self.chkbuttons_frame.winfo_children():
            w.grid_forget()
            w.pack_forget()
        col = 0
        for chk in self.info['chk']:
            chk.grid(row=0,column=col,sticky='n', padx=5)
            col += 1
        self.chkbuttons_frame.grid(sticky='n', pady=5, columnspan=2)

        buttons_frame = ttk.Frame(master=self.frame)
        ttk.Button(master=buttons_frame, text='<', command=self.left).grid(row=0,column=0, sticky='s')
        ttk.Button(master=buttons_frame, text='>', command=self.right).grid(row=0,column=1, sticky='s')
        buttons_frame.grid(sticky='s', columnspan=2)


if __name__ == "__main__":
    unlabeled_images_folder = r'C:\Users\sierr\Documents\Uni\TFM\furgonetas_27_06_2023'
    image_folder = r'C:\Users\sierr\Documents\Uni\TFM\images'
    label_folder = r'C:\Users\sierr\Documents\Uni\TFM\labels'
    new_label_folder = r'C:\Users\sierr\Documents\Uni\TFM\new_labels'

    root = Display(image_folder, label_folder, new_label_folder, unlabeled_images_folder)

    root.mainloop()
