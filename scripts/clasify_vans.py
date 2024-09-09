import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import os
from colorama import just_fix_windows_console
import cv2
from networkx import center

NEW_IMG_H = 550

class Display(tk.Tk):
    def __init__(self, image_folder, label_folder, new_label_folder, bb_image_folder, dataset_info_folder):
        super().__init__()
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.new_label_folder = new_label_folder
        self.bb_image_folder = bb_image_folder
        self.dataset_info_folder = dataset_info_folder

        self.geometry("1200x800")
        
        self.id = 0

        self.file_names = [f[:-4] for f in os.listdir(label_folder) if f.endswith(('.txt',))]
        self.n_pics = len(self.file_names)
        self.imgs_info = [None]*self.n_pics

        self.frame = ttk.Frame(master=self)
        self.inputs_frame = ttk.Frame(self.frame)
        self.idpp_entry_var = tk.StringVar(value=str(self.id+1))
        self.idpp_entry = ttk.Entry(master=self.frame, width=5, textvariable=self.idpp_entry_var)

        self.update_disp()

        self.frame.pack(expand=True)
        self.bind('<Left>', func=self.left)
        self.bind('<Right>', func=self.right)
        self.bind('<Button-3>', func=self.right)
        self.bind('<Button-2>', func=self.left)
        #self.bind('<space>', func=self.right)
        self.bind('.', func=self.toggle_pesada_ligera)
        self.bind('l', func=self.focus_next_van)
        self.bind('k', func=self.focus_previous_van)

    def toggle_pesada_ligera(self, *ignore):
        if self.focus_get() not in self.info["chk"]:
            self.focus_next_van(None)

        index = self.info["chk"].index(self.focus_get())

        if str(self.info["chk"][index]['state']) in ['normal', 'selected']:
            self.info["chk"][index].invoke()
        

    def focus_next_van(self, *ignore):
        if self.focus_get() not in self.info["chk"]:
            index = 0
            while str(self.info["chk"][index]['state']) == 'disabled':
                index += 1
                if index == len(self.info["chk"]):
                    return
            self.info["chk"][index].focus_set()
        else:
            index = self.info["chk"].index(self.focus_get()) + 1
        if index == len(self.info["chk"]):
            index = 0
        first_index = index
        while str(self.info["chk"][index]['state']) == 'disabled':
            index += 1
            if index == len(self.info["chk"]):
                index = 0
            if index == first_index:
                return
        if index <= len(self.info["chk"]) - 1:
            self.info["chk"][index].focus_set()
        else:
            index = 0
            while str(self.info["chk"][index]['state']) == 'disabled':
                index += 1
                if index == len(self.info["chk"]):
                    return
            self.info["chk"][index].focus_set()

    def focus_previous_van(self, *ignore):
        if self.focus_get() not in self.info["chk"]:
            index = len(self.info["chk"]) - 1
            while str(self.info["chk"][index]['state']) == 'disabled':
                index -= 1
                if index < 0:
                    return
            self.info["chk"][index].focus_set()
        else:
            index = self.info["chk"].index(self.focus_get()) - 1
        first_index = index
        while str(self.info["chk"][index]['state']) == 'disabled':
            index -= 1
            if index < 0:
                index = len(self.info["chk"]) - 1
            if index == first_index:
                return
        if index >= 0:
            self.info["chk"][index].focus_set()
        else:
            print("index1", index)
            index = len(self.info["chk"]) + index
            print("index2", index)
            while str(self.info["chk"][index]['state']) == 'disabled':
                index -= 1
                if index < 0:
                    return
            self.info["chk"][index].focus_set()

    def read_yolo_labels(self, label_path):
        with open(label_path, 'r') as file:
            lines = file.readlines()
        classes = []
        for line in lines:
            values = line.strip().split()
            classes.append(int(values[0]))
        return classes

    def read_db_stats(self, path):
        with open(path, 'r') as file:
            lines = file.readlines()
        colors = lines[0].strip().split()
        fronts = lines[1].strip().split()
        radios = lines[2].strip().split()
        return (colors, fronts, radios)

    def get_image_info(self, image_path: str):
        height, width, _ = cv2.imread(image_path).shape
        return height, width

    def save_classes(self):
        if self.info is None:
            return
        for index, var in enumerate(self.info['chk_vars']):
            if str(self.info['chk'][index]['state']) != 'disabled':
                self.info['classes'][index] = var.get()+2

        original_txt = os.path.join(self.label_folder, self.info['label_file'])
        new_txt = os.path.join(self.new_label_folder, self.info['label_file'])
        stats_txt = os.path.join(self.dataset_info_folder, self.info['label_file'])

        with open(original_txt, 'r') as file:
            lines = file.readlines()
        with open(new_txt, 'w+') as file:
            for n_line, line in enumerate(lines):
                if len(self.info['classes']) > n_line:
                    clase = str(self.info['classes'][n_line])
                    line = clase + line[1:]
                    file.write(line)

        # Save stats
        with open(stats_txt, 'w+') as f:
            lines = []
            f.write(" ".join(color.get() for color in self.info['clr_vars']))  # type: ignore
            f.write("\n"+" ".join(str(front.get()) for front in self.info['front_vars']))  # type: ignore
            f.write("\n"+" ".join(str(radio.get()) for radio in self.info['radio_vars'])+"\n")  # type: ignore

        self.imgs_info[self.id] = self.info

    def right(self, *ignore):
        self.save_classes()
        self.id += 1
        if self.id >= self.n_pics:
            self.id = 0
        self.update_disp()

    def left(self, *ignore):
        self.save_classes()
        self.id -= 1
        if self.id < 0:
            self.id = self.n_pics - 1
        self.update_disp()

    def new_n_image(self, *ignore):
        self.save_classes()
        self.id = int(self.idpp_entry.get()) - 1
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
        image_file = file + '.jpg'
        image_path = os.path.join(self.image_folder, image_file)
        label_file = file + '.txt'
        img_h, img_w = self.get_image_info(image_path)
        new_img_w = int(img_w*NEW_IMG_H/img_h)
        img = ImageTk.PhotoImage(Image.open(image_path).resize((new_img_w, NEW_IMG_H)))
        if os.path.isfile(os.path.join(self.new_label_folder, label_file)):
            classes = self.read_yolo_labels(os.path.join(self.new_label_folder, label_file))
        else:
            classes = self.read_yolo_labels(os.path.join(self.label_folder, label_file))

        if os.path.isfile(os.path.join(self.dataset_info_folder, label_file)):
            colors, fronts, distances = self.read_db_stats(os.path.join(self.dataset_info_folder, label_file))
        else:
            colors = ['wh']*len(classes)
            fronts = [-1]*len(classes)
            distances = [-1]*len(classes)

        chkbuttons_vars = []
        chkbuttons = []
        clr_entries_vars = []
        clr_entries = []
        front_chks_vars = []
        front_chks = []
        radios = []
        radios_vars = []
        for index, (clase, color, front_saved, distance) in enumerate(zip(classes, colors, fronts, distances)):
            if clase in [2, 3]:
                value = clase - 2
                state = tk.NORMAL
            else:
                value = 0
                state = tk.DISABLED
            clr_var = tk.StringVar(value=color)
            front_var = tk.IntVar(value=int(front_saved))
            radio_var = tk.IntVar(value=int(distance))
            var = tk.IntVar(value=value)

            chkbutton = ttk.Checkbutton(
                master=self.inputs_frame,
                text=index,
                variable=var,
                state=state)
            chkbuttons_vars.append(var)
            chkbuttons.append(chkbutton)

            clr_entry = ttk.Entry(
                master=self.inputs_frame,
                textvariable=clr_var,
                width=10,
                state=state)
            clr_entries_vars.append(clr_var)
            clr_entries.append(clr_entry)

            front_chk = ttk.Checkbutton(
                master=self.inputs_frame,
                variable=front_var,
                state=state)
            front_chks_vars.append(front_var)
            front_chks.append(front_chk)

            radio_c = ttk.Radiobutton(
                master=self.inputs_frame,
                variable=radio_var,
                state=state,
                value=0)

            radio_m = ttk.Radiobutton(
                master=self.inputs_frame,
                variable=radio_var,
                state=state,
                value=1)

            radio_l = ttk.Radiobutton(
                master=self.inputs_frame,
                variable=radio_var,
                state=state,
                value=2)
            radio_ml = ttk.Radiobutton(
                master=self.inputs_frame,
                variable=radio_var,
                state=state,
                value=3)
            radios_vars.append(radio_var)
            radios.append((radio_c, radio_m, radio_l, radio_ml))

        self.imgs_info[self.id] = {  # type: ignore
            'img_file': image_file,
            'img_path': image_path,
            'label_file': label_file,
            'img': img,
            'classes': classes,
            'chk_vars': chkbuttons_vars,
            'chk': chkbuttons,
            'clr_vars': clr_entries_vars,
            'clr': clr_entries,
            'front_vars': front_chks_vars,
            'front': front_chks,
            'radio_vars': radios_vars,
            'radios': radios,
        }

        self.info = self.imgs_info[self.id]

        ttk.Label(master=self.frame, text='Imagen número: ').grid(row=0, column=0, sticky='e', pady=10)

        self.idpp_entry_var.set(str(self.id + 1))
        self.idpp_entry.grid(row=0, column=1, pady=10, sticky='w')
        self.idpp_entry.bind("<Return>", self.new_n_image)

        img_name = tk.Text(master=self.frame, height=1, width=50, borderwidth=0, background=self.cget('background'))
        img_name.tag_configure("center_text", justify='center')
        img_name.insert(tk.END, self.info['label_file'] if self.info else "")
        img_name.config(state=tk.DISABLED)
        img_name.tag_add("center_text", "1.0", tk.END)
        img_name.grid(row=1, column=0, columnspan=2, sticky='n')

        label = ttk.Label(self.frame, image=self.info['img'] if self.info else "")
        label.image = self.info['img'] if self.info else ""  # type: ignore
        label.grid(sticky='n', column=0, row=2)

        unlabelled_img = ImageTk.PhotoImage(Image.open(os.path.join(self.bb_image_folder, image_file)).resize((new_img_w, NEW_IMG_H)))
        label = ttk.Label(self.frame, image=unlabelled_img)  # type: ignore
        label.image = unlabelled_img   # type: ignore
        label.grid(sticky='n', column=1, row=2)

        for w in self.inputs_frame.winfo_children():
            w.grid_forget()
            w.pack_forget()

        ttk.Label(master=self.inputs_frame, text='Es pesada:').grid(row=0, column=0, sticky='n', padx=0)
        col = 1
        for chk in self.info['chk']:  # type: ignore
            chk.grid(row=0, column=col, sticky='n', padx=10)
            col += 1

        ttk.Label(master=self.inputs_frame, text='Color:').grid(row=1, column=0, sticky='n', padx=0)
        col = 1
        for clr in self.info['clr']:  # type: ignore
            clr.grid(row=1, column=col, sticky='n', padx=10)
            col += 1

        ttk.Label(master=self.inputs_frame, text='de frente?').grid(row=2, column=0, sticky='n', padx=0)
        col = 1
        for front in self.info['front']:  # type: ignore
            front.grid(row=2, column=col, sticky='n', padx=5)
            col += 1
        self.inputs_frame.grid(sticky='n', pady=5, columnspan=2)

        ttk.Label(master=self.inputs_frame, text='Cerca').grid(row=3, column=0, sticky='n', padx=0)
        ttk.Label(master=self.inputs_frame, text='Media').grid(row=4, column=0, sticky='n', padx=0)
        ttk.Label(master=self.inputs_frame, text='Lejos').grid(row=5, column=0, sticky='n', padx=0)
        ttk.Label(master=self.inputs_frame, text='Muy lejos').grid(row=6, column=0, sticky='n', padx=0)
        col = 1
        for radios_info in self.info['radios']:  # type: ignore
            r_c, r_m, r_l, r_ml = radios_info
            r_c.grid(row=3, column=col, sticky='n', padx=5)
            r_m.grid(row=4, column=col, sticky='n', padx=5)
            r_l.grid(row=5, column=col, sticky='n', padx=5)
            r_ml.grid(row=6, column=col, sticky='n', padx=5)
            col += 1
        self.inputs_frame.grid(sticky='n', pady=5, columnspan=2)

        buttons_frame = ttk.Frame(master=self.frame)
        ttk.Button(master=buttons_frame, text='<', command=self.left).grid(row=0, column=0, sticky='s')
        ttk.Button(master=buttons_frame, text='>', command=self.right).grid(row=0, column=1, sticky='s')
        buttons_frame.grid(sticky='s', columnspan=2)


if __name__ == "__main__":
    general_dir = r'C:\Users\sierr\Documents\Uni\TFM\archive\for_relabelling'
    bb_image_folder = general_dir + r'\unlabeled_images'
    image_folder = general_dir + r'\images'
    label_folder = general_dir + r'\labels'
    new_label_folder = general_dir + r'\new_labels'
    dataset_info_folder = general_dir + r'\dataset_info'
    root = Display(image_folder, label_folder, new_label_folder, bb_image_folder, dataset_info_folder)

    root.mainloop()
