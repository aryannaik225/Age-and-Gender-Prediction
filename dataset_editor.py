import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import csv
import glob
import os
import chromadb

# --- CONFIG ---
CSV_FILE = "footfall_analytics_store.csv"
HARVEST_DIR = "harvested_images_store"
DB_DIR = "./chroma_reid_db"

class DataEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Footfall Data Studio (SDE Edition)")
        self.root.geometry("1100x800") 
        self.root.configure(bg="#1e1e1e")
        
        self.records = [] 
        self.unique_ids = []
        
        self.current_idx = 0
        
        self.anchor_idx = 0
        self.compare_idx = 1
        
        self.connect_to_db()
        self.load_csv()
        self.setup_ui()
        self.refresh_ui()

    def connect_to_db(self):
        try:
            self.chroma_client = chromadb.PersistentClient(path=DB_DIR)
            self.collection = self.chroma_client.get_collection(name="store_visitors")
            print("Connected to ChromaDB! Edits will be synchronized globally.")
        except Exception as e:
            self.collection = None
            print("Warning: ChromaDB not found. Is the path correct? Will only edit CSV.")

    def load_csv(self):
        if not os.path.exists(CSV_FILE):
            messagebox.showerror("Error", f"Could not find {CSV_FILE}")
            self.root.destroy()
            return
            
        with open(CSV_FILE, 'r') as f:
            reader = csv.reader(f)
            self.headers = next(reader)
            self.records = list(reader)
            
        seen = set()
        self.unique_ids = []
        for row in self.records:
            if len(row) >= 4 and row[1] not in seen:
                self.unique_ids.append(row[1])
                seen.add(row[1])

    def save_csv_to_disk(self):
        self.records.sort(key=lambda x: x[0])

        cleaned_records = []
        last_state = {}

        for row in self.records:
            if len(row) < 5: continue
            timestamp, gid, gender, age, event = row[0], row[1], row[2], row[3], row[4]

            prev_event = last_state.get(gid)

            if prev_event == event:
                continue
            if prev_event in ["ENTERED ZONE", "ALREADY IN STORE"] and event == "ALREADY IN STORE":
                continue
            if prev_event == "EXITED ZONE" and event == "EXITED ZONE":
                continue

            cleaned_records.append(row)
            last_state[gid] = event

        self.records = cleaned_records

        try:
            with open(CSV_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                writer.writerows(self.records)
        except PermissionError:
            messagebox.showerror("Error", f"Close {CSV_FILE} in Excel first! It is locking the file.")

    def get_person_image(self, target_id, target_height=450):
        search_pattern = os.path.join(HARVEST_DIR, "**", f"id_{target_id}_body.jpg")
        matches = glob.glob(search_pattern, recursive=True)
        
        if not matches: 
            search_pattern = os.path.join(HARVEST_DIR, "**", f"id_{target_id}_face.jpg")
            matches = glob.glob(search_pattern, recursive=True)

        if matches:
            try:
                img = Image.open(matches[0])
                w, h = img.size
                aspect_ratio = w / h
                new_width = int(target_height * aspect_ratio)
                img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
                return ImageTk.PhotoImage(img), new_width
            except:
                return None, 0
        return None, 0

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook.Tab', font=('Arial', 12, 'bold'), padding=[15, 5])
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        self.tab1 = tk.Frame(self.notebook, bg="#2b2b2b")
        self.tab2 = tk.Frame(self.notebook, bg="#2b2b2b")
        
        self.notebook.add(self.tab1, text="1. Fix Demographics")
        self.notebook.add(self.tab2, text="2. Merge Duplicate IDs")
        
        self.setup_tab1()
        self.setup_tab2()

    def setup_tab1(self):
        self.img_label_t1 = tk.Label(self.tab1, text="Loading...", bg="#1e1e1e", fg="white")
        self.img_label_t1.pack(pady=20)
        
        self.info_label_t1 = tk.Label(self.tab1, text="", font=("Arial", 16, "bold"), bg="#2b2b2b", fg="white")
        self.info_label_t1.pack()

        control_frame = tk.Frame(self.tab1, bg="#2b2b2b")
        control_frame.pack(pady=15)

        tk.Label(control_frame, text="Gender:", font=("Arial", 12), bg="#2b2b2b", fg="white").grid(row=0, column=0, padx=5)
        self.gender_var = tk.StringVar()
        self.gender_cb = ttk.Combobox(control_frame, textvariable=self.gender_var, values=["Male", "Female", "Unknown", ""], font=("Arial", 12))
        self.gender_cb.grid(row=0, column=1, padx=10)

        tk.Label(control_frame, text="Age:", font=("Arial", 12), bg="#2b2b2b", fg="white").grid(row=0, column=2, padx=5)
        self.age_var = tk.StringVar()
        self.age_cb = ttk.Combobox(control_frame, textvariable=self.age_var, values=["0-12", "13-19", "20-35", "36-55", "56+", "Unknown", ""], font=("Arial", 12))
        self.age_cb.grid(row=0, column=3, padx=10)

        btn_frame = tk.Frame(self.tab1, bg="#2b2b2b")
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="⏪ Previous", command=self.prev_person, font=("Arial", 12), width=12).grid(row=0, column=0, padx=15)
        tk.Button(btn_frame, text="💾 Save Changes", command=self.save_demographics, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", width=15).grid(row=0, column=1, padx=15)
        tk.Button(btn_frame, text="Next ⏩", command=self.next_person, font=("Arial", 12), width=12).grid(row=0, column=2, padx=15)

    def setup_tab2(self):
        self.header_t2 = tk.Label(self.tab2, text="Are these the exact same person?", font=("Arial", 20, "bold"), bg="#2b2b2b", fg="white")
        self.header_t2.pack(pady=15)
        
        self.progress_t2 = tk.Label(self.tab2, text="", font=("Arial", 12), bg="#2b2b2b", fg="gray")
        self.progress_t2.pack()

        img_frame = tk.Frame(self.tab2, bg="#2b2b2b")
        img_frame.pack(pady=20)
        
        anchor_container = tk.Frame(img_frame, bg="#2b2b2b")
        anchor_container.grid(row=0, column=0, padx=20)
        self.anchor_id_label = tk.Label(anchor_container, text="Target ID", font=("Arial", 14, "bold"), bg="#2b2b2b", fg="#4CAF50")
        self.anchor_id_label.pack()
        self.img_label_anchor = tk.Label(anchor_container, bg="#1e1e1e", width=40, height=20)
        self.img_label_anchor.pack()

        tk.Label(img_frame, text="VS", font=("Arial", 24, "bold"), bg="#2b2b2b", fg="white").grid(row=0, column=1)

        compare_container = tk.Frame(img_frame, bg="#2b2b2b")
        compare_container.grid(row=0, column=2, padx=20)
        self.compare_id_label = tk.Label(compare_container, text="Compare ID", font=("Arial", 14, "bold"), bg="#2b2b2b", fg="#FF9800")
        self.compare_id_label.pack()
        self.img_label_compare = tk.Label(compare_container, bg="#1e1e1e", width=40, height=20)
        self.img_label_compare.pack()

        btn_frame = tk.Frame(self.tab2, bg="#2b2b2b")
        btn_frame.pack(pady=30)
        
        tk.Button(btn_frame, text="✅ YES (Merge them)", command=self.merge_ids, font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", width=20, height=2).grid(row=0, column=0, padx=20)
        tk.Button(btn_frame, text="❌ NO (Keep Separate)", command=self.skip_merge, font=("Arial", 14, "bold"), bg="#f44336", fg="white", width=20, height=2).grid(row=0, column=1, padx=20)

    def refresh_ui(self):
        if self.current_idx >= len(self.unique_ids): self.current_idx = max(0, len(self.unique_ids) - 1)
        
        if self.anchor_idx >= len(self.unique_ids) - 1:
            self.header_t2.config(text="🎉 All Comparisons Complete!")
            self.img_label_anchor.config(image='', text="Done", width=40, height=20)
            self.img_label_compare.config(image='', text="Done", width=40, height=20)
            self.anchor_id_label.config(text="")
            self.compare_id_label.config(text="")
            self.progress_t2.config(text="")
        else:
            if self.compare_idx >= len(self.unique_ids):
                self.anchor_idx += 1
                self.compare_idx = self.anchor_idx + 1
            
            if self.anchor_idx < len(self.unique_ids) - 1:
                self.load_tab2()

        self.load_tab1()

    def load_tab1(self):
        if not self.unique_ids: return
        target_id = self.unique_ids[self.current_idx]
        self.info_label_t1.config(text=f"Editing Global ID: {target_id} ({self.current_idx + 1} of {len(self.unique_ids)})")
        
        current_gender, current_age = "", ""
        for row in self.records:
            if row[1] == target_id:
                current_gender = row[2]
                current_age = row[3]
                break
                
        self.gender_var.set(current_gender)
        self.age_var.set(current_age)
        
        photo, w = self.get_person_image(target_id, 450)
        if photo:
            self.img_label_t1.config(image=photo, text="", width=w, height=450)
            self.img_label_t1.image = photo 
        else:
            self.img_label_t1.config(image='', text="No Image", width=40, height=20)

    def save_demographics(self):
        target_id = self.unique_ids[self.current_idx]
        new_g = self.gender_var.get()
        new_a = self.age_var.get()
        
        for row in self.records:
            if row[1] == target_id:
                row[2] = new_g
                row[3] = new_a
        self.save_csv_to_disk()
        
        if self.collection:
            try:
                db_res = self.collection.get(where={"global_id": int(target_id)})
                if db_res and db_res['ids']:
                    new_metas = []
                    for m in db_res['metadatas']:
                        m['gender'] = new_g  
                        m['age'] = new_a    
                        new_metas.append(m)
                    
                    self.collection.update(ids=db_res['ids'], metadatas=new_metas)
                    print(f"🗄️ DB SYNC: Updated Demographics for ID {target_id} across {len(db_res['ids'])} poses.")
            except Exception as e:
                print(f"DB Update failed: {e}")

        print(f"Saved Demographics for ID {target_id}")

    def next_person(self):
        if self.current_idx < len(self.unique_ids) - 1:
            self.current_idx += 1
            self.refresh_ui()

    def prev_person(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.refresh_ui()

    def load_tab2(self):
        if not self.unique_ids or self.anchor_idx >= len(self.unique_ids) - 1: return
        
        a_id = self.unique_ids[self.anchor_idx]
        c_id = self.unique_ids[self.compare_idx]
        
        self.progress_t2.config(text=f"Anchor Progress: {self.anchor_idx + 1} / {len(self.unique_ids) - 1}")
        self.anchor_id_label.config(text=f"Target ID: {a_id}")
        self.compare_id_label.config(text=f"Is this also ID {a_id}? (Currently ID {c_id})")

        p_anchor, wa = self.get_person_image(a_id, 400)
        p_compare, wc = self.get_person_image(c_id, 400)

        if p_anchor:
            self.img_label_anchor.config(image=p_anchor, text="", width=wa, height=400)
            self.img_label_anchor.image = p_anchor
        else:
            self.img_label_anchor.config(image='', text="No Image", width=30, height=20)
            
        if p_compare:
            self.img_label_compare.config(image=p_compare, text="", width=wc, height=400)
            self.img_label_compare.image = p_compare
        else:
            self.img_label_compare.config(image='', text="No Image", width=30, height=20)

    def merge_ids(self):
        anchor_id = self.unique_ids[self.anchor_idx]
        absorbed_id = self.unique_ids.pop(self.compare_idx) 
        
        for row in self.records:
            if row[1] == absorbed_id:
                row[1] = anchor_id
        self.save_csv_to_disk()
        
        if self.collection:
            try:
                db_res = self.collection.get(where={"global_id": int(absorbed_id)})
                if db_res and db_res['ids']:
                    new_metas = []
                    for m in db_res['metadatas']:
                        m['global_id'] = int(anchor_id)
                        new_metas.append(m)
                    
                    self.collection.update(ids=db_res['ids'], metadatas=new_metas)
                    print(f"🗄️ DB SYNC: Transferred {len(db_res['ids'])} vectors from ID {absorbed_id} to ID {anchor_id}")
            except Exception as e:
                print(f"DB Merge failed: {e}")

        print(f"MERGED: ID {absorbed_id} is now permanently ID {anchor_id}")
        self.refresh_ui()

    def skip_merge(self):
        self.compare_idx += 1
        self.refresh_ui()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataEditorApp(root)
    root.mainloop()