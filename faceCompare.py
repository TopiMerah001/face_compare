import tkinter as tk
from tkinter import filedialog, Label, Frame, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras_facenet import FaceNet
import mediapipe as mp

class FaceComparerApp:
    def __init__(self, master):
        self.master = master
        master.title("FaceNet Face Comparator")
        master.geometry("800x700")
        master.configure(bg="#f0f0f0")

        self.img1_path = None
        self.img2_path = None
        self.similarity_threshold = 0.6
        
        self.facenet_model = FaceNet()
        
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1, 
            refine_landmarks=True, 
            min_detection_confidence=0.5
        )

        
        main_frame = Frame(master, bg="#f0f0f0")
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)

       
        title_label = Label(main_frame, text="FaceNet Face Comparator", 
                           font=("Helvetica", 18, "bold"), bg="#f0f0f0", fg="#333")
        title_label.pack(pady=(0, 20))

        
        images_frame = Frame(main_frame, bg="#f0f0f0")
        images_frame.pack(pady=20, fill="both", expand=True)

        
        img1_frame = Frame(images_frame, bg="white", relief="raised", bd=2)
        img1_frame.pack(side="left", padx=10, pady=10, fill="both", expand=True)

        Label(img1_frame, text="Image 1", font=("Helvetica", 12, "bold"), 
              bg="white", fg="#333").pack(pady=(10, 5))

        
        self.img1_original_preview = Label(img1_frame, bg="white", text="Original", fg="#666")
        self.img1_original_preview.pack(padx=20, pady=(0, 10))

        self.img1_landmark_preview = Label(img1_frame, bg="white", text="With Landmarks", fg="#666")
        self.img1_landmark_preview.pack(padx=20, pady=(0, 20))

        self.btn1 = tk.Button(img1_frame, text="Select Image 1", command=self.load_img1,
                             bg="#4CAF50", fg="white", font=("Helvetica", 10, "bold"),
                             padx=20, pady=5)
        self.btn1.pack(pady=(0, 20))

        
        img2_frame = Frame(images_frame, bg="white", relief="raised", bd=2)
        img2_frame.pack(side="right", padx=10, pady=10, fill="both", expand=True)

        Label(img2_frame, text="Image 2", font=("Helvetica", 12, "bold"), 
              bg="white", fg="#333").pack(pady=(10, 5))

        
        self.img2_original_preview = Label(img2_frame, bg="white", text="Original", fg="#666")
        self.img2_original_preview.pack(padx=20, pady=(0, 10))

        self.img2_landmark_preview = Label(img2_frame, bg="white", text="With Landmarks", fg="#666")
        self.img2_landmark_preview.pack(padx=20, pady=(0, 20))

        self.btn2 = tk.Button(img2_frame, text="Select Image 2", command=self.load_img2,
                             bg="#2196F3", fg="white", font=("Helvetica", 10, "bold"),
                             padx=20, pady=5)
        self.btn2.pack(pady=(0, 20))

        
        self.compare_btn = tk.Button(main_frame, text="Compare Faces", command=self.compare,
                                    bg="#FF5722", fg="white", font=("Helvetica", 14, "bold"),
                                    padx=40, pady=10)
        self.compare_btn.pack(pady=20)

        
        result_frame = Frame(main_frame, bg="white", relief="raised", bd=2)
        result_frame.pack(pady=20, padx=20, fill="both", expand=True)

        result_title = Label(result_frame, text="Comparison Result", 
                           font=("Helvetica", 16, "bold"), bg="white", fg="#333")
        result_title.pack(pady=(20, 10))

        self.result_label = Label(result_frame, text="Select two images and click Compare", 
                                 font=("Helvetica", 12), bg="white", fg="#666",
                                 justify="center", wraplength=300)
        self.result_label.pack(pady=(0, 20), padx=20, fill="both", expand=True)

        
        self.status_label = Label(main_frame, text="Ready - FaceNet loaded", 
                                 font=("Helvetica", 10), bg="#f0f0f0", fg="#666")
        self.status_label.pack(side="bottom", pady=10)

    def preprocess_image(self, img_path):
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb
        except Exception as e:
            print(f"Error in image preprocessing: {e}")
            return None

    def get_embedding(self, img_path):
        
        try:
            img = self.preprocess_image(img_path)
            if img is None:
                return None
                
            faces = self.facenet_model.extract(img, threshold=0.7)
            if faces:
                return faces[0]["embedding"]
            return None
        except Exception as e:
            print(f"Error in embedding extraction: {e}")
            return None

    def compare_faces(self, img1_path, img2_path):
        
        self.status_label.config(text="Processing faces...")
        self.master.update()
        
        emb1 = self.get_embedding(img1_path)
        emb2 = self.get_embedding(img2_path)
        
        if emb1 is None or emb2 is None:
            self.status_label.config(text="Error: Could not extract face embeddings")
            return None
        
       
        dot_product = np.dot(emb1, emb2)
        norm_a = np.linalg.norm(emb1)
        norm_b = np.linalg.norm(emb2)
        cosine_similarity = dot_product / (norm_a * norm_b)
  
        euclidean_distance = np.linalg.norm(emb1 - emb2)
        
        self.status_label.config(text="Comparison complete - FaceNet used")
        return {
            'cosine_similarity': cosine_similarity,
            'euclidean_distance': euclidean_distance
        }
    
    def draw_face_landmarks(self, img_bgr):
        """Draw face landmarks using MediaPipe"""
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=img_rgb,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error drawing face mesh: {e}")
            return img_bgr

    def load_img1(self):
        """Load and display first image"""
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.img1_path = path

            img_cv = cv2.imread(path)
            img_pil_original = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            img_pil_original.thumbnail((200, 150), Image.Resampling.LANCZOS)
            self.tk_img1_original = ImageTk.PhotoImage(img_pil_original)
            self.img1_original_preview.config(image=self.tk_img1_original, text="")

            img_with_landmarks = self.draw_face_landmarks(img_cv)
            img_pil_landmarked = Image.fromarray(cv2.cvtColor(img_with_landmarks, cv2.COLOR_BGR2RGB))
            img_pil_landmarked.thumbnail((200, 150), Image.Resampling.LANCZOS)
            self.tk_img1_landmark = ImageTk.PhotoImage(img_pil_landmarked)
            self.img1_landmark_preview.config(image=self.tk_img1_landmark, text="")

            self.status_label.config(text="Image 1 loaded successfully")

    def load_img2(self):
       
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.img2_path = path

            # Load original image
            img_cv = cv2.imread(path)
            img_pil_original = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            img_pil_original.thumbnail((200, 150), Image.Resampling.LANCZOS)
            self.tk_img2_original = ImageTk.PhotoImage(img_pil_original)
            self.img2_original_preview.config(image=self.tk_img2_original, text="")

            # Create landmark version
            img_with_landmarks = self.draw_face_landmarks(img_cv)
            img_pil_landmarked = Image.fromarray(cv2.cvtColor(img_with_landmarks, cv2.COLOR_BGR2RGB))
            img_pil_landmarked.thumbnail((200, 150), Image.Resampling.LANCZOS)
            self.tk_img2_landmark = ImageTk.PhotoImage(img_pil_landmarked)
            self.img2_landmark_preview.config(image=self.tk_img2_landmark, text="")

            self.status_label.config(text="Image 2 loaded successfully")

    def compare(self):
  
        if not self.img1_path or not self.img2_path:
            messagebox.showwarning("Warning", "Please select both images first!")
            return

        try:
            result = self.compare_faces(self.img1_path, self.img2_path)
            if result is None:
                self.result_label.config(
                    text="❌ Could not detect face in one or both images.\n\nPlease try with clearer face images.", 
                    fg="red"
                )
            else:
                cosine_similarity = result['cosine_similarity']
                euclidean_distance = result['euclidean_distance']

                is_match = cosine_similarity > self.similarity_threshold
                
                if cosine_similarity > 0.80:
                    confidence = "Very High"
                elif cosine_similarity > 0.70:
                    confidence = "High"
                elif cosine_similarity > 0.60:
                    confidence = "Good"
                elif cosine_similarity > 0.50:
                    confidence = "Neutral"
                elif cosine_similarity > 0.40:
                    confidence = "Low"
                else:
                    confidence = "Very Low"
                
       
                is_match = cosine_similarity > 0.60
                match_text = "✅ MATCH" if is_match else "❌ DIFFERENT"
                match_color = "#4CAF50" if is_match else "#F44336"
                
                result_text = f"Cosine Similarity: {cosine_similarity:.4f}\n"
                result_text += f"Euclidean Distance: {euclidean_distance:.4f}\n"
                result_text += f"Confidence: {confidence}\n"

                result_text += f"Result: {match_text}"
                
                self.result_label.config(text=result_text, fg=match_color, 
                                       font=("Helvetica", 11, "bold"))
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during comparison: {str(e)}")
            self.status_label.config(text="Error occurred during comparison")


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceComparerApp(root)
    root.mainloop()
