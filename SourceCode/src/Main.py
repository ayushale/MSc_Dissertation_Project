import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import platform

class IDSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intrusion Detection System")

        # Cross-platform fullscreen adjustment
        if platform.system() == "Windows":
            self.root.state('zoomed')
        else:
            try:
                self.root.attributes('-zoomed', True)
            except:
                self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")

        # Title Label
        tk.Label(root, text="Intrusion Detection System", font=("Arial", 20, "bold")).pack(pady=10)

        # Frame for buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        # Buttons
        tk.Button(button_frame, text="Upload Train Dataset", command=self.load_train_dataset, width=20).grid(row=0, column=0, padx=10, pady=5)
        tk.Button(button_frame, text="Upload Test Dataset", command=self.load_test_dataset, width=20).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(button_frame, text="Train Model", command=self.train_model, width=20).grid(row=0, column=2, padx=10, pady=5)
        tk.Button(button_frame, text="Predict & Show Results", command=self.predict_and_visualize, width=20).grid(row=0, column=3, padx=10, pady=5)
        tk.Button(button_frame, text="Close Application", command=self.close_application, width=20).grid(row=0, column=4, padx=10, pady=5)

        # Frame for output and plots
        self.output_frame = tk.Frame(root)
        self.output_frame.pack(fill="both", expand=True, pady=10)

        # Output Text Box
        self.output_text = tk.Text(self.output_frame, height=15, width=80, font=("Arial", 12))
        self.output_text.pack(side="left", fill="y", padx=10)

        # Scrollbar for Output Text Box
        scrollbar = tk.Scrollbar(self.output_frame, command=self.output_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.output_text.configure(yscrollcommand=scrollbar.set)

        # Canvas for displaying plots
        self.plot_canvas = None

        # Initialize variables
        self.train_data = None
        self.test_data = None
        self.models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier()
        }
        self.feature_order = None
        self.model_trained = False

    def load_train_dataset(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filepath:
            self.train_data = pd.read_csv(filepath)
            messagebox.showinfo("Success", "Train dataset loaded successfully!")

    def load_test_dataset(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filepath:
            self.test_data = pd.read_csv(filepath)
            messagebox.showinfo("Success", "Test dataset loaded successfully!")

    def preprocess_data(self, data, fit_encoder=False):
        if fit_encoder:
            # Map labels to numeric values for training
            label_mapping = {"benign": 0, "outlier": 1, "malicious": 2}
            data['label'] = data['label'].map(label_mapping)

        # Drop unnecessary columns (e.g., IP addresses, which are anonymized and non-informative)
        columns_to_drop = ['src_ip', 'dest_ip', 'time_start', 'time_end']
        data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        # Handle missing values using SimpleImputer
        imputer = SimpleImputer(strategy="mean")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

        # Save feature order during training
        if fit_encoder:
            self.feature_order = data.drop(columns=['label'], errors='ignore').columns.tolist()

        # Ensure test data uses the same feature order
        else:
            data = data.reindex(columns=self.feature_order, fill_value=0)

        return data

    def clear_display(self):
        """Clear all existing content from the output and plot areas."""
        self.output_text.delete('1.0', tk.END)
        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()
            self.plot_canvas = None

    def train_model(self):
        self.clear_display()
        if self.train_data is not None:
            try:
                # Preprocess data
                self.train_data = self.preprocess_data(self.train_data, fit_encoder=True)
                X = self.train_data.drop(columns=['label'])
                y = self.train_data['label']

                # Split data into training and validation sets
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

                # Train and evaluate all models
                for model_name, model in self.models.items():
                    self.output_text.insert(tk.END, f"--- {model_name} Training ---\n")

                    # Handle NaN values for KNN
                    if model_name == "K-Nearest Neighbors":
                        imputer = SimpleImputer(strategy="mean")
                        X_train = imputer.fit_transform(X_train)
                        X_val = imputer.transform(X_val)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    accuracy = accuracy_score(y_val, y_pred)
                    self.output_text.insert(tk.END, f"Validation Accuracy: {accuracy:.2f}\n")

                    # Display classification report
                    report = classification_report(y_val, y_pred, target_names=['benign', 'outlier', 'malicious'])
                    self.output_text.insert(tk.END, f"\nClassification Report:\n{report}\n\n")

                    # Plot feature importance (if applicable)
                    if hasattr(model, "feature_importances_"):
                        self.plot_feature_importance(model.feature_importances_, X.columns, model_name)

                self.model_trained = True

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Please load the training dataset first.")

    def predict_and_visualize(self):
        self.clear_display()
        if not self.model_trained:
            messagebox.showwarning("Warning", "Please train the model first.")
            return

        if self.test_data is not None:
            try:
                # Preprocess test data
                self.test_data = self.preprocess_data(self.test_data.copy(), fit_encoder=False)
                X_test = self.test_data.drop(columns=['label'], errors='ignore')

                # Use only Random Forest for prediction
                model = self.models["Random Forest"]
                self.output_text.insert(tk.END, f"--- Prediction Results ---\n")

                y_pred = model.predict(X_test)
                class_counts = pd.Series(y_pred).value_counts()

                # Display all prediction counts (benign, outlier, malicious)
                for cls, count in class_counts.items():
                    cls_name = ['benign', 'outlier', 'malicious'][int(cls)]  # Ensure the index is an integer
                    self.output_text.insert(tk.END, f"{cls_name}: {count}\n")

                # Plot prediction distribution as a bar graph
                self.plot_prediction_distribution(class_counts)

                self.output_text.insert(tk.END, "\n")

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Please ensure both the test dataset and model are loaded.")

    def plot_prediction_distribution(self, class_counts):
        try:
            # Clear any existing plots
            if self.plot_canvas:
                self.plot_canvas.get_tk_widget().destroy()
                self.plot_canvas = None

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(
                x=class_counts.index.map({0: 'benign', 1: 'outlier', 2: 'malicious'}),
                y=class_counts.values,
                ax=ax
            )
            ax.set_title("Prediction Distribution", fontsize=16)
            ax.set_ylabel("Count", fontsize=14)
            ax.set_xlabel("Classes", fontsize=14)

            # Show plot in GUI
            self.plot_canvas = FigureCanvasTkAgg(fig, self.root)
            self.plot_canvas.get_tk_widget().pack()
            self.plot_canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def plot_feature_importance(self, importances, feature_names, model_name):
        try:
            # Clear any existing plots
            if self.plot_canvas:
                self.plot_canvas.get_tk_widget().destroy()
                self.plot_canvas = None

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x=importances, y=feature_names, ax=ax)
            ax.set_title(f" Feature Importance", fontsize=16)
            ax.set_xlabel("Importance", fontsize=14)
            ax.set_ylabel("Features", fontsize=14)

            # Show plot in GUI
            self.plot_canvas = FigureCanvasTkAgg(fig, self.root)
            self.plot_canvas.get_tk_widget().pack()
            self.plot_canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def close_application(self):
        self.root.quit()
        self.root.destroy()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = IDSApp(root)
    root.mainloop()
