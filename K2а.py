import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Глобальні змінні для зберігання мінімальних та максимальних векторів
min_vectors = {}
max_vectors = {}

# Функції для обробки зображень
def crop_image(image):
    coordinates = np.column_stack(np.where(image == 0))
    if coordinates.size == 0:
        return image

    top_left = coordinates.min(axis=0)
    bottom_right = coordinates.max(axis=0)

    cropped_image = image[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]
    return cropped_image

def process_image(file_path, threshold_value):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    _, thresh_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    cropped_image = crop_image(thresh_image)
    return cropped_image

def calculate_feature_vector(image, num_segments):
    height, width = image.shape
    feature_vector = []
    angles = np.linspace(0, 90, num_segments + 1)[1:]  # Кути для поділу

    for angle in angles:
        rad_angle = np.radians(angle)
        x_start = width  # Початок з правого нижнього кута
        y_start = height
        x_end = 0
        y_end = int((width - x_end) * np.tan(rad_angle))

        if y_end > height:  # Якщо y_end перевищує висоту зображення
            y_end = height
            x_end = width - int(height / np.tan(rad_angle))

        # Створення маски для кожного сегмента
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon_points = np.array([[width, height], [x_end, height - y_end], [0, 0], [width, 0]], np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [polygon_points], 255)

        masked_image = np.zeros((height, width), dtype=np.uint8)
        for x in range(width):
            for y in range(height):
                if image[y, x] == mask[y, x] and image[y, x] == 0:
                    masked_image[y, x] = 1

        black_pixels = np.sum(masked_image == 1)
        previous_sum = sum(feature_vector)
        adjusted_black_pixels = black_pixels - previous_sum
        feature_vector.append(adjusted_black_pixels)

    return feature_vector


def display_image(image, num_segments, name_image):
    height, width = image.shape
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    angles = np.linspace(0, 90, num_segments + 1)[:-1]

    for angle in angles:
        rad_angle = np.radians(angle)
        x_start = width  # Початок з правого нижнього кута
        y_start = height
        x_end = 0
        y_end = int((width - x_end) * np.tan(rad_angle))

        if y_end > height:
            y_end = height
            x_end = width - int(height / np.tan(rad_angle))

        if angle == 90:
            x_end = 0
            y_end = 0

        ax.plot([width, x_end], [height, height - y_end], color='r', linestyle='-', linewidth=1)
        ax.text(x_end + 2, height - y_end + 2, str(int(angle)), color='r', fontsize=10, verticalalignment='center')

    plt.title(name_image)
    plt.show()


def normalize_sum(vector):
    total_sum = sum(vector)
    if total_sum == 0:
        return [0 for v in vector]
    return [v / total_sum for v in vector]

def normalize_max(vector):
    max_value = max(vector)
    if max_value == 0:
        return [0 for v in vector]
    return [v / max_value for v in vector]

def print_feature_info(image_file, feature_vector, normalized_sum_vector, normalized_max_vector):
    print("-" * 50)
    print(f"Image: {image_file}")
    print(f"absolute vector: {[int(val) for val in feature_vector]}")
    print(f"Normalized vector (by sum): {[float(val) for val in normalized_sum_vector]}")
    print(f"Normalized vector (by max): {[float(val) for val in normalized_max_vector]}")

def classify_unknown_image(unknown_vector, min_vector, max_vector):
    for i in range(len(min_vector)):
        if not (min_vector[i] <= unknown_vector[i] <= max_vector[i]):
            return False
    return True

def classify_image(normalized_vector_sum):
    class_names = ["Class_1", "Class_2", "Class_3"]
    for class_num in range(1, 4):
        if classify_unknown_image(normalized_vector_sum, min_vectors[class_num], max_vectors[class_num]):
            return class_names[class_num - 1]
    return None

def process_image_file():
    image_path = file_entry.get()
    if not os.path.isfile(image_path):
        result_label.config(text="Неправильний шлях до зображення!")
        return

    try:
        threshold = int(threshold_entry.get())
        num_segments = int(segments_entry.get())
    except ValueError:
        result_label.config(text="Будь ласка, введіть дійсні числа для порогового значення і кількості сегментів!")
        return

    image = process_image(image_path, threshold)
    feature_vector = calculate_feature_vector(image, num_segments)

    normalized_vector_sum = normalize_sum(feature_vector)
    normalized_vector_max = normalize_max(feature_vector)

    print_feature_info(os.path.basename(image_path), feature_vector, normalized_vector_sum, normalized_vector_max)
    display_image(image, num_segments, os.path.basename(image_path))

def classify_uploaded_image():
    image_path = file_entry.get()
    if not os.path.isfile(image_path):
        result_label.config(text="Неправильний шлях до зображення!")
        return

    try:
        threshold = int(threshold_entry.get())
        num_segments = int(segments_entry.get())
    except ValueError:
        result_label.config(text="Будь ласка, введіть дійсні числа для порогового значення і кількості сегментів!")
        return

    image = process_image(image_path, threshold)
    feature_vector = calculate_feature_vector(image, num_segments)
    normalized_vector_sum = normalize_sum(feature_vector)

    result = classify_image(normalized_vector_sum)
    if result:
        result_label.config(text=f"Unknown image belongs to: {result}")
    else:
        result_label.config(text="Unknown image does not belong to any class.")

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

def main():
    global min_vectors, max_vectors

    # Введення еталонних образів
    class1_images = [
        "C:/image/class1/1.png",
        "C:/image/class1/2.png",
        "C:/image/class1/3.png",
        "C:/image/class1/4.png",
        "C:/image/class1/5.png",
        "C:/image/class1/6.png"
    ]

    class2_images = [
        "C:/image/class2/1.png",
        "C:/image/class2/2.png",
        "C:/image/class2/3.png",
        "C:/image/class2/4.png",
        "C:/image/class2/5.png",
        "C:/image/class2/6.png"
    ]

    class3_images = [
        "C:/image/class3/1.png",
        "C:/image/class3/2.png",
        "C:/image/class3/3.png",
        "C:/image/class3/4.png",
        "C:/image/class3/5.png",
        "C:/image/class3/6.png"
    ]

    threshold = 150
    num_segments = 5

    # Обробка еталонних образів
    class_vectors = {1: [], 2: [], 3: []}
    for class_num, images in zip(range(1, 4), [class1_images, class2_images, class3_images]):
        for image_file in images:
            image = process_image(image_file, threshold)
            feature_vector = calculate_feature_vector(image, num_segments)
            normalized_vector_sum = normalize_sum(feature_vector)
            normalized_vector_max = normalize_max(feature_vector)

            class_vectors[class_num].append((feature_vector, normalized_vector_sum, normalized_vector_max))

    # Обчислення мінімальних та максимальних векторів
    min_vectors = {1: [], 2: [], 3: []}
    max_vectors = {1: [], 2: [], 3: []}
    for class_num in range(1, 4):
        min_vectors[class_num] = [min(vec[1][i] for vec in class_vectors[class_num]) for i in range(num_segments)]
        max_vectors[class_num] = [max(vec[1][i] for vec in class_vectors[class_num]) for i in range(num_segments)]

    root = tk.Tk()
    root.title("Image Classifier")
    root.geometry("500x400")
    root.configure(bg="#f0f0f0")

    # Вибір зображення
    tk.Label(root, text="Виберіть зображення:", bg="#f0f0f0").pack(pady=5)
    global file_entry
    file_entry = tk.Entry(root, width=50)
    file_entry.pack(pady=5)
    tk.Button(root, text="Browse", command=browse_file).pack(pady=5)

    # Порогове значення
    tk.Label(root, text="Порогове значення:", bg="#f0f0f0").pack(pady=5)
    global threshold_entry
    threshold_entry = tk.Entry(root, width=10)
    threshold_entry.pack(pady=5)
    threshold_entry.insert(0, "150")

    # Кількість сегментів
    tk.Label(root, text="Кількість сегментів:", bg="#f0f0f0").pack(pady=5)
    global segments_entry
    segments_entry = tk.Entry(root, width=10)
    segments_entry.pack(pady=5)
    segments_entry.insert(0, "5")

    # Кнопка обробки зображення
    tk.Button(root, text="Process Image", command=process_image_file).pack(pady=10)
    
    # Кнопка класифікації зображення
    tk.Button(root, text="Classify Image", command=classify_uploaded_image).pack(pady=10)

    # Мітка для відображення результату класифікації
    global result_label
    result_label = tk.Label(root, text="", bg="#f0f0f0", font=("Arial", 12))
    result_label.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
