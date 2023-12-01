import numpy as np
import cv2
from sklearn.cluster import KMeans
import time

def load_brain_tumor_image(image_path):
    return cv2.imread(image_path)

def two_layer_kmeans(image, k1, k2, max_iters=100):
    height, width, _ = image.shape
    data = image.reshape((height * width, 3))

    # First layer K-means clustering
    start = time.time()
    kmeans1 = KMeans(n_clusters=k1, max_iter=max_iters, random_state=0)
    labels1 = kmeans1.fit_predict(data)
    stop = time.time()
    sequential_time_1 = stop - start

    # Assign each pixel to a new cluster based on the labels from the first layer
    augmented_data = np.concatenate((data, labels1[:, np.newaxis]), axis=1)

    # Second layer K-means clustering
    start = time.time()
    kmeans2 = KMeans(n_clusters=k2, max_iter=max_iters, random_state=0)
    labels2 = kmeans2.fit_predict(augmented_data[:, :-1])
    stop = time.time()
    sequential_time_2 = stop - start

    sequential_time = sequential_time_1 + sequential_time_2
    print('Sequential execution time in seconds:', sequential_time)

    # Reshape the labels to the original image shape
    segmented_image = labels2.reshape((height, width))

    return segmented_image, sequential_time

if __name__ == "__main__":
    # Load an example brain tumor image
    image_path = "file/image2.jpg"  # Replace with the actual path
    image = load_brain_tumor_image(image_path)

    # Define parameters
    k1 = 5  # Number of clusters in the first layer
    k2 = 3  # Number of clusters in the second layer

    # Perform two-layer K-means clustering for brain tumor image segmentation
    segmented_image, sequential_time = two_layer_kmeans(image, k1, k2)

    # Display the original and segmented images
    cv2.imshow("Original Image", image)
    cv2.imshow("Segmented Image", (segmented_image * (255 / k2)).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('Sequential execution time in seconds:', sequential_time)