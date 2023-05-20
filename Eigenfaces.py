import numpy as np
import os
from PIL import Image
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import random

#function to load the images in sets, every set includes the number of pictures of every face according to the list defined below. The images are name person01_01.png etc.
def loadImages(path, set_number):
    # Define the sets
    sets = {
        1: (1, 7),
        2: (8, 19),
        3: (20, 31),
        4: (32, 45),
        5: (46, 64) }

    images = []
    labels = []

    # Get the image range for the specified set number
    start_image, end_image = sets[set_number]
    image_files = os.listdir(path)

    for image_name in image_files:
        person, image_no = image_name.split('_')
        person = int(person[6:])
        image_no = int(image_no.split('.')[0])

        if (image_no >= start_image) and (image_no <= end_image):
            image_path = os.path.join(path, image_name)
            image = Image.open(image_path)

            # Convert the image to a NumPy array and flatten it
            image_array = np.array(image).flatten()
            images.append(image_array)

            label = person
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels
 
#chech that everything went well in the creation of the matrices and their shapes are correct
set_number = list(range(1,6))

for Z in set_number:

    images, labels = loadImages(path, Z)
    images= pd.DataFrame(images)
    labels= pd.DataFrame(labels)
    # Print the shape of the data matrix and labels
    print(f"Images of set {Z}: \n", images)
    print("Shape of images:", images.shape)
    print(f"Labels of set {Z}: \n", labels)
    print("Shape of labels:", labels.shape)
    
#create the images/label sets and bring them to float type needed for the math of preprocessing

images_1, labels_1 = loadImages(path, 1)
images_2, labels_2 = loadImages(path, 2)
images_3, labels_3 = loadImages(path, 3)
images_4, labels_4 = loadImages(path, 4)
images_5, labels_5 = loadImages(path, 5)

images_1 = images_1.astype(float)
images_2 = images_2.astype(float)
images_3 = images_3.astype(float)
images_4 = images_4.astype(float)
images_5 = images_5.astype(float)

#preprocess by extracting the mean of each image and divide by st.dev.
def preprocess_images(images):
    # Subtract the average value of each image
    images -= np.mean(images, axis=0)
    # Divide by the standard deviation of each image's values
    images /= np.std(images, axis=0)
    
    return images

pre_images_1 = preprocess_images(images_1)
pre_images_2 = preprocess_images(images_2)
pre_images_3 = preprocess_images(images_3)
pre_images_4 = preprocess_images(images_4)
pre_images_5 = preprocess_images(images_5)


def create_eigenfaces(images, d):
    pca = PCA(n_components=d)
    pca.fit(images)
    eigenfaces = pca.components_
    return eigenfaces
    
def project_images(images, eigenfaces):
    return np.dot(images, eigenfaces.T)
    
def face_identification(X_train, y_train, X_test):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return y_pred
    
# Apply PCA to obtain eigenfaces
d = 9  # Set the number of components
eigenfaces_9 = create_eigenfaces(pre_images_1, d)

# Transform images into eigen space
X_train = project_images(pre_images_1, eigenfaces_9)
y_train = labels_1


#apply face id for the rest of the sets here is an example for set 2, you can try with different components eg d=30
Set_2_test = project_images(pre_images_2, eigenfaces_9)
y_pred = face_identification(X_train, y_train, Set_2_test)

accuracy = accuracy_score(labels_2, y_pred)
print("Accuracy for Set 2:", accuracy*100,"%")


#construct the images of eigenvectors (top 9)

for e, ax in enumerate(axes.flat):
    eigenface = eigenfaces_30[e].reshape(50, 50) 
    ax.imshow(eigenface, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Eigenface {e+1}')

plt.tight_layout()
plt.show()

#compare eigenvectors for different pca components

# Create a list of pre_images variables
pre_images = [pre_images_1, pre_images_2, pre_images_3, pre_images_4, pre_images_5]

# Create subplots for original and reconstructed images
fig, axes = plt.subplots(5, 3, figsize=(6, 10))
fig.suptitle('Original and Reconstructed Images')

for i, ax_row in enumerate(axes):
    # Randomly select an image from each set
    set_images = [pre_images_1, pre_images_2, pre_images_3, pre_images_4, pre_images_5]
    chosen_image = random.choice(set_images[i])

    # Reshape the chosen image
    original_image = chosen_image.reshape(50, 50)

    # Reshape the chosen image for the dot product
    chosen_image_reshaped = chosen_image.reshape(-1, 1)

    # Reconstruct the image using eigenface for component 9
    reconstructed_image_9 = (eigenfaces_9.T @ eigenfaces_9 @ chosen_image_reshaped).reshape(50, 50)

    # Reconstruct the image using eigenface for component 30
    reconstructed_image_30 = (eigenfaces_30.T @ eigenfaces_30 @ chosen_image_reshaped).reshape(50, 50)

    # Display the original chosen image
    ax_row[0].imshow(original_image, cmap='gray')
    ax_row[0].axis('off')
    ax_row[0].set_title(f'Original from Set {i+1}', fontsize=10)

    # Display the reconstructed image using eigenface for component 9
    ax_row[1].imshow(reconstructed_image_9, cmap='gray')
    ax_row[1].axis('off')
    ax_row[1].set_title('Reconstructed (d=9)', fontsize=10)

    # Display the reconstructed image using eigenface for component 30
    ax_row[2].imshow(reconstructed_image_30, cmap='gray')
    ax_row[2].axis('off')
    ax_row[2].set_title('Reconstructed (d=30)', fontsize=10)

plt.tight_layout()
plt.show()

#compare pca with svd on eigenfaces

# Perform SVD on the images in set 1
U, S, VT = np.linalg.svd(pre_images_1.T, full_matrices=False)

# Create subplots for the top 9 singular vectors
fig, axes = plt.subplots(2, 9, figsize=(15, 5))
fig.suptitle('Top 9 Singular Vectors (SVD) vs Top 9 Eigenfaces')

# Display the top 9 singular vectors
for i, ax in enumerate(axes.flatten()[:9]):
    singular_vector = U[:, i].reshape(50, 50)

    ax.imshow(singular_vector, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Singular Vector {i+1}', fontsize=10)

for i, ax in enumerate(axes.flatten()[9:]):
    eigenface = eigenfaces_30[i, :].reshape(50, 50)

    ax.imshow(eigenface, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Eigenface {i+1}', fontsize=10)

plt.tight_layout()
plt.show()


