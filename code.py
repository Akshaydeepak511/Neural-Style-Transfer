import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import time

def load_and_process_image(image_path, target_size=(512, 512)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(img):
    img = img.reshape((512, 512, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def get_vgg_model():
    model = vgg19.VGG19(weights='imagenet', include_top=False)
    model.trainable = False
    return model

def compute_loss(model, content_image, style_image, generated_image):
    content_layer = 'block5_conv2'
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    
    content_weight = 1e4
    style_weight = 1e-2
    
    content_output = model.get_layer(content_layer).output
    style_outputs = [model.get_layer(layer).output for layer in style_layers]
    
    loss = tf.zeros(shape=())
    content_loss = tf.reduce_mean(tf.square(content_output - model(content_image)[0]))
    loss += content_weight * content_loss
    
    for style_output in style_outputs:
        loss += style_weight * tf.reduce_mean(tf.square(style_output - model(style_image)))
    
    return loss

def train_nst(content_image_path, style_image_path, epochs=100):
    model = get_vgg_model()
    content_image = load_and_process_image(content_image_path)
    style_image = load_and_process_image(style_image_path)
    generated_image = tf.Variable(content_image, dtype=tf.float32)
    
    optimizer = tf.optimizers.Adam(learning_rate=5.0)
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, content_image, style_image, generated_image)
        grads = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(grads, generated_image)])
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Loss = {loss.numpy()}')
    
    return deprocess_image(generated_image.numpy())

# Example Usage:
styled_image = train_nst('content.jpg', 'style.jpg', epochs=100)
plt.imshow(styled_image)
plt.axis('off')
plt.show()
