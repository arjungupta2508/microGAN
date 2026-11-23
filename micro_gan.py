import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(X_train,_), (_, _) = mnist.load_data() # for unconditionalGAN we dont need the labels for training

X_train = X_train.astype(np.float32) # convert to floating point value
X_train = (X_train / 127.5) - 1.0  #normalization of pixel values to range(1,-1)
X_train = X_train.reshape(-1, 784) #reshape images to a single vector of 784 pixels ( -1 auto adjusts according to no of samples)

print("Dataset shape:", X_train.shape)

class LeakyReLu:

    def __init__(self, alpha=0.2): #this is a small value multiplied to negatuve values so they are not exactly 0
        self.alpha = alpha

    def forward(self, X):   # same value if x> 0 and alpha*x if x < 0
        self.X = X
        return np.where(X > 0, X, self.alpha * X)

    def backward(self, grad): # 1 if x > 0 and alpha if x < 0 ( -0.2)
        derivative = np.where(self.X > 0, 1.0, self.alpha)
        return grad * derivative

class Tanh:  #tanh converts output to be on a range of [-1,1]

    def forward(self, X):
        self.output = np.tanh(X)
        return self.output

    def backward(self, grad): # derivative of tanh is 1- tanh**2
        return grad * (1.0 - self.output ** 2)

class Sigmoid: #squashes the real vale between [0,1] and introduces non linearity

    def forward(self, X):
        self.output = 1.0 / (1.0 + np.exp(-np.clip(X, -50, 50)))
        return self.output

    def backward(self, grad):
        return grad * self.output * (1.0 - self.output)

class Dropout:    #dropout layer deactivates random neurons during training so our model does not
                  #overfit and doesnt particularly depend on some certain neurons :)
    def __init__(self, rate=0.3):
        self.rate = rate
        self.mask = None

    def forward(self, X, training=True):
        if not training or self.rate == 0.0:
            return X
        keep_prob = 1.0 - self.rate
        self.mask = (np.random.rand(*X.shape) < keep_prob) / keep_prob
        return X * self.mask

    def backward(self, grad):
        return grad * self.mask

class Dense:

    def __init__(self, input_dim, output_dim, beta1=0.5, beta2=0.999, eps=1e-8):

        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, grad, accumulate=False):
        dW = self.X.T @ grad
        db = np.sum(grad, axis=0, keepdims=True)

        if accumulate:
            self.dW += dW
            self.db += db
        else:
            self.dW = dW
            self.db = db

        return grad @ self.W.T

    def update(self, lr):
        self.t += 1

        # Update Weights
        self.mW = self.beta1 * self.mW + (1.0 - self.beta1) * self.dW
        self.vW = self.beta2 * self.vW + (1.0 - self.beta2) * (self.dW ** 2)
        mW_hat = self.mW / (1.0 - self.beta1 ** self.t)
        vW_hat = self.vW / (1.0 - self.beta2 ** self.t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + self.eps)

        # Update Biases
        self.mb = self.beta1 * self.mb + (1.0 - self.beta1) * self.db
        self.vb = self.beta2 * self.vb + (1.0 - self.beta2) * (self.db ** 2)
        mb_hat = self.mb / (1.0 - self.beta1 ** self.t)
        vb_hat = self.vb / (1.0 - self.beta2 ** self.t)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

        # Reset accumulated gradients
        self.dW.fill(0.0)
        self.db.fill(0.0)


# ==========================================
# 4. GENERATOR
# ==========================================

class Generator:

    def __init__(self):
        self.dense1 = Dense(100, 256)
        self.dense2 = Dense(256, 512)
        self.dense3 = Dense(512, 784)

        self.lrelu1 = LeakyReLu(0.2)
        self.lrelu2 = LeakyReLu(0.2)
        self.tanh = Tanh()

    def forward(self, z):
        z = self.lrelu1.forward(self.dense1.forward(z))
        z = self.lrelu2.forward(self.dense2.forward(z))
        z = self.tanh.forward(self.dense3.forward(z))
        return z

    def backward(self, grad):
        grad = self.dense3.backward(self.tanh.backward(grad))
        grad = self.dense2.backward(self.lrelu2.backward(grad))
        grad = self.dense1.backward(self.lrelu1.backward(grad))
        return grad

    def update(self, lr):
        self.dense1.update(lr)
        self.dense2.update(lr)
        self.dense3.update(lr)




class Discriminator:

    def __init__(self):
        self.dense1 = Dense(784, 512)
        self.dense2 = Dense(512, 256)
        self.dense3 = Dense(256, 1)

        self.lrelu1 = LeakyReLu(0.2)
        self.lrelu2 = LeakyReLu(0.2)
        self.drop1 = Dropout(0.3)
        self.drop2 = Dropout(0.3)
        self.sigmoid = Sigmoid()

    def forward(self, image, training=True):
        image = self.drop1.forward(self.lrelu1.forward(self.dense1.forward(image)), training)
        image = self.drop2.forward(self.lrelu2.forward(self.dense2.forward(image)), training)
        image = self.sigmoid.forward(self.dense3.forward(image))
        return image

    def backward(self, grad, accumulate=False):
        grad = self.dense3.backward(self.sigmoid.backward(grad), accumulate=accumulate)
        grad = self.dense2.backward(self.lrelu2.backward(self.drop2.backward(grad)), accumulate=accumulate)
        grad = self.dense1.backward(self.lrelu1.backward(self.drop1.backward(grad)), accumulate=accumulate)
        return grad

    def update(self, lr):
        self.dense1.update(lr)
        self.dense2.update(lr)
        self.dense3.update(lr)


# ==========================================
# 6. BINARY CROSS ENTROPY
# ==========================================

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.mean(
        y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
    )


def binary_cross_entropy_backward(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    N = y_true.shape[0]
    return (1.0 / N) * (-y_true / y_pred + (1.0 - y_true) / (1.0 - y_pred))




latent_dim = 100
batch_size = 64
epochs = 50       # With Adam, 50 epochs is plenty for crisp MNIST digits
lr = 0.0002

generator = Generator()
discriminator = Discriminator()

d_history = []
g_history = []


def generate_samples(generator, epoch, num_images=16):
    noise = np.random.randn(num_images, latent_dim)
    generated_images = generator.forward(noise).reshape(-1, 28, 28)

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i], cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")

    plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()
    plt.show()



for epoch in range(epochs):

    np.random.shuffle(X_train)

    d_losses = []
    g_losses = []

    for start in range(0, len(X_train), batch_size):

        real_images = X_train[start:start + batch_size]
        current_batch = real_images.shape[0]

        # ----------------------------------
        # STEP 1: DISCRIMINATOR ON REAL
        # ----------------------------------
        real_predictions = discriminator.forward(real_images, training=True)

        # One-sided label smoothing (0.9 instead of 1.0) prevents overconfidence
        real_labels = np.full_like(real_predictions, 0.9)

        d_real_loss = binary_cross_entropy(real_labels, real_predictions)
        real_grad = binary_cross_entropy_backward(real_labels, real_predictions)

        # Store gradients without updating yet
        discriminator.backward(real_grad, accumulate=False)

        # ----------------------------------
        # STEP 2: DISCRIMINATOR ON FAKE
        # ----------------------------------
        noise = np.random.randn(current_batch, latent_dim)
        fake_images = generator.forward(noise)

        fake_predictions = discriminator.forward(fake_images, training=True)
        fake_labels = np.zeros_like(fake_predictions)

        d_fake_loss = binary_cross_entropy(fake_labels, fake_predictions)
        fake_grad = binary_cross_entropy_backward(fake_labels, fake_predictions)

        # Accumulate fake gradients with real gradients, then update once
        discriminator.backward(fake_grad, accumulate=True)
        discriminator.update(lr)

        # ----------------------------------
        # STEP 3: TRAIN GENERATOR
        # ----------------------------------
        noise = np.random.randn(current_batch, latent_dim)
        fake_images = generator.forward(noise)

        fake_predictions = discriminator.forward(fake_images, training=True)
        generator_labels = np.ones_like(fake_predictions)

        g_loss = binary_cross_entropy(generator_labels, fake_predictions)
        gen_grad = binary_cross_entropy_backward(generator_labels, fake_predictions)

        # Backpropagate through discriminator (do NOT update discriminator)
        grad_fake_images = discriminator.backward(gen_grad, accumulate=False)

        # Clear discriminator's temporary gradients so they don't linger
        for layer in (discriminator.dense1, discriminator.dense2, discriminator.dense3):
            layer.dW.fill(0.0)
            layer.db.fill(0.0)

        # Backpropagate and update generator
        generator.backward(grad_fake_images)
        generator.update(lr)

        d_losses.append(d_real_loss + d_fake_loss)
        g_losses.append(g_loss)

    avg_d_loss = np.mean(d_losses)
    avg_g_loss = np.mean(g_losses)

    d_history.append(avg_d_loss)
    g_history.append(avg_g_loss)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"D Loss: {avg_d_loss:.4f} | "
        f"G Loss: {avg_g_loss:.4f}"
    )

    if (epoch + 1) % 5 == 0:
        generate_samples(generator, epoch + 1)


# ==========================================
# 9. PLOT TRAINING LOSSES & FINAL SAMPLES
# ==========================================

plt.figure(figsize=(10, 5))
plt.plot(d_history, label="Discriminator Loss")
plt.plot(g_history, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training Loss")
plt.legend()
plt.grid()
plt.show()

generate_samples(generator, "Final", 16)