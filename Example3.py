import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

c = 1
noise_stddev = 0.1  # 噪声标准差

# Define the domain
geom = dde.geometry.geometry_2d.Rectangle([-1, -1], [1, 1])
timedomain = dde.geometry.TimeDomain(0, 2)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# Define the kernel function
def K(x1, x2, y1, y2):
    return tf.exp(-tf.square(x1 - y1) - tf.square(x2 - y2))


def true_solution(X):
    x1, x2, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    return np.exp(-t / c) * np.exp(-(x1 ** 2 + x2 ** 2))

# 函数I使用高斯积分
def I(x1, x2, t):
    def integrand(y1, y2, x1, x2):
        return tf.exp(-tf.square(x1 - y1) - tf.square(x2 - y2) - tf.square(y1) - tf.square(y2))

    # Use Gauss-Legendre nodes and weights
    num_points = 10
    y1_vals, w1 = np.polynomial.legendre.leggauss(num_points)
    y2_vals, w2 = np.polynomial.legendre.leggauss(num_points)
    y1_vals = y1_vals.astype(np.float32)
    y2_vals = y2_vals.astype(np.float32)
    w1 = w1.astype(np.float32)
    w2 = w2.astype(np.float32)

    integral = tf.constant(0, dtype=tf.float32)
    for i in range(num_points):
        for j in range(num_points):
            y1 = y1_vals[i]
            y2 = y2_vals[j]
            weight = w1[i] * w2[j]
            integral += integrand(y1, y2, x1, x2) * weight

    return -tf.exp(-t / c) * integral



def gauss_legendre_weights(n):  # Define Gauss-Legendre nodes and weights
    x, w = np.polynomial.legendre.leggauss(n)
    return x.astype(np.float32), w.astype(np.float32)


def pde(x, V):
    x1, x2, t = x[:, 0:1], x[:, 1:2], x[:, 2:3]

    # Call Python function I using tf.py_function
    I_val = tf.py_function(I, [x1, x2, t], tf.float32)
    V_val = V

    def integrand(y1, y2, x1, x2, t):
        K_val = tf.exp(-tf.square(x1 - y1) - tf.square(x2 - y2))
        V_y = tf.exp(-t) * tf.exp(-tf.square(y1) - tf.square(y2))
        return K_val * V_y

    # Use Gauss-Legendre nodes and weights
    num_points = 10
    y1_vals, w1 = gauss_legendre_weights(num_points)
    y2_vals, w2 = gauss_legendre_weights(num_points)

    integral = tf.zeros_like(x1, dtype=tf.float32)
    for i in range(num_points):
        for j in range(num_points):
            y1 = y1_vals[i]
            y2 = y2_vals[j]
            weight = w1[i] * w2[j]
            integral += integrand(y1, y2, x1, x2, t) * weight
        # 生成随机噪声项
    noise = tf.random.normal(shape=tf.shape(V), mean=0.0, stddev=noise_stddev)
    V_t = dde.grad.jacobian(V, x, i=0, j=2)

    # return I_val - V_val + integral - c * V_t
    return I_val - V_val + integral - c * V_t + noise


def init_cond(x):
    return np.exp(-x[:, 0:1] ** 2 - x[:, 1:2] ** 2).astype(np.float32)


ic = dde.IC(geomtime, init_cond, lambda _, on_initial: on_initial)

data = dde.data.TimePDE(
    geomtime, pde, [ic], num_domain=180, num_boundary=20, num_initial=20
)
net = dde.nn.FNN([3] + [32] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Train the model
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=5000)
# 预测和测试解
X_test = geomtime.random_points(1000)
y_test = true_solution(X_test)
y_pred = model.predict(X_test)

# 计算L2误差
error = dde.metrics.l2_relative_error(y_test, y_pred)
print(f"Relative L2 error: {error}")

losshistory, train_state = model.train(iterations=5000)  # 保存损失历史和历史状态
model.compile("L-BFGS-B")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

# Plot predicted solution, true solution, and error for different t values
t_values = [0, 0.5, 1, 1.5, 2]

for t in t_values:
    # 预测解图
    plt.figure()  # 创建一个新的图像
    ax1 = plt.axes(projection='3d')
    X_pred = np.linspace(-1, 1, 100)
    Y_pred = np.linspace(-1, 1, 100)
    X_pred, Y_pred = np.meshgrid(X_pred, Y_pred)
    T_pred = np.ones_like(X_pred) * t
    X_pred_flat = X_pred.flatten()
    Y_pred_flat = Y_pred.flatten()
    T_pred_flat = T_pred.flatten()
    Z_pred = model.predict(np.vstack((X_pred_flat, Y_pred_flat, T_pred_flat)).T)
    Z_pred = Z_pred.reshape(X_pred.shape)
    ax1.plot_surface(X_pred, Y_pred, Z_pred, cmap='viridis')
    ax1.set_title(f"Example 3: t={t} predicted solutions")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.tight_layout()
    plt.show()

    # 真实解图
    plt.figure()  # 创建一个新的图像
    ax2 = plt.axes(projection='3d')
    Z_true = true_solution(np.vstack((X_pred_flat, Y_pred_flat, T_pred_flat)).T)
    Z_true = Z_true.reshape(X_pred.shape)
    ax2.plot_surface(X_pred, Y_pred, Z_true, cmap='viridis')
    ax2.set_title(f"Example 3: t={t} true solutions")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.tight_layout()
    plt.show()

    # 误差图
    plt.figure()  # 创建一个新的图像
    ax3 = plt.axes(projection='3d')
    error = np.abs(Z_pred - Z_true)
    ax3.plot_surface(X_pred, Y_pred, error, cmap='viridis')
    ax3.set_title(f"Example 3: t={t} Error plot between true and predicted solutions")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Error')
    plt.tight_layout()
    plt.show()
