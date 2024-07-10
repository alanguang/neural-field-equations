'''
使用中值法处理积分项
'''
import numpy as np
import deepxde as dde
import tensorflow as tf
import matplotlib.pyplot as plt


# 常数
sigma = 1.0
lamda = 1.0
c = 1.0
pi = np.pi
num_points = 50
# noise_stddev = 0.1  # 噪声标准差


# 定义K
def K(x, y):
    diff1 = x[:, 0:1] - y[:, 0:1]
    diff2 = x[:, 1:2] - y[:, 1:2]
    return tf.exp(-lamda * tf.square(diff1) - lamda * tf.square(diff2))


def S(v):
    return tf.math.tanh(sigma * v)

# 精确解
def true_solution(X):
    t = X[:, 2:3]
    return t

# 积分函数
def approximate_integral(x, S_V_func):
    # Sample uniformly spaced points
    y_points = np.linspace(-1, 1, num_points)
    # y_points_tf = tf.constant(y_points, dtype=tf.float32)

    # 初始化积分
    integral_approx = 0

    # 循环每个采样点
    for y in y_points:
        y_vec = tf.convert_to_tensor([[y, y]], dtype=tf.float32)
        integral_approx += K(x, y_vec) * S_V_func(y_vec)


    return integral_approx / num_points

# 定义PDE
def pde(X, V):
    x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]

    #  I(x, t)
    sqrt_lamda = tf.sqrt(lamda)
    I_val = 1 + t -tf.math.tanh(sigma * t) * (pi / (4 * lamda)) * (
        tf.math.erf(sqrt_lamda * (1 - x)) + tf.math.erf(sqrt_lamda * (1 + x))
    ) * (
        tf.math.erf(sqrt_lamda * (1 - y)) + tf.math.erf(sqrt_lamda * (1 + y))
    )

    # 计算 K * S(V)的积分
    integral = approximate_integral(tf.concat([x, y], axis=1), lambda z: S(V))

    # 生成随机噪声项
    # noise = tf.random.normal(shape=tf.shape(V), mean=0.0, stddev=noise_stddev)

    # 计算dV/dt
    dV_dt = dde.grad.jacobian(V, X, i=0, j=2)


    # return I_val - V + integral - c * dV_dt + noise
    return I_val - V + integral - c * dV_dt


geom = dde.geometry.Rectangle([-1, -1], [1, 1])
timedomain = dde.geometry.TimeDomain(0, 0.1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)



# Initial condition V(x, y, 0) = 0

def ic(X):
    return np.zeros_like(X[:, 2:3])

ic_func = dde.icbc.IC(geomtime, ic, lambda _, on_initial: on_initial)


data = dde.data.TimePDE(
    geomtime, pde, [ic_func], num_domain=2540, num_boundary=80, num_initial=160
)
net = dde.nn.FNN([3] + [10] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)


model.compile("adam", lr=0.0001)
losshistory, train_state = model.train(iterations=40000)

# 选择固定的空间位置，例如 x=0.5, y=0.5
x_fixed = 1
y_fixed = 1
t_values = np.linspace(0, 0.1, 100)
X_fixed = np.array([[x_fixed, y_fixed, t] for t in t_values])

# 预测和真实解
V_pred_fixed = model.predict(X_fixed)
V_true_fixed = true_solution(X_fixed)

# 画图
plt.figure(figsize=(10, 6))
plt.plot(t_values, V_pred_fixed, label='Predicted V', linestyle='--')
plt.plot(t_values, V_true_fixed, label='True V')
plt.xlabel('Time')
plt.ylabel('V')
plt.title('Comparison of Predicted and True V at (x, y) = (1, 1)')
plt.legend()
plt.grid(True)
plt.show()

# 计算L2误差
X_test = geomtime.random_points(1000)
y_test = true_solution(X_test)
y_pred = model.predict(X_test)

error = dde.metrics.l2_relative_error(y_test, y_pred)
print(f"Relative L2 error: {error}")

# 保存损失历史和训练状态
losshistory, train_state = model.train(iterations=1500)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=False, isplot=True)
print(f"Final Relative L2 error: {error}")
