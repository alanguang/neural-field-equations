'''
神经场方程例1的求解
'''
# import numpy as np
# import tensorflow as tf
# import deepxde as dde
# import matplotlib.pyplot as plt
# from scipy.special import roots_legendre
#
# # 设置例1中的参数
# sigma = 1.0
# lamda = 1.0
# c = 1.0
# pi = np.pi
# num_points = 100  # 高斯积分点数
#
# def K(x, y):  # 定义函数K
#     diff1 = x[:, 0:1] - y[:, 0:1]
#     diff2 = x[:, 1:2] - y[:, 1:2]
#     return tf.exp(-lamda * tf.square(diff1) - lamda * tf.square(diff2))
#
# def S(x):  # 定义函数S
#     return tf.math.tanh(sigma * x)
#
# def approximate_integral(x, S_V_func):  # 定义积分项
#     integral_approx = 0
#     roots, weights = roots_legendre(num_points)
#     for i in range(num_points):
#         y = roots[i]
#         y_vec = tf.convert_to_tensor([[y, y]], dtype=tf.float32)
#         integral_approx += weights[i] * K(x, y_vec) * S_V_func(y_vec)
#     return integral_approx
#
# def true_solution(X):  # 设置精确解用来求解L2误差变化
#     t = X[:, 2:3]  # 第三列是时间信息
#     return np.exp(-t / c)  # 真实解
#
# # 定义ide方程
# def ide(X, V):  # 定义积分微分方程
#     x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
#
#     # 定义 I(x, t)
#     sqrt_lamda = tf.sqrt(lamda)
#     I_val = -tf.math.tanh(sigma * tf.exp(-t / c)) * (pi / (4 * lamda)) * (
#         tf.math.erf(sqrt_lamda * (1 - x)) + tf.math.erf(sqrt_lamda * (1 + x))
#     ) * (
#         tf.math.erf(sqrt_lamda * (1 - y)) + tf.math.erf(sqrt_lamda * (1 + y))
#     )
#
#     # 逼近 K * S(V)的积分
#     integral = approximate_integral(tf.concat([x, y], axis=1), lambda z: S(V))
#
#     # 使用 TensorFlow 计算dV/dt
#     dV_dt = dde.grad.jacobian(V, X, i=0, j=2)
#
#     # 定义PDE方程
#     return I_val - V + integral - c * dV_dt
#
# # 时间空间域
# geom = dde.geometry.Rectangle([-1, -1], [1, 1])  # 二维区域
# timedomain = dde.geometry.TimeDomain(0, 0.1)  # 时间域（0,0.1）
# geomtime = dde.geometry.GeometryXTime(geom, timedomain)  # 时间和空间区域
#
# def ic(X):  # 初始条件V(x, y, 0) = 1，返回全1矩阵作为初始条件
#     return np.ones_like(X[:, 2:3])
#
# # 初始条件
# ic_func = dde.icbc.IC(geomtime, ic, lambda _, on_initial: on_initial)
#
# # 定义pinn模型
# data = dde.data.TimePDE(geomtime, ide, [ic_func], num_domain=2580, num_boundary=80, num_initial=160)
# net = dde.nn.FNN([3] + [10] * 3 + [1], "tanh", "Glorot normal")
# model = dde.Model(data, net)
#
# # 训练模型
# model.compile('adam', lr=0.001)
# losshistory, train_state = model.train(iterations=15000)
#
# # 预测和测试解
# X_test = geomtime.random_points(1000)
# y_test = true_solution(X_test)
# y_pred = model.predict(X_test)
#
# # 计算L2误差
# error = dde.metrics.l2_relative_error(y_test, y_pred)
# print(f"Relative L2 error: {error}")
#
# # 画图
# x_plot = np.linspace(-1, 1, 100)
# y_plot = np.linspace(-1, 1, 100)
# t_plot = 0.05  # 固定某个时间点t=0.05
# X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
# T_plot = np.full_like(X_plot, t_plot)
#
# # 准备数据
# X_flat = X_plot.flatten()
# Y_flat = Y_plot.flatten()
# T_flat = T_plot.flatten()
# XYT = np.vstack((X_flat, Y_flat, T_flat)).T
#
# # 预测和真实解
# V_pred = model.predict(XYT).reshape(X_plot.shape)
# V_true = true_solution(XYT).reshape(X_plot.shape)
#
# fig = plt.figure(figsize=(14, 6))
#
# # 画近似解
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.plot_surface(X_plot, Y_plot, V_pred, cmap='viridis')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('V')
# ax1.set_title('Approximated Solution V')
#
# # 画真实解
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot_surface(X_plot, Y_plot, V_true, cmap='viridis')
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('V')
# ax2.set_title('True Solution V')
#
# plt.tight_layout()
# plt.show()
#
# # 误差图
# error = np.abs(V_pred - V_true)
# fig, ax = plt.subplots(figsize=(8, 6))
# c = ax.pcolormesh(X_plot, Y_plot, error, shading='auto', cmap='viridis')
# fig.colorbar(c, ax=ax)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Error between predicted and true solutions')
# plt.show()

'''
例1添加噪声项
'''
# import numpy as np
# import tensorflow as tf
# import deepxde as dde
# import matplotlib.pyplot as plt
# from scipy.special import roots_legendre
#
# # 设置例1中的参数
# sigma = 1.0
# lamda = 1.0
# c = 1.0
# pi = np.pi
# num_points = 50  # 高斯积分点数
# noise_stddev = 0.1  # 噪声标准差
#
# def K(x, y):  # 定义函数K
#     diff1 = x[:, 0:1] - y[:, 0:1]
#     diff2 = x[:, 1:2] - y[:, 1:2]
#     return tf.exp(-lamda * tf.square(diff1) - lamda * tf.square(diff2))
#
#
# def S(x):  # 定义函数S
#     return tf.math.tanh(sigma * x)
#
#
# def approximate_integral(x, S_V_func):  # 定义积分项
#     integral_approx = 0
#     roots, weights = roots_legendre(num_points)
#     for i in range(num_points):
#         y = roots[i]
#         y_vec = tf.convert_to_tensor([[y, y]], dtype=tf.float32)
#         integral_approx += weights[i] * K(x, y_vec) * S_V_func(y_vec)
#     return integral_approx
#
#
# def true_solution(X):  # 设置精确解用来求解L2误差变化
#     t = X[:, 2:3]  # 第三列是时间信息
#     return np.exp(-t / c)  # 真实解
#
#
# # 定义ide方程
# def ide(X, V):  # 定义积分微分方程
#     x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
#
#     # 定义 I(x, t)
#     sqrt_lamda = tf.sqrt(lamda)
#     I_val = -tf.math.tanh(sigma * tf.exp(-t / c)) * (pi / (4 * lamda)) * (
#             tf.math.erf(sqrt_lamda * (1 - x)) + tf.math.erf(sqrt_lamda * (1 + x))
#     ) * (
#                     tf.math.erf(sqrt_lamda * (1 - y)) + tf.math.erf(sqrt_lamda * (1 + y))
#             )
#
#     # 逼近 K * S(V)的积分
#     integral = approximate_integral(tf.concat([x, y], axis=1), lambda z: S(V))
#     # 生成随机噪声项
#     noise = tf.random.normal(shape=tf.shape(V), mean=0.0, stddev=noise_stddev)
#     # 使用 TensorFlow 计算dV/dt
#     dV_dt = dde.grad.jacobian(V, X, i=0, j=2)
#
#     # 定义PDE方程
#     # return I_val - V + integral - c * dV_dt
#     return I_val - V + integral - c * dV_dt + noise
#
#
# # 时间空间域
# geom = dde.geometry.Rectangle([-1, -1], [1, 1])  # 二维区域
# timedomain = dde.geometry.TimeDomain(0, 0.1)  # 时间域（0,0.1）
# geomtime = dde.geometry.GeometryXTime(geom, timedomain)  # 时间和空间区域
#
#
# def ic(X):  # 初始条件V(x, y, 0) = 1，返回全1矩阵作为初始条件
#     return np.ones_like(X[:, 2:3])
#
#
# # 初始条件1
# ic_func = dde.icbc.IC(geomtime, ic, lambda _, on_initial: on_initial)
#
# # 定义pinn模型
# data = dde.data.TimePDE(geomtime, ide, [ic_func], num_domain=2540, num_boundary=80, num_initial=160)
# net = dde.nn.FNN([3] + [10] * 3 + [1], "tanh", "Glorot normal")
# model = dde.Model(data, net)
#
# # 训练模型
# model.compile('adam', lr=0.001)
# losshistory, train_state = model.train(iterations=20000)
#
# # 预测和测试解
# X_test = geomtime.random_points(1000)
# y_test = true_solution(X_test)
# y_pred = model.predict(X_test)
#
# # 计算L2误差
# error = dde.metrics.l2_relative_error(y_test, y_pred)
# print(f"Relative L2 error: {error}")
#
# losshistory, train_state = model.train(iterations=20000)  # 保存损失历史和历史状态
# # model.compile("L-BFGS")
# # losshistory, train_state = model.train()
# dde.saveplot(losshistory, train_state, issave=False, isplot=True)
#
# # 选择固定的空间位置，例如 x=0.5, y=0.5
#
#
# x_fixed = 0.5  # 0.5
# y_fixed = 0.5  # 0.5
# t_values = np.linspace(0, 0.1, 1000)
# X_fixed = np.array([[x_fixed, y_fixed, t] for t in t_values])
#
# # 预测和真实解
# V_pred_fixed = model.predict(X_fixed)
# V_true_fixed = true_solution(X_fixed)
#
# # 画图
# plt.figure(figsize=(10, 6))
# plt.plot(t_values, V_pred_fixed, label='Predicted V', linestyle='--')
# plt.plot(t_values, V_true_fixed, label='True V')
# plt.xlabel('Time')
# plt.ylabel('V')
# plt.title('Comparison of Predicted and True V at (x, y) = (0.5, 0.5)')
# plt.legend()
# plt.grid(True)
# plt.show()