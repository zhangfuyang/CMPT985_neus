import numpy as np
import skfmm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

grid = np.load('grid.npy')

grid[grid <=0] = -1

sdf = skfmm.distance(-grid, dx=1e-2)

figs, axes = plt.subplots(nrows=1, ncols=2)
fig = axes[0].imshow(grid)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title('Scene')
fig = axes[1].imshow(sdf)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_title('SDF')

# set camera view
camera_view = 45
s = 1
max_density_button = True
max_weight_button = True

figs, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
axes[0].imshow(grid)
# draw a horizontal dash red line
f0_0 = axes[0].axhline(y=camera_view, color='r', linestyle='--')
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title('Camera view')

sdf_line = sdf[camera_view, :]
s_density = np.exp(-sdf_line*s) / (1 + np.exp(-sdf_line*s))**2

# naive method
sigma_naive = s_density
alpha_naive = 1.0 - np.exp(-sigma_naive)
T_naive = np.cumprod(1.0 - alpha_naive)
weight_naive = T_naive * alpha_naive
max_weight_idx_naive = np.argmax(weight_naive)
max_sigma_idx_naive = np.argmax(sigma_naive)

f1_0 = axes[1].plot(sdf_line, label='SDF')
f1_1 = axes[1].plot(sigma_naive, label='Density')
f1_2 = axes[1].plot(T_naive, label='Transmittance')
f1_3 = axes[1].plot(weight_naive, label='Weight')
f1_4 = axes[1].axvline(x=max_weight_idx_naive, color='black', linestyle='--') # draw line for max weight
f1_5 = axes[1].axhline(y=0, color='black')
f1_6 = axes[1].set_yticks([0], '0')
f1_7 = axes[1].axvline(x=max_sigma_idx_naive, color='black', linestyle='--') # draw line for max sigma
axes[1].set_xticks([])
axes[1].legend()
axes[1].set_title('Naive method')

# neus method
Phi_s = (1+np.exp(-sdf_line*s))**-1
Phi_s2 = np.concatenate([Phi_s[1:], [Phi_s[-1]]])
sigma_neus = (Phi_s - Phi_s2) / Phi_s
sigma_neus[sigma_neus < 0] = 0
max_sigma_idx_neus = np.argmax(sigma_neus)
T_neus = np.cumprod(1.0 - sigma_neus)
T_neus = np.concatenate([[1.0], T_neus[:-1]])
weight_neus = T_neus * sigma_neus
max_weight_idx_neus = np.argmax(weight_neus)

f2_0 = axes[2].plot(sdf_line, label='SDF')
f2_1 = axes[2].plot(sigma_neus, label='Density')
f2_2 = axes[2].axhline(y=0, color='black')
f2_3 = axes[2].axvline(x=max_sigma_idx_neus, color='black', linestyle='--') # draw line for max sigma
f2_4 = axes[2].plot(T_neus, label='Transmittance')
f2_5 = axes[2].plot(weight_neus, label='Weight')
f2_6 = axes[2].set_yticks([0], '0')
f2_7 = axes[2].axvline(x=max_weight_idx_neus, color='black', linestyle='--') # draw line for max weight
axes[2].legend() 
axes[2].set_xticks([])
axes[2].set_title('Neus method')

def update_value():
    global sdf_line, s_density, sigma_naive, alpha_naive, T_naive, weight_naive, max_weight_idx_naive, max_sigma_idx_naive
    global Phi_s, Phi_s2, sigma_neus, max_sigma_idx_neus, T_neus, weight_neus, max_weight_idx_neus

    sdf_line = sdf[camera_view, :]
    s_density = np.exp(-sdf_line*s) / (1 + np.exp(-sdf_line*s))**2

    # naive method
    sigma_naive = s_density
    alpha_naive = 1.0 - np.exp(-sigma_naive)
    T_naive = np.cumprod(1.0 - alpha_naive)
    weight_naive = T_naive * alpha_naive
    max_weight_idx_naive = np.argmax(weight_naive)
    max_sigma_idx_naive = np.argmax(sigma_naive)

    # neus method
    Phi_s = (1+np.exp(-sdf_line*s))**-1
    Phi_s2 = np.concatenate([Phi_s[1:], [Phi_s[-1]]])
    sigma_neus = (Phi_s - Phi_s2) / Phi_s
    sigma_neus[sigma_neus < 0] = 0
    max_sigma_idx_neus = np.argmax(sigma_neus)
    T_neus = np.cumprod(1.0 - sigma_neus)
    T_neus = np.concatenate([[1.0], T_neus[:-1]])
    weight_neus = T_neus * sigma_neus
    max_weight_idx_neus = np.argmax(weight_neus)

def update_plot():
    f0_0.set_ydata(camera_view)
    f1_0[0].set_ydata(sdf_line)
    f1_1[0].set_ydata(sigma_naive)
    f1_2[0].set_ydata(T_naive)
    f1_3[0].set_ydata(weight_naive)
    f1_4.set_xdata(max_weight_idx_naive)
    f1_7.set_xdata(max_sigma_idx_naive)
    f2_0[0].set_ydata(sdf_line)
    f2_1[0].set_ydata(sigma_neus)
    f2_3.set_xdata(max_sigma_idx_neus)
    f2_4[0].set_ydata(T_neus)
    f2_5[0].set_ydata(weight_neus)
    f2_7.set_xdata(max_weight_idx_neus)
    update_value()


axb = plt.axes([0.32, 0.05, 0.33, 0.03])
sb = Slider(axb, 'training (S value)', 1, 70, valinit=1)
def update(val):
    global s
    s = sb.val
    update_plot()
sb.on_changed(update)

axb2 = plt.axes([0.32, 0.01, 0.33, 0.03])
sb2 = Slider(axb2, 'camera height', 0, 128, valinit=45)
def update2(val):
    global camera_view
    camera_view = int(sb2.val)
    update_plot()
sb2.on_changed(update2)

axb3 = plt.axes([0.8, 0.0, 0.06, 0.04])
button1 = Button(axb3, 'Show max Weight')
def show_max_weight(event):
    f2_7.set_visible(not f2_7.get_visible())
    f1_4.set_visible(not f1_4.get_visible())
    plt.draw()
button1.on_clicked(show_max_weight)

axb4 = plt.axes([0.8, 0.05, 0.06, 0.04])
button2 = Button(axb4, 'Show max Density')
def show_max_sigma(event):
    f1_7.set_visible(not f1_7.get_visible())
    f2_3.set_visible(not f2_3.get_visible())
    plt.draw()
button2.on_clicked(show_max_sigma)

plt.show()
