import matplotlib.pyplot as plt

# Plot
def draw_sin_cos(x_values):
    y1_values = np.sin(x_values *np.pi)
    y2_values = np.cos(x_values *np.pi)
    plt.plot(x_values, y1_values, label = 'Sin')
    plt.plot(x_values, y2_values, label = 'Cos')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('values')
    plt.title('sine and cos scaled by pi')

x_values = np.arange(0, 20, 0.001) 
# same as python range but returns ndarray instance
# use numpy.linspace if you want a fixed number of points

plt.figure(figsize=(10,3), dpi=100) #640 x 450
draw_sin_cos(x_values)
plt.savefig('tutorial_sin_cos.jpg')
plt.show() # required if using in terminal or scripts. Not required in notebooks
