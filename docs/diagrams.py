import matplotlib.pyplot as plt
import numpy as np
# time, precision, n_parameters
factor = 4
values = np.array([(6.04, 92, 20.30 * factor),
                   (5.62, 94, 22.97 * factor),
                   (3.42, 93, 5.37 * factor),
                   (3.01, 83, 25.6 * factor),
                   (0.52, 84, 1.74 * factor),
                   (0.46, 91, 1.26 * factor),
                   (0.6, 92, 2.03 * factor),
                   (0.9, 92, 3.58 * factor),
                   ])

labels = ['DenseNet201',
          'Xception',
          'NASNetMobile',
          'ResNet-50',
          'MobileNetV2 ($\\alpha=0.35$)',
          'SqueezeNet-v1.1',
          'MobileNetV2 ($\\alpha=0.5$)',
          'MobileNetV2 ($\\alpha=1$)',
          ]

scat1 = plt.scatter(values[:, 0], values[:, 1], s=values[:, 2])
# scat2 = ax2.scatter(values[:, 0], values[:, 1], s=values[:, 2])

plt.ylim(82, 95)
# ax2.set_ylim(74, 76)

# ax.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)

# ax.tick_params(bottom=False, top=False)
# ax2.xaxis.tick_bottom()

# d = .015
# kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
# ax.plot((-d, +d), (-d, +d), **kwargs)
# ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
# kwargs.update(transform=ax2.transAxes)
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

plt.grid(alpha=0.3, linestyle=':')
# ax2.grid(alpha=0.3, linestyle=':')

for (x, y, _), label in zip(values[:1], labels[:1]):  # DenseNet
    plt.annotate(label,  # this is the text
                 (x, y),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(10, -15),  # distance from text to points (x,y)
                 ha='right')

    # ax2.annotate(label,  # this is the text
    #              (x, y),  # this is the point to label
    #              textcoords="offset points",  # how to position the text
    #              xytext=(10, -15),  # distance from text to points (x,y)
    #              ha='right')

for (x, y, _), label in zip(values[1:2], labels[1:2]):  # Xception
    plt.annotate(label,  # this is the text
                 (x, y),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, -15),  # distance from text to points (x,y)
                 ha='center')

for (x, y, _), label in zip(values[2:3], labels[2:3]):  # NASNetMobile
    plt.annotate(label,  # this is the text
                 (x, y),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, -15),  # distance from text to points (x,y)
                 ha='center')

for (x, y, _), label in zip(values[3:4], labels[3:4]):  # ResNet-50
    plt.annotate(label,  # this is the text
                 (x, y),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(7, 0)  # distance from text to points (x,y)
                 )

for (x, y, _), label in zip(values[4:5], labels[4:5]):  # MobileNetV2 ($alpha=0.35$)
    plt.annotate(label,  # this is the text
                 (x, y),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(5, 0)  # distance from text to points (x,y)
                 )

for (x, y, _), label in zip(values[5:6], labels[5:6]):  # SqueezeNet-v1.1
    plt.annotate(label,  # this is the text
                 (x, y),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, -12),  # distance from text to points (x,y)
                 # ha='center'
                 )

for (x, y, _), label in zip(values[6:7], labels[6:7]):  # MobileNetV2 ($\\alpha=0.5$)
    plt.annotate(label,  # this is the text
                 (x, y),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, -12),  # distance from text to points (x,y)
                 )

for (x, y, _), label in zip(values[7:], labels[7:]):  # MobileNetV2 ($\\alpha=1$)
    plt.annotate(label,  # this is the text
                 (x, y),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(5, 0),  # distance from text to points (x,y)
                 )


# interp_values = np.array([#(0.46, 88),
#                           (0.6, 90),
#                           (0.9, 92),
#                           (5.62, 93)])

# from scipy import interpolate
#
# tck, u = interpolate.splrep(interp_values[:, 0],
#                             interp_values[:, 1])
# unew = np.arange(0, 1.01, 0.01)
# out = interpolate.splev(unew, tck)
# ax.plot(out[0], out[1], color='orange')

# from scipy.interpolate import make_interp_spline
#
# # 300 represents number of points to make between T.min and T.max
# xnew = np.linspace(0, 6, 300)
#
# spl = make_interp_spline(interp_values[:, 0], interp_values[:, 1], k=2)
# power_smooth = spl(xnew)
#
# ax.plot(np.linspace(0, 6, 300), power_smooth[:300])

plt.title('Pareto Front of Time vs Accuracy', fontsize=14)
plt.xlabel('Time (s)', fontsize=13)
plt.ylabel('Accuracy (%)', fontsize=13)
# f.text(0.06, 0.5, 'Precisión (%)', fontsize=12, ha='center', va='center',
#          rotation='vertical')

handles, labels = scat1.legend_elements(prop="sizes", alpha=0.6, func=lambda
    x: x/factor, num=30, fmt="{x:.0f} M")

handles = [handles[i] for i in [0, 2, 9, 29]]
labels = [labels[i] for i in [0, 2, 9, 29]]

plt.legend(handles, labels, loc='lower right', title='Parameters')

plt.savefig('tiempo_vs_precision.png', dpi=200)
plt.show()