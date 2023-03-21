# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.

circle1 = plt.Circle((0, 0), 0.2, color='r')

circle2 = plt.Circle((0.5, 0.5), 0.2, color='blue',fill= False)

circle3 = plt.Circle((1, 1), 0.2, color='g', clip_on=False)

fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot

ax.add_artist(circle1)

ax.add_artist(circle2)

ax.add_artist(circle3)

fig.savefig('plotcircles.png')
plt.Circle((0, 0), 0.3,
           color="blue")
plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
