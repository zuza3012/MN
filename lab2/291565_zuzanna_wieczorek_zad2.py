import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: 3 * x * x * x - 4 * x * x - 1


def set_range():
    while True:
        val1 = input("Type a:")
        val2 = input("Type b:")
        try:
            v1 = float(val1)
            v2 = float(val2)
            f1 = f(v1)
            f2 = f(v2)

            if f1 * f2 < 0:
                if v1 > v2:
                    b = v1
                    a = v2
                    break
                if v2 > v1:
                    a = v1
                    b = v2
                    break
                else:
                    print("Two the same values, try again.")
            else:
                print("Note that f(a)*f(b) must be < 0!")
        except ValueError:
            print("Please, enter a number!")

    return a, b


def find_root(a, b, precision):
    counter = 0
    d = []
    while True:
        x0 = a + (b - a) / 2
        d.append((x0, a, b, f(a), f(b), f(x0)))
        print(counter)
        if abs(f(x0)) < precision:
            root = x0
            break
        elif f(a) * f(x0) > 0:
            tmp = x0
            a = tmp
        elif f(b) * f(x0) > 0:
            tmp = x0
            b = tmp

    df = pd.DataFrame(d, columns=('x0', 'a', 'b', 'f(a)', 'f(b)', 'f(x0)'))
    return root, df


print("Program tries to find real roots of a function f(x) = 3x^3-4x^2-1 at given range <a,b>")
a, b = set_range()
root, df = find_root(a, b, 0.0000001)
print(df)
print("Root: ", root)

x1 = np.linspace(a, b, 100)
y1 = f(x1)
x2 = df[['x0']].values
y2 = df[['f(x0)']].values

plt.plot(x1, y1, label="Function")
plt.plot(x2, y2, 'o', color='red', label="Indirect results")
plt.plot(df[['x0']].iloc[-1], df[['f(x0)']].iloc[-1], 'o', color='black', label="Root")

plt.xlim(a, b)
plt.title('Bisection method: f(x) = 3x^3-4x^2-1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
