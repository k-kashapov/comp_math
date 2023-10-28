#!/bin/python3

import math
import matplotlib.pyplot as plt
import numpy as np

# ========< Functions >========
func_list = ["sinsqr", "cossin", "expsincos", "ln3", "power"]

def sinsqr(x: float) -> float:
    return math.sin(x*x)

def cossin(x: float) -> float:
    return math.cos(math.sin(x))

def expsincos(x: float) -> float:
    return math.exp(math.sin(math.cos(x)))

def ln3(x: float) -> float:
    return math.log(x + 3)

def power(x: float) -> float:
    return (x + 3)**0.5

# ======< Target values >======
def sinsqr_tgt(x: float) -> float:
    return (math.cos(x*x)*2*x)

def cossin_tgt(x: float) -> float:
    return -math.sin(math.sin(x)) * math.cos(x)

def expsincos_tgt(x: float) -> float:
    return -math.sin(x) * math.cos(math.cos(x)) * expsincos(x)

def ln3_tgt(x: float) -> float:
    return 1.0 / (x + 3)

def power_tgt(x: float) -> float:
    return 1.0 / (2.0 * (x + 3)**0.5)

# ========< Formulae >=========
def formula1(func: str, x: float, h: float) -> float:
    return (eval(func)(x + h) - eval(func)(x)) / h
    
def formula2(func: str, x: float, h: float) -> float:
    return (eval(func)(x) - eval(func)(x - h)) / h

def formula3(func: str, x: float, h: float) -> float:
    return (eval(func)(x + h) - eval(func)(x - h)) / 2 / h

def formula3_1(func: str, x: float, h: float) -> float:
    return (eval(func)(x + 2 * h) - eval(func)(x - 2 * h)) / 4.0 / h

def formula3_2(func: str, x: float, h: float) -> float:
    return (eval(func)(x + 3 * h) - eval(func)(x - 3 * h)) / 6.0 / h

def formula4(func: str, x: float, h: float) -> float:
    return (4.0 * formula3(func, x, h) / 3.0) - (formula3_1(func, x, h)) / 3.0

def formula5(func: str, x: float, h: float) -> float:
    return (3.0 * formula3(func, x, h) / 2.0) - 3.0 * (formula3_1(func, x, h)) / 5.0 + formula3_2(func, x, h) / 10.0

# ==========< Misc >===========
def one_f_to_rule_them_all(form: str, func: str, x: float, h: float):
    return abs(eval(form)(func, x, h) - eval(func + "_tgt")(x))

def calc_h(n: int) -> float:
    return 1.0 / 2**(n - 1)

# ==========< main >===========
def main():
    x: float = 4.0
    
    print(f"x = {x:3.4f}. Calculating...\n")

    errors = np.empty((5, 5, 21))
    hs     = []

    n: int = 1
    while n <= 21:
        h = calc_h(n)
        print(f"n = {n}, h = {h:3.10f}")
        
        hs.append(h)
        for subj in range(len(func_list)):
            for form in range(0, 5):
                errors[subj][form][n - 1] = one_f_to_rule_them_all("formula" + str(form + 1), func_list[subj], x, h)
        n += 1

    print("\nPlotting...")
    for func in range(len(func_list)):
        print(func_list[func])
        plt.xscale('log')
        plt.yscale('log')
        plt.grid()

        plt.title(func_list[func])
        plt.ylabel("Error")
        plt.xlabel("Step")

        for i in range(5):
            plt.plot(hs, errors[func][i], '.-', ms=5.0)
        plt.savefig("img/"+func_list[func])
        plt.clf()

    print("Finished")

if __name__ == '__main__':
    main()
