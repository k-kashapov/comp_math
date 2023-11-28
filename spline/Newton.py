#! python3

import matplotlib.pyplot as plt

def div_diff(xs, ys, k_start, k_end):
    if k_start == k_end:
        return ys[k_start]
    return (div_diff(xs, ys, k_start + 1, k_end) - div_diff(xs, ys, k_start, k_end-1))/(xs[k_end] - xs[k_start])

def Newton(xs, diffs, x_tgt):
    res = diffs[0]
    mult = 1.

    for i in range(1, len(xs)):
        mult *= x_tgt - xs[i - 1]
        res += diffs[i] * mult

    return res

def main():
    years      = [1910., 1920., 1930., 1940., 1950., 1960., 1970., 1980., 1990., 2000.]
    population = [92228496., 106021537., 123202624., 132164569., 151325798., 179323175., 203211926., 226545805., 248709873., 281421906., 308745538.]

    diffs = [0] * (len(years) + 1)
    predict = [0] * (len(years) + 1)

    for i in range(len(years)):
        diffs[i] = div_diff(years, population, 0, i)

    years.append(2010)

    for i in range(11):
        predict[i] = Newton(years, diffs, 1910 + i * 10)
        print(f"year = {years[i]}, population = {predict[i]}")

    plt.scatter(years, predict)
    plt.grid()
    
    plt.title("Values, Newton method")
    plt.ylabel("Population")
    plt.xlabel("Year")

    # plt.show()
    plt.savefig("img/NewtonValue.png")

if __name__ == "__main__":
    main()
