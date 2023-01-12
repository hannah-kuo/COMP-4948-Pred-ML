weights = [0.5, 2.3, 2.9]
heights = [1.4, 1.9, 3.2]


def getSlopeOfLossFunction(weights, heights, intercept):
    sum = 0
    BETA = 0.64
    for i in range(0, len(weights)):
        sum += -2 * (heights[i] - intercept - BETA * weights[i])

    print("Intercept: " + str(intercept) + " Res (Slope): " + str(round(sum, 2)))  # Res = Slope of the curve


intercept = 0
getSlopeOfLossFunction(weights, heights, intercept)
