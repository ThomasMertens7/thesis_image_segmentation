def convert_to_0_1(contour):
    for i in range(len(contour)):
        for j in range(len(contour[0])):
            if contour[i][j] <= 0:
                contour[i][j] = 1
            elif contour[i][j] > 0:
                contour[i][j] = 0

    return contour