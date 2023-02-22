import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

def nevilles_method(x_points, y_points, x):

    matrix = np.zeros((3, 3))

    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]


    num_of_points = 3

    for i in range(1, num_of_points):
        for j in range(1, i + 1):
            first_multiplication = (x - x_points[i-j]) * matrix[i][j-1]
            second_multiplication = (x - x_points[i]) * matrix[i-1][j-1]
            denominator = x_points[i] - x_points[i-j]

            coefficient = (first_multiplication - second_multiplication)/denominator
            matrix[i][j] = coefficient
    
    return matrix[2][2]

#----------------------------------------------------------------------------------
def divided_difference_table(x_points, y_points):
    size: int = 4
    matrix: np.array = np.zeros((4,4))

    for index, row in enumerate(matrix):
        row[0] = y_points[index]

    for i in range(1, size):
        for j in range(1, i + 1):
            numerator =  matrix[i][j-1] - matrix[i-1][j-1]
            denominator = x_points[i] - x_points[i-j]
            operation = numerator / denominator
            matrix[i][j] = '{0:.7g}'.format(operation)


    print(matrix[1,1])
    print(matrix[2,2])
    print(matrix[3,3])
    return matrix

def get_approximate_result(matrix, x_points, value):

    reoccuring_x_span = 1
    reoccuring_px_result = 23.5492

    for i in range(1, 4):

        reoccuring_x_span *= (value - x_points[i-1])
        mult_operation = matrix[i][i] * reoccuring_x_span
        reoccuring_px_result += mult_operation

    print(reoccuring_px_result)
    return reoccuring_px_result

#----------------------------------------------------------------------------------    
def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue
            
            # get left cell entry
            left: float = matrix[i][j-1]
            # get diagonal left entry
            diagonal_left: float = matrix[i-1][j-1]

            # order of numerator is SPECIFIC.
            numerator: float = left - diagonal_left

            # denominator is current i's x_val minus the starting i's x_val....
            denominator = matrix[i][0] - matrix[i - (j-1)][0]

            # something save into matrix
            operation = numerator / denominator
            matrix[i][j] = operation
    
    return matrix


def hermite_interpolation():
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]
 
    num_of_points = len(x_points)
    matrix = np.zeros((6, 6))

    for x in range(0, 6):
        matrix[x][0] = x_points[int(x/2)]
   
    for x in range(0, 6):
        matrix[x][1] = y_points[int(x/2)]

    for i in range(1, 5, 2):
        matrix[i][2] = slopes[int(i/2)]
        i = i + 1
        matrix[i][2] = (matrix[i][1] - matrix[i-1][1])/(matrix[i][0] - matrix[i-2][0])

    matrix[5][2] = slopes[2]

    filled_matrix = apply_div_dif(matrix)
    print(filled_matrix)
#----------------------------------------------------------------------------------
def cubic_spline_interpolation(x , y):
    
    #------------------------------
    #Find Matrix A
    h = np.zeros(3)

    for i in range(0, 3):
        h[i] = x[i+1] - x[i]

    A = np.zeros((4, 4))

    A[0][0] = 1
    A[3][3] = 1
    

    for i in range (1, 3):
        A[i][i-1] = h[i-1]
        A[i][i] = 2*(h[i-1]+h[i])
        A[i][i+1] = h[i]

    print(A)
    #------------------------------

     
    #------------------------------
    #Find Vector b
    alpha = np.zeros(3)

    for i in range(1, 3):
        alpha[i] = (3/h[i])*(y[i+1]-y[i]) - (3/h[i-1])*(y[i]-y[i-1])

    print(alpha)
    #------------------------------


    #------------------------------
    #Find Vector x
    l = np.zeros(4)
    u = np.zeros(3)
    z = np.zeros(4)

    l[0] = 1

    for i in range(1, 3):
        l[i] = 2*(x[i+1] - x[i-1]) - h[i-1]*u[i-1]
        u[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]

    
    n = 3

    l[3] = 1
    c = np.zeros(4)
    b = np.zeros(3)
    d = np.zeros(3)

    for i in range(1, 4):
        c[n-i] = z[n-i] - u[n-i]*c[n-i+1]
        b[n-i] = (y[n-i+1] - y[n-i])/h[n-i] - h[n-i]*(c[n-i+1]+2*c[n-i])/3
        d[n-i] = (c[n-i+1] - c[n-i])/(3*h[n-i])


    print(c)



if __name__ == "__main__":


    #----------------------------------------------------------------------------------   
     
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    approximating_value = 3.7
    print(nevilles_method(x_points, y_points, approximating_value))
    print("---------------------------------------------------------------------------------- ")

    #----------------------------------------------------------------------------------    

    
    x_points = [7.2, 7.4, 7.5, 7.6]
    #x_points = [8.1, 8.3, 8.6, 8.7]
    y_points = [23.5492, 25.3913, 26.8224, 27.4589]
    #y_points = [16.94410, 17.56492, 18.50515, 18.82091]
  
    divided_table = divided_difference_table(x_points, y_points)
    print("---------------------------------------------------------------------------------- ")

    #----------------------------------------------------------------------------------

    approximating_x = 7.3
    final_approximation = get_approximate_result(divided_table, x_points, approximating_x)
    print("---------------------------------------------------------------------------------- ")

    #---------------------------------------------------------------------------------- 

    hermite_interpolation() #for the third row and fourth column, the value is too small to print so it is approximated to 0
    print("---------------------------------------------------------------------------------- ")

     #---------------------------------------------------------------------------------- 

    x_points = [2, 5, 8, 10]
    y_points = [3, 5, 7, 9]

    cubic_spline_interpolation(x_points , y_points)