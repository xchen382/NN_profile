def find_closest_value(A, C):
    sorted_A = sorted(A)[-1::-1]
    for value in sorted_A:
        if value<=C:
            return value
    

