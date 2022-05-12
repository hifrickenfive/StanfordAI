def factor(position1, position2):
    def is_same_row(position1, position2):
        return position1[0] == position2[0]
    
    def is_same_column(position1, position2):
        return position1[1] == position2[1]

    def is_same_column(poition1, position2):
        return abs(position2[0]-position1[0]) == abs(position2[1] - position1[1])
    
    return (not is_same_column) and (not is_same_row) and (not is_same_column)


# print(factor((0,0), (1,1)))
print(factor((0,0), (3,2)))