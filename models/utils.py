def find_best(data, value_range, function, n):
    list_score = []
    for element in value_range:
        score = function(data, element)
        list_score.append(score)
    max_score = max(list_score)
    best_value = list_score.index(max_score) + n
    return max_score, best_value, list_score
