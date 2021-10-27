def find_word_in_circle(circle, word):
    len_word = len(word)
    len_circle = len(circle)
    if word == " " or circle == " " or len_circle == 0 or len_word == 0:
        return -1
    temp = circle[::-1]
    temp = temp[:-1]
    temp = circle[0] + temp
    str_4 = temp*2
    str_3 = circle*2
    if len_word <= len_circle:
        num_1 = str_3.find(word)
        num_2 = str_4.find(word)
        if num_1 == -1 and num_2 == -1:
            return -1
        if num_1 != -1:
            return (num_1, 1)
        if num_2 != -1:
            return (len_circle-num_2, -1)
    else:
        str_1 = circle*int(len_word/len_circle + 1)
        str_2 = temp*int(len_circle/len_circle + 1)
        num_1 = str_1.find(word)
        num_2 = str_2.find(word)
        if num_1 == -1 and num_2 == -1:
            return -1
        if num_1 != -1:
            return (num_1, 1)
        if num_2 != -1:
            return (len_circle-num_2, -1)
