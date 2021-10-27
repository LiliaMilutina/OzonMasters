def find_max_substring_occurrence(input_string):
    n = len(input_string)
    if (n == 1):
        return 1
    len_st = 1
    while (len_st <= n):
        ss = input_string[0:len_st]
        k = int(n/len(ss))
        st = ss*k
        if (st == input_string):
            return k
        else:
            len_st += 1
