def check_first_sentence_is_second(s_1, s_2):
    d = {}
    for word in s_1.split(' '):
        if word:
            d[word] = d.get(word, 0) + 1
    fl = True
    for word in s_2.split(' '):
        if word:
            if d.get(word) is None:
                fl = False
                break
            else:
                if d[word] == 0:
                    fl = False
                    break
                else:
                    d[word] = d.get(word, 0) - 1
    return fl
