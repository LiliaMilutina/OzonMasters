def get_new_dictionary(input_dict_name, output_dict_name):
    f1 = open(input_dict_name)
    f2 = open(output_dict_name, 'w')
    words_eng = []
    words_rus = []
    words = []
    for line in f1:
        sents = line.split('-')
        if len(sents) != 1:
            words_eng.append(sents[0].rstrip())
            if sents[1].lstrip().split('\n')[0]:
                words.append(sents[1].lstrip().split('\n')[0])
    i = 0
    d = dict()
    for word in words:
        if word.find(',') == -1:
            if d.get(word) is None:
                d[word] = words_eng[i]
            else:
                s = d.get(word) + ',' + words_eng[i]
                d[word] = s
        else:
            temp = word.split(',')
            for t in temp:
                if d.get(t.lstrip()) is None:
                    d[t.lstrip()] = words_eng[i]
                else:
                    s = d.get(t.lstrip()) + ',' + words_eng[i]
                    d[t.lstrip()] = s
        i += 1
    f2.write(str(len(d)) + '\n')
    for key in sorted(d.keys()):
        if (d[key].find(',') == -1):
            str_f = key + ' - ' + d[key] + '\n'
            f2.write(str_f)
        else:
            list_tr = d[key].split(',')
            str_f = key + ' - '
            s_temp = ''
            for el in sorted(list_tr):
                s_temp = s_temp + str(el) + ', '
            s_temp = s_temp[:-2] + '\n'
            str_f = str_f + s_temp
            f2.write(str_f)
    f2.close()
    return output_dict_name
