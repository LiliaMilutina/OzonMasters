def encode_rle(x):
    count_el = []
    count_el.append(1)
    res = []
    res.append(x[0])
    el_prev = x[0]
    k = 0
    for i in range(1, len(x)):
        el_next = x[i]
        if (el_next == el_prev):
            count_el[k] += 1
        else:
            k += 1
            count_el.append(1)
            res.append(el_next)
        el_prev = el_next
    return (res, count_el)
