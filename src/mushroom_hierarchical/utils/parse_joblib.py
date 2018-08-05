def parse_joblib(res, n_results=2):
    res_list = list()

    for i in range(n_results):
        res_i = [r[i] for r in res]
        res_list.append(res_i)

    return tuple(res_list)
