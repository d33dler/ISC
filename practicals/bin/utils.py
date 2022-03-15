def min_dict(prms: list[dict], comp: str) -> dict:
    """ """
    out: dict = prms[0]
    print(prms)
    for obj in prms:
        if out[comp] > obj[comp]:
            out = obj
    return out
