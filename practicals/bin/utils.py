def min_dict(prms: list[dict], comp: str) -> dict:
    """ """
    out: dict = prms[0]
    for obj in prms:
        if out[comp] > obj[comp]:
            out = obj
    return out
