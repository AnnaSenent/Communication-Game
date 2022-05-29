

def dict2string(d):
    s = []

    for k, v in d.items():
        if type(v) in (int, float):
            s.append(f"--{k}={v}")
        elif type(v) is bool and v:
            s.append(f"--{k}")
        elif type(v) is str:
            assert (
                    '"' not in v
            ), f"Key {k} has string value {v} which contains forbidden quotes."
            s.append(f"--{k}={v}")
        else:
            raise Exception(f"Key {k} has value {v} of unsupported type {type(v)}.")
    return s

def grid():
    """
    Should return an iterable of the parameter strings, e.g.
    `--param1=value1 --param2`
    """
    for game_mode in ['gs', 'rf']:
        for rate in [0.001, 0.005, 0.01, 0.05, 0.1]:

            params = dict(
                train="data/train.txt",
                val="data/validation.txt",
                mode=game_mode,
                N=20,
                n_integers=2,
                n_epochs=50,
                batch_size=512,
                validation_batch_size=512,
                vocab_size=256,
                sender_hidden=256,
                receiver_hidden=512,
                lr=rate
            )

            yield dict2string(params)
