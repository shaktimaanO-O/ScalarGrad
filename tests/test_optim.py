from scalargrad import Adam, Momentum, RMSProp, SGD, Value


def _one_step(optimizer_cls, **kwargs):
    p = Value(2.0)
    loss = p * p
    loss.backward()
    optimizer = optimizer_cls([p], **kwargs)
    optimizer.step()
    return p.data


def test_sgd_updates_parameter_downhill():
    assert _one_step(SGD, lr=0.1) < 2.0


def test_momentum_updates_parameter_downhill():
    assert _one_step(Momentum, lr=0.1) < 2.0


def test_rmsprop_updates_parameter_downhill():
    assert _one_step(RMSProp, lr=0.01) < 2.0


def test_adam_updates_parameter_downhill():
    assert _one_step(Adam, lr=0.1) < 2.0
