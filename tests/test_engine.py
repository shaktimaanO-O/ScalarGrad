from scalargrad import MLP, Value


def test_basic_backward():
    a = Value(2.0)
    b = Value(-3.0)
    c = a * b + 10

    c.backward()

    assert c.data == 4.0
    assert a.grad == -3.0
    assert b.grad == 2.0


def test_expression_backward():
    x = Value(2.0)
    y = (x * 2 + 1) ** 2

    y.backward()

    assert y.data == 25.0
    assert x.grad == 20.0


def test_mlp_has_parameters():
    model = MLP(3, [4, 4, 1])

    assert len(model.parameters()) == 41
