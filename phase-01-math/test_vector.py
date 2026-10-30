from Vector import Vector

# 判断两个数足够接近，1e-9 是一个很小的数
def close(a,b):
    return abs(a-b) < 1e-9


# 测试 __init__ 这个构造器
def test___init__():
    a = [1,2]
    v = Vector(a)                       # ① 准备
    a.append(3)
    assert v.components == [1, 2], v.components   # ③ 断言
    assert v.dim == 2, v.dim


# list() 不只是复制，还统一了输入类型：喂什么进去，存下来的都是 list
def test_init_accepts_any_iterable():
    assert Vector(range(5)).components == list(range(5))
    assert Vector(range(5)).dim == 5

    assert Vector((1, 2, 3)).components == list([1, 2, 3])
    assert Vector((1,2,3)).dim == 3


def test___add__():
    a = Vector([3, 4, 2])                       # ① 准备
    b = Vector([5, 6, 3])
    c = a +b
    assert c.components == [8, 10, 5], [8, 10, 5]
    assert c.dim == 3, 3
    assert isinstance(c,Vector), c.__class__


def test___sub__():
    a = Vector([3, 4, 2])                       # ① 准备
    b = Vector([5, 6, 3])
    c = b - a
    assert c.components == [2, 2, 1], [2, 2, 1]
    assert c.dim == 3, 3
    assert isinstance(c,Vector), c.__class__


# 加减都不该动到操作数。原地修改的实现会在这里红。
def test_operators_do_not_mutate_operands():
    a = Vector([3, 4, 2])
    b = Vector([5, 6, 3])

    _ = a + b      # 结果不要，只看操作数有没有被动过
    _ = a - b      # 下划线是惯例：「这个值我故意不用」

    assert a.components == [3, 4, 2], a.components
    assert a.dim == 3, a.dim
    assert b.components == [5, 6, 3], b.components
    assert b.dim == 3, b.dim


# 性质测试：不依赖具体数值，换任何向量都该成立
def test_addition_is_commutative():
    a = Vector([3, 4, 2])
    b = Vector([5, 6, 3])
    assert (a + b).components == (b + a).components
    assert (a + b).dim == (b + a).dim


# 性质测试：一条同时约束 + 和 - 之间的关系
def test_add_then_subtract_restores_original():
    a = Vector([3, 4, 2])
    b = Vector([5, 6, 3])
    e = a + b - b
    assert e.components == a.components, e.components
    assert e.dim == a.dim, e.dim
