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

def test___add__():
    a = Vector([3, 4, 2])                       # ① 准备
    b = Vector([5, 6, 3])
    c = a +b
    assert c.components == [8, 10, 5], [8, 10, 5]
    assert c.dim == 3, 3
    assert isinstance(c,Vector), c.__class__
    assert c.dim == 3, c.dim

def test___sub__():
    a = Vector([3, 4, 2])                       # ① 准备
    b = Vector([5, 6, 3])
    c = b - a
    assert c.components == [2, 2, 1], [2, 2, 1]
    assert c.dim == 3, 3
    assert isinstance(c,Vector), c.__class__
    assert c.dim == 3, c.dim

# if __name__ == "__main__":
#     test___init__()
#     test___add__()
#     test___sub__()
#     print("全过了")


