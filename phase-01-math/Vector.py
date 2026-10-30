

"""
list 的意思就是 从 list 复制出一个新 list, 例如 list([1, 2, 3])      → [1, 2, 3]
 如果写成 self.components = components，存进去的是同一个 list 的引用，不是副本。

 __add__ 这种方法之所以要前后加__ 是因为 你不需要写成a.__add__(b), 你直接写成 a + b
 python 会给你执行 这个方法

 a @ b       __matmul__
 a[3]        __getitem__
 len(a)      __len__
 print(a)    __repr__

 zip 是拉链
 例如：
    x  = [1, 2, 3]
    y = [10, 20, 30]
    zip(x, y) → (1,10)  (2,20)  (3,30)

[a+b for a , b in zip(self.components, other.components)] 与   list (a+b for a , b in zip(self.components, other.components)) 是一样的

注意:

a 和 b 是数字（components 里的元素），所以 a + b 走的是 int.__add__ / float.__add__，跟这里的 Vector.__add__毫无关系

"""



class Vector:
    def __init__(self, components):
        self.components = list(components) # list 的意思就是
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a+b for a , b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a-b for a, b in zip(self.components, other.components)])

