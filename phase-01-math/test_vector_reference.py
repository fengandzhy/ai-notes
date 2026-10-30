"""⛔ 别打开这个文件，直到你自己那份测试全绿。

里面是 13 条现成的测试 + 你问过的那些知识点的笔记。提前看了，
你就再也想不出「还该测什么」—— 而那一步才是写测试真正难的地方。

你自己那份跑绿之后，再回来 diff：看的是「我漏了哪些」，不是对错。

跑法: python3 test_vector_reference.py
"""

"""给自己敲的 Vector 用的红绿灯 —— 带上你问过的那些点。

跑法（在 phase-01-math 目录下）:
    python3 test_vector.py

第一次跑会直接 ImportError（Vector.py 还是空的），写出 class Vector 之后
才开始一条条变绿。⛔ 跑之前不要打开 01_linear_algebra.py，卡住了先自己想两分钟。


═══════════════════════════════════════════════════════════════════════
 要实现的东西 —— 只给「能干什么」，不给「怎么写」；⚠ 是会踩的坑
═══════════════════════════════════════════════════════════════════════

Vector(components)
    components 是个列表，任意长度（768 维才是真实场景，2 维只是画得出图）

    self.components = list(components)
        ⓘ list 是 Python 的数组类型，≈ TypeScript 的 Array。
          这里 list(...) 是在「调用」它 —— 把任何可遍历的东西复制成新 list。
          所以 Vector(range(5))、Vector((1,2,3)) 也能用。
        ⚠ 为什么不直接写 self.components = components：
          那样存进去的是外面那个 list 的「引用」。外面改一下，你的向量
          跟着变；两个 Vector 还可能共用同一份数据。
          TS 里 this.x = arr 是一模一样的坑，解法也一样是 [...arr]。

    self.dim = len(self.components)
        ⓘ dim 这个名字是作者随便起的 —— components / dot / magnitude 也都是。
          Python 没有「先声明字段」这回事，self.任何名字 = ... 当场就建出来。
          所以打错字不报错，只是默默多一个属性，等到「用」的时候才炸。
        ⚠ 但测试断言了 v.dim，所以这里必须叫 dim —— 名字一旦有人用就不自由了。
        ⚠ dim 是「存下来的」不是「每次现算的」。components 一变它就过期。
          这里安全，是因为整个类被当成不可变对象在用（见下面 __add__）。

    关于 self：
        ⓘ self 就是那个对象本身，不是副本 —— __init__ 里的 self 和外面
          拿到的 v 是同一个地址。
        ⓘ self 不是关键字，它只是第一个参数，Python 自动把对象填进去。
          v.dot(w) 和 Vector.dot(v, w) 完全等价，前者是后者的语法糖。
        ⚠ 给 self 本身赋值没用（self = ...只改了局部变量）；你能改的是 self.xxx。
        ⚠ __init__ 不要写 return —— 对象在 __init__ 之前就造好了，
          __init__ 只是进去挂属性。真写了 return 会报 TypeError。


a + b                    →  __add__(self, other)
        ⓘ 你在外面只写 a + b，从不写 a.__add__(b)。解释器执行到 `+` 时
          自己去 type(a) 上找 __add__ 并调用 —— 调用点在解释器里，不在你的代码里。
          （字节码里只有一条 BINARY_ADD，连方法名都没有。）
        ⓘ 左边的是 self，右边的是 other。加法可交换看不出来，
          以后写 __sub__ 时搞反就全错。
        ⚠ 必须 return Vector([...]) 造新对象，不能去改 self.components。
        ⚠ 没实现 → TypeError: unsupported operand type(s) for +
        ⓘ 里面那行 [a + b for a, b in zip(X, Y)] 叫「列表推导式」，
          里面没有赋值。从右往左读：for 先跑，a + b 是每一轮算什么，
          算出来的结果收成一个新 list。等价于：
              out = []
              for a, b in zip(X, Y):
                  out.append(a + b)
          ≈ TypeScript 的 arr.map(([a, b]) => a + b)。
          zip 负责配对：zip([1,2], [10,20]) → [(1,10), (2,20)]，
          for a, b in ... 再把每个二元组「拆包」成两个变量。
        ⚠ 这里的 a、b 是循环变量，是「数字」。它们跟外面 a + b
          里那两个 Vector 只是重名：同样三个字符，外层是向量加法
          （走 __add__），内层是普通数字相加。嫌绕就改成 x, y，
          测试不管变量叫什么。
        ⓘ 推导式在 Python 3 里有自己的作用域，不会污染外面的同名变量。
        ⓘ zip 是「拉链」—— 按位置把两个序列配成对，也可以超过两个：
              zip([1,2,3], [10,20,30]) → (1,10) (2,20) (3,30)
          它返回的是一次性的迭代器，不是 list。for 里遍历一次没问题，
          想反复用得先 list(zip(...)) 存下来。zip(*pairs) 能反向拆开，
          相当于转置。
        ⚠ zip 在两边长度不等时按最短的截断 —— 不报错，不警告。于是：
              Vector([1,2]) + Vector([10,20,30]) = Vector([11,22])
          2 维加 3 维在数学上非法，代码却悄悄算完了，第 3 个分量被吃掉。
          点积同理，会少算一项。这是参考实现里一个真实的缺陷：错误被吞掉，
          真正报错的地方会离犯错的地方很远，甚至根本不报错只是结果不对。
          Python 3.10+ 可以写 zip(..., strict=True)，长度不等当场 ValueError。
          本项目 venv 是 3.14，能用。
        ⓘ 里面那个 a + b 不会递归调到你自己的 __add__。规则只有一条：
              a + b  →  Python 去查 type(a) 上的 __add__
          跟这行代码写在哪个方法里一点关系都没有。这里 a、b 是 components
          里的「数字」，所以走的是 int.__add__ / float.__add__。
          （只有当分量本身就是 Vector 时才会真递归 —— 那是另一回事。）
          这叫动态派发，≈ TS 里虚方法按运行时类型找实现，只不过这里的
          「方法名」藏在 + 这个符号后面。对所有 dunder 都成立。

a.dot(b)                 →  点积，返回一个「数」，不是 Vector
        ⚠ 第 1 节最常见的手滑：返回了 Vector。点积吃两个向量，吐一个标量。

a.magnitude()            →  长度
a.normalize()            →  同方向、长度为 1 的新 Vector
a.cosine_similarity(b)   →  夹角余弦，范围 [-1, 1]
        ⚠ 它和两条向量的长度无关；点积有关。这是第 1 条错觉，测试里专门验了。

a.project_onto(b)        →  a 投影到 b 方向上的那个「分量向量」
        ⚠ 最容易写错的一个，留最后写。
          忘了除 b·b，拿 [7,0] 去投会得到 7 倍的结果 —— 那个分母的作用
          就是把 b 的长度消掉，让结果只跟 b 的「方向」有关。

repr(a)                  →  __repr__，print 出来人能看懂就行
        ⚠ 不实现的话 print(v) 打出 <__main__.Vector object at 0x...>


═══════════════════════════════════════════════════════════════════════
 顺手记住的三条
═══════════════════════════════════════════════════════════════════════

① 名字分三档，判据是「名字是你起的，还是 Python 来找你的」

      语言的钩子   __init__ __add__ __repr__ __matmul__   ⛔ 一个字母都不能改
      约定         self、_private                          能改，但永远别改
      自由         dot / magnitude / dim / components      ✅ 随便起

   __add__ 这种前后双下划线的叫 dunder（double underscore），它跟「私有」
   毫无关系 —— 恰恰相反，它存在的全部意义就是给外面用（外面写的是 `+`）。
   它也不是 override：object 身上根本没有 __add__，没东西可盖。
   准确的说法是「实现协议」，≈ TS 里实现 Symbol.iterator / toString()。

② Python 没有块级注释。只有 # 到行尾。

   批量注释掉代码 → PyCharm 里选中按 Cmd + /，加一排 #。
   ⚠ 不要拿 \"\"\"...\"\"\" 当 /* */ 用：它是字符串字面量不是注释语法，
     而且会撞上里面的 docstring，把注释提前关掉。
   \"\"\"...\"\"\" 放在函数/类/模块的第一条语句时是 docstring（存进 __doc__），
   那是正当用法 —— 比如你正在读的这一段。

③ 写完全绿之后，再去 diff 01_linear_algebra.py —— 看的是差异，不是对错。


写这个类的时候，实现顺序建议：
    __init__ → magnitude → dot → __add__ → normalize
    → cosine_similarity → project_onto → __repr__
"""

from Vector import Vector

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def vclose(v, expected, tol=1e-9):
    if len(v.components) != len(expected):
        return False
    return all(close(a, b, tol) for a, b in zip(v.components, expected))


@case("构造：记住分量，知道自己几维")
def _():
    v = Vector([3, 4])
    assert v.components == [3, 4], v.components
    assert v.dim == 2, v.dim
    assert Vector([1, 2, 3, 4, 5]).dim == 5


@case("加法：对应位置相加，且返回新对象不改原件")
def _():
    a, b = Vector([1, 2, 3]), Vector([4, 5, 6])
    assert vclose(a + b, [5, 7, 9])
    assert a.components == [1, 2, 3], "a 被改了 —— 加法不该修改原向量"


@case("点积：返回的是数，不是向量")
def _():
    r = Vector([1, 2, 3]).dot(Vector([4, 5, 6]))
    assert not isinstance(r, Vector), "点积吃两个向量，吐一个标量"
    assert close(r, 32), r


@case("点积的符号：>0 同向、=0 垂直、<0 反向")
def _():
    assert Vector([1, 0]).dot(Vector([2, 0])) > 0
    assert close(Vector([1, 0]).dot(Vector([0, 1])), 0)
    assert Vector([1, 0]).dot(Vector([-3, 0])) < 0


@case("长度：3-4-5")
def _():
    assert close(Vector([3, 4]).magnitude(), 5)
    assert close(Vector([1, 1, 1, 1]).magnitude(), 2)


@case("归一化：长度变 1，方向不变")
def _():
    u = Vector([3, 4]).normalize()
    assert close(u.magnitude(), 1), u.magnitude()
    assert vclose(u, [0.6, 0.8]), u


@case("余弦相似度：垂直 0 / 同向 1 / 反向 -1")
def _():
    assert close(Vector([1, 0]).cosine_similarity(Vector([0, 1])), 0)
    assert close(Vector([1, 0]).cosine_similarity(Vector([2, 0])), 1)
    assert close(Vector([1, 0]).cosine_similarity(Vector([-5, 0])), -1)


@case("⚠ 余弦和长度无关，点积和长度有关 —— 这是第 1 条错觉")
def _():
    a, short, long = Vector([1, 1]), Vector([1, 0]), Vector([100, 0])
    assert close(a.cosine_similarity(short), a.cosine_similarity(long)), \
        "把 b 拉长 100 倍，余弦必须一个字都不变"
    assert not close(a.dot(short), a.dot(long)), \
        "点积必须跟着变 —— 它含长度"


@case("投影：落在 b 的方向上")
def _():
    p = Vector([3, 4]).project_onto(Vector([1, 0]))
    assert vclose(p, [3, 0]), p


@case("⚠ 投影只看 b 的方向，不看 b 的长度")
def _():
    a = Vector([3, 4])
    assert vclose(a.project_onto(Vector([1, 0])),
                  a.project_onto(Vector([7, 0])).components), \
        "分母 b·b 的作用就是把 b 的长度消掉"


@case("投影的残差必须垂直于 b")
def _():
    a, b = Vector([3, 4]), Vector([2, 1])
    p = a.project_onto(b)
    residual = Vector([x - y for x, y in zip(a.components, p.components)])
    assert close(residual.dot(b), 0), residual.dot(b)


@case("任意维都要能跑 —— 768 维才是真实场景")
def _():
    a = Vector(list(range(768)))
    b = Vector([1] * 768)
    assert a.dim == 768
    assert close(a.dot(b), sum(range(768)))
    assert close(a.normalize().magnitude(), 1)


@case("repr：print 出来不报错")
def _():
    assert len(repr(Vector([1, 2, 3]))) > 0


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in CASES:
        try:
            fn()
            print("  \033[32m✓\033[0m %s" % name)
            passed += 1
        except Exception as e:
            kind = type(e).__name__
            msg = str(e) or "(没有说明)"
            print("  \033[31m✗\033[0m %s\n      %s: %s" % (name, kind, msg))
            failed += 1
    print("\n  %d 过 / %d 挂" % (passed, failed))
    if failed == 0:
        print("  全绿了。现在再去 diff 01_linear_algebra.py，看的是差异不是对错。")
