"""Phase 01 · 01 线性代数直觉 —— 手写一遍，再把错的直觉挑出来。

出处：phases/01-math-foundations/01-linear-algebra-intuition

    python3 01_linear_algebra.py

纯标准库，macOS 自带的 python3（3.9.6）直接能跑。
"""


# ══════════════════════════════════════════════════════════════════════
# Build It —— 先手写，不许 import numpy
# ══════════════════════════════════════════════════════════════════════

class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x ** 2 for x in self.components) ** 0.5

    def normalize(self):
        m = self.magnitude()
        return Vector([x / m for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def project_onto(self, other):
        """a 在 b 方向上的分量。残差 (a - proj) 和 b 垂直。"""
        k = self.dot(other) / other.dot(other)
        return Vector([k * x for x in other.components])

    def __repr__(self):
        return "Vector(%s)" % [round(x, 4) for x in self.components]


class Matrix:
    def __init__(self, rows):
        self.rows = [list(r) for r in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        # ⚠ 这是 @ 运算符，不是 *。见下面「错的直觉 ①」
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        return Matrix([
            [sum(self.rows[i][k] * other.rows[k][j] for k in range(self.shape[1]))
             for j in range(other.shape[1])]
            for i in range(self.shape[0])
        ])

    def __repr__(self):
        return "Matrix(%s)" % self.rows


def rank(vectors, tol=1e-9):
    """行化简数一下有几个非零行 —— 就是秩。tol 是「多小算零」的阈值。"""
    m = [list(v.components) for v in vectors]
    rows, cols = len(m), len(m[0])
    r = 0
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if abs(m[i][c]) > tol:
                pivot = i
                break
        if pivot is None:
            continue
        m[r], m[pivot] = m[pivot], m[r]
        for i in range(rows):
            if i != r and abs(m[i][c]) > tol:
                f = m[i][c] / m[r][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        r += 1
    return r


def gram_schmidt(vectors):
    """把一组无关向量拧成互相垂直、长度为 1 的基。QR 分解内部就是这个。"""
    basis = []
    for v in vectors:
        w = Vector(v.components)
        for u in basis:
            p = w.project_onto(u)
            w = Vector([a - b for a, b in zip(w.components, p.components)])
        basis.append(w.normalize())
    return basis


# ══════════════════════════════════════════════════════════════════════
print("═" * 68)
print("① 一层神经网络 = 一次矩阵乘法，没有别的")
print("═" * 68)

W = Matrix([[0.1, -0.2, 0.3],
            [0.4, 0.5, -0.1]])
x = Vector([1.0, 0.5, -0.3])
print("  输入 3 维: %s" % x)
print("  W 的形状 : %s  —— 3 维进，2 维出" % (W.shape,))
print("  输出 2 维: %s" % (W @ x))
print()
print("  「权重」就是这 6 个数。训练就是调这 6 个数。")
print("  层数再多，也只是把这件事重复做 —— 中间插非线性才让它不塌成一层。")


# ══════════════════════════════════════════════════════════════════════
print()
print("═" * 68)
print("⛔ 错的直觉 ① —— 「点积就是相似度」")
print("═" * 68)

short = Vector([1, 1])
long_same = Vector([100, 100])     # 和 short 方向完全一样，只是长 100 倍
other_dir = Vector([1, 5])         # 方向明显不同

base = Vector([1, 1])
print("  base          = %s" % base)
print()
print("  %-28s dot=%8.2f   cos=%.4f" % ("方向相同但很长 " + str(long_same),
                                         base.dot(long_same), base.cosine_similarity(long_same)))
print("  %-28s dot=%8.2f   cos=%.4f" % ("方向不同 " + str(other_dir),
                                         base.dot(other_dir), base.cosine_similarity(other_dir)))
print()
print("  ⛔ 点积把「长度」和「方向」混在一起了 —— 长向量天然分高。")
print("  ✅ 余弦相似度除掉长度，只剩方向。")
print()
print("  那为什么到处都在说「RAG 用点积检索」？因为 embedding 模型的输出")
print("  **已经归一化**（长度都是 1）。长度都一样时，点积和余弦完全等价：")
a1, b1 = base.normalize(), other_dir.normalize()
print("    归一化后  dot=%.4f   cos=%.4f   ← 一模一样" % (a1.dot(b1), a1.cosine_similarity(b1)))
print()
print("  ⚠ 所以这条不是「点积能当相似度」，是「归一化之后才能」。")
print("    自己拼向量（比如把几个 embedding 加起来）就破坏了归一化，这时")
print("    还用点积，长的那条永远排第一 —— 而且不报错，只是结果慢慢变差。")


# ══════════════════════════════════════════════════════════════════════
print()
print("═" * 68)
print("⛔ 错的直觉 ② —— 「形状就是维度」")
print("═" * 68)

v1 = Vector([1, 0, 0])
v2 = Vector([0, 1, 0])
v3 = Vector([2, 1, 0])     # v3 = 2*v1 + v2，不带来任何新方向

print("  三个 3 维向量: %s %s %s" % (v1, v2, v3))
print("  形状看着是 3×3，秩却是 %d" % rank([v1, v2, v3]))
print()
print("  ⛔ 形状是「容器有多大」，秩是「真实自由度有几个」。")
print("  这三条全躺在 xy 平面上，怎么组合都够不到 [0,0,1]。")
print()
print("  在数据里：feature_3 = 2*feature_1 + feature_2 时，加上 feature_3")
print("  给模型的信息是 **零**。更糟的是正规方程变奇异 —— 权重没有唯一解，")
print("  训练不报错，但权重会乱飘，输入动一点输出就甩很远。")


# ══════════════════════════════════════════════════════════════════════
print()
print("═" * 68)
print("⛔ 错的直觉 ③ —— 「线性相关 / 无关是个是非题」")
print("═" * 68)

almost = Vector([2, 1, 1e-10])     # 几乎就在平面上，但差了一丁点
for tol in (1e-9, 1e-12):
    print("  tol=%-7s → rank([v1, v2, almost]) = %d" % (tol, rank([v1, v2, almost], tol=tol)))

print()
print("  ⛔ 同一组向量，阈值不同答案不同。浮点世界里没有「精确相关」。")
print()
print("  真实数据永远落在中间：不是满秩，也不是真亏秩，而是 **接近亏秩**。")
print("  后果是矩阵病态 —— 输入的微小噪声被放大成输出的大幅摆动。")
print("  这就是后面 ridge 回归、SVD 截断、以及训练里各种正则化要处理的东西。")
print("  ⓘ 记住这个口子：它是 Phase 01·13「数值稳定性」的入口。")


# ══════════════════════════════════════════════════════════════════════
print()
print("═" * 68)
print("⛔ 错的直觉 ④ —— 「投影是在算什么，很高级」")
print("═" * 68)

a = Vector([3, 4])
onto = Vector([1, 0])
p = a.project_onto(onto)
residual = Vector([x - y for x, y in zip(a.components, p.components)])

print("  a = %s  投到 %s 上 → %s" % (a, onto, p))
print("  残差 = %s，和 %s 的点积 = %.1f（垂直）" % (residual, onto, residual.dot(onto)))
print()
print("  ⛔ 投影不是在「算」什么，是在 **扔掉** 什么 —— y 分量没了。")
print("  这就是最朴素的降维。")
print()
print("  同一件事换了四个名字：")
print("    线性回归  = 把观测值投到特征张成的空间上")
print("    PCA      = 投到方差最大的几个方向上")
print("    注意力    = query 投到 key 上，投得多的权重就高")
print("    RAG 检索  = 同上，只是 key 存在向量库里")


# ══════════════════════════════════════════════════════════════════════
print()
print("═" * 68)
print("② Gram-Schmidt —— 把歪的基拧正")
print("═" * 68)

basis = gram_schmidt([Vector([1, 1, 0]), Vector([1, 0, 1]), Vector([0, 1, 1])])
for i, u in enumerate(basis):
    print("  u%d = %s   |u%d| = %.6f" % (i + 1, u, i + 1, u.magnitude()))
print()
print("  两两点积（应该全是 0）：")
for i in range(len(basis)):
    for j in range(i + 1, len(basis)):
        print("    u%d · u%d = %+.1e" % (i + 1, j + 1, basis[i].dot(basis[j])))
print()
print("  ⚠ 不是精确的 0，是 1e-16 这个量级 —— 浮点误差。")
print("    这正是 ③ 说的：所有「等于零」的判断，实际都是「小于某个阈值」。")
print()
print("  这个过程就是 QR 分解的内部实现：Q 是拧正后的基，R 记录投影系数。")


