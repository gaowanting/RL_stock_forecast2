from functools import reduce
# x = [1, 2, 3, 4]
# # 循环
# y = [x**2 for i in x]
# print(y)
# # lambda
# f = lambda x :x**3
# y2 = map(f, x)  # 遍历x，用f映射
# for i in y2:
#     print(i)
# y3 = list(map(f, x))
# print(y3)
#
# x = [1,2,3]
# a, b, c = x

# 函数可以return 也可以用yield, 使用yield是生成器
# def fib(x, y):
#     while True:
#         yield x + y
#         x, y = y, x + y
#
#
# a = list(fib(1, 1))
# print(a)
#
# a = [1,2,3]
# def func():
#     pass
# # 按func返回值(T,F), 过滤列表a - 相当于将
# filter(func, a)
# reduce(func, a, initial)
#
# add = 0
# for i in range(1, 51):
#     add += i
# print(add)

'''
reduce 对集合依次做lambda的运算
'''
# print(reduce(lambda a, b: a + b, range(1, 51)))
# print(reduce(lambda a, b: a * b, range(1, 51)))

'''
合并字典
'''
x = {"a": 1, "d": 2}
y = {"c": 113, "b": 5}
# z = dict(x.items() + y.items())  python 2.x
z = dict(x, **y)  # this is method1
print(z)
c = x | y # this is method2
# c = x |= y  #将y添加到x中
print(c)
