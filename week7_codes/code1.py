

# 闭包
def outer():
    x = 10
    def inner():
        for i in range(x):
            print(i)

    return inner  # 返回内嵌函数

fun = outer()  # 调用函数
fun()
