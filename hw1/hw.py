def hw2_6_6(num):
    if num==0:
        return 2
    elif num==1:
        return 1
    else:
        a = hw2_6_6(num-1)
        b = hw2_6_6(num-2)
        return a+b

answer = hw2_6_6(32)
print(answer)