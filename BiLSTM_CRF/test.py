answer = []

n = 123456

while n != 0 :
    answer.append(n % 10)
    n //= 10
print(answer)
    