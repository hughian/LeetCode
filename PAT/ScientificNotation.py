
import re

s = input()
m = re.match(r'([+-])(\d{1}.\d+)(E)([+-]\d+)',s)
print(m.groups())
sign = m.group(1)
e = m.group(2)
x = m.group(4)
res = 0
if sign == '+':
    res = 1
else:
    res = -1
res =res * float(e) * pow(10,int(x))
print(res)


