



s = input()
n = int(s)
s = input()
L = s.split(' ')

for t in L:
    x = int(t)
    r = []
    if x%2==0:
        x = x/2
        r += [x]
    else:
        x = 3*(x+1)/2
        r += [x]




