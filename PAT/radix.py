
def c2int(c):
    if c>='0' and c<='9':
        return ord(c)-ord('0')
    else:
        return ord(c)-ord('a')+10;

def str2num(s,base):
    sum = 0
    for i in range(len(s)):
        sum = sum * base + c2int(s[i])
    return sum

def getbiggest(s):
    m = -1
    for c in list(s):
        t = c2int(c)
        if m < t:
            t = m
    return m
	
def trans():
    st = input()
    L = st.split(' ')
    n1 = L[0]
    n2 = L[1]
    tag = int(L[2])
    radix = int(L[3])
    if tag == 2 :
        n1,n2 = n2,n1
    d1 = str2num(n1,radix)
    low = getbiggest(n2)+1
    high = max(d1+1,low)
    minR = fix = high + 1;
    while low<=high:
        rmid = int((low + high) / 2)
        d2 = str2num(n2,rmid)
        if d1 == d2:
            minR = rmid
            high = rmid -1;
        else:
            if d2 >= d1 or d2 <= 0:
                high = rmid -1
            else:
                low = rmid + 1
    if minR != fix:
        print(minR)
    else:
        print('Impossible')
    return None

if __name__ == '__main__':
    trans()

