
def reverse():
    s = input()
    L = s.split(' ')[::-1]
    st = ''
    for t in L:
        st += t + ' '
    print(st.strip())

if __name__ == '__main__':
    reverse()

