def filter(s):
    t = ""
    for c in s:
        if c=='#':
            t = t[:-1]
        else:
            t += c
    return t
class Solution(object):
    def backspaceCompare(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: bool
        """
        s = filter(S)
        t = filter(T)
        return s==t
def main():
    S = "ab##"
    T = "c#d#"
    s = Solution().backspaceCompare(S,T)
    print(s)
    
if __name__=='__main__':
    main()