# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num):
NUM_TO_GUESS = 0
N = 0
def guess(num):
    global N,NUM_TO_GUESS
    if NUM_TO_GUESS>=0 and NUM_TO_GUESS<=N:
        if num==NUM_TO_GUESS:return 0
        elif num>NUM_TO_GUESS:return -1
        else :return 1
    else:
        raise(ValueError)
def set(n,k):
    global N,NUM_TO_GUESS
    N = n
    NUM_TO_GUESS = k
class Solution(object):
    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        left, right = 0,n
        while left<=right:
            mid = (left + right) // 2
            r = guess(mid)
            if r==0:
                return mid
            elif r==-1:
                right=mid-1
            else:
                left=mid+1
                
def main():
    n,k=10,6
    set(n,k)
    s = Solution().guessNumber(n)
    print(s)
if __name__=='__main__':
    main()