class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        primes = [2,3,5]
        dp = [1] + [0] * (n-1)
        index = [0,0,0]
        for i in range(1,n):
            uglies = [dp[index[j]] * primes[j] for j in range(3)]
            dp[i] = min(uglies)
            index = [index[j] + 1 if uglies[j]==dp[i] else index[j] for j in range(3)]
        return dp[n-1]
        
def main():
    n = 10
    s = Solution().nthSuperUglyNumber(n)
    print(s)
    
if __name__=='__main__':
    main()