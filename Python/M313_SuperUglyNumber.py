class Solution(object):
    def nthSuperUglyNumber(self, n, primes):
        """
        :type n: int
        :type primes: List[int]
        :rtype: int
        """
        k = len(primes)
        dp, index = [1] + [0] * n, [0] * k
        for i in range(1, n):
            uglies = [dp[index[j]] * primes[j] for j in range(0, k)] # Build next possible ugly numbers.
            dp[i] = min(uglies)     # Find the next one.
            index = [index[j] + 1 if uglies[j] == dp[i] else index[j] for j in range(0, k)] # Advance the index(es)
        return dp[n-1]
        
def main():
    primes = [2,7,13,19]
    n = 12
    s = Solution().nthSuperUglyNumber(n,primes)
    print(s)
    
if __name__=='__main__':
    main()