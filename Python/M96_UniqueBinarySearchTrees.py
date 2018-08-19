from functools import reduce #python3 
class Solution:
    def fact(self, n):
        return reduce(lambda x,y: x*y, [1] + list(range(1,n+1)))
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        #calculate catalan number
        fcn = self.fact(n)
        return self.fact(2*n) // (fcn * fcn * (n+1))
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        r = [1,1]
        for i in range(2,n+1):
            ix = i - 1
            t = ((4*ix+2) *r[ix])//(ix+2)
            r.append(t)
        return r[n]
        
        
       
def main():
    n = 4
    s = Solution().numTrees(n)
    print(s)

if __name__=='__main__':
    main()