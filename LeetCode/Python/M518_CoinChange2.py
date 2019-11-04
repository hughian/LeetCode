class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        self.ans = 0
        def select(i,target): #TLE: Time Limit Exceeded
            if target < 0:
                return
            elif target==0:
                self.ans += 1
                return
            elif i < len(coins) :
                select(i,target-coins[i])
                select(i+1,target)
        select(0,amount)
        return self.ans
        
def main():
    coins = [3,5,7,8,9,10,11]
    amount = 1000
    r = Solution().change(amount,coins)
    print(r)
    
if __name__=='__main__':
    main()