
class Solution:
    def deleteAndEarn(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        h = [0] * 10001
        for k in nums:
            h[k] += k
        last_ = False
        for i in range(1,10000):
            if last_==False and h[i] >= h[i+1]:
                last_ = True
            elif last_==True:
                h[i] = 0
                last_ = False
            else:
                h[i] = 0
        print(h[:10])
        return sum(h)

def main():
    nums = [1,1,1,2,3,4,5,6]
    #nums = [4,10,10,8,1,4,10,9,7,6]
    a = Solution()
    print(a.deleteAndEarn(nums))

if __name__ == '__main__':
    main()