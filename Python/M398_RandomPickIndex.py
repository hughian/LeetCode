class Solution:
    
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.L = nums
        

    def pick(self, target):
        """
        :type target: int
        :rtype: int
        """
        s = [i for i in range(len(self.L)) if self.L[i] == target]
        return random.choice(s)

def main():
    nums = [1,2,3,3,3]
    target = 3
    s = Solution(nums)
    r = s.pick(target)
    print(r)

if __name__=='__main__':
    main()
