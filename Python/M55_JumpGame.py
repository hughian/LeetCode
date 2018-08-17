class Solution:
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(nums)==0:
            return True
        target = len(nums) - 1
        for i in range(len(nums)-1, -1, -1):
            if i + nums[i] >= target: #can jump to target
                target = i
        return target == 0

def main():
    nums = [2,3,1,1,4]
    s = Solution()
    s.canJump(nums)

if __name__ == '__main__':
    main()