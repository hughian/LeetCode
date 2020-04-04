//
// Created by Hughian on 2020/4/4.
//


// 两段有序的，可以用二分来找，这里比较的时重点 mid 和 右端点 hi 的大小
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        if(rotateArray.size()==0)
            return 0;
        int lo=0, hi=rotateArray.size() - 1, mid;
        while (lo < hi){
            mid = lo + (hi - lo) / 2;
            if (rotateArray[mid] > rotateArray[hi]){    // 大于右端点，说明最小的点在右半部分
                lo = mid + 1;
            }else if (rotateArray[mid] < rotateArray[hi]){  // 小于右端点，说明最小的点在左半部分（包括 mid)
                hi = mid;
            }else{ // 和右端点相等，无法判读，我们将右端点前移一位
                hi -= 1;
            }
        }
        return rotateArray[lo];
    }
};