//
// Created by Hughian on 2020/4/4.
//


// 使用 mask 来区分 a,b 两组数
class Solution {
public:
    void FindNumsAppearOnce(vector<int> data, int *num1, int *num2) {
        int s = 0;
        for (auto x: data) {
            s ^= x;
        }
        // # s = a^b
        *num1 = *num2 = 0;
        int mask = s & ~(s - 1);
        for (auto x: data) {
            if (x & mask)
                *num1 ^= x;
            else
                *num2 ^= x;
        }
    }
};