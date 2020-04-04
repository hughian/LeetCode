//
// Created by Hughian on 2020/4/4.
//

class Solution {
public:
    // 用空间的 DP
    int _FindGreatestSumOfSubArray(vector<int> array) {
        int m = INT_MIN;
        vector<int> dp(array.size(), 0);
        for (auto i = 0; i < array.size(); i++) {
            if (i == 0 || dp[i - 1] <= 0)
                dp[i] = array[i];
            else
                dp[i] = dp[i - 1] + array[i];
            m = ::max(m, dp[i]);
        }
        return m;
    }

    // 常数空间，只要当前数的累计和大于 0 就可以继续加
    int FindGreatestSumOfSubArray(vector<int> array) {
        int m = INT_MIN;
        int cur_sum = 0;
        for (auto x: array) {
            if (cur_sum <= 0)
                cur_sum = x;
            else
                cur_sum += x;
            m = ::max(m, cur_sum);
        }
        return m;
    }
};