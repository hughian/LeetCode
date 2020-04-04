//
// Created by Hughian on 2020/4/4.
//

// 题目一：滑动窗口最大值
// 题目的经典解法是单调队列，我用的是优化过的暴力方法
class Solution {
public:
    vector<int> maxInWindows(const vector<int> &num, unsigned int size) {
        vector<int> ans;
        if (size > num.size() || num.size() == 0 || size <= 0)
            return ans;

        int m = INT_MIN;
        for (auto i = 0; i < size; i++) {
            m = ::max(num[i], m);
        }
        ans.push_back(m);
        for (auto i = size; i < num.size(); i++) {
            if (num[i] >= ans.back()) {
                ans.push_back(num[i]);
            } else if (num[i - size] < ans.back()) {
                ans.push_back(ans.back());
            } else {
                m = INT_MIN;
                for (auto j = i - size + 1; j <= i; j++) {
                    m = ::max(num[j], m);
                }
                ans.push_back(m);
            }
        }
        return ans;
    }
};