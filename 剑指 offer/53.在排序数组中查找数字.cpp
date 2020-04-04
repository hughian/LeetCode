//
// Created by Hughian on 2020/4/4.
//

// 使用二分找一下上下界就 ok 了
// 注意 include <algorithm>
class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        //int lo = 0, hi = data.size()-1;
        auto lo = lower_bound(data.begin(), data.end(), k);
        auto hi = upper_bound(data.begin(), data.end(), k);
        return (hi - lo);
    }
};