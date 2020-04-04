//
// Created by Hughian on 2020/4/4.
//

// 升序数组，使用双指针
class Solution {
public:
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        int lo = 0, hi = array.size() - 1;
        int s = INT_MAX;
        vector<int> res(2, 0);
        while(lo < hi){
            if(array[lo]+array[hi] < sum){
                lo++;
            }else if(array[lo]+array[hi] > sum){
                hi--;
            }else{
                // 多对和找乘积最小的
                if(array[lo] * array[hi] < s){
                    s = array[lo] * array[hi];
                    res[0] = array[lo];
                    res[1] = array[hi];
                }
                lo++;
                hi--;
            }
        }
        if (s<INT_MAX)
            return res;
        return vector<int>();
    }
};


// 题目二：和为 s 的连续正数序列
// 从 i ~ N // 2 + 1 ，开始枚举就 ok
class Solution {
public:
    vector<vector<int> > FindContinuousSequence(int sum) {
        if (sum <= 0)
            return vector<vector<int> >();
        vector<vector<int> > res;
        for(auto i=1;i<= sum/2+1;i++){
            vector<int> tmp;
            int s = 0;

            for (auto j=i;j<=sum/2+1 && s< sum; j++){
                tmp.emplace_back(j);
                s += j;
            }
            // cout<<s<<" "<<endl;
            if (s == sum && tmp.size() > 1)
                res.emplace_back(tmp);
        }
        return res;
    }
};