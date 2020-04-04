//
// Created by Hughian on 2020/4/4.
//


// 用个大顶堆来处理
class Solution {
public:
    int GetUglyNumber_Solution(int n) {
        if (n < 1)
            return 0;
        priority_queue<long, vector<long>, greater<>> pq;

        pq.push(1);
        for (int i=0;i<n-1;i++){
            long x = pq.top();
            // 相同值直接弹出
            while (pq.size()>0 && pq.top() == x) pq.pop();
            pq.push(x * 2);
            pq.push(x * 3);
            pq.push(x * 5);
        }
        return pq.top();
    }
};