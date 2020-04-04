//
// Created by Hughian on 2020/4/4.
//

class Solution {
public:
    // 排序解
    vector<int> _GetLeastNumbers_Solution(vector<int> input, int k) {
        if(input.size() < k){
            return vector<int>();
        }
        sort(input.begin(), input.end());
        return vector<int>(input.begin(), input.begin()+k);
    }

    // 使用大顶堆
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        if(k <= 0 || input.size() == 0 || k > input.size())
            return vector<int>();

        priority_queue<int> heap;
        for(auto x: input){
            if(heap.size() < k)
                heap.push(x);
            else{
                if (x < heap.top()){
                    heap.pop();
                    heap.push(x);
                }
            }
        }

        vector<int> res;
        while (heap.size()){
            res.push_back(heap.top());
            heap.pop();
        }
        reverse(res.begin(), res.end());
        return res;
    }
};