//
// Created by Hughian on 2020/4/4.
//


// 大小堆
class Solution {
    priority_queue<int, vector<int>, greater<int>> min_heap;
    priority_queue<int, vector<int>, less<int>> max_heap;
public:
    void Insert(int num) {
        if (min_heap.size() != max_heap.size()) {
            max_heap.push(num);
            min_heap.push(max_heap.top());
            max_heap.pop();
        } else {
            if (min_heap.size() == 0) {
                max_heap.push(num);
            } else {
                if (num > min_heap.top()) {
                    min_heap.push(num);
                    max_heap.push(min_heap.top());
                    min_heap.pop();
                } else {
                    max_heap.push(num);
                }
            }
        }
    }

    double GetMedian() {
        if (min_heap.size() == max_heap.size()) {
            return (min_heap.top() + max_heap.top()) / 2.0;
        } else {
            return max_heap.top();
        }
    }

};