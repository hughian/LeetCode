//
// Created by Hughian on 2020/4/4.
//


// 逆序对，使用类似 merge sort 的思路
class Solution {
public:
    int merge(vector<int> &data, int lo, int hi) {
        int temp[hi - lo + 1];
        int index = 0, count = 0, mid = (lo + hi) / 2;
        int i = lo, j = mid + 1;
        while (i <= mid && j <= hi) {
            if (data[i] <= data[j]) {
                temp[index++] = data[i++];
            } else {
                count += mid - i + 1;
                temp[index++] = data[j++];
            }
        }
        while (i <= mid) {
            temp[index++] = data[i++];
        }
        while (j <= hi) {
            temp[index++] = data[j++];
        }
        for (int i = lo; i <= hi; i++) {
            data[i] = temp[i - lo];
        }
        return count;
    }

    int InversePairs(vector<int> data) {
        int lo = 0, hi = data.size() - 1;
        int mid = (lo + hi) / 2;

        return merge(data, 0, data.size() - 1);
    }
};