//
// Created by Hughian on 2020/4/4.
//


// 奇数放前面，先用额外的空间存一下偶数
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        vector<int> vec;
        for(auto x: array){
            if (x%2==0)
                vec.push_back(x);
        }
        int i=0, j=0;
        while (j < array.size()){
            if (array[j] & 1){
                array[i++] = array[j];
            }
            j++;
        }
        int t = i;
        while (i < array.size()){
            array[i] = vec[i-t];
            i++;
        }
    }
};

// 交换的解法会破坏原有元素的相对顺序
/*
class Solution{
public:
    void reOrderArray(vector<int> &array){
        if (array.size() == 0)
            return;
        int lo = 0, hi = array.size() - 1;
        while (lo < hi){
            while (lo < hi && array[lo] & 1)
                lo ++;
            while (lo < hi && array[hi] & 1 == 0)
                hi --;
            if (lo < hi){
                int t = array[lo];
                array[lo] = array[hi];
                array[hi] = t;
            }
        }
    }
};
 */