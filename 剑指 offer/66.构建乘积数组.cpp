//
// Created by Hughian on 2020/4/4.
//

// 左右两边扫描的办法
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        if (A.size() == 0)
            return vector<int>();
        vector<int> left, B(A.size(), 0);
        left.push_back(1);
        for(auto i=0;i<A.size();i++){
            left.push_back(A[i] * left.back());
        }
        int right = 1;
        for(int i=B.size()-1; i>=0;i--){
            B[i] = left[i] * right;
            right *= A[i];
        }
        return B;
    }
};