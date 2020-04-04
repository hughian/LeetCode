//
// Created by Hughian on 2020/4/4.
//


// 使用 rootval 作为划分点，然后判断右子树是否都是大于根节点的
// 然后递归遍历左右两边
class Solution {
public:
    bool helper(vector<int> &vec, int lo, int hi) {
        if (hi - lo + 1 <= 2)
            return true;

        int rootval = vec[hi], i = lo, idx = -1;
        for (; i < hi; i++) {
            if (rootval < vec[i]) {
                break;
            }
        }
        for (int j = i; j < hi; j++) {
            if (rootval > vec[j]) {
                return false;
            }
        }
        return helper(vec, lo, i - 1) && helper(vec, i, hi - 1);
    }

    bool VerifySquenceOfBST(vector<int> sequence) {
        if (sequence.size() == 0)
            return false;  // 空树返回 false?
        return helper(sequence, 0, sequence.size() - 1);
    }
};