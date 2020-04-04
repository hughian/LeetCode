//
// Created by Hughian on 2020/4/4.
//

class Solution {
    // 二维数组中得查找
public:
    // 暴力方法，TLE
    bool _Find(int target, vector<vector<int> > array) {
        for(size_t i=0; i<array.size();i++){
            for (size_t j=0; j < array[0].size(); j++){
                if (array[i][j] == target)
                    return true;
            }
        }
        return false;
    }
    // 使用右上角作为划分点进行二分查找
    bool helper(int a, int b, int c, int d, int target, vector<vector<int> > &arr){
        if (a > c || b > d)
            return false;
        else if (a == c && b == d){
            return arr[a][b] == target;
        }
        else{
            if (arr[a][d] == target){
                return true;
            }else if (arr[a][d] > target){
                return helper(a, b, c, d-1, target, arr);
            }else{
                return helper(a+1, b, c, d, target, arr);
            }
        }
    }
    bool Find(int target, vector<vector<int> > array){
        if (array.size() == 0 || array[0].size() == 0)
            return false;
        else
            return helper(0, 0, array.size()-1, array[0].size()-1, target, array);
    }
};