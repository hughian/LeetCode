//
// Created by Hughian on 2020/4/4.
//

// 大小王是 0，大小王的个数能够填充其他牌中的 gap 就可以组成顺子

class Solution {
public:
    bool IsContinuous( vector<int> numbers ) {
        if (numbers.size() == 0)
            return false;
        sort(numbers.begin(), numbers.end()); // 排序
        int num_zeros = 0, gap = 0;
        for(int i=0;i<numbers.size();i++){
            if (numbers[i] == 0){
                num_zeros ++;  //大小王的个数
            }else{
                if(i>0){
                    if(numbers[i-1] == numbers[i]) // 有相等的，不可能是顺子
                        return false;
                    if(numbers[i-1] != 0 && numbers[i] - numbers[i-1] > 1)
                        gap += numbers[i] - numbers[i-1] - 1;
                }
            }
        }
        return num_zeros >= gap;
    }
};