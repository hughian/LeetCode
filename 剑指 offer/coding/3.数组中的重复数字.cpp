//
// Created by Hughian on 2020/4/4.
//

class Solution {
public:
    // Parameters:
    //        numbers:     an array of integers
    //        length:      the length of array numbers
    //        duplication: (Output) the duplicated number in the array number
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false

    // 移动到 index 位置， 如果不让改动改动原数组的化，可以使用额外的空间
    bool duplicate(int numbers[], int length, int* duplication) {
        int t;
        int i = 0;
        while (i<length){

            if (numbers[i] == i){
                i ++;
            }else{
                t = numbers[i];
                if (numbers[t] == t){
                    *duplication = t; // duplication 用来返回重复的值
                    return true;
                }
                numbers[i] = numbers[t];
                numbers[t] = t;
            }
        }
        return false;
    }
};
