//
// Created by Hughian on 2020/4/4.
//

class Solution {
public:
    // 直接使用另外的数组来存，然后复制回去就可以了
    void replaceSpace(char *str,int length) {
        char tmp[length*3+1];
        memset(tmp, 0, length*3+1);
        int i, j;
        for(i=0, j=0;i<length;i++){
            if (str[i] == ' '){
                tmp[j++] = '%';
                tmp[j++] = '2';
                tmp[j++] = '0';
            }else{
                tmp[j++] = str[i];
            }
        }
        strcpy(str, tmp);
    }
};