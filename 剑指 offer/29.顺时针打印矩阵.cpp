//
// Created by Hughian on 2020/4/4.
//

// 旋转打印矩阵，关键就是下标的处理
class Solution {
public:
    vector<int> printMatrix(vector<vector<int> > matrix) {
        vector<int> vec;
        if (matrix.size() == 0 || matrix[0].size() == 0)
            return vec;
        // 使用 a, b, c, d 表示行列来 bound 未遍历的位置
        int a=0, b=0, c=matrix.size() - 1, d = matrix[0].size() - 1;
        while (a <= c && b <= d){
            for(int j=b;j<=d;j++){
                vec.push_back(matrix[a][j]);
            }
            for (int i=a+1; i<=c;i++){
                vec.push_back(matrix[i][d]);
            }
            if (a < c) //这里很重要，只有一行的时候，可以避免重复
                for (int j=d-1; j>=b;j--){
                    vec.push_back(matrix[c][j]);
                }
            if (b < d) // 这里很重要，只有一列的时候，可以避免重复
                for (int i=c-1; i>=a+1;i--){
                    vec.push_back(matrix[i][b]);
                }
            a += 1;
            b += 1;
            c -= 1;
            d -= 1;
        }
        return vec;
    }
};