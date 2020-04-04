//
// Created by Hughian on 2020/4/4.
//


typedef vector <vector<int>> vvi;
// 回溯
class Solution {
    vvi dirt = {{-1, 0},
                {1,  0},
                {0,  -1},
                {0,  1}};
    int m, n;
    int len;
public:
    bool dfs(vvi &visit, char *matrix, int i, int j, char *str, int k) {
        //cout<<i<<" "<<j<<" "<<k<<" "<<matrix[i*n+j]<<" "<<str[k]<<endl;
        if (i < 0 || i >= m || j < 0 || j >= n || visit[i][j] == 1 || matrix[i * n + j] != str[k])
            return false;
        if (k == len - 1)
            return true;

        visit[i][j] = 1;
        for (auto &v : dirt) {
            int x = i + v[0], y = j + v[1];
            if (dfs(visit, matrix, x, y, str, k + 1))
                return true;
        }
        visit[i][j] = 0;
        return false;
    }

    bool hasPath(char *matrix, int rows, int cols, char *str) {
        m = rows;
        n = cols;
        len = strlen(str);
        vvi visit(rows, vector<int>(cols, 0));
        // 枚举每一个位置作为起始位置
        for (auto i = 0; i < rows; i++) {
            for (auto j = 0; j < cols; j++) {
                if (dfs(visit, matrix, i, j, str, 0))
                    return true;
            }
        }
        return false;
    }
};