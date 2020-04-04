//
// Created by Hughian on 2020/4/4.
//


// 回溯
class Solution {
    int sum_digits(int x, int y) {
        int res = 0;
        while (x) {
            res += x % 10;
            x /= 10;
        }
        while (y) {
            res += y % 10;
            y /= 10;
        }
        return res;
    }

public:
    void dfs(vector <vector<int>> &visit, vector <vector<int>> &dirt, int i, int j, int rows, int cols, int k) {
        visit[i][j] = 1;
        for (auto &v: dirt) {
            int x = i + v[0];
            int y = j + v[1];
            if (x >= 0 && x < rows && y >= 0 && y < cols && sum_digits(x, y) <= k && visit[x][y] == 0) {
                dfs(visit, dirt, x, y, rows, cols, k);
            }
        }
    }

    int movingCount(int threshold, int rows, int cols) {
        if (rows <= 0 || cols <= 0 || threshold < 0)
            return 0;
        vector <vector<int>> dirt = {{0,  -1},
                                     {0,  1},
                                     {1,  0},
                                     {-1, 0}};
        vector <vector<int>> visit(rows, vector<int>(cols, 0));
        dfs(visit, dirt, 0, 0, rows, cols, threshold);
        int res = 0;
        for (auto i = 0; i < rows; i++) {
            for (auto j = 0; j < cols; j++) {
                res += visit[i][j];
            }
        }
        return res;
    }
};