//
// Created by Hughian on 2020/4/4.
//

class Solution {
public:
    map<pair<int, int>, bool> dp;
    // memo + dfs
    bool foo(char *s, char *p, int i, int j) {
        if (dp.count(make_pair(i, j)))
            return dp[make_pair(i, j)];
        bool res = false;
        if (i == 0 && j == 0)
            res = true;
        else if (i == 0) {
            if (p[j - 1] == '*')
                res = foo(s, p, i, j - 2);
            else
                res = false;
        } else if (j == 0)
            res = false;
        else {
            if (p[j - 1] == '*') {
                res = foo(s, p, i, j - 2);
                if (p[j - 2] == '.' || p[j - 2] == s[i - 1]) {
                    res = res || foo(s, p, i - 1, j);
                }
            } else if (p[j - 1] == '.')
                res = foo(s, p, i - 1, j - 1);
            else
                res = s[i - 1] == p[j - 1] && foo(s, p, i - 1, j - 1);
        }
        dp[make_pair(i, j)] = res;
        return res;
    }

    // dp 方法
    bool match(char *str, char *pattern) {
        // dp
        // return foo(str, pattern, strlen(str), strlen(pattern));
        int m = strlen(str), n = strlen(pattern);
        vector <vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
        dp[m][n] = true;
        char *s = str, *p = pattern;
        for (int i = m; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                bool first = i < m && (p[j] == s[i] || p[j] == '.');
                if (j + 1 < n && p[j + 1] == '*')
                    dp[i][j] = dp[i][j + 2] || first && dp[i + 1][j];
                else
                    dp[i][j] = first && dp[i + 1][j + 1];
            }
        }
        return dp[0][0];
    }
};