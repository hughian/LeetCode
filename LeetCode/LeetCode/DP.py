from typing import List
import collections
import bisect
import functools


################################################################################################################
# 字符串有关的DP

class Solution_10:
    # String, DP, 正则匹配
    def isMatch(self, s: str, p: str) -> bool:
        # 暴力解
        def foo(i, j):
            if i == j == 0:
                return True
            elif i == 0:
                # 处理 din*, .*, din*b* 这种特殊情况
                if p[j - 1] == '*':
                    return foo(i, j - 2)
                else:
                    return False
            elif j == 0:
                return False
            else:
                if p[j - 1] == '*':
                    r = foo(i, j - 2)
                    if p[j - 2] in {'.', s[i - 1]}:
                        r = r or foo(i - 1, j)
                    return r
                elif p[j - 1] == '.':
                    return foo(i - 1, j - 1)
                else:
                    return s[i - 1] == p[j - 1] and foo(i - 1, j - 1)

        return foo(len(s), len(p))

    def _isMatch(self, s: str, p: str) -> bool:
        # 另一种暴力写法
        def match(t, p):
            if len(p) == 0:
                return not len(t)
            r = (len(t) and (p[0] == t[0] or p[0] == '.'))
            if len(p) >= 2 and p[1] == '*':
                return match(t, p[2:]) or (r and match(t[1:], p))
            else:
                return r and match(t[1:], p[1:])

        return match(s, p)

    def isMatch(self, s: str, p: str) -> bool:
        # 把第一种暴力解转换为 带 memo 的递归搜索
        memo = {}

        def foo(i, j):
            nonlocal memo
            if (i, j) in memo:
                return memo[(i, j)]

            if i == j == 0:
                res = True
            elif i == 0:
                res = foo(i, j - 2) if p[j - 1] == '*' else False
            elif j == 0:
                res = False
            else:
                if p[j - 1] == '*':
                    res = foo(i, j - 2)
                    if p[j - 2] in {'.', s[i - 1]}:
                        res = res or foo(i - 1, j)
                elif p[j - 1] == '.':
                    res = foo(i - 1, j - 1)
                else:
                    res = s[i - 1] == p[j - 1] and foo(i - 1, j - 1)
            memo[(i, j)] = res
            return res

        return foo(len(s), len(p))

    def isMatch(self, s: str, p: str) -> bool:
        # 将带 memo 的递归转化为 DP
        m, n = len(s), len(p)
        dp = [[False for _ in range(n + 1)] for _ in range(m + 1)]
        dp[0][0] = True
        # 处理 din*, .*, din*b* 这种特殊情况
        for i in range(1, n + 1):
            if p[i - 1] == '*':
                dp[0][i] = dp[0][i - 2]
        for i in range(1, m + 1):
            cs = s[i - 1]
            for j in range(1, n + 1):
                cp = p[j - 1]
                if cp == '*':
                    dp[i][j] = dp[i][j - 2]
                    if p[j - 2] in {cs, '.'}:
                        dp[i][j] = dp[i][j] or dp[i - 1][j]
                elif cp == '.':
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = (cs == cp) and dp[i - 1][j - 1]
        for x in dp:
            print(list(map(int, x)))
        return dp[m][n]

    def isMatch(self, s: str, p: str) -> bool:
        # 空间优化
        m, n = len(s), len(p)
        dp = [False for _ in range(n + 1)]
        dp[0] = True
        for i in range(1, n + 1):
            if p[i - 1] == '*':
                dp[i] = dp[i - 2]

        for i in range(1, m + 1):
            cs = s[i - 1]
            new_dp = [False for _ in range(n + 1)]
            for j in range(1, n + 1):
                cp = p[j - 1]

                if cp == '*':
                    new_dp[j] = new_dp[j - 2]
                    if p[j - 2] in {cs, '.'}:
                        new_dp[j] = new_dp[j] or dp[j]
                elif cp == '.':
                    new_dp[j] = dp[j - 1]
                else:
                    new_dp[j] = (cs == cp) and dp[j - 1]
            # print(list(map(int, new_dp)))
            dp[:] = new_dp
        # for x in dp:
        #     print(list(map(int, x)))
        return dp[n]

    def isMatch(self, s, p):
        # 反向DP
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]

        dp[-1][-1] = True
        for i in range(m, -1, -1):
            for j in range(n - 1, -1, -1):
                first_match = i < m and p[j] in {s[i], '.'}
                if j + 1 < n and p[j + 1] == '*':
                    dp[i][j] = dp[i][j + 2] or first_match and dp[i + 1][j]
                else:
                    dp[i][j] = first_match and dp[i + 1][j + 1]

        return dp[0][0]

    @staticmethod
    def debug():
        s = Solution_10()
        assert s.isMatch("aa", "din*") is True
        assert s.isMatch("aa", "din") is False
        assert s.isMatch("ab", ".*") is True
        assert s.isMatch("aab", "c*din*b") is True
        assert s.isMatch("mississippi", "mis*is*p*.") is False


class Solution_44:
    # String, DP, 正则匹配
    def isMatch(self, s: str, p: str) -> bool:
        # 暴力搜索
        def foo(i, j):
            if i == j == 0:
                return True
            elif j == 0:
                return False
            elif i == 0:
                return all(p[t] == '*' for t in range(j))
            else:
                if p[j - 1] == '*':
                    return foo(i - 1, j) or foo(i, j - 1)
                elif p[j - 1] == '?':
                    return foo(i - 1, j - 1)
                else:
                    return s[i - 1] == p[j - 1] and foo(i - 1, j - 1)

        return foo(len(s), len(p))

    def isMatch(self, s: str, p: str) -> bool:
        # 带memo 的搜索
        memo = {}

        def foo(i, j):
            nonlocal memo
            if (i, j) in memo:
                return memo[(i, j)]
            if i == j == 0:
                res = True
            elif j == 0:
                res = False
            elif i == 0:
                res = all(p[t] == '*' for t in range(j))
            else:
                if p[j - 1] == '*':
                    res = foo(i - 1, j) or foo(i, j - 1)
                elif p[j - 1] == '?':
                    res = foo(i - 1, j - 1)
                else:
                    res = s[i - 1] == p[j - 1] and foo(i - 1, j - 1)
            memo[(i, j)] = res
            return res

        return foo(len(s), len(p))

    def isMatch(self, s: str, p: str) -> bool:
        # DP
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for j in range(n + 1):
            dp[0][j] = all(p[t] == '*' for t in range(j))

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
                elif p[j - 1] == '?':
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = s[i - 1] == p[j - 1] and dp[i - 1][j - 1]
        for ls in dp:
            print(list(map(lambda x: int(x), ls)))
        print('#')
        return dp[m][n]

    def isMatch(self, s: str, p: str) -> bool:
        # 空间优化，不太好优化，使用两个数组滚动并没有节约多少空间的样子
        m, n = len(s), len(p)
        dp = [False] * (n + 1)
        dp[0] = True
        for j in range(n + 1):
            dp[j] = all(p[t] == '*' for t in range(j))
        # print(list(map(lambda x: int(x), dp)))
        for i in range(1, m + 1):
            new_dp = [False] * (n + 1)
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    new_dp[j] = dp[j] or new_dp[j - 1]
                elif p[j - 1] == '?':
                    new_dp[j] = dp[j - 1]
                else:
                    new_dp[j] = s[i - 1] == p[j - 1] and dp[j - 1]
            dp[:] = new_dp
            # print(list(map(lambda x: int(x), dp)))
        return dp[n]


class Solution_32:

    def longestValidParentheses(self, s: str) -> int:
        # 括号匹配问题，使用 Stack
        # 然后使用dp数组来记录状态
        stack = []
        dp = [0] * len(s)
        m = 0
        for i, c in enumerate(s):
            if c == ')':
                if stack:
                    # 有一对括号 match, 对应的 '(' 的 index 是 j
                    j = stack.pop()
                    # 那么到 i 最长的有效匹配的串为 从 j 到当前 i 的长度 i - j + 1，加上 j 之前已经匹配过的值 dp[j - 1]
                    dp[i] = i - j + 1 + (dp[j - 1] if j > 0 else 0)
                    m = max(m, dp[i])
            else:
                stack.append(i)
        return m

    def longestValidParentheses(self, s: str) -> int:
        # 使用 left, right 来分别记录 '(' 和 ')' 出现的次数，我们不需要再使用Stack, 正反向两次扫描就可以了
        # 扫描过程中，如果 left == right, 即出现 '(' 和 ')' 的次数相等，此时得到了一个括号匹配的串，记录一下长度
        # 另外，正向扫描中，如果 right 的值大于 left, 即已经多出了一个 ')'，此时左边部分的已经是不匹配的，
        #      我们将 left, right 重新置零。从当前位置继续向后扫描
        # 同样地，反向扫描中，如果 left 的值大于right, 我们同样将 left, right 重新置零。从当前位置继续向前扫描
        m = 0
        # 正向扫描
        left = right = 0
        for i, c in enumerate(s):
            if c == '(':
                left += 1
            else:
                right += 1

            if left == right:
                m = max(m, left + right)
            elif right > left:
                left = right = 0
        # 反向扫描
        left = right = 0
        for i in range(len(s) - 1, -1, -1):
            if s[i] == '(':
                left += 1
            else:
                right += 1
            if left == right:
                m = max(m, left + right)
            elif left > right:
                left = right = 0
        return m


class Solution_72:
    # String, DP
    def minDistance(self, word1: str, word2: str) -> int:
        # 暴力搜索
        # i == j == 0, 两个word长度都是0，需要0步
        # i == 0 and j > 0, 需要插入j个字符都，共j步
        # i > 0 and j == 0, 需要把i个字符都删除，共i步
        # i > 0 and j > 0, 比较一下最后一个字符是否相等，不相等要加一步修改（f = 1）
        #                  最终需要的步数是下列三项中的最小值：
        #                      删除 word1 最后一个字符： 1 + foo(i-1, j)
        #                      插入 word2 最后一个字符： 1 + foo(i, j-1)
        #                      修改最后一个字符使相等：   f + foo(i-1, j-1)

        def foo(i, j):
            if i == j == 0:
                return 0
            elif i == 0:
                return j
            elif j == 0:
                return i
            else:
                f = 1 if word1[i - 1] != word2[j - 1] else 0
                return min(foo(i, j - 1) + 1, foo(i - 1, j) + 1, foo(i - 1, j - 1) + f)

        return foo(len(word1), len(word2))

    def minDistance(self, word1: str, word2: str) -> int:
        # 转换为带memo的搜索
        memo = {}

        def foo(i, j):
            nonlocal memo
            if (i, j) in memo:
                return memo[(i, j)]
            if i == j == 0:
                memo[(i, j)] = 0
            elif i == 0:
                memo[(i, j)] = j
            elif j == 0:
                memo[(i, j)] = i
            else:
                f = 1 if word1[i - 1] != word2[j - 1] else 0
                memo[(i, j)] = min(foo(i, j - 1) + 1, foo(i - 1, j) + 1, foo(i - 1, j - 1) + f)
            return memo[(i, j)]

        return foo(len(word1), len(word2))

    def minDistance(self, word1: str, word2: str) -> int:
        # 其实暴力就已经把状态和状态转移式子定义的很清楚了，写出 DP 也很简单
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                f = 1 if word1[i - 1] != word2[j - 1] else 0
                dp[i][j] = min(dp[i][j - 1] + 1, dp[i - 1][j] + 1, dp[i - 1][j - 1] + f)
        return dp[m][n]


class Solution_91:
    def numDecodings(self, s: str) -> int:
        # 暴力解法
        def foo(i):
            if i == 0:
                return 1
            ans = foo(i - 1) if 1 <= int(s[i - 1]) <= 9 else 0
            if i >= 2 and 10 <= int(s[i - 2:i]) <= 26:
                ans += foo(i - 2)
            return ans

        return foo(len(s))

    def numDecodings(self, s: str) -> int:
        # 转换为 DP
        dp = [0] * (len(s) + 1)
        dp[0] = 1
        # 要么当前位数字在 1~9 之间，可以组成字母映射
        # 要么当前两位（加前一位）在10~26之间，可以组成字母映射
        for i in range(1, len(s) + 1):
            if 1 <= int(s[i - 1]) <= 9:
                dp[i] = dp[i - 1]

            if i >= 2:
                if 10 <= int(s[i - 2:i]) <= 26:  # i-2,i-1
                    dp[i] += dp[i - 2]

        print(dp)
        return dp[len(s)]


class Solution_97:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        # 暴力搜索，TLE
        if len(s1) + len(s2) != len(s3):
            return False

        def foo(i, j, k):
            if i == len(s1) and j == len(s2) and k == len(s3):
                return True
            elif i == len(s1):
                return s2[j:] == s3[k:]
            elif j == len(s2):
                return s1[i:] == s3[k:]

            else:
                if s1[i] == s2[j] and s3[k] == s1[i]:
                    return foo(i + 1, j, k + 1) or foo(i, j + 1, k + 1)
                elif s1[i] == s3[k]:
                    return foo(i + 1, j, k + 1)
                elif s2[j] == s3[k]:
                    return foo(i, j + 1, k + 1)
                else:
                    return False

        return foo(0, 0, 0)

    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        # 将暴力搜搜索转换为带memo的搜索， AC
        if len(s1) + len(s2) != len(s3):
            return False
        memo = {}

        def foo(i, j, k):
            nonlocal memo
            if (i, j, k) in memo:
                return memo[(i, j, k)]
            if i == len(s1) and j == len(s2) and k == len(s3):
                memo[(i, j, k)] = True
            elif i == len(s1):
                memo[(i, j, k)] = s2[j:] == s3[k:]
            elif j == len(s2):
                memo[(i, j, k)] = s1[i:] == s3[k:]

            else:
                if s1[i] == s2[j] and s3[k] == s1[i]:
                    memo[(i, j, k)] = foo(i + 1, j, k + 1) or foo(i, j + 1, k + 1)
                elif s1[i] == s3[k]:
                    memo[(i, j, k)] = foo(i + 1, j, k + 1)
                elif s2[j] == s3[k]:
                    memo[(i, j, k)] = foo(i, j + 1, k + 1)
                else:
                    memo[(i, j, k)] = False
            return memo[(i, j, k)]

        return foo(0, 0, 0)

    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        # 约简掉 k 因为 k == i + j
        if len(s1) + len(s2) != len(s3):
            return False
        memo = {}

        def foo(i, j):
            nonlocal memo
            if (i, j) in memo:
                return memo[(i, j)]
            if i == len(s1) and j == len(s2):
                memo[(i, j)] = True
            elif i == len(s1):
                memo[(i, j)] = s2[j:] == s3[i + j:]
            elif j == len(s2):
                memo[(i, j)] = s1[i:] == s3[i + j:]

            else:
                if s1[i] == s2[j] and s3[i + j] == s1[i]:
                    memo[(i, j)] = foo(i + 1, j) or foo(i, j + 1)
                elif s1[i] == s3[i + j]:
                    memo[(i, j)] = foo(i + 1, j)
                elif s2[j] == s3[i + j]:
                    memo[(i, j)] = foo(i, j + 1)
                else:
                    memo[(i, j)] = False
            return memo[(i, j)]

        return foo(0, 0)

    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        # 将带memo的搜索转化为 从后向前 DP
        if len(s1) + len(s2) != len(s3):
            return False
        dp = [[False] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        dp[len(s1)][len(s2)] = True

        for i in range(len(s1)):
            dp[i][len(s2)] = s1[i:] == s3[i + len(s2):]
        for j in range(len(s2)):
            dp[len(s1)][j] = s2[j:] == s3[len(s1) + j:]

        for i in range(len(s1) - 1, -1, -1):
            for j in range(len(s2) - 1, -1, -1):
                if s1[i] == s2[j] and s3[i + j] == s1[i]:
                    dp[i][j] = dp[i + 1][j] or dp[i][j + 1]
                elif s1[i] == s3[i + j]:
                    dp[i][j] = dp[i + 1][j]
                elif s2[j] == s3[i + j]:
                    dp[i][j] = dp[i][j + 1]
                else:
                    dp[i][j] = False
        # debug()
        for ls in dp:
            print(list(map(int, ls)))

        return dp[0][0]

    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        # 优化空间为一维
        # dp[i][j] 只与 dp[i+1][j] 或者 dp[i][j+1] 有关
        if len(s1) + len(s2) != len(s3):
            return False
        dp = [False] * (len(s2) + 1)
        dp[len(s2)] = True

        for i in range(len(s1)):
            dp[len(s2)] = s1[i:] == s3[i + len(s2):]
        for j in range(len(s2)):
            dp[j] = s2[j:] == s3[len(s1) + j:]

        for i in range(len(s1) - 1, -1, -1):
            for j in range(len(s2) - 1, -1, -1):
                if s1[i] == s2[j] and s3[i + j] == s1[i]:
                    dp[j] = dp[j] or dp[j + 1]
                elif s1[i] == s3[i + j]:
                    dp[j] = dp[j]
                elif s2[j] == s3[i + j]:
                    dp[j] = dp[j + 1]
                else:
                    dp[j] = False

        return dp[0]

    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        # 前向方法 dp
        if len(s1) + len(s2) != len(s3):
            return False
        dp = [False] * (len(s2) + 1)
        for i in range(len(s1) + 1):
            for j in range(len(s2) + 1):
                if i == j == 0:
                    dp[j] = True
                elif i == 0:
                    dp[j] = dp[j - 1] and s2[j - 1] == s3[i + j - 1]
                elif j == 0:
                    dp[j] = dp[j] and s1[i - 1] == s3[i + j - 1]
                else:
                    dp[j] = (dp[j] and s1[i - 1] == s3[i + j - 1]) or (dp[j - 1] and s2[j - 1] == s3[i + j - 1])
        print(list(map(lambda x: int(x), dp)))
        return dp[len(s2)]


class Solution_115:
    # String DP
    def numDistinct(self, s: str, t: str) -> int:
        # 带memo的递归搜索
        memo = {}

        def foo(i, j):
            nonlocal memo

            if (i, j) in memo:
                return memo[(i, j)]
            if j == len(t):
                memo[(i, j)] = 1
                return 1
            if i == len(s):
                memo[(i, j)] = 0
                return 0
            a = 0
            if s[i] == t[j]:
                a += foo(i + 1, j + 1)
            a += foo(i + 1, j)
            memo[(i, j)] = a
            return a

        return foo(0, 0)

    def numDistinct(self, s: str, t: str) -> int:
        # 把带memo的搜索转为后向的dp
        m, n = len(s), len(t)
        if m == 0 or n == 0:
            return 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(n, m + 1):
            dp[i][n] = 1

        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                dp[i][j] += dp[i + 1][j]
                if s[i] == t[j]:
                    dp[i][j] += dp[i + 1][j + 1]
        for ls in dp:
            print(ls)
        return dp[0][0]

    def numDistinct(self, s: str, t: str) -> int:
        # 优化空间，注意j的循环要反一下
        m, n = len(s), len(t)
        if m == 0 or n == 0:
            return 0
        dp = [0] * (n + 1)
        dp[n] = 1
        for i in range(m - 1, -1, -1):
            for j in range(n):
                if s[i] == t[j]:
                    dp[j] += dp[j + 1]
        return dp[0]


class Solution_139:
    # 参考 472
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        memo = set(wordDict)
        dp = [False] * (len(s) + 1)
        dp[0] = True
        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in memo:
                    dp[i] = True
                    break
        return dp[len(s)]


class Solution_140:
    # 前序题目 139， 参考472
    def _brute_force_TLE_wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        # 暴力解， TLE
        res = []
        memo = {}
        for w in wordDict:
            memo[w] = memo.get(w, 0) + 1

        def dfs(memo, lo, hi, tmp):
            nonlocal res
            if hi == len(s):
                if s[lo:hi] in memo:
                    tmp += [s[lo:hi]]
                    res.append(tmp)
            elif hi < len(s):
                if s[lo:hi + 1] in memo:
                    dfs(memo, hi + 1, hi + 1, tmp + [s[lo:hi + 1]])
                dfs(memo, lo, hi + 1, tmp)

        dfs(memo, 0, 0, [])
        return [' '.join(x) for x in res]

    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        # 转换为带 memo 的递归, AC
        words = set(wordDict)
        memo = {}

        def search(s):
            nonlocal memo
            res = []
            for i in range(len(s)):
                left = s[:i + 1]
                if left in words:
                    right = s[i + 1:]
                    if len(right) == 0:
                        res.append([left])
                        return res
                    if right not in memo:
                        memo[right] = search(right)
                    for ls in memo[right]:
                        res.append([left] + ls)
            return res

        res = search(s)
        return [' '.join(ls) for ls in res]

    def _trie_memo_wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        # 使用前缀树 + memo 的递归, AC
        trie = {}
        for w in wordDict:
            p = trie
            for c in w:
                if c not in p:
                    p[c] = {}
                p = p[c]
            p['#'] = w

        def dfs_find(s, memo):
            res = []
            p = trie
            for i, c in enumerate(s):
                if c not in p:
                    break
                p = p[c]
                if '#' in p:
                    right = s[i + 1:]
                    if len(right) == 0:
                        res.append([p['#']])
                        return res
                    if right not in memo:
                        memo[right] = dfs_find(right, memo)  # 递归查找
                    for ls in memo[right]:
                        res.append([p['#']] + ls)
            return res

        res = dfs_find(s, {})
        return [' '.join(ls) for ls in res]

    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        # DP + 重建 path, AC
        if len(wordDict) == 0:
            return []
        mlen = len(max(wordDict, key=len))
        memo = set(wordDict)
        res = []
        dp = [0] * (len(s) + 1)
        dp[0] = 1
        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in memo:
                    dp[i] = 1

        def build_path(left, idx):
            nonlocal res, dp, memo
            if idx == 0:
                res.append(left[::-1])
                return
            for j in range(max(0, idx - mlen), idx):
                w = s[j:idx]
                if dp[j] and w in memo:
                    build_path(left + [w], j)

        build_path([], len(s))
        return [' '.join(ls) for ls in res]


class Solution_472:
    # 前序题目是139, 212， DP, Trie, DFS
    def _findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        def check(w, mm):
            for i in range(len(w)):
                left, right = w[:i + 1], w[i + 1:]
                if left in mm:
                    if right in mm or check(right, mm):
                        return True
            return False

        memo = set([w for w in words if len(w)])
        res = []
        for w in words:
            if len(w) > 0:
                if check(w, memo):
                    res.append(w)
        return res

    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:

        def create_trie(words):
            root = {}
            for w in words:
                if len(w) == 0: continue
                p = root
                for c in w:
                    if c not in p:
                        p[c] = {}
                    p = p[c]
                p['#'] = w
            return root

        def check(root, w):
            if len(w) == 0:
                return False
            p = root
            for c in w:
                if c not in p:
                    return False
                p = p[c]
            return '#' in p

        def search(root, word):
            for i in range(len(word)):
                left, right = word[:i + 1], word[i + 1:]
                if check(root, left):
                    if check(root, right) or search(root, right):
                        return True
            return False

        root = create_trie(words)
        res = []
        for w in words:
            if len(w) > 0 and search(root, w):
                res.append(w)
        return res

    def _DP_findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:

        def check(word, memo):
            # TODO this also solve 139 word break， and also used for 140 (DP + rebuild)
            dp = [0] * (len(word) + 1)
            dp[0] = 1
            for i in range(1, len(word) + 1):
                for j in range(i):
                    if dp[j] and word[j:i] in memo:
                        dp[i] = 1
                        break
            return dp[len(word)]

        memo = set(words)
        res = []
        for w in words:
            if len(w) > 0:
                memo.remove(w)
                if check(w, memo):
                    res.append(w)
                memo.add(w)
        return res


class Solution_467:
    # String, DP
    def findSubstringInWraproundString(self, p: str) -> int:
        # 这题 p 的长度可能超过 1000， O(n^2) 的解会超时，所以要找更好的解
        # 无穷串s:  ...abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz...
        # 注意到无穷串 s 中任意两个相邻字符都有：
        #  (ord(s[i+1]) - ord(s[i])) % 26 == 1
        # 对于 p ，使用一个例子来说明：abcde
        #   din: 5, b: 4, c: 3, d:2, e:1, 对应表示以字符开头的，是 s 的子串的个数
        # 求和 5 + 4 + 3 + 2 + 1就是最终结果，由于一个字符可能回多次出现，我们只
        # 要连续长度最长的一个就好（求max）。
        res = {c: 1 for c in p}
        ln = 1
        for i in range(1, len(p)):
            ln = ln + 1 if (ord(p[i]) - ord(p[i - 1])) % 26 == 1 else 1
            res[p[i]] = max(res[p[i]], ln)
        print(res)
        return sum(res.values())


######################################################################################################################
# 回文串
class Solution_5:
    # 回文串
    def longestPalindrome(self, s: str) -> str:
        # 纯暴力解，对每一个i->j的子串检查是否是回文串 O(n^3) 注意比较是否是回文串需要O(n), AC
        n = len(s)
        ms = ''
        for i in range(n):
            for j in range(n, i, -1):
                t = s[i:j]
                if t == t[::-1] and len(ms) < len(t):
                    ms = t
        return ms

    def longestPalindrome(self, s: str) -> str:
        # 中心扩展法
        max_len = st = i = 0
        while i < len(s):
            j = i + 1
            while j < len(s) and s[j - 1] == s[j]:  # 处理"bb", "bbb" 这种情况
                j += 1
            left = i
            right = j - 1
            while left > 0 and right < len(s) - 1 and s[left - 1] == s[right + 1]:
                left -= 1
                right += 1
            i = j
            if right - left + 1 > max_len:
                max_len = right - left + 1
                st = left

        return s[st:st + max_len]

    def longestPalindrome(self, s: str) -> str:
        # 由于回文串的正序部分和逆序部分是相同的，所以我们可以将其转换为 s 和 s[::-1] 的 LCS 问题
        # 但是如果 s 种存在一个非回文但是逆向也有相同部分的子串，如 aacdefcaa 这种，LCS 找出来的
        # 最长串并不是最长回文串，所以我们需要另外判断 LCS 找出的是不是一个回文串。
        t = s[::-1]
        n = len(s)
        dp = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
        m, ms = 0, ''
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > m:
                    # 当前最长公共子串的在字符串 s 中的终止点是当前下标 i-1, 起始点是：
                    # (i-1) - dp[i][j] + 1 = i - dp[i][j]
                    # 在 t 中终止点是 j-1, 对应到翻转前的下标应该是 n - 1 - (j-1) = n - j
                    # 如果 i - dp[i][j] == n - j, 那么这一部分公共子串的首尾就是相同的，也即是回文串
                    if i - dp[i][j] == n - j:
                        m = dp[i][j]
                        ms = s[i - m:i]
        for ls in dp:
            print(ls)
        return ms

    def longestPalindrome(self, s):
        # dp, 对于回文串 aba, 显然有 xabax 也是回文串，于是我们有
        # dp[i][i] = s[i] == s[j] and dp[i+1][j-1]
        # 初始值：dp[i][i] = 1
        #        dp[i][i+1] = s[i] == s[j]

        dp = [[0 for _ in range(len(s))] for _ in range(len(s))]
        st, ed = 0, 0
        for i in range(len(s)):
            dp[i][i] = 1
            for j in range(i - 1, -1, -1):
                if j + 1 == i:
                    dp[j][i] = (s[j] == s[i])
                else:
                    dp[j][i] = dp[j + 1][i - 1] and (s[j] == s[i])
                if dp[j][i] and (i - j + 1) > (ed - st + 1):
                    st = j
                    ed = i
        return s[st:ed + 1]

    def longestPalindrome(self, s):
        def manacher(s):
            ma = '^#' + '#'.join(s) + '#$'  # 预处理 长度会变成2n-1+4 = 2(n+1) + 1 奇数
            p = [0] * len(ma)  # 要计算的数组p
            cid = 0
            mx = 0
            max_len = 0
            max_idx = 0
            # 循环从1~len-1, 第0个字符和最后一个字符，'^'和 '$'不会有回文子串，
            # 忽略头尾这两个字符可保证扩展时不检查边界也不会越界。
            for i in range(1, len(ma) - 1):
                if mx > i:  # mx在i的右边
                    p[i] = min(p[2 * cid - i], mx - i)
                else:  # mx在i的左边
                    p[i] = 1

                # 暴力的向两边扩展，注意这里我们并没有检查边界，因为设定了不相等
                # 的字符'^'和 '$'，确保了在遇到边界一定会出现不等而退出循环, bonus~
                while ma[i + p[i]] == ma[i - p[i]]:
                    p[i] += 1

                if i + p[i] > mx:
                    mx = p[i] + i  # 更新 mx 和 cid
                    cid = i
                if p[i] > max_len:
                    max_len = p[i]
                    max_idx = i

            # 映射回原来的字符串中，起始点是最长的回文串的中心的减去长度 // 2，
            # 长度是 max_len - 1(如果半径不算中心的化，那就是max_len)
            st = (max_idx - max_len) // 2
            return s[st: st + max_len - 1]

        return manacher(s)


class Solution_647:
    # 回文串
    def countSubstrings(self, s: str) -> int:
        # expand to both sides, 算上两个字符中间的空，总共有2N-1个位置，检查以这些位置为中心
        # 点对称的字符串是否为回文串。检查的时候同时向两端扩展就可以了。
        # 唯一值得注意的是，偶数位置是指向原字符串的字符，奇数位置指向的两个字符中间的位置。
        # 如例子
        #         din  |  b  |  b  |  din  |  c
        #  index: 0  1  2  3  4  5  6  7  8
        n = len(s)
        res = 0
        for idx in range(2 * n - 1):
            lo = idx // 2
            hi = (idx + 1) // 2  # 奇数位置是两个字符中间，所以hi指向右边字符
            while lo >= 0 and hi < n and s[lo] == s[hi]:
                lo -= 1
                hi += 1
                res += 1
        return res

    def countSubstrings(self, s: str) -> int:
        # dp 方法
        # 状态(i, j)，dp[i][j]表示s的子串s[i:j+1]是回文串
        # 显然：
        #   i > j， dp[i][j] = False, 空串不是回文串（dp数组是个上三角矩阵）
        #   i ==j， dp[i][j] = True, 一个字符是回文串
        #   i < j， 我们可以比较这个子串的两端。
        #      * 如果 s[i] == s[j], dp[i][j] = dp[i+1, j-1]
        #      * 否则 dp[i][j] = False
        # 我们有了一个递推式：dp[i][j] = dp[i+1][j-1]， 如果我们能够先算出dp[i+1, j-1]，
        # 就可以动规解这道题了。注意到(i+1, j-1)是(i, j)的左下位置。因此我们要从主对∩线开始，
        # 逐渐向右上算。
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        cnt = 0
        for d in range(n):  # d表示第几条斜对角线，d=0表示主对角线
            for i in range(n - d):
                j = i + d
                if s[i] == s[j]:  # 这里有两种情况 1. i==j, 2. i < j。所以用i+1 >= j-1判断是否有左下元素
                    dp[i][j] = 1 if i + 1 >= j - 1 else dp[i + 1][j - 1]
                    print(i, j)
                    if dp[i][j]:
                        cnt += 1
        return cnt

    def countSubstrings(self, s: str) -> int:
        # Manacher 算法，在O(n)时间内求最长的回文子串的长度。
        # 给定一个字符串s, 长度为 n
        # 首先对字符串预处理，在每个字符中间插入不会出现的字符，比如'#',
        # 然后在两头插入一对不相同的字符且和中间插的不同，比如'^'和 '$'。
        # 经过这样的处理后字符串长度变为 2n+3(奇数)，记处理后字符串为ma。
        # 我们使用一个数组p, p[i]表示以i位置字符为中心的回文子串的半径。（P[i]的值恰好对应删去预处理插入的字符后的回文串的长度）
        # 我们的目的就是求出数组p.
        # 根据动态规划的思想，如果我们要求p[i]，则一定已经计算过了所有j<i的p[j]的值。
        # 使用mx表示 0 + p[0] 到 i-1 + p[i-1] 的最大值，即向右匹配到的最右的位置
        # 对应的cid记录匹配的最右mx的中心位置，即有 cid + p[cid] == mx
        # 设j是i关于cid的对应位置，即  -----j---cid---i----
        # 现在已知p[j], cid, p[cid], mx的情况下，求p[i].
        # 情况如下：
        #    1. mx 在i的左边(mx<=i),--j-mx'--cid--mx-i----, 那就只能暴力从i向两边扩展求p[i]的长度
        #    2. mx 在i的右边(mx>i), ---mx'--j---cid---i--mx--, 显然以cid为中心的回文子串包括了i位置。
        #          我们写的清楚一点，ma[j] == ma[i], ma[j-1] == ma[i+1], ..., ma[mx'] == ma[mx]
        #          而且很明显以i为中心的回文子串与以j为中心的回文子串有关，p[i]的值有一个下界 p[j]
        #          如果p[j] + i > mx，而以i为中心，mx之后部分是未检查过的，因此另一个下界是 mx - i
        #          综合起来 p[i] = min(p[j], mx-i), 然后在此基础上向外扩展。
        # 对于j,我们有j = cid - (i - cid) = 2 * cid - i
        def manacher(s):
            ma = '^#' + '#'.join(s) + '#$'  # 预处理 长度会变成2n-1+4 = 2(n+1) + 1 奇数
            p = [0] * len(ma)  # 要求的数组p
            cid = 0
            mx = 0
            # 循环从1~len-1, 第0个字符和最后一个字符，'^'和 '$'不会有回文子串，
            # 忽略头尾这两个字符可保证扩展时不检查边界也不会越界。
            for i in range(1, len(ma) - 1):
                if mx > i:  # mx在i的右边
                    p[i] = min(p[2 * cid - i], mx - i)
                else:  # mx在i的左边
                    p[i] = 1

                # 暴力的向两边扩展，注意这里我们并没有检查边界，因为设定了不相等
                # 的字符'^'和 '$'，确保了在遇到边界一定会出现不等而退出循环。bonus~
                while ma[i + p[i]] == ma[i - p[i]]:
                    p[i] += 1

                if i + p[i] > mx:
                    mx = p[i] + i  # 更新 mx 和 cid
                    cid = i
            return p

        return sum(v // 2 for v in manacher(s))


class Solution_132:
    def minCut(self, s: str) -> int:
        def foo(idx):
            if idx <= 0:
                return 0
            ans = len(s)
            for i in range(idx - 1, -1, -1):
                t = s[i:idx]
                if t == t[::-1]:
                    ans = min(ans, 1 + foo(i))
            return ans

        return foo(len(s)) - 1

    def minCut(self, s: str) -> int:
        # 带 memo 的递归
        memo = {}

        def foo(idx):
            nonlocal memo
            if idx <= 0:
                return 0
            if idx in memo:
                return memo[idx]
            ans = len(s)  # 最大的值是 len(s) - 1，我们在外面减一，这里取len(s)
            for i in range(idx - 1, -1, -1):
                t = s[i:idx]
                if t == t[::-1]:
                    ans = min(ans, 1 + foo(i))
            memo[idx] = ans
            return ans

        return foo(len(s)) - 1

    def minCut(self, s: str) -> int:
        # DP
        dp = [len(s)] * (len(s) + 1)
        dp[0] = 0
        for i in range(len(s) + 1):
            for j in range(i):
                t = s[j:i]
                if t == t[::-1]:
                    dp[i] = min(dp[i], 1 + dp[j])
        return dp[len(s)] - 1

    def minCut(self, s: str) -> int:
        # 上面检查子串 t 是否回文的时候用的是暴力方法，我们可以利用
        # 回文的性质，使用 DP 来检查一个子串是否是回文。
        n = len(s)
        dp = [[False] * (n + 1) for _ in range(n + 1)]
        cut = [0] * n
        for i in range(n):
            cut[i] = i
            for j in range(i + 1):
                # j + 1 > i - 1 means j is i-1 or i，in both case, s[j:i+1] is palindrome
                if s[i] == s[j] and (j + 1 > i - 1 or dp[j + 1][i - 1]):
                    dp[j][i] = True
                    cut[i] = 0 if j == 0 else min(cut[i], cut[j - 1] + 1)
        return cut[n - 1]

    def minCut(self, s: str) -> int:
        # 同样，我们也可以使用 中心扩展法，而不使用 DP 数组来记录子串是否是回文。
        n = len(s)
        cut = [i for i in range(-1, n)]
        for idx in range(1, n):
            for lo, hi in [(idx, idx), (idx - 1, idx)]:  # 分别对应奇数长度和偶数长度的回文串两种情况
                while lo >= 0 and hi < n and s[lo] == s[hi]:
                    cut[hi + 1] = min(cut[hi + 1], cut[lo] + 1)
                    lo -= 1
                    hi += 1
        return cut[-1]


######################################################################################################################
#
class Solution_312:
    def maxCoins(self, nums: List[int]) -> int:
        # 暴力搜索
        def foo(arr, lo, hi):
            if lo + 1 == hi:  # 没有元素了。只剩下头尾的 1
                return 0
            ans = 0
            for i in range(lo + 1, hi):
                # 这里是 key idea
                # 先把 lo~i 内的和 i~hi 都 burst，然后把 i burst
                # 这样分别对应 lo~i: foo(arr, lo, i)
                #             i~hi: foo(arr, i, hi)
                #               i : arr[lo] * arr[i] * arr[hi]
                ans = max(ans, arr[lo] * arr[i] * arr[hi] + foo(arr, lo, i) + foo(arr, i, hi))
            return ans

        nums = [1] + nums + [1]
        return foo(nums, 0, len(nums) - 1)

    def maxCoins(self, nums: List[int]) -> int:
        # 带 memo 的递归
        memo = {}

        def foo(arr, lo, hi):
            nonlocal memo
            if lo + 1 == hi:  # 没有元素了。只剩下头尾的 1
                return 0
            if (lo, hi) in memo:
                return memo[(lo, hi)]
            ans = 0
            for i in range(lo + 1, hi):
                ans = max(ans, arr[lo] * arr[i] * arr[hi] + foo(arr, lo, i) + foo(arr, i, hi))
            memo[(lo, hi)] = ans
            return ans

        nums = [1] + nums + [1]
        return foo(nums, 0, len(nums) - 1)

    def maxCoins(self, nums: List[int]) -> int:
        # 转化为 DP
        nums = [1] + nums + [1]
        n = len(nums)
        dp = [[0] * n for _ in range(n)]
        # 注意这里并不是:
        # for lo in range(n):
        #     for hi in range(lo+2, hi):
        # 因为这样迭代在求 dp[lo][hi] 时 dp[i][hi]还没有被计算过
        #
        # 使用下面的方式沿主对角线方向依次往右上算，这样可以保证在求 dp[lo][hi] 时，
        # dp[lo][i] 和 dp[i][hi]都已经被计算过。如下，第一次计算的时 + 表示斜线
        # - - + - - -
        # - - - + - -
        # - - - - + -
        # - - - - - +
        # - - - - - -
        # - - - - - -
        # 空间优化，lo < hi 所以 dp 是个上三角矩阵，可以压缩存储（省一半空间）
        for k in range(2, n):
            for lo in range(n - k):
                hi = lo + k
                for i in range(lo + 1, hi):
                    dp[lo][hi] = max(dp[lo][hi], nums[lo] * nums[i] * nums[hi] + dp[lo][i] + dp[i][hi])
        for ls in dp:
            print(ls)
        return dp[0][n - 1]


class Solution_174:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        # 关键点： 找最小的初始 HP, 也就是找消耗 HP 路径最少的路径（但是在这个路径上，HP值一定要是正的）。
        #         要反过来思考，找每一个位置到右下角需要的最小的 HP 值。这样更简单，而且可以定义最优子结构，
        #         方便转化为 DP 方法。
        m = len(dungeon)
        n = len(dungeon[0]) if m else 0
        if m == 0 or n == 0:
            return 0

        def foo(x, y):
            if x == m - 1 and y == n - 1:
                # 右下角位置，需要的 hp 是：
                #   1                   if dungeon[x][y] >= 0
                #   1-dungeon[x][y]     if dungeon[x][y] < 0
                return max(1, 1 - dungeon[x][y])
            elif x == m - 1:
                # 下边界，只能向右，需要的 hp 是：
                #   1      如果右边需要的 hp = foo(x, y+1) <= dungeon[x][y]
                #   foo(x, y+1) - dungeon[x][y]  如果右边需要的 hp = foo(x, y+1) > dungeon[x][y]
                return max(1, foo(x, y + 1) - dungeon[x][y])
            elif y == n - 1:
                # 右边界，只能向下，需要的 hp 是：
                #   1      如果下边需要的 hp = foo(x+1, y) <= dungeon[x][y]
                #   foo(x+1, y) - dungeon[x][y]  如果下边需要的 hp = foo(x+1, y) > dungeon[x][y]
                return max(1, foo(x + 1, y) - dungeon[x][y])
            else:
                # 有向右和向下两种选择，需要的 hp 是：
                #   min(foo(x+1, y), foo(x, y+1) - dungeon[x][y]    如果右/下需要的 hp 的最小值 > dungeon[x][y]
                #   1       otherwise
                return max(1, min(foo(x + 1, y), foo(x, y + 1)) - dungeon[x][y])

        return foo(0, 0)

    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        # 转化为 memo 递归
        m = len(dungeon)
        n = len(dungeon[0]) if m else 0
        if m == 0 or n == 0:
            return 0
        memo = {}

        def foo(x, y):
            nonlocal memo
            if (x, y) in memo:
                return memo[(x, y)]
            if x == m - 1 and y == n - 1:
                res = max(1, 1 - dungeon[x][y])
            elif x == m - 1:
                res = max(1, foo(x, y + 1) - dungeon[x][y])
            elif y == n - 1:
                res = max(1, foo(x + 1, y) - dungeon[x][y])
            else:
                res = max(1, min(foo(x + 1, y), foo(x, y + 1)) - dungeon[x][y])
            memo[(x, y)] = res
            return res

        return foo(0, 0)

    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        # 转化为 DP
        m = len(dungeon)
        n = len(dungeon[0]) if m else 0
        if m == 0 or n == 0:
            return 0
        dp = [[0] * n for _ in range(m)]
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if i == m - 1 and j == n - 1:
                    dp[i][j] = max(1, 1 - dungeon[i][j])
                elif i == m - 1:
                    dp[i][j] = max(1, dp[i][j + 1] - dungeon[i][j])
                elif j == n - 1:
                    dp[i][j] = max(1, dp[i + 1][j] - dungeon[i][j])
                else:
                    dp[i][j] = max(1, min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j])
        return dp[0][0]

    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        # 空间优化，注意到只依赖dp[i+1][j] 和 dp[i][j+1]
        m = len(dungeon)
        n = len(dungeon[0]) if m else 0
        if m == 0 or n == 0:
            return 0
        dp = [0] * n
        dp[n - 1] = max(1, 1 - dungeon[m - 1][n - 1])
        for j in range(n - 2, -1, -1):
            dp[j] = max(1, dp[j + 1] - dungeon[m - 1][j])

        for i in range(m - 2, -1, -1):
            dp[n - 1] = max(1, dp[n - 1] - dungeon[i][n - 1])
            for j in range(n - 2, -1, -1):
                dp[j] = max(1, min(dp[j], dp[j + 1]) - dungeon[i][j])
        return dp[0]

    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        m = len(dungeon)
        n = len(dungeon[0]) if m else 0
        if m == 0 or n == 0:
            return 0

        # 我们可以通过一个小 trick 来简化前面的代码，略去一些条件判断。将状态转移统一成：
        # foo(x, y) = max(1, min(foo(x+1, y), foo(x, y+1)) - dungeon[x][y])
        # 的形式。我们通过把状态由(m-1, n-1)扩展到(m, n)来实现。 为此，我们要考虑以下三种情况：
        #    1. 在原本的right-bottom 有 max(1, 1-dungeon[m-1][n-1])，所以 min(foo(m-1, n), foo(m, n-1)) = 1。两项都为 1
        #    2. 在原本的下边界上，只能向右，有max(1, foo(m-1, y + 1) - dungeon[m-1][y])。
        #       所以 min(foo(m-1, y+1), foo(m, y)) = foo(m-1, y+1)。即：对 y != n-1 有 foo(m, y) = float('inf')
        #    3. 在原本的右边界上，只能向下，有max(1, foo(x + 1, n-1) - dungeon[x][n-1])。
        #       所以 min(foo(x, n), foo(x+1, n-1)) = foo(x+1, n-1)。即：对 x != m-1 有 foo(x, n) = float('inf')
        #
        # 于是，搜索过程如下：

        def foo(x, y):
            if (x == m and y == n - 1) or (x == m - 1 and y == n):
                return 1
            elif x == m or y == n:
                return float('inf')
            else:
                return max(1, min(foo(x + 1, y), foo(x, y + 1)) - dungeon[x][y])

        # 直接把搜索过程转换为 DP 过程如下
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        dp[m][n - 1] = dp[m - 1][n] = 1
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                dp[i][j] = max(1., min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j])

        return int(dp[0][0])

    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        # 空间压缩
        m = len(dungeon)
        n = len(dungeon[0]) if m else 0
        if m == 0 or n == 0:
            return 0

        dp = [float('inf')] * (n + 1)
        dp[n - 1] = dp[n] = 1
        for j in range(n - 1, -1, -1):
            dp[j] = max(1., min(dp[j], dp[j + 1]) - dungeon[m - 1][j])

        for i in range(m - 2, -1, -1):
            dp[n] = float('inf')
            for j in range(n - 1, -1, -1):
                dp[j] = max(1., min(dp[j], dp[j + 1]) - dungeon[i][j])

        return int(dp[0])


class Solution_375:
    # MinMax问题
    def getMoneyAmount(self, n: int) -> int:
        # 暴力搜索
        def foo(lo, hi):
            if lo >= hi:
                return 0
            res = float('inf')
            for i in range(lo, hi):
                # 在局部取最糟糕的情况，max(foo(lo, i-1), foo(i+1, hi)), 走向花费大的分支
                # 在所有局部最糟糕（花费大的分支）的情况下，选最小的
                res = min(res, i + max(foo(lo, i - 1), foo(i + 1, hi)))
            return res

        return foo(1, n)

    def getMoneyAmount(self, n: int) -> int:
        # 带 memo 的递归搜索
        memo = {}

        def foo(lo, hi):
            nonlocal memo
            if lo >= hi:
                return 0
            if (lo, hi) in memo:
                return memo[(lo, hi)]
            res = float('inf')
            for i in range(lo, hi):
                res = min(res, i + max(foo(lo, i - 1), foo(i + 1, hi)))
            memo[(lo, hi)] = res
            return res

        return foo(1, n)

    def getMoneyAmount(self, n: int) -> int:
        # 转化为 O(n^3) DP, 和 312.题差不多， dp[lo][hi] 为猜 lo ~ hi 范围内的最少花费
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(1, n):
            dp[i][i + 1] = i + 1  # 初始化，区间长度为 1(lo+1 == hi)
        for k in range(2, n):  # 区间长度为 2 ~ n-1.
            for lo in range(1, n + 1 - k):
                hi = lo + k
                m = float('inf')
                for i in range(lo, hi):
                    m = min(m, i + max(dp[lo][i - 1], dp[i + 1][hi]))
                dp[lo][hi] = m
            print('####')
            for ls in dp:
                print(ls)
        return dp[1][n]

    def getMoneyAmount(self, n: int) -> int:
        # 优化时间复杂度为O(n^2)
        # > 在二分查找的题目：887.丢鸡蛋问题中，我们有一个类似的式子：
        # >      dp(k, n) = min_{1<=x_train<=N} (max(dp(k-1, x-1), dp(k, n-x)))
        # > 其中t1 = dp(k-1, x-1), 随x单调增
        # >     t2 = dp(k, n-x),  随x单调减
        # > 于是max(t1,t2)是二者的上半部分
        # 同样，这里我们有： dp(lo, hi) = min_{lo <= i < hi}{i+max(dp(lo, i-1), dp(i+1, hi))}
        # 其中 dp(lo, i-1)，随 i 单调增
        #      dp(i+1, hi), 随 i 单调减
        # 所以整体的最小值点在两者的“交点”处。
        # 记： i0(lo, hi) = max{i: dp(lo, i-1) <= dp(i+1, hi)}, 即 i0 为区间 lo~hi 内的“交点”
        # 这样 max(dp(lo, i-1), dp(i+1, hi)) 的值为：
        #    dp(i+1, hi),  if lo <= i <= i0(lo, hi)
        #    dp(lo, i-1),  if i0(lo, hi) < i <= hi
        # 即取值为dp(i+1, hi)和dp(lo, i-1)图像的上班部分，在“交点”左边，取值dp(i+1, hi)， 在“交点”右边，取值dp(lo, i-1)
        # 因此：
        #    dp(lo, hi) = min_{lo <= i < hi}{t1, t2}
        #    t1 = min_{lo <= i <= i0(lo, hi)} {dp(i+1, hi) + i}
        #    t2 = min_{i0(lo, hi) < i <= hi} {dp(lo, i-1) + i} = dp(lo, i0(lo, hi)) + i0(lo,hi) + 1
        # 与887. 不同之处在于，这里的 max 项还加上了 i。
        # therefore: i + dp(lo, i-1)，随 i 单调增
        #            i + dp(i+1, hi)，随 i 的单调性无法判断
        # 对于 i + dp(lo, i-1)，由于单调增，所以我们可以利用上一次求得t2最小值点 i0 左移来找新的t2最小值点，避免扫描整个区间
        # 对于 i + dp(i+1, hi)，我们需要使用一个升序的 deque 来手动维护单调性。
        # 参考 https://artofproblemsolving.com/community/c296841h1273742
        # 参考 https://blog.csdn.net/Site1997/article/details/100168676
        dp = [[0] * (n + 1) for _ in range(n + 1)]

        for hi in range(2, n + 1):
            i0 = hi - 1  # 用来在 lo 的循环中保存上一次的“交点”, i0 < hi
            deque = collections.deque()
            for lo in range(hi - 1, 0, -1):  # lo 从 hi-1 到 1
                while dp[lo][i0 - 1] > dp[i0 + 1][hi]:  # 在“交点”右侧，将 i0 左移(i0 -= 1)
                    if len(deque) and deque[0][1] == i0:
                        # deque 中保存得可能点 i == i0, 紧接着i0左移之后 i > i0, 所以要从 deque 中丢掉
                        deque.popleft()
                    i0 -= 1  # i0 左移

                v = lo + dp[lo + 1][hi]
                while len(deque) and v < deque[-1][0]:
                    deque.pop()
                # 随着 lo 减小（左边界减小），把 vn 放入 deque 就是把 min_{lo <= i <= i0}(f(i+1,b) + i) 放入了 deque。
                # 其实就是通过 deque 维护了一个升序序列，队列首元素为 最小值。
                # 这是通过两个部分来实现的：
                #     1. deque.popleft() with deque[0][1] == i0 保证了所有 i > i0 的值
                #        都被丢掉了（i0 左移减小之前把队列中的 i0丢掉）
                #     2. deque.pop() with vn < deque[-1][0] 保证了 deque 中是一个升序序
                #        列，所有大于 vn 的都被丢弃了
                deque.append((v, lo))
                # dp(lo, (i0 + 1) - 1) + (i0 + 1)， i0 + 1取得最小值
                # 队首元素, 是 dp(i+1, hi) + i 的最小值
                # 两者中的 min 是区间 lo ~ hi 的最小值
                dp[lo][hi] = min(dp[lo][i0] + i0 + 1, deque[0][0])

        # debug
        print('####')
        for ls in dp:
            print(ls)
        return dp[1][n]


class Solution_1140:
    # MinMax Game
    def stoneGameII(self, piles: List[int]) -> int:
        # 使用 game tree 结构进行搜索, turn代表游戏双方，
        # 0 表示 Alex，他的目标是最大化收益,
        # 1 表示对手 Lee，他的目标是最小化Alex的收益（等价于最大化自己的）
        def foo(piles, m, turn=0):
            if len(piles) == 0:
                return 0
            if turn == 0:
                ans = 0
                for x in range(1, min(len(piles), 2 * m) + 1):  # 1 <= x <= 2*m
                    ans = max(ans, foo(piles[x:], max(m, x), turn=1 - turn) + sum(piles[:x]))
            else:
                ans = float('inf')
                for x in range(1, min(len(piles), 2 * m) + 1):  # 1 <= x <= 2*m
                    ans = min(ans, foo(piles[x:], max(m, x), turn=1 - turn))
            return ans

        return foo(piles, 1)

    def stoneGameII(self, piles: List[int]) -> int:
        # maxmin game
        # memo + 递归, TLE
        n = len(piles)
        memo = {}

        def foo(idx, m):
            nonlocal memo
            if idx >= n:
                return 0
            if (idx, m) in memo:
                return memo[(idx, m)]
            ans = 0
            for x in range(1, min(n, 2 * m) + 1):  # 1 <= x <= 2*m
                mt = max(m, x)
                res = float('inf')
                for t in range(1, min(n, 2 * mt) + 1):  # 1 <= t <= 2*mt
                    res = min(res, foo(idx + t + x, max(mt, t)))
                res = 0 if res == float('inf') else res
                ans = max(ans, res + sum(piles[idx:idx + x]))
            memo[(idx, m)] = ans
            return ans

        return foo(0, 1)

    def stoneGameII(self, piles: List[int]) -> int:
        N = len(piles)
        # 后缀和
        for i in range(N - 2, -1, -1):
            piles[i] += piles[i + 1]

        from functools import lru_cache

        @lru_cache(None)
        def dp(i, m):
            if i + 2 * m >= N: return piles[i]  # 当前可以取完，就取完， 当 i>=N 时取到的是零
            return piles[i] - min(dp(i + x, max(m, x)) for x in range(1, 2 * m + 1))  # 最优自己=>最小化对手

        return dp(0, 1)


class Solution_1262:
    def maxSumDivThree(self, nums: List[int]) -> int:

        def _foo(i, a):
            # 这个暴力写起来简单，但是有向下递归的 din 以及向上回溯的 return max.
            # 而且状态定义不明确，所以不好转换成dp
            if i == len(nums):
                return a if a % 3 == 0 else 0
            return max(foo(i + 1, a), foo(i + 1, a + nums[i]))

        def foo(i, k):
            if i == -1:
                return 0
            # k = 0/1/2
            ans = foo(i - 1, k)
            for rk in range(3):
                prev = foo(i - 1, rk)
                t = prev + nums[i]
                if t % 3 == k:
                    ans = max(ans, t)

            return ans

        return foo(len(nums) - 1, 0)

    def maxSumDivThree(self, nums: List[int]) -> int:
        memo = {}

        def foo(i, k):
            nonlocal memo
            if i == -1:
                return 0
            if (i, k) in memo:
                return memo[(i, k)]
            # k = 0/1/2
            ans = foo(i - 1, k)
            for rk in range(3):
                prev = foo(i - 1, rk)
                t = prev + nums[i]
                if t % 3 == k:
                    ans = max(ans, t)
            memo[(i, k)] = ans
            return ans

        return foo(len(nums) - 1, 0)

    def maxSumDivThree(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[0] * (n + 1) for _ in range(3)]
        for i in range(1, n + 1):
            for k in range(3):
                dp[k][i] = dp[k][i - 1]
            for k in range(3):
                t = dp[k][i - 1] + nums[i - 1]
                dp[t % 3][i] = max(t, dp[t % 3][i])

        for ls in dp:
            print(ls)
        return dp[0][n]

    def maxSumDivThree(self, nums: List[int]) -> int:
        # 空间优化，第 i 列只跟 i-1 列有关，可以优化到O(1)空间
        dp = [0, 0, 0]
        for x in nums:
            tmp = dp[:]  # slice 很重要，使用一个副本，因为下面的循环中会修改数组
            for prev in tmp:
                t = prev + x
                dp[t % 3] = max(dp[t % 3], t)
        return dp[0]


class Solution_1227:
    def nthPersonGetsNthSeat(self, n: int) -> float:
        # 根据题设，n 个乘客 n 个座位，第一个乘客随机选一个座位，之后的每个乘客：
        #    1. 如果自己的座位没被选，选自己的座位
        #    2. 如果自己的座位被选了，随机从剩下的选一个
        # 不妨假设原本第 i 个乘客的座位是 i。
        # 那么对于第一个乘客来说，最后一个客人能选到自己原本位子有两种情况：
        #    1. 第一个乘客随机选到了自己的位子，对应概率为 1/n
        #    2. 第一个乘客既没选到自己位子，也没选到第 n 个乘客的位子, 假设选到了乘客 i(i<n) 的位子（概率为 (n-2)/n ），
        #       那么乘客 i 需要随机选位子。
        #         i). 如果乘客 i 选到了第一个乘客的位置，乘客 n 一定可以选到自己的位子，对应概率为  1/(n-1)
        #        ii). 如果乘客 i 选到了既不是第一个，也不是乘客 n 的位置（概率为 ((n-1)-2)/(n-1)）, 这是一个递归
        # 综合起来：
        #   f(n) = 1/n + (1 - 2/n) * f(n-1)
        # 上式子是一个迭代式，我们可以使用递归来实现它：
        def foo(n):
            if n == 1:
                return 1.0
            return 1 / n + (1 - 2 / n) * foo(n - 1)

        # 同样给我们也可将其转化为 DP
        def DP(n):
            dp = [0] * (n + 1)
            dp[1] = 1.0
            for i in range(2, n + 1):
                dp[i] = 1 / i + (1 - 2 / i) * dp[i - 1]
            return dp[n]

        return DP(n)

    def nthPersonGetsNthSeat(self, n: int) -> float:
        # 根据递推式 f(n) = 1/n + (1 - 2/n) * f(n-1), 以及初始值 f(1) = 1.0
        # 我们可以求 f(2) = 0.5， f(3) = 0.5, f(4) = 0.5, ......
        # 使用归纳法证明: 对于 n >= 2, f(n) = 0.5
        #   1. 初值 f(2) = 0.5
        #   2. 假设 f(k) = 0.5，则有
        #         f(k+1) = 1/(k+1) - (1 - 2/(k+1)) * f(k)
        #                = 1/(k+1) - (1 - 2/(k+1)) * 1/2
        #                = (K+1)/(2*(k+1)) = 1/2
        # 综和 1 和 2 得证。
        return 1.0 if n == 1 else 0.5


class Solution_1320:
    # DP, String
    def minimumDistance(self, word: str) -> int:
        def _dist(a, b):
            a = divmod(ord(a) - ord('A'), 6)
            b = divmod(ord(b) - ord('A'), 6)
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # 第一字母选择 finger 1, 之后的字母两种选择，finger 2/1，找一个最小值
        # 关键点是找到 figure 2 的入点
        # 使用 DFS + memo
        memo = {}

        def foo(f1, f2, j):
            nonlocal memo
            d = 0
            i = j + 1
            if (f1, f2, j) in memo:
                return memo[(f1, f2, j)]
            if i < len(word):
                d1, d2 = _dist(f1, word[i]), _dist(f2, word[i])
                d = min(d1 + foo(word[i], f2, i), d2 + foo(f1, word[i], i))
            memo[(f1, f2, j)] = d
            return d

        pre = [0] * len(word)  # 表示加到当前为止的 dist
        dp = [[]]  # pass
        res = float('inf')
        for j in range(1, len(word)):
            if j > 1:
                pre[j - 1] += pre[j - 2] + _dist(word[j - 2], word[j - 1])

            dist = pre[j - 1]
            f1, f2 = word[j - 1], word[j]
            dist += foo(f1, f2, j)
            print(dist)
            res = min(res, dist)
        print(pre)
        return res

    def minimumDistance(self, word: str) -> int:
        # DP 3D
        def _dist(a, b):
            if a == 26:
                return 0
            return abs(a // 6 - b // 6) + abs(a % 6 - b % 6)

        dp = [[[0] * 27 for _ in range(27)] for _ in range(301)]
        for k in range(len(word) - 1, -1, -1):
            b = ord(word[k]) - ord('A')
            for i in range(27):  # finger 1
                for j in range(27):  # finger 2
                    dp[k][i][j] = min(dp[k + 1][i][b] + _dist(j, b), dp[k + 1][j][b] + _dist(i, b))
        return dp[0][26][26]

    def minimumDistance(self, word: str) -> int:
        # DP 2D
        def _dist(a, b):
            if a == 26:
                return 0
            return abs(a // 6 - b // 6) + abs(a % 6 - b % 6)

        # f1, f2 -> dist
        dp, dp_ = {(26, 26): 0}, {}
        for c in word:
            t = ord(c) - ord('A')
            for a, b in dp:
                dp_[t, b] = min(dp_.get((t, b), float('inf')), dp[a, b] + _dist(a, t))  # f1 移动到t
                dp_[a, t] = min(dp_.get((a, t), float('inf')), dp[a, b] + _dist(b, t))  # f2 移动到t
            dp, dp_ = dp_, {}
        return min(dp.values())

    def minimumDistance(self, word: str) -> int:
        # DP 1D 超出我的理解范围了
        def _dist(a, b):
            return abs(a // 6 - b // 6) + abs(a % 6 - b % 6)

        dp = [0] * 26
        s = 0
        for i in range(1, len(word)):
            b, c = ord(word[i - 1]) - ord('A'), ord(word[i]) - ord('A')
            s += _dist(b, c)
            dp[b] = max(dp[a] + _dist(b, c) - _dist(a, c) for a in range(26))

        return s - max(dp)


class Solution_1312:

    def minInsertions(self, s):
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        # dp(i, j) 表示 s[i:j+1] 是需要插入的次数

        # 计算顺序是主对角线，从左上到右下。之后在计算右上一行，一直到计算出 dp[0][n-1]
        for L in range(n - 1, -1, -1):
            for i in range(L):
                j = i + n - L
                # 如果 s[i] == s[j]，则情况等价于 s[i+1:(j-1)+1] 的情况 dp[i+1][j-1]
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1]
                else:
                    # 不相等的情况，就是左边插入一个/右边插入一个的最小值，即 dp[i+1][j] / dp[i][j-1]
                    dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j - 1])

        return dp[0][n - 1]


#######################################################################################################################
# 以下两个都用了 mono stack + DP 的方法

class Solution_1340:
    def maxJumps(self, arr: List[int], d: int) -> int:
        # 只能跳到在 [i-d, i+d] 之间比 arr[i] 小的，并且当前位置 i 和目标位置 j
        # 之间的数字都比当前低。

        # DP 问题，dp[i] 表示从当前位置能完成的最大跳数
        # 递推式：dp[i] = 1 + max(dp[j]) , j 表示从 i 出发所有能到达的位置

        def foo(i, dp):
            if dp[i]:
                return dp[i]
            r = 1

            for k in range(1, d + 1):
                if i - k >= 0 and arr[i - k] < arr[i]:
                    r = max(r, 1 + foo(i - k, dp))
                else:
                    break
            for k in range(1, d + 1):
                if i + k < len(arr) and arr[i + k] < arr[i]:
                    r = max(r, 1 + foo(i + k, dp))
                else:
                    break

            dp[i] = r
            return r

        dp = [0] * len(arr)
        for i in range(len(arr)):
            foo(i, dp)

        return max(dp)

    def maxJumps(self, arr: List[int], d: int) -> int:
        # Bottom Up DP
        # 对于每个 状态的值 其依赖与比它小的状态值
        # 则我们可以对原数组进行排序后再DP
        # 时间复杂度： O(nlogn + nd)
        dp = [1] * len(arr)
        for a, i in sorted([a, i] for i, a in enumerate(arr)):
            for di in [-1, 1]:  # 方向
                for j in range(i + di, i + di * d + di, di):
                    if not (0 <= j < len(arr) and arr[j] < arr[i]): break
                    dp[i] = max(dp[i], dp[j] + 1)
        print(dp)
        return max(dp)

    def maxJumps(self, arr: List[int], d: int) -> int:
        # 单调栈 + DP
        # Decreasing Stack + DP
        # 显然初始值是 1
        # 使用一个降序栈，每次从栈中弹出的元素，就是当前位置往左可以跳到的元素(满足 i-j <= d)，
        # 左边的 i-d 范围内，低于 arr[i] 的元素都是可以跳到的。
        # 同时，弹出的元素是此时栈顶元素往右可以跳到的位置(满足 j-stack[-1] <= d), 右边的 i+d 范围
        # 内，低于 arr[stack[-1]] 的元素都是可以跳到的。
        # 这样，递推的过程就解决了。
        # 另外，我们在数组最后添加一个 float('inf') 作为终止节点，这样我们会在遇到终止节点时将栈弹空。
        n = len(arr)
        dp = [1] * (n + 1)
        stack = []
        for i, x in enumerate(arr + [float('inf')]):
            while stack and arr[stack[-1]] < x:
                L = [stack.pop()]

                # 弹出一堆相同的数
                while stack and arr[stack[-1]] == arr[L[0]]:
                    L.append(stack.pop())

                for j in L:
                    if i - j <= d:
                        dp[i] = max(dp[i], dp[j] + 1)  # i 左边可以跳到的位置

                    if stack and j - stack[-1] <= d:  # stack[-1] 右边可以跳到的位置
                        dp[stack[-1]] = max(dp[stack[-1]], dp[j] + 1)
            stack.append(i)
        return max(dp[:-1])


class Solution_1335:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        # 数组，划分成 d 非空子数组，使得每个子数组的最的值的和最小
        # min(sum(max(Ai) for i in range(d)))
        # 枚举所有的子划分
        # dp(i, d) 表示从 i 开始 剩余 d 次划分的结果
        # d(i, d) = min(max(A[i:j+1])+dp(j+1, d-1) for j in range(i, n-d+1))
        n = len(jobDifficulty)
        if n < d: return -1

        dp = [[-1] * (d + 1) for _ in range(n)]

        def dfs(i, d, dp):
            if dp[i][d] != -1:
                return dp[i][d]

            if d == 1:
                return max(jobDifficulty[i:])
            res, maxd = float('inf'), 0
            for j in range(i, n - d + 1):  # 从 i 开始到 n - d 保留一个
                maxd = max(maxd, jobDifficulty[j])
                res = min(res, maxd + dfs(j + 1, d - 1, dp))
            dp[i][d] = res
            return res

        return dfs(0, d, dp)

    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        # 将上述 dfs + memo 转换为 DP
        # d(i, d) = min(max(A[i:j+1])+dp(j+1, d-1) for j in range(i, n-d+1))
        n = len(jobDifficulty)
        if n < d: return -1

        dp = [[float('inf')] * n + [0] for _ in range(d + 1)]
        for i in range(1, d + 1):
            for k in range(n - i + 1):
                m = 0
                for j in range(k, n - i + 1):
                    m = max(m, jobDifficulty[j])
                    dp[i][k] = min(dp[i][k], m + dp[i - 1][j + 1])

        return dp[d][0]

    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        # 空间优化，每次只依赖于前一行(d-1), 因此可以对空间进行优化
        # 对于行内，计算 dp[i] 时 用到的是 range(i, n-i+1) 都是 i 后面的，正向循环时还未更新过，
        # 恰好是 d-1 时对应的值。
        n = len(jobDifficulty)
        if n < d: return -1

        dp = [float('inf')] * n + [0]
        for i in range(1, d + 1):
            for k in range(n - i + 1):
                m, dp[k] = 0, float('inf')
                for j in range(k, n - i + 1):
                    m = max(m, jobDifficulty[j])
                    dp[k] = min(dp[k], m + dp[j + 1])

        return dp[0]

    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        # 单调栈 + DP
        # TODO 理解
        n = len(jobDifficulty)
        if n < d: return -1
        dp, dp_ = [float('inf')] * n, [0] * n

        for d in range(d):
            stack = []
            for i in range(d, n):
                dp_[i] = dp[i - 1] + jobDifficulty[i] if i else jobDifficulty[i]
                # 单调减的栈
                while stack and jobDifficulty[stack[-1]] <= jobDifficulty[i]:
                    j = stack.pop()  # 弹出比当前 objDifficulty[i] 小的元素
                    dp_[i] = min(dp_[i], dp_[j] - jobDifficulty[j] + jobDifficulty[i])
                if stack:
                    dp_[i] = min(dp_[i], dp_[stack[-1]])
                stack.append(i)
            dp, dp_ = dp_, [0] * n
        return dp[-1]

    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        n = len(jobDifficulty)
        if n < d: return -1

        dp, dp_ = [0] * n, [0] * n

        # 初始存储到当前位置的最大值
        dp[0] = jobDifficulty[0]
        for i in range(1, n):
            dp[i] = max(dp[i - 1], jobDifficulty[i])

        # 外层循环 1 ~ d-1
        for d in range(1, d):
            dp, dp_ = dp_, dp  # 交换
            # monotonic and minimum stack {dp_old_Best, curMax, bestSoFar}
            stack = [[float('inf'), float('inf'), float('inf')]]
            for i in range(d, n):
                old_best = dp_[i - 1]
                while stack[-1][1] <= jobDifficulty[i]:
                    old_best = min(old_best, stack[-1][0])
                    stack.pop()

                stack.append([old_best, jobDifficulty[i], min(old_best + jobDifficulty[i], stack[-1][2])])
                dp[i] = stack[-1][2]
        return dp[-1]


class Solution_1130:
    def mctFromLeafValues(self, arr: List[int]) -> int:
        # 单调栈
        res = 0
        stack = [float('inf')]
        for x in arr:
            # 我们事先存了一个 inf, 避免了判断 stack 不为空
            while stack[-1] <= x:  # 栈顶比当前元素小
                t = stack.pop()
                res += t * min(stack[-1], x)
            stack.append(x)

        while len(stack) > 2:
            res += stack.pop() * stack[-1]

        return res

    def mctFromLeafValues(self, arr: List[int]) -> int:
        # dp(i, j) = dp(i, k) + dp(k+1, j) + root_val
        def foo(i, j, memo):
            if i >= j: return 0
            if (i, j) not in memo:
                res = float('inf')
                for k in range(i + 1, j + 1):
                    res = min(res, foo(i, k - 1, memo) + foo(k, j, memo) + max(arr[i:k]) * max(arr[k:j + 1]))
                memo[i, j] = res
            return memo[i, j]

        return foo(0, len(arr) - 1, {})


class Solution_403:
    def canCross(self, stones: List[int]) -> bool:
        dp = {}

        def foo(i, k, dp):
            if stones[i] >= stones[-1]:
                return True
            if (i, k) in dp:
                return dp[i, k]

            for jump in (k - 1, k, k + 1):
                for j in range(i + 1, len(stones)):
                    if stones[i] + jump < stones[j]: break
                    if stones[i] + jump == stones[j]:
                        if foo(j, jump, dp):
                            dp[i, k] = True
                            return True
            dp[i, k] = False
            return False

        return foo(0, 0, dp)


class Solution_583:
    def minDistance(self, word1: str, word2: str) -> int:
        # LCS?
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        L = dp[m][n]
        return (m - L) + (n - L)


class Solution_678:
    def checkValidString(self, s: str) -> bool:
        def foo(i, cnt, dp):
            if i >= len(s):
                return cnt == 0
            if cnt < 0:
                return False
            if (i, cnt) in dp:
                return dp[i, cnt]
            if s[i] == '*':
                dp[i, cnt] = foo(i + 1, cnt, dp) or foo(i + 1, cnt + 1, dp) or foo(i + 1, cnt - 1, dp)
            elif s[i] == '(':
                dp[i, cnt] = foo(i + 1, cnt + 1, dp)
            else:
                dp[i, cnt] = foo(i + 1, cnt - 1, dp)
            return dp[i, cnt]

        return foo(0, 0, {})


class Solution_514:
    def findRotateSteps(self, ring: str, key: str) -> int:
        # memo + dfs
        N = len(ring)
        memo = {}

        def dfs(idx, i):
            nonlocal memo
            if i >= len(key):
                return 0
            if (idx, i) in memo:
                return memo[idx, i]

            if ring[idx] == key[i]:
                memo[idx, i] = dfs(idx, i + 1) + 1
                return memo[idx, i]

            j = (idx + 1) % N
            step = 1
            while j != idx and ring[j] != key[i]:
                j = (j + 1) % N
                step += 1
            res = dfs(j, i + 1) + step + 1

            j = (idx - 1 + N) % N
            step = 1
            while j != idx and ring[j] != key[i]:
                j = (j - 1 + N) % N
                step += 1
            res = min(res, dfs(j, i + 1) + step + 1)
            memo[idx, i] = res
            return res

        return dfs(0, 0)

    def findRotateSteps(self, ring: str, key: str) -> int:
        N = len(ring)
        indexes = collections.defaultdict(list)
        for i, c in enumerate(ring):
            indexes[c].append(i)

        memo = {}

        def dfs(idx, i):
            nonlocal memo
            if i >= len(key):
                return 0

            if (idx, i) in memo:
                return memo[idx, i]

            if ring[idx] == key[i]:
                memo[idx, i] = dfs(idx, i + 1) + 1
                return memo[idx, i]

            res = float('inf')
            for j in indexes[key[i]]:
                tmp = min(abs(j - idx), N - abs(j - idx))
                res = min(res, dfs(j, i + 1) + tmp + 1)
            memo[idx, i] = res
            return res

        return dfs(0, 0)

    def findRotateSteps(self, ring: str, key: str) -> int:
        # 换成 迭代 DP 速度变慢了，
        N = len(ring)
        M = len(key)
        indexes = collections.defaultdict(list)
        for i, c in enumerate(ring):
            indexes[c].append(i)

        dp = [[0] * (N + 1) for _ in range(M + 1)]

        for i in range(M - 1, -1, -1):
            for idx in range(N - 1, -1, -1):
                if ring[idx] == key[i]:
                    dp[i][idx] = dp[i + 1][idx] + 1
                else:
                    res = float('inf')
                    for j in indexes[key[i]]:
                        tmp = min(abs(j - idx), N - abs(j - idx))
                        res = min(res, dp[i + 1][j] + tmp + 1)
                    dp[i][idx] = res
        return dp[0][0]


class Solution_446:
    def numberOfArithmeticSlices(self, A: List[int]) -> int:
        # 1 2 3 4 [5]
        # 1 3 [5]
        # TLE
        dp = [[0] * len(A) for _ in range(len(A))]
        for i in range(2, len(A)):
            for j in range(1, i):
                for k in range(j):
                    if A[i] - A[j] == A[j] - A[k]:
                        # print(k, j, i)
                        dp[i][j] += dp[j][k] + 1

        return sum(sum(dp, []))

    def numberOfArithmeticSlices(self, A: List[int]) -> int:
        ans = 0
        cnt = [collections.defaultdict(int) for _ in range(len(A))]
        for i in range(1, len(A)):
            for j in range(i):
                diff = A[i] - A[j]
                s = cnt[j][diff]
                cnt[i][diff] += s + 1
                ans += s

        return ans


class Solution_474:
    # 背包
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        memo = {}

        def dfs(idx, mt, nt):
            nonlocal memo
            if idx >= len(strs):
                return 0
            if mt <= 0 and nt <= 0:
                return 0
            if (idx, mt, nt) in memo:
                return memo[idx, mt, nt]
            cnt = collections.Counter(strs[idx])
            res = 0
            if cnt['0'] <= mt and cnt['1'] <= nt:
                res = 1 + dfs(idx + 1, mt - cnt['0'], nt - cnt['1'])
            res = max(res, dfs(idx + 1, mt, nt))
            memo[idx, mt, nt] = res
            return res

        return dfs(0, m, n)

    def findMaxForm(self, strs, m, n):
        #
        dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(len(strs) + 1)]
        for i in range(len(strs) - 1, -1, -1):
            cnt = collections.Counter(strs[i])
            for j in range(m + 1):
                for k in range(n + 1):
                    res = dp[i + 1][j][k]
                    if cnt['0'] <= j and cnt['1'] <= k:
                        res = max(res, 1 + dp[i + 1][j - cnt['0']][k - cnt['1']])
                    dp[i][j][k] = res

        return dp[0][m][n]

    def findMaxForm(self, strs, m, n):
        # 空间优化，显然可以优化到 m*n, 同时要把后面两个更新顺序倒过来，本轮使用的就是上一轮中的数据 参见 0-1 背包。
        # 另外一个小 trick, 我们只用更新 cnt['0'] <= j <= m 和 cnt['1'] <= k <= n, 所以循环可以改一下。
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(len(strs) - 1, -1, -1):
            cnt = collections.Counter(strs[i])
            for j in range(m, cnt['0'] - 1, -1):
                for k in range(n, cnt['1'] - 1, -1):
                    dp[j][k] = max(dp[j][k], 1 + dp[j - cnt['0']][k - cnt['1']])

        return dp[m][n]


class Solution_546:
    def removeBoxes(self, boxes: List[int]) -> int:
        # len(boxes) <= 100
        # max points
        memo = {}

        def dfs(boxes):
            if not boxes: return 0
            if tuple(boxes) in memo:
                return memo[tuple(boxes)]
            tmp = []
            s = 0
            k = 1
            for i in range(1, len(boxes)):
                if boxes[i] == boxes[i - 1]:
                    k += 1
                else:
                    tmp.append((s, k))
                    s = i
                    k = 1
            if s == len(boxes) - 1 or k > 1:
                tmp.append((s, k))
            res = -1
            for s, k in tmp:
                res = max(res, dfs(boxes[:s] + boxes[s + k:]) + k * k)
            memo[tuple(boxes)] = res
            return res

        return dfs(boxes)

    def removeBoxes(self, boxes: List[int]) -> int:
        # len(boxes) <= 100
        # dp(i, j, k) 表示 从 i ~ j 能的取得的最大值，k 表示前面有 k 个元素 color 和 boxes[j] 一样
        # 具体就是表示 [b_1, b_2, ..., b_k， box_i, box_{i+1}, ..., box_j] 其中 b_1~b_k 的颜色相同且都等于 box_j
        # 如果 i~j 中间有一个 m 满足 box_m == ... == box_i == box_j == b_* ，我们可以将 i~m + 前面 中间的先处理了。
        # 对于另外 t 在 (m+1) ~ j 范围内，有 boxes[t] == b_*，我们可以先消去 boxes[m+1:t] , 这样我们我们就有了 boxes[t:j] +
        # 前面 k+1 个元素等同于 b_*
        N = len(boxes)
        memo = [[[0] * N for _ in range(N)] for _ in range(N)]

        def dp(i, j, k):
            if i > j: return 0
            if not memo[i][j][k]:
                m = i
                while m + 1 < j and boxes[m + 1] == boxes[i]:
                    m += 1
                # A[i:m+1] 都是同一个元素 + 加上前面有 k 个颜色与 boxes[i] 相同的
                i, k = m, k + m - i
                ans = dp(i + 1, j, 0) + (k + 1) ** 2
                for t in range(i + 1, j + 1):
                    if boxes[i] == boxes[t]:
                        ans = max(ans, dp(i + 1, t - 1, 0) + dp(t, j, k + 1))
                memo[i][j][k] = ans
            return memo[i][j][k]

        return dp(0, N - 1, 0)


class Solution_300:
    # 这个题目Python卡时间太严了，O(n^2)也会 TLE
    def lengthOfLIS(self, nums: List[int]) -> int:
        # O(n^2) dp
        # 0-1背包的变形题，枚举（递归） --> 带memo的递归 --> dp --> dp优化

        def foo(idx, path):
            # 这种前向和后向混在一起，而且状态不明确，不好转换为带memo的搜索
            print('#', idx, path)
            if idx == len(nums):
                return len(path) - 1
            r = 0
            if nums[idx] > path[-1]:
                r = foo(idx + 1, path + [nums[idx]])
            return max(r, foo(idx + 1, path))

        def helper(last, idx):
            if idx == len(nums):
                print('#', last, idx, '0')
                return 0
            r = 0
            if last < 0 or nums[last] < nums[idx]:
                r = 1 + helper(idx, idx + 1)
            t = max(r, helper(last, idx + 1))
            print('#', last, idx, t)
            return t

        return helper(-1, 0)

    def lengthOfLIS(self, nums: List[int]) -> int:
        # 暴力转为带memo的搜索
        memo = {}

        def helper(last, idx):
            nonlocal memo
            if idx == len(nums):
                return 0

            if (last, idx) in memo:
                return memo[(last, idx)]

            r = 0
            if last < 0 or nums[last] < nums[idx]:
                r = 1 + helper(idx, idx + 1)
            memo[(last, idx)] = max(r, helper(last, idx + 1))
            return memo[(last, idx)]

        return helper(-1, 0)

    def lengthOfLIS(self, nums: List[int]) -> int:
        # 带memo的搜索 转换为 后向DP
        # 这里i 对应last, j对应idx. 存的时候因为last有-1， 所以+1存储
        n = len(nums)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        max_len = 0
        for last in range(n - 1, -2, -1):
            for idx in range(n - 1, -1, -1):
                r = 0
                if last < 0 or nums[last] < nums[idx]:
                    r = 1 + dp[idx + 1][idx]
                dp[last + 1][idx] = max(r, dp[last + 1][idx + 1])
                max_len = max(dp[last + 1][idx], max_len)
        for ls in dp:
            print(ls)
        return max_len

    def lengthOfLIS(self, nums: List[int]) -> int:
        # 优化空间，前向DP, 跟上面的DP过程不太一样了
        # dp[i] = max(dp[j]) + 1,  0 <= j < i
        # ans = max(dp)

        n = len(nums)
        if n == 0:
            return 0  # 边界条件
        dp = [0] * n
        dp[0] = 1
        max_len = 0
        for i in range(n):
            tmp = 0
            for j in range(i):
                if nums[j] < nums[i]:
                    tmp = max(tmp, dp[j])

            dp[i] = tmp + 1
            max_len = max(max_len, dp[i])
        print(dp)
        return max_len

    def lengthOfLIS(self, nums: List[int]) -> int:
        # dp + binary search
        # O(n log n)
        n = len(nums)
        dp = [0] * n
        length = 0
        for x in nums:
            idx = bisect.bisect_left(dp, x, 0, length)
            dp[idx] = x
            if idx == length:
                length += 1

        return length

    def lengthOfLIS(self, nums: List[int]) -> int:
        res = []
        for x in nums:
            if not res or x > res[-1]:
                res.append(x)
            else:
                idx = bisect.bisect_left(res, x)
                res[idx] = x
        return len(res)

    def lengthOfLIS(self, nums: List[int]) -> int:
        # dp + binary search的另一种写法
        # 这里使用 dp 和 path 数组, 以及length 能够重建出其中一个最长子序列
        # 例如：[3, 4, -1, 5, 8, 2, 3, 12, 7, 9, 10]
        # 对应的dp: [2, 5, 6, 8, 9, 10, 0, 0, 0, 0, 0]
        # 对应的path: [-1, 0, -1, 1, 3, 2, 5, 4, 6, 8, 9]
        # length为：6
        # 首先 dp[length - 1] = 10
        # path[10] = 9
        # path[9] = 8
        # path[8] = 6
        # path[6] = 5
        # path[5] = 2
        # path[2] = -1
        # 重建的最长子序列为
        # index   2   5   6   8   9   10
        #  num   -1   2   3   7   9   10
        n = len(nums)
        tmp = [0] * n
        dp = [0] * n
        path = [-1] * n
        length = 0
        for i, x in enumerate(nums):
            idx = bisect.bisect_left(tmp, x, 0, length)
            dp[idx] = i
            tmp[idx] = x
            if idx > 0:
                path[i] = dp[idx - 1]
            if idx == length:
                length += 1
        print(dp)
        print(path)
        return length


class Solution_887:
    def superEggDrop(self, K: int, N: int) -> int:
        # DP(memo+search) + binary search
        # 状态(K, N)
        # 如果我们从X层丢鸡蛋，碎了状态变成(K-1, x_train-1), 没碎状态变成(K, N-x_train)
        # 定义dp(k, n) = 在状态(k, n)下解这个问题需要的最多的步数，则
        #   dp(k, n) = min_{1<=x_train<=N} (max(dp(k-1, x-1), dp(k, n-x)))
        # 注意到 t1 = dp(k-1, x-1), 随x单调增
        #        t2 = dp(k, n-x), 随x单调减
        # 于是max(t1,t2)是二者的上半部分, 因此查找x可以二分进行。
        # 边界条件是两者不一定相交于整数X, 因此这时要检查两个。
        # Time: O(KN log N)
        # space: O(KN)
        # https://leetcode.com/articles/super-egg-drop/ Solution1
        dp = {}

        def foo(k, n):
            if (k, n) in dp:
                return dp[(k, n)]

            if n == 0:
                ans = 0
            elif k == 1:
                ans = n
            else:
                lo, hi = 1, n
                while lo + 1 < hi:
                    x = (lo + hi) >> 1
                    t1 = foo(k - 1, x - 1)
                    t2 = foo(k, n - x)

                    if t1 < t2:
                        lo = x
                    elif t1 > t2:
                        hi = x
                    else:
                        lo = hi = x
                ans = 1 + min(max(foo(k - 1, x - 1), foo(k, n - x)) for x in (lo, hi))
            dp[(k, n)] = ans
            return dp[(k, n)]

        return foo(K, N)

    def superEggDrop(self, K: int, N: int) -> int:
        # 使用上述定义  dp(k, n) = min_{1<=x_train<=N} (max(dp(k-1, x-1), dp(k, n-x)))
        #   t1 = dp(k-1, x-1)，注意到t1随x单调增，但是与 n 无关
        #   t2 = t2 = dp(k, n-x), 随x单调减, 但随 n 单调增
        # 于是可以得到 dp(k, n) 随 n 的增加而增加。https://LeetCode.com/articles/super-egg-drop/ 这里有图
        # Time: O(kN)
        # space: O(N)
        dp = list(range(N + 1))
        for k in range(2, K + 1):
            print(dp)
            dp_tmp = [0]  # dp_tmp = dp(k, ·)
            x = 1
            for n in range(1, N + 1):
                # Increase our optimal x while we can make our answer better.
                while x < n and max(dp[x - 1], dp_tmp[n - x]) > max(dp[x], dp_tmp[n - x - 1]):
                    x += 1
                    print('#')
                dp_tmp.append(1 + max(dp[x - 1], dp_tmp[n - x]))
            dp = dp_tmp
        return dp[-1]

    def superEggDrop(self, K, N):
        # 反向思考这个问题，假如给出步数T，K个鸡蛋,，f(T,K)表示我们能够解原问题的楼层数的最大值。
        # （能够解原问题指：找到0<=F<=f(T,K)即确定找到楼层F).
        # 则问题转换为寻找满足 f(T,K) >= N 的T的最小值。即 min T s.t. f(T, K) >= N
        # 在最优策略下，我们在解X' 层丢一个鸡蛋，如果破了，我们可以解f(T-1, K-1), 如果没有碎，可以解f(T-1, K)
        # 于是   f(T, K) = 1 + f(T-1, K-1) + f(T-1, K) 且显然f(t, 1) = t (t>=1), f(1, k) = 1(k>=1)
        # 接下来用两种方式解的f(T, K)的通项
        #    i). 记g(t, k) = f(T, K) - F(T, K-1)
        #          f(T, K)    = 1 + f(T-1, K-1) + f(T-1, K)
        #          f(T, K-1)  = 1 + f(T-1, K-2) + f(T-1, K-1)
        #          g(t, k) = f(T, K) - f(T, K-1) = f(T-1, k) - f(T-1, K-2) = g(t-1, k) + g(t-1, k-1)
        #          上式子 g(t, k) = g(t-1, k) + g(t-1, k-1) 是一个二项分布的递归式，其解为g(t, k) = C(t, k+1)
        #        则：f(t, k) = \sum_{1<=x<=K} g(t, x) = \sum C(t, x)
        #
        #   ii). 另一个角度来看，我们有t次尝试和k个鸡蛋，因此这是一个长度为t,失败（鸡蛋碎）次数最多为k的尝试序列。
        #        没有失败的是C(n, 0), 一次失败是C(n, 1)..., 综合起来就是\sum C(t, x)
        #
        #  by using  C(n, k + 1) = C(n, k) * (n-k)/(k+1) 可以简化计算

        def combination(x):
            # C(n, k + 1) = C(n, k) * (n-k)/(k+1)
            ans = 0
            r = 1
            for i in range(1, K + 1):
                r *= x - i + 1  # r = r * (x - (i-1)) // ((i-1) + 1)
                r //= i
                ans += r
                if ans >= N:
                    break
            return ans

        lo, hi = 1, N
        while lo < hi:
            mi = (lo + hi) >> 1
            if combination(mi) < N:
                lo = mi + 1
            else:
                hi = mi
        return lo


class Solution_1043:
    def maxSumAfterPartitioning(self, A: List[int], K: int) -> int:

        @functools.lru_cache(None)
        def dfs(i):  # [:i]
            if i <= 0: return 0
            res = -1
            for j in range(1, K + 1):
                res = max(res, dfs(i - j) + max(A[i - j:i] or [0]) * j)
            return res

        return dfs(len(A))

    def maxSumAfterPartitioning(self, A: List[int], K: int) -> int:
        dp = [0] * (len(A) + 1)
        for i in range(1, len(A) + 1):
            cur_max = 0
            for j in range(1, min(K, i) + 1):
                cur_max = max(cur_max, A[i - j])
                # print(i, j, cur_max, max(A[i-j:i] or [0]))
                t = dp[i - j] if i >= j else 0
                dp[i] = max(dp[i], t + cur_max * j)
        return dp[len(A)]


class Solution_712:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:

        @functools.lru_cache(None)
        def dfs(i, j):
            if i >= len(s1) or j >= len(s2):
                return (sum(ord(s2[k]) for k in range(j, len(s2))) +
                        sum(ord(s1[k]) for k in range(i, len(s1))))

            res = dfs(i + 1, j + 1) if s1[i] == s2[j] else float('inf')

            res = min(res,
                      dfs(i + 1, j) + ord(s1[i]),
                      dfs(i, j + 1) + ord(s2[j]),
                      dfs(i + 1, j + 1) + ord(s1[i]) + ord(s2[j]))
            # print(i, j, res)
            return res

        return dfs(0, 0)

    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        A, B = list(map(ord, s1)), list(map(ord, s2))
        m, n = len(A), len(B)

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m - 1, -1, -1):
            dp[i][n] = dp[i + 1][n] + A[i]

        for j in range(n - 1, -1, -1):
            dp[m][j] = dp[m][j + 1] + B[j]

        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if A[i] == B[j]:
                    dp[i][j] = dp[i + 1][j + 1]
                else:
                    dp[i][j] = min(dp[i + 1][j] + A[i],
                                   dp[i][j + 1] + B[j])

        return dp[0][0]
