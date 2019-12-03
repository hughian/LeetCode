
class MyHashSet_705:
    # TODO 开放定址法，平方探测
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.primes = 10007
        self.hash_table = [None] * self.primes

    def add(self, key: int) -> None:
        idx = key % self.primes
        i = 1
        sign = 1
        while self.hash_table[idx] is not None:
            if self.hash_table[idx] == key:
                return
            idx = (key + i * i * sign) % self.primes
            if sign == -1:
                i += 1
            sign *= -1

        self.hash_table[idx] = key

    def remove(self, key: int) -> None:
        idx = key % self.primes
        i = 1
        sign = 1
        while self.hash_table[idx] is not None and self.hash_table[idx] != key:
            idx = (key + i * i * sign) % self.primes
            if sign == -1:
                i += 1
            sign *= -1
        self.hash_table[idx] = None

    def contains(self, key: int) -> bool:
        """
        Returns true if this set contains the specified element
        """
        idx = key % self.primes
        i = 1
        sign = 1
        while self.hash_table[idx] is not None:
            if self.hash_table[idx] == key:
                return True
            idx = (key + i * i * sign) % self.primes
            if sign == -1:
                i += 1
            sign *= -1
        return False
