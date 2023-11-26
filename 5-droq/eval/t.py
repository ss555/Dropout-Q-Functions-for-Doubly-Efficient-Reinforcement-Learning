from typing import List
class Solution:
    def grayCode(self, n: int) -> List[int]:
        result = []
        for i in range(pow(2, n)):
            print(f'i={i},i>>1={i>>1},i^(i>>1)={i ^ (i >> 1)}')
            result.append(i ^ (i // 2))
        return result
    # def grayCode(self, n: int) -> List[int]:
    #     result = [0]
    #     for i in range(1, 1 << n):
    #         print(f'i={i},i>>1={i>>1},i^(i>>1)={i ^ (i >> 1)}')
    #         result.append(i ^ (i >> 1))
    #     return result


s = Solution()
print(s.grayCode(2))