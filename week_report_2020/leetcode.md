# leetcode 

- 快排

  ```python
  class Solution:
      def sortArray(self, nums: List[int]) -> List[int]:
          def partition(nums, l, r):
              p = l 
              while l<r:
                  while l<r and nums[r]>=nums[p]:
                      r -= 1
                  while l<r and nums[l]<=nums[p]:
                      l += 1
                  nums[l], nums[r] = nums[r], nums[l] 
              nums[l], nums[p] = nums[p], nums[l] 
              return l 
          def quickSort(nums,l,r):
              if l>=r:
                  return 
              mid = partition(nums,l,r)
              quickSort(nums,l,mid-1)
              quickSort(nums,mid+1,r)
  
          quickSort(nums,0,len(nums)-1)
          return nums
  ```

- [169. 多数元素](https://leetcode-cn.com/problems/majority-element/) 

  ```python
  class Solution:
      def majorityElement(self, nums: List[int]) -> int:
          n, c = None, 0
          for num in nums:
              if n == num:
                  c += 1
              elif c==0:
                  n, c = num, 1
              else:
                  c -= 1
          return n 
  ```

- [229. 求众数 II](https://leetcode-cn.com/problems/majority-element-ii/)

  ```python
  class Solution:
      def majorityElement(self, nums: List[int]) -> List[int]:
          c1, c2 = None, None 
          count1, count2 = 0, 0
          for n in nums:
              if c1 == n:   # 注意顺序
                  count1 += 1
              elif c2 == n:
                  count2 += 1
              elif count1 == 0:
                  c1, count1 = n, 1
              elif count2 == 0:
                  c2, count2 = n, 1
              else:
                  count1 -= 1
                  count2 -= 1
          return [i for i in [c1, c2] if nums.count(i)>len(nums)/3]
  ```

- [31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/) 

  ```python
  class Solution:
      def nextPermutation(self, nums: List[int]) -> None:
          """
          Do not return anything, modify nums in-place instead.
          """
          i = len(nums) - 1
          while i >0:
              if nums[i]>nums[i-1]:
                  break 
              i -= 1
          if i > 0:
              j = len(nums)-1
              while j >= i:
                  if nums[j]>nums[i-1]:
                      break 
                  j -= 1
              nums[j], nums[i-1] = nums[i-1], nums[j] 
          start, end = i, len(nums)-1 
          while start<end:
              nums[start], nums[end] = nums[end], nums[start]
              start += 1
              end -= 1
  ```

- [78. 子集](https://leetcode-cn.com/problems/subsets/) 

  ```python
  class Solution:
      def subsets(self, nums: List[int]) -> List[List[int]]:
          if nums == []:
              return [[]]
          res = self.subsets(nums[1:])   # 注意
          return res + [[nums[0]]+ s for s in res]
  ```

  ```python
  class Solution:
      def subsets(self, nums: List[int]) -> List[List[int]]:
          def helper(res,lst,nums,p):
              res.append(lst[:])
              for i in range(p,len(nums)):
                  lst.append(nums[i])
                  helper(res,lst,nums,i+1)  # 注意是i，不是p
                  lst.pop()
  
          res = []
          lst = []
          helper(res,lst,nums,0)
          return res 
  ```

- [46. 全排列](https://leetcode-cn.com/problems/permutations/) 

  ```python
  class Solution:
      def permute(self, nums: List[int]) -> List[List[int]]:
          if len(nums) == 0:
              return [[]]
          res = []
          for i in range(len(nums)):
              res += [[nums[i]] + r for r in self.permute(nums[:i]+nums[i+1:])]
          return res 
  ```

  ```python
  class Solution:
      def permute(self, nums: List[int]) -> List[List[int]]:
          def helper(res,lst,nums):
              if len(lst) == len(nums):
                  res.append(lst[:])
              for n in nums:
                  if n in lst:
                      continue 
                  lst.append(n)
                  helper(res,lst,nums)
                  lst.pop()
          res = []
          lst = []
          helper(res,lst,nums)
          return res 
  ```

- #### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

  ```python
  class Solution:
      def lengthOfLongestSubstring(self, s: str) -> int:
          if len(s) == 0:
              return 0
          start, end = 0, 0
          l = len(s) 
          seen = set()
          global_max = 0
          while start< l and end < l:
              if s[end] not in seen:
                  seen.add(s[end])
                  end += 1
                  global_max = max(global_max, end-start)
              else:
                  seen.remove(s[start])
                  start += 1 
          return global_max
  ```

  ```python
  class Solution:
      def lengthOfLongestSubstring(self, s: str) -> int:
          start, end = 0, 0
          dic = {}
          global_max = 0
          for idx, n in enumerate(s):
              if n in dic and start<=dic[n]:
                  start = dic[n] + 1
              else:
                  global_max = max(global_max,idx-start+1)
              dic[n] = idx 
          return global_max
  ```

- [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/) 

  ```python
  class Solution:
      def minWindow(self, s: str, t: str) -> str:
          if len(t) > len(s):
              return ""
          ct = collections.Counter(t)
          count = len(t)
          start, end = 0, 0
          global_min = len(s)+1
          n = len(s)
          res = ""
          while start< n :
              if count >0 and end < n:
                  if s[end] in ct:
                      if ct[s[end]] > 0:
                          count -= 1
                      ct[s[end]] -= 1
                  end += 1
              elif count == 0:
                  if end - start < global_min:
                      res = s[start:end]
                      global_min = end -start
                  if s[start] in ct:
                      ct[s[start]] += 1
                      if ct[s[start]] > 0:
                          count += 1
                  start += 1
              else:
                  break 
          return res 
  ```

- #### [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

  ```python
  class Solution:
      def minDistance(self, word1: str, word2: str) -> int:
          l1, l2 = len(word1), len(word2)
          if l1*l2 == 0:
              return max(l1,l2)
          dp = [[0]*(l2+1) for _ in range(l1+1)]
          for i in range(1,l1+1):
              dp[i][0] = i 
          for i in range(1,l2+1):
              dp[0][i] = i 
          for i in range(1,l1+1):
              for j in range(1,l2+1):
                  if word1[i-1] == word2[j-1]:
                      dp[i][j] = dp[i-1][j-1]
                  else:
                      dp[i][j] = 1+ min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
          return dp[l1][l2]
  ```

- #### [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

  ```python
  class Solution:
      def longestPalindrome(self, s: str) -> str:
          if len(s) == 0:
              return ""
          res = s[0]
          global_max = 1 
          n = len(s)
          dp = [[False]*n for _ in range(n)] 
          for i in range(n):
              dp[i][i] = True 
          
          for j in range(1,n):
              for i in range(j):
                  if s[i] == s[j]:
                      if j-i<3 or dp[i+1][j-1]:
                          dp[i][j] = True 
                          if j-i+1 > global_max:
                              global_max = j-i+1
                              res = s[i:j+1]
          return res             
  ```

- #### [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

  ```python
  class Solution:
      def findKthLargest(self, nums: List[int], k: int) -> int:
          idx = len(nums) - k 
          self.helper(nums,0,len(nums)-1,idx)
          return nums[idx]
  
      def helper(self,nums,l,r,idx):
          m = self.partition(nums,l,r)
          if m == idx:
              return
          elif m < idx:
              self.helper(nums,m+1,r,idx)
          else:
              self.helper(nums,l,m-1,idx)
  
      def partition(self,nums,l,r):
          mid = (l + r)//2
          nums[l] ,nums[mid] = nums[mid], nums[l]#此处是（伪）随机选择pivot，可以大幅减少运行时间
          p = l
          while l<r:
              while l<r and nums[r] >= nums[p]:
                  r -= 1
              while l<r and nums[l] <= nums[p]:
                  l += 1
              nums[l], nums[r] = nums[r], nums[l]
          nums[l], nums[p] = nums[p], nums[l]
          return l # 注意返回值是l，不是p
  ```

  ```python
  class Solution:
      def findKthLargest(self, nums: List[int], k: int) -> int:
          size = len(nums)
          if k > size:
              raise Exception('程序出错')
  
          L = []
          for index in range(k):
              # heapq 默认就是小顶堆
              heapq.heappush(L, nums[index])
  
          for index in range(k, size):
              top = L[0]
              if nums[index] > top:
                  # 看一看堆顶的元素，只要比堆顶元素大，就替换堆顶元素
                  heapq.heapreplace(L, nums[index])
          # 最后堆顶中的元素就是堆中最小的，整个数组中的第 k 大元素
          return L[0]
  ```


- #### [剑指 Offer 26. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
      def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
          def sameTree(a,b):
              if b is None:
                  return True 
              if a is None:
                  return False
              if a.val != b.val:
                  return False
              return sameTree(a.left,b.left) and sameTree(a.right, b.right)
  
          if A is None or B is None:
              return False
          res = False
          if A.val == B.val:
              res = sameTree(A,B)
          if not res:
              res = self.isSubStructure(A.left,B)
          if not res:
              res = self.isSubStructure(A.right,B)
          return res 
  ```

- 蓄水池抽样

  ```python
  import random
  class ReservoirSample(object):
      def __init__(self, size):
          self._size = size
          self._counter = 0
          self._sample = []
  
      def feed(self, item):
          self._counter += 1
          # 第i个元素（i <= k），直接进入池中
          if len(self._sample) < self._size:
              self._sample.append(item)
              return self._sample
          # 第i个元素（i > k），以k / i的概率进入池中
          rand_int = random.randint(1, self._counter)
          if rand_int <= self._size:
              self._sample[rand_int - 1] = item
          return self._sample
  
  ```

  蓄水池采样算法（Reservoir Sampling）原理，证明和代码 https://blog.csdn.net/anshuai_aw1/article/details/88750673

- 将一个数组分割成两个差值最小的部分

  ```python
  def split_nums(nums):
      s = sum(nums)
      half_sum = s // 2
      length = len(nums)
      dp = [[0]*(half_sum+1) for _ in range(length+1)]
      
      for i in range(1, length+1):
          for j in range(1, half_sum+1):
              if nums[i-1] > j :           # 这里出过错，注意
                  dp[i][j] = dp[i-1][j]
              else:
                  dp[i][j] = max(dp[i-1][j], dp[i-1][j-nums[i-1]]+nums[i-1] ) 
      i = length
      v = half_sum 
      a, b = [], []
      while i >0:
          if dp[i][v] == dp[i-1][v]:
              b.append(nums[i-1]) 
              
          elif dp[i][v] == dp[i-1][j-nums[i-1]]+nums[i-1]:
              a.append(nums[i-1])
              v -= nums[i-1]
          i -= 1
      
      return dp[-1][-1],a,b
  
  arr = [1, 2, 3, 4, 5]
  print(split_nums(arr))
  ```

- [平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

  ```python
  class Solution:
      def isBalanced(self, root: TreeNode) -> bool:
          def helper(node):
              if node is None:
                  return True, 0
              leftBalanced, leftDepth =  helper(node.left)
              if not leftBalanced:
                  return False, leftDepth + 1   # 此时返回False，第二项的值无关紧要
              rightBalabced, rightDepth = helper(node.right)
              if not rightBalabced:
                  return False, rightDepth + 1    # 此时返回False，第二项的值无关紧要        
              return abs(leftDepth - rightDepth)<2 , max(rightDepth,leftDepth) + 1
          return helper(root)[0]
  ```

  

- [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

  ```python
  # 后续遍历
  class Solution:
      def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
          ans = TreeNode(None)
          def dfs(root, p, q):
              if root is None:
                  return False 
              left = dfs(root.left, p, q)
              right = dfs(root.right, p, q)
              if (left and right) or ((root.val==p.val or root.val==q.val) and (left or right)):
                  ans.val = root.val 
                  return 
              return left or right or root.val==p.val or root.val==q.val
          dfs(root,p,q)
          return ans
  ```

  ```python
  class Solution:
      def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
          parents = {}
          visited = set()
          def dfs(node):
              if node is None:
                  return 
              if node.left:
                  parents[node.left.val] = node 
                  dfs(node.left)
              if node.right:
                  parents[node.right.val] = node 
                  dfs(node.right)
          dfs(root)
          parents[root.val] = None 
          node = p 
          while node:
              visited.add(node.val)
              node = parents[node.val] 
          node = q
          while node:
              if node.val in visited:
                  return node 
              node = parents[node.val] 
          return None 
  ```

- [297. 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)

  ```python
  class Codec:
  
      def serialize(self, root):
          """Encodes a tree to a single string.
          
          :type root: TreeNode
          :rtype: str
          """
          if root is None:
              return ''
          queue = collections.deque()
          queue.append(root) 
          res = []
          while queue:
              node = queue.popleft()
              if node:
                  res.append(str(node.val)) 
                  queue.append(node.left)
                  queue.append(node.right)
              else:
                  res.append("None") 
          while res[-1] == "None":
              res.pop() 
          return ",".join(res) 
      
      def deserialize(self, data):
          """Decodes your encoded data to tree.
          
          :type data: str
          :rtype: TreeNode
          """
          if data == "":
              return None 
          queue = collections.deque()
          l = data.split(',')
          root = TreeNode(int(l[0]))
          queue.append(root) 
          cnt = 0
          for v in l[1:]:
              if cnt == 0:
                  parent = queue.popleft()
              if v != "None":
                  node = TreeNode(int(v))
                  queue.append(node) 
                  if cnt == 0:
                      parent.left = node 
                  if cnt == 1:
                      parent.right = node 
              cnt ^= 1 
          return root 
  ```

- #### [887. 鸡蛋掉落](https://leetcode-cn.com/problems/super-egg-drop/)

  ```python
  class Solution:
      def superEggDrop(self, K: int, N: int) -> int:
          memory = {}
          def helper(k,n):
              if (k,n) not in memory:
                  if n == 0:
                      res = 0
                  elif k == 1:
                      res = n 
                  else:
                      l,r = 1, n 
                      while l+1<r:
                          x = (l+r)//2
                          t1 = helper(k-1,x-1)
                          t2 = helper(k,n-x) 
                          if t1<t2:
                              l = x
                          elif t1>t2:
                              r = x
                          else:
                              l, r = x, x
                      res = 1 + min([ max(helper(k-1,x-1),helper(k,n-x))  for x in (l,r) ])
                  memory[(k,n)] = res
              return memory[(k,n)]
          return helper(K,N) 
  ```

- #### [410. 分割数组的最大值](https://leetcode-cn.com/problems/split-array-largest-sum/)

  ```python
  class Solution:
      def splitArray(self, nums: List[int], m: int) -> int:
          length = len(nums)
          dp = [[sys.maxsize]*(m+1) for _ in range(length+1)]
          dp[0][0] = 0
          accumulate = [0] * (length+1)
          for i in range(1,length+1):
              accumulate[i] = accumulate[i-1] + nums[i-1] 
          for i in range(1,length+1):
              for j in range(1, min(i,m)+1):
                  for k in range(i):
                      dp[i][j] = min(dp[i][j], max(dp[k][j-1], accumulate[i]-accumulate[k]))
          return dp[-1][-1]
  ```

  ```python
  # O(N*log(sum-maxn))
  class Solution:
      def splitArray(self, nums: List[int], m: int) -> int:
          l, r = max(nums), sum(nums)
          while l < r:
              mid = l+(r-l)//2
              c = 1
              s = 0
              for n in nums:
                  if s+n<=mid:
                      s += n 
                  else:
                      c += 1
                      s = n 
              if c>m:
                  l = mid + 1
              else:
                  r = mid 
          return l 
  ```

- #### [120. 三角形最小路径和](https://leetcode-cn.com/problems/triangle/)

  ```python
  class Solution:
      def minimumTotal(self, triangle: List[List[int]]) -> int:
          length = len(triangle)
          dp = [0] * length
          dp[0] = triangle[0][0]
          for i in range(1,length):
              dp[i] = dp[i-1] + triangle[i][i]
              for j in range(i-1,0,-1):
                  dp[j] = min(dp[j-1],dp[j]) + triangle[i][j] 
              dp[0] += triangle[i][0] 
          return min(dp)
  ```

  