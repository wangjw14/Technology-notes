# leetcode

### 快排

```python
def quick_sort(nums):
    l, r = 0, len(nums)-1
    helper(nums, l, r)
    return nums 

def helper(nums, l, r):
    if l >= r:
        return 
    mid = partition(nums, l , r)
    helper(nums, l , mid-1)
    helper(nums, mid+1, r)
    

def partition(nums, l, r):
    p = l 
    while l < r:
        while l < r and nums[r] >= nums[p]:
            r -= 1
        while l < r and nums[l] <= nums[p]:
            l += 1
        nums[l], nums[r] = nums[r], nums[l]
    nums[l], nums[p] = nums[p], nums[l]
    return l

l = [1,4,5,2,5,2,0]
print(quick_sort(l))
```

