import pandas as pd

df = pd.DataFrame(columns=['date', 'values', 'name'])

print(df)
print(2**5)
for i in range(4):
    if i % 2 == 0:
        print(i)


def gemstones(stones):
    stone_set = []
    for stone in stones:
        stone_set.append(set(list(stone)))
    ans = len(set.intersection(*stone_set))
    return ans


string = ['the quick brown fox jumps over the lazy dog', 'this is not a pangram']


def is_pangram(string):
    ans = ''
    for st in string:
        if len(set(list(st.replace(' ', '')))) == 26:
            ans += '1'
        else:
            ans += '0'

    return ans


print(is_pangram(string))

import math


def volleyball(A, B):
    mod = 1000000007
    if B + 1 < A == 25:
        ans = math.factorial(24 + B) / (math.factorial(B) * math.factorial(24))
    elif A + 1 < B == 25:
        ans = math.factorial(24 + A) / (math.factorial(A) * math.factorial(24))
    elif A > 24 and B > 24 and abs(A - B) == 2:
        ps = min(A,B)
        part1 = math.factorial(48) / (math.factorial(24) * math.factorial(24))
        part2 = pow(2, ps - 26, mod)
        ans = part1 * part2 %mod
    else:
        ans = 0

    return int(ans)


print(volleyball(28, 30))


# Python3 Program for recursive binary search.

# Returns index of x in arr if present, else -1


def binarySearch(arr, l, r, x):
    # Check base case
    if r >= l:

        mid = l + (r - l) // 2

        # If element is present at the middle itself
        if arr[mid] == x:
            return mid

        # If element is smaller than mid, then it
        # can only be present in left subarray
        elif arr[mid] > x:
            return binarySearch(arr, l, mid - 1, x)

        # Else the element can only be present
        # in right subarray
        else:
            return binarySearch(arr, mid + 1, r, x)

    else:
        # Element is not present in the array
        return -1


# Driver Code
arr = [0]
x = 10

# Function call
result = binarySearch(arr, 0, len(arr) - 1, x)

if result != -1:
    print("Element is present at index % d" % result)
else:
    print("Element is not present in array")

#USing Dictionary


def duplicate(input_list):
    new_dict, new_list = {}, []

    for i in input_list:
        if not i in new_dict:
            new_dict[i] = 1
        else:
            new_dict[i] += 1

    for key, values in new_dict.items():
        if values > 1:
            new_list.append(key)

    return new_list


dcit = {}
lst = []

for i in lst:
    if not i in dcit:
