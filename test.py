import concurrent.futures

def testFunc(a, b, c ,i):
    print("worker "+ str(i))
    return a * b, b * c

pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)

# Submitting tasks to the thread pool
future1 = pool.submit(testFunc, 1, 2, 3 ,1)
future2 = pool.submit(testFunc, 1, 2, 3 ,2)
future3 = pool.submit(testFunc, 1, 2, 3 ,3)
future4 = pool.submit(testFunc, 1, 2, 3, 4)

# Retrieving results from the completed tasks
temp1, temp11 = future1.result()
temp2, temp22 = future2.result()
temp3, temp33 = future3.result()
temp4, temp44 = future4.result()

# Shutdown the thread pool
pool.shutdown(wait=True)

# Print the results
print(temp1, temp11)
print(temp2, temp22)
print(temp3, temp33)
print(temp4, temp44)
