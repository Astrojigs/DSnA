## Python Learning Guide for Catherine

Welcome, Catherine! Let’s learn Python step by step. We’ll keep things simple but clear.

1. **Basic Object Types**
2. **Conditionals (`if`, `else`)**
3. **Lists**
4. **Tuples**
5. **For Loops**
6. **While Loops**
7. **Functions**
8. **Input & Output**

---

### 1. Basic Object Types

Python stores different kinds of data in **objects**. Here are the most common:

* **int**: whole numbers, like `5` or `-2`.
* **float**: decimal numbers, like `3.14` or `0.5`.
* **str**: text in quotes, like `'hello'` or `"Python"`.
* **bool**: `True` or `False`.

```python
age = 5            # int
temperature = 36.6 # float
name = "Catherine" # str
is_happy = True    # bool
```

**Practice**: Create one variable of each type. Use `print()` and `type()` to show its value and type.

---

### 2. Conditionals (`if`, `else`)

Make decisions in your code based on conditions.

```python
if condition:
    # do this if condition is True
else:
    # do this if condition is False
```

**Example**: Check if a number is even or odd.

```python
number = 7
if number % 2 == 0:
    print("Even")
else:
    print("Odd")
```

**Practice**:

* Print “Positive” if a number > 0.
* Print “Zero” if it is 0.
* Otherwise, print “Negative.”

---

### 3. Lists

An **ordered**, **changeable** collection of items.

```python
fruits = ["apple", "banana", "cherry"]
```

* **Access**: `fruits[0]` → `'apple'`
* **Add**: `fruits.append("date")`
* **Remove**: `fruits.pop()` removes last item

**Practice**:

* Make a list of your favorite foods.
* Add one item, then remove the last item.

---

### 4. Tuples

An **ordered**, **unchangeable** collection.

```python
point = (10, 20)
```

* **Access**: `point[0]` → `10`
* **Cannot** add or remove items once set
* **Useful** for fixed groups of values

**Practice**:

* Create a tuple with three numbers.
* Try changing one element and see the error.

---

### 5. For Loops

Repeat actions for each item in a sequence.

```python
for fruit in fruits:
    print(fruit)
```

Use `range()` to loop a certain number of times:

```python
for i in range(3):  # i = 0,1,2
    print(i)
```

**Practice**:

* Print numbers 1 to 5.
* Print each letter in the word "Python."
* Use a loop to print squares of numbers from 1 to 5.

---

### 6. While Loops

Keep repeating **while** a condition is True.

```python
count = 3
while count > 0:
    print(count)
    count -= 1
print("Done!")
```

**Practice**:

* Countdown from 5 to 1, then print “Go!”
* Ask the user to type "yes" and repeat until they do.

---

### 7. Functions

Group reusable code into a named block.

```python
def greet(name):
    return "Hello, " + name

print(greet("Catherine"))
```

* **def** starts a function
* **parameters** go in `()`
* **return** sends back a result

**Practice**:

* Write `add(a, b)` that returns `a + b`.
* Write `is_even(n)` that returns `True` if `n` is even, else `False`.

---

### 8. Input & Output

Interact with the user.

* **print()** displays text or variables.
* **input()** reads user input (always as a string).

```python
name = input("Enter your name: ")
print("Hello, " + name)

age = int(input("How old are you? "))  # convert to int
print("In 5 years you will be", age + 5)
```

**Practice**:

* Ask the user for two numbers and print their sum.
* Ask for their favorite color and print a custom message.

---
