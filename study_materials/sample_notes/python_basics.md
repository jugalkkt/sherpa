# Python Basics: Variables, Functions and Scope

These notes cover the foundations you need before closures and decorators
make sense: how names are bound, how functions work, and how Python decides
which variable a name refers to.

## Variables are names bound to objects

A Python variable is not a box that holds a value. It is a **name** that
refers to an object. Assignment binds a name; it never copies the object.

```python
a = [1, 2, 3]
b = a          # b and a now name the SAME list
b.append(4)
print(a)       # [1, 2, 3, 4]
```

Rebinding a name does not affect other names that referred to the old object:

```python
x = 10
y = x
x = 20
print(y)       # 10, because y still refers to the int 10
```

### Mutable vs immutable

- **Immutable**: `int`, `float`, `str`, `tuple`, `frozenset`. "Changing" one
  creates a new object.
- **Mutable**: `list`, `dict`, `set`, most user-defined objects. They can be
  changed in place, and every name referring to them sees the change.

## Functions

Functions are defined with `def`, take parameters, and return a value
(`None` if there is no `return` statement).

```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

greet("Ada")                 # 'Hello, Ada!'
greet("Ada", greeting="Hi")  # 'Hi, Ada!'
```

### The mutable default argument gotcha

Default values are evaluated **once**, when the function is defined, not on
every call:

```python
def add_item(item, bucket=[]):   # one shared list for every call!
    bucket.append(item)
    return bucket

add_item(1)  # [1]
add_item(2)  # [1, 2], which is surprising
```

The fix is to use `None` as a sentinel:

```python
def add_item(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket
```

## Scope

A **scope** is the region of code where a name is visible. Every function
call creates a new local scope. Names assigned inside a function are local
to it unless declared otherwise.

```python
def f():
    secret = 42      # local to f
    return secret

f()
print(secret)        # NameError: secret only exists inside f
```

## The LEGB rule

When Python looks up a name, it searches four scopes in order:

1. **L**ocal: names assigned inside the current function.
2. **E**nclosing: locals of any enclosing (outer) functions.
3. **G**lobal: names at the top level of the module.
4. **B**uilt-in: names like `len`, `print`, `range`.

The first match wins.

```python
x = "global"

def outer():
    x = "enclosing"
    def inner():
        x = "local"
        print(x)     # local
    inner()
    print(x)         # enclosing

outer()
print(x)             # global
```

### Assignment makes a name local

If a function assigns to a name *anywhere* in its body, that name is local
for the whole function. Reading it before the assignment fails:

```python
count = 0

def bump():
    count += 1       # UnboundLocalError: count is local here
```

Use `global` to rebind a module-level name, or `nonlocal` to rebind a name
in an enclosing function (see the closures notes):

```python
count = 0

def bump():
    global count
    count += 1
```

## Key takeaways

- Variables are names bound to objects; assignment never copies.
- Default arguments are evaluated once, at definition time.
- Name lookup follows LEGB: Local, Enclosing, Global, Built-in.
- Assigning to a name makes it local unless you use `global` or `nonlocal`.
