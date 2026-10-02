# Closures in Python

## Definition

A **closure** is a nested function that remembers the variables from its
enclosing scope, even after the enclosing function has finished running.
The inner function "closes over" those variables, which is where the name
comes from.

Closures build directly on the **E** in the LEGB rule: the inner function
looks names up in its enclosing scope.

## The three requirements

A closure exists when all three of these are true:

1. There is a **nested function** (a function defined inside another
   function).
2. The nested function **references a variable from the enclosing
   function** (a *free variable*).
3. The enclosing function **returns the nested function** (or otherwise
   lets it outlive the enclosing call).

```python
def make_multiplier(factor):
    def multiply(x):
        return x * factor      # factor is a free variable
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)
double(5)   # 10
triple(5)   # 15
```

`make_multiplier` has returned, but `double` still remembers `factor = 2`.

## Example: make_counter and `nonlocal`

Reading an enclosing variable just works. *Rebinding* it does not, because
assignment makes a name local. Use `nonlocal` to rebind it:

```python
def make_counter():
    count = 0
    def increment():
        nonlocal count     # rebind the enclosing count, not a new local
        count += 1
        return count
    return increment

counter = make_counter()
counter()   # 1
counter()   # 2
counter()   # 3

other = make_counter()
other()     # 1, because each call to make_counter makes a fresh count
```

Without `nonlocal`, `count += 1` raises `UnboundLocalError`.

Mutating a mutable object does not need `nonlocal`, because you are not
rebinding the name:

```python
def make_logger():
    lines = []
    def log(msg):
        lines.append(msg)   # mutation, not rebinding
        return lines
    return log
```

## Gotcha: late binding in loops

Closures capture **variables, not values**. The variable is looked up when
the inner function *runs*, not when it is defined:

```python
funcs = []
for i in range(3):
    funcs.append(lambda: i)

[f() for f in funcs]   # [2, 2, 2], not [0, 1, 2]
```

All three lambdas share the same `i`, which is `2` when the loop ends.

Fix 1: bind the current value with a default argument (evaluated at
definition time):

```python
funcs = [lambda i=i: i for i in range(3)]
[f() for f in funcs]   # [0, 1, 2]
```

Fix 2: create a new enclosing scope per iteration:

```python
def make_f(i):
    return lambda: i

funcs = [make_f(i) for i in range(3)]
```

## Inspecting closures with `__closure__`

Every function has a `__closure__` attribute. For a closure it is a tuple of
*cell* objects, one per free variable. A plain function has `None`.

```python
counter = make_counter()
counter.__code__.co_freevars          # ('count',)
counter.__closure__                   # (<cell at 0x...: int object ...>,)
counter.__closure__[0].cell_contents  # 0
counter()
counter.__closure__[0].cell_contents  # 1

def plain():
    return 1
plain.__closure__                     # None
```

## When to use closures

- **Function factories** such as `make_multiplier`.
- **Keeping private state** without writing a class, as in `make_counter`.
- **Callbacks** that need extra context.
- **Decorators**, which are closures that wrap another function.

## Key takeaways

- A closure is a nested function plus the enclosing variables it references.
- It needs a nested function, a free variable, and an outer function that
  returns the inner one.
- Use `nonlocal` to rebind an enclosing variable.
- Closures capture variables, not values (late binding).
- `__closure__` holds the captured cells.
